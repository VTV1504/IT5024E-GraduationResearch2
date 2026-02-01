from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.config import load_settings
from src.storage.pg import PostgresStorage
from src.utils.reporting import make_report_path, write_report

logger = logging.getLogger(__name__)


class Agent4TrainPredictJoint:
    def __init__(self, pg: PostgresStorage, min_samples: int = 200):
        self.pg = pg
        self.min_samples = min_samples
        self.max_iter = int(os.getenv("LR_MAX_ITER", "200"))
        self.class_weight = os.getenv("LR_CLASS_WEIGHT")
        self.train_horizon = os.getenv("TRAIN_HORIZON_DAYS")

    def run(self) -> None:
        rows = self.pg.fetch_all(
            """
            SELECT fj.event_id, fj.symbol, fj.t0, fj.horizon_days, fj.feature_json,
                   mr.label_up, mr.ret_h
            FROM features_joint fj
            JOIN market_reactions mr ON fj.event_id = mr.event_id
            ORDER BY fj.t0 ASC
            """
        )
        if self.train_horizon:
            try:
                horizon = int(self.train_horizon)
                rows = [row for row in rows if row.get("horizon_days") == horizon]
            except ValueError:
                pass
        report_lines = ["Agent 4 Report"]
        if rows:
            report_lines.append(f"dataset size: {len(rows)}")
            report_lines.append(f"t0 range: {rows[0].get('t0')} -> {rows[-1].get('t0')}")
        else:
            report_lines.append("dataset size: 0")
        if len(rows) < self.min_samples:
            logger.warning("Insufficient samples for training: %s", len(rows))
            report_lines.append(f"status: insufficient samples ({len(rows)})")
            path = make_report_path("agent4")
            write_report(path, "\n".join(report_lines))
            return
        labels = [row.get("label_up") for row in rows]
        if len(set(labels)) < 2:
            logger.warning("Only one class present; skipping training.")
            report_lines.append("status: single class present")
            path = make_report_path("agent4")
            write_report(path, "\n".join(report_lines))
            return
        feature_dicts = [self._sanitize_features(row.get("feature_json") or {}) for row in rows]
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(feature_dicts)
        y = np.array(labels)
        split_idx = int(len(rows) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            logger.warning("Train/validation split lacks class diversity; skipping training.")
            report_lines.append("status: train/val split lacks class diversity")
            path = make_report_path("agent4")
            write_report(path, "\n".join(report_lines))
            return
        class_weight = None
        if self.class_weight == "balanced":
            class_weight = "balanced"
        model = LogisticRegression(max_iter=self.max_iter, class_weight=class_weight)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = self._evaluate(y_val, preds, proba)
        backtest = self._backtest(rows[split_idx:], proba)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_id = f"joint_{timestamp}"
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        artifact_path = os.path.join(artifacts_dir, f"model_{model_id}.joblib")
        joblib.dump({"model": model, "vectorizer": vectorizer}, artifact_path)
        meta_json = {
            "feature_count": len(vectorizer.feature_names_),
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "metrics": metrics,
            "backtest": backtest,
        }
        self.pg.upsert_model(model_id, meta_json)
        predictions = []
        for row, proba_up in zip(rows[split_idx:], proba):
            pred_id = hashlib.md5(f"{model_id}|{row['event_id']}".encode("utf-8")).hexdigest()
            predictions.append(
                {
                    "pred_id": pred_id,
                    "model_id": model_id,
                    "event_id": row.get("event_id"),
                    "symbol": row.get("symbol"),
                    "t0": row.get("t0"),
                    "horizon_days": row.get("horizon_days"),
                    "proba_up": float(proba_up),
                    "meta_json": {"ret_h": row.get("ret_h")},
                }
            )
        inserted = self.pg.upsert_predictions(predictions)
        logger.info("Agent4 stored model %s with %s predictions", model_id, inserted)
        report_lines.extend(
            [
                f"train samples: {len(y_train)}",
                f"val samples: {len(y_val)}",
                f"feature dimensions: {len(vectorizer.feature_names_)}",
                f"model: LogisticRegression(max_iter={self.max_iter}, class_weight={class_weight})",
                f"metrics: {metrics}",
                f"backtest: {backtest}",
            ]
        )
        if rows:
            report_lines.append(f"train t0 range: {rows[0].get('t0')} -> {rows[split_idx - 1].get('t0')}")
            report_lines.append(f"val t0 range: {rows[split_idx].get('t0')} -> {rows[-1].get('t0')}")
        path = make_report_path("agent4")
        write_report(path, "\n".join(report_lines))

    @staticmethod
    def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        }
        try:
            metrics["auc"] = roc_auc_score(y_true, proba)
        except Exception:
            metrics["auc"] = None
        return metrics

    @staticmethod
    def _backtest(rows: list[dict[str, Any]], proba: np.ndarray) -> dict[str, Any]:
        returns = []
        hits = []
        for row, p in zip(rows, proba):
            if p >= 0.55:
                ret = row.get("ret_h")
                if ret is not None:
                    returns.append(ret)
                    hits.append(1 if ret > 0 else 0)
        if not returns:
            return {"n_trades": 0}
        return {
            "n_trades": len(returns),
            "mean_ret_h": float(np.mean(returns)),
            "hit_rate": float(np.mean(hits)),
        }

    @staticmethod
    def _sanitize_features(features: dict[str, Any]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for key, value in features.items():
            if value is None:
                cleaned[key] = 0.0
            else:
                try:
                    cleaned[key] = float(value)
                except Exception:
                    cleaned[key] = 0.0
        return cleaned


def main() -> None:
    settings = load_settings()
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is required")
    pg = PostgresStorage(settings.database_url)
    Agent4TrainPredictJoint(pg).run()


if __name__ == "__main__":
    main()
