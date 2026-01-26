from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.storage.pg import PostgresStorage

logger = logging.getLogger(__name__)


class Agent4TrainPredictJoint:
    def __init__(self, pg: PostgresStorage, min_samples: int = 200):
        self.pg = pg
        self.min_samples = min_samples

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
        if len(rows) < self.min_samples:
            logger.warning("Insufficient samples for training: %s", len(rows))
            return
        labels = [row.get("label_up") for row in rows]
        if len(set(labels)) < 2:
            logger.warning("Only one class present; skipping training.")
            return
        feature_dicts = [row.get("feature_json") or {} for row in rows]
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(feature_dicts)
        y = np.array(labels)
        split_idx = int(len(rows) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            logger.warning("Train/validation split lacks class diversity; skipping training.")
            return
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = self._evaluate(y_val, preds, proba)
        backtest = self._backtest(rows[split_idx:], proba)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_id = f"joint_{timestamp}"
        artifact_path = f"stock-agent-lab/artifacts/model_{model_id}.joblib"
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
