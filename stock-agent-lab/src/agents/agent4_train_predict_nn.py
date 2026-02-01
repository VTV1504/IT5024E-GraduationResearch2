from __future__ import annotations

import hashlib
import logging
import os
import random
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from src.config import load_settings
from src.storage.pg import PostgresStorage
from src.utils.reporting import make_report_path, write_report

logger = logging.getLogger(__name__)


class Agent4TrainPredictNN:
    def __init__(self, pg: PostgresStorage, min_samples: int = 200, seed: int = 42):
        self.pg = pg
        self.min_samples = min_samples
        self.seed = seed
        self.epoch_logs: list[dict[str, Any]] = []
        self.train_horizon = os.getenv("TRAIN_HORIZON_DAYS")

    def run(self) -> dict[str, Any] | None:
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
        report_lines = ["Agent 4 NN Report"]
        if rows:
            report_lines.append(f"dataset size: {len(rows)}")
            report_lines.append(f"t0 range: {rows[0].get('t0')} -> {rows[-1].get('t0')}")
        else:
            report_lines.append("dataset size: 0")
        if len(rows) < self.min_samples:
            logger.warning("NN: insufficient samples for training: %s", len(rows))
            report_lines.append(f"status: insufficient samples ({len(rows)})")
            path = make_report_path("agent4_nn")
            write_report(path, "\n".join(report_lines))
            return None
        labels = [row.get("label_up") for row in rows]
        if len(set(labels)) < 2:
            logger.warning("NN: only one class present; skipping training.")
            report_lines.append("status: single class present")
            path = make_report_path("agent4_nn")
            write_report(path, "\n".join(report_lines))
            return None
        feature_dicts = [self._sanitize_features(row.get("feature_json") or {}) for row in rows]
        vectorizer = DictVectorizer(sparse=True)
        X_sparse = vectorizer.fit_transform(feature_dicts)
        y = np.array(labels)
        split_idx = int(len(rows) * 0.8)
        if split_idx <= 0 or split_idx >= len(rows):
            logger.warning("NN: insufficient data for train/validation split.")
            report_lines.append("status: insufficient train/validation split")
            path = make_report_path("agent4_nn")
            write_report(path, "\n".join(report_lines))
            return None
        X_train = X_sparse[:split_idx]
        X_val = X_sparse[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]
        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            logger.warning("NN: train/validation split lacks class diversity; skipping training.")
            report_lines.append("status: train/validation split lacks class diversity")
            path = make_report_path("agent4_nn")
            write_report(path, "\n".join(report_lines))
            return None

        svd = None
        if X_train.shape[1] > 20000:
            svd = TruncatedSVD(n_components=256, random_state=self.seed)
            X_train_dense = svd.fit_transform(X_train)
            X_val_dense = svd.transform(X_val)
        else:
            X_train_dense = X_train.toarray()
            X_val_dense = X_val.toarray()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_dense)
        X_val_scaled = scaler.transform(X_val_dense)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        self.epoch_logs = []
        model, val_proba = self._train_torch(X_train_scaled, y_train, X_val_scaled, y_val)
        if model is None or val_proba is None:
            model, val_proba = self._train_sklearn(X_train_scaled, y_train, X_val_scaled, y_val)
        if model is None or val_proba is None:
            return None

        preds = (val_proba >= 0.5).astype(int)
        metrics = self._evaluate(y_val, preds, val_proba)
        backtest = self._backtest(rows[split_idx:], val_proba)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_id = f"mlp_96_48_24_{timestamp}"
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        artifact_path = os.path.join(artifacts_dir, f"mlp_{model_id}.joblib")
        artifact_payload = {
            "vectorizer": vectorizer,
            "scaler": scaler,
            "svd": svd,
            "model_type": "MLP_96_48_24",
        }
        if hasattr(model, "state_dict"):
            artifact_payload["model_state"] = model.state_dict()
            artifact_payload["input_dim"] = int(X_train_scaled.shape[1])
        else:
            artifact_payload["model"] = model
        joblib.dump(artifact_payload, artifact_path)

        meta_json = {
            "model_type": "MLP_96_48_24",
            "feature_dim": int(X_train_scaled.shape[1]),
            "scaler_used": True,
            "svd_used": svd is not None,
            "train_range": {
                "start": rows[0].get("t0").isoformat() if rows[0].get("t0") else None,
                "end": rows[split_idx - 1].get("t0").isoformat() if rows[split_idx - 1].get("t0") else None,
            },
            "val_range": {
                "start": rows[split_idx].get("t0").isoformat() if rows[split_idx].get("t0") else None,
                "end": rows[-1].get("t0").isoformat() if rows[-1].get("t0") else None,
            },
            "metrics": metrics,
            "backtest": backtest,
        }
        self.pg.upsert_model(model_id, meta_json)
        predictions = []
        for row, proba_up in zip(rows[split_idx:], val_proba):
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
                    "meta_json": {"ret_h": row.get("ret_h"), "model_type": "MLP_96_48_24"},
                }
            )
        inserted = self.pg.upsert_predictions(predictions)
        logger.info("NN: stored model %s with %s predictions", model_id, inserted)
        report_lines.extend(
            [
                f"train samples: {len(y_train)}",
                f"val samples: {len(y_val)}",
                f"feature dimensions: {int(X_train_scaled.shape[1])}",
                f"model: MLP_96_48_24",
                f"metrics: {metrics}",
                f"backtest: {backtest}",
                f"train t0 range: {rows[0].get('t0')} -> {rows[split_idx - 1].get('t0')}",
                f"val t0 range: {rows[split_idx].get('t0')} -> {rows[-1].get('t0')}",
            ]
        )
        if self.epoch_logs:
            report_lines.append("")
            report_lines.append("Epoch logs:")
            for entry in self.epoch_logs:
                report_lines.append(
                    f"- epoch {entry['epoch']}: train_loss={entry['train_loss']}, "
                    f"val_loss={entry['val_loss']}, val_auc={entry.get('val_auc')}"
                )
        path = make_report_path("agent4_nn")
        write_report(path, "\n".join(report_lines))
        return {"model_id": model_id, "metrics": metrics, "backtest": backtest}

    def _train_torch(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> tuple[Any | None, np.ndarray | None]:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except Exception as exc:
            logger.warning("NN: torch not available (%s); falling back to sklearn MLP.", exc)
            return None, None

        self._set_seeds(torch)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCELoss()

        best_metric = None
        patience = 8
        patience_left = patience
        best_state = None

        batch_size = 128
        epochs = 80
        for epoch in range(epochs):
            model.train()
            indices = torch.randperm(X_train_t.size(0))
            epoch_losses = []
            for start in range(0, X_train_t.size(0), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_x = X_train_t[batch_idx]
                batch_y = y_train_t[batch_idx]
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_probs = val_outputs.detach().cpu().numpy().reshape(-1)
            try:
                val_auc = roc_auc_score(y_val, val_probs)
                current_metric = val_auc
            except Exception:
                val_auc = None
                current_metric = -val_loss
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else None
            self.epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                }
            )

            if best_metric is None or current_metric > best_metric:
                best_metric = current_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1

            if patience_left <= 0:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_t).detach().cpu().numpy().reshape(-1)
        return model, val_probs

    def _train_sklearn(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> tuple[Any | None, np.ndarray | None]:
        logger.warning("NN: using sklearn MLPClassifier (dropout not supported; using L2 + early stopping).")
        model = MLPClassifier(
            hidden_layer_sizes=(96, 48, 24),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=self.seed,
        )
        model.fit(X_train, y_train)
        try:
            val_probs = model.predict_proba(X_val)[:, 1]
        except Exception as exc:
            logger.warning("NN: sklearn MLP failed to produce probabilities: %s", exc)
            return None, None
        return model, val_probs

    def _set_seeds(self, torch_module: Any) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch_module.manual_seed(self.seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(self.seed)

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
    Agent4TrainPredictNN(pg).run()


if __name__ == "__main__":
    main()
