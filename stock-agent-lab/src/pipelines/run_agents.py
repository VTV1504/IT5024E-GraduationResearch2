from __future__ import annotations

import json
import logging

from src.agents.agent1_news_event import Agent1NewsEvent
from src.agents.agent2_market_react import Agent2MarketReact
from src.agents.agent3_features_joint import Agent3FeaturesJoint
from src.agents.agent4_train_predict_joint import Agent4TrainPredictJoint
from src.agents.agent4_train_predict_nn import Agent4TrainPredictNN
from src.storage.pg import PostgresStorage

logger = logging.getLogger(__name__)


def run_agents(pg: PostgresStorage, horizon_days: int = 40, model_mode: str = "both") -> None:
    Agent1NewsEvent(pg).run()
    Agent2MarketReact(pg, horizon_days=horizon_days).run()
    Agent3FeaturesJoint(pg, horizon_days=horizon_days).run()
    lr_summary = None
    if model_mode in ("lr", "both"):
        Agent4TrainPredictJoint(pg).run()
        lr_summary = _fetch_lr_summary(pg)
    nn_result = None
    if model_mode in ("nn", "both"):
        nn_result = Agent4TrainPredictNN(pg).run()
    if model_mode == "both":
        lr_summary = lr_summary or "LR: no training result"
        nn_summary = (
            f"NN: AUC={nn_result['metrics'].get('auc')}, Acc={nn_result['metrics'].get('accuracy')}, "
            f"Prec={nn_result['metrics'].get('precision')}, Rec={nn_result['metrics'].get('recall')}, "
            f"mean_ret_h={nn_result['backtest'].get('mean_ret_h')}, hitrate={nn_result['backtest'].get('hit_rate')}"
            if nn_result
            else "NN: no training result"
        )
        logger.info("Model comparison: %s | %s", lr_summary, nn_summary)
    logger.info("Agents pipeline complete")


def _fetch_lr_summary(pg: PostgresStorage) -> str:
    rows = pg.fetch_all(
        """
        SELECT model_id, meta_json
        FROM models
        WHERE model_id LIKE 'joint_%%'
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    if not rows:
        return "LR: no training result"
    meta = rows[0].get("meta_json")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    backtest = meta.get("backtest", {}) if isinstance(meta, dict) else {}
    return (
        f"LR: AUC={metrics.get('auc')}, Acc={metrics.get('accuracy')}, "
        f"Prec={metrics.get('precision')}, Rec={metrics.get('recall')}, "
        f"mean_ret_h={backtest.get('mean_ret_h')}, hitrate={backtest.get('hit_rate')}"
    )
