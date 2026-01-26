from __future__ import annotations

import logging

from src.agents.agent1_news_event import Agent1NewsEvent
from src.agents.agent2_market_react import Agent2MarketReact
from src.agents.agent3_features_joint import Agent3FeaturesJoint
from src.agents.agent4_train_predict_joint import Agent4TrainPredictJoint
from src.storage.pg import PostgresStorage

logger = logging.getLogger(__name__)


def run_agents(pg: PostgresStorage, horizon_days: int = 40) -> None:
    Agent1NewsEvent(pg).run()
    Agent2MarketReact(pg, horizon_days=horizon_days).run()
    Agent3FeaturesJoint(pg, horizon_days=horizon_days).run()
    Agent4TrainPredictJoint(pg).run()
    logger.info("Agents pipeline complete")
