from __future__ import annotations

import logging
from datetime import date, timedelta

from src.collectors.corp_actions_collector import CorpActionsCollector
from src.collectors.vn30_universe import VN30_SYMBOLS
from src.config import Settings
from src.storage.pg import PostgresStorage
from src.utils.http import HttpClient

logger = logging.getLogger(__name__)


def collect_corp_actions(
    pg: PostgresStorage,
    settings: Settings,
    symbols: list[str],
    start: date,
    end: date,
    smoke_mode: bool = False,
) -> None:
    if smoke_mode:
        symbols = settings.smoke_symbols
        start = end - timedelta(days=settings.smoke_days)
        logger.info("Smoke mode corp actions: %s %s-%s", symbols, start, end)
    http = HttpClient(settings.http_timeout_secs, settings.http_max_retries, settings.http_sleep_secs)
    collector = CorpActionsCollector(pg, http, settings.http_sleep_secs)
    collector.collect(symbols, start, end)


def collect_corp_actions_default(
    pg: PostgresStorage, settings: Settings, start: date, end: date, smoke_mode: bool = False
) -> None:
    collect_corp_actions(pg, settings, VN30_SYMBOLS, start, end, smoke_mode)
