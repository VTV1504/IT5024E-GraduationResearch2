from __future__ import annotations

import logging
from datetime import date, timedelta

from src.collectors.worldnews_collector import WorldNewsCollector
from src.collectors.vn30_universe import VN30_SYMBOLS
from src.config import Settings
from src.storage.pg import PostgresStorage
from src.utils.http import HttpClient

logger = logging.getLogger(__name__)


def collect_news(
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
        logger.info("Smoke mode news: %s %s-%s", symbols, start, end)
    http = HttpClient(settings.http_timeout_secs, settings.http_max_retries, settings.http_sleep_secs)
    collector = WorldNewsCollector(pg, http, settings.worldnews_api_key, settings.worldnews_page_size)
    collector.collect(symbols, start, end, settings.worldnews_source_country, settings.worldnews_language)


def collect_news_default(pg: PostgresStorage, settings: Settings, start: date, end: date, smoke_mode: bool = False) -> None:
    symbols = VN30_SYMBOLS
    collect_news(pg, settings, symbols, start, end, smoke_mode)
