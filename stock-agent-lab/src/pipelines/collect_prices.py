from __future__ import annotations

import logging
from datetime import date, timedelta

from src.collectors.fireant_price_collector import FireAntPriceCollector
from src.collectors.vn30_universe import VN30_SYMBOLS
from src.config import Settings
from src.storage.pg import PostgresStorage
from src.utils.http import HttpClient
from src.utils.coverage import find_missing_trading_ranges

logger = logging.getLogger(__name__)


def collect_prices(
    pg: PostgresStorage,
    settings: Settings,
    symbols: list[str],
    start: date,
    end: date,
    smoke_mode: bool = False,
) -> None:
    gap_fill = not smoke_mode
    if smoke_mode:
        symbols = settings.smoke_symbols
        start = end - timedelta(days=settings.smoke_days)
        logger.info("Smoke mode prices: %s %s-%s", symbols, start, end)
    http = HttpClient(settings.http_timeout_secs, settings.http_max_retries, settings.http_sleep_secs)
    collector = FireAntPriceCollector(pg, http, settings.fireant_base_url)
    collector.collect(symbols, start, end)
    if gap_fill:
        _gap_fill_prices(pg, collector, symbols, start, end)
    _log_price_summary(pg, symbols)


def collect_prices_default(pg: PostgresStorage, settings: Settings, start: date, end: date, smoke_mode: bool = False) -> None:
    symbols = VN30_SYMBOLS
    collect_prices(pg, settings, symbols, start, end, smoke_mode)


def _gap_fill_prices(
    pg: PostgresStorage,
    collector: FireAntPriceCollector,
    symbols: list[str],
    start: date,
    end: date,
) -> None:
    max_ranges_per_symbol = 12
    for symbol in symbols:
        ranges = find_missing_trading_ranges(pg, symbol, start, end)
        if not ranges:
            continue
        logger.info("Price gaps for %s: %s ranges", symbol, len(ranges))
        for gap_start, gap_end in ranges[:max_ranges_per_symbol]:
            rows = collector._fetch_symbol(symbol, gap_start, gap_end)
            if rows:
                inserted = pg.upsert_prices(rows)
                logger.info("Gap fill prices %s %s-%s rows=%s", symbol, gap_start, gap_end, inserted)


def _log_price_summary(pg: PostgresStorage, symbols: list[str]) -> None:
    for symbol in symbols:
        rows = pg.fetch_all(
            """
            SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS cnt
            FROM prices
            WHERE symbol = %(symbol)s
            """,
            {"symbol": symbol},
        )
        if rows:
            logger.info(
                "Price summary %s: %s to %s (%s rows)",
                symbol,
                rows[0].get("min_date"),
                rows[0].get("max_date"),
                rows[0].get("cnt"),
            )


def gap_fill_prices(
    pg: PostgresStorage,
    settings: Settings,
    symbols: list[str],
    start: date,
    end: date,
) -> None:
    http = HttpClient(settings.http_timeout_secs, settings.http_max_retries, settings.http_sleep_secs)
    collector = FireAntPriceCollector(pg, http, settings.fireant_base_url)
    _gap_fill_prices(pg, collector, symbols, start, end)
