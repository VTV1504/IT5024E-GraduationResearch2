from __future__ import annotations

import logging
from datetime import date, timedelta

import os

from src.collectors.news_sitemap_backfill import NewsSitemapBackfill, SOURCE_CONFIG
from src.config import Settings
from src.storage.pg import PostgresStorage
from src.utils.http import HttpClient
from src.collectors.vn30_universe import VN30_SYMBOLS

logger = logging.getLogger(__name__)


def collect_news(
    pg: PostgresStorage,
    settings: Settings,
    start: date,
    end: date,
    smoke_mode: bool = False,
    symbols_override: list[str] | None = None,
) -> None:
    symbols = symbols_override or VN30_SYMBOLS
    http = HttpClient(settings.http_timeout_secs, settings.http_max_retries, settings.http_sleep_secs)
    news_start = start
    news_end = end
    min_chars = 200 if smoke_mode else 800
    max_urls_per_source = None
    deepen = os.getenv("NEWS_DEEPEN", "0") == "1"
    deepen_limit = int(os.getenv("NEWS_DEEPEN_PER_SYMBOL", "20"))
    news_mode = os.getenv("NEWS_MODE", "category").lower()
    hard_filter = os.getenv("NEWS_HARD_FILTER", "1") == "1"
    max_pages_per_entry = int(os.getenv("MAX_PAGES_PER_ENTRY", "2000"))
    if smoke_mode:
        max_pages_per_entry = int(os.getenv("SMOKE_MAX_PAGES_PER_ENTRY", "50"))
    if smoke_mode:
        symbols = settings.smoke_symbols
        news_start = end - timedelta(days=settings.smoke_days)
        max_urls_per_source = int(os.getenv("SMOKE_MAX_URLS_PER_SOURCE", "30"))
        logger.info("Smoke mode news: %s to %s (cap %s/source)", news_start, news_end, max_urls_per_source)
    else:
        max_urls_env = os.getenv("MAX_URLS_PER_SOURCE")
        if max_urls_env:
            try:
                max_urls_per_source = int(max_urls_env)
            except ValueError:
                logger.warning("Invalid MAX_URLS_PER_SOURCE=%s; ignoring", max_urls_env)
    for source in ("cafef", "vietstock", "vnexpress"):
        logger.info("Collecting %s news (%s)", source, SOURCE_CONFIG[source]["base_domain"])
        collector = NewsSitemapBackfill(
            pg,
            http,
            settings.http_sleep_secs,
            news_start,
            news_end,
            min_content_chars=min_chars,
            max_urls_per_source=max_urls_per_source,
            hard_filter=hard_filter,
        )
        collector.run_source(
            source,
            deepen_symbols=symbols if deepen and not smoke_mode else None,
            deepen_limit=deepen_limit,
            discover_sitemaps=(news_mode == "sitemap"),
            category_crawl=(news_mode == "category"),
            max_pages_per_entry=max_pages_per_entry,
        )


def collect_news_default(pg: PostgresStorage, settings: Settings, start: date, end: date, smoke_mode: bool = False) -> None:
    collect_news(pg, settings, start, end, smoke_mode)


def collect_news_deepen(
    pg: PostgresStorage,
    settings: Settings,
    start: date,
    end: date,
    symbols: list[str],
    per_symbol_limit: int,
) -> None:
    http = HttpClient(settings.http_timeout_secs, settings.http_max_retries, settings.http_sleep_secs)
    collector = NewsSitemapBackfill(
        pg,
        http,
        settings.http_sleep_secs,
        start,
        end,
        min_content_chars=800,
        max_urls_per_source=per_symbol_limit,
    )
    for source in ("cafef", "vietstock", "vnexpress"):
        collector.run_source(
            source,
            deepen_symbols=symbols,
            deepen_limit=per_symbol_limit,
            discover_sitemaps=False,
        )


def collect_news_deepen_months(
    pg: PostgresStorage,
    settings: Settings,
    symbols_by_month: dict[str, list[str]],
    per_symbol_limit: int,
) -> None:
    http = HttpClient(settings.http_timeout_secs, settings.http_max_retries, settings.http_sleep_secs)
    for ym, symbols in symbols_by_month.items():
        symbols = sorted(set(symbols))
        start = date.fromisoformat(f"{ym}-01")
        if start.month == 12:
            end = date(start.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(start.year, start.month + 1, 1) - timedelta(days=1)
        collector = NewsSitemapBackfill(
            pg,
            http,
            settings.http_sleep_secs,
            start,
            end,
            min_content_chars=800,
            max_urls_per_source=per_symbol_limit,
        )
        for source in ("cafef", "vietstock", "vnexpress"):
            collector.run_source(
                source,
                deepen_symbols=symbols,
                deepen_limit=per_symbol_limit,
                discover_sitemaps=False,
            )
