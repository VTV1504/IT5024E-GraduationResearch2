from __future__ import annotations

import logging
from datetime import date
from typing import Any

from dateutil.parser import parse

from src.collectors.vn30_universe import SYMBOL_ALIASES
from src.storage.pg import PostgresStorage
from src.utils.http import HttpClient
from src.utils.time_ranges import monthly_ranges

logger = logging.getLogger(__name__)


class WorldNewsCollector:
    def __init__(self, pg: PostgresStorage, http: HttpClient, api_key: str, page_size: int):
        self.pg = pg
        self.http = http
        self.api_key = api_key
        self.page_size = page_size
        self.base_url = "https://api.worldnewsapi.com/search-news"

    def _build_query(self, symbol: str) -> str:
        aliases = SYMBOL_ALIASES.get(symbol, [])
        parts = [symbol] + aliases
        quoted = [f'"{p}"' for p in parts]
        return " OR ".join(quoted)

    def collect(self, symbols: list[str], start: date, end: date, source_country: str, language: str) -> None:
        if not self.api_key:
            logger.warning("WORLDNEWS_API_KEY missing; skipping news collection.")
            return
        for symbol in symbols:
            query = self._build_query(symbol)
            for chunk_start, chunk_end in monthly_ranges(start, end):
                state = self.pg.get_collector_state("worldnews", symbol, chunk_start, chunk_end)
                if state and state.get("done"):
                    logger.info("Skip %s %s-%s: already done", symbol, chunk_start, chunk_end)
                    continue
                offset = state.get("offset", 0) if state else 0
                done = False
                while not done:
                    params = {
                        "text": query,
                        "source-country": source_country,
                        "language": language,
                        "earliest-publish-date": chunk_start.isoformat(),
                        "latest-publish-date": chunk_end.isoformat(),
                        "sort": "publish-time",
                        "offset": offset,
                        "number": self.page_size,
                    }
                    headers = {"x-api-key": self.api_key}
                    try:
                        response = self.http.get(self.base_url, params=params, headers=headers)
                        payload = response.json()
                    except Exception as exc:
                        logger.warning("WorldNews request failed for %s offset %s: %s", symbol, offset, exc)
                        break
                    news_items = payload.get("news", []) or []
                    if not news_items:
                        done = True
                        self.pg.upsert_collector_state("worldnews", symbol, chunk_start, chunk_end, offset, True)
                        break
                    rows = []
                    for item in news_items:
                        try:
                            publish_date = parse(item.get("publish_date")) if item.get("publish_date") else None
                        except Exception:
                            publish_date = None
                        rows.append(
                            {
                                "id": item.get("id"),
                                "source": item.get("source"),
                                "source_country": item.get("source_country"),
                                "language": item.get("language"),
                                "category": item.get("category"),
                                "publish_date": publish_date,
                                "title": item.get("title"),
                                "summary": item.get("summary"),
                                "text": item.get("text"),
                                "url": item.get("url"),
                                "image": item.get("image"),
                                "sentiment": item.get("sentiment"),
                            }
                        )
                    inserted = self.pg.upsert_news_raw([row for row in rows if row.get("id") is not None])
                    logger.info("WorldNews %s %s-%s offset %s: %s rows", symbol, chunk_start, chunk_end, offset, inserted)
                    offset += self.page_size
                    self.pg.upsert_collector_state("worldnews", symbol, chunk_start, chunk_end, offset, False)
                    if len(news_items) < self.page_size:
                        done = True
                        self.pg.upsert_collector_state("worldnews", symbol, chunk_start, chunk_end, offset, True)
