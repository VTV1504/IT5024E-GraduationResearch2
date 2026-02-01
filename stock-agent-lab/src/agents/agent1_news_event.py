from __future__ import annotations

import hashlib
import logging
import re
from datetime import date
from typing import Any

from src.collectors.vn30_universe import SYMBOL_ALIASES
from src.storage.pg import PostgresStorage
from src.utils.reporting import make_report_path, write_report

logger = logging.getLogger(__name__)

EVENT_PATTERNS = {
    "earnings": ["profit", "earnings", "revenue", "lợi nhuận", "doanh thu"],
    "governance": ["board", "ceo", "appointment", "bổ nhiệm", "management"],
    "mna": ["acquisition", "merger", "mua lại", "sáp nhập"],
    "legal": ["lawsuit", "regulator", "investigation", "khởi tố", "phạt"],
    "macro": ["interest rate", "inflation", "macro", "lãi suất", "tỷ giá"],
    "rumor": ["rumor", "tin đồn"],
}

CORP_ACTION_KEYWORDS = [
    "chot quyen",
    "gdkhq",
    "co tuc",
    "tam ung co tuc",
    "quyen mua",
    "phat hanh them",
    "esop",
    "thuong co phieu",
    "chia tach",
    "gop co phieu",
    "niem yet bo sung",
]

SENTIMENT_WORDS = {
    "positive": ["beat", "growth", "up", "record", "tăng", "kỷ lục", "lợi nhuận"],
    "negative": ["decline", "down", "loss", "drop", "giảm", "thua lỗ"],
}


class Agent1NewsEvent:
    def __init__(self, pg: PostgresStorage):
        self.pg = pg

    def run(self, start: date | None = None, end: date | None = None) -> None:
        query = "SELECT * FROM news_raw"
        params: dict[str, Any] = {}
        if start and end:
            query += " WHERE publish_date >= %(start)s AND publish_date <= %(end)s"
            params = {"start": start, "end": end}
        rows = self.pg.fetch_all(query, params)
        if not rows:
            logger.warning("No news_raw rows to process.")
            self._write_report(params if params else None, 0)
            return
        events = []
        for row in rows:
            text_blob = " ".join(
                [str(row.get("title") or ""), str(row.get("summary") or ""), str(row.get("text") or "")]
            )
            symbols = self._detect_symbols(text_blob)
            if not symbols:
                continue
            event_type = self._classify_event(text_blob)
            sentiment = row.get("sentiment")
            if sentiment is None:
                sentiment = self._heuristic_sentiment(text_blob)
            impact_hint = self._impact_hint(event_type, text_blob)
            for symbol in symbols:
                event_id = self._event_id(symbol, row)
                events.append(
                    {
                        "event_id": event_id,
                        "symbol": symbol,
                        "publish_date": row.get("publish_date"),
                        "title": row.get("title"),
                        "url": row.get("url"),
                        "event_type": event_type,
                        "sentiment": sentiment,
                        "impact_hint": impact_hint,
                        "evidence_json": {"news_id": row.get("id")},
                    }
                )
        inserted = self.pg.upsert_news_events(events)
        logger.info("Agent1 inserted %s news_events", inserted)
        self._write_report(params if params else None, inserted)

    def _write_report(self, date_params: dict[str, Any] | None, inserted: int) -> None:
        range_clause = ""
        if date_params:
            range_clause = " WHERE publish_date >= %(start)s AND publish_date <= %(end)s"
        total_news = self.pg.fetch_all(f"SELECT COUNT(*) AS cnt FROM news_raw{range_clause}", date_params or {})
        total_news_count = total_news[0]["cnt"] if total_news else 0
        total_events = self.pg.fetch_all(f"SELECT COUNT(*) AS cnt FROM news_events{range_clause}", date_params or {})
        total_events_count = total_events[0]["cnt"] if total_events else 0
        per_symbol = self.pg.fetch_all(
            """
            SELECT symbol, COUNT(*) AS cnt
            FROM news_events
            GROUP BY symbol
            ORDER BY cnt DESC
            LIMIT 10
            """
        )
        per_month = self.pg.fetch_all(
            """
            SELECT to_char(date_trunc('month', publish_date), 'YYYY-MM') AS ym, COUNT(*) AS cnt
            FROM news_events
            GROUP BY ym
            ORDER BY ym
            """
        )
        type_dist = self.pg.fetch_all(
            """
            SELECT event_type, COUNT(*) AS cnt
            FROM news_events
            GROUP BY event_type
            ORDER BY cnt DESC
            """
        )
        covered = self.pg.fetch_all(
            """
            SELECT COUNT(DISTINCT (evidence_json->>'news_id')) AS cnt
            FROM news_events
            """
        )
        covered_count = covered[0]["cnt"] if covered else 0
        recall_pct = (covered_count / total_news_count * 100) if total_news_count else 0.0
        lines = [
            "Agent 1 Report",
            f"news_raw processed: {total_news_count}",
            f"news_events upserted: {inserted}",
            f"total news_events: {total_events_count}",
            f"percent articles -> events: {recall_pct:.2f}%",
            "",
            "Top 10 symbols:",
            *[f"- {row['symbol']}: {row['cnt']}" for row in per_symbol],
            "",
            "Events per month:",
            *[f"- {row['ym']}: {row['cnt']}" for row in per_month],
            "",
            "Event type distribution:",
            *[f"- {row['event_type']}: {row['cnt']}" for row in type_dist],
        ]
        path = make_report_path("agent1")
        write_report(path, "\n".join(lines))

    def _detect_symbols(self, text: str) -> list[str]:
        detected = []
        for symbol, aliases in SYMBOL_ALIASES.items():
            for alias in [symbol] + aliases:
                pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
                if pattern.search(text):
                    detected.append(symbol)
                    break
        return sorted(set(detected))

    def _classify_event(self, text: str) -> str:
        lowered = text.lower()
        if any(keyword in lowered for keyword in CORP_ACTION_KEYWORDS):
            return "corp_action"
        for event_type, keywords in EVENT_PATTERNS.items():
            if any(keyword in lowered for keyword in keywords):
                return event_type
        return "other"

    def _heuristic_sentiment(self, text: str) -> float:
        lowered = text.lower()
        score = 0
        for word in SENTIMENT_WORDS["positive"]:
            if word in lowered:
                score += 1
        for word in SENTIMENT_WORDS["negative"]:
            if word in lowered:
                score -= 1
        return float(score)

    def _impact_hint(self, event_type: str, text: str) -> float:
        base = {
            "earnings": 0.6,
            "governance": 0.4,
            "mna": 0.7,
            "legal": 0.5,
            "macro": 0.5,
            "rumor": 0.3,
            "other": 0.2,
        }.get(event_type, 0.2)
        numbers = re.findall(r"\d+", text)
        boost = min(len(numbers) * 0.05, 0.3)
        return base + boost

    def _event_id(self, symbol: str, row: dict[str, Any]) -> str:
        raw = f"{symbol}|{row.get('publish_date')}|{row.get('url')}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
