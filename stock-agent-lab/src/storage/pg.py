from __future__ import annotations

import json
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row


class PostgresStorage:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or {})

    def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or {})
                return list(cur.fetchall())

    def upsert_news_raw(self, rows: Iterable[dict[str, Any]]) -> int:
        rows_list = list(rows)
        if not rows_list:
            return 0
        query = """
            INSERT INTO news_raw (id, source, source_country, language, category, publish_date, title, summary, text, url, image, sentiment)
            VALUES (%(id)s, %(source)s, %(source_country)s, %(language)s, %(category)s, %(publish_date)s, %(title)s, %(summary)s, %(text)s, %(url)s, %(image)s, %(sentiment)s)
            ON CONFLICT (id) DO UPDATE SET
                source = EXCLUDED.source,
                source_country = EXCLUDED.source_country,
                language = EXCLUDED.language,
                category = EXCLUDED.category,
                publish_date = EXCLUDED.publish_date,
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                text = EXCLUDED.text,
                url = EXCLUDED.url,
                image = EXCLUDED.image,
                sentiment = EXCLUDED.sentiment
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows_list)
        return len(rows_list)

    def upsert_prices(self, rows: Iterable[dict[str, Any]]) -> int:
        rows_list = list(rows)
        if not rows_list:
            return 0
        query = """
            INSERT INTO prices (symbol, date, open, high, low, close, volume)
            VALUES (%(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows_list)
        return len(rows_list)

    def upsert_news_events(self, rows: Iterable[dict[str, Any]]) -> int:
        rows_list = list(rows)
        if not rows_list:
            return 0
        query = """
            INSERT INTO news_events (event_id, symbol, publish_date, title, url, event_type, sentiment, impact_hint, evidence_json)
            VALUES (%(event_id)s, %(symbol)s, %(publish_date)s, %(title)s, %(url)s, %(event_type)s, %(sentiment)s, %(impact_hint)s, %(evidence_json)s)
            ON CONFLICT (event_id) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                publish_date = EXCLUDED.publish_date,
                title = EXCLUDED.title,
                url = EXCLUDED.url,
                event_type = EXCLUDED.event_type,
                sentiment = EXCLUDED.sentiment,
                impact_hint = EXCLUDED.impact_hint,
                evidence_json = EXCLUDED.evidence_json
        """
        rows_list = [dict(row, evidence_json=json.dumps(row.get("evidence_json", {}))) for row in rows_list]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows_list)
        return len(rows_list)

    def upsert_market_reactions(self, rows: Iterable[dict[str, Any]]) -> int:
        rows_list = list(rows)
        if not rows_list:
            return 0
        query = """
            INSERT INTO market_reactions (reaction_id, event_id, symbol, t0, horizon_days, ret_1d, ret_5d, ret_h, vol_5d, dd_h, label_up, meta_json)
            VALUES (%(reaction_id)s, %(event_id)s, %(symbol)s, %(t0)s, %(horizon_days)s, %(ret_1d)s, %(ret_5d)s, %(ret_h)s, %(vol_5d)s, %(dd_h)s, %(label_up)s, %(meta_json)s)
            ON CONFLICT (reaction_id) DO UPDATE SET
                ret_1d = EXCLUDED.ret_1d,
                ret_5d = EXCLUDED.ret_5d,
                ret_h = EXCLUDED.ret_h,
                vol_5d = EXCLUDED.vol_5d,
                dd_h = EXCLUDED.dd_h,
                label_up = EXCLUDED.label_up,
                meta_json = EXCLUDED.meta_json
        """
        rows_list = [dict(row, meta_json=json.dumps(row.get("meta_json", {}))) for row in rows_list]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows_list)
        return len(rows_list)

    def upsert_features_joint(self, rows: Iterable[dict[str, Any]]) -> int:
        rows_list = list(rows)
        if not rows_list:
            return 0
        query = """
            INSERT INTO features_joint (event_id, symbol, t0, horizon_days, feature_json)
            VALUES (%(event_id)s, %(symbol)s, %(t0)s, %(horizon_days)s, %(feature_json)s)
            ON CONFLICT (event_id) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                t0 = EXCLUDED.t0,
                horizon_days = EXCLUDED.horizon_days,
                feature_json = EXCLUDED.feature_json
        """
        rows_list = [dict(row, feature_json=json.dumps(row.get("feature_json", {}))) for row in rows_list]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows_list)
        return len(rows_list)

    def upsert_model(self, model_id: str, meta_json: dict[str, Any]) -> None:
        query = """
            INSERT INTO models (model_id, meta_json)
            VALUES (%(model_id)s, %(meta_json)s)
            ON CONFLICT (model_id) DO UPDATE SET
                meta_json = EXCLUDED.meta_json
        """
        self.execute(query, {"model_id": model_id, "meta_json": json.dumps(meta_json)})

    def upsert_predictions(self, rows: Iterable[dict[str, Any]]) -> int:
        rows_list = list(rows)
        if not rows_list:
            return 0
        query = """
            INSERT INTO predictions (pred_id, model_id, event_id, symbol, t0, horizon_days, proba_up, meta_json)
            VALUES (%(pred_id)s, %(model_id)s, %(event_id)s, %(symbol)s, %(t0)s, %(horizon_days)s, %(proba_up)s, %(meta_json)s)
            ON CONFLICT (pred_id) DO UPDATE SET
                proba_up = EXCLUDED.proba_up,
                meta_json = EXCLUDED.meta_json
        """
        rows_list = [dict(row, meta_json=json.dumps(row.get("meta_json", {}))) for row in rows_list]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows_list)
        return len(rows_list)

    def upsert_collector_state(
        self,
        source: str,
        symbol: str,
        chunk_start,
        chunk_end,
        offset: int,
        done: bool,
    ) -> None:
        query = """
            INSERT INTO collector_state (source, symbol, chunk_start, chunk_end, offset, done)
            VALUES (%(source)s, %(symbol)s, %(chunk_start)s, %(chunk_end)s, %(offset)s, %(done)s)
            ON CONFLICT (source, symbol, chunk_start, chunk_end) DO UPDATE SET
                offset = EXCLUDED.offset,
                done = EXCLUDED.done,
                updated_at = now()
        """
        self.execute(
            query,
            {
                "source": source,
                "symbol": symbol,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "offset": offset,
                "done": done,
            },
        )

    def get_collector_state(self, source: str, symbol: str, chunk_start, chunk_end) -> dict[str, Any] | None:
        query = """
            SELECT source, symbol, chunk_start, chunk_end, offset, done
            FROM collector_state
            WHERE source = %(source)s AND symbol = %(symbol)s AND chunk_start = %(chunk_start)s AND chunk_end = %(chunk_end)s
        """
        rows = self.fetch_all(
            query,
            {
                "source": source,
                "symbol": symbol,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            },
        )
        return rows[0] if rows else None

    def table_counts(self) -> dict[str, int]:
        tables = [
            "news_raw",
            "prices",
            "news_events",
            "market_reactions",
            "features_joint",
            "models",
            "predictions",
        ]
        counts = {}
        with self.connect() as conn:
            with conn.cursor() as cur:
                for table in tables:
                    cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                    counts[table] = cur.fetchone()["count"]
        return counts

    def latest_price_date(self) -> dict[str, Any] | None:
        rows = self.fetch_all("SELECT MAX(date) AS latest_date FROM prices")
        return rows[0] if rows else None
