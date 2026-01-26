from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from src.storage.pg import PostgresStorage
from src.utils.features import moving_average, rsi, safe_std, zscore

logger = logging.getLogger(__name__)

KEYWORDS = [
    "profit",
    "revenue",
    "dividend",
    "investigation",
    "interest rate",
    "inflation",
    "lãi suất",
    "doanh thu",
    "cổ tức",
]


class Agent3FeaturesJoint:
    def __init__(self, pg: PostgresStorage, horizon_days: int = 40):
        self.pg = pg
        self.horizon_days = horizon_days
        self.vectorizer = HashingVectorizer(n_features=128, alternate_sign=False, norm=None)

    def run(self) -> None:
        rows = self.pg.fetch_all(
            """
            SELECT ne.*, mr.t0, nr.title AS raw_title, nr.summary AS raw_summary, nr.text AS raw_text
            FROM news_events ne
            LEFT JOIN market_reactions mr ON ne.event_id = mr.event_id
            LEFT JOIN news_raw nr ON (nr.id = (ne.evidence_json->>'news_id')::bigint)
            WHERE mr.t0 IS NOT NULL
            """
        )
        if not rows:
            logger.warning("No news_events with market reactions for features.")
            return
        feature_rows: list[dict[str, Any]] = []
        for row in rows:
            symbol = row.get("symbol")
            t0 = row.get("t0")
            if not symbol or not t0:
                continue
            news_features = self._news_features(row)
            price_features = self._price_features(symbol, t0)
            feature_json = {**news_features, **price_features}
            feature_rows.append(
                {
                    "event_id": row.get("event_id"),
                    "symbol": symbol,
                    "t0": t0,
                    "horizon_days": self.horizon_days,
                    "feature_json": feature_json,
                }
            )
        inserted = self.pg.upsert_features_joint(feature_rows)
        logger.info("Agent3 inserted %s joint features", inserted)

    def _news_features(self, row: dict[str, Any]) -> dict[str, Any]:
        features: dict[str, Any] = {}
        event_type = row.get("event_type") or "other"
        for label in ["earnings", "governance", "mna", "legal", "macro", "rumor", "other"]:
            features[f"event_{label}"] = 1 if event_type == label else 0
        sentiment = row.get("sentiment")
        impact_hint = row.get("impact_hint")
        features["sentiment"] = sentiment
        features["impact_hint"] = impact_hint
        title = row.get("title") or row.get("raw_title") or ""
        summary = row.get("raw_summary") or ""
        text = row.get("raw_text") or ""
        combined = f"{title} {summary}".strip()
        features["title_len"] = len(title)
        features["text_len"] = len(text)
        lowered = combined.lower()
        for keyword in KEYWORDS:
            features[f"kw_{keyword.replace(' ', '_')}"] = lowered.count(keyword)
        hashed = self.vectorizer.transform([combined])
        hashed = hashed.toarray()[0]
        for idx, value in enumerate(hashed):
            if value:
                features[f"h_{idx}"] = float(value)
        return features

    def _price_features(self, symbol: str, t0: date) -> dict[str, Any]:
        rows = self.pg.fetch_all(
            """
            SELECT date, open, close, volume
            FROM prices
            WHERE symbol = %(symbol)s AND date <= %(t0)s
            ORDER BY date ASC
            """,
            {"symbol": symbol, "t0": t0},
        )
        if not rows:
            return {}
        closes = [row.get("close") for row in rows if row.get("close") is not None]
        volumes = [row.get("volume") for row in rows if row.get("volume") is not None]
        if len(closes) < 2:
            return {}
        # use data up to t0-1 for pre-event
        pre_closes = closes[:-1] if len(closes) > 1 else closes
        pre_volumes = volumes[:-1] if len(volumes) > 1 else volumes
        features: dict[str, Any] = {}
        features["ret_5d_pre"] = self._ret_window(pre_closes, 5)
        features["ret_20d_pre"] = self._ret_window(pre_closes, 20)
        features["vol_20d_pre"] = self._vol_window(pre_closes, 20)
        features["volume_z_20d"] = zscore(pre_volumes[-1] if pre_volumes else None, pre_volumes[-20:])
        ma5 = moving_average(pre_closes, 5)
        ma20 = moving_average(pre_closes, 20)
        if ma5 is not None and ma20:
            features["ma_ratio_5_20"] = ma5 / ma20 - 1
        else:
            features["ma_ratio_5_20"] = None
        features["rsi_14_pre"] = rsi(pre_closes, 14)
        if len(rows) >= 2:
            prev_close = rows[-2].get("close")
            open_t0 = rows[-1].get("open")
            if prev_close and open_t0:
                features["gap_open_pre"] = open_t0 / prev_close - 1
            else:
                features["gap_open_pre"] = None
        return features

    @staticmethod
    def _ret_window(values: list[float], window: int) -> float | None:
        if len(values) <= window:
            return None
        if values[-window - 1] in (None, 0):
            return None
        return values[-1] / values[-window - 1] - 1

    @staticmethod
    def _vol_window(values: list[float], window: int) -> float | None:
        if len(values) <= window:
            return None
        returns = []
        for i in range(-window, 0):
            prev = values[i - 1]
            curr = values[i]
            if prev:
                returns.append(curr / prev - 1)
        return safe_std(returns)
