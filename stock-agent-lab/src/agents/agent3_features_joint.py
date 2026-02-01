from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from src.storage.pg import PostgresStorage
from src.utils.features import moving_average, rsi, safe_std, zscore
from src.utils.reporting import make_report_path, write_report

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

CORP_ACTION_TYPES = [
    "dividend_cash",
    "dividend_stock",
    "rights_issue",
    "bonus_issue",
    "esop",
    "split",
    "consolidation",
    "additional_listing",
    "record_date",
    "ex_rights",
    "other",
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
            self._write_report(0, 0, 0, 0, 0, {})
            return
        symbols = sorted({row.get("symbol") for row in rows if row.get("symbol")})
        min_t0 = min(row.get("t0") for row in rows if row.get("t0"))
        max_t0 = max(row.get("t0") for row in rows if row.get("t0"))
        corp_actions = self._load_corp_actions(symbols, min_t0, max_t0)
        feature_rows: list[dict[str, Any]] = []
        missing_news = 0
        missing_price = 0
        missing_corp = 0
        feature_non_null: dict[str, int] = {}
        for row in rows:
            symbol = row.get("symbol")
            t0 = row.get("t0")
            if not symbol or not t0:
                continue
            news_features = self._news_features(row)
            price_features = self._price_features(symbol, t0)
            corp_features = self._corp_action_features(symbol, t0, corp_actions.get(symbol, []))
            feature_json = {**news_features, **price_features, **corp_features}
            self._apply_interactions(feature_json)
            if news_features.get("sentiment") is None or news_features.get("title_len", 0) == 0:
                missing_news += 1
            if price_features.get("ret_20d_pre") is None or price_features.get("vol_20d_pre") is None:
                missing_price += 1
            if corp_features.get("days_to_next_record") is None and corp_features.get("days_since_prev_record") is None:
                missing_corp += 1
            for key, value in feature_json.items():
                if value is not None:
                    feature_non_null[key] = feature_non_null.get(key, 0) + 1
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
        self._write_report(len(rows), inserted, missing_news, missing_price, missing_corp, feature_non_null)

    def _write_report(
        self,
        attempted: int,
        produced: int,
        missing_news: int,
        missing_price: int,
        missing_corp: int,
        feature_non_null: dict[str, int],
    ) -> None:
        top_features = sorted(feature_non_null.items(), key=lambda kv: kv[1], reverse=True)[:20]
        lines = [
            "Agent 3 Report",
            f"events attempted: {attempted}",
            f"features produced: {produced}",
            f"news features missing: {missing_news}",
            f"price context missing: {missing_price}",
            f"corp milestone features missing: {missing_corp}",
            "",
            "Top 20 features by non-null occurrence:",
            *[f"- {k}: {v}" for k, v in top_features],
        ]
        path = make_report_path("agent3")
        write_report(path, "\n".join(lines))

    def _news_features(self, row: dict[str, Any]) -> dict[str, Any]:
        features: dict[str, Any] = {}
        event_type = row.get("event_type") or "other"
        for label in ["earnings", "governance", "mna", "legal", "macro", "rumor", "corp_action", "other"]:
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

    def _load_corp_actions(self, symbols: list[str], start: date, end: date) -> dict[str, list[dict[str, Any]]]:
        if not symbols:
            return {}
        window_start = start - timedelta(days=120)
        window_end = end + timedelta(days=120)
        rows = self.pg.fetch_all(
            """
            SELECT symbol, action_type, record_date, ex_date, effective_date
            FROM public.corp_actions
            WHERE symbol = ANY(%(symbols)s)
              AND (
                (record_date >= %(start)s AND record_date <= %(end)s)
                OR (ex_date >= %(start)s AND ex_date <= %(end)s)
                OR (effective_date >= %(start)s AND effective_date <= %(end)s)
              )
            """,
            {"symbols": symbols, "start": window_start, "end": window_end},
        )
        actions: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            actions.setdefault(row["symbol"], []).append(row)
        return actions

    def _corp_action_features(
        self, symbol: str, t0: date, actions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        features: dict[str, Any] = {}
        record_dates = [row.get("record_date") for row in actions if row.get("record_date")]
        ex_dates = [row.get("ex_date") for row in actions if row.get("ex_date")]
        record_dates = sorted(record_dates)
        ex_dates = sorted(ex_dates)
        prev_record = max([d for d in record_dates if d <= t0], default=None)
        next_record = min([d for d in record_dates if d >= t0], default=None)
        features["days_since_prev_record"] = (t0 - prev_record).days if prev_record else None
        features["days_to_next_record"] = (next_record - t0).days if next_record else None
        features["within_window_record_7"] = 1 if self._within_window(record_dates, t0, 7) else 0
        features["within_window_record_14"] = 1 if self._within_window(record_dates, t0, 14) else 0
        features["within_window_record_30"] = 1 if self._within_window(record_dates, t0, 30) else 0
        features["within_window_ex_7"] = 1 if self._within_window(ex_dates, t0, 7) else 0
        features["within_window_ex_14"] = 1 if self._within_window(ex_dates, t0, 14) else 0
        features["within_window_ex_30"] = 1 if self._within_window(ex_dates, t0, 30) else 0
        features["count_corp_actions_last_180d"] = self._count_recent_actions(actions, t0, 180)
        most_recent = self._most_recent_action(actions, t0)
        for action_type in CORP_ACTION_TYPES:
            features[f"last_action_{action_type}"] = 1 if most_recent == action_type else 0
        return features

    @staticmethod
    def _within_window(dates: list[date], t0: date, window: int) -> bool:
        if not dates:
            return False
        for dt in dates:
            if abs((t0 - dt).days) <= window:
                return True
        return False

    @staticmethod
    def _count_recent_actions(actions: list[dict[str, Any]], t0: date, window: int) -> int:
        count = 0
        for action in actions:
            action_date = action.get("record_date") or action.get("ex_date") or action.get("effective_date")
            if action_date and 0 <= (t0 - action_date).days <= window:
                count += 1
        return count

    @staticmethod
    def _most_recent_action(actions: list[dict[str, Any]], t0: date) -> str | None:
        latest = None
        latest_date = None
        for action in actions:
            action_date = action.get("record_date") or action.get("ex_date") or action.get("effective_date")
            if action_date and action_date <= t0:
                if latest_date is None or action_date > latest_date:
                    latest_date = action_date
                    latest = action.get("action_type")
        return latest

    @staticmethod
    def _apply_interactions(features: dict[str, Any]) -> None:
        sentiment = features.get("sentiment")
        record_14 = features.get("within_window_record_14")
        if sentiment is not None and record_14 is not None:
            features["sentiment_x_record_14"] = sentiment * record_14
        momentum_20 = features.get("ret_20d_pre")
        ex_14 = features.get("within_window_ex_14")
        if momentum_20 is not None and ex_14 is not None:
            features["pre_momentum_20d_x_ex_14"] = momentum_20 * ex_14

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
