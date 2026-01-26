from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import Any

from src.storage.pg import PostgresStorage
from src.utils.features import max_drawdown, safe_std

logger = logging.getLogger(__name__)


class Agent2MarketReact:
    def __init__(self, pg: PostgresStorage, horizon_days: int = 40):
        self.pg = pg
        self.horizon_days = horizon_days

    def run(self) -> None:
        events = self.pg.fetch_all("SELECT * FROM news_events")
        if not events:
            logger.warning("No news_events available for market reactions.")
            return
        reactions: list[dict[str, Any]] = []
        for event in events:
            publish_date = event.get("publish_date")
            if not publish_date:
                continue
            symbol = event.get("symbol")
            t0 = self._resolve_t0(symbol, publish_date.date())
            if not t0:
                continue
            price_rows = self.pg.fetch_all(
                "SELECT date, open, close, volume FROM prices WHERE symbol = %(symbol)s AND date >= %(t0)s ORDER BY date ASC",
                {"symbol": symbol, "t0": t0},
            )
            if not price_rows:
                continue
            closes = [row.get("close") for row in price_rows if row.get("close") is not None]
            if len(closes) < 2:
                continue
            ret_1d = self._calc_return(closes, 1)
            ret_5d = self._calc_return(closes, 5)
            ret_h = self._calc_return(closes, self.horizon_days)
            vol_5d = self._calc_volatility(closes[:6])
            dd_h = max_drawdown(closes[: self.horizon_days + 1])
            label_up = 1 if ret_h is not None and ret_h > 0 else 0
            reaction_id = hashlib.md5(f"{event['event_id']}|{self.horizon_days}".encode("utf-8")).hexdigest()
            reactions.append(
                {
                    "reaction_id": reaction_id,
                    "event_id": event["event_id"],
                    "symbol": symbol,
                    "t0": t0,
                    "horizon_days": self.horizon_days,
                    "ret_1d": ret_1d,
                    "ret_5d": ret_5d,
                    "ret_h": ret_h,
                    "vol_5d": vol_5d,
                    "dd_h": dd_h,
                    "label_up": label_up,
                    "meta_json": {"price_points": len(closes)},
                }
            )
        inserted = self.pg.upsert_market_reactions(reactions)
        logger.info("Agent2 inserted %s market_reactions", inserted)

    def _resolve_t0(self, symbol: str, publish_date: date) -> date | None:
        rows = self.pg.fetch_all(
            "SELECT date FROM prices WHERE symbol = %(symbol)s AND date >= %(date)s ORDER BY date ASC LIMIT 1",
            {"symbol": symbol, "date": publish_date},
        )
        return rows[0]["date"] if rows else None

    @staticmethod
    def _calc_return(closes: list[float], horizon: int) -> float | None:
        if len(closes) <= horizon:
            return None
        if closes[0] in (None, 0):
            return None
        return closes[horizon] / closes[0] - 1

    @staticmethod
    def _calc_volatility(closes: list[float]) -> float | None:
        if len(closes) < 2:
            return None
        returns = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            curr = closes[i]
            if prev:
                returns.append(curr / prev - 1)
        return safe_std(returns)
