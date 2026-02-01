from __future__ import annotations

import bisect
import hashlib
import logging
from datetime import date
from typing import Any

from src.storage.pg import PostgresStorage
from src.utils.features import max_drawdown, safe_std
from src.utils.reporting import make_report_path, write_report

logger = logging.getLogger(__name__)


class Agent2MarketReact:
    def __init__(self, pg: PostgresStorage, horizon_days: int = 40):
        self.pg = pg
        self.horizon_days = horizon_days

    def run(self) -> None:
        events = self.pg.fetch_all("SELECT * FROM news_events")
        if not events:
            logger.warning("No news_events available for market reactions.")
            self._write_report(0, [], {})
            return
        symbols = sorted({event.get("symbol") for event in events if event.get("symbol")})
        prices_map = self._load_prices(symbols)
        drop_reasons = {
            "missing_publish_date": 0,
            "missing_t0": 0,
            "no_price_rows": 0,
            "insufficient_closes": 0,
            "insufficient_horizon": 0,
        }
        reactions: list[dict[str, Any]] = []
        for event in events:
            publish_date = event.get("publish_date")
            if not publish_date:
                drop_reasons["missing_publish_date"] += 1
                continue
            symbol = event.get("symbol")
            if not symbol or symbol not in prices_map:
                drop_reasons["no_price_rows"] += 1
                continue
            dates, closes = prices_map[symbol]
            t0_index = self._resolve_t0_index(dates, publish_date.date())
            if t0_index is None:
                drop_reasons["missing_t0"] += 1
                continue
            closes_from_t0 = closes[t0_index:]
            if len(closes_from_t0) < 2:
                drop_reasons["insufficient_closes"] += 1
                continue
            ret_1d = self._calc_return(closes_from_t0, 1)
            ret_5d = self._calc_return(closes_from_t0, 5)
            ret_h = self._calc_return(closes_from_t0, self.horizon_days)
            if ret_h is None:
                drop_reasons["insufficient_horizon"] += 1
                continue
            vol_5d = self._calc_volatility(closes_from_t0[:6])
            dd_h = max_drawdown(closes_from_t0[: self.horizon_days + 1])
            label_up = 1 if ret_h is not None and ret_h > 0 else 0
            reaction_id = hashlib.md5(f"{event['event_id']}|{self.horizon_days}".encode("utf-8")).hexdigest()
            reactions.append(
                {
                    "reaction_id": reaction_id,
                    "event_id": event["event_id"],
                    "symbol": symbol,
                    "t0": dates[t0_index],
                    "horizon_days": self.horizon_days,
                    "ret_1d": ret_1d,
                    "ret_5d": ret_5d,
                    "ret_h": ret_h,
                    "vol_5d": vol_5d,
                    "dd_h": dd_h,
                    "label_up": label_up,
                    "meta_json": {"price_points": len(closes_from_t0)},
                }
            )
        inserted = self.pg.upsert_market_reactions(reactions)
        logger.info("Agent2 inserted %s market_reactions", inserted)
        self._write_report(len(events), reactions, drop_reasons)

    def _load_prices(self, symbols: list[str]) -> dict[str, tuple[list[date], list[float]]]:
        if not symbols:
            return {}
        rows = self.pg.fetch_all(
            """
            SELECT symbol, date, close
            FROM prices
            WHERE symbol = ANY(%(symbols)s)
            ORDER BY symbol, date ASC
            """,
            {"symbols": symbols},
        )
        prices: dict[str, tuple[list[date], list[float]]] = {}
        for row in rows:
            symbol = row.get("symbol")
            close = row.get("close")
            dt = row.get("date")
            if symbol is None or close is None or dt is None:
                continue
            if symbol not in prices:
                prices[symbol] = ([], [])
            prices[symbol][0].append(dt)
            prices[symbol][1].append(close)
        return prices

    @staticmethod
    def _resolve_t0_index(dates: list[date], publish_date: date) -> int | None:
        if not dates:
            return None
        idx = bisect.bisect_left(dates, publish_date)
        if idx >= len(dates):
            return None
        return idx

    def _write_report(self, total_events: int, reactions: list[dict[str, Any]], drop_reasons: dict[str, int]) -> None:
        label_up = sum(1 for r in reactions if r.get("label_up") == 1)
        label_down = sum(1 for r in reactions if r.get("label_up") == 0)
        ret_h_values = [r.get("ret_h") for r in reactions if r.get("ret_h") is not None]
        if ret_h_values:
            mean_ret = sum(ret_h_values) / len(ret_h_values)
            min_ret = min(ret_h_values)
            max_ret = max(ret_h_values)
            sorted_vals = sorted(ret_h_values)
            mid = len(sorted_vals) // 2
            median_ret = sorted_vals[mid] if len(sorted_vals) % 2 == 1 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
            std_ret = safe_std(ret_h_values)
        else:
            mean_ret = min_ret = max_ret = median_ret = std_ret = None
        lines = [
            "Agent 2 Report",
            f"news_events considered: {total_events}",
            f"market_reactions produced: {len(reactions)}",
            "",
            "Drop reasons:",
            *[f"- {k}: {v}" for k, v in drop_reasons.items()],
            "",
            f"label balance: up={label_up}, down={label_down}",
            f"ret_h summary: mean={mean_ret}, std={std_ret}, min={min_ret}, max={max_ret}, median={median_ret}",
        ]
        path = make_report_path("agent2")
        write_report(path, "\n".join(lines))

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
