from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd

from src.storage.pg import PostgresStorage
from src.utils.http import HttpClient

logger = logging.getLogger(__name__)


class FireAntPriceCollector:
    def __init__(self, pg: PostgresStorage, http: HttpClient, base_url: str):
        self.pg = pg
        self.http = http
        self.base_url = base_url.rstrip("/")

    def collect(self, symbols: list[str], start: date, end: date) -> None:
        for symbol in symbols:
            rows = self._fetch_symbol(symbol, start, end)
            if not rows:
                logger.warning("No price rows for %s", symbol)
                continue
            inserted = self.pg.upsert_prices(rows)
            logger.info("Prices %s %s rows", symbol, inserted)

    def _fetch_symbol(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        for candidate in (f"HO{symbol}", symbol):
            rows = self._request_symbol(candidate, symbol, start, end)
            if rows:
                return rows
        return []

    def _request_symbol(self, api_symbol: str, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        url = f"{self.base_url}/api/Data/Markets/HistoricalQuotes"
        params = {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
            "symbol": api_symbol,
        }
        try:
            response = self.http.get(url, params=params)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("FireAnt request failed for %s: %s", api_symbol, exc)
            return []
        content = response.content
        rows = self._parse_xml(content)
        for row in rows:
            row["symbol"] = symbol
        return rows

    def _parse_xml(self, content: bytes) -> list[dict[str, Any]]:
        try:
            df = pd.read_xml(io.BytesIO(content))
            if df is None or df.empty:
                return []
            df.columns = [c.lower() for c in df.columns]
            mapping = {
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
            rows = []
            for _, row in df.iterrows():
                rows.append({
                    "date": pd.to_datetime(row.get("date")).date() if row.get("date") is not None else None,
                    "open": self._safe_float(row.get("open")),
                    "high": self._safe_float(row.get("high")),
                    "low": self._safe_float(row.get("low")),
                    "close": self._safe_float(row.get("close")),
                    "volume": self._safe_float(row.get("volume")),
                })
            return [r for r in rows if r.get("date")]
        except Exception:
            return self._parse_xml_fallback(content)

    def _parse_xml_fallback(self, content: bytes) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            root = ET.fromstring(content)
        except Exception:
            return rows
        for item in root.iter():
            if item.tag.lower() in {"historicalquote", "item"}:
                row = {child.tag.lower(): child.text for child in item}
                if not row:
                    continue
                try:
                    dt = pd.to_datetime(row.get("date")).date() if row.get("date") else None
                except Exception:
                    dt = None
                rows.append(
                    {
                        "date": dt,
                        "open": self._safe_float(row.get("open")),
                        "high": self._safe_float(row.get("high")),
                        "low": self._safe_float(row.get("low")),
                        "close": self._safe_float(row.get("close")),
                        "volume": self._safe_float(row.get("volume")),
                    }
                )
        return [r for r in rows if r.get("date")]

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except Exception:
            return None
