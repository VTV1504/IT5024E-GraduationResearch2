from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from datetime import date
from typing import Any

from bs4 import BeautifulSoup
from dateutil.parser import parse

from src.collectors.vn30_universe import SYMBOL_ALIASES, VN30_SYMBOLS
from src.storage.pg import PostgresStorage
from src.utils.html import clean_text
from src.utils.http import HttpClient

logger = logging.getLogger(__name__)

ACTION_KEYWORDS = {
    "dividend_cash": ["co tuc", "tam ung co tuc", "bang tien", "tien mat", "chi tra co tuc"],
    "dividend_stock": ["co tuc bang co phieu", "co phieu thuong", "thuong co phieu"],
    "rights_issue": ["quyen mua", "phat hanh them", "chao ban"],
    "bonus_issue": ["thuong co phieu", "co phieu thuong", "phat hanh thuong"],
    "esop": ["esop", "co phieu esop"],
    "split": ["chia tach"],
    "consolidation": ["gop co phieu"],
    "additional_listing": ["niem yet bo sung", "giao dich bo sung"],
    "record_date": ["chot quyen", "ngay dang ky cuoi cung", "record date"],
    "ex_rights": ["gdkhq", "ngay gdkhq", "ex-rights", "ex-date"],
}


class CorpActionsCollector:
    def __init__(self, pg: PostgresStorage, http: HttpClient, sleep_secs: float):
        self.pg = pg
        self.http = http
        self.sleep_secs = sleep_secs
        self.base_url = "https://www.vsd.vn"
        self.listing_url = "https://www.vsd.vn/vi/tin-tuc"
        self.user_agent = "Mozilla/5.0 (compatible; StockAgentLab/1.0; +https://example.com/bot)"
        self._cached_articles: list[dict[str, Any]] | None = None

    def collect(self, symbols: list[str], start: date, end: date) -> None:
        if not symbols:
            return
        articles = self._fetch_vsd_articles(start, end)
        if not articles:
            logger.warning("No corporate action articles found.")
            return
        rows = self._rows_from_articles(articles, start, end, symbols)
        if rows:
            self._upsert(rows)
        else:
            logger.warning("No corporate actions matched any VN30 symbols.")

    def _fetch_vsd_articles(self, start: date, end: date) -> list[dict[str, Any]]:
        if self._cached_articles is not None:
            return self._cached_articles
        articles: list[dict[str, Any]] = []
        seen: set[str] = set()
        current_url = self.listing_url
        for _ in range(200):
            listing_html = self._fetch_page(current_url)
            if not listing_html:
                break
            soup = BeautifulSoup(listing_html, "lxml")
            links = self._extract_listing_links(soup)
            new_links = [link for link in links if link not in seen]
            if not new_links:
                break
            for link in new_links:
                seen.add(link)
                article = self._fetch_article(link)
                if not article:
                    continue
                publish_date = article.get("publish_date")
                if publish_date and (publish_date < start or publish_date > end):
                    continue
                articles.append(article)
            next_url = self._find_next_listing(soup, current_url)
            if not next_url:
                break
            current_url = next_url
        self._cached_articles = articles
        return articles

    def _fetch_page(self, url: str) -> str | None:
        try:
            response = self.http.get(url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            logger.warning("Corp actions listing fetch failed %s: %s", url, exc)
            return None
        time.sleep(self.sleep_secs)
        if response.status_code != 200:
            logger.warning("Corp actions listing HTTP %s for %s", response.status_code, url)
            return None
        return response.text

    def _extract_listing_links(self, soup: BeautifulSoup) -> list[str]:
        links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            if href.startswith("/"):
                href = f"{self.base_url}{href}"
            if self.base_url not in href:
                continue
            if "/tin-tuc" not in href:
                continue
            if href.rstrip("/").endswith("/vi/tin-tuc"):
                continue
            if "page=" in href:
                continue
            links.append(href.split("#")[0])
        return list(dict.fromkeys(links))

    def _find_next_listing(self, soup: BeautifulSoup, current_url: str) -> str | None:
        next_link = soup.select_one("a[rel='next']") or soup.select_one("a.next") or soup.select_one(".pagination a.next")
        if next_link and next_link.get("href"):
            href = next_link.get("href")
            if href.startswith("/"):
                return f"{self.base_url}{href}"
            return href
        match = re.search(r"[?&]page=(\d+)", current_url)
        if match:
            page = int(match.group(1)) + 1
            return re.sub(r"page=\d+", f"page={page}", current_url)
        separator = "&" if "?" in current_url else "?"
        return f"{current_url}{separator}page=2" if "page=" not in current_url else None

    def _fetch_article(self, url: str) -> dict[str, Any] | None:
        try:
            response = self.http.get(url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            logger.warning("Corp actions article fetch failed %s: %s", url, exc)
            return None
        time.sleep(self.sleep_secs)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "lxml")
        title_node = soup.find("h1")
        title = clean_text(title_node.get_text(" ", strip=True)) if title_node else ""
        publish_date = self._extract_publish_date(soup)
        content = self._extract_content_text(soup)
        return {"url": url, "title": title, "publish_date": publish_date, "content": content}

    def _extract_publish_date(self, soup: BeautifulSoup) -> date | None:
        meta = soup.find("meta", attrs={"property": "article:published_time"})
        if meta and meta.get("content"):
            parsed = self._parse_date(meta.get("content"))
            if parsed:
                return parsed
        time_node = soup.find("time")
        if time_node:
            parsed = self._parse_date(time_node.get_text(" ", strip=True))
            if parsed:
                return parsed
        text = soup.get_text(" ", strip=True)
        return self._parse_date(text)

    def _extract_content_text(self, soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        article = soup.find("article")
        if article:
            return clean_text(article.get_text(" ", strip=True))
        content = soup.find("div", class_=re.compile(r"content|detail|body", re.IGNORECASE))
        if content:
            return clean_text(content.get_text(" ", strip=True))
        return clean_text(soup.get_text(" ", strip=True))

    def _rows_from_articles(
        self, articles: list[dict[str, Any]], start: date, end: date, symbols: list[str]
    ) -> list[dict[str, Any]]:
        symbol_patterns: dict[str, re.Pattern[str]] = {}
        for symbol in symbols or VN30_SYMBOLS:
            aliases = SYMBOL_ALIASES.get(symbol, [])
            tokens = [symbol] + aliases
            token_pattern = r"\b(" + "|".join([re.escape(t) for t in tokens if t]) + r")\b"
            symbol_patterns[symbol] = re.compile(token_pattern, re.IGNORECASE)
        rows = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')}"
            normalized = self._normalize_text(text)
            matched_symbols = [s for s, pat in symbol_patterns.items() if pat.search(normalized)]
            if not matched_symbols:
                continue
            record_date, effective_date, ex_date_hint = self._extract_dates_from_text(normalized)
            ex_date = ex_date_hint
            action_type = self._classify_action(normalized)
            action_date = record_date or ex_date or effective_date or article.get("publish_date")
            if not action_date:
                continue
            if action_date < start or action_date > end:
                continue
            cash_amount = self._parse_cash_amount(normalized)
            ratio = self._parse_ratio(normalized)
            for symbol in matched_symbols:
                action_id = self._action_id("vsd", symbol, action_type, record_date, effective_date, article.get("url"))
                rows.append(
                    {
                        "action_id": action_id,
                        "symbol": symbol,
                        "action_type": action_type,
                        "record_date": record_date,
                        "ex_date": ex_date,
                        "effective_date": effective_date,
                        "ratio": ratio,
                        "cash_amount": cash_amount,
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "source": "vsd",
                        "raw_json": {
                            "publish_date": article.get("publish_date").isoformat() if article.get("publish_date") else None,
                        },
                    }
                )
        return rows

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        normalized = unicodedata.normalize("NFKD", lowered)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    @staticmethod
    def _parse_date(value: str) -> date | None:
        if not value:
            return None
        try:
            return parse(value, dayfirst=True, fuzzy=True).date()
        except Exception:
            return None

    def _extract_dates_from_text(self, text: str) -> tuple[date | None, date | None, date | None]:
        dates = self._find_dates(text)
        record_date = None
        effective_date = None
        ex_date = None
        if "chot quyen" in text or "dang ky cuoi cung" in text:
            record_date = dates[0] if dates else None
        if "gdkhq" in text:
            ex_date = dates[0] if dates else None
        if "thanh toan" in text or "thuc hien" in text or "niem yet bo sung" in text:
            effective_date = dates[-1] if dates else None
        return record_date, effective_date, ex_date

    @staticmethod
    def _find_dates(text: str) -> list[date]:
        matches = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
        dates = []
        for match in matches:
            try:
                dates.append(parse(match, dayfirst=True).date())
            except Exception:
                continue
        return dates

    def _classify_action(self, text: str) -> str:
        for action_type, keywords in ACTION_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return action_type
        if "co tuc" in text:
            return "dividend_cash"
        return "other"

    @staticmethod
    def _parse_cash_amount(text: str) -> float | None:
        if "tien" not in text and "co tuc" not in text:
            return None
        match = re.search(r"(\d+[.,]?\d*)\s*(d|vnd|dong|vnd)", text)
        if match:
            value = match.group(1).replace(",", "")
            try:
                return float(value)
            except Exception:
                return None
        match = re.search(r"(\d+[.,]?\d*)\s*%", text)
        if match:
            value = match.group(1).replace(",", "")
            try:
                return float(value)
            except Exception:
                return None
        return None

    @staticmethod
    def _parse_ratio(text: str) -> str | None:
        match = re.search(r"\b(\d+\s*[:/]\s*\d+)\b", text)
        if match:
            return match.group(1).replace(" ", "")
        match = re.search(r"\b(\d+[.,]?\d*)\s*%\b", text)
        if match:
            return match.group(0)
        return None

    @staticmethod
    def _action_id(
        source: str,
        symbol: str,
        action_type: str,
        record_date: date | None,
        effective_date: date | None,
        url: str | None,
    ) -> str:
        raw = f"{source}|{symbol}|{action_type}|{record_date}|{effective_date}|{url or ''}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _upsert(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        query = """
            INSERT INTO public.corp_actions (
                action_id, symbol, action_type, record_date, ex_date, effective_date,
                ratio, cash_amount, title, url, source, raw_json
            )
            VALUES (
                %(action_id)s, %(symbol)s, %(action_type)s, %(record_date)s, %(ex_date)s, %(effective_date)s,
                %(ratio)s, %(cash_amount)s, %(title)s, %(url)s, %(source)s, %(raw_json)s
            )
            ON CONFLICT (action_id) DO UPDATE SET
                record_date = EXCLUDED.record_date,
                ex_date = EXCLUDED.ex_date,
                effective_date = EXCLUDED.effective_date,
                ratio = EXCLUDED.ratio,
                cash_amount = EXCLUDED.cash_amount,
                title = EXCLUDED.title,
                url = EXCLUDED.url,
                raw_json = EXCLUDED.raw_json
        """
        rows = [
            dict(row, raw_json=json.dumps(row.get("raw_json", {})))
            if isinstance(row.get("raw_json"), dict)
            else row
            for row in rows
        ]
        with self.pg.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows)
