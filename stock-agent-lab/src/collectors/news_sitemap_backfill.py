from __future__ import annotations

import gzip
import hashlib
import logging
import os
import time
from collections import deque
from datetime import date, datetime
from typing import Any, Iterable

from bs4 import BeautifulSoup
from dateutil.parser import parse

from src.collectors.vn30_universe import VN30_SYMBOLS, SYMBOL_ALIASES
from src.storage.pg import PostgresStorage
import re
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

from src.utils.html import (
    extract_canonical_url,
    extract_content,
    extract_publish_date,
    extract_title,
    parse_date_from_text,
    select_text,
    summarize,
)
from src.utils.http import HttpClient

logger = logging.getLogger(__name__)

BUSINESS_KEYWORDS = (
    "chung-khoan",
    "co-phieu",
    "co-tuc",
    "co-phan",
    "gdkkhq",
    "gdkhq",
    "chot-quyen",
    "quyen-mua",
    "phat-hanh",
    "niem-yet",
    "dai-hoi-co-dong",
)


SOURCE_CONFIG = {
    "cafef": {
        "base_domain": "https://cafef.vn",
        "entry_pages": ["https://cafef.vn/thi-truong-chung-khoan.chn", "https://cafef.vn/doanh-nghiep.chn"],
        "category_slugs": ["thi-truong-chung-khoan", "doanh-nghiep"],
        "robots": "https://cafef.vn/robots.txt",
        "title_selectors": ["h1.title", "h1#title", "h1"],
        "date_selectors": [".time", ".timeDate", "span.pdate", "time"],
        "content_selectors": ["div#mainContent", "div.detail-content", "div.contentdetail", "div#content_detail"],
        "listing_date_selectors": [".time", ".timeDate", "span.pdate", "time"],
        "pagination_next_selectors": ["a[rel='next']", "a.next", "a[title*='Trang tiếp']", ".pagination a.next"],
        "allowlist": [r"\.chn$", r"\.htm$", r"/\d{4}/\d{2}/\d{2}/"],
        "denylist": [r"/video", r"/photo", r"/tag", r"/topic", r"/rss", r"/amp", r"/mobile", r"/sitemap", r"/tim-kiem"],
        "search_templates": ["https://cafef.vn/tim-kiem.chn?keyword={query}"],
    },
    "vietstock": {
        "base_domain": "https://vietstock.vn",
        "entry_pages": ["https://vietstock.vn/chung-khoan.htm", "https://vietstock.vn/doanh-nghiep.htm"],
        "category_slugs": ["chung-khoan", "doanh-nghiep"],
        "robots": "https://vietstock.vn/robots.txt",
        "title_selectors": ["h1.title", "h1#title", "h1.detail-title", "h1.news-title", "h1"],
        "date_selectors": ["span.date", ".news-date", ".date", "time"],
        "content_selectors": ["div#news-content", "div#content_detail", "div.news-content", "div.detail-content"],
        "listing_date_selectors": ["span.date", ".news-date", ".date", "time"],
        "pagination_next_selectors": ["a[rel='next']", "a.next", ".pagination a.next", ".pagination a[rel='next']"],
        "allowlist": [r"\.htm$", r"/\d{4}/\d{2}/\d{2}/"],
        "denylist": [r"/video", r"/photo", r"/tag", r"/rss", r"/amp", r"/mobile", r"/sitemap", r"/tim-kiem"],
        "search_templates": ["https://vietstock.vn/tim-kiem?kw={query}"],
    },
    "vnexpress": {
        "base_domain": "https://vnexpress.net",
        "entry_pages": [
            "https://vnexpress.net/kinh-doanh",
            "https://vnexpress.net/kinh-doanh/doanh-nghiep",
            "https://vnexpress.net/kinh-doanh/chung-khoan",
        ],
        "category_slugs": ["kinh-doanh", "doanh-nghiep", "chung-khoan"],
        "robots": "https://vnexpress.net/robots.txt",
        "title_selectors": ["h1.title-detail", "h1.title_news_detail", "h1"],
        "date_selectors": ["span.date", "span.date-time", "p.date", "time"],
        "content_selectors": ["article.fck_detail", "article#fck_detail", "div.fck_detail", "div#article_content"],
        "listing_date_selectors": ["span.date", "span.date-time", "p.date", "time"],
        "pagination_next_selectors": ["a[rel='next']", "a.next", ".pagination a.next", ".pagination a[rel='next']"],
        "allowlist": [r"/\d{4}/\d{2}/\d{2}/", r"\.html$", r"\.htm$"],
        "denylist": [r"/video", r"/photo", r"/tag", r"/topic", r"/rss", r"/amp", r"/mobile", r"/sitemap", r"/tim-kiem"],
        "search_templates": ["https://vnexpress.net/tim-kiem?q={query}"],
    },
}


class NewsSitemapBackfill:
    def __init__(
        self,
        pg: PostgresStorage,
        http: HttpClient,
        sleep_secs: float,
        news_start: date,
        news_end: date,
        min_content_chars: int,
        max_urls_per_source: int | None = None,
        hard_filter: bool = False,
    ):
        self.pg = pg
        self.http = http
        self.sleep_secs = sleep_secs
        self.news_start = news_start
        self.news_end = news_end
        self.min_content_chars = min_content_chars
        self.max_urls_per_source = max_urls_per_source
        self.hard_filter = hard_filter
        self.user_agent = "Mozilla/5.0 (compatible; StockAgentLab/1.0; +https://example.com/bot)"
        self._relevance_pattern = self._build_relevance_pattern()

    def run_source(
        self,
        source: str,
        deepen_symbols: list[str] | None = None,
        deepen_limit: int = 0,
        discover_sitemaps: bool = True,
        category_crawl: bool = False,
        max_pages_per_entry: int = 2000,
    ) -> None:
        config = SOURCE_CONFIG[source]
        if category_crawl:
            logger.info("Category crawl discovery for %s", source)
            self._discover_from_categories(source, config, max_pages_per_entry)
        elif discover_sitemaps:
            logger.info("Phase 1 discovery for %s", source)
            self._discover_from_sitemaps(source, config["robots"], config)
        if deepen_symbols and deepen_limit > 0:
            self._discover_symbol_pages(source, config, deepen_symbols, deepen_limit)
        logger.info("Phase 2 fetch for %s", source)
        self._fetch_articles(source, config)

    def _discover_from_sitemaps(self, source: str, robots_url: str, config: dict[str, Any]) -> None:
        sitemap_urls = self._fetch_robots_sitemaps(robots_url)
        if not sitemap_urls:
            logger.warning("No sitemap URLs found for %s", source)
            return
        ordered = self._prioritize_sitemaps(sitemap_urls)
        last_done = self._get_state(source, "sitemap")
        queue = deque(ordered)
        visited: set[str] = set()
        skipping = bool(last_done and last_done in ordered)
        while queue:
            sitemap_url = queue.popleft()
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)
            if skipping:
                if sitemap_url == last_done:
                    skipping = False
                continue
            urls, sitemap_children = self._fetch_sitemap_entries(sitemap_url, config)
            inserted = self._upsert_news_urls(source, urls)
            logger.info("Discovered %s URLs from %s", inserted, sitemap_url)
            self._set_state(source, "sitemap", sitemap_url)
            for child in self._prioritize_sitemaps(sitemap_children):
                if child not in visited:
                    queue.append(child)

    def _fetch_robots_sitemaps(self, robots_url: str) -> list[str]:
        try:
            response = self.http.get(robots_url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            logger.warning("Robots fetch failed for %s: %s", robots_url, exc)
            return []
        time.sleep(self.sleep_secs)
        sitemaps = []
        for line in response.text.splitlines():
            if line.lower().startswith("sitemap:"):
                _, value = line.split(":", 1)
                url = value.strip()
                if url:
                    sitemaps.append(url)
        return list(dict.fromkeys(sitemaps))

    def _prioritize_sitemaps(self, sitemaps: Iterable[str]) -> list[str]:
        def score(url: str) -> int:
            lowered = url.lower()
            return sum(1 for key in BUSINESS_KEYWORDS if key in lowered)

        scored = sorted(set(sitemaps), key=lambda u: (-score(u), u))
        return scored

    def _fetch_sitemap_entries(self, sitemap_url: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        try:
            response = self.http.get(sitemap_url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            logger.warning("Sitemap fetch failed for %s: %s", sitemap_url, exc)
            return [], []
        time.sleep(self.sleep_secs)
        content = response.content
        if sitemap_url.endswith(".gz"):
            try:
                content = gzip.decompress(content)
            except Exception as exc:
                logger.warning("Failed to decompress %s: %s", sitemap_url, exc)
                return [], []
        soup = BeautifulSoup(content, "xml")
        sitemap_children = [loc.get_text(strip=True) for loc in soup.find_all("loc") if loc.find_parent("sitemap")]
        url_entries = []
        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            if not loc:
                continue
            url = loc.get_text(strip=True)
            if not self._is_allowed_url(url, config):
                continue
            lastmod_tag = url_tag.find("lastmod")
            lastmod_date = self._parse_lastmod(lastmod_tag.get_text(strip=True) if lastmod_tag else "")
            if lastmod_date and (lastmod_date < self.news_start or lastmod_date > self.news_end):
                continue
            url_entries.append({"url": url, "lastmod": lastmod_date})
        return url_entries, sitemap_children

    def _parse_lastmod(self, value: str) -> date | None:
        if not value:
            return None
        try:
            parsed = parse(value, dayfirst=True, fuzzy=True)
            return parsed.date()
        except Exception:
            return None

    def _upsert_news_urls(self, source: str, rows: Iterable[dict[str, Any]]) -> int:
        rows_list = list(rows)
        if not rows_list:
            return 0
        query = """
            INSERT INTO news_urls (url, source, lastmod)
            VALUES (%(url)s, %(source)s, %(lastmod)s)
            ON CONFLICT (url) DO UPDATE SET
                source = EXCLUDED.source,
                lastmod = COALESCE(EXCLUDED.lastmod, news_urls.lastmod)
        """
        payload = [dict(row, source=source) for row in rows_list]
        with self.pg.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, payload)
        return len(payload)

    def _fetch_articles(self, source: str, config: dict[str, Any]) -> None:
        batch_size = 50
        total = 0
        while True:
            rows = self._next_batch(source, batch_size)
            if not rows:
                break
            for row in rows:
                total += 1
                self._fetch_one(source, config, row["url"])
            logger.info("Fetched %s URLs for %s so far", total, source)

    def _next_batch(self, source: str, limit: int) -> list[dict[str, Any]]:
        if self.max_urls_per_source is not None and self.max_urls_per_source <= 0:
            return []
        actual_limit = limit
        if self.max_urls_per_source is not None:
            actual_limit = min(limit, self.max_urls_per_source)
        limit_clause = "LIMIT %(limit)s"
        priority_case = self._url_priority_case()
        query = f"""
            SELECT url, lastmod
            FROM news_urls
            WHERE source = %(source)s
              AND status = 'new'
              AND (lastmod IS NULL OR (lastmod >= %(start)s AND lastmod <= %(end)s))
            ORDER BY {priority_case}, lastmod NULLS LAST, url
            {limit_clause}
        """
        rows = self.pg.fetch_all(
            query,
            {
                "source": source,
                "start": self.news_start,
                "end": self.news_end,
                "limit": actual_limit,
            },
        )
        if self.max_urls_per_source is not None:
            self.max_urls_per_source = max(self.max_urls_per_source - len(rows), 0)
        return rows

    def _url_priority_case(self) -> str:
        clauses = []
        for keyword in BUSINESS_KEYWORDS:
            clauses.append(f"url ILIKE '%%{keyword}%%'")
        if not clauses:
            return "0"
        return f"CASE WHEN {' OR '.join(clauses)} THEN 0 ELSE 1 END"

    def _discover_symbol_pages(
        self, source: str, config: dict[str, Any], symbols: list[str], per_symbol_limit: int
    ) -> None:
        templates = config.get("search_templates", [])
        if not templates:
            return
        for symbol in symbols:
            discovered = 0
            for template in templates:
                if discovered >= per_symbol_limit:
                    break
                search_url = template.format(query=symbol)
                links = self._fetch_search_links(search_url, config["base_domain"], config)
                if not links:
                    continue
                rows = [{"url": url, "lastmod": None} for url in links]
                inserted = self._upsert_news_urls(source, rows)
                logger.info("Deepen %s %s: %s urls", source, symbol, inserted)
                discovered += inserted

    def _fetch_search_links(self, url: str, domain: str, config: dict[str, Any]) -> list[str]:
        try:
            response = self.http.get(url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            logger.warning("Search fetch failed %s: %s", url, exc)
            return []
        time.sleep(self.sleep_secs)
        soup = BeautifulSoup(response.text, "lxml")
        links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                href = f"{domain}{href}"
            if not href.startswith(domain):
                continue
            if any(ext in href for ext in [".jpg", ".png", ".gif"]):
                continue
            if not (href.endswith(".html") or href.endswith(".htm") or href.endswith(".chn")):
                continue
            if not self._is_allowed_url(href, config):
                continue
            links.append(href.split("#")[0])
        return list(dict.fromkeys(links))

    def _fetch_one(self, source: str, config: dict[str, Any], url: str) -> None:
        if not self._is_allowed_url(url, config):
            self._mark_url(url, "skipped", None, "denylist")
            return
        try:
            response = self.http.get(url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            logger.warning("Fetch failed %s: %s", url, exc)
            self._mark_url(url, "error", None, str(exc))
            return
        time.sleep(self.sleep_secs)
        status = response.status_code
        if status != 200:
            self._mark_url(url, "error", status, f"HTTP {status}")
            return
        soup = BeautifulSoup(response.text, "lxml")
        canonical = extract_canonical_url(soup)
        if canonical:
            url = canonical
        title = self._extract_title(soup, config)
        publish_date = self._extract_publish_date(soup, config)
        if publish_date:
            logger.info("Fetched article date %s | %s", publish_date.date(), url)
        content = self._extract_content(soup, config)
        if self.hard_filter and not self._is_relevant_article(url, f"{title} {content}"):
            self._mark_url(url, "skipped", status, "hard_filter")
            return
        if publish_date and (publish_date.date() < self.news_start or publish_date.date() > self.news_end):
            self._mark_url(url, "skipped", status, "publish_date_out_of_range")
            return
        if len(content) < self.min_content_chars:
            self._mark_url(url, "skipped", status, "content_too_short")
            return
        summary = summarize(content, max_sentences=3)
        news_id = self._news_id(source, url)
        row = {
            "id": news_id,
            "source": source,
            "source_country": "vn",
            "language": "vi",
            "category": "business",
            "publish_date": publish_date,
            "title": title,
            "summary": summary,
            "text": content,
            "url": url,
            "image": None,
            "sentiment": None,
        }
        self.pg.upsert_news_raw([row])
        self._mark_url(url, "fetched", status, None)

    def _extract_title(self, soup: BeautifulSoup, config: dict[str, Any]) -> str:
        title = select_text(soup, config.get("title_selectors", []))
        return title or extract_title(soup)

    def _extract_publish_date(self, soup: BeautifulSoup, config: dict[str, Any]) -> datetime | None:
        for selector in config.get("date_selectors", []):
            node = soup.select_one(selector)
            if node:
                text = node.get("datetime") or node.get_text(" ", strip=True)
                parsed = self._parse_date(text)
                if parsed:
                    return parsed
        return extract_publish_date(soup)

    def _extract_content(self, soup: BeautifulSoup, config: dict[str, Any]) -> str:
        selectors = config.get("content_selectors", [])
        return extract_content(soup, selectors)

    def _parse_date(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return parse(value, dayfirst=True, fuzzy=True)
        except Exception:
            return parse_date_from_text(value)

    def _is_allowed_url(self, url: str, config: dict[str, Any]) -> bool:
        allowlist = [re.compile(pat, re.IGNORECASE) for pat in config.get("allowlist", [])]
        denylist = [re.compile(pat, re.IGNORECASE) for pat in config.get("denylist", [])]
        if any(pat.search(url) for pat in denylist):
            return False
        if allowlist and not any(pat.search(url) for pat in allowlist):
            return False
        if self.hard_filter:
            slugs = config.get("category_slugs", [])
            if slugs and not any(slug in url for slug in slugs):
                return False
        return True

    def _build_relevance_pattern(self) -> re.Pattern[str]:
        terms = set()
        for symbol in VN30_SYMBOLS:
            terms.add(symbol)
        for symbol, aliases in SYMBOL_ALIASES.items():
            terms.add(symbol)
            for alias in aliases:
                if alias:
                    terms.add(alias)
        corp_terms = [
            "dividend",
            "cash dividend",
            "stock dividend",
            "rights issue",
            "bonus share",
            "esop",
            "record date",
            "ex-right",
            "additional listing",
            "split",
            "consolidation",
        ]
        terms.update(corp_terms)
        escaped = [re.escape(t) for t in terms if t]
        pattern = r"(?i)\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern)

    def _is_relevant_text(self, text: str) -> bool:
        if not text:
            return False
        return bool(self._relevance_pattern.search(text))

    def _is_relevant_article(self, url: str, text: str) -> bool:
        url_hit = any(keyword in url for keyword in BUSINESS_KEYWORDS)
        if url_hit:
            return True
        return self._is_relevant_text(text)

    def _discover_from_categories(self, source: str, config: dict[str, Any], max_pages_per_entry: int) -> None:
        entry_pages = config.get("entry_pages", [])
        if not entry_pages:
            logger.warning("No entry pages configured for %s", source)
            return
        seen_urls: set[str] = set()
        for entry_url in entry_pages:
            self._crawl_listing(entry_url, source, config, max_pages_per_entry, seen_urls)

    def _crawl_listing(
        self,
        start_url: str,
        source: str,
        config: dict[str, Any],
        max_pages: int,
        seen_urls: set[str],
    ) -> None:
        current_url = start_url
        base_domain = config.get("base_domain", "")
        allowed_listing_prefix = start_url
        for page_idx in range(max_pages):
            listing_html = self._fetch_listing_page(current_url)
            if not listing_html:
                break
            soup = BeautifulSoup(listing_html, "lxml")
            urls = self._extract_listing_urls(soup, base_domain, config)
            new_urls = [u for u in urls if u not in seen_urls]
            for u in new_urls:
                seen_urls.add(u)
            if new_urls:
                self._upsert_news_urls(source, [{"url": u, "lastmod": None} for u in new_urls])
            else:
                break
            stop, oldest = self._should_stop_by_date(soup, config)
            if oldest:
                logger.info("Listing page %s oldest date %s | %s", page_idx + 1, oldest.date(), current_url)
            else:
                logger.info("Listing page %s has no detectable dates | %s", page_idx + 1, current_url)
            if stop:
                logger.info(
                    "Reached date bound for %s at %s (start=%s)",
                    current_url,
                    oldest.date() if oldest else None,
                    self.news_start,
                )
                break
            next_url = self._find_next_page_url(current_url, soup, config)
            if not next_url:
                break
            if not self._is_allowed_listing_url(next_url, base_domain, allowed_listing_prefix):
                logger.warning("Stopping listing crawl due to off-category next page: %s", next_url)
                break
            current_url = next_url

    def _fetch_listing_page(self, url: str) -> str | None:
        try:
            response = self.http.get(url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            logger.warning("Listing fetch failed %s: %s", url, exc)
            return None
        time.sleep(self.sleep_secs)
        if response.status_code != 200:
            logger.warning("Listing HTTP %s for %s", response.status_code, url)
            return None
        return response.text

    def _extract_listing_urls(self, soup: BeautifulSoup, base_domain: str, config: dict[str, Any]) -> list[str]:
        allowlist = [re.compile(pat, re.IGNORECASE) for pat in config.get("allowlist", [])]
        denylist = [re.compile(pat, re.IGNORECASE) for pat in config.get("denylist", [])]
        urls: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(base_domain, href)
            if base_domain and not href.startswith(base_domain):
                continue
            href = href.split("#")[0]
            if any(pat.search(href) for pat in denylist):
                continue
            if allowlist and not any(pat.search(href) for pat in allowlist):
                continue
            urls.append(href)
        return list(dict.fromkeys(urls))

    def _find_next_page_url(self, current_url: str, soup: BeautifulSoup, config: dict[str, Any]) -> str | None:
        for selector in config.get("pagination_next_selectors", []):
            node = soup.select_one(selector)
            if node and node.get("href"):
                return urljoin(current_url, node.get("href"))
        # numeric pattern fallback
        parsed = urlparse(current_url)
        qs = parse_qs(parsed.query)
        for key in ["page", "p"]:
            if key in qs and qs[key]:
                try:
                    current = int(qs[key][0])
                    qs[key] = [str(current + 1)]
                    query = urlencode(qs, doseq=True)
                    return urlunparse(parsed._replace(query=query))
                except Exception:
                    continue
        path = parsed.path
        match = re.search(r"(?:/page/|/p|/trang-)(\d+)", path, re.IGNORECASE)
        if match:
            current = int(match.group(1))
            next_path = re.sub(match.group(0), match.group(0).replace(match.group(1), str(current + 1)), path)
            return urlunparse(parsed._replace(path=next_path))
        return None

    def _is_allowed_listing_url(self, url: str, base_domain: str, prefix: str) -> bool:
        if base_domain and not url.startswith(base_domain):
            return False
        # keep pagination strictly within the same entry page path
        prefix_path = prefix.split("?")[0]
        if prefix_path and not url.startswith(prefix_path):
            return False
        return True

    def _should_stop_by_date(self, soup: BeautifulSoup, config: dict[str, Any]) -> tuple[bool, datetime | None]:
        date_selectors = config.get("listing_date_selectors", [])
        dates: list[datetime] = []
        for selector in date_selectors:
            for node in soup.select(selector):
                text = node.get("datetime") or node.get_text(" ", strip=True)
                parsed = self._parse_date(text)
                if parsed:
                    dates.append(parsed)
        if dates:
            oldest = min(dates)
            if oldest.date() <= self.news_start:
                return True, oldest
            return False, oldest
        # lightweight probe from links
        links = self._extract_listing_urls(soup, config.get("base_domain", ""), config)[:3]
        probe_dates = []
        for link in links:
            parsed = self._probe_article_date(link, config)
            if parsed:
                probe_dates.append(parsed)
        if probe_dates:
            oldest = min(probe_dates)
            if oldest.date() <= self.news_start:
                return True, oldest
            return False, oldest
        return False, None

    def _probe_article_date(self, url: str, config: dict[str, Any]) -> datetime | None:
        try:
            response = self.http.get(url, headers={"User-Agent": self.user_agent})
        except Exception:
            return None
        time.sleep(self.sleep_secs)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "lxml")
        return self._extract_publish_date(soup, config)

    def _news_id(self, source: str, url: str) -> int:
        raw = f"{source}|{url}".encode("utf-8")
        digest = hashlib.sha1(raw).digest()
        return int.from_bytes(digest[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF

    def _mark_url(self, url: str, status: str, http_status: int | None, error_text: str | None) -> None:
        query = """
            UPDATE news_urls
            SET status = %(status)s,
                http_status = %(http_status)s,
                error_text = %(error_text)s,
                fetched_at = now()
            WHERE url = %(url)s
        """
        self.pg.execute(
            query,
            {"status": status, "http_status": http_status, "error_text": error_text, "url": url},
        )

    def _get_state(self, source: str, cursor: str) -> str | None:
        query = """
            SELECT cursor_value
            FROM collector_state
            WHERE source = %(source)s AND cursor = %(cursor)s
        """
        rows = self.pg.fetch_all(query, {"source": source, "cursor": cursor})
        return rows[0]["cursor_value"] if rows else None

    def _set_state(self, source: str, cursor: str, cursor_value: str) -> None:
        query = """
            INSERT INTO collector_state (source, cursor, cursor_value)
            VALUES (%(source)s, %(cursor)s, %(cursor_value)s)
            ON CONFLICT (source, cursor) DO UPDATE SET
                cursor_value = EXCLUDED.cursor_value,
                updated_at = now()
        """
        self.pg.execute(
            query,
            {"source": source, "cursor": cursor, "cursor_value": cursor_value},
        )
