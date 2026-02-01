from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from bs4 import BeautifulSoup
from dateutil.parser import parse


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def select_text(soup: BeautifulSoup, selectors: Iterable[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                return text
    return ""


def select_attr(soup: BeautifulSoup, selector: str, attr: str) -> str:
    node = soup.select_one(selector)
    if not node:
        return ""
    value = node.get(attr) or ""
    return clean_text(value)


def extract_canonical_url(soup: BeautifulSoup) -> str:
    canonical = select_attr(soup, "link[rel='canonical']", "href")
    if canonical:
        return canonical
    og_url = select_attr(soup, "meta[property='og:url']", "content")
    return og_url


def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return parse(value, dayfirst=True, fuzzy=True)
    except Exception:
        return None


def extract_title(soup: BeautifulSoup) -> str:
    og = select_attr(soup, "meta[property='og:title']", "content")
    if og:
        return og
    title = select_text(soup, ["title"])
    if title:
        return title
    return select_text(soup, ["h1"])


def extract_publish_date(soup: BeautifulSoup) -> datetime | None:
    meta_candidates = [
        "meta[property='article:published_time']",
        "meta[property='og:published_time']",
        "meta[itemprop='datePublished']",
        "meta[name='pubdate']",
        "meta[name='publish-date']",
        "meta[name='publishdate']",
        "meta[name='date']",
        "meta[name='dcterms.date']",
    ]
    for selector in meta_candidates:
        value = select_attr(soup, selector, "content")
        parsed = parse_date(value)
        if parsed:
            return parsed
    time_node = soup.find("time")
    if time_node:
        datetime_attr = clean_text(time_node.get("datetime") or "")
        parsed = parse_date(datetime_attr) or parse_date(time_node.get_text(" ", strip=True))
        if parsed:
            return parsed
    text_date = parse_date_from_text(soup.get_text(" ", strip=True))
    if text_date:
        return text_date
    return None


def parse_date_from_text(text: str) -> datetime | None:
    if not text:
        return None
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
        r"\b\d{1,2}:\d{2}\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            parsed = parse_date(match.group(0))
            if parsed:
                return parsed
    return None


def _text_from_container(container) -> str:
    if not container:
        return ""
    paragraphs = [clean_text(p.get_text(" ", strip=True)) for p in container.find_all("p")]
    paragraphs = [p for p in paragraphs if p]
    if paragraphs:
        return clean_text(" ".join(paragraphs))
    return clean_text(container.get_text(" ", strip=True))


def extract_longest_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside"]):
        tag.decompose()
    candidates = []
    for container in soup.find_all(["article", "section", "div"]):
        text = _text_from_container(container)
        if len(text) > 200:
            candidates.append(text)
    if candidates:
        return max(candidates, key=len)
    body = soup.body or soup
    return clean_text(body.get_text(" ", strip=True))


def extract_content(soup: BeautifulSoup, selectors: Iterable[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = _text_from_container(node)
            if text:
                return text
    return extract_longest_text(soup)


def summarize(text: str, max_sentences: int = 3) -> str:
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return clean_text(" ".join(sentences[:max_sentences]))
