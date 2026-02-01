from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Iterable

from src.collectors.vn30_universe import SYMBOL_ALIASES
from src.storage.pg import PostgresStorage

logger = logging.getLogger(__name__)


def month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def iter_months(start: date, end: date) -> list[str]:
    months = []
    current = date(start.year, start.month, 1)
    while current <= end:
        months.append(month_key(current))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def upsert_coverage(pg: PostgresStorage, rows: Iterable[dict[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    query = """
        INSERT INTO coverage_monthly (key, domain, symbol, year_month, count)
        VALUES (%(key)s, %(domain)s, %(symbol)s, %(year_month)s, %(count)s)
        ON CONFLICT (key) DO UPDATE SET
            count = EXCLUDED.count,
            updated_at = now()
    """
    with pg.connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, rows_list)


def compute_price_monthly(pg: PostgresStorage, start: date, end: date) -> dict[tuple[str, str], int]:
    rows = pg.fetch_all(
        """
        SELECT symbol, to_char(date_trunc('month', date), 'YYYY-MM') AS ym, COUNT(*) AS cnt
        FROM prices
        WHERE date >= %(start)s AND date <= %(end)s
        GROUP BY symbol, ym
        """,
        {"start": start, "end": end},
    )
    return {(row["symbol"], row["ym"]): row["cnt"] for row in rows}


def compute_corp_monthly(pg: PostgresStorage, start: date, end: date) -> dict[tuple[str, str], int]:
    rows = pg.fetch_all(
        """
        SELECT symbol,
               to_char(date_trunc('month', COALESCE(record_date, ex_date, effective_date)), 'YYYY-MM') AS ym,
               COUNT(*) AS cnt
        FROM public.corp_actions
        WHERE COALESCE(record_date, ex_date, effective_date) >= %(start)s
          AND COALESCE(record_date, ex_date, effective_date) <= %(end)s
        GROUP BY symbol, ym
        """,
        {"start": start, "end": end},
    )
    return {(row["symbol"], row["ym"]): row["cnt"] for row in rows}


def compute_news_monthly(
    pg: PostgresStorage, start: date, end: date, symbols: list[str]
) -> dict[tuple[str, str], int]:
    rows = pg.fetch_all(
        """
        SELECT publish_date, title, summary, text
        FROM news_raw
        WHERE publish_date >= %(start)s AND publish_date <= %(end)s
        """,
        {"start": start, "end": end},
    )
    patterns: dict[str, re.Pattern[str]] = {}
    for symbol in symbols:
        aliases = SYMBOL_ALIASES.get(symbol, [])
        tokens = [symbol] + aliases
        escaped = [re.escape(t) for t in tokens if t]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        patterns[symbol] = re.compile(pattern, re.IGNORECASE)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        published = row.get("publish_date")
        if not published:
            continue
        ym = month_key(published.date())
        text = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("summary") or ""),
                str(row.get("text") or ""),
            ]
        )
        for symbol, pattern in patterns.items():
            if pattern.search(text):
                counts[(symbol, ym)] += 1
    return counts


def build_coverage_rows(
    domain: str,
    symbols: list[str],
    months: list[str],
    counts: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for ym in months:
            count = counts.get((symbol, ym), 0)
            key = f"{domain}|{symbol}|{ym}"
            rows.append(
                {
                    "key": key,
                    "domain": domain,
                    "symbol": symbol,
                    "year_month": ym,
                    "count": count,
                }
            )
    return rows


def report_missing(domain: str, symbols: list[str], months: list[str], counts: dict[tuple[str, str], int]) -> list[tuple[str, str]]:
    missing = []
    for symbol in symbols:
        for ym in months:
            if counts.get((symbol, ym), 0) == 0:
                missing.append((symbol, ym))
    logger.info("%s coverage missing %s/%s symbol-month pairs", domain, len(missing), len(symbols) * len(months))
    if missing:
        logger.info("%s coverage sample missing: %s", domain, missing[:10])
    return missing


def report_below_threshold(
    domain: str,
    symbols: list[str],
    months: list[str],
    counts: dict[tuple[str, str], int],
    threshold: int,
) -> list[tuple[str, str]]:
    missing = []
    for symbol in symbols:
        for ym in months:
            if counts.get((symbol, ym), 0) < threshold:
                missing.append((symbol, ym))
    logger.info(
        "%s coverage below threshold %s: %s/%s symbol-month pairs",
        domain,
        threshold,
        len(missing),
        len(symbols) * len(months),
    )
    if missing:
        logger.info("%s below-threshold sample: %s", domain, missing[:10])
    return missing


def report_corp_yearly(
    pg: PostgresStorage,
    symbols: list[str],
    start: date,
    end: date,
    min_per_year: int,
) -> list[tuple[str, int]]:
    rows = pg.fetch_all(
        """
        SELECT symbol, EXTRACT(YEAR FROM COALESCE(record_date, ex_date, effective_date))::int AS yr, COUNT(*) AS cnt
        FROM public.corp_actions
        WHERE COALESCE(record_date, ex_date, effective_date) >= %(start)s
          AND COALESCE(record_date, ex_date, effective_date) <= %(end)s
        GROUP BY symbol, yr
        """,
        {"start": start, "end": end},
    )
    counts: dict[tuple[str, int], int] = {(row["symbol"], row["yr"]): row["cnt"] for row in rows}
    missing: list[tuple[str, int]] = []
    for symbol in symbols:
        for year in range(start.year, end.year + 1):
            if counts.get((symbol, year), 0) < min_per_year:
                missing.append((symbol, year))
    logger.info(
        "corp yearly below threshold %s: %s/%s symbol-years",
        min_per_year,
        len(missing),
        len(symbols) * (end.year - start.year + 1),
    )
    if missing:
        logger.info("corp yearly below-threshold sample: %s", missing[:10])
    return missing


def find_missing_trading_ranges(pg: PostgresStorage, symbol: str, start: date, end: date) -> list[tuple[date, date]]:
    rows = pg.fetch_all(
        """
        SELECT date
        FROM prices
        WHERE symbol = %(symbol)s AND date >= %(start)s AND date <= %(end)s
        ORDER BY date ASC
        """,
        {"symbol": symbol, "start": start, "end": end},
    )
    existing = {row["date"] for row in rows}
    missing_dates: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in existing:
            missing_dates.append(current)
        current += timedelta(days=1)
    if not missing_dates:
        return []
    ranges: list[tuple[date, date]] = []
    range_start = missing_dates[0]
    prev = missing_dates[0]
    for dt in missing_dates[1:]:
        if (dt - prev).days > 1:
            ranges.append((range_start, prev))
            range_start = dt
        prev = dt
    ranges.append((range_start, prev))
    return ranges
