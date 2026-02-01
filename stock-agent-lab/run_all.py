from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime, timedelta

from src.config import load_settings
from src.collectors.vn30_universe import VN30_SYMBOLS
from src.pipelines.collect_corp_actions import collect_corp_actions_default
from src.pipelines.collect_news import collect_news_default
from src.pipelines.collect_news import collect_news_deepen_months
from src.pipelines.collect_prices import collect_prices_default, gap_fill_prices
from src.pipelines.run_agents import run_agents
from src.storage.pg import PostgresStorage
from src.utils.coverage import (
    build_coverage_rows,
    report_below_threshold,
    report_corp_yearly,
    compute_corp_monthly,
    compute_news_monthly,
    compute_price_monthly,
    iter_months,
    report_missing,
    upsert_coverage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VN30 joint news+price pipeline")
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is required")
    pg = PostgresStorage(settings.database_url)
    if args.smoke:
        end = date.today()
        start = end - timedelta(days=settings.smoke_days)
        collect_news_default(pg, settings, start, end, smoke_mode=True)
        collect_prices_default(pg, settings, start, end, smoke_mode=True)
        collect_corp_actions_default(pg, settings, start, end, smoke_mode=True)
        run_agents(pg, horizon_days=args.horizon)
    else:
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required unless --smoke")
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
        collect_news_default(pg, settings, start, end, smoke_mode=False)
        collect_prices_default(pg, settings, start, end, smoke_mode=False)
        collect_corp_actions_default(pg, settings, start, end, smoke_mode=False)
        run_agents(pg, horizon_days=args.horizon)
        _run_coverage_and_retry(pg, settings, start, end)
    counts = pg.table_counts()
    logging.info("Table counts: %s", counts)
    top_symbols = pg.fetch_all(
        """
        SELECT symbol, COUNT(*) AS cnt
        FROM news_events
        GROUP BY symbol
        ORDER BY cnt DESC
        LIMIT 10
        """
    )
    logging.info("Top symbols by events: %s", top_symbols)
    latest_price = pg.latest_price_date()
    logging.info("Latest price date: %s", latest_price)


def _run_coverage_and_retry(pg: PostgresStorage, settings, start: date, end: date) -> None:
    news_target = 5
    price_target = 24
    corp_target_per_year = 1
    months = iter_months(start, end)
    symbols = VN30_SYMBOLS
    price_counts = compute_price_monthly(pg, start, end)
    news_counts = compute_news_monthly(pg, start, end, symbols)
    corp_counts = compute_corp_monthly(pg, start, end)
    upsert_coverage(pg, build_coverage_rows("prices", symbols, months, price_counts))
    upsert_coverage(pg, build_coverage_rows("news", symbols, months, news_counts))
    upsert_coverage(pg, build_coverage_rows("corp", symbols, months, corp_counts))
    missing_prices = report_below_threshold("prices", symbols, months, price_counts, price_target)
    missing_news = report_below_threshold("news", symbols, months, news_counts, news_target)
    missing_corp_years = report_corp_yearly(pg, symbols, start, end, corp_target_per_year)
    if missing_prices:
        logging.info("Retry price gaps (1 pass)")
        gap_fill_prices(pg, settings, symbols, start, end)
    if missing_corp_years:
        missing_symbols = sorted({symbol for symbol, _ in missing_corp_years})
        logging.info("Retry corp actions for %s symbols", len(missing_symbols))
        collect_corp_actions_default(pg, settings, start, end, smoke_mode=False)
    if missing_news and os.getenv("NEWS_DEEPEN", "0") == "1":
        missing_symbols = sorted({symbol for symbol, _ in missing_news})
        per_symbol_limit = int(os.getenv("NEWS_DEEPEN_PER_SYMBOL", "20"))
        logging.info("Retry news deepen for %s symbols", len(missing_symbols))
        symbols_by_month: dict[str, list[str]] = {}
        for symbol, ym in missing_news:
            symbols_by_month.setdefault(ym, []).append(symbol)
        collect_news_deepen_months(pg, settings, symbols_by_month, per_symbol_limit)

    # Recompute coverage after one retry pass
    price_counts = compute_price_monthly(pg, start, end)
    news_counts = compute_news_monthly(pg, start, end, symbols)
    corp_counts = compute_corp_monthly(pg, start, end)
    upsert_coverage(pg, build_coverage_rows("prices", symbols, months, price_counts))
    upsert_coverage(pg, build_coverage_rows("news", symbols, months, news_counts))
    upsert_coverage(pg, build_coverage_rows("corp", symbols, months, corp_counts))
    remaining_prices = report_below_threshold("prices", symbols, months, price_counts, price_target)
    remaining_news = report_below_threshold("news", symbols, months, news_counts, news_target)
    remaining_corp_years = report_corp_yearly(pg, symbols, start, end, corp_target_per_year)
    if remaining_prices or remaining_news or remaining_corp_years:
        logging.error("Coverage targets not met after retry pass.")
        logging.error("Remaining price gaps: %s", len(remaining_prices))
        logging.error("Remaining news gaps: %s", len(remaining_news))
        logging.error("Remaining corp year gaps: %s", len(remaining_corp_years))
        raise SystemExit("Coverage targets not met; see logs for missing symbol-months/years.")


if __name__ == "__main__":
    main()
