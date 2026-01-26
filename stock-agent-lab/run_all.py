from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta

from src.collectors.vn30_universe import VN30_SYMBOLS
from src.config import load_settings
from src.pipelines.collect_news import collect_news_default
from src.pipelines.collect_prices import collect_prices_default
from src.pipelines.run_agents import run_agents
from src.storage.pg import PostgresStorage

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
        run_agents(pg, horizon_days=args.horizon)
    else:
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required unless --smoke")
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
        collect_news_default(pg, settings, start, end, smoke_mode=False)
        collect_prices_default(pg, settings, start, end, smoke_mode=False)
        run_agents(pg, horizon_days=args.horizon)
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


if __name__ == "__main__":
    main()
