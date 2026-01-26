from __future__ import annotations

import logging
from datetime import date, timedelta

from src.config import load_settings
from src.pipelines.collect_news import collect_news_default
from src.pipelines.collect_prices import collect_prices_default
from src.pipelines.run_agents import run_agents
from src.storage.pg import PostgresStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    settings = load_settings()
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is required")
    pg = PostgresStorage(settings.database_url)
    end = date.today()
    start = end - timedelta(days=settings.smoke_days)
    collect_news_default(pg, settings, start, end, smoke_mode=True)
    collect_prices_default(pg, settings, start, end, smoke_mode=True)
    run_agents(pg, horizon_days=40)
    counts = pg.table_counts()
    logging.info("Table counts: %s", counts)


if __name__ == "__main__":
    main()
