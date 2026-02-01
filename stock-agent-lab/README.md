# VN30 Stock Agent Lab

End-to-end pipeline to collect VN30 news + daily prices, store in PostgreSQL, and build joint news+price learning agents.

## Setup
1) docker compose up -d
2) python -m venv .venv && activate
3) pip install -r requirements.txt
4) copy .env.example .env
5) python run_smoke.py
6) python run_all.py

## Run all expectations
- Full 3-year backfill for VN30 can take hours. Keep the terminal open.
- To deepen news coverage, set `NEWS_DEEPEN=1` and `NEWS_DEEPEN_PER_SYMBOL=20`.
- Coverage targets enforced in full mode: news >= 5/month per symbol, prices >= 24 trading days/month, corp actions >= 1/year.
- Use `NEWS_MODE=category` (default) for faster category crawling; `NEWS_MODE=sitemap` is available as fallback.
- Reports are written to `reports/` as `agent*_report_YYYYMMDD_HHMMSS.txt`.

## Run all sanity
- `python run_all.py --smoke`

## Troubleshooting
- If a site blocks requests: increase HTTP_SLEEP_SECS.
- If parsing fails: adjust selectors in src/utils/html.py.
