# VN30 Stock Agent Lab

End-to-end pipeline to collect VN30 news + daily prices, store in PostgreSQL, and build joint news+price learning agents.

## Prerequisites
- Docker
- Python 3.10+

## Setup
1) Start PostgreSQL:
```bash
docker compose up -d
```

2) Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Configure environment variables:
```bash
cp .env.example .env
```
Edit `.env` and set `WORLDNEWS_API_KEY`.

## Run smoke mode
```bash
python run_smoke.py
```

## Run full backfill
```bash
python run_all.py --start 2023-01-01 --end 2026-01-19
```

## Troubleshooting
- WorldNewsAPI quota issues: reduce symbols, increase sleep, run incremental, rely on resume checkpoints.
- FireAnt HO prefix failures: collector auto retries with the raw ticker.
