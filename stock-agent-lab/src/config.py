from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    worldnews_api_key: str
    worldnews_source_country: str
    worldnews_language: str
    worldnews_page_size: int
    fireant_base_url: str
    http_sleep_secs: float
    http_timeout_secs: float
    http_max_retries: int
    smoke_symbols: list[str]
    smoke_days: int


def load_settings() -> Settings:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "")
    worldnews_api_key = os.getenv("WORLDNEWS_API_KEY", "")
    worldnews_source_country = os.getenv("WORLDNEWS_SOURCE_COUNTRY", "vn")
    worldnews_language = os.getenv("WORLDNEWS_LANGUAGE", "vi")
    worldnews_page_size = int(os.getenv("WORLDNEWS_PAGE_SIZE", "100"))
    fireant_base_url = os.getenv("FIREANT_BASE_URL", "https://www.fireant.vn")
    http_sleep_secs = float(os.getenv("HTTP_SLEEP_SECS", "0.2"))
    http_timeout_secs = float(os.getenv("HTTP_TIMEOUT_SECS", "30"))
    http_max_retries = int(os.getenv("HTTP_MAX_RETRIES", "5"))
    smoke_symbols = [
        s.strip().upper()
        for s in os.getenv("SMOKE_SYMBOLS", "HPG,FPT").split(",")
        if s.strip()
    ]
    smoke_days = int(os.getenv("SMOKE_DAYS", "7"))
    return Settings(
        database_url=database_url,
        worldnews_api_key=worldnews_api_key,
        worldnews_source_country=worldnews_source_country,
        worldnews_language=worldnews_language,
        worldnews_page_size=worldnews_page_size,
        fireant_base_url=fireant_base_url,
        http_sleep_secs=http_sleep_secs,
        http_timeout_secs=http_timeout_secs,
        http_max_retries=http_max_retries,
        smoke_symbols=smoke_symbols,
        smoke_days=smoke_days,
    )
