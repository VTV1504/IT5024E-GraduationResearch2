from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class HttpClient:
    def __init__(self, timeout: float, max_retries: int, sleep_secs: float):
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_secs = sleep_secs
        self.session = requests.Session()

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> requests.Response:
        attempt = 0
        while True:
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"Retryable status {response.status_code}", response=response)
                return response
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.warning("Max retries exceeded for %s: %s", url, exc)
                    raise
                backoff = self.sleep_secs * (2 ** (attempt - 1))
                logger.warning("Retrying %s in %.2fs due to %s", url, backoff, exc)
                time.sleep(backoff)
