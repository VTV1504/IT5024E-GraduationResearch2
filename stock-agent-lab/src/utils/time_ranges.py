from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator

from dateutil.relativedelta import relativedelta


def monthly_ranges(start: date, end: date) -> Iterator[tuple[date, date]]:
    current = date(start.year, start.month, 1)
    while current <= end:
        next_month = current + relativedelta(months=1)
        chunk_end = min(end, next_month - timedelta(days=1))
        yield current, chunk_end
        current = next_month
