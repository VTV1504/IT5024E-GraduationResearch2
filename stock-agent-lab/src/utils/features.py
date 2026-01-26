from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def returns(series: Iterable[float]) -> list[float]:
    values = list(series)
    rets = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]
        if prev is None or prev == 0:
            rets.append(0.0)
        else:
            rets.append(curr / prev - 1)
    return rets


def moving_average(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return float(np.mean(values[-window:]))


def rsi(values: list[float], window: int = 14) -> float | None:
    if len(values) < window + 1:
        return None
    gains = []
    losses = []
    for i in range(1, window + 1):
        delta = values[-i] - values[-i - 1]
        if delta >= 0:
            gains.append(delta)
        else:
            losses.append(abs(delta))
    avg_gain = sum(gains) / window if gains else 0
    avg_loss = sum(losses) / window if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def max_drawdown(values: list[float]) -> float | None:
    if not values:
        return None
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v / peak) - 1.0 if peak else 0.0
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def zscore(value: float | None, series: list[float]) -> float | None:
    if value is None:
        return None
    if len(series) < 2:
        return None
    mean = float(np.mean(series))
    std = float(np.std(series, ddof=1))
    if math.isclose(std, 0.0):
        return None
    return (value - mean) / std
