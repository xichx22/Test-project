"""합성 캔들 소스.

네트워크 없이 파이프라인 전체(지표→신호→검증)를 돌리고 테스트하기 위한 소스.
GBM + 장중 U자 거래량 + 세션 경계까지 재현해서, 분봉 로직의 세션 처리 버그가
합성 데이터에서도 드러나게 만들었다.
"""

from __future__ import annotations

import zlib
from datetime import datetime

import numpy as np
import pandas as pd

from ..ohlcv import KST, validate
from .base import DataSource, Interval

SESSION_OPEN = (9, 0)
SESSION_CLOSE = (15, 30)


def _session_index(days: int, interval: Interval, end: pd.Timestamp) -> pd.DatetimeIndex:
    if interval is Interval.D1:
        return pd.bdate_range(end=end.normalize(), periods=days, tz=KST)
    step = interval.minutes or 1
    stamps: list[pd.Timestamp] = []
    for day in pd.bdate_range(end=end.normalize(), periods=days, tz=KST):
        start = day + pd.Timedelta(hours=SESSION_OPEN[0], minutes=SESSION_OPEN[1])
        stop = day + pd.Timedelta(hours=SESSION_CLOSE[0], minutes=SESSION_CLOSE[1])
        stamps.extend(pd.date_range(start, stop, freq=f"{step}min", inclusive="left"))
    return pd.DatetimeIndex(stamps)


class SyntheticDataSource(DataSource):
    name = "synthetic"

    def __init__(
        self,
        *,
        seed: int = 20260828,
        start_price: float = 70_000.0,
        annual_vol: float = 0.35,
        drift: float = 0.0,
    ) -> None:
        self.seed = seed
        self.start_price = start_price
        self.annual_vol = annual_vol
        self.drift = drift

    def candles(
        self,
        code: str,
        interval: Interval,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        # 코드별로 다른(그러나 재현 가능한) 시계열을 준다.
        # 파이썬 내장 hash() 는 PYTHONHASHSEED 로 프로세스마다 달라져 재현성이 깨진다.
        # 재현 가능한 검증이 이 프로젝트의 전제이므로 안정 해시를 쓴다.
        rng = np.random.default_rng(self.seed + (zlib.crc32(code.encode("utf-8")) % 10_000))
        last = pd.Timestamp(end, tz=KST) if end is not None else pd.Timestamp.now(tz=KST)

        bars_per_day = 1 if interval is Interval.D1 else max(1, 390 // (interval.minutes or 1))
        days = max(5, -(-(count or 2000) // bars_per_day) + 2)
        idx = _session_index(days, interval, last)
        if count is not None:
            idx = idx[-count:]

        n = len(idx)
        bars_per_year = 252 * bars_per_day
        sigma = self.annual_vol / np.sqrt(bars_per_year)
        mu = self.drift / bars_per_year

        ret = rng.normal(mu, sigma, n)
        close = self.start_price * np.exp(np.cumsum(ret))
        open_ = np.concatenate([[self.start_price], close[:-1]])
        wick = np.abs(rng.normal(0, sigma, n)) * close
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick

        if interval is Interval.D1:
            shape = np.ones(n)
        else:  # 장중 U자 거래량
            pos = np.arange(n) % bars_per_day / max(1, bars_per_day - 1)
            shape = 1.0 + 2.5 * (np.exp(-6 * pos) + np.exp(-6 * (1 - pos)))
        volume = np.round(rng.lognormal(9.5, 0.4, n) * shape)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        df.index.name = "dt"
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz=KST)]
        return validate(df.astype("float64"))
