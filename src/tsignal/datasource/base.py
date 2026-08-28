"""데이터 소스 추상화.

토스든 KIS든 CSV든 `DataSource` 하나만 만족하면 파이프라인 전체가 그대로 돈다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd


class Interval(str, Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"

    @property
    def pandas_rule(self) -> str:
        return {
            "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "1h", "1d": "1D",
        }[self.value]

    @property
    def minutes(self) -> int | None:
        """일봉은 None. 분 단위 환산값 (연율화/기간 환산에 쓴다)."""
        return {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(self.value)


@dataclass(frozen=True)
class Symbol:
    code: str          # 6자리 단축코드, 예: "005930"
    name: str = ""
    market: str = ""   # KOSPI / KOSDAQ

    @property
    def toss_code(self) -> str:
        return self.code if self.code.startswith("A") else f"A{self.code}"


class DataSource(ABC):
    """캔들 공급자."""

    name: str = "base"

    @abstractmethod
    def candles(
        self,
        code: str,
        interval: Interval,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        """OHLCV 규격 DataFrame 반환."""

    def symbols(self) -> list[Symbol]:
        """유니버스. 지원하지 않으면 빈 리스트."""
        return []

    def supports(self, interval: Interval) -> bool:
        return True
