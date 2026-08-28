"""로컬 CSV/Parquet 소스.

한 번 받아둔 캔들을 재현 가능하게 다시 읽기 위한 소스.
검증 실험은 항상 같은 입력에서 같은 결과가 나와야 하므로, 실전 수집은
CSV 로 떨어뜨려 두고 실험은 이 소스로 도는 것을 기본으로 한다.

디렉터리 규약: {root}/{interval}/{code}.csv   예) data/1m/005930.csv
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..ohlcv import normalize, validate
from .base import DataSource, Interval, Symbol


class CsvDataSource(DataSource):
    name = "csv"

    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)

    def path_for(self, code: str, interval: Interval) -> Path:
        return self.root / interval.value / f"{code}.csv"

    def candles(
        self,
        code: str,
        interval: Interval,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        path = self.path_for(code, interval)
        if not path.exists():
            raise FileNotFoundError(f"캔들 파일이 없습니다: {path}")
        df = validate(normalize(pd.read_csv(path)))
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]
        if count is not None:
            df = df.tail(count)
        return df

    def save(self, df: pd.DataFrame, code: str, interval: Interval) -> Path:
        path = self.path_for(code, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index_label="dt")
        return path

    def symbols(self) -> list[Symbol]:
        codes = {p.stem for p in self.root.rglob("*.csv")}
        return [Symbol(code=c) for c in sorted(codes)]
