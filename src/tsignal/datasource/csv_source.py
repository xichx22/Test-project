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

from ..ohlcv import KST, normalize, validate
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

    def extras_path(self, code: str, interval: Interval) -> Path:
        return self.root / "extras" / interval.value / f"{code}.csv"

    def save_extras(self, df: pd.DataFrame, code: str, interval: Interval) -> Path:
        path = self.extras_path(code, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index_label="dt")
        return path

    def load_extras(self, code: str, interval: Interval) -> pd.DataFrame:
        """부가 데이터(수급 등). 없으면 빈 DataFrame."""
        path = self.extras_path(code, interval)
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path, parse_dates=["dt"]).set_index("dt")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize(KST)
        return frame

    def load_all_extras(self, interval: Interval) -> dict[str, pd.DataFrame]:
        out = {}
        for symbol in self.symbols(interval):
            frame = self.load_extras(symbol.code, interval)
            if not frame.empty:
                out[symbol.code] = frame
        return out

    def flow_path(self, code: str, interval: Interval) -> Path:
        return self.root / "flow" / interval.value / f"{code}.csv"

    def load_flow(self, code: str, interval: Interval) -> pd.DataFrame:
        """투자자별 수급(기관·외국인 순매매량). 없으면 빈 DataFrame."""
        path = self.flow_path(code, interval)
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path, parse_dates=["dt"]).set_index("dt")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize(KST)
        return frame.sort_index()

    def load_all_flow(self, interval: Interval) -> dict[str, pd.DataFrame]:
        out = {}
        for symbol in self.symbols(interval):
            frame = self.load_flow(symbol.code, interval)
            if not frame.empty:
                out[symbol.code] = frame
        return out

    def symbols(self, interval: Interval | None = None) -> list[Symbol]:
        """저장된 종목 목록.

        interval 을 주면 그 타임프레임 디렉터리만 본다. 루트를 통째로 훑으면
        `data/universe.csv` 같은 메타 파일까지 종목으로 오인한다.
        """
        roots = [self.root / interval.value] if interval else [
            self.root / iv.value for iv in Interval
        ]
        codes = {p.stem for root in roots if root.is_dir() for p in root.glob("*.csv")}
        return [Symbol(code=c) for c in sorted(codes)]

    def load_all(self, interval: Interval) -> dict[str, pd.DataFrame]:
        """그 타임프레임에 저장된 모든 종목을 한 번에 읽는다."""
        return {s.code: self.candles(s.code, interval) for s in self.symbols(interval)}
