from .base import DataSource, Interval, Symbol
from .csv_source import CsvDataSource
from .naver import NaverDataSource
from .universe_list import fetch_listing, fetch_universe
from .synthetic import SyntheticDataSource
from .toss import TossClient, TossDataSource

__all__ = [
    "DataSource", "Interval", "Symbol",
    "CsvDataSource", "NaverDataSource", "fetch_listing", "fetch_universe", "SyntheticDataSource", "TossClient", "TossDataSource",
]


def get_source(name: str, **kwargs) -> DataSource:
    table = {
        "toss": TossDataSource,
        "csv": CsvDataSource,
        "naver": NaverDataSource,
        "synthetic": SyntheticDataSource,
    }
    if name not in table:
        raise KeyError(f"알 수 없는 데이터 소스 '{name}'. 사용 가능: {sorted(table)}")
    return table[name](**kwargs)
