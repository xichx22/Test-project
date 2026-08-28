from .base import DataSource, Interval, Symbol
from .csv_source import CsvDataSource
from .synthetic import SyntheticDataSource
from .toss import TossClient, TossDataSource

__all__ = [
    "DataSource", "Interval", "Symbol",
    "CsvDataSource", "SyntheticDataSource", "TossClient", "TossDataSource",
]


def get_source(name: str, **kwargs) -> DataSource:
    table = {
        "toss": TossDataSource,
        "csv": CsvDataSource,
        "synthetic": SyntheticDataSource,
    }
    if name not in table:
        raise KeyError(f"알 수 없는 데이터 소스 '{name}'. 사용 가능: {sorted(table)}")
    return table[name](**kwargs)
