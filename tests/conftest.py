import pandas as pd
import pytest

from tsignal.datasource import Interval, SyntheticDataSource


@pytest.fixture(scope="session")
def candles_5m() -> pd.DataFrame:
    return SyntheticDataSource(seed=1234).candles("005930", Interval.M5, count=2500)


@pytest.fixture(scope="session")
def candles_1d() -> pd.DataFrame:
    return SyntheticDataSource(seed=1234).candles("005930", Interval.D1, count=600)
