import pandas as pd
import pytest

from tsignal.ohlcv import OhlcvError, normalize, resample, validate


def test_normalize_accepts_common_aliases():
    df = pd.DataFrame({
        "Date": ["2024-01-02 09:00", "2024-01-02 09:01"],
        "O": [100, 101], "H": [102, 103], "L": [99, 100], "C": [101, 102], "V": [10, 20],
    })
    out = validate(normalize(df))
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "Asia/Seoul"


def test_validate_rejects_inconsistent_high_low(candles_5m):
    broken = candles_5m.copy()
    broken.iloc[5, broken.columns.get_loc("high")] = broken["low"].iloc[5] - 1
    with pytest.raises(OhlcvError):
        validate(broken)


def test_validate_rejects_unsorted_index(candles_5m):
    with pytest.raises(OhlcvError):
        validate(candles_5m.iloc[::-1])


def test_resample_preserves_ohlc_semantics(candles_5m):
    fifteen = resample(candles_5m, "15min")
    assert len(fifteen) < len(candles_5m)
    assert fifteen["high"].max() <= candles_5m["high"].max() + 1e-9
    assert fifteen["volume"].sum() == pytest.approx(candles_5m["volume"].sum())
