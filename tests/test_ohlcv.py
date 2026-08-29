import pandas as pd
import pytest

from tsignal.ohlcv import OhlcvError, normalize, repair, resample, validate


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


def test_repair_fixes_a_one_tick_violation_on_a_cheap_stock():
    """저가주의 1원 반올림 오차는 보정돼야 한다.

    비율 문턱만 쓰면 169원짜리 주가의 1원 차이(0.59%)가 0.5% 를 넘어
    종목이 통째로 버려진다. 하필 가장 오래된 종목들만 골라서 빠지므로
    표본에서 옛날 구간이 계통적으로 사라진다.
    """
    index = pd.date_range("2004-01-01", periods=3, freq="D", tz="Asia/Seoul")
    frame = pd.DataFrame(
        {"open": [169.0, 100.0, 100.0], "high": [169.0, 101.0, 101.0],
         "low": [169.0, 99.0, 99.0], "close": [170.0, 100.0, 100.0],
         "volume": [10.0, 10.0, 10.0]},
        index=index,
    )
    fixed, log = repair(frame)
    assert len(log) == 1
    validate(fixed)                       # 이제 통과해야 한다
    assert fixed["high"].iloc[0] == 170.0


def test_repair_still_rejects_genuinely_broken_rows():
    """거래정지 행(가격 0)은 절대 오차도 가격 전체라 보정되면 안 된다."""
    index = pd.date_range("2004-01-01", periods=2, freq="D", tz="Asia/Seoul")
    frame = pd.DataFrame(
        {"open": [50_000.0, 100.0], "high": [0.0, 101.0], "low": [0.0, 99.0],
         "close": [50_000.0, 100.0], "volume": [0.0, 10.0]},
        index=index,
    )
    fixed, log = repair(frame)
    assert log.empty
    with pytest.raises(OhlcvError):
        validate(fixed)


def test_repair_tick_allowance_scales_with_price_band():
    """허용치는 가격대별 호가 단위를 따라야 한다.

    500원짜리의 3원과 50,000원짜리의 3원은 전혀 다른 이야기다.
    """
    from tsignal.ohlcv import tick_size

    price = pd.Series([500.0, 3_000.0, 30_000.0, 300_000.0])
    assert list(tick_size(price)) == [1.0, 5.0, 50.0, 500.0]


def test_repair_rejects_a_gap_far_beyond_a_few_ticks():
    """호가 몇 틱을 크게 넘는 차이는 반올림이 아니다."""
    index = pd.date_range("2020-01-01", periods=1, freq="D", tz="Asia/Seoul")
    frame = pd.DataFrame(
        {"open": [100.0], "high": [100.0], "low": [100.0], "close": [120.0],
         "volume": [10.0]},
        index=index,
    )
    _, log = repair(frame)               # 20원 = 20틱, 20%
    assert log.empty
