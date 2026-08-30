"""조건을 얹은 신호는 '같은 조건을 건 기준선' 과만 비교해야 한다."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.conditional import (
    cell, consistent, context_table, grid, signal_mask,
)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-02", periods=n, freq="B", tz="Asia/Seoul")
    close = 10_000 * np.cumprod(1 + rng.normal(0, 0.02, n))
    return pd.DataFrame({
        "open": close, "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": rng.integers(1e4, 1e6, n).astype(float),
    }, index=index)


def _features(frame):
    close = frame["close"]
    return pd.DataFrame({
        "ema_20": close.ewm(span=20).mean(), "ema_60": close.ewm(span=60).mean(),
        "rsi_14": pd.Series(np.linspace(20, 80, len(close)), index=close.index),
        "williams_r_14": pd.Series(-50.0, index=close.index),
        "macd_hist": close.diff().fillna(0.0),
        "stoch_k": pd.Series(50.0, index=close.index),
        "stoch_d": pd.Series(40.0, index=close.index),
        "cmf_20": pd.Series(0.1, index=close.index),
        "mfi_14": pd.Series(55.0, index=close.index),
        "bb_upper": close * 1.05, "bb_lower": close * 0.95,
        "senkou_a": close * 0.98, "senkou_b": close * 0.97,
    }, index=close.index)


def test_forward_column_enters_at_next_open():
    """결과는 다음 봉 시가 진입 기준이다. 종가 진입으로 재면 갭이 섞인다."""
    frame = _frame(120)
    frame.loc[frame.index[50], "open"] = frame["close"].iloc[50] * 1.10
    table = context_table({"A": frame}, {"A": _features(frame)}, horizons=(20,))
    row = table.loc[frame.index[49]]
    expected = frame["close"].iloc[49 + 21] / frame["open"].iloc[50] - 1
    assert row["fwd20"] == pytest.approx(expected)


def test_context_columns_do_not_look_ahead():
    """정황 컬럼을 미래 값으로 바꿔도 값이 변하면 안 된다."""
    frame = _frame(300)
    table = context_table({"A": frame}, {"A": _features(frame)}, horizons=(20,))
    cut = 200
    tampered = frame.copy()
    tampered.iloc[cut + 1:] *= 3.0
    later = context_table({"A": tampered}, {"A": _features(tampered)}, horizons=(20,))
    cols = ["RSI", "거래량배수", "52주고점대비", "3개월수익률", "변동성"]
    a = table[cols].iloc[:cut].astype(float)
    b = later[cols].iloc[:cut].astype(float)
    pd.testing.assert_frame_equal(a, b)


def test_baseline_is_restricted_to_the_same_condition():
    """조건을 걸면 기준선도 같이 걸려야 한다.

    조건이 그냥 좋은 날을 고른 것뿐이면 차이는 0 이 나와야 한다.
    """
    frame = _frame(400, seed=3)
    table = context_table({"A": frame}, {"A": _features(frame)}, horizons=(20,))
    good = (table["fwd20"] > table["fwd20"].median()).to_numpy()
    rng = np.random.default_rng(1)
    # 신호는 조건 안에서 무작위로 고른다 — 정보가 없다
    sig = good & (rng.random(len(table)) < 0.5)
    got = cell(table, sig, good, horizon=20, min_signals=10,
               min_year_signals=5, min_year_baseline=10)
    assert abs(got["차이"]) < 0.01


def test_ignoring_the_condition_manufactures_a_fake_edge():
    """같은 신호도 기준선을 전체로 두면 이긴 것처럼 보인다."""
    frame = _frame(400, seed=3)
    table = context_table({"A": frame}, {"A": _features(frame)}, horizons=(20,))
    good = (table["fwd20"] > table["fwd20"].median()).to_numpy()
    rng = np.random.default_rng(1)
    sig = good & (rng.random(len(table)) < 0.5)
    everything = np.ones(len(table), bool)
    honest = cell(table, sig, good, horizon=20, min_signals=10,
                  min_year_signals=5, min_year_baseline=10)
    inflated = cell(table, sig, everything, horizon=20, min_signals=10,
                    min_year_signals=5, min_year_baseline=10)
    assert inflated["차이"] > honest["차이"] + 0.02


def test_signal_mask_matches_code_and_date():
    """마스크는 종목과 날짜가 **둘 다** 맞을 때만 켜져야 한다."""
    a, b = _frame(60, seed=1), _frame(60, seed=2)
    table = context_table({"A": a, "B": b},
                          {"A": _features(a), "B": _features(b)}, horizons=(20,))
    hit = pd.Series(False, index=a.index)
    hit.iloc[10] = True
    mask = signal_mask(table, {"A": hit})
    assert mask.sum() == 1
    assert table.loc[mask, "code"].iloc[0] == "A"
    assert table.index[mask][0] == a.index[10]


def test_consistent_drops_cells_carried_by_one_year():
    frame = pd.DataFrame([
        {"신호": "x", "조건": "c", "차이": 0.05, "연도승": 6, "연도수": 6},
        {"신호": "y", "조건": "c", "차이": 0.09, "연도승": 1, "연도수": 6},
        {"신호": "z", "조건": "c", "차이": -0.01, "연도승": 6, "연도수": 6},
    ])
    kept = consistent(frame)
    assert list(kept["신호"]) == ["x"]


def test_grid_returns_empty_frame_when_nothing_qualifies():
    frame = _frame(80)
    table = context_table({"A": frame}, {"A": _features(frame)}, horizons=(20,))
    out = grid(table, {"s": np.zeros(len(table), bool)},
               {"c": np.ones(len(table), bool)})
    assert out.empty
