"""경로 프로파일러 — 기술통계이지 검정이 아니다."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.pathprofile import (
    baseline_paths, default_exit_rules, exit_timing,
    forward_paths, path_shape, turn_signature,
)


def _frame(close, volume=None, open_=None):
    close = np.asarray(close, float)
    n = len(close)
    index = pd.date_range("2020-01-01", periods=n, freq="B", tz="Asia/Seoul")
    return pd.DataFrame(
        {"open": close if open_ is None else np.asarray(open_, float),
         "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": np.full(n, 1000.0) if volume is None else np.asarray(volume, float)},
        index=index,
    )


def _signal(frame, positions):
    hit = pd.Series(False, index=frame.index)
    for p in positions:
        hit.iloc[p] = True
    return hit


def test_entry_is_the_next_bar_open():
    """종가 기준으로 재면 사기 전에 벌어진 갭이 성과에 들어간다."""
    close = np.r_[np.full(10, 100.0), np.full(30, 110.0)]
    open_ = close.copy()
    open_[10] = 110.0                       # 신호 다음 봉이 갭 상승해 시작
    frame = _frame(close, open_=open_)
    paths = forward_paths({"A": _signal(frame, [9])}, {"A": frame}, horizon=5)
    assert paths.shape == (1, 5)
    assert paths[0, 0] == pytest.approx(0.0)   # 110/110 - 1, 갭을 못 먹는다


def test_short_windows_are_kept_as_nan_not_dropped():
    """구간이 모자란 신호를 버리면 최근 신호가 통째로 빠진다."""
    frame = _frame(np.full(20, 100.0))
    paths = forward_paths({"A": _signal(frame, [17])}, {"A": frame}, horizon=10)
    assert paths.shape == (1, 10)
    assert np.isnan(paths[0]).sum() > 0
    assert not np.isnan(paths[0, 0])


def test_path_shape_finds_the_peak_day():
    up = np.r_[np.linspace(100, 120, 10), np.linspace(120, 90, 10)]
    frame = _frame(np.r_[np.full(5, 100.0), up])
    paths = forward_paths({"A": _signal(frame, [4])}, {"A": frame}, horizon=20)
    shape = path_shape(paths)
    assert shape.n == 1
    assert 8 <= shape.days_to_peak[0] <= 12
    assert shape.mfe[0] > 0 > shape.mae[0]


def test_path_shape_survives_all_nan_rows():
    """전부 NaN 인 행이 있으면 argmax 가 터진다 — 마스크로 걸러야 한다."""
    paths = np.full((3, 5), np.nan)
    paths[0] = np.linspace(0.01, 0.05, 5)
    shape = path_shape(paths)
    assert shape.n == 1
    assert np.isfinite(shape.mfe).all()


def test_path_shape_handles_no_events():
    shape = path_shape(np.empty((0, 10)))
    assert shape.n == 0
    assert shape.summary().empty


def test_baseline_samples_the_whole_period():
    """신호 경로를 이것과 나란히 놓지 않으면 시장 상승과 구별할 수 없다."""
    frame = _frame(np.linspace(100, 200, 300))
    base = baseline_paths({"A": frame}, horizon=20, step=20, seed=1)
    assert len(base) > 5
    assert np.nanmedian(base[:, -1]) > 0        # 우상향 구간이므로 양수


def test_exit_timing_prefers_the_rule_that_beats_holding():
    """조건에 판 쪽이 나으면 그 조건이 정보를 담고 있다는 뜻이다."""
    # 10봉 오르고 20봉 내리는 경로 — 일찍 파는 쪽이 유리해야 한다
    close = np.r_[np.full(30, 100.0), np.linspace(100, 130, 10),
                  np.linspace(130, 80, 25)]
    frame = _frame(close)
    feat = pd.DataFrame({"ema_20": frame["close"].rolling(20, min_periods=1).mean()},
                        index=frame.index)
    rules = {"오르막 끝": lambda c, f: c["close"].diff() < 0}
    table = exit_timing({"A": _signal(frame, [29])}, {"A": frame}, {"A": feat},
                        rules=rules, horizon=30, cost_bps=0.0)
    assert not table.empty
    assert table.iloc[0]["만기보유 대비"] > 0


def test_exit_timing_reports_the_hold_baseline():
    frame = _frame(np.full(60, 100.0))
    feat = pd.DataFrame({"ema_20": 100.0}, index=frame.index)
    table = exit_timing({"A": _signal(frame, [10])}, {"A": frame}, {"A": feat},
                        rules={"never": lambda c, f: pd.Series(False, index=c.index)},
                        horizon=20, cost_bps=0.0)
    assert table.empty                       # 켜진 조건이 없다
    assert list(table.columns)                # 그래도 컬럼은 있어야 한다
    assert table.attrs["n_events"] == 1
    assert table.attrs["held_median"] == pytest.approx(0.0, abs=1e-9)


def test_default_exit_rules_are_all_callable():
    rules = default_exit_rules()
    assert len(rules) >= 6
    n = 80
    frame = _frame(np.linspace(100, 130, n))
    feat = pd.DataFrame({
        "macd_hist": np.zeros(n), "stoch_k": np.full(n, 50.0),
        "stoch_d": np.full(n, 50.0), "ema_5": frame["close"] * 0.99,
        "ema_20": frame["close"] * 0.98, "kijun": frame["close"] * 0.97,
        "senkou_a": frame["close"] * 0.95, "senkou_b": frame["close"] * 0.94,
        "williams_r_14": np.full(n, -50.0)}, index=frame.index)
    for name, fn in rules.items():
        out = fn(frame, feat)
        assert len(out) == n, name
        assert out.dtype == bool or out.isin([True, False]).all(), name


def test_turn_signature_compares_signal_day_to_pre_peak():
    n = 60
    close = np.r_[np.full(20, 100.0), np.linspace(100, 130, 20), np.linspace(130, 100, 20)]
    frame = _frame(close)
    feat = pd.DataFrame({"rsi_14": np.r_[np.full(20, 30.0), np.linspace(30, 75, 20),
                                         np.linspace(75, 40, 20)]}, index=frame.index)
    out = turn_signature({"A": _signal(frame, [19])}, {"A": frame}, {"A": feat},
                         horizon=40, columns=("rsi_14",))
    assert len(out) == 1
    assert out.iloc[0]["rsi_14_신호일"] < out.iloc[0]["rsi_14_고점직전"]
    assert out.iloc[0]["days_to_peak"] > 0
