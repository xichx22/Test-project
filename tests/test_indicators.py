import numpy as np
import pandas as pd
import pytest

from tsignal import indicators as ind
from tsignal.datasource import Interval
from tsignal.indicators import _util, momentum, trend, volatility, volume


def test_ema_matches_manual_recursion():
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
    got = _util.ema(s, 3)
    alpha = 2 / 4
    # pandas ewm(adjust=False) 는 첫 값을 시드로 재귀한다 — 직접 재귀로 검증
    seed = s.iloc[0]
    for v in s.iloc[1:]:
        seed = alpha * v + (1 - alpha) * seed
    assert got.iloc[-1] == pytest.approx(seed)
    assert got.iloc[:2].isna().all()


def test_rsi_bounds_and_extremes():
    up = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                       "close": np.arange(1, 60, dtype=float), "volume": 1.0},
                      index=pd.date_range("2024-01-01", periods=59, freq="D", tz="Asia/Seoul"))
    r = momentum.rsi(up, 14).dropna()
    assert np.allclose(r, 100.0)               # 하락이 전혀 없으면 RSI=100


def test_williams_r_range(candles_5m):
    wr = momentum.williams_r(candles_5m, 14).dropna()
    assert wr.between(-100, 0).all()


def test_envelope_band_is_fixed_percentage(candles_5m):
    env = volatility.envelope(candles_5m, n=20, pct=2.0).dropna()
    assert (env["env_upper"] / env["env_mid"] - 1).round(10).eq(0.02).all()
    assert (1 - env["env_lower"] / env["env_mid"]).round(10).eq(0.02).all()


def test_dema_leads_ema_on_a_ramp():
    """DEMA 는 EMA 보다 지연이 작다 — 상승 램프에서 항상 EMA 위에 있어야 한다."""
    s = pd.Series(np.arange(200, dtype=float))
    assert (_util.dema(s, 20).dropna() > _util.ema(s, 20).reindex(_util.dema(s, 20).dropna().index)).all()


def test_vwap_resets_each_session(candles_5m):
    v = volume.vwap(candles_5m)
    first_of_day = v.index.to_series().groupby(candles_5m.index.date).idxmin()
    tp = (candles_5m["high"] + candles_5m["low"] + candles_5m["close"]) / 3
    # 세션 첫 봉의 VWAP 은 그 봉의 typical price 와 같아야 한다.
    assert np.allclose(v.loc[first_of_day, "vwap"], tp.loc[first_of_day])


def test_supertrend_direction_is_binary(candles_5m):
    d = trend.supertrend(candles_5m)["supertrend_dir"].dropna()
    assert set(d.unique()) <= {-1.0, 1.0}


def test_compute_all_covers_registry(candles_5m):
    features = ind.compute_all(candles_5m, interval=Interval.M5)
    assert features.shape[1] > len(ind.REGISTRY)      # 다중 컬럼 지표가 있으므로
    assert features.index.equals(candles_5m.index)
    assert not features.columns.duplicated().any()
    assert features.iloc[-1].notna().all()            # 마지막 봉은 전부 계산돼야 실전에 쓴다
