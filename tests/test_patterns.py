"""차트 형태 패턴 테스트.

지표와 달리 '모양'을 보므로, 합성 캔들로 모양을 직접 그려 판정을 확인한다.
"""

import numpy as np
import pandas as pd
import pytest

from tsignal.datasource import Interval, SyntheticDataSource
from tsignal.evaluation.eventstudy import calendar_time_portfolio
from tsignal.signals.patterns import CupHandleParams, cup_with_handle


def _candles(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.bdate_range("2020-01-01", periods=n, tz="Asia/Seoul")
    close = np.asarray(closes, dtype=float)
    volume = np.asarray(volumes if volumes is not None else [1000.0] * n, dtype=float)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": volume,
    }, index=idx)


def _cup_shape(*, prior_gain: float = 0.30, depth: float = 0.25, cup_len: int = 60,
               handle_len: int = 8, handle_depth: float = 0.05,
               v_shape: bool = False) -> tuple[list[float], list[float]]:
    """선행 상승 → 컵 → 핸들 → 돌파 형태의 종가/거래량을 만든다."""
    prior = list(np.linspace(100 / (1 + prior_gain), 100, 70))
    rim = prior[-1]
    trough = rim * (1 - depth)

    if v_shape:                                   # 저점이 앞쪽에 몰린 V자
        down = list(np.linspace(rim, trough, max(3, cup_len // 12)))
        up = list(np.linspace(trough, rim, cup_len - len(down)))
    else:                                         # 저점이 가운데인 U자
        half = cup_len // 2
        down = list(np.linspace(rim, trough, half))
        up = list(np.linspace(trough, rim, cup_len - half))
    cup = down + up

    handle_low = rim * (1 - handle_depth)
    handle = list(np.linspace(rim, handle_low, handle_len // 2)) + \
             list(np.linspace(handle_low, rim * 0.995, handle_len - handle_len // 2))
    breakout = [rim * 1.05]

    closes = prior + cup + handle + breakout
    volumes = ([1000.0] * len(prior) + [1000.0] * len(cup)
               + [400.0] * len(handle) + [3000.0])
    return closes, volumes


def test_detects_a_textbook_cup_with_handle():
    closes, volumes = _cup_shape()
    hits = cup_with_handle(_candles(closes, volumes))
    assert hits.iloc[-1], "교과서 형태를 못 잡았다"
    assert hits.sum() == 1                        # 돌파 봉에서만 True


def test_rejects_v_shaped_bottom():
    """저점에서 곧장 튄 V자 반등은 컵이 아니다."""
    closes, volumes = _cup_shape(v_shape=True)
    assert not cup_with_handle(_candles(closes, volumes)).any()


def test_rejects_without_prior_uptrend():
    """컵앤핸들은 '오르던 종목의 조정'이다. 선행 상승이 없으면 다른 사건이다.

    (실측에서도 이 조건을 빼면 연환산 초과수익이 14.2% → 0.1% 로 사라졌다.)
    """
    closes, volumes = _cup_shape(prior_gain=0.0)
    assert not cup_with_handle(_candles(closes, volumes)).any()


def test_rejects_shallow_cup():
    closes, volumes = _cup_shape(depth=0.03)
    assert not cup_with_handle(_candles(closes, volumes)).any()


def test_rejects_deep_handle():
    """핸들이 깊으면 조정이지 핸들이 아니다."""
    closes, volumes = _cup_shape(handle_depth=0.30)
    assert not cup_with_handle(_candles(closes, volumes)).any()


def test_rejects_breakout_without_volume():
    closes, volumes = _cup_shape()
    volumes[-1] = 900.0                           # 돌파봉 거래량이 평균 이하
    assert not cup_with_handle(_candles(closes, volumes)).any()


def test_is_causal():
    """t 봉 판정은 t 이후 데이터에 영향을 받지 않아야 한다."""
    source = SyntheticDataSource(seed=555)
    candles = source.candles("AAA", Interval.D1, count=900)
    full = cup_with_handle(candles)
    for cut in (600, 750):
        truncated = cup_with_handle(candles.iloc[:cut])
        assert full.iloc[:cut].equals(truncated), f"cut={cut} 에서 판정이 달라졌다"


def test_no_edge_on_random_walk():
    """엣지가 0인 랜덤워크에서는 이 패턴도 성과를 내면 안 된다."""
    source = SyntheticDataSource(seed=8080, drift=0.0, annual_vol=0.30)
    data = {f"S{i:02d}": source.candles(f"S{i:02d}", Interval.D1, count=1000) for i in range(15)}
    events = {code: cup_with_handle(candles) for code, candles in data.items()}
    if sum(int(s.sum()) for s in events.values()) < 20:
        pytest.skip("합성 데이터에서 패턴 표본이 부족하다")
    result = calendar_time_portfolio(events, data, holding_days=60)
    assert abs(result.t_stat) < 3.0, result.summary()


def test_benchmark_uses_the_whole_universe():
    """벤치마크는 이벤트가 있는 종목이 아니라 유니버스 전체여야 한다.

    이벤트 종목만으로 평균을 내면 벤치마크가 표본과 함께 움직여 초과수익이
    왜곡된다 (전체기간과 부분기간 결과가 모순되는 형태로 드러났던 버그).
    """
    source = SyntheticDataSource(seed=99)
    data = {f"S{i:02d}": source.candles(f"S{i:02d}", Interval.D1, count=400) for i in range(6)}
    index = data["S00"].index
    events = {"S00": pd.Series(False, index=index)}
    events["S00"].iloc[100] = True

    result = calendar_time_portfolio(events, data, holding_days=20)
    assert result.n_events == 1
    # 한 종목만 이벤트가 있어도 나머지 5종목이 벤치마크에 들어가므로
    # 초과수익이 항상 0 이 되지 않는다.
    assert not np.allclose(result.daily.to_numpy(), 0.0)
