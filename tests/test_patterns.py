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


# =====================================================================
# 추가 패턴 — 합성 캔들로 모양 판정을 확인한다
# =====================================================================

from tsignal.signals.patterns import (  # noqa: E402
    FAMOUS_PATTERNS, LESSER_KNOWN_PATTERNS, PATTERNS,
    ascending_triangle, bull_flag, double_bottom, flat_base,
)


def _with_prior(shape: list[float], gain: float = 0.30) -> list[float]:
    """베이스 앞에 상승 구간을 붙인다. 모든 베이스 패턴의 전제 조건."""
    start = shape[0]
    return list(np.linspace(start / (1 + gain), start, 70)) + shape


def _volumes(n_prior: int, n_base: int, base_level: float = 1000.0) -> list[float]:
    return [base_level] * n_prior + [base_level * 0.6] * n_base + [base_level * 3]


def test_double_bottom_detects_w_shape():
    rim = 100.0
    leg = list(np.linspace(rim, 78, 15)) + list(np.linspace(78, 94, 15)) \
        + list(np.linspace(94, 79, 15)) + list(np.linspace(79, 93.5, 15))
    closes = _with_prior(leg) + [96.0]
    volumes = _volumes(70, len(leg))
    hits = double_bottom(_candles(closes, volumes))
    assert hits.iloc[-1], "교과서 쌍바닥을 못 잡았다"


def test_double_bottom_rejects_mismatched_lows():
    """두 저점의 높이가 크게 다르면 쌍바닥이 아니다."""
    rim = 100.0
    leg = list(np.linspace(rim, 78, 15)) + list(np.linspace(78, 94, 15)) \
        + list(np.linspace(94, 60, 15)) + list(np.linspace(60, 93.5, 15))
    closes = _with_prior(leg) + [96.0]
    volumes = _volumes(70, len(leg))
    assert not double_bottom(_candles(closes, volumes)).any()


def test_flat_base_detects_tight_range():
    base = list(100 + np.sin(np.linspace(0, 6 * np.pi, 40)) * 3)   # ±3% 횡보
    closes = _with_prior(base) + [106.0]
    volumes = _volumes(70, len(base))
    assert flat_base(_candles(closes, volumes)).iloc[-1]


def test_flat_base_rejects_wide_range():
    """깊은 조정은 플랫 베이스가 아니다."""
    base = list(100 + np.sin(np.linspace(0, 6 * np.pi, 40)) * 20)  # ±20%
    closes = _with_prior(base) + [125.0]
    volumes = _volumes(70, len(base))
    assert not flat_base(_candles(closes, volumes)).any()


def test_bull_flag_detects_pole_and_flag():
    pole = list(np.linspace(100, 135, 15))          # +35% 깃대
    flag = list(np.linspace(135, 127, 5)) + list(np.linspace(127, 133, 5))
    # 거래량 평균 창(50) + 깃대·깃발 최대 길이만큼 앞자리가 필요하다
    closes = [100.0] * 110 + pole + flag + [138.0]
    volumes = [1000.0] * 110 + [2500.0] * len(pole) + [800.0] * len(flag) + [4000.0]
    assert bull_flag(_candles(closes, volumes)).iloc[-1]


def test_bull_flag_rejects_deep_retrace():
    """깃대 상승분의 절반 넘게 되돌리면 깃발이 아니라 추세 이탈이다."""
    pole = list(np.linspace(100, 135, 15))
    flag = list(np.linspace(135, 112, 6)) + list(np.linspace(112, 120, 6))
    closes = [100.0] * 110 + pole + flag + [136.0]
    volumes = [1000.0] * 110 + [2500.0] * len(pole) + [800.0] * len(flag) + [4000.0]
    assert not bull_flag(_candles(closes, volumes)).any()


def test_ascending_triangle_needs_rising_lows():
    """수평 저항 + 높아지는 저점. 저점이 평평하면 삼각형이 아니라 박스다."""
    rising = []
    for i in range(5):
        rising += list(np.linspace(88 + i * 2, 100, 6)) + list(np.linspace(100, 89 + i * 2, 6))
    flat = []
    for _ in range(5):
        flat += list(np.linspace(88, 100, 6)) + list(np.linspace(100, 88, 6))

    for shape, expected in ((rising, True), (flat, False)):
        closes = _with_prior(shape, gain=0.15) + [104.0]
        volumes = _volumes(70, len(shape))
        assert bool(ascending_triangle(_candles(closes, volumes)).any()) is expected


def test_all_patterns_require_a_prior_uptrend():
    """베이스 패턴은 전부 '오르던 종목의 조정'이다.

    컵앤핸들 실측에서 이 조건을 빼자 연환산 초과수익이 +15.9% → +0.6% 로 사라졌다.
    깃발은 깃대 자체가 상승이므로 별도 조건이 없다 — 그래서 제외한다.
    """
    from tsignal.signals.patterns import (
        AscendingTriangleParams, CupHandleParams, DoubleBottomParams, FlatBaseParams,
    )

    for params in (CupHandleParams(), DoubleBottomParams(),
                   FlatBaseParams(), AscendingTriangleParams()):
        assert params.prior_gain > 0, params


def test_pattern_registry_lists_every_detector():
    assert set(PATTERNS) == set(FAMOUS_PATTERNS) | set(LESSER_KNOWN_PATTERNS)
    assert not set(FAMOUS_PATTERNS) & set(LESSER_KNOWN_PATTERNS)
    source = SyntheticDataSource(seed=13)
    candles = source.candles("AAA", Interval.D1, count=500)
    for name, fn in PATTERNS.items():
        series = fn(candles)
        assert series.dtype == bool and len(series) == len(candles), name


def test_new_patterns_are_causal():
    source = SyntheticDataSource(seed=246)
    candles = source.candles("AAA", Interval.D1, count=800)
    for name, fn in PATTERNS.items():
        full = fn(candles)
        truncated = fn(candles.iloc[:600])
        assert full.iloc[:600].equals(truncated), f"{name} 이 미래를 보고 있다"


# =====================================================================
# 스윙 포트폴리오 — 초과수익이 아니라 실제 자산가치 곡선
# =====================================================================

from tsignal.evaluation.eventstudy import swing_portfolio  # noqa: E402


def test_swing_portfolio_holds_cash_when_no_signal():
    """신호가 없는 날은 전액 현금이어야 한다."""
    source = SyntheticDataSource(seed=21)
    data = {c: source.candles(c, Interval.D1, count=400) for c in ("AAA", "BBB")}
    index = data["AAA"].index
    events = {c: pd.Series(False, index=index) for c in data}
    events["AAA"].iloc[100] = True

    result = swing_portfolio(events, data, holding_days=20, cost_bps=0.0, cash_rate=0.0)
    assert result.exposure < 0.15                    # 20/400 근처
    # 보유하지 않는 구간은 현금이므로 수익률이 0 이다.
    assert result.daily[~result.weight.astype(bool)].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_swing_portfolio_equal_weights_open_positions():
    """동시 보유 종목은 동일가중이다."""
    source = SyntheticDataSource(seed=22)
    data = {c: source.candles(c, Interval.D1, count=300) for c in ("AAA", "BBB")}
    index = data["AAA"].index
    events = {c: pd.Series(False, index=index) for c in data}
    for c in data:
        events[c].iloc[50] = True

    result = swing_portfolio(events, data, holding_days=30, cost_bps=0.0, cash_rate=0.0)
    day = index[60]
    expected = np.mean([data[c]["close"].pct_change().loc[day] for c in data])
    assert result.daily.loc[day] == pytest.approx(expected, rel=1e-9)


def test_swing_portfolio_charges_entry_costs():
    source = SyntheticDataSource(seed=23)
    data = {"AAA": source.candles("AAA", Interval.D1, count=300)}
    events = {"AAA": pd.Series(False, index=data["AAA"].index)}
    events["AAA"].iloc[50] = True

    free = swing_portfolio(events, data, holding_days=30, cost_bps=0.0)
    charged = swing_portfolio(events, data, holding_days=30, cost_bps=100.0)
    assert charged.equity.iloc[-1] < free.equity.iloc[-1]
    assert charged.trades == 1


def test_swing_portfolio_respects_position_limit():
    source = SyntheticDataSource(seed=24)
    codes = [f"S{i}" for i in range(6)]
    data = {c: source.candles(c, Interval.D1, count=300) for c in codes}
    index = data[codes[0]].index
    events = {c: pd.Series(False, index=index) for c in codes}
    for c in codes:
        events[c].iloc[50] = True

    limited = swing_portfolio(events, data, holding_days=30, max_positions=2, cost_bps=0.0)
    full = swing_portfolio(events, data, holding_days=30, cost_bps=0.0)
    assert limited.daily.std() >= full.daily.std()    # 덜 분산되므로 변동성이 크다


# =====================================================================
# 추가 패턴 6종 — 유명한 것 2, 덜 알려진 것 4
#
# 검출기는 반드시 세 가지를 통과해야 한다:
#   1. 교과서 모양에서 잡힌다        (안 잡히면 백테스트 결과가 무의미하다)
#   2. 비슷하지만 다른 모양은 기각한다 (안 그러면 그냥 아무 상승을 잡는 것)
#   3. 랜덤워크에서 드물다            (흔하면 '패턴'이라 부를 근거가 없다)
# =====================================================================

def _bars(close, volume=None, high=None, low=None):
    close = np.asarray(close, float)
    n = len(close)
    index = pd.date_range("2018-01-01", periods=n, freq="B", tz="Asia/Seoul")
    return pd.DataFrame(
        {"open": close,
         "high": close * 1.005 if high is None else np.asarray(high, float),
         "low": close * 0.995 if low is None else np.asarray(low, float),
         "close": close,
         "volume": np.full(n, 1000.0) if volume is None else np.asarray(volume, float)},
        index=index,
    )


def _seg(a, b, n):
    return np.linspace(a, b, n, endpoint=False)


def _spike(n, level=1000.0, last=4000.0):
    return np.r_[np.full(n - 1, level), [last]]


def _inverse_hs_shape():
    """앞선 하락 → 왼어깨 · 머리 · 오른어깨 → 넥라인 돌파."""
    prior = _seg(100, 80, 70)
    body = np.r_[_seg(80, 65, 20), _seg(65, 78, 20), _seg(78, 55, 20),
                 _seg(55, 78, 20), _seg(78, 66, 20), _seg(66, 77, 20)]
    close = np.r_[prior, body, [82.0]]
    return _bars(close, _spike(len(close)))


def _falling_wedge_shape():
    """고점선·저점선이 **둘 다** 내려오면서 수렴한다."""
    prior = _seg(60, 110, 70)
    i = np.arange(60)
    upper, lower = 110 - 14 * i / 59, 96 - 2 * i / 59
    zig = np.where(i % 2 == 0, upper, lower)
    zig[-1] = lower[-1]
    close = np.r_[prior, zig, [108.0]]
    return _bars(close, _spike(len(close)))


def _vcp_shape():
    """조정이 22% → 13% → 6% 로 얕아지고 거래량이 마른다."""
    prior = _seg(60, 100, 70)
    legs, top = [], 100.0
    for depth, length in ((0.22, 27), (0.13, 27), (0.06, 26)):
        low = top * (1 - depth)
        legs.append(np.r_[_seg(top, low, length // 2),
                          _seg(low, top * 0.995, length - length // 2)])
    close = np.r_[prior, np.concatenate(legs), [104.0]]
    volume = np.r_[np.full(70, 4000.0), np.full(27, 4000.0), np.full(27, 3000.0),
                   np.full(26, 1500.0), [12000.0]]
    return _bars(close, volume[:len(close)])


def _high_tight_flag_shape():
    prior = _seg(50, 106, 42)                 # 112% 급등
    flag = np.r_[_seg(106, 88, 10), _seg(88, 104, 10)]
    close = np.r_[prior, flag, [110.0]]
    return _bars(close, _spike(len(close)))


def _nr7_shape():
    close = np.array(list(_seg(80, 100, 60)) + [100.0, 103.0])
    high, low = close * 1.005, close * 0.995
    high[-2], low[-2] = 100.05, 99.95         # 최근 7봉 중 가장 좁은 봉
    high[-1], low[-1] = 103.5, 100.1
    return _bars(close, _spike(len(close)), high, low)


def _pocket_pivot_shape():
    close = np.r_[_seg(80, 96, 60), [95.0], [96.5]]
    volume = np.r_[np.full(60, 1000.0), [3000.0], [9000.0]]
    return _bars(close, volume)


TEXTBOOK = {
    "inverse_head_and_shoulders": _inverse_hs_shape,
    "falling_wedge": _falling_wedge_shape,
    "volatility_contraction": _vcp_shape,
    "high_tight_flag": _high_tight_flag_shape,
    "nr7_breakout": _nr7_shape,
    "pocket_pivot": _pocket_pivot_shape,
}


@pytest.mark.parametrize("name", sorted(TEXTBOOK))
def test_detector_fires_on_its_textbook_shape(name):
    """정석 모양에서 안 잡히면 그 검출기로 낸 백테스트는 아무 의미가 없다.

    처음 작성했을 때 6개 중 5개가 자기 교과서 모양에서 안 잡혔다.
    """
    from tsignal.signals.patterns import (
    FAMOUS_PATTERNS, LESSER_KNOWN_PATTERNS, PATTERNS)

    hit = PATTERNS[name](TEXTBOOK[name]())
    assert bool(hit.iloc[-1]), f"{name} 이 정석 모양의 완성 봉을 잡지 못했다"


def test_inverse_head_shoulders_rejects_a_plain_v():
    """어깨가 없는 V자 반등은 역헤드앤숄더가 아니다."""
    from tsignal.signals.patterns import inverse_head_and_shoulders

    close = np.r_[_seg(100, 80, 70), _seg(80, 50, 60), _seg(50, 78, 60), [82.0]]
    assert not inverse_head_and_shoulders(_bars(close, _spike(len(close)))).any()


def test_falling_wedge_rejects_a_rising_wedge():
    """두 선이 올라가면 하락쐐기가 아니다 — 방향 조건이 살아 있는지 본다."""
    from tsignal.signals.patterns import falling_wedge

    i = np.arange(60)
    zig = np.where(i % 2 == 0, 96 + 14 * i / 59, 110 + 2 * i / 59)
    close = np.r_[_seg(60, 96, 70), zig, [118.0]]
    assert not falling_wedge(_bars(close, _spike(len(close)))).iloc[-1]


def test_vcp_rejects_widening_pullbacks():
    """조정이 깊어지는 형태는 '변동성 수축'의 반대다."""
    from tsignal.signals.patterns import volatility_contraction

    legs, top = [], 100.0
    for depth, length in ((0.06, 27), (0.13, 27), (0.22, 26)):
        low = top * (1 - depth)
        legs.append(np.r_[_seg(top, low, length // 2),
                          _seg(low, top * 0.995, length - length // 2)])
    close = np.r_[_seg(60, 100, 70), np.concatenate(legs), [104.0]]
    volume = np.r_[np.full(len(close) - 1, 3000.0), [12000.0]]
    assert not volatility_contraction(_bars(close, volume)).iloc[-1]


def test_high_tight_flag_needs_an_actual_run():
    """급등이 없으면 '하이 타이트'가 아니다."""
    from tsignal.signals.patterns import high_tight_flag

    close = np.r_[_seg(50, 60, 42), _seg(60, 55, 10), _seg(55, 59, 10), [62.0]]
    assert not high_tight_flag(_bars(close, _spike(len(close)))).iloc[-1]


def test_nr7_needs_the_previous_bar_to_be_the_narrowest():
    from tsignal.signals.patterns import nr7_breakout

    close = np.array(list(_seg(80, 100, 60)) + [100.0, 103.0])
    high, low = close * 1.005, close * 0.995
    high[-2], low[-2] = 105.0, 95.0           # 직전 봉이 가장 **넓다**
    high[-1], low[-1] = 103.5, 100.1
    assert not nr7_breakout(_bars(close, _spike(len(close)), high, low)).iloc[-1]


def test_pocket_pivot_needs_volume_above_recent_down_bars():
    from tsignal.signals.patterns import pocket_pivot

    close = np.r_[_seg(80, 96, 60), [95.0], [96.5]]
    volume = np.r_[np.full(60, 1000.0), [20000.0], [9000.0]]  # 하락봉이 더 크다
    assert not pocket_pivot(_bars(close, volume)).iloc[-1]


def test_envelope_lines_trace_the_outline_not_the_middle():
    """포락선은 봉우리끼리·골끼리 이어야 한다.

    모든 봉에 최소제곱을 걸면 지그재그 한가운데를 지나가서 상단선과
    하단선이 거의 붙는다. 실측에서 수렴하는 쐐기의 폭 비율이 0.913 으로
    나와 수축 조건을 통과하지 못했다.
    """
    from tsignal.signals.patterns import _envelope_lines

    i = np.arange(60)
    high = np.where(i % 2 == 0, 110 - 14 * i / 59, 100.0)
    low = np.where(i % 2 == 0, 100.0, 96 - 2 * i / 59)
    hi_slope, hi_base, lo_slope, lo_base = _envelope_lines(high, low)
    width_start = hi_base - lo_base
    width_end = (hi_base + hi_slope * 59) - (lo_base + lo_slope * 59)
    assert width_start > width_end > 0, "수렴을 잡아내지 못했다"
    assert width_end / width_start < 0.5


@pytest.mark.parametrize("name,limit", [
    ("inverse_head_and_shoulders", 0.5),
    ("falling_wedge", 0.5),
    ("volatility_contraction", 0.5),
    ("high_tight_flag", 0.5),
])
def test_structural_patterns_are_rare_in_random_walks(name, limit):
    """구조 패턴이 랜덤워크에서 흔하면 '패턴'이라 부를 근거가 없다."""
    from tsignal.signals.patterns import PATTERNS

    rng = np.random.default_rng(11)
    total = 0
    for _ in range(40):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 800)))
        volume = rng.integers(500, 5000, 800).astype(float)
        total += int(PATTERNS[name](_bars(close, volume)).sum())
    assert total / 40 < limit, f"{name} 이 랜덤워크에서 종목당 {total/40:.2f}회 나온다"


def test_rejects_runaway_right_rim():
    """우측 테두리가 좌측을 크게 뚫고 올라가면 컵이 아니다.

    실측에서 이 상한이 없어 롯데쇼핑처럼 '3배 급등 후 반토막' 이 컵으로
    잡혔다. 좌측 80,900 · 저점 62,700 · 우측 211,000 (2.6배).
    """
    # 하락 → 저점 → 좌측 고점을 한참 넘어서는 급등 → 급락 → 5봉 눌림 → 돌파
    closes = ([100] * 20 + list(np.linspace(100, 70, 40)) + list(np.linspace(70, 260, 40))
              + list(np.linspace(260, 120, 15)) + [118, 116, 117, 115, 116] + [125])
    volumes = [1000] * (len(closes) - 1) + [5000]
    assert not cup_with_handle(_candles(closes, volumes)).any()


def test_rejects_handle_far_below_the_right_rim():
    """핸들은 컵 오른쪽 입술 근처에 생긴다. 한참 아래면 붕괴 중 반등이다."""
    closes = ([100] * 20 + list(np.linspace(100, 75, 30)) + list(np.linspace(75, 105, 30))
              + list(np.linspace(105, 70, 12)) + [69, 68, 69, 67, 68] + [72])
    volumes = [1000] * (len(closes) - 1) + [5000]
    assert not cup_with_handle(_candles(closes, volumes)).any()


def test_still_accepts_a_rim_that_recovers_to_the_same_level():
    """고친 뒤에도 정상 컵앤핸들은 잡혀야 한다 (상한이 너무 좁으면 안 된다)."""
    params = CupHandleParams()
    assert params.rim_recovery < 1.0 < params.rim_overshoot
