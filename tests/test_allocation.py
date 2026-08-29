"""전술적 자산배분 / 추세추종 테스트.

이 모듈은 개별 종목 신호와 목적이 다르다 — 수익률이 아니라 위험 조정 성과를
다루므로, 검사할 것도 다르다: 미래참조, 비용 반영, 그리고 비중 논리.
"""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.allocation import (
    STRATEGIES, absolute_momentum, buy_and_hold, daily_sma, monthly_sma, volatility_target,
)


def _series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2010-01-01", periods=len(values), tz="Asia/Seoul")
    return pd.Series(values, index=idx, dtype=float)


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    """추세와 급락이 모두 들어간 합성 가격."""
    rng = np.random.default_rng(4242)
    up = 100 * np.cumprod(1 + rng.normal(0.0006, 0.010, 900))
    crash = up[-1] * np.cumprod(1 + rng.normal(-0.010, 0.030, 60))
    recover = crash[-1] * np.cumprod(1 + rng.normal(0.0006, 0.012, 500))
    return _series(list(up) + list(crash) + list(recover))


def test_buy_and_hold_matches_the_asset(prices):
    result = buy_and_hold(prices, one_way_bps=0.0)
    assert result.equity.iloc[-1] == pytest.approx(prices.iloc[-1] / prices.iloc[0], rel=1e-6)
    assert result.exposure == pytest.approx(1.0)


def test_strategies_never_look_ahead(prices):
    """미래 데이터를 잘라내도 그 이전 비중이 바뀌면 안 된다."""
    for name, fn in STRATEGIES.items():
        full = fn(prices).weight
        for cut in (700, 1000):
            part = fn(prices.iloc[:cut]).weight
            assert full.loc[part.index].equals(part), f"{name} 이 미래를 보고 있다"


def test_weights_stay_within_bounds(prices):
    for name, fn in STRATEGIES.items():
        weight = fn(prices).weight
        assert weight.between(0.0, 1.0).all(), name
        assert weight.notna().all(), name


def test_costs_reduce_returns(prices):
    free = daily_sma(prices, one_way_bps=0.0)
    charged = daily_sma(prices, one_way_bps=50.0)
    assert charged.equity.iloc[-1] < free.equity.iloc[-1]
    assert charged.trades == free.trades          # 비용은 매매 횟수를 바꾸지 않는다


def test_rebalance_band_cuts_turnover(prices):
    """밴드 없이 매일 조정하면 매매가 수천 번 나온다 — 현실성이 없다."""
    daily_rebal = volatility_target(prices, band=0.0)
    banded = volatility_target(prices, band=0.10)
    assert banded.trades < daily_rebal.trades / 5


def test_trend_rules_reduce_exposure_in_a_downtrend():
    """하락 추세에서는 비중이 0 으로 내려가야 한다."""
    falling = _series(list(np.linspace(100, 40, 400)))
    for fn in (daily_sma, monthly_sma, absolute_momentum):
        weight = fn(falling).weight
        assert weight.iloc[-50:].mean() < 0.1, fn.__name__


def test_volatility_target_cuts_exposure_when_volatility_spikes():
    """방향이 아니라 변동성만 본다 — 급등락이 시작되면 즉시 줄인다."""
    rng = np.random.default_rng(7)
    calm = 100 * np.cumprod(1 + rng.normal(0.0003, 0.004, 300))
    wild = calm[-1] * np.cumprod(1 + rng.normal(0.0003, 0.040, 200))
    weight = volatility_target(_series(list(calm) + list(wild)), target=0.10).weight
    assert weight.iloc[250:295].mean() > weight.iloc[400:].mean() * 3


def test_summary_reports_risk_metrics(prices):
    summary = buy_and_hold(prices).summary()
    assert {"CAGR%", "변동성%", "샤프", "MDD%", "칼마", "노출%", "매매횟수"} <= set(summary)
    assert summary["MDD%"] <= 0
    assert summary["변동성%"] > 0


def test_max_drawdown_is_computed_on_the_equity_curve():
    """낙폭은 자산가치 곡선에서 재야 한다 (가격이 아니라)."""
    flat_then_half = _series([100.0] * 250 + [50.0] * 250)
    result = buy_and_hold(flat_then_half, cash_rate=0.0, one_way_bps=0.0)
    assert result.max_drawdown == pytest.approx(-0.5, abs=1e-6)


# =====================================================================
# 정적 자산배분 + 리밸런싱
# =====================================================================

from tsignal.evaluation.allocation import static_mix  # noqa: E402


def _two_assets(n: int = 1500, corr: float = 0.0, seed: int = 11):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0004, 0.012, n)
    b = corr * a + np.sqrt(max(0.0, 1 - corr**2)) * rng.normal(0.0002, 0.004, n)
    idx = pd.bdate_range("2010-01-01", periods=n, tz="Asia/Seoul")
    return (pd.Series(100 * np.cumprod(1 + a), index=idx),
            pd.Series(100 * np.cumprod(1 + b), index=idx))


def test_mixing_uncorrelated_assets_lowers_volatility_more_than_return():
    """분산 효과: 상관이 낮은 자산을 섞으면 변동성이 수익보다 더 많이 준다.

    이건 예측이 아니라 수학적 성질이다 — 이 프로젝트에서 유일하게
    모든 구간(6/6)에서 일관되게 통과한 결과의 근거이기도 하다.
    """
    risky, safe = _two_assets(corr=0.0)
    stock_only = buy_and_hold(risky, one_way_bps=0.0)
    mixed = static_mix({"a": risky, "b": safe}, {"a": 0.6, "b": 0.4},
                       rebalance="QE", one_way_bps=0.0)
    assert mixed.volatility < stock_only.volatility
    assert mixed.sharpe > stock_only.sharpe


def test_diversification_benefit_shrinks_as_correlation_rises():
    """상관이 1에 가까워지면 분산 효과가 사라진다 — 이 전략의 유일한 전제."""
    gains = []
    for corr in (0.0, 0.5, 0.95):
        risky, safe = _two_assets(corr=corr)
        stock_only = buy_and_hold(risky, one_way_bps=0.0)
        mixed = static_mix({"a": risky, "b": safe}, {"a": 0.6, "b": 0.4},
                           rebalance="QE", one_way_bps=0.0)
        gains.append(mixed.sharpe - stock_only.sharpe)
    assert gains[0] > gains[2], gains


def test_static_mix_weights_sum_to_one():
    risky, safe = _two_assets()
    mixed = static_mix({"a": risky, "b": safe}, {"a": 3.0, "b": 2.0},   # 정규화되어야 한다
                       rebalance="QE", one_way_bps=0.0)
    other = static_mix({"a": risky, "b": safe}, {"a": 0.6, "b": 0.4},
                       rebalance="QE", one_way_bps=0.0)
    assert mixed.equity.iloc[-1] == pytest.approx(other.equity.iloc[-1], rel=1e-9)


def test_rebalancing_costs_are_charged():
    risky, safe = _two_assets()
    free = static_mix({"a": risky, "b": safe}, {"a": 0.6, "b": 0.4},
                      rebalance="QE", one_way_bps=0.0)
    charged = static_mix({"a": risky, "b": safe}, {"a": 0.6, "b": 0.4},
                         rebalance="QE", one_way_bps=100.0)
    assert charged.equity.iloc[-1] < free.equity.iloc[-1]
    assert charged.trades > 0


def test_no_rebalance_lets_weights_drift():
    risky, safe = _two_assets()
    drifting = static_mix({"a": risky, "b": safe}, {"a": 0.6, "b": 0.4},
                          rebalance=None, one_way_bps=0.0)
    assert drifting.trades == 0


# =====================================================================
# 꾸준함 척도 — 샤프가 답하지 못하는 질문
# =====================================================================

from tsignal.evaluation.allocation import risk_parity  # noqa: E402


def test_rolling_positive_measures_consistency():
    """단조 상승이면 롤링 수익이 항상 양수여야 한다."""
    idx = pd.bdate_range("2010-01-01", periods=1200, tz="Asia/Seoul")
    rising = pd.Series(100 * 1.0004 ** np.arange(1200), index=idx)
    result = buy_and_hold(rising, one_way_bps=0.0)
    assert result.rolling_positive(12) == pytest.approx(1.0)
    assert result.worst_rolling(12) > 0


def test_longest_underwater_counts_days_not_depth():
    """낙폭의 '깊이'가 아니라 '길이'를 센다.

    3년간 원금 회복을 못 하는 것이 한 달 만에 -30% 를 겪고 회복하는 것보다
    견디기 어렵다 — MDD 는 그 차이를 못 잡는다.
    """
    idx = pd.bdate_range("2010-01-01", periods=400, tz="Asia/Seoul")
    values = np.concatenate([np.full(50, 100.0), np.full(300, 90.0), np.full(50, 110.0)])
    result = buy_and_hold(pd.Series(values, index=idx), cash_rate=0.0, one_way_bps=0.0)
    assert 295 <= result.longest_underwater_days <= 305


def test_ulcer_index_penalises_long_shallow_drawdowns():
    """같은 MDD 라도 오래 잠겨 있으면 궤양지수가 커진다."""
    idx = pd.bdate_range("2010-01-01", periods=600, tz="Asia/Seoul")
    quick = np.concatenate([np.full(50, 100.0), np.full(10, 90.0), np.full(540, 100.0)])
    slow = np.concatenate([np.full(50, 100.0), np.full(500, 90.0), np.full(50, 100.0)])
    a = buy_and_hold(pd.Series(quick, index=idx), cash_rate=0.0, one_way_bps=0.0)
    b = buy_and_hold(pd.Series(slow, index=idx), cash_rate=0.0, one_way_bps=0.0)
    assert a.max_drawdown == pytest.approx(b.max_drawdown, abs=1e-9)   # 깊이는 같다
    assert b.ulcer_index > a.ulcer_index * 3                            # 길이가 다르다


def test_sortino_ignores_upside_volatility():
    """상승 변동성에는 벌점을 주지 않는다 — 샤프와 갈리는 지점."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-01", periods=1500, tz="Asia/Seoul")
    base = rng.normal(0.0005, 0.008, 1500)
    spiky = base.copy()
    spiky[::50] += 0.05                                   # 상승만 뾰족하게
    a = buy_and_hold(pd.Series(100 * np.cumprod(1 + base), index=idx), one_way_bps=0.0)
    b = buy_and_hold(pd.Series(100 * np.cumprod(1 + spiky), index=idx), one_way_bps=0.0)
    assert b.sortino > a.sortino                          # 소르티노는 보상한다
    assert b.sortino - a.sortino > b.sharpe - a.sharpe     # 샤프보다 더 후하게


def test_risk_parity_gives_low_volatility_assets_more_weight():
    """변동성 역가중이므로 조용한 자산이 큰 비중을 받는다."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2010-01-01", periods=900, tz="Asia/Seoul")
    loud = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.020, 900)), index=idx)
    quiet = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.002, 900)), index=idx)

    result = risk_parity({"loud": loud, "quiet": quiet}, window=120, one_way_bps=0.0)
    assert result.weight.mean() < 0.3                     # loud 비중이 작아야 한다
    assert result.volatility < buy_and_hold(loud, one_way_bps=0.0).volatility


def test_consistency_summary_has_the_right_fields():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2010-01-01", periods=1600, tz="Asia/Seoul")
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.010, 1600)), index=idx)
    summary = buy_and_hold(prices).consistency()
    assert {"12개월 양수율", "36개월 양수율", "최악의12개월%",
            "최장무회복일", "궤양지수", "소르티노"} <= set(summary)
    assert 0 <= summary["12개월 양수율"] <= 1
