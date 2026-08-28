"""횡단면 팩터 분석 테스트.

핵심은 두 개다: 전방수익률 겹침과 베타. 둘 다 보정하지 않으면 없는 알파가 보인다.
"""

import numpy as np
import pandas as pd
import pytest

from tsignal.datasource import Interval, SyntheticDataSource
from tsignal.evaluation.factor import (
    build_factor_panel, dose_response, double_sort, factor_correlations, market_regression,
)
from tsignal.evaluation.metrics import non_overlapping_t_stat, t_stat


@pytest.fixture(scope="module")
def panel():
    source = SyntheticDataSource(seed=2468, drift=0.2, annual_vol=0.30)
    data = {f"S{i:02d}": source.candles(f"S{i:02d}", Interval.D1, count=900) for i in range(20)}
    return build_factor_panel(data, horizon=20)


def test_overlapping_returns_inflate_t_by_about_sqrt_horizon():
    """겹치는 관측을 독립으로 세면 t 가 √h 배 부풀려진다."""
    rng = np.random.default_rng(0)
    daily = rng.normal(0.0004, 0.01, 4000)
    horizon = 20
    # 20일 누적수익률을 매일 계산 → 인접 관측이 19/20 을 공유한다
    overlapping = np.convolve(daily, np.ones(horizon), mode="valid")

    naive = abs(t_stat(pd.Series(overlapping)))
    corrected = abs(non_overlapping_t_stat(overlapping, horizon))
    assert naive > corrected
    # 이론적 축소배율 √20 ≈ 4.5 근방이어야 한다.
    assert 2.0 < naive / max(corrected, 1e-9) < 9.0


def test_non_overlapping_matches_naive_when_horizon_is_one():
    rng = np.random.default_rng(1)
    values = rng.normal(0.001, 0.02, 600)
    assert non_overlapping_t_stat(values, 1) == pytest.approx(abs(t_stat(pd.Series(values))), rel=1e-9)


def test_non_overlapping_needs_enough_data():
    assert np.isnan(non_overlapping_t_stat(np.arange(10.0), 20))


def test_non_overlapping_rejects_bad_horizon():
    with pytest.raises(ValueError, match="horizon"):
        non_overlapping_t_stat(np.arange(100.0), 0)


def test_excess_is_cross_sectionally_demeaned(panel):
    per_day = panel.frame.groupby("day")["excess"].mean()
    assert np.allclose(per_day.to_numpy(), 0.0, atol=1e-12)


def test_dose_response_reports_both_t_variants(panel):
    table = dose_response(panel, "ema60_gap", n_buckets=5)
    assert {"t_overlap", "t_edge"} <= set(table.columns)
    assert len(table) == 5
    assert {"spread_t", "spread_t_overlap", "monotone_rho"} <= set(table.attrs)
    # 겹침 t 가 비겹침 t 보다 크게 나오는 것이 정상이다.
    assert abs(table.attrs["spread_t_overlap"]) >= abs(table.attrs["spread_t"])


def test_no_factor_survives_on_noise(panel):
    """드리프트만 있는 합성 시장에서는 어떤 팩터도 알파를 내면 안 된다."""
    for factor in ("ema60_gap", "ret_20", "rsi_14", "atrp_14"):
        reg = market_regression(panel, factor, n_buckets=5)
        assert abs(reg["alpha_t_nonoverlap"]) < 3.0, (factor, reg)


def test_market_regression_separates_beta(panel):
    reg = market_regression(panel, "atrp_14", n_buckets=5)
    assert {"spread", "alpha", "beta", "market_r2", "alpha_t_nonoverlap"} <= set(reg)
    assert 0.0 <= reg["market_r2"] <= 1.0


def test_buckets_are_formed_within_each_day(panel):
    """분위는 날짜마다 새로 매겨야 한다 — 전체 표본에 한 번 매기면 시점 효과와 섞인다."""
    from tsignal.evaluation.factor import _cross_sectional_bucket

    frame = panel.frame.copy()
    frame["bucket"] = _cross_sectional_bucket(frame, "ema60_gap", 5)
    counts = frame.dropna(subset=["bucket"]).groupby("day")["bucket"].nunique()
    assert (counts == 5).mean() > 0.95        # 거의 모든 날짜에 5개 분위가 다 있어야 한다


def test_double_sort_returns_a_grid(panel):
    table = double_sort(panel, "ema60_gap", "ret_120", n_buckets=4)
    assert table.shape == (4, 4)
    assert "spreads" in table.attrs


def test_correlations_are_symmetric_with_unit_diagonal(panel):
    corr = factor_correlations(panel, ["ema60_gap", "ret_20", "rsi_14"])
    assert np.allclose(np.diag(corr.to_numpy(dtype=float)), 1.0)
    assert np.allclose(corr.to_numpy(dtype=float), corr.to_numpy(dtype=float).T, atol=1e-9)
