"""종목을 고르지 않는 매매 규칙 + 복리·적립식 검증."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.allocation import static_mix
from tsignal.evaluation.compounding import (
    band_rebalance,
    compare_rules,
    dca,
    momentum_rotation,
    trend_filtered_mix,
)


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n, freq="B", tz="Asia/Seoul")


def _walk(seed: int, n: int = 1500, drift: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))),
                     index=_days(n))


def _ramp(n: int = 1500, rate: float = 0.0004) -> pd.Series:
    return pd.Series(100 * np.exp(np.arange(n) * rate), index=_days(n))


def test_momentum_rotation_holds_only_the_winners():
    """한 자산만 오르면 그 자산만 들고 있어야 한다."""
    n = 1500
    winner = _ramp(n, 0.0008)
    loser = pd.Series(100 * np.exp(-np.arange(n) * 0.0004), index=_days(n))
    result = momentum_rotation({"w": winner, "l": loser}, top_n=1, lookback=252)
    both = static_mix({"w": winner, "l": loser}, {"w": 1.0, "l": 1.0})
    assert result.cagr > both.cagr


def test_absolute_momentum_goes_to_cash_when_everything_falls():
    """전부 하락 중이면 아무것도 사지 않고 현금이어야 한다."""
    n = 1500
    falling = {c: pd.Series(100 * np.exp(-np.arange(n) * 0.0006), index=_days(n))
               for c in ("a", "b", "c")}
    parked = momentum_rotation(falling, top_n=2, lookback=252, absolute=True,
                               cash_rate=0.02)
    exposed = momentum_rotation(falling, top_n=2, lookback=252, absolute=False)
    assert parked.cagr > 0 > exposed.cagr


def test_trend_filter_sits_out_a_falling_asset():
    n = 1500
    up, down = _ramp(n, 0.0006), pd.Series(
        100 * np.exp(-np.arange(n) * 0.0006), index=_days(n))
    filtered = trend_filtered_mix({"u": up, "d": down}, window=200)
    plain = static_mix({"u": up, "d": down}, {"u": 1.0, "d": 1.0})
    assert filtered.cagr > plain.cagr


def test_band_rebalance_trades_less_than_the_calendar():
    """밴드 방식이 분기 리밸런싱보다 매매가 적어야 존재 이유가 있다."""
    assets = {"a": _walk(1), "b": _walk(2), "c": _walk(3)}
    banded = band_rebalance(assets, {k: 1.0 for k in assets}, band=0.05)
    calendar = static_mix(assets, {k: 1.0 for k in assets}, rebalance="QE")
    assert 0 < banded.trades < calendar.trades


def test_wide_band_never_rebalances():
    """밴드를 아주 넓게 잡으면 방치와 같아야 한다 (경계 조건)."""
    assets = {"a": _walk(4), "b": _walk(5)}
    banded = band_rebalance(assets, {k: 1.0 for k in assets}, band=0.99)
    drift = static_mix(assets, {k: 1.0 for k in assets}, rebalance=None)
    assert banded.trades == 0
    assert banded.cagr == pytest.approx(drift.cagr, abs=1e-9)


def test_dca_irr_matches_cagr_on_a_constant_grower():
    """수익률이 매일 일정하면 적립식 IRR 은 거치식 CAGR 과 같아야 한다.

    현금흐름 계산이 틀리면 여기서 갈라진다.
    """
    n = 2000
    steady = _ramp(n, 0.0004)
    result = static_mix({"a": steady}, {"a": 1.0}, rebalance=None, one_way_bps=0.0)
    plan = dca(result, monthly=500_000)
    assert plan["연환산수익률"] == pytest.approx(result.cagr, abs=2e-3)


def test_dca_principal_and_multiple_are_consistent():
    result = static_mix({"a": _walk(6)}, {"a": 1.0}, rebalance=None)
    plan = dca(result, monthly=300_000)
    assert plan["원금"] == pytest.approx(300_000 * plan["납입월수"])
    assert plan["수익배수"] == pytest.approx(plan["최종자산"] / plan["원금"])


def test_dca_multiple_is_below_lump_sum_multiple():
    """나중에 넣은 돈은 굴러간 기간이 짧다 — 배수를 CAGR 로 착각하지 않도록."""
    n = 2000
    steady = _ramp(n, 0.0004)
    result = static_mix({"a": steady}, {"a": 1.0}, rebalance=None, one_way_bps=0.0)
    plan = dca(result, monthly=500_000)
    assert plan["수익배수"] < float(result.equity.iloc[-1])


def test_dca_needs_a_year_of_data():
    short = static_mix({"a": _walk(7, n=100)}, {"a": 1.0}, rebalance=None)
    with pytest.raises(ValueError):
        dca(short)


def test_compare_rules_returns_every_rule_ranked():
    assets = {"a": _walk(8), "b": _walk(9), "c": _walk(10), "d": _walk(11)}
    table = compare_rules(assets, monthly=500_000)
    assert len(table) == 8
    assert table["적립 최종자산"].is_monotonic_decreasing
    assert (table["적립 최종자산"] > 0).all()


def test_sign_test_excludes_ties():
    """동점을 패로 세면 증거가 통째로 약해진다."""
    from tsignal.evaluation.compounding import sign_test

    assert sign_test(5, 0) == pytest.approx(0.0625)
    assert sign_test(5, 2) > 0.4
    assert sign_test(0, 0) == 1.0
    assert sign_test(3, 3) == 1.0


def test_subperiod_test_cannot_reach_significance_with_five_periods():
    """구간 5개면 전승해도 p=0.0625 — 0.05 를 넘길 수 없다.

    이걸 모르고 "유의하지 않다"고 적으면 검정력 부족을 증거 부재로 오독한다.
    """
    from tsignal.evaluation.compounding import subperiod_test

    assets = {"a": _walk(12), "b": _walk(13), "c": _walk(14), "d": _walk(15)}
    table = subperiod_test(assets, periods=5)
    assert table.attrs["min_p"] > 0.05
    assert len(table.attrs["segments"]) == 5
    assert set(table["규칙"]) == {"5%밴드", "모멘텀상위2", "모멘텀상위3", "추세필터"}


def test_sign_test_excludes_ties():
    """동점을 패로 세면 증거가 통째로 약해진다."""
    from tsignal.evaluation.compounding import sign_test

    assert sign_test(5, 0) == pytest.approx(0.0625)
    assert sign_test(5, 2) > 0.4
    assert sign_test(0, 0) == 1.0
    assert sign_test(3, 3) == 1.0


def test_subperiod_test_cannot_reach_significance_with_five_periods():
    """구간 5개면 전승해도 p=0.0625 — 0.05 를 넘길 수 없다.

    이걸 모르고 "유의하지 않다"고 적으면 검정력 부족을 증거 부재로 오독한다.
    """
    from tsignal.evaluation.compounding import subperiod_test

    assets = {"a": _walk(12), "b": _walk(13), "c": _walk(14), "d": _walk(15)}
    table = subperiod_test(assets, periods=5)
    assert table.attrs["min_p"] > 0.05
    assert len(table.attrs["segments"]) == 5
    assert set(table["규칙"]) == {"5%밴드", "모멘텀상위2", "모멘텀상위3", "추세필터"}
