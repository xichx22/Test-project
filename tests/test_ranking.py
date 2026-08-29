"""시가총액 순위 기반 매수법 검증."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.ranking import (
    common_start,
    market_caps,
    survivor_note,
    top_n_portfolio,
    turnover_report,
)


def _frames(n: int = 800, n_c: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", periods=n, freq="B", tz="Asia/Seoul")
    codes = [f"{i:06d}" for i in range(n_c)]
    steps = rng.normal(0.0003, 0.015, (n, n_c))
    close = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)),
                         index=index, columns=codes)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, (n, n_c)))
    shares = pd.Series(rng.integers(1_000_000, 50_000_000, n_c).astype(float),
                       index=codes)
    return open_, close, shares


def test_market_caps_scale_with_shares():
    _, close, shares = _frames()
    caps = market_caps(close, shares)
    assert caps.shape == close.shape
    # 주식수를 2배로 하면 시총도 2배
    doubled = market_caps(close, shares * 2)
    assert np.allclose(doubled.to_numpy(), caps.to_numpy() * 2)


def test_market_caps_ignore_codes_without_share_counts():
    _, close, shares = _frames()
    caps = market_caps(close, shares.iloc[:10])
    assert list(caps.columns) == list(shares.index[:10])


def test_common_start_skips_the_thin_beginning():
    """상장 종목이 몇 개뿐인 앞 구간을 잘라내야 한다.

    합집합 인덱스의 첫 날에는 종목이 2개뿐일 수 있다. 거기서부터 재면
    "상위 20"이 성립하지 않는 구간이 결과에 섞인다.
    """
    _, close, _ = _frames(n=300, n_c=10)
    close.iloc[:100, 2:] = np.nan          # 앞 100봉은 2종목만 상장
    start = common_start(close, min_listed=5)
    assert start == close.index[100]
    assert close.loc[start].notna().sum() >= 5


def test_common_start_raises_when_never_enough():
    _, close, _ = _frames(n=100, n_c=3)
    with pytest.raises(ValueError):
        common_start(close, min_listed=10)


def test_top_n_holds_exactly_n_names():
    """상위 N 만 담아야 한다 — 비중 합이 1이고 0이 아닌 항목이 N개."""
    open_, close, shares = _frames()
    caps = market_caps(close, shares)
    result = top_n_portfolio(caps, open_, close, top_n=5, rebalance="QE",
                            cost_bps=0.0)
    assert result.trades > 0
    assert len(result.equity) == len(close)


def test_cap_weighting_differs_from_equal_weighting():
    open_, close, shares = _frames()
    caps = market_caps(close, shares)
    equal = top_n_portfolio(caps, open_, close, top_n=5, weighting="equal")
    cap = top_n_portfolio(caps, open_, close, top_n=5, weighting="cap")
    assert not np.allclose(equal.daily.to_numpy(), cap.daily.to_numpy())


def test_costs_reduce_return():
    open_, close, shares = _frames()
    caps = market_caps(close, shares)
    free = top_n_portfolio(caps, open_, close, top_n=5, cost_bps=0.0)
    paid = top_n_portfolio(caps, open_, close, top_n=5, cost_bps=28.0)
    assert paid.cagr < free.cagr


def test_yearly_rebalancing_trades_less_than_quarterly():
    open_, close, shares = _frames()
    caps = market_caps(close, shares)
    quarterly = top_n_portfolio(caps, open_, close, top_n=5, rebalance="QE")
    yearly = top_n_portfolio(caps, open_, close, top_n=5, rebalance="YE")
    assert yearly.trades < quarterly.trades


def test_rebalance_waits_for_enough_listed_names():
    """상장 종목이 N보다 적으면 리밸런싱하지 않아야 한다."""
    open_, close, shares = _frames(n=400, n_c=10)
    close.iloc[:200, 3:] = np.nan
    caps = market_caps(close, shares)
    result = top_n_portfolio(caps, open_, close, top_n=8, rebalance="QE")
    # 앞 200봉에는 8종목이 없으므로 그 구간 수익률은 전부 0(현금)이어야 한다
    assert result.daily.iloc[:200].abs().max() == pytest.approx(0.0)


def test_turnover_report_counts_add_up():
    open_, close, shares = _frames()
    caps = market_caps(close, shares)
    table = turnover_report(caps, top_n=6, rebalance="QE")
    assert not table.empty
    assert (table["신규편입"] + table["유지"] == 6).all()
    assert (table["신규편입"] == table["탈락"]).all()


def test_survivor_note_ignores_rows_too_thin_to_rank():
    open_, close, shares = _frames(n=400, n_c=10)
    close.iloc[:100, :] = np.nan
    close.iloc[:100, :2] = 100.0          # 앞 100봉은 2종목만
    caps = market_caps(close, shares)
    note = survivor_note(caps, top_n=5)
    assert 0.0 <= note["교체율"] <= 1.0
    assert note["끝까지 상위N 유지"] <= 5
