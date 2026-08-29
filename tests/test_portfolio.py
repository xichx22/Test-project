"""실행 계층(ETF 선택·세금) 검증."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.allocation import _rebalance_marks, static_mix
from tsignal.evaluation.portfolio import (
    ETF_CATALOG,
    after_tax_mix,
    account_comparison,
    horizon_gap,
    liquidity,
    spec_for,
)


def _days(n: int = 500) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B", tz="Asia/Seoul")


def _walk(seed: int, n: int = 500, drift: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.01, n)
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=_days(n))


def test_rebalance_marks_land_on_real_trading_days():
    """분기말 라벨이 아니라 그 분기의 마지막 실제 봉이어야 한다.

    라벨 방식(`resample("QE").last().index`)은 휴장일이면 어떤 봉과도 맞지
    않아 리밸런싱이 조용히 건너뛰어진다. 그 회귀를 여기서 막는다.
    """
    index = _days(500)
    # 분기말(3/31, 6/30, 9/30, 12/31)을 통째로 빼서 휴장을 흉내낸다.
    holidays = [d for d in index if (d.month, d.day) in
                {(3, 31), (6, 30), (9, 30), (12, 31)}]
    trimmed = index.drop(holidays)
    marks = _rebalance_marks(trimmed, "QE")

    assert marks, "분기가 여러 개인데 리밸런싱 시점이 하나도 안 잡혔다"
    assert marks <= set(trimmed), "실제 봉에 없는 날짜가 리밸런싱 시점으로 잡혔다"
    quarters = {(d.year, d.quarter) for d in trimmed}
    assert len(marks) == len(quarters), "분기마다 정확히 한 번이어야 한다"


def test_skipped_rebalance_bug_would_change_the_answer():
    """라벨 방식이었다면 리밸런싱 횟수가 실제로 줄어든다 — 무해한 버그가 아니다."""
    index = _days(1200)
    frame = pd.DataFrame({"a": _walk(1, 1200), "b": _walk(2, 1200)}, index=index)
    label_marks = set(frame.resample("QE").last().index) & set(frame.index)
    real_marks = _rebalance_marks(frame.index, "QE")
    assert len(label_marks) < len(real_marks)


def test_pension_account_matches_a_tax_free_backtest():
    """세율을 0 으로 준 after_tax_mix 는 static_mix 와 같아야 한다.

    이게 맞아야 두 계좌 비교에서 남는 차이가 오직 세금이라고 말할 수 있다.
    """
    assets = {"a": _walk(3), "b": _walk(4), "c": _walk(5)}
    weights = {k: 1.0 for k in assets}
    plain = static_mix(assets, weights, rebalance="QE")
    free = after_tax_mix(assets, weights, rebalance="QE",
                         tax_rates={k: 0.0 for k in assets})
    assert free.rebalances == plain.trades
    # 완전히 같지는 않다: static_mix 는 수수료를 그 날 수익률에서 빼고(가산),
    # 여기서는 평가액에 곱해서 뺀다(승산). 차이는 2차항이라 1e-6 아래다.
    assert free.cagr == pytest.approx(plain.cagr, abs=5e-6)


def test_tax_only_ever_costs_money():
    """과세 계좌가 비과세 계좌보다 나은 경우는 없어야 한다."""
    assets = {"a": _walk(6), "b": _walk(7), "c": _walk(8)}
    weights = {k: 1.0 for k in assets}
    free = after_tax_mix(assets, weights, tax_rates={k: 0.0 for k in assets})
    taxed = after_tax_mix(assets, weights, tax_rates={k: 0.154 for k in assets})
    assert taxed.tax_paid > 0
    assert taxed.cagr < free.cagr


def test_tax_is_charged_on_gains_not_on_principal():
    """전 자산이 같은 값으로 제자리걸음이면 실현이익이 없어 세금도 0 이어야 한다.

    취득원가 추적이 틀리면 원금에까지 세금이 붙는데, 그 실수를 여기서 잡는다.
    """
    flat = pd.Series(100.0, index=_days(400))
    assets = {"a": flat, "b": flat.copy()}
    taxed = after_tax_mix(assets, {"a": 1.0, "b": 1.0},
                          tax_rates={"a": 0.154, "b": 0.154}, one_way_bps=0.0)
    assert taxed.tax_paid == pytest.approx(0.0, abs=1e-12)
    assert taxed.equity.iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_more_rebalancing_realizes_more_tax():
    """월 리밸런싱은 연 1회보다 더 자주 이익을 실현하므로 세금이 더 나온다."""
    assets = {"a": _walk(9, drift=0.0008), "b": _walk(10, drift=-0.0002)}
    weights = {k: 1.0 for k in assets}
    rates = {k: 0.154 for k in assets}
    monthly = after_tax_mix(assets, weights, rebalance="ME", tax_rates=rates)
    yearly = after_tax_mix(assets, weights, rebalance="YE", tax_rates=rates)
    assert monthly.tax_paid > yearly.tax_paid


def test_domestic_equity_is_exempt_in_account_comparison():
    """국내주식형만 담으면 일반계좌와 연금계좌의 운용 중 세금 차이가 없다."""
    assets = {"069500": _walk(11), "102110": _walk(12)}
    table = account_comparison(assets, {k: 1.0 for k in assets},
                               domestic_equity=("069500", "102110"))
    assert table.attrs["drag"] == pytest.approx(0.0, abs=1e-9)


def test_liquidity_uses_median_not_mean():
    """거래대금 하루치 폭등이 결과를 끌어올리면 안 된다."""
    index = _days(300)
    volume = pd.Series(1_000.0, index=index)
    volume.iloc[-1] = 10_000_000.0        # 테마가 붙은 하루
    frame = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                          "close": 100.0, "volume": volume}, index=index)
    table = liquidity({"069500": frame}, days=250)
    assert table.loc["069500", "median_turnover"] == pytest.approx(100_000.0)


def test_horizon_gap_compounds():
    table = horizon_gap(0.0044, 0.1072, years=(10, 30))
    gaps = table["차이"].tolist()
    assert gaps[1] > gaps[0] * 3, "복리라면 30년 차이가 10년의 3배를 넘어야 한다"


def test_catalog_codes_are_unique_and_named():
    codes = [spec.code for spec in ETF_CATALOG]
    assert len(codes) == len(set(codes))
    assert all(spec.name and spec.asset for spec in ETF_CATALOG)
    assert spec_for("069500").taxable is False   # 국내주식형은 매매차익 비과세
    assert spec_for("133690").taxable is True
    assert spec_for("000000") is None
