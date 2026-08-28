"""유니버스 풀링 검증 테스트.

핵심은 마지막 테스트다: 시장이 우상향할 때 원시 t 로 판정하면 엣지가 없는
신호도 유의해 보인다. 초과수익으로 검정해야 그 착시가 사라진다.
"""

import numpy as np
import pandas as pd
import pytest

from tsignal.datasource import Interval, SyntheticDataSource
from tsignal.evaluation.universe import screen_universe, signal_returns


@pytest.fixture(scope="module")
def drifting_universe():
    """연 60% 드리프트를 준 랜덤워크 12종목. 신호와 무관하게 무조건부 기대값이 크다.

    종목 수가 너무 적으면 횡단면 기준선(같은 날 전 종목 평균)이 잡음투성이가 되어
    시장 성분을 제대로 걷어내지 못한다. 실제 유니버스와 같은 이유로 넉넉히 잡는다.
    """
    source = SyntheticDataSource(seed=99, drift=0.6, annual_vol=0.30)
    return {f"S{i:02d}": source.candles(f"S{i:02d}", Interval.D1, count=700) for i in range(12)}


def test_signal_returns_uses_a_cross_sectional_baseline(drifting_universe):
    """초과수익의 기준선은 '같은 날 전 종목 평균'이어야 한다.

    종목별 창 안 평균을 기준선으로 쓰면 인공적인 평균회귀가 생긴다
    (combine.CombinationLab._excess 독스트링 참고). 여기서는 같은 날짜의
    초과수익이 전 종목에 걸쳐 상쇄되는지로 그 성질을 확인한다.
    """
    pooled, base = signal_returns(drifting_universe, interval=Interval.D1, horizon=5)
    assert (base > 0).all()                                  # 드리프트가 있으므로 무조건부는 양수
    frame = pooled["macd_cross_up"]
    assert {"code", "ret", "excess", "day"} <= set(frame.columns)

    # 시장이 오른 날에는 excess < ret, 빠진 날에는 excess > ret 이 된다.
    # 종목별 평균을 빼던 예전 방식이었다면 항상 excess < ret 이었다.
    assert (frame["excess"] > frame["ret"]).any()
    assert (frame["excess"] < frame["ret"]).any()

    # 전체 패널에서 초과수익의 평균은 0 근처여야 한다.
    everything = pd.concat(pooled.values(), ignore_index=True)
    assert abs(float(everything["excess"].mean())) < abs(float(everything["ret"].mean()))


def test_raw_t_overstates_edge_in_a_rising_market(drifting_universe):
    """드리프트만 있는 시장에서 t_raw 는 부풀고 t_edge 는 0 근처여야 한다."""
    out = screen_universe(
        drifting_universe, interval=Interval.D1, horizon=20, min_events=50,
    )
    sizeable = out[out["n"] >= 50]
    assert not sizeable.empty

    # 원시 t 는 드리프트 때문에 여러 신호에서 크게 나온다.
    assert (sizeable["t_raw"] > 2.0).any()
    # 그러나 초과수익 기준으로는 어떤 신호도 보정 문턱을 넘지 못한다.
    assert (sizeable["t_edge"].abs() < out.attrs["threshold"]).all()
    assert "채택후보" not in set(out["verdict"])


def test_breadth_flags_concentration(drifting_universe):
    out = screen_universe(drifting_universe, interval=Interval.D1, horizon=5, min_events=1)
    assert out["breadth"].between(0, 1).all()
    assert out["n_codes"].max() <= len(drifting_universe)


def test_verdict_requires_both_significance_and_breadth():
    from tsignal.evaluation.universe import _verdict
    import pandas as pd

    strong_broad = pd.Series({"n": 500, "t_edge": 4.0, "breadth": 0.8})
    strong_narrow = pd.Series({"n": 500, "t_edge": 4.0, "breadth": 0.3})
    weak = pd.Series({"n": 500, "t_edge": 0.5, "breadth": 0.9})
    tiny = pd.Series({"n": 10, "t_edge": 9.0, "breadth": 1.0})

    assert _verdict(strong_broad, threshold=3.0, min_events=100) == "채택후보"
    assert _verdict(strong_narrow, threshold=3.0, min_events=100) == "쏠림주의"
    assert _verdict(weak, threshold=3.0, min_events=100) == "기각"
    assert _verdict(tiny, threshold=3.0, min_events=100) == "표본부족"
