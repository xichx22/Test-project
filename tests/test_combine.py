"""조합 탐색 테스트.

핵심은 마지막 두 개다: 조합을 수천 개 훑으면 엣지가 없어도 좋아 보이는 게
나오지만, IS 선정 → OOS 채점 프로토콜은 그걸 통과시키지 않아야 한다.
"""

import numpy as np
import pytest

from tsignal.datasource import Interval, SyntheticDataSource
from tsignal.evaluation.combine import (
    BARE,
    Combo,
    CombinationLab,
    filter_contribution,
    select_and_validate,
    split_labs,
)
from tsignal.signals import filters


@pytest.fixture(scope="module")
def universe():
    source = SyntheticDataSource(seed=4242, drift=0.1, annual_vol=0.30)
    return {c: source.candles(c, Interval.D1, count=900) for c in ("AAA", "BBB", "CCC", "DDD", "EEE")}


@pytest.fixture(scope="module")
def lab(universe):
    return CombinationLab(universe, interval=Interval.D1, horizon=5)


def test_lab_pools_all_symbols(lab, universe):
    assert lab.n_codes == len(universe)
    assert len(lab.excess) == len(lab.ret) == len(lab.code_idx)
    assert lab.trig.shape[0] == len(lab.excess)
    assert np.isfinite(lab.excess).all()


def test_enumerate_skips_same_axis_pairs(lab):
    axis_of = {name: spec.axis for name, spec in filters.REGISTRY.items()}
    combos = lab.enumerate_combos(max_filters=2)
    for combo in combos:
        axes = [axis_of[f] for f in combo.filters]
        assert len(axes) == len(set(axes)), f"같은 축끼리 결합됨: {combo.name}"
    assert any(c.filters == () for c in combos)          # 필터 없는 기준선도 포함


def test_adding_a_filter_never_increases_sample(lab):
    """필터는 조건을 좁히므로 표본이 늘어날 수 없다 — AND 논리의 정합성 검사."""
    trigger = lab.trigger_names[0]
    bare_n = lab.stats(Combo(trigger, ()))["n"]
    for name in lab.filter_names:
        assert lab.stats(Combo(trigger, (name,)))["n"] <= bare_n


def test_lift_is_measured_against_the_bare_trigger(lab):
    table = lab.search(max_filters=1, min_events=1)
    bare_rows = table[table["n_filters"] == 0]
    assert np.allclose(bare_rows["lift"].dropna(), 0.0)   # 필터가 없으면 lift 는 0
    assert table["edge_bare"].notna().any()


def test_threshold_scales_with_number_of_combos_tested(lab):
    few = lab.search(max_filters=0, min_events=1)
    many = lab.search(max_filters=2, min_events=1)
    assert many.attrs["n_trials"] > few.attrs["n_trials"]
    assert many.attrs["threshold"] > few.attrs["threshold"]


def test_no_combo_survives_multiple_testing_on_noise(lab):
    """드리프트만 있는 합성 시장에서는 어떤 조합도 보정 문턱을 넘으면 안 된다."""
    table = lab.search(max_filters=2, min_events=60)
    assert not table["pass_sidak"].any(), list(table[table["pass_sidak"]].index[:5])


def test_oos_selection_protocol_rejects_noise(universe):
    """IS 에서 최고를 골라도 OOS 생존이 우연 범위여야 한다 — 프로토콜의 존재 이유."""
    is_lab, oos_lab = split_labs(universe, train_ratio=0.6, interval=Interval.D1, horizon=5)
    res = select_and_validate(is_lab, oos_lab, top_k=15, min_events=40, min_oos_events=15)
    assert res.n_selected > 0
    assert res.binomial_p > 0.05, f"노이즈에서 OOS 생존이 유의하게 나왔다 (p={res.binomial_p})"
    assert "우연 범위" in res.verdict


def test_dedupe_collapses_identical_masks(lab):
    """느슨한 필터를 붙여도 봉 집합이 같으면 같은 가설이다 — 하나로 합쳐야 한다."""
    trigger = lab.trigger_names[0]
    combos = lab.enumerate_combos(max_filters=2)
    deduped = lab.dedupe(combos)
    assert len(deduped) < len(combos)

    # 남은 조합끼리는 마스크가 모두 달라야 한다.
    seen = {lab.mask(c).tobytes() for c in deduped}
    assert len(seen) == len(deduped)

    # 같은 마스크 그룹에서는 가장 단순한(필터 적은) 표현이 남는다.
    bare = Combo(trigger, ())
    same_as_bare = [c for c in combos if c.trigger == trigger
                    and np.array_equal(lab.mask(c), lab.mask(bare))]
    if len(same_as_bare) > 1:
        kept = [c for c in deduped if np.array_equal(lab.mask(c), lab.mask(bare))]
        assert len(kept) == 1 and kept[0].filters == ()


def test_dedupe_lowers_the_multiple_testing_threshold(lab):
    """중복을 남기면 독립이 아닌 가설을 개수에 넣어 문턱이 과대 계산된다."""
    with_dupes = lab.search(max_filters=2, min_events=40, dedupe=False)
    without = lab.search(max_filters=2, min_events=40, dedupe=True)
    assert without.attrs["n_combos"] < with_dupes.attrs["n_combos"]
    assert without.attrs["threshold"] < with_dupes.attrs["threshold"]


def test_filter_contribution_reports_every_axis(lab):
    contrib = filter_contribution(lab, min_events=40)
    assert not contrib.empty
    assert contrib["improve_rate"].between(0, 1).all()
    assert (contrib["n_improved"] <= contrib["n_pairs"]).all()
    assert contrib["sign_p"].between(0, 1).all()


def test_combo_name_marks_the_bare_case():
    assert Combo("x", ()).name.endswith(BARE)
    assert Combo("x", ("a", "b")).name == "x + a & b"


# =====================================================================
# 날짜 군집 보정 — 이 프로젝트에서 가장 크게 결과를 바꾼 수정
# =====================================================================

def test_clustered_t_deflates_cross_sectionally_correlated_samples():
    """같은 날 여러 종목에서 동시에 뜬 신호는 독립 관측이 아니다.

    극단적인 예: 20개 종목이 완전히 같은 수익률을 갖는다면 실질 표본은 1개다.
    순진한 t 는 √20 만큼 부풀지만 군집 보정 t 는 그렇지 않아야 한다.
    """
    from tsignal.evaluation.metrics import clustered_t_stat, t_stat
    import pandas as pd

    rng = np.random.default_rng(0)
    day_effect = rng.normal(0.001, 0.02, 60)           # 날짜 60개
    values = np.repeat(day_effect, 20)                 # 날짜당 20종목이 완전 동조
    days = np.repeat(np.arange(60), 20)

    naive = t_stat(pd.Series(values))
    clustered = clustered_t_stat(values, days)
    assert abs(naive) > abs(clustered) * 3, (naive, clustered)
    # 완전 동조라면 실질 표본은 날짜 수뿐이다.
    assert clustered == pytest.approx(t_stat(pd.Series(day_effect)), rel=1e-9)


def test_clustered_t_matches_naive_when_one_observation_per_day():
    from tsignal.evaluation.metrics import clustered_t_stat, t_stat
    import pandas as pd

    rng = np.random.default_rng(1)
    values = rng.normal(0.002, 0.03, 200)
    days = np.arange(200)
    assert clustered_t_stat(values, days) == pytest.approx(t_stat(pd.Series(values)), rel=1e-9)


def test_lab_reports_both_naive_and_clustered_t(lab):
    row = lab.stats(Combo(lab.trigger_names[0], ()))
    assert "t_edge" in row and "t_naive" in row and "n_days" in row
    assert row["n_days"] <= row["n"]                   # 날짜 수는 표본 수를 넘을 수 없다


def test_eligibility_requires_enough_distinct_days(lab):
    """표본 수만 채우고 날짜가 적은 조합은 검정 대상에서 빠져야 한다."""
    loose = lab.search(max_filters=1, min_events=30, min_days=1)
    strict = lab.search(max_filters=1, min_events=30, min_days=200)
    assert strict.attrs["n_trials"] < loose.attrs["n_trials"]
