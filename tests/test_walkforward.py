"""롤링 워크포워드 테스트."""

import numpy as np
import pandas as pd
import pytest

from tsignal.datasource import Interval, SyntheticDataSource
from tsignal.evaluation import walkforward as wf
from tsignal.evaluation.combine import build_panels


@pytest.fixture(scope="module")
def panels():
    source = SyntheticDataSource(seed=777, drift=0.1, annual_vol=0.30)
    data = {c: source.candles(c, Interval.D1, count=1100) for c in ("AAA", "BBB", "CCC", "DDD")}
    return build_panels(data, interval=Interval.D1, horizon=5)


def test_folds_are_cut_by_date_not_row_number():
    folds = wf.make_folds(
        pd.Timestamp("2021-01-01", tz="Asia/Seoul"), pd.Timestamp("2026-01-01", tz="Asia/Seoul"),
        train_months=24, test_months=6,
    )
    assert len(folds) > 1
    for fold in folds:
        assert fold.train_start < fold.train_end == fold.test_start < fold.test_end
    # rolling 은 학습창 길이가 일정하다.
    spans = {(f.train_end - f.train_start).days // 30 for f in folds}
    assert spans == {24}


def test_anchored_scheme_grows_the_training_window():
    folds = wf.make_folds(
        pd.Timestamp("2021-01-01", tz="Asia/Seoul"), pd.Timestamp("2026-01-01", tz="Asia/Seoul"),
        train_months=24, test_months=6, scheme="anchored",
    )
    starts = {f.train_start for f in folds}
    assert len(starts) == 1                                  # 시작점이 고정
    assert folds[-1].train_end > folds[0].train_end          # 끝점만 밀린다


def test_test_windows_do_not_overlap():
    folds = wf.make_folds(
        pd.Timestamp("2021-01-01", tz="Asia/Seoul"), pd.Timestamp("2026-01-01", tz="Asia/Seoul"),
        train_months=24, test_months=6,
    )
    for a, b in zip(folds, folds[1:]):
        assert a.test_end <= b.test_start


def test_make_folds_raises_when_window_exceeds_data(panels):
    with pytest.raises(ValueError, match="폴드를 만들 수 없습니다"):
        wf.run(panels=panels[0], panel_columns=panels[1:], horizon=5, train_months=600)


def test_walkforward_rejects_noise(panels):
    """합성 노이즈에서는 폴드를 넘어 일관된 결과가 나오면 안 된다."""
    result = wf.run(
        panels=panels[0], panel_columns=(panels[1], panels[2]), horizon=5,
        train_months=24, test_months=6, top_k=10, min_events=40, min_test_events=15,
    )
    assert len(result.folds) >= 3
    assert "우연 범위" in result.combo_verdict

    # 폴드를 단위로 세야 한다 — 조합을 단위로 세면 상관 때문에 p 가 가짜로 작아진다.
    stats = result.combo_stats
    assert stats["n_folds"] == len(result.folds)
    assert 0 <= stats["folds_above_null"] <= stats["n_folds"]


def test_filter_summary_covers_all_folds(panels):
    result = wf.run(
        panels=panels[0], panel_columns=(panels[1], panels[2]), horizon=5,
        train_months=24, test_months=6, top_k=10, min_events=40, min_test_events=15,
    )
    summary = result.filter_summary
    assert not summary.empty
    assert (summary["folds_positive"] <= summary["n_folds"]).all()
    assert summary["consistency"].between(0, 1).all()
    assert summary["sign_p"].between(0, 1).all()


# =====================================================================
# 기준선(baseline) 선택이 만드는 인공물
# =====================================================================

def _trend_down_lift(lab, min_events: int = 30) -> float:
    """모든 트리거에 trend_down 을 걸었을 때 lift 의 중앙값."""
    from tsignal.evaluation.combine import Combo

    lifts = []
    for trigger in lab.trigger_names:
        bare = lab.stats(Combo(trigger, ()))
        with_filter = lab.stats(Combo(trigger, ("trend_down",)))
        if bare["n"] < min_events or with_filter["n"] < min_events:
            continue
        if not (np.isfinite(bare["edge"]) and np.isfinite(with_filter["edge"])):
            continue
        lifts.append(with_filter["edge"] - bare["edge"])
    return float(np.median(lifts)) if lifts else np.nan


@pytest.fixture(scope="module")
def noise_panels():
    """엣지가 0인 순수 랜덤워크. 어떤 필터도 도움이 되면 안 된다."""
    source = SyntheticDataSource(seed=31337, drift=0.0, annual_vol=0.30)
    data = {f"S{i:02d}": source.candles(f"S{i:02d}", Interval.D1, count=900) for i in range(12)}
    return build_panels(data, interval=Interval.D1, horizon=5)


def test_time_series_baseline_manufactures_edge_in_short_windows(noise_panels):
    """창 안 평균을 기준선으로 쓰면 엣지가 없어도 trend_down 이 좋아 보인다.

    `trend_down`(종가<60EMA)은 정의상 '자기 평균 아래'를 고르고, 창 안 평균으로
    빼면 그 뒤의 평균회귀가 초과수익으로 잡힌다. 짧은 창일수록 심하다.
    이 인공물 때문에 워크포워드가 12/12 폴드 양수를 뱉었었다.
    """
    from tsignal.evaluation.combine import CombinationLab

    panels, trig_cols, filt_cols = noise_panels
    # 한 창은 우연히 어느 쪽으로도 나올 수 있다. 인공물은 통계적 현상이므로
    # 여러 창에 걸쳐 체계적으로 나타나는지를 본다.
    begin = max(p.index[0] for p in panels) + pd.DateOffset(months=7)   # 지표 워밍업 이후
    last = min(p.index[-1] for p in panels)

    ts_lifts, cs_lifts = [], []
    start = begin
    while start + pd.DateOffset(months=3) <= last:
        end = start + pd.DateOffset(months=3)
        for baseline, bucket in (("time_series", ts_lifts), ("cross_sectional", cs_lifts)):
            lab = CombinationLab.from_panels(
                panels, (trig_cols, filt_cols), horizon=5,
                start=start, end=end, baseline=baseline,
            )
            value = _trend_down_lift(lab, min_events=20)
            if np.isfinite(value):
                bucket.append(value)
        start = end

    assert len(ts_lifts) >= 4 and len(ts_lifts) == len(cs_lifts)
    ts_positive = float(np.mean(np.array(ts_lifts) > 0))
    cs_positive = float(np.mean(np.array(cs_lifts) > 0))

    # 엣지가 0인 데이터인데도 시계열 기준선은 절반을 넘겨 양수를 만들어낸다.
    assert ts_positive > 0.5, ts_lifts
    # 그리고 횡단면 기준선보다 체계적으로 후하다.
    assert np.median(ts_lifts) > np.median(cs_lifts), (ts_lifts, cs_lifts)
    assert ts_positive >= cs_positive


def test_cross_sectional_baseline_is_the_default(noise_panels):
    from tsignal.evaluation.combine import CombinationLab

    panels, trig_cols, filt_cols = noise_panels
    lab = CombinationLab.from_panels(panels, (trig_cols, filt_cols), horizon=5)
    assert lab.baseline == "cross_sectional"
    # 같은 날짜의 초과수익 합은 0 이어야 한다 (그날 평균을 뺐으므로).
    frame = pd.DataFrame({"day": lab.day, "excess": lab.excess})
    per_day = frame.groupby("day")["excess"].sum()
    assert np.allclose(per_day.to_numpy(), 0.0, atol=1e-9)


def test_unknown_baseline_is_rejected(noise_panels):
    from tsignal.evaluation.combine import CombinationLab

    panels, trig_cols, filt_cols = noise_panels
    with pytest.raises(ValueError, match="알 수 없는 baseline"):
        CombinationLab.from_panels(panels, (trig_cols, filt_cols), horizon=5, baseline="nope")
