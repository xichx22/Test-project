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
