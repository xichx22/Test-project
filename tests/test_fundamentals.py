"""재무 스크리너 — 거르기만 하고 순위는 매기지 않는다."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.fundamentals import Screen, distribution, screen


def _frame():
    return pd.DataFrame({
        "code": ["000001", "000002", "000003", "000004", "000005"],
        "name": ["싼회사", "비싼회사", "빚많은회사", "결측회사", "작은회사"],
        "cap": [1e13, 1e13, 1e13, 1e13, 1e11],
        "per": [7.0, 40.0, 8.0, np.nan, 6.0],
        "pbr": [0.7, 3.0, 0.6, 0.8, 0.5],
        "roe": [12.0, 15.0, 9.0, 10.0, 20.0],
        "div": [3.0, 0.0, 2.0, 1.0, 4.0],
        "debt": [50.0, 60.0, 400.0, 70.0, 30.0],
    })


def test_screen_drops_each_kind_of_failure():
    out = screen(_frame())
    kept = set(out.passed["code"])
    assert "000001" in kept                      # 전부 통과
    assert "000002" not in kept                  # PER·PBR 초과
    assert "000003" not in kept                  # 부채비율 초과
    assert "000005" not in kept                  # 시총 미달
    assert out.total == 5


def test_missing_values_are_dropped_by_default():
    """재무를 공시하지 않는 종목을 통과시키면 스크리너가 무의미해진다."""
    out = screen(_frame())
    assert "000004" not in set(out.passed["code"])


def test_missing_values_can_be_kept_explicitly():
    out = screen(_frame(), Screen(require_all=False))
    assert "000004" in set(out.passed["code"])


def test_drop_reasons_are_counted():
    out = screen(_frame())
    assert out.dropped
    assert sum(out.dropped.values()) >= 3
    assert all(v > 0 for v in out.dropped.values())


def test_disabled_rules_are_skipped():
    """문턱이 None 이면 그 조건은 아예 걸지 않는다."""
    loose = Screen(max_per=None, max_pbr=None, min_roe=None,
                   max_debt=None, min_cap=None)
    out = screen(_frame(), loose)
    assert len(out.passed) == 5


def test_result_is_sorted_by_size_not_by_a_score():
    """순위를 매기면 1등을 사게 되고, 그건 검증되지 않은 예측이다."""
    frame = _frame()
    frame.loc[frame["code"] == "000001", "cap"] = 5e12
    out = screen(frame, Screen(max_per=50, max_pbr=5, min_roe=0,
                               max_debt=1000, min_cap=1e12, require_all=False))
    caps = out.passed["cap"].tolist()
    assert caps == sorted(caps, reverse=True)


def test_empty_result_reports_clearly():
    out = screen(_frame(), Screen(max_per=1.0))
    assert out.passed.empty
    assert "없습니다" in out.report()


def test_distribution_reports_missing_rate():
    table = distribution(_frame())
    per = table[table["지표"] == "per"].iloc[0]
    assert per["표본"] == 4
    assert per["결측률"] == pytest.approx(0.2)


def test_distribution_skips_all_missing_columns():
    frame = _frame()
    frame["div"] = np.nan
    table = distribution(frame)
    assert "div" not in set(table["지표"])


def test_nan_comparison_is_handled_explicitly():
    """pandas 에서 `NaN <= 15` 는 NaN 이 아니라 False 다.

    fillna 로 결측을 되살릴 수 있다고 생각하면 조용히 틀린다 —
    결측 종목이 require_all 설정과 무관하게 항상 탈락한다.
    """
    values = pd.Series([np.nan, 10.0])
    assert list(values <= 15) == [False, True]
    assert list((values <= 15).fillna(True)) == [False, True]     # 안 바뀐다
    assert list((values <= 15).where(values.notna(), True)) == [True, True]
