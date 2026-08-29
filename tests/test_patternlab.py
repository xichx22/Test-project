"""패턴 검증 랩 — 문턱 계산과 판정 규칙."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.patternlab import (
    count_events,
    detect,
    min_sign_p,
    score,
    sidak_t,
    sign_test,
    subperiods,
)


def _candles(seed: int, n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", periods=n, freq="B", tz="Asia/Seoul")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.integers(1_000, 9_000, n).astype(float)},
        index=index,
    )


def _every_nth(step: int):
    def detector(candles: pd.DataFrame) -> pd.Series:
        hit = np.zeros(len(candles), dtype=bool)
        hit[::step] = True
        hit[-70:] = False                  # 보유 기간이 끝을 넘지 않도록
        return pd.Series(hit, index=candles.index)
    return detector


@pytest.fixture(scope="module")
def universe():
    return {f"{i:06d}": _candles(i) for i in range(1, 13)}


def test_sidak_threshold_rises_with_the_number_of_hypotheses():
    """단일 가설의 t=1.96 감각을 수천 개에 그대로 쓰면 안 된다."""
    assert sidak_t(1) == pytest.approx(1.96, abs=0.01)
    assert sidak_t(11) > sidak_t(1)
    assert sidak_t(5176) > sidak_t(11)
    assert sidak_t(5176) == pytest.approx(4.42, abs=0.05)


def test_sidak_rejects_a_nonsense_count():
    with pytest.raises(ValueError):
        sidak_t(0)


def test_sign_test_excludes_ties():
    assert sign_test(5, 0) == pytest.approx(0.0625)
    assert sign_test(8, 0) == pytest.approx(0.0078, abs=1e-4)
    assert sign_test(5, 2) > 0.4
    assert sign_test(0, 0) == 1.0


def test_min_sign_p_shows_when_a_design_cannot_reach_significance():
    """구간 4~5개로는 전승해도 0.05 를 못 넘긴다 — 설계 단계에서 알아야 한다."""
    assert min_sign_p(4) > 0.05
    assert min_sign_p(5) > 0.05
    assert min_sign_p(6) < 0.05
    assert min_sign_p(8) < 0.01


def test_detect_drops_symbols_with_no_events(universe):
    detectors = {"never": lambda c: pd.Series(False, index=c.index),
                 "sometimes": _every_nth(120)}
    found = detect(universe, detectors)
    assert found["never"] == {}
    assert len(found["sometimes"]) == len(universe)
    assert count_events(found["sometimes"]) > 0


def test_score_marks_thin_samples_as_unmeasured_not_ineffective(universe):
    """발생이 적으면 성과를 내지 않아야 한다.

    "효과 없음"과 "못 쟀음"을 같은 칸에 적으면 결론이 뒤집힌다.
    """
    detectors = {"thin": _every_nth(800), "thick": _every_nth(25)}
    found = detect(universe, detectors)
    table = score(found, universe, min_events=50)
    thin = table[table["패턴"] == "thin"].iloc[0]
    thick = table[table["패턴"] == "thick"].iloc[0]
    assert thin["판정"] == "표본부족"
    assert pd.isna(thin["연초과"]) and pd.isna(thin["t"])
    assert thick["판정"] in {"통과", "미달"}
    assert np.isfinite(thick["t"])


def test_score_threshold_comes_from_the_number_of_patterns(universe):
    detectors = {f"p{i}": _every_nth(25 + i) for i in range(4)}
    found = detect(universe, detectors)
    table = score(found, universe, min_events=10)
    assert table.attrs["n_hypotheses"] == 4
    assert table.attrs["threshold"] == pytest.approx(sidak_t(4))
    for _, row in table.iterrows():
        if row["판정"] == "표본부족":
            continue
        expected = "통과" if abs(row["t"]) >= table.attrs["threshold"] else "미달"
        assert row["판정"] == expected


def test_subperiods_reports_its_own_power_limit(universe):
    found = detect(universe, {"p": _every_nth(20)})["p"]
    out = subperiods(found, universe, periods=4, min_events=1)
    assert out["min_p"] == pytest.approx(0.125)
    assert len(out["표"]) == 4
    assert out["승"] + out["패"] <= 4
    assert out["p"] >= out["min_p"]


def test_subperiods_leaves_thin_windows_unscored(universe):
    found = detect(universe, {"p": _every_nth(300)})["p"]
    out = subperiods(found, universe, periods=8, min_events=10_000)
    assert out["표"]["연초과"].isna().all()
    assert out["승"] == out["패"] == 0
    assert out["p"] == 1.0
