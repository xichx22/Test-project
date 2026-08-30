"""신호가 날짜에 몰리면 건수는 부풀지만 정보는 늘지 않는다."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.breadth import (
    ClusterReport, daily_counts, date_clustered, split_check, threshold_table,
)


def _events(rows):
    """rows: (날짜, fwd20) 목록."""
    index = pd.DatetimeIndex([pd.Timestamp(d, tz="Asia/Seoul") for d, _ in rows])
    return pd.DataFrame({"fwd20": [v for _, v in rows]}, index=index)


def test_one_huge_day_does_not_make_an_edge():
    """하루에 몰린 대박이 건수 가중 평균은 올려도, 날짜 평균은 못 올린다."""
    rows = [("2020-03-20", 0.40)] * 200
    rows += [(f"2021-01-{d:02d}", -0.01) for d in range(1, 29)]
    events = _events(rows)
    naive = float((events["fwd20"] - 0.0028).mean())
    report = date_clustered(events, cost=0.0028, lag=5)
    assert naive > 0.30                      # 건수로 세면 큰 이익
    assert report.mean < 0.02                # 날짜로 세면 사라진다
    assert report.days == 29
    assert report.trades == 228


def test_top_share_reports_concentration():
    rows = [("2020-03-20", 0.50)] * 50
    rows += [(f"2021-02-{d:02d}", 0.0) for d in range(1, 20)]
    report = date_clustered(_events(rows), cost=0.0, lag=3, tops=(1,))
    assert report.top_share[1] == pytest.approx(1.0, abs=1e-6)


def test_t_stat_shrinks_when_returns_are_autocorrelated():
    """겹치는 보유구간을 무시하면 t 값이 부풀려진다."""
    n = 200
    index = pd.date_range("2020-01-01", periods=n, freq="B", tz="Asia/Seoul")
    base = np.random.default_rng(0).normal(0.01, 0.01, n)
    smooth = pd.Series(base).rolling(20, min_periods=1).mean().to_numpy()
    events = pd.DataFrame({"fwd20": smooth}, index=index)
    naive_t = smooth.mean() / (smooth.std(ddof=1) / np.sqrt(n))
    report = date_clustered(events, cost=0.0, lag=20)
    assert report.t_stat < naive_t


def test_daily_counts_counts_rows_per_day():
    events = _events([("2020-01-02", 0.0), ("2020-01-02", 0.1), ("2020-01-03", 0.0)])
    counts = daily_counts(events)
    assert counts.iloc[0] == 2 and counts.iloc[1] == 1


def test_threshold_table_only_keeps_days_over_the_bar():
    rows = [(f"2020-01-{d:02d}", 0.05) for d in range(1, 11) for _ in range(60)]
    rows += [(f"2020-02-{d:02d}", -0.05) for d in range(1, 11)]
    table = threshold_table(_events(rows), thresholds=(1, 50), columns=("fwd20",))
    low = table[table["문턱"] == 1].iloc[0]
    high = table[table["문턱"] == 50].iloc[0]
    assert low["해당 날"] == 20
    assert high["해당 날"] == 10
    assert high["fwd20"] > low["fwd20"]


def test_split_check_separates_train_and_test():
    rows = [(f"2021-01-{d:02d}", 0.05) for d in range(1, 6) for _ in range(20)]
    rows += [(f"2024-01-{d:02d}", -0.05) for d in range(1, 6) for _ in range(20)]
    out = split_check(_events(rows), threshold=10, split_year=2023)
    train = out[out["구간"] == "학습"].iloc[0]
    test = out[out["구간"] == "검증"].iloc[0]
    assert train["평균"] > 0 > test["평균"]
    assert train["날 수"] == test["날 수"] == 5


def test_report_summary_is_printable():
    report = date_clustered(_events([("2020-01-02", 0.01), ("2020-01-03", 0.02)]),
                            cost=0.0, lag=1, tops=(1,))
    text = report.summary()
    assert "신호 난 날" in text and "t =" in text
    assert isinstance(report, ClusterReport)
