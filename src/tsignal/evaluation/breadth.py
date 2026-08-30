"""신호가 **몇 건 났는지**를 신호로 쓴다.

발견 경위
---------
RSI 과매도 반등은 5년 내내 기준선을 이겼고 비용·유동성·틱 관문을 다 통과했다.
그런데 날짜를 관측치 1개로 세자 무너졌다 — 신호 16,318건이 1,125일에 몰려
있었고, **상위 20일이 전체 이익의 100%** 였다. 상위 20일을 빼면 정확히 0.00%,
겹침 보정 t 값은 0.34.

그 20일은 전부 시장이 무너진 날이었다 (2020-03 코로나 바닥, 2024-12-10,
2025-04 관세 충격). 즉 이것은 종목을 고르는 신호가 아니라 **날을 고르는
신호**다. 그리고 "그날 과매도 반등이 몇 건 났는가" 는 장 마감 시점에 셀 수
있으므로 미래참조가 아니다.

실측 결론
---------
문턱을 올릴수록 성적이 단조롭게 좋아지고(10건 +1.37% → 150건 +10.93%),
학습(2020~22) → 검증(2023~25) 을 통과했으며, 2020년을 빼도 남았다.
**하지만 연 4~13회뿐이라 계좌 수익으로는 지수 매수보유에 크게 진다.**
현금 몫 운용법으로 덧붙여도 코로나를 빼면 개선이 0.1%p 안쪽이었다.

그래서 이 모듈은 "쓰라" 가 아니라 **"이런 것도 재 봤고 이렇게 졌다"** 의 기록이다.
날짜 쏠림 검사(`date_clustered`)는 다른 신호에도 그대로 쓸 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterReport:
    """날짜를 관측치 1개로 셌을 때의 성적."""

    days: int
    trades: int
    mean: float
    median: float
    win_rate: float
    t_stat: float
    top_share: dict[int, float]

    def summary(self) -> str:
        lines = [
            f"신호 난 날 {self.days:,}일, 총 {self.trades:,}건",
            f"날짜 평균 {self.mean:+.2%}  중앙값 {self.median:+.2%}  "
            f"이익 난 날 {self.win_rate:.1%}",
            f"겹침 보정 t = {self.t_stat:.2f}",
        ]
        for k, share in sorted(self.top_share.items()):
            lines.append(f"  상위 {k:3d}일이 전체 이익의 {share:6.1%}")
        return "\n".join(lines)


def daily_counts(events: pd.DataFrame) -> pd.Series:
    """날짜별 신호 건수. `events` 는 날짜 인덱스를 가진 신호 표."""
    return events.groupby(events.index).size()


def date_clustered(
    events: pd.DataFrame,
    *,
    column: str = "fwd20",
    cost: float = 0.0028,
    lag: int = 20,
    tops: tuple[int, ...] = (5, 10, 20, 50),
) -> ClusterReport:
    """같은 날 신호를 하나로 묶고, 날짜 단위로 성적을 다시 잰다.

    신호가 날짜에 몰리면 건수는 부풀지만 독립 정보는 늘지 않는다. 겹치는
    보유구간 때문에 자기상관이 생기므로 t 값은 Newey-West 로 보정한다.
    """
    net = events[column] - cost
    day = net.groupby(events.index).mean()
    size = events.groupby(events.index).size()
    x = day.to_numpy(float)
    n = len(x)
    if n < 2:
        return ClusterReport(n, int(size.sum()), float("nan"), float("nan"),
                             float("nan"), float("nan"), {})

    mu = float(x.mean())
    err = x - mu
    var = float(err @ err) / n
    for l in range(1, lag + 1):
        if l >= n:
            break
        var += 2 * (1 - l / (lag + 1)) * float(err[l:] @ err[:-l]) / n
    se = np.sqrt(max(var, 1e-12) / n)

    contribution = (size * day).sort_values(ascending=False)
    total = float(contribution.sum())
    share = {k: float(contribution.head(k).sum() / total) if total else float("nan")
             for k in tops if k <= n}
    return ClusterReport(
        days=n, trades=int(size.sum()), mean=mu, median=float(np.median(x)),
        win_rate=float((x > 0).mean()), t_stat=float(mu / se) if se else float("nan"),
        top_share=share,
    )


def threshold_table(
    events: pd.DataFrame,
    *,
    thresholds: tuple[int, ...] = (10, 20, 30, 50, 80, 100, 150),
    columns: tuple[str, ...] = ("fwd10", "fwd20", "fwd60"),
    cost: float = 0.0028,
) -> pd.DataFrame:
    """'하루 신호가 N건 이상 난 날' 만 샀다면.

    건수는 그날 종가 시점에 세므로 미래참조가 아니다. 진입은 다음 봉 시가.
    """
    grouped = events.groupby(events.index)
    day = pd.DataFrame({"건수": grouped.size()})
    for col in columns:
        if col in events.columns:
            day[col] = grouped[col].mean() - cost
    span_years = max((day.index[-1] - day.index[0]).days / 365.25, 1e-9)

    rows = []
    for th in thresholds:
        hit = day[day["건수"] >= th]
        if len(hit) < 5:
            continue
        row = {"문턱": th, "해당 날": len(hit), "연 평균 횟수": len(hit) / span_years}
        for col in columns:
            if col in hit:
                row[col] = float(hit[col].mean())
        main = columns[1] if len(columns) > 1 else columns[0]
        if main in hit:
            row["이익난날"] = float((hit[main] > 0).mean())
            row["최악"] = float(hit[main].min())
        rows.append(row)
    return pd.DataFrame(rows)


def split_check(
    events: pd.DataFrame,
    *,
    threshold: int,
    column: str = "fwd20",
    cost: float = 0.0028,
    split_year: int = 2023,
) -> pd.DataFrame:
    """학습 기간에서 고른 문턱을 검증 기간에 그대로 쓰면.

    이 프로젝트에서 신호가 무너진 자리는 거의 언제나 여기였다.
    """
    grouped = events.groupby(events.index)
    day = pd.DataFrame({"건수": grouped.size(), "수익": grouped[column].mean() - cost})
    hit = day[day["건수"] >= threshold]
    rows = []
    for label, sub in (("학습", hit[hit.index.year < split_year]),
                       ("검증", hit[hit.index.year >= split_year])):
        rows.append({"구간": label, "날 수": len(sub),
                     "평균": float(sub["수익"].mean()) if len(sub) else np.nan,
                     "이익난날": float((sub["수익"] > 0).mean()) if len(sub) else np.nan})
    return pd.DataFrame(rows)
