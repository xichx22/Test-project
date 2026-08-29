"""차트 패턴 검증 랩 — 검출 · 채점 · 구간 분할을 한 곳에서.

왜 별도 모듈인가
----------------
패턴 검증은 세 단계가 늘 붙어 다닌다. 종목마다 패턴을 검출하고, 캘린더-타임
포트폴리오로 초과수익을 내고, 구간을 쪼개 방향이 유지되는지 본다. 이걸
매번 스크립트로 쓰면 문턱 계산(시험한 패턴 수에 따른 Šidák)을 빠뜨리기 쉽다.

이 모듈이 강제하는 것
---------------------
1. **문턱은 시험한 개수에서 나온다.** 패턴 11종을 동시에 재면 |t| 2.81 이
   필요하다. 이걸 함수가 계산해서 표에 박아 넣는다.
2. **표본이 얇으면 판정하지 않는다.** 발생 50회 미만은 "표본부족"으로
   표시하고 성과를 내지 않는다. "효과 없음"과 "못 쟀음"은 다른 결론이다.
3. **구간 분할의 최소 가능 p 를 같이 낸다.** 구간 4개면 전승해도 0.125 다.
   이걸 모르면 검정력 부족을 증거 부재로 오독한다.
"""

from __future__ import annotations

from math import comb, erfc, sqrt
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .eventstudy import calendar_time_portfolio

STOCK_ROUND_TRIP_BPS = 28.0
MIN_EVENTS = 50


def sidak_t(n_hypotheses: int, alpha: float = 0.05) -> float:
    """동시에 n개를 시험할 때 필요한 |t| (정규 근사, 양측).

    단일 가설의 t=2 감각을 그대로 쓰면 안 된다. 11종을 재면 문턱이 2.81,
    5,000개를 재면 4.42 가 된다.
    """
    if n_hypotheses < 1:
        raise ValueError("가설 수는 1 이상이어야 합니다")
    target = 1 - (1 - alpha) ** (1 / n_hypotheses)
    low, high = 0.0, 12.0
    for _ in range(200):
        mid = (low + high) / 2
        if erfc(mid / sqrt(2)) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def sign_test(wins: int, losses: int) -> float:
    """동점을 제외한 양측 부호검정."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n))


def min_sign_p(periods: int) -> float:
    """구간 `periods` 개에서 나올 수 있는 가장 작은 p.

    이 값이 0.05 보다 크면 **어떤 결과도 유의할 수 없다.** 구간 수를
    정할 때 먼저 확인해야 하는 값이다.
    """
    return sign_test(periods, 0)


def detect(
    data: Mapping[str, pd.DataFrame],
    detectors: Mapping[str, Callable[[pd.DataFrame], pd.Series]],
) -> dict[str, dict[str, pd.Series]]:
    """종목별로 각 패턴을 검출한다. 신호가 하나도 없는 종목은 뺀다."""
    out: dict[str, dict[str, pd.Series]] = {}
    for name, fn in detectors.items():
        found = {}
        for code, candles in data.items():
            series = fn(candles)
            if series.any():
                found[code] = series
        out[name] = found
    return out


def count_events(events: Mapping[str, pd.Series]) -> int:
    return sum(int(s.sum()) for s in events.values())


def score(
    events_by_pattern: Mapping[str, Mapping[str, pd.Series]],
    data: Mapping[str, pd.DataFrame],
    *,
    holding_days: int = 60,
    cost_bps: float = STOCK_ROUND_TRIP_BPS,
    min_events: int = MIN_EVENTS,
    labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """패턴별 초과수익과 t, 그리고 다중검정 통과 여부."""
    threshold = sidak_t(len(events_by_pattern))
    rows = []
    for name, events in events_by_pattern.items():
        total = count_events(events)
        row = {"패턴": name, "분류": (labels or {}).get(name, ""), "발생": total}
        if total < min_events:
            # 성과를 내지 않는다. "효과 없음"이 아니라 "못 쟀음"이다.
            rows.append({**row, "연초과": np.nan, "t": np.nan, "판정": "표본부족"})
            continue
        result = calendar_time_portfolio(events, data, holding_days=holding_days,
                                         cost_bps=cost_bps)
        rows.append({**row, "연초과": result.annualized, "t": result.t_stat,
                     "판정": "통과" if abs(result.t_stat) >= threshold else "미달"})
    frame = pd.DataFrame(rows).sort_values("t", ascending=False, na_position="last")
    frame.attrs["threshold"] = threshold
    frame.attrs["n_hypotheses"] = len(events_by_pattern)
    return frame.reset_index(drop=True)


def subperiods(
    events: Mapping[str, pd.Series],
    data: Mapping[str, pd.DataFrame],
    *,
    periods: int = 8,
    holding_days: int = 60,
    cost_bps: float = STOCK_ROUND_TRIP_BPS,
    min_events: int = 20,
) -> dict:
    """구간을 쪼개 방향이 유지되는지 본다.

    전 구간 하나의 t 는 한 시대의 결과일 수 있다. 구간마다 부호가 같아야
    "이 시장에서 꾸준히 그랬다"고 말할 수 있다. 반환에 `min_p` 를 같이
    담는다 — 그 값이 0.05 를 넘으면 애초에 유의할 수 없는 설계다.
    """
    index = sorted(set().union(*[d.index for d in data.values()]))
    segments = np.array_split(np.arange(len(index)), periods)
    rows, wins, losses = [], 0, 0
    for segment in segments:
        start, end = index[segment[0]], index[segment[-1]]
        window = {c: d[(d.index >= start) & (d.index <= end)] for c, d in data.items()}
        window = {c: d for c, d in window.items() if len(d) > 80}
        sliced = {c: events[c].reindex(window[c].index).fillna(False)
                  for c in window if c in events}
        sliced = {c: s for c, s in sliced.items() if s.any()}
        total = count_events(sliced)
        if total < min_events:
            rows.append({"구간": f"{start.date()}~{end.date()}", "발생": total,
                         "연초과": np.nan})
            continue
        result = calendar_time_portfolio(sliced, window, holding_days=holding_days,
                                         cost_bps=cost_bps)
        rows.append({"구간": f"{start.date()}~{end.date()}", "발생": total,
                     "연초과": result.annualized})
        if result.annualized > 0:
            wins += 1
        elif result.annualized < 0:
            losses += 1
    return {
        "표": pd.DataFrame(rows),
        "승": wins, "패": losses,
        "p": sign_test(wins, losses),
        "min_p": min_sign_p(periods),
    }
