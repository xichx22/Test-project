"""캘린더-타임 포트폴리오 — 겹치는 이벤트를 데이터 손실 없이 다루는 표준 방법.

문제
----
"신호 발생 후 60일 수익률"을 이벤트마다 계산하면 창이 서로 겹친다. 겹침을
피하려고 이벤트를 60일씩 띄워 고르면 표본의 대부분을 버리게 된다
(실측: 이벤트 1,920개 → 유효 관측 약 20개).

해결
----
관점을 뒤집는다. 이벤트별 수익률이 아니라 **매일의 포트폴리오 수익률**을 본다.

    매일: 지난 h일 안에 신호가 뜬 종목을 동일가중으로 담는다
          그날의 포트폴리오 초과수익 = 담긴 종목들의 (수익률 − 그날 전 종목 평균) 평균

이렇게 만든 일별 계열은 **겹치지 않는다** — 하루는 한 번만 세어진다.
그래서 평범한 t검정이 그대로 유효하고, 표본도 버리지 않는다.
덤으로 실제 운용에 가깝다: "신호가 뜨면 사서 h일 들고 있는다"를 그대로 옮긴 것.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class CalendarTimeResult:
    daily: pd.Series            # 일별 포트폴리오 초과수익
    positions: pd.Series        # 일별 보유 종목 수
    holding_days: int
    n_events: int

    @property
    def mean_daily(self) -> float:
        return float(self.daily.mean())

    @property
    def t_stat(self) -> float:
        """자기상관 보정 t. 포트폴리오가 매일 거의 같은 종목을 들고 있으므로
        평범한 t 는 부풀려진다 — Newey-West 로 잡는다."""
        return metrics.newey_west_t_stat(self.daily.to_numpy(), lags=self.holding_days)

    @property
    def t_stat_naive(self) -> float:
        return metrics.t_stat(self.daily)

    @property
    def annualized(self) -> float:
        """연 환산 초과수익 (252영업일 복리)."""
        return float((1 + self.daily.mean()) ** 252 - 1)

    @property
    def hit_rate(self) -> float:
        return float((self.daily > 0).mean())

    def summary(self) -> dict[str, float]:
        return {
            "이벤트": self.n_events,
            "보유일": self.holding_days,
            "포지션있는날": int((self.positions > 0).sum()),
            "평균보유종목": float(self.positions[self.positions > 0].mean())
            if (self.positions > 0).any() else np.nan,
            "일평균초과수익%": self.mean_daily * 100,
            "연환산%": self.annualized * 100,
            "t_순진": self.t_stat_naive,
            "t": self.t_stat,
            "양의날비율": self.hit_rate,
        }


def calendar_time_portfolio(
    events_by_code: Mapping[str, pd.Series],
    candles_by_code: Mapping[str, pd.DataFrame],
    *,
    holding_days: int = 20,
    cost_bps: float = 0.0,
) -> CalendarTimeResult:
    """이벤트 → 캘린더-타임 포트폴리오.

    체결 가정은 다른 곳과 같다. 신호는 t 봉 종가에 확정되므로 t+1 봉부터
    holding_days 동안 보유한다.

    cost_bps 를 주면 진입일에 왕복 비용을 한 번에 차감한다 (보수적).
    """
    # 벤치마크(그날 전 종목 평균)는 **유니버스 전체**로 계산해야 한다.
    # 이벤트가 있는 종목만으로 평균을 내면 벤치마크가 표본과 같이 움직여
    # 초과수익이 왜곡된다 (실측: 전체기간 +12.4%/년인데 부분기간은 전부 음수라는
    # 모순이 났고, 원인이 이것이었다).
    daily_returns: dict[str, pd.Series] = {}
    holding_flags: dict[str, pd.Series] = {}
    entry_flags: dict[str, pd.Series] = {}
    n_events = 0

    for code, candles in candles_by_code.items():
        daily_returns[code] = candles["close"].pct_change()
        events = events_by_code.get(code)
        if events is None or not events.any():
            continue
        n_events += int(events.sum())
        # 신호는 t 봉 종가에 확정 → t+1 부터 보유하므로 한 칸 민다.
        # bool 로 먼저 맞춘 뒤 shift(fill_value=) 로 민다. object dtype 을 거치면
        # pandas 가 downcasting 경고를 봉마다 뱉고, 그 출력이 실행을 붙잡는다
        # (실측: 로그 3.4MB, 리포트 생성이 30분 넘게 늘어졌다).
        entered = (
            events.reindex(candles.index).fillna(False).astype(bool)
            .shift(1, fill_value=False).astype(bool)
        )
        entry_flags[code] = entered
        holding_flags[code] = (
            entered.rolling(holding_days, min_periods=1).max().astype(bool)
        )

    if not daily_returns or not holding_flags:
        empty = pd.Series(dtype=float)
        return CalendarTimeResult(empty, empty, holding_days, 0)

    returns = pd.DataFrame(daily_returns)
    held = pd.DataFrame(holding_flags).reindex(
        index=returns.index, columns=returns.columns
    ).fillna(False).astype(bool)
    entries = pd.DataFrame(entry_flags).reindex(
        index=returns.index, columns=returns.columns
    ).fillna(False).astype(bool)

    # 초과수익 = 그날 전 종목 평균 대비. 시장 움직임을 제거한다.
    excess = returns.sub(returns.mean(axis=1), axis=0)
    if cost_bps:
        excess = excess - entries * (cost_bps / 10_000.0)

    selected = excess.where(held)
    portfolio = selected.mean(axis=1)
    positions = held.sum(axis=1)

    portfolio = portfolio[positions > 0].dropna()
    return CalendarTimeResult(portfolio, positions, holding_days, n_events)


def compare_holdings(
    events_by_code: Mapping[str, pd.Series],
    candles_by_code: Mapping[str, pd.DataFrame],
    *,
    holdings: tuple[int, ...] = (5, 10, 20, 60),
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for days in holdings:
        result = calendar_time_portfolio(
            events_by_code, candles_by_code, holding_days=days, cost_bps=cost_bps
        )
        if result.daily.empty:
            continue
        rows.append(result.summary())
    return pd.DataFrame(rows).set_index("보유일")
