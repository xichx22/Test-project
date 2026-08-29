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


def swing_portfolio(
    events_by_code,
    candles_by_code,
    *,
    holding_days: int = 60,
    cost_bps: float = 28.0,
    cash_rate: float = 0.02,
    max_positions: int | None = None,
    risk_free: float = 0.02,
):
    """이벤트 → **실제 투자 가능한** 자산가치 곡선.

    `calendar_time_portfolio` 는 초과수익(시장 대비)을 재므로 "이 신호에 알파가
    있는가"를 답한다. 반면 실제로 돈을 굴렸을 때의 꾸준함 — 롤링 12개월 양수율,
    최장 무회복 기간 — 을 재려면 **원시 수익률**로 된 자산가치 곡선이 필요하다.

    규칙
      - 신호 봉 다음 날부터 holding_days 동안 보유 (다른 곳과 같은 체결 가정)
      - 보유 종목은 동일가중, 신호가 없는 날은 전액 현금
      - 매수·매도 시 왕복 비용을 진입일에 한 번에 차감 (보수적)
      - max_positions 를 주면 동시 보유를 제한한다 (자금 한도 반영)

    `allocation.BacktestResult` 를 돌려주므로 자산배분 전략과 같은 잣대로 비교된다.
    """
    from .allocation import BacktestResult

    returns, held, entries = {}, {}, {}
    for code, candles in candles_by_code.items():
        series = events_by_code.get(code)
        if series is None or not series.any():
            continue
        returns[code] = candles["close"].pct_change()
        entered = (
            series.reindex(candles.index).fillna(False).astype(bool)
            .shift(1, fill_value=False).astype(bool)
        )
        entries[code] = entered
        held[code] = entered.rolling(holding_days, min_periods=1).max().astype(bool)

    if not returns:
        empty = pd.Series(dtype=float)
        return BacktestResult("스윙(신호없음)", empty, empty, empty, 0, risk_free=risk_free)

    ret = pd.DataFrame(returns).sort_index()
    hold = pd.DataFrame(held).reindex(index=ret.index, columns=ret.columns).fillna(False).astype(bool)
    ent = pd.DataFrame(entries).reindex(index=ret.index, columns=ret.columns).fillna(False).astype(bool)

    if max_positions:
        # 한도를 넘으면 먼저 진입한 종목을 우선한다 (컬럼 순서로 결정론적)
        rank = hold.cumsum(axis=1)
        hold = hold & (rank <= max_positions)

    count = hold.sum(axis=1)
    invested = count > 0
    # 보유 종목 동일가중 수익. 보유가 없으면 현금.
    gross = (ret.where(hold).sum(axis=1) / count.replace(0, np.nan)).fillna(0.0)
    cash_daily = (1 + cash_rate) ** (1 / 252) - 1
    daily = np.where(invested, gross, cash_daily)

    # 진입 비중만큼 왕복 비용을 그날 차감
    new_share = (ent & hold).sum(axis=1) / count.replace(0, np.nan)
    daily = daily - new_share.fillna(0.0).to_numpy() * (cost_bps / 10_000.0)

    series = pd.Series(daily, index=ret.index)
    equity = (1 + series).cumprod()
    weight = invested.astype(float)
    trades = int(ent.sum().sum())
    name = f"스윙 {holding_days}일보유"
    return BacktestResult(name, equity, weight, series, trades, risk_free=risk_free)


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
