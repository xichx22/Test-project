"""신호 하나를 **실제로 굴릴 수 있는 규칙**으로 바꾼다.

왜 필요한가
-----------
"이 패턴에 사면 된다"는 아직 매매 방법이 아니다. 실측(플랫베이스, 1,064종목):
신호가 연 239회 뜨고, 60일 보유하면 동시 보유가 **중앙값 46종목, 최대 280종목**
이 된다. 개인이 못 한다. 자리 수를 제한하는 순간 두 질문이 새로 생긴다 —
**자리가 모자라면 무엇을 버릴 것인가**, 그리고 **언제 팔 것인가**.

이 모듈은 그 둘을 명시적인 파라미터로 만들고, 체결을 한 건씩 따라가며
시뮬레이션한다. 벡터 연산으로는 손절을 정확히 재현할 수 없다 —
"장중 저가가 손절가를 건드렸는가"는 봉 단위로 봐야 한다.

체결 가정 (이 프로젝트 전체와 동일)
-----------------------------------
- 신호는 t봉 종가에 확정, 매수는 t+1봉 **시가**
- 손절은 장중 저가가 손절가에 닿으면 그 가격에 체결.
  단, 시가가 이미 손절가 아래면 **시가**에 체결한다 (갭하락에서 손절가로
  나갈 수 있다고 가정하면 성과가 부풀려진다)
- 시간 청산은 보유 만기 봉의 **종가**
- 비용은 진입·청산 각각에 편도로 부과
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .allocation import BacktestResult

STOCK_ONE_WAY_BPS = 14.0    # 왕복 28bp 의 절반


@dataclass(frozen=True)
class Plan:
    """운용 규칙. 전부 사람이 지킬 수 있는 값이어야 한다."""

    max_positions: int = 10
    holding_days: int = 60
    stop_loss: float | None = 0.08      # 진입가 대비 하락률. None 이면 손절 없음
    take_profit: float | None = None
    trail: float | None = None          # 고점 대비 하락률로 따라가는 손절
    rank_by: str = "liquidity"          # 자리보다 후보가 많을 때 무엇을 먼저 살 것인가
    one_way_bps: float = STOCK_ONE_WAY_BPS
    cash_rate: float = 0.02
    min_turnover: float = 3e8           # 일 거래대금 하한 (3억). 못 사는 종목을 뺀다
    exit_on_gate_off: bool = False      # 시장 게이트가 닫히면 보유분도 전부 청산


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry: float
    exit: float
    bars: int
    reason: str                          # stop / target / trail / time

    @property
    def gross(self) -> float:
        return self.exit / self.entry - 1


def _rank_key(rule: str, frame: pd.DataFrame, row: int) -> float:
    """자리 경쟁에서 우선순위. 값이 **클수록** 먼저 산다."""
    if rule == "liquidity":
        # 거래대금이 큰 종목부터. 슬리피지 가정이 가장 잘 성립하는 쪽이다.
        window = frame["close"].to_numpy()[max(0, row - 20):row]
        volume = frame["volume"].to_numpy()[max(0, row - 20):row]
        return float(np.median(window * volume)) if len(window) else 0.0
    if rule == "tight":
        # 베이스가 좁을수록 먼저. 오닐이 말한 '조여진' 베이스에 가깝다.
        high = frame["high"].to_numpy()[max(0, row - 40):row]
        low = frame["low"].to_numpy()[max(0, row - 40):row]
        if len(high) == 0 or high.max() <= 0:
            return 0.0
        return float(-(high.max() - low.min()) / high.max())
    raise ValueError(f"알 수 없는 rank_by: {rule}")


def run_plan(
    events: Mapping[str, pd.Series],
    candles: Mapping[str, pd.DataFrame],
    plan: Plan = Plan(),
    gate: pd.Series | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    """규칙대로 한 건씩 체결해 자산곡선과 체결 내역을 낸다.

    `gate` 는 "오늘 새로 사도 되는가"를 날짜별로 담은 boolean Series 다
    (예: 지수가 200일선 위). `plan.exit_on_gate_off` 를 켜면 게이트가
    닫히는 날 보유분도 전부 판다 — 진입만 막으면 이미 산 것이 하락장을
    그대로 맞기 때문이다.
    """
    index = pd.DatetimeIndex(sorted(set().union(*[d.index for d in candles.values()])))
    pos_of = {ts: i for i, ts in enumerate(index)}
    cost = plan.one_way_bps / 10_000.0

    # 날짜별 진입 후보 (신호 다음 봉에 체결하므로 하루 밀어 둔다)
    pending: dict[pd.Timestamp, list[tuple[float, str]]] = {}
    for code, series in events.items():
        frame = candles[code]
        rows = {ts: i for i, ts in enumerate(frame.index)}
        for stamp in series[series].index:
            row = rows.get(stamp)
            if row is None or row + 1 >= len(frame):
                continue
            fill_at = frame.index[row + 1]
            score = _rank_key(plan.rank_by, frame, row + 1)
            liquidity = float(np.median(
                (frame["close"] * frame["volume"]).to_numpy()[max(0, row - 20):row + 1]))
            if liquidity < plan.min_turnover:
                continue
            pending.setdefault(fill_at, []).append((score, code))

    open_slots: dict[str, dict] = {}
    trades: list[Trade] = []
    equity = 1.0
    cash = 1.0
    daily: list[float] = []
    cash_daily = (1 + plan.cash_rate) ** (1 / 252) - 1

    for stamp in index:
        start_equity = equity
        # --- 청산 먼저 (자리를 비워야 새로 산다) ---
        for code in list(open_slots):
            slot = open_slots[code]
            frame = candles[code]
            if stamp not in frame.index:
                continue
            bar = frame.loc[stamp]
            slot["bars"] += 1
            slot["peak"] = max(slot["peak"], float(bar["high"]))

            exit_price, reason = None, ""
            stop = slot["stop"]
            if plan.trail is not None:
                stop = max(stop or 0.0, slot["peak"] * (1 - plan.trail))
            if stop is not None and float(bar["low"]) <= stop:
                # 갭하락이면 손절가가 아니라 시가에 나간다 (보수적)
                exit_price = min(float(bar["open"]), stop)
                reason = "trail" if plan.trail is not None else "stop"
            elif plan.take_profit is not None and \
                    float(bar["high"]) >= slot["entry"] * (1 + plan.take_profit):
                exit_price = slot["entry"] * (1 + plan.take_profit)
                reason = "target"
            elif slot["bars"] >= plan.holding_days:
                exit_price = float(bar["close"])
                reason = "time"

            if exit_price is not None:
                proceeds = slot["shares"] * exit_price * (1 - cost)
                cash += proceeds
                trades.append(Trade(code, slot["entry_date"], stamp, slot["entry"],
                                    exit_price, slot["bars"], reason))
                del open_slots[code]

        # --- 시장 게이트가 닫히면 전량 청산 ---
        gate_open = True if gate is None else bool(gate.get(stamp, False))
        if plan.exit_on_gate_off and not gate_open and open_slots:
            for code in list(open_slots):
                slot = open_slots[code]
                frame = candles[code]
                if stamp not in frame.index:
                    continue
                price = float(frame.loc[stamp, "close"])
                cash += slot["shares"] * price * (1 - cost)
                trades.append(Trade(code, slot["entry_date"], stamp, slot["entry"],
                                    price, slot["bars"], "gate"))
                del open_slots[code]

        # --- 진입 ---
        candidates = [] if not gate_open else sorted(pending.get(stamp, []), reverse=True)
        for _, code in candidates:
            if len(open_slots) >= plan.max_positions or code in open_slots:
                continue
            frame = candles[code]
            if stamp not in frame.index:
                continue
            price = float(frame.loc[stamp, "open"])
            if price <= 0:
                continue
            budget = equity / plan.max_positions
            if cash < budget:
                continue
            shares = budget * (1 - cost) / price
            cash -= budget
            open_slots[code] = {
                "entry": price, "entry_date": stamp, "shares": shares,
                "bars": 0, "peak": price,
                "stop": price * (1 - plan.stop_loss) if plan.stop_loss else None,
            }

        # --- 평가 ---
        cash *= 1 + cash_daily
        holdings = 0.0
        for code, slot in open_slots.items():
            frame = candles[code]
            if stamp in frame.index:
                slot["last"] = float(frame.loc[stamp, "close"])
            holdings += slot["shares"] * slot.get("last", slot["entry"])
        equity = cash + holdings
        daily.append(equity / start_equity - 1 if start_equity > 0 else 0.0)

    # 끝까지 안 팔린 포지션도 기록한다. 빼놓으면 손실을 계속 들고 있는 규칙이
    # 승률이 좋아 보인다 — 진 거래만 로그에서 사라지기 때문이다.
    for code, slot in open_slots.items():
        last = slot.get("last", slot["entry"])
        trades.append(Trade(code, slot["entry_date"], index[-1], slot["entry"],
                            last, slot["bars"], "open"))

    series = pd.Series(daily, index=index)
    curve = (1 + series).cumprod()
    exposure = pd.Series(
        [1.0 if v != 0 else 0.0 for v in series], index=index)
    result = BacktestResult(
        f"플랫베이스 {plan.max_positions}자리 {plan.holding_days}일",
        curve, exposure, series, len(trades),
    )
    log = pd.DataFrame([{
        "code": t.code, "entry": t.entry_date, "exit": t.exit_date,
        "bars": t.bars, "reason": t.reason, "수익률": t.gross,
    } for t in trades])
    return result, log


def summarize(log: pd.DataFrame) -> dict:
    """체결 내역 요약 — 승률과 손익비는 규칙을 지킬 수 있는지와 직결된다."""
    if log.empty:
        return {}
    wins = log[log["수익률"] > 0]["수익률"]
    losses = log[log["수익률"] <= 0]["수익률"]
    return {
        "체결": len(log),
        "승률": len(wins) / len(log),
        "평균수익": float(wins.mean()) if len(wins) else 0.0,
        "평균손실": float(losses.mean()) if len(losses) else 0.0,
        "손익비": abs(float(wins.mean()) / float(losses.mean()))
                 if len(wins) and len(losses) and losses.mean() else np.nan,
        "평균보유": float(log["bars"].mean()),
        "청산사유": log["reason"].value_counts().to_dict(),
    }
