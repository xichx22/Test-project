"""주문 수량 계산 — 자리 수와 손절 폭, 두 가지로 각각 재고 작은 쪽을 쓴다.

왜 두 번 재는가
---------------
자리 기준만 쓰면 "계좌의 10%" 가 되는데, 손절 폭이 넓은 종목에서는 그 10% 가
계좌 전체에 큰 손실을 낼 수 있다. 손절 기준만 쓰면 손절이 아주 좁은 종목에서
자리보다 큰 금액이 나온다. **둘 다 재고 작은 쪽을 쓴다.**

한국 시장 규약
--------------
- 소수점 매매가 안 되므로 정수 주로 내린다
- 호가 단위로 손절가를 맞춘다 (그 사이 가격에는 주문이 안 들어간다)
- 왕복 비용 28bp(수수료 1.5bp×2 + 증권거래세 15bp + 슬리피지 5bp×2)를 반영한다
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..ohlcv import tick_size

STOCK_ROUND_TRIP_BPS = 28.0


@dataclass(frozen=True)
class SizingResult:
    shares: int
    price: float
    stop_price: float
    notional: float
    risk_amount: float          # 손절까지 갔을 때 잃는 금액
    risk_pct_of_account: float
    binding: str                # "자리" 또는 "손절" — 무엇이 수량을 정했는가
    cost_estimate: float

    def summary(self) -> str:
        return (
            f"{self.shares:,}주 × {self.price:,.0f}원 = {self.notional:,.0f}원 "
            f"(계좌의 {self.notional and self.risk_pct_of_account:.2%} 위험, "
            f"{self.binding} 기준)"
        )


def round_to_tick(price: float, *, mode: str = "down") -> float:
    """호가 단위에 맞춰 가격을 내림/올림한다.

    손절가는 **내림**이 맞다. 호가 사이 가격으로 주문하면 체결되지 않고,
    올리면 의도보다 일찍 잘린다.
    """
    tick = float(tick_size(pd.Series([price])).iloc[0])
    if not math.isfinite(tick) or tick <= 0:
        return float(price)
    q = price / tick
    return float((math.floor(q) if mode == "down" else math.ceil(q)) * tick)


def size_order(
    account: float,
    price: float,
    *,
    max_positions: int = 10,
    stop_loss: float = 0.08,
    max_risk_pct: float = 0.02,
    one_way_bps: float = STOCK_ROUND_TRIP_BPS / 2,
) -> SizingResult:
    """한 종목의 주문 수량.

    `max_risk_pct` 는 "이 한 건으로 계좌의 몇 %까지 잃어도 되는가"다.
    자리 10개 × 손절 8% 면 한 건의 위험이 계좌의 0.8% 이므로 기본값 2% 는
    여유가 있다. 손절이 넓은 종목에서만 이쪽이 구속력을 갖는다.
    """
    if account <= 0 or price <= 0:
        raise ValueError("계좌 금액과 가격은 0보다 커야 합니다")
    if not 0 < stop_loss < 1:
        raise ValueError("손절 폭은 0과 1 사이여야 합니다")
    if max_positions < 1:
        raise ValueError("자리 수는 1 이상이어야 합니다")

    stop_price = round_to_tick(price * (1 - stop_loss), mode="down")
    per_share_risk = price - stop_price
    if per_share_risk <= 0:
        raise ValueError("손절가가 진입가 이상입니다 — 호가 단위를 확인하세요")

    by_slot = int((account / max_positions) // price)
    by_risk = int((account * max_risk_pct) // per_share_risk)
    shares = max(0, min(by_slot, by_risk))
    binding = "자리" if by_slot <= by_risk else "손절"

    notional = shares * price
    risk = shares * per_share_risk
    cost = notional * (one_way_bps / 10_000.0) * 2
    return SizingResult(
        shares=shares, price=float(price), stop_price=stop_price,
        notional=notional, risk_amount=risk,
        risk_pct_of_account=risk / account if account else 0.0,
        binding=binding, cost_estimate=cost,
    )
