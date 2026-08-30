"""손실 차단기 — 오늘 새 매매를 해도 되는가.

원본과 무엇이 다른가
--------------------
tradermonty/claude-trading-skills 의 drawdown-circuit-breaker 는 기본값으로
"연속 2패 → 쿨다운", "일 손실 2% → 정지" 를 쓴다. 그 값을 이 전략에 그대로
적용하면 21년 실측에서:

    연속 2패 쿨다운   45번 발동  ← 사실상 영구 정지
    일 손실 2% 정지  113일 발동  (전체 거래일의 2.1%)

이 규칙은 승률 34%, 최장 연속 손실 32회다. **연속 손실은 고장이 아니라 설계**다.
그래서 연속 패배 문턱은 아예 쓰지 않고, 나머지 문턱은 실측 분포의 꼬리에서
유도했다. 아래 기본값은 21년 동안 각각 몇 번 발동하는지 세어 정한 것이다.

    일간   하위 0.1% 가 −3.98%, 최악 −8.09%  →  −5%
    주간   하위 1% 가 −5.14%, 최악 −10.18%   →  −8%
    월간   하위 1% 가 −7.73%, 최악 −11.13%   →  −12%
    12개월 최악 −23.5%                        →  −25%  (계획서의 중단 규칙)

문턱을 정하는 원칙: **정상 범위의 나쁜 구간에서는 울리지 않고, 백테스트가
설명하지 못하는 구간에서만 울린다.** 자주 울리는 차단기는 전략을 죽인다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from .ledger import Fill

ALLOWED, COOLDOWN, HALTED = "매매가능", "쿨다운", "중단"


@dataclass(frozen=True)
class Thresholds:
    """전부 계좌 대비 손실률(양수로 적는다)."""

    daily: float = 0.05
    weekly: float = 0.08
    monthly: float = 0.12
    yearly: float = 0.25          # 계획서의 스윙 중단 규칙과 같은 값
    max_rule_breaks: int = 2      # 규칙 위반 2회면 중단 (성과와 무관)


@dataclass
class GuardResult:
    state: str
    reasons: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def can_trade(self) -> bool:
        return self.state == ALLOWED

    def summary(self) -> str:
        if self.can_trade:
            return "매매 가능"
        return f"{self.state} — " + "; ".join(self.reasons)


def _pnl_since(fills: list[Fill], start: date, end: date) -> float:
    total = 0.0
    for f in fills:
        if not f.exit_date:
            continue
        try:
            d = date.fromisoformat(f.exit_date)
        except ValueError:
            continue
        if start <= d <= end and f.pnl is not None:
            total += f.pnl
    return total


def evaluate_guard(
    fills: list[Fill],
    account: float,
    *,
    as_of: date | str | None = None,
    thresholds: Thresholds = Thresholds(),
) -> GuardResult:
    """기간별 실현손익과 규칙 위반을 보고 상태를 낸다.

    평가에는 **실현손익만** 쓴다. 평가손익으로 차단기를 돌리면 보유 중에
    흔들릴 때마다 울려서, 버티는 것이 전부인 규칙과 정면으로 충돌한다.
    """
    if account <= 0:
        raise ValueError("계좌 금액은 0보다 커야 합니다")
    today = date.fromisoformat(as_of) if isinstance(as_of, str) else (as_of or date.today())

    windows = {
        "일간": (today, thresholds.daily),
        "주간": (today - timedelta(days=today.weekday()), thresholds.weekly),
        "월간": (today.replace(day=1), thresholds.monthly),
        "12개월": (today - timedelta(days=365), thresholds.yearly),
    }
    reasons, detail, state = [], {}, ALLOWED
    for label, (start, limit) in windows.items():
        pnl = _pnl_since(fills, start, today)
        ratio = pnl / account
        detail[label] = ratio
        if ratio <= -limit:
            reasons.append(f"{label} 손실 {ratio:.1%} (문턱 −{limit:.0%})")
            state = HALTED

    breaks = sum(len(f.rule_breaks) for f in fills)
    detail["규칙위반"] = breaks
    if breaks >= thresholds.max_rule_breaks:
        reasons.append(f"규칙 위반 {breaks}회 (문턱 {thresholds.max_rule_breaks}회)")
        state = HALTED

    return GuardResult(state=state, reasons=reasons, detail=detail)
