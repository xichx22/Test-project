"""진입 전 규율 게이트 — 이 주문을 내도 되는가.

이 게이트는 종목을 고르지 않는다. **주문을 내기 전에 스스로에게 묻는 질문**을
기계가 대신 물어 주는 것뿐이다. 답이 하나라도 아니오면 주문하지 않는다.

확인할 수 없으면 통과가 아니라 차단(fail-closed)이다. 모의투자 환경 없이
실계좌에 주문을 내는 상황에서는, 모르는 것을 통과시키는 쪽이 훨씬 비싸다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .guard import GuardResult

PASS, BLOCK, REVIEW = "통과", "차단", "확인필요"


@dataclass
class Answers:
    """주문 직전에 채우는 항목. 전부 사람이 답한다."""

    written_plan: bool | None = None      # 계획에 적어 둔 진입인가
    stop_defined: bool | None = None      # 손절가를 정했는가
    size_within_plan: bool | None = None  # 수량이 계획 범위인가
    gate_open: bool | None = None         # 이번 달 시장 게이트가 열려 있는가
    slots_free: int | None = None         # 남은 자리
    already_held: bool | None = None      # 이미 보유 중인 종목인가
    planned_risk: float | None = None     # 계획한 위험 금액
    actual_risk: float | None = None      # 실제 주문의 위험 금액
    note: str = ""


@dataclass
class ChecklistResult:
    decision: str
    reasons: list[str] = field(default_factory=list)
    answers: Answers | None = None

    @property
    def ok(self) -> bool:
        return self.decision == PASS

    def summary(self) -> str:
        if self.ok:
            return "통과 — 주문 가능"
        return f"{self.decision} — " + "; ".join(self.reasons)


def evaluate_checklist(
    answers: Answers,
    guard: GuardResult | None = None,
) -> ChecklistResult:
    """답변과 차단기 상태로 주문 가부를 낸다."""
    reasons: list[str] = []
    unknown: list[str] = []

    checks = [
        ("written_plan", answers.written_plan, "계획에 없는 진입"),
        ("stop_defined", answers.stop_defined, "손절가 미정"),
        ("size_within_plan", answers.size_within_plan, "수량이 계획 밖"),
        ("gate_open", answers.gate_open, "시장 게이트가 닫혀 있음"),
    ]
    for name, value, message in checks:
        if value is None:
            unknown.append(name)
        elif not value:
            reasons.append(message)

    if answers.already_held is None:
        unknown.append("already_held")
    elif answers.already_held:
        reasons.append("이미 보유 중인 종목 (중복 진입)")

    if answers.slots_free is None:
        unknown.append("slots_free")
    elif answers.slots_free <= 0:
        reasons.append("남은 자리 없음")

    if answers.planned_risk is None or answers.actual_risk is None:
        unknown.append("risk")
    elif answers.actual_risk > answers.planned_risk:
        reasons.append(
            f"실제 위험 {answers.actual_risk:,.0f}원 > 계획 {answers.planned_risk:,.0f}원")

    if guard is not None and not guard.can_trade:
        reasons.append(f"차단기: {guard.summary()}")

    if reasons:
        return ChecklistResult(BLOCK, reasons, answers)
    if unknown:
        # 모르면 통과시키지 않는다.
        return ChecklistResult(REVIEW, [f"미응답: {', '.join(unknown)}"], answers)
    return ChecklistResult(PASS, [], answers)
