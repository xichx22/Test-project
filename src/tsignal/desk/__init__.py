"""매매 데스크 — 계획·기록·차단·리뷰.

이 패키지는 수익을 만들지 않는다. 이 프로젝트가 21년 1,064종목으로 확인한 것은
**가격 데이터로 매매 타이밍의 우위를 찾을 수 없다**는 것이었다. 그렇다면 남는
변수는 규칙을 지키는가 하나뿐이고, 이 패키지는 그것만 다룬다.

  계획 → 매매 → 기록 → 리뷰 → 개선

구성
----
`sizing`     자리 수와 손절 폭에서 주문 수량을 낸다
`guard`      오늘 새 매매를 해도 되는가 (손실 차단기)
`checklist`  이 주문을 내도 되는가 (진입 전 규율 게이트)
`ledger`     체결과 청산을 남긴다
`review`     기간별로 집계해 규칙을 지켰는지 본다

출처
----
구조와 용어는 tradermonty/claude-trading-skills (MIT, © 2026 TraderMonty) 의
position-sizer · pre-trade-discipline-gate · drawdown-circuit-breaker ·
trader-memory-core 를 참고했다. 자세한 내용과 무엇을 바꿨는지는
`docs/ported-from.md` 에 적어 두었다.

문턱은 그대로 가져오지 않았다 — 원본 기본값(연속 2패 쿨다운)을 이 전략에
적용하면 21년간 45번 발동해 사실상 영구 정지가 된다. 승률 34%, 최장 연속
손실 32회인 규칙이기 때문이다. 모든 문턱을 실측 분포에서 다시 유도했다.
"""

from .checklist import Answers, ChecklistResult, evaluate_checklist
from .guard import GuardResult, Thresholds, evaluate_guard
from .ledger import Fill, Ledger
from .review import review
from .sizing import SizingResult, size_order

__all__ = [
    "Answers", "ChecklistResult", "evaluate_checklist",
    "GuardResult", "Thresholds", "evaluate_guard",
    "Fill", "Ledger", "review",
    "SizingResult", "size_order",
]
