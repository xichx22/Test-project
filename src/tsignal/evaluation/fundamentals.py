"""재무 수치로 종목을 거르는 스크리너 — 그리고 그 한계.

왜 이것만 백테스트가 없는가
---------------------------
이 프로젝트의 다른 모든 것은 21년치로 검증했다. 재무 팩터는 못 했다.
무료로 구할 수 있는 한국 재무 이력이 **연간 3년치**뿐이기 때문이다
(네이버 API 기준. DART OpenAPI 는 키가 필요하고 FnGuide 는 막혀 있다).

3년이면 리밸런싱 시점이 두 번이다. 관측 2개로는 아무것도 검정할 수 없다.
그래서 이 모듈은 **검증된 규칙이 아니라 걸러내기 도구**다. 순위를 매겨
1등을 사라고 말하지 않고, "이 조건들을 동시에 만족하지 못하는 종목"을
후보에서 빼는 데 쓴다.

가격에서 계산되는 팩터는 왜 안 되는가
-------------------------------------
같은 21년 데이터로 고전 팩터 7종(소형주·장기역행·모멘텀·저변동성·
저회전율·단기역행)을 걸었다. 다중검정 문턱 |t| ≥ 2.68 을 넘은 것은
소형주(t=6.83)와 저회전율(t=−2.99) 둘인데, 둘 다 생존편향이었다.

    시작 시점 하위 20% 종목의 시총 배수 중앙값   14.86배
    대형 상위 20%                              1.15배

유니버스가 "오늘 시총 상위 1,200종목" 이라, 과거에 작았던 종목이 지금
목록에 있으려면 그 사이 15배 커졌어야 한다. 소형 십분위는 구조적으로
**성공한 소형주만** 담는다. 유동성 문턱을 올리면 사라진다
(3억 t=5.58 → 100억 t=1.13 → 300억 t=0.57).

그래서 이 프로젝트가 검증한 매수 규칙은 여전히 하나뿐이다 —
6자산 동일가중 분기 리밸런싱. 그것도 "수치를 보고 사는" 규칙이다.
분기마다 평가액÷6 을 계산해 그보다 적은 자산을 산다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COLUMNS = ("code", "name", "cap", "price", "per", "pbr", "roe",
           "div", "debt", "op_margin", "op_growth")


@dataclass(frozen=True)
class Screen:
    """거르기 조건. 전부 '이보다 나쁘면 뺀다' 는 뜻이다.

    문턱은 백테스트가 아니라 **분포**에서 왔다. 한국 시총 상위 400종목의
    중앙값이 PER 12.1 · PBR 0.88 · ROE 6.2% · 배당 2.55% · 부채 99% 다.
    기본값은 대체로 그 중앙값 근처에 두어, "평균보다 나쁘지 않은 것"만
    남기도록 했다. 상위 몇 %를 노리는 값이 아니다.
    """

    max_per: float | None = 15.0
    max_pbr: float | None = 1.5
    min_roe: float | None = 5.0
    min_div: float | None = None
    max_debt: float | None = 150.0
    min_op_margin: float | None = None
    min_op_growth: float | None = None
    min_cap: float | None = 5_000e8      # 시총 5,000억 이상 (유동성 대용)
    require_all: bool = True             # 결측이 있으면 뺄 것인가


@dataclass
class ScreenResult:
    passed: pd.DataFrame
    dropped: dict[str, int] = field(default_factory=dict)
    total: int = 0

    def report(self, top: int = 20) -> str:
        lines = [f"통과 {len(self.passed)} / {self.total}종목"]
        if self.dropped:
            lines.append("탈락 사유: " + ", ".join(
                f"{k} {v}" for k, v in sorted(
                    self.dropped.items(), key=lambda x: -x[1])))
        if self.passed.empty:
            lines.append("  조건을 만족하는 종목이 없습니다 — 문턱을 확인하세요.")
            return "\n".join(lines)
        view = self.passed.head(top).copy()
        view["시총"] = (view["cap"] / 1e8).map("{:,.0f}억".format)
        for col, fmt in (("per", "{:.1f}"), ("pbr", "{:.2f}"),
                         ("roe", "{:.1f}%"), ("div", "{:.2f}%"),
                         ("debt", "{:.0f}%")):
            view[col] = view[col].map(lambda v, f=fmt: "-" if pd.isna(v) else f.format(v))
        cols = ["code", "name", "시총", "per", "pbr", "roe", "div", "debt"]
        lines.append(view[cols].to_string(index=False))
        return "\n".join(lines)


def screen(frame: pd.DataFrame, spec: Screen = Screen()) -> ScreenResult:
    """조건을 만족하지 못하는 종목을 뺀다. 순위는 매기지 않는다.

    순위를 매기지 않는 이유: 어떤 지표로 줄을 세워야 하는지가 검증되지
    않았다. 줄을 세우면 1등을 사게 되고, 그건 검증되지 않은 예측이다.
    """
    data = frame.copy()
    keep = pd.Series(True, index=data.index)
    dropped: dict[str, int] = {}

    rules = [
        ("PER", "per", lambda s: s <= spec.max_per, spec.max_per),
        ("PBR", "pbr", lambda s: s <= spec.max_pbr, spec.max_pbr),
        ("ROE", "roe", lambda s: s >= spec.min_roe, spec.min_roe),
        ("배당", "div", lambda s: s >= spec.min_div, spec.min_div),
        ("부채비율", "debt", lambda s: s <= spec.max_debt, spec.max_debt),
        ("영업이익률", "op_margin", lambda s: s >= spec.min_op_margin, spec.min_op_margin),
        ("영업이익증가", "op_growth", lambda s: s >= spec.min_op_growth, spec.min_op_growth),
        ("시총", "cap", lambda s: s >= spec.min_cap, spec.min_cap),
    ]
    for label, col, test, threshold in rules:
        if threshold is None or col not in data:
            continue
        values = pd.to_numeric(data[col], errors="coerce")
        # pandas 에서 `NaN <= 15` 는 NaN 이 아니라 **False** 다. 그래서
        # fillna 로는 결측을 되살릴 수 없고, 명시적으로 덮어써야 한다.
        # 결측을 통과시키면 재무를 공시하지 않는 종목이 다 남는다.
        ok = test(values).where(values.notna(), not spec.require_all)
        lost = int((keep & ~ok).sum())
        if lost:
            dropped[label] = lost
        keep &= ok

    passed = data[keep].sort_values("cap", ascending=False)
    return ScreenResult(passed=passed, dropped=dropped, total=len(data))


def distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """유니버스의 재무 분포. 문턱을 정하기 전에 먼저 본다."""
    rows = []
    for col in ("per", "pbr", "roe", "div", "debt", "op_margin", "op_growth"):
        if col not in frame:
            continue
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "지표": col, "표본": len(s), "결측률": 1 - len(s) / len(frame),
            "하위25%": s.quantile(0.25), "중앙값": s.median(),
            "상위25%": s.quantile(0.75),
        })
    return pd.DataFrame(rows)
