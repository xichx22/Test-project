"""신호 정의 규약.

신호 = 특정 봉에서 True 가 되는 boolean Series (이벤트).

미래참조 금지 규칙 (이 프로젝트의 1번 규칙)
------------------------------------------
- 신호는 t 봉 **종가 확정 후** 계산된다. t 봉의 종가/고가/저가를 써도 된다.
- 체결은 t+1 봉 **시가**에서 일어난다고 가정한다. 검증 코드가 이 지연을 강제한다.
- shift(-k) 처럼 미래를 당겨오는 연산은 지표/신호 어디에도 없어야 한다
  (`tests/test_no_lookahead.py` 가 이걸 기계적으로 검사한다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd

# fn(candles, features) -> boolean Series
SignalFunc = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]

LONG = "long"
SHORT = "short"


@dataclass(frozen=True)
class SignalSpec:
    name: str
    func: SignalFunc
    category: str                 # breakout / pullback / reversion / momentum / volatility / volume
    side: str = LONG              # long: 매수 진입, short: 매도(청산 또는 공매도) 신호
    kind: str = "entry"           # entry / exit
    rationale: str = ""           # 왜 이 신호가 의미를 가질 수 있는가 (가설)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def evaluate(self, candles: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        out = self.func(candles, features)
        if not isinstance(out, pd.Series):
            raise TypeError(f"신호 '{self.name}' 가 Series 를 반환하지 않았습니다: {type(out)}")
        return out.fillna(False).astype(bool).rename(self.name)


REGISTRY: dict[str, SignalSpec] = {}


def signal(
    name: str,
    category: str,
    *,
    side: str = LONG,
    kind: str = "entry",
    rationale: str = "",
    tags: Iterable[str] = (),
) -> Callable[[SignalFunc], SignalFunc]:
    """신호 등록 데코레이터."""

    def wrap(func: SignalFunc) -> SignalFunc:
        REGISTRY[name] = SignalSpec(name, func, category, side, kind, rationale, tuple(tags))
        return func

    return wrap


def evaluate_all(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    *,
    names: Iterable[str] | None = None,
    kind: str | None = None,
    exclude_tags: Iterable[str] = (),
) -> pd.DataFrame:
    """등록된 신호를 모두 평가해 boolean 행렬로 만든다.

    exclude_tags: 해당 태그가 붙은 신호를 뺀다. 일봉에서 장중 전용 신호
    (VWAP·개장레인지 등)를 제외할 때 쓴다 — 일봉에서는 한 봉이 곧 한 세션이라
    이 신호들이 자명하거나 무의미해진다.
    """
    specs = [REGISTRY[n] for n in names] if names else list(REGISTRY.values())
    if kind:
        specs = [s for s in specs if s.kind == kind]
    if exclude_tags:
        banned = set(exclude_tags)
        specs = [s for s in specs if not banned & set(s.tags)]
    data = {s.name: s.evaluate(candles, features) for s in specs}
    return pd.DataFrame(data, index=candles.index)


def catalog() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"name": s.name, "kind": s.kind, "side": s.side, "category": s.category,
         "rationale": s.rationale, "tags": ",".join(s.tags)}
        for s in REGISTRY.values()
    ]
    return pd.DataFrame(rows).sort_values(["kind", "category", "name"]).reset_index(drop=True)
