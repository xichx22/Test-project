"""매매 기록 — 사람이 읽고 고칠 수 있는 한 줄 한 건.

왜 JSON Lines 인가
------------------
한 줄이 한 건이라 손으로 열어 고칠 수 있고, 중간이 깨져도 나머지가 살아남는다.
기록은 도구가 죽어도 남아야 하는 것이므로 형식이 단순해야 한다.

무엇을 남기는가
---------------
체결만 남기지 않는다. **왜 샀는지**와 **규칙을 지켰는지**를 같이 남긴다.
나중에 볼 때 필요한 것은 수익률이 아니라 "그때 무슨 생각이었나" 이기 때문이다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator

from ..ohlcv import KST

OPEN, CLOSED = "open", "closed"


@dataclass
class Fill:
    """한 건의 매매. 청산 전에는 exit_* 가 비어 있다."""

    code: str
    name: str = ""
    entry_date: str = ""
    entry_price: float = 0.0
    shares: int = 0
    stop_price: float = 0.0
    reason: str = ""                 # 왜 샀는가 (신호 이름)
    plan_note: str = ""              # 계획에 적어 둔 말
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""            # stop / time / gate / manual
    rule_breaks: list[str] = field(default_factory=list)
    lesson: str = ""                 # 사후에 적는다

    @property
    def status(self) -> str:
        return CLOSED if self.exit_date else OPEN

    @property
    def gross_return(self) -> float | None:
        if not self.exit_date or self.entry_price <= 0:
            return None
        return self.exit_price / self.entry_price - 1

    @property
    def pnl(self) -> float | None:
        r = self.gross_return
        return None if r is None else r * self.entry_price * self.shares

    def net_return(self, round_trip_bps: float = 28.0) -> float | None:
        r = self.gross_return
        return None if r is None else r - round_trip_bps / 10_000.0


class Ledger:
    """파일 하나에 담긴 매매 기록."""

    def __init__(self, path: str | Path = "state/ledger.jsonl") -> None:
        self.path = Path(path)

    # ------------------------------------------------------------- 읽기/쓰기
    def load(self) -> list[Fill]:
        if not self.path.exists():
            return []
        out: list[Fill] = []
        for line_no, line in enumerate(self.path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Fill(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                # 한 줄이 깨져도 나머지는 살린다. 조용히 넘어가지는 않는다.
                raise ValueError(f"{self.path}:{line_no} 를 읽을 수 없습니다: {exc}") from exc
        return out

    def save(self, fills: Iterable[Fill]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.dumps(asdict(f), ensure_ascii=False) for f in fills]
        self.path.write_text("\n".join(rows) + ("\n" if rows else ""))
        return self.path

    # --------------------------------------------------------------- 조작
    def add(self, fill: Fill) -> Fill:
        """새 매매를 기록한다. 같은 종목이 이미 열려 있으면 거부한다.

        중복 진입을 막는 것이 요점이다 — 같은 종목을 두 자리에 담으면
        분산이 깨지고, 보통은 주문 실수다.
        """
        fills = self.load()
        if any(f.code == fill.code and f.status == OPEN for f in fills):
            raise ValueError(f"{fill.code} 는 이미 보유 중입니다 (중복 진입)")
        if not fill.entry_date:
            fill.entry_date = datetime.now(tz=None).astimezone().strftime("%Y-%m-%d")
        fills.append(fill)
        self.save(fills)
        return fill

    def close(self, code: str, *, exit_date: str, exit_price: float,
              exit_reason: str, lesson: str = "") -> Fill:
        fills = self.load()
        for f in fills:
            if f.code == code and f.status == OPEN:
                f.exit_date, f.exit_price = exit_date, float(exit_price)
                f.exit_reason, f.lesson = exit_reason, lesson
                self.save(fills)
                return f
        raise ValueError(f"{code} 의 열린 포지션이 없습니다")

    def open_positions(self) -> list[Fill]:
        return [f for f in self.load() if f.status == OPEN]

    def closed(self) -> list[Fill]:
        return [f for f in self.load() if f.status == CLOSED]

    def between(self, start: str | date, end: str | date) -> list[Fill]:
        """청산일 기준으로 기간을 자른다."""
        s, e = str(start), str(end)
        return [f for f in self.closed() if s <= f.exit_date <= e]
