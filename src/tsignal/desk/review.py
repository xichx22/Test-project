"""리뷰 — 기간을 잘라 규칙을 지켰는지 본다.

무엇을 보고 무엇을 보지 않는가
------------------------------
**보지 않는 것: 짧은 기간의 수익률.** 이 전략은 승률 34%, 최장 연속 손실
32회다. 한 달, 한 분기 수익률은 잡음이고, 그걸 보고 규칙을 바꾸면 그때부터는
검증된 것을 하는 게 아니다.

**보는 것: 규칙을 지켰는가.** 손절을 미뤘는가, 자리를 늘렸는가, 게이트를
무시했는가. 이건 표본이 적어도 셀 수 있고, 통제할 수 있는 유일한 변수다.

수익률은 참고로 같이 내되, 백테스트 분포의 어디에 있는지와 함께 낸다.
"백테스트 최악이 −23.5% 였는데 지금 −8% 다" 는 판단할 수 있는 문장이고,
"이번 분기 −8% 다" 는 판단할 수 없는 문장이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ledger import Fill

# 21년 백테스트에서 나온 참조 분포 (10자리·120일·손절 −8%·월말 게이트)
BACKTEST = {
    "승률": 0.34,
    "평균수익": 0.213,
    "평균손실": -0.069,
    "12개월 최악": -0.235,
    "12개월 양수율": 0.70,
    "최장 연속손실": 32,
}


@dataclass
class ReviewResult:
    period: str
    n_trades: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    total_return: float | None
    rule_breaks: int
    longest_loss_streak: int
    exits: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"== {self.period} =="]
        if self.n_trades == 0:
            lines.append("  청산된 매매 없음")
            return "\n".join(lines)
        lines.append(f"  체결 {self.n_trades}건  승률 {self.win_rate:.0%} "
                     f"(백테스트 {BACKTEST['승률']:.0%})")
        if self.avg_win is not None:
            lines.append(f"  평균수익 {self.avg_win:+.1%} / 평균손실 {self.avg_loss:+.1%} "
                         f"(백테스트 {BACKTEST['평균수익']:+.1%} / {BACKTEST['평균손실']:+.1%})")
        lines.append(f"  최장 연속손실 {self.longest_loss_streak}회 "
                     f"(백테스트 최장 {BACKTEST['최장 연속손실']}회)")
        lines.append(f"  청산 사유 {self.exits}")
        lines.append(f"  규칙 위반 {self.rule_breaks}회" +
                     ("  ← 2회면 중단" if self.rule_breaks >= 1 else ""))
        for f in self.flags:
            lines.append(f"  ! {f}")
        return "\n".join(lines)


def _streak(fills: list[Fill]) -> int:
    best = run = 0
    for f in sorted(fills, key=lambda x: x.exit_date):
        r = f.gross_return
        if r is not None and r <= 0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def review(fills: list[Fill], *, period: str = "전체",
           round_trip_bps: float = 28.0) -> ReviewResult:
    """청산된 매매만 집계한다."""
    closed = [f for f in fills if f.status == "closed" and f.gross_return is not None]
    breaks = sum(len(f.rule_breaks) for f in fills)
    if not closed:
        return ReviewResult(period, 0, None, None, None, None, breaks, 0)

    rets = [f.net_return(round_trip_bps) for f in closed]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    exits: dict[str, int] = {}
    for f in closed:
        exits[f.exit_reason or "미기재"] = exits.get(f.exit_reason or "미기재", 0) + 1

    flags = []
    if breaks >= 2:
        flags.append("규칙 위반 2회 — 계획서상 중단 조건")
    if len(closed) < 20:
        flags.append(f"표본 {len(closed)}건 — 성과로 판단하기에 부족하다 (규칙 준수만 본다)")
    if _streak(closed) > BACKTEST["최장 연속손실"]:
        flags.append("연속 손실이 백테스트 최장을 넘었다 — 규칙 점검 필요")

    total = 1.0
    for r in rets:
        total *= 1 + r
    return ReviewResult(
        period=period, n_trades=len(closed),
        win_rate=len(wins) / len(closed),
        avg_win=sum(wins) / len(wins) if wins else None,
        avg_loss=sum(losses) / len(losses) if losses else None,
        total_return=total - 1, rule_breaks=breaks,
        longest_loss_streak=_streak(closed), exits=exits, flags=flags,
    )
