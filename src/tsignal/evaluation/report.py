"""검증 리포트 생성.

이 프로젝트의 최종 산출물. "어떤 신호에 진입하고 어떤 신호에 청산할 수 있다"는
주장을, 표본 수 / 기대값 / OOS 재현 / 다중검정 보정까지 붙여서 문서로 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasource.base import Interval
from .. import indicators as ind
from .. import signals as sig
from . import validation
from .forward import DEFAULT_HORIZONS, screen_signals
from .trades import CostModel, ExitPolicy, exit_reason_breakdown, simulate, suggest_barriers


@dataclass
class ReportConfig:
    code: str
    interval: Interval
    horizon: int = 5
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    train_ratio: float = 0.6
    min_events: int = 30
    alpha: float = 0.05
    top_k: int = 5
    costs: CostModel = CostModel()
    policy: ExitPolicy = ExitPolicy()


def _pct(x: float, digits: int = 3) -> str:
    return "-" if not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _num(x: float, digits: int = 2) -> str:
    return "-" if not np.isfinite(x) else f"{x:.{digits}f}"


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """의존성 없이 마크다운 표를 만든다."""
    if df.empty:
        return "_(데이터 없음)_\n"
    body = df.reset_index() if df.index.name else df.copy()
    header = "| " + " | ".join(str(c) for c in body.columns) + " |"
    sep = "| " + " | ".join("---" for _ in body.columns) + " |"
    lines = [header, sep]
    for _, row in body.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("-" if not np.isfinite(v) else floatfmt.format(v))
            elif isinstance(v, pd.Timestamp):
                cells.append(v.strftime("%Y-%m-%d %H:%M"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build(candles: pd.DataFrame, config: ReportConfig) -> str:
    features = ind.compute_all(candles, interval=config.interval)
    entries = sig.evaluate_all(candles, features, kind="entry")
    exits = sig.evaluate_all(candles, features, kind="exit")

    screen = screen_signals(candles, entries, horizons=config.horizons)
    split = validation.split_sample(
        candles, entries, horizon=config.horizon, train_ratio=config.train_ratio
    )
    threshold = validation.deflated_threshold(len(entries.columns), config.alpha)

    merged = screen.join(split[["n_is", "exp_is", "t_is", "n_oos", "exp_oos", "t_oos", "sign_agree"]])
    merged["verdict"] = merged.apply(
        lambda r: validation.verdict(r, t_threshold=threshold, min_events=config.min_events), axis=1
    )
    merged = merged.sort_values(["verdict", "t_oos"], ascending=[True, False])

    out: list[str] = []
    add = out.append

    add(f"# 단타 신호 검증 리포트 — {config.code} ({config.interval.value})\n")
    add(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 데이터 구간: {candles.index[0]:%Y-%m-%d %H:%M} ~ {candles.index[-1]:%Y-%m-%d %H:%M} "
        f"({len(candles):,}봉)")
    add(f"- 지표 {len(ind.REGISTRY)}종 → 피처 {features.shape[1]}개, "
        f"진입 신호 {entries.shape[1]}개 / 청산 신호 {exits.shape[1]}개")
    add(f"- 왕복 매매비용: {config.costs.round_trip_bps:.1f}bp "
        f"(수수료 {config.costs.fee_bps}bp×2 + 세금 {config.costs.tax_bps}bp + 슬리피지 {config.costs.slippage_bps}bp×2)")
    add("")

    add("## 0. 이 리포트를 읽는 법\n")
    add("판단 기준은 셋이고, 셋을 **모두** 통과해야 '채택후보'다.\n")
    add(f"1. **표본**: 신호 발생 {config.min_events}회 미만은 통계로 다루지 않는다 (표본부족).")
    add(f"2. **유의성**: 진입 신호 {len(entries.columns)}개를 동시에 시험했으므로 "
        f"다중검정 보정(Šidák, α={config.alpha}) 후 문턱은 **|t| ≥ {threshold:.2f}** 다. "
        "흔히 쓰는 t=2.0 을 그대로 쓰면 우연히 좋아 보이는 신호를 채택하게 된다.")
    add(f"3. **재현**: 앞 {config.train_ratio:.0%}(IS)에서 나온 결론이 뒤 {1 - config.train_ratio:.0%}(OOS)에서도 "
        "같은 부호로 재현돼야 한다.\n")
    add("> `exp_k` 는 신호 다음 봉 **시가 진입** 기준 k봉 뒤 수익률의 평균이다. "
        "`edge_k` 는 같은 기간 무조건부 평균을 뺀 값 — 이게 0 근처면 그 신호는 "
        "'시장이 올라서 번 것'이지 신호가 번 게 아니다.\n")

    add("## 1. 진입 신호 스크리닝\n")
    cols = ["n", f"exp_{config.horizon}", f"edge_{config.horizon}", f"win_{config.horizon}",
            f"t_{config.horizon}", "t_is", "t_oos", "sign_agree", "verdict"]
    add(_md_table(merged[cols].round(5)))
    add("")

    counts = merged["verdict"].value_counts().to_dict()
    add(f"판정 집계: {counts}\n")

    accepted = merged[merged["verdict"] == "채택후보"]
    if accepted.empty:
        add("> **채택후보 없음.** 이 데이터·기간에서는 어떤 단일 신호도 보정된 문턱을 넘지 못했다. "
            "신호를 버리라는 뜻이 아니라, *단독 신호로는 근거가 부족*하다는 뜻이다. "
            "다음 단계는 (a) 조건 결합(추세 필터 + 트리거), (b) 다른 타임프레임, "
            "(c) 종목 유니버스 확대 순이다.\n")
    else:
        add(f"채택후보 {len(accepted)}개: {', '.join(accepted.index)}\n")

    add("## 2. 상위 후보 상세 — 구간 안정성\n")
    add("한 구간에서 몰아서 번 신호는 실전에서 재현되지 않는다. "
        "5등분한 구간마다 기대값 부호가 유지되는지 본다.\n")
    ranked = merged.sort_values("t_oos", ascending=False).head(config.top_k)
    for name in ranked.index:
        table = validation.stability(candles, entries[name], horizon=config.horizon)
        positive = int((table["expectancy"] > 0).sum())
        spec = sig.REGISTRY[name]
        add(f"### {name}\n")
        add(f"- 가설: {spec.rationale}")
        add(f"- 구간 5개 중 양(+) 기대값: **{positive}/5**\n")
        add(_md_table(table.drop(columns=["from", "to"]).round(5)))
        add("")

    add("## 3. 청산 규칙 비교 — '언제 팔 것인가'\n")
    add("같은 진입 신호에 청산 규칙만 바꿔 붙였을 때 순손익이 어떻게 달라지는지가 "
        "이 프로젝트가 답하려는 두 번째 질문이다. 아래는 상위 후보 진입에 대해 "
        "청산 규칙을 갈아끼운 결과다 (모두 비용 차감 후).\n")

    exit_names = [None] + list(exits.columns)
    for name in ranked.index[:3]:
        rows = []
        for exit_name in exit_names:
            policy = ExitPolicy(
                stop_atr=config.policy.stop_atr,
                target_atr=config.policy.target_atr,
                max_bars=config.policy.max_bars,
                exit_signal=exit_name,
                close_at_session_end=config.policy.close_at_session_end,
            )
            trades = simulate(
                candles, entries[name], policy=policy, costs=config.costs,
                exit_events=exits[exit_name] if exit_name else None,
            )
            if trades.empty:
                continue
            r = trades["ret_net"]
            rows.append({
                "청산규칙": exit_name or "(배리어만)",
                "거래수": len(trades),
                "평균순익": float(r.mean()),
                "승률": float((r > 0).mean()),
                "누적": float((1 + r).prod() - 1),
                "평균보유봉": float(trades["bars_held"].mean()),
            })
        if not rows:
            continue
        table = pd.DataFrame(rows).set_index("청산규칙").sort_values("평균순익", ascending=False)
        add(f"### 진입: {name}\n")
        add(_md_table(table.round(5)))
        add("")

    add("## 4. 익절/손절 폭 근거 (MFE/MAE 분포)\n")
    add("진입 후 실제로 얼마나 유리하게/불리하게 움직였는지의 분위수. "
        "익절폭은 MFE 분포 안쪽에, 손절폭은 MAE 분포 바깥에 두는 게 출발점이다.\n")
    for name in ranked.index[:3]:
        barriers = suggest_barriers(candles, entries[name], horizon=config.policy.max_bars)
        if barriers.empty:
            continue
        add(f"### {name}\n")
        add(_md_table(barriers.round(3)))
        add("")

    add("## 5. 청산 사유 분해\n")
    top = ranked.index[0]
    trades = simulate(
        candles, entries[top], policy=config.policy, costs=config.costs,
        exit_events=exits[config.policy.exit_signal] if config.policy.exit_signal else None,
    )
    add(f"진입 `{top}` / 청산 `{config.policy.describe()}`\n")
    add(_md_table(exit_reason_breakdown(trades).round(4)))
    if not trades.empty:
        add(f"\n- 총 {len(trades)}거래, 평균 순수익 {_pct(float(trades['ret_net'].mean()))}, "
            f"누적 {_pct(float((1 + trades['ret_net']).prod() - 1), 2)}")
        add(f"- 비용 전 평균 {_pct(float(trades['ret_gross'].mean()))} → "
            f"비용이 회당 {config.costs.round_trip_bps / 100:.2f}%를 가져간다. "
            "단타에서 비용은 전략의 일부가 아니라 전략 그 자체다.\n")

    add("## 6. 한계\n")
    add("- 단일 종목·단일 기간 결과다. 종목/기간을 늘려 같은 결론이 나오는지 확인하기 전에는 채택하지 말 것.")
    add("- 체결은 다음 봉 시가로 가정했다. 실제로는 호가 잔량과 시장가 슬리피지가 더 붙는다.")
    add("- 한 봉 안에서 익절·손절이 같이 닿으면 손절로 처리했다. 실제 순서는 틱 데이터가 있어야 안다.")
    add("- 상장폐지·거래정지 종목이 표본에서 빠졌다면 생존 편향이 남아 있다.")
    add("- 이 리포트는 매매 권유가 아니다. 통계적 근거를 기록하는 문서다.\n")

    return "\n".join(out)


def write(candles: pd.DataFrame, config: ReportConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(candles, config), encoding="utf-8")
    return target
