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


# =====================================================================
# 유니버스 리포트 — 여러 종목을 묶어서 검증
# =====================================================================

def build_universe(
    candles_by_code: dict[str, pd.DataFrame],
    *,
    interval: Interval,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
    exclude_tags: tuple[str, ...] = (),
    train_ratio: float = 0.6,
    min_events: int = 100,
    alpha: float = 0.05,
    costs: CostModel = CostModel(),
    names: dict[str, str] | None = None,
) -> str:
    """유니버스 전체 검증 리포트."""
    from .universe import screen_universe, split_universe

    names = names or {}
    spans = [(c, df.index[0], df.index[-1], len(df)) for c, df in candles_by_code.items()]
    total_bars = sum(s[3] for s in spans)

    out: list[str] = []
    add = out.append

    add(f"# 단타 신호 검증 리포트 — 유니버스 {len(candles_by_code)}종목 ({interval.value})\n")
    add(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 데이터: {min(s[1] for s in spans):%Y-%m-%d} ~ {max(s[2] for s in spans):%Y-%m-%d}, "
        f"총 {total_bars:,}봉")
    add(f"- 종목: {', '.join(names.get(c, c) for c in candles_by_code)}")
    add(f"- 왕복 매매비용 {costs.round_trip_bps:.1f}bp\n")

    add("## 0. 판정 기준\n")
    add("**초과수익(edge)으로 검정한다.** 원시 수익률의 t(`t_raw`)로 판정하면 "
        "보유기간을 늘릴수록 아무 신호나 유의해진다 — 시장이 우상향하면 "
        "'아무 때나 사서 들고 있기'의 기대값이 양수이기 때문이다. "
        "종목별 무조건부 평균을 뺀 `edge` 의 t(`t_edge`)만 신호의 기여분이다.\n")
    add("| 판정 | 조건 |")
    add("| --- | --- |")
    add(f"| 채택후보 | 표본 {min_events}회 이상 · `t_edge` ≥ 보정 문턱 · breadth ≥ 0.6 |")
    add("| 쏠림주의 | t는 넘었으나 소수 종목에 성과가 몰림 (breadth < 0.6) |")
    add("| 보류 | t_edge > 1.0 |")
    add("| 기각 | 그 외 |\n")

    for horizon in horizons:
        screen = screen_universe(
            candles_by_code, interval=interval, horizon=horizon,
            exclude_tags=exclude_tags, min_events=min_events, alpha=alpha,
        )
        threshold = screen.attrs["threshold"]
        add(f"## 보유 {horizon}봉\n")
        add(f"무조건부 기대값 {_pct(screen.attrs['baseline'])} · 보정 문턱 |t| ≥ {threshold:.2f} "
            f"(신호 {len(screen)}개 동시 검정)\n")
        cols = ["n_codes", "n", "expectancy", "edge", "win_rate", "t_raw", "t_edge", "breadth", "verdict"]
        add(_md_table(screen[cols].head(12).round(4)))
        counts = screen["verdict"].value_counts().to_dict()
        add(f"\n판정 집계: {counts}\n")

        gap = (screen["t_raw"] - screen["t_edge"]).abs().max()
        if np.isfinite(gap) and gap > 1.0:
            worst = (screen["t_raw"] - screen["t_edge"]).abs().idxmax()
            add(f"> `t_raw` 와 `t_edge` 의 최대 격차 {gap:.2f} (`{worst}`). "
                "이 격차가 곧 '시장이 벌어준 몫'이다.\n")

    add("## OOS 재현 검증\n")
    add(f"종목마다 앞 {train_ratio:.0%} 를 IS, 뒤 {1 - train_ratio:.0%} 를 OOS 로 잘라 "
        "초과수익 t 를 각각 낸다. 부호가 뒤집히면 IS 결과는 우연이었다고 본다.\n")
    split = split_universe(
        candles_by_code, interval=interval, horizon=horizons[min(2, len(horizons) - 1)],
        train_ratio=train_ratio, exclude_tags=exclude_tags,
    )
    add(_md_table(split.head(12).round(4)))
    survivors = split[(split["t_edge_is"] > 1.5) & (split["t_edge_oos"] > 1.5) & split["sign_agree"]]
    add(f"\nIS·OOS 모두 t_edge > 1.5 이고 부호가 일치하는 신호: "
        f"{', '.join(survivors.index) if len(survivors) else '**없음**'}\n")

    add("## 한계\n")
    add(f"- 종목 {len(candles_by_code)}개는 유니버스로 작다. 코스피200 전체로 넓혀야 한다.")
    add("- 현재 상장된 종목만 담겼다 — 상장폐지 종목이 빠진 **생존 편향**이 남아 있다.")
    add("- 일봉 결과이므로 분봉 단타에 그대로 옮길 수 없다. 타임프레임이 바뀌면 다시 재야 한다.")
    add("- 이 리포트는 매매 권유가 아니다.\n")

    return "\n".join(out)


def write_universe(
    candles_by_code: dict[str, pd.DataFrame], path: str | Path, **kwargs
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_universe(candles_by_code, **kwargs), encoding="utf-8")
    return target
