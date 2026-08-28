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


# =====================================================================
# 조합 탐색 리포트
# =====================================================================

def build_combination(
    candles_by_code: dict[str, pd.DataFrame],
    *,
    interval: Interval,
    horizons: tuple[int, ...] = (3, 5, 10),
    exclude_tags: tuple[str, ...] = (),
    train_ratio: float = 0.6,
    max_filters: int = 2,
    min_events: int = 80,
    top_k: int = 20,
    names: dict[str, str] | None = None,
) -> str:
    """트리거 × 상태필터 조합 탐색 리포트."""
    from .combine import compare_filter_contribution, select_and_validate, split_labs

    names = names or {}
    out: list[str] = []
    add = out.append

    add(f"# 신호 결합 탐색 리포트 — {len(candles_by_code)}종목 ({interval.value})\n")
    add(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 종목: {', '.join(names.get(c, c) for c in candles_by_code)}")
    add(f"- 분할: 앞 {train_ratio:.0%} IS / 뒤 {1 - train_ratio:.0%} OOS, 종목마다 동일 비율\n")

    add("## 0. 왜 조합을 재는가, 그리고 왜 위험한가\n")
    add("단독 트리거가 전부 기각된 뒤의 가설: 트리거는 시장 상태를 구분하지 않아 "
        "서로 다른 사건을 한 통에 넣고 세고 있었다. 상승추세의 RSI 반등과 "
        "하락추세의 RSI 반등은 다른 사건이다.\n")
    add("동시에 조합 탐색은 **과최적화 기계**다. 조합 수천 개를 훑으면 엣지가 전혀 없어도 "
        "t>3 짜리가 수십 개 나온다. 그래서 프로토콜을 고정했다.\n")
    add("1. 탐색 공간을 먼저 줄인다 — 같은 축의 필터끼리는 결합하지 않고, 필터는 최대 "
        f"{max_filters}개")
    add("2. 실제로 시험한 조합 수만큼 문턱을 올린다 (Šidák) + BH-FDR 을 나란히 본다")
    add("3. **IS 에서 고르고 OOS 로 채점한다** — 상위 K개 중 몇 개가 살아남는지를 "
        "우연히 기대되는 개수와 이항검정으로 비교")
    add("4. 단독 트리거 대비 **증분(lift)** 을 본다. 필터가 트리거를 개선하지 못하면 "
        "그 조합은 의미가 없다\n")

    labs = {h: split_labs(candles_by_code, train_ratio=train_ratio, interval=interval,
                          horizon=h, exclude_tags=exclude_tags) for h in horizons}

    add("## 1. 조합 탐색 — 전체 구간\n")
    for horizon in horizons:
        from .combine import CombinationLab

        lab = CombinationLab(candles_by_code, interval=interval, horizon=horizon,
                             exclude_tags=exclude_tags)
        table = lab.search(max_filters=max_filters, min_events=min_events)
        meta = table.attrs
        add(f"### 보유 {horizon}봉\n")
        add(f"조합 {meta['n_combos']:,}개 (표본 {min_events}회 이상: {meta['n_trials']:,}개) · "
            f"Šidák 문턱 |t| ≥ {meta['threshold']:.2f} · "
            f"Šidák 통과 **{int(table['pass_sidak'].sum())}개** · "
            f"BH-FDR({meta['fdr_alpha']:.0%}) 통과 **{int(table['pass_fdr'].sum())}개**\n")
        cols = ["n", "edge", "edge_bare", "lift", "win_rate", "t_edge", "breadth"]
        add(_md_table(table[table["n"] >= min_events][cols].head(8).round(4)))
        add("")

    add("## 2. 정직한 판정 — IS 에서 고르고 OOS 로 채점\n")
    add("IS 성적으로만 상위를 고른 뒤, 그 선택을 OOS 로 채점한다. "
        "엣지가 전혀 없다면 생존 확률은 약 16%(부호가 반반 × t>1 조건)이므로, "
        f"상위 {top_k}개 중 우연히 기대되는 생존은 약 {top_k * 0.16:.1f}개다.\n")

    for horizon in horizons:
        is_lab, oos_lab = labs[horizon]
        res = select_and_validate(is_lab, oos_lab, top_k=top_k, max_filters=max_filters,
                                  min_events=min_events, min_oos_events=25)
        add(f"### 보유 {horizon}봉\n")
        add(f"- IS 상위 {res.n_selected}개 중 OOS 생존 **{res.n_survived}개** "
            f"(우연 기대 {res.expected_by_chance:.1f}개, 이항 p={res.binomial_p:.3f})")
        add(f"- **{res.verdict}**\n")
        add(_md_table(res.table.head(10).round(4)))
        add("")

    add("## 3. 필터 기여도 — 가설을 2,835개에서 15개로 줄이면\n")
    add("조합 하나를 고르는 것은 가설을 수천 개 세우는 짓이다. "
        "질문을 바꿔서 **\"필터 X 는 트리거 종류와 무관하게 도움이 되는가\"** 를 물으면 "
        "가설이 필터 개수만큼으로 줄어들고, 같은 데이터로 훨씬 강한 결론이 나온다.\n")
    add("> 검정은 **부호검정**을 쓴다. 트리거별 lift 는 표본이 겹쳐 서로 독립이 아니므로 "
        "t검정은 유의성을 부풀린다. \"몇 개 트리거에서 개선됐는가\"는 그 상관에 훨씬 덜 휘둘린다.\n")

    consistent_by_horizon: dict[int, list[str]] = {}
    for horizon in horizons:
        is_lab, oos_lab = labs[horizon]
        contrib = compare_filter_contribution(is_lab, oos_lab, min_events=min_events)
        consistent_by_horizon[horizon] = list(contrib[contrib["consistent"]].index)
        add(f"### 보유 {horizon}봉\n")
        add(_md_table(contrib.round(4)))
        add(f"\nIS·OOS 방향이 일치하는 필터: "
            f"**{', '.join(consistent_by_horizon[horizon]) or '없음'}**\n")

    add("## 4. 레짐 경고 — 구간 사이에서 부호가 뒤집히는 필터\n")
    add("변동성·레짐 축 필터는 IS 에서 강하게 도움이 되다가 OOS 에서 반대로 뒤집히는 "
        "패턴을 보인다. 이건 엣지가 아니라 **그 구간의 시장 성격을 외운 것**이다. "
        "IS 구간과 OOS 구간의 변동성 레짐이 달랐다면, 변동성 필터는 신호가 아니라 "
        "날짜를 인코딩하고 있는 셈이다.\n")
    add("이런 필터를 '검증된 조건'으로 채택하면, 레짐이 바뀌는 순간 정확히 반대로 작동한다. "
        "IS 성적만 보고 골랐다면 알 수 없었을 함정이다.\n")

    add("## 5. 결론\n")
    add("**개별 조합 수준에서는 얻은 것이 없다.** 어떤 보유기간에서도 IS 상위 "
        f"{top_k}개의 OOS 생존이 우연 기대치를 유의하게 넘지 못했다. "
        "IS 에서 t>3 이던 조합이 OOS 에서 부호가 뒤집히는 사례가 반복된다.\n")

    tally: dict[str, list[int]] = {}
    for horizon, filters_ in consistent_by_horizon.items():
        for name in filters_:
            tally.setdefault(name, []).append(horizon)
    total = len(consistent_by_horizon)
    always = sorted(n for n, hs in tally.items() if len(hs) == total)
    majority = sorted((n for n, hs in tally.items() if 1 < len(hs) < total),
                      key=lambda n: -len(tally[n]))

    add("**필터 수준에서는 하나가 남는다.** 보유기간별 IS·OOS 방향 일치 여부:\n")
    if tally:
        add("| 필터 | 일치한 보유기간 |")
        add("| --- | --- |")
        for name, hs in sorted(tally.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            add(f"| `{name}` | {', '.join(f'{h}봉' for h in hs)} ({len(hs)}/{total}) |")
        add("")
    if always:
        add(f"모든 보유기간에서 일관: **{', '.join(always)}**\n")
    elif majority:
        add(f"과반 보유기간에서 일관: **{', '.join(majority)}**\n")
        add("전 구간 일관은 아니지만, 단타가 사는 짧은 보유기간에서 방향이 유지되는 것은 "
            "의미가 있다. 단독으로 매매 근거가 되기엔 부족하고, 다른 신호에 붙이는 "
            "**확인 조건(confirmation)** 으로 다음 단계 검증 대상이다.\n")
    else:
        add("어떤 필터도 두 개 이상의 보유기간에서 일관되지 않았다. "
            "조합 탐색으로 얻은 재현 가능한 결과가 없다.\n")

    add("## 6. 한계\n")
    add("- IS/OOS 분할은 한 번뿐이다. 분할 지점을 바꾸면 결론이 달라질 수 있다 — "
        "롤링 워크포워드로 여러 번 확인해야 한다.")
    add("- 필터 lift 들은 표본이 겹쳐 서로 독립이 아니다. 부호검정으로 완화했지만 완전하진 않다.")
    add(f"- 종목 {len(candles_by_code)}개, 일봉 기준이다. 분봉 단타에 그대로 옮길 수 없다.")
    add("- 매매비용을 뺀 순수익이 아니라 초과수익(edge) 기준이다. 실제 채택 전에는 "
        "거래 시뮬로 비용 차감 후 성적을 다시 봐야 한다.")
    add("- 이 리포트는 매매 권유가 아니다.\n")

    return "\n".join(out)


def write_combination(
    candles_by_code: dict[str, pd.DataFrame], path: str | Path, **kwargs
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_combination(candles_by_code, **kwargs), encoding="utf-8")
    return target


# =====================================================================
# 워크포워드 리포트
# =====================================================================

def build_walkforward(
    candles_by_code: dict[str, pd.DataFrame],
    *,
    interval: Interval,
    horizons: tuple[int, ...] = (3, 5, 10),
    exclude_tags: tuple[str, ...] = (),
    train_months: int = 24,
    test_months: int = 3,
    step_months: int | None = None,
    scheme: str = "rolling",
    top_k: int = 20,
    min_events: int = 120,
    min_test_events: int = 25,
    alpha: float = 0.05,
) -> str:
    """롤링 워크포워드 리포트."""
    from .combine import build_panels
    from .walkforward import run as run_wf

    out: list[str] = []
    add = out.append

    add(f"# 롤링 워크포워드 리포트 — {len(candles_by_code)}종목 ({interval.value})\n")
    add(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 방식: {scheme} · 학습 {train_months}개월 / 검증 {test_months}개월 / "
        f"{step_months or test_months}개월씩 이동\n")

    add("## 0. 왜 분할을 여러 번 하는가\n")
    add("IS/OOS 분할을 한 번만 하면 결론이 \"어느 날짜에 잘랐는가\"에 걸린다. "
        "특히 변동성·레짐 필터는 두 구간의 시장 성격 차이를 그대로 반영해버린다. "
        "창을 밀어가며 반복하고, **폴드를 넘어 방향이 유지되는지**를 본다.\n")

    add("### 두 가지 중요한 보정\n")
    add("**날짜 군집 보정.** 한국 주식 199개는 같은 날 같이 움직인다. 시장이 빠진 날에는 "
        "수십~백 종목에서 같은 신호가 동시에 뜬다. 이걸 독립 관측으로 세면 표준오차가 "
        "√n 만큼 작아져 t 가 부풀려진다. 실측 인플레이션은 최대 **10배**였다 "
        "(`bollinger_lower_reclaim`: 순진한 t 5.93 → 보정 t 0.58, "
        "`donchian_breakout`: 3.56 → −0.79 로 부호까지 반전). "
        "유효 표본은 신호 발생 횟수가 아니라 **발생 날짜 수**다.\n")
    add("**폴드 단위 집계.** 한 폴드 안의 조합 20개는 표본이 크게 겹쳐 독립이 아니다. "
        "조합을 단위로 이항검정을 하면 p 가 가짜로 작아진다. 폴드별 생존률이 귀무값(16%)을 "
        "넘는지만 세고, 폴드 사이에서 부호검정을 한다.\n")

    threshold_p = 1 - (1 - alpha) ** (1 / len(__import__("tsignal.signals", fromlist=["filters"]).filters.REGISTRY))
    add(f"> 필터를 동시에 검정하므로 보정된 p 문턱은 **{threshold_p:.4f}** 다. "
        "폴드가 적으면 부호검정의 최소 p 가 이 문턱보다 커서 **어떤 결과도 통과할 수 없다** "
        f"(폴드 6개면 최소 p=0.031). 이 리포트가 폴드를 {'' if test_months >= 6 else '짧은 검증창으로 '}"
        "늘려 잡은 이유다.\n")

    passed: dict[str, list[int]] = {}
    harmful: dict[str, list[int]] = {}
    for horizon in horizons:
        panels, trig_cols, filt_cols = build_panels(
            candles_by_code, interval=interval, horizon=horizon, exclude_tags=exclude_tags
        )
        result = run_wf(
            panels=panels, panel_columns=(trig_cols, filt_cols), horizon=horizon,
            train_months=train_months, test_months=test_months, step_months=step_months,
            scheme=scheme, top_k=top_k, min_events=min_events, min_test_events=min_test_events,
        )
        stats = result.combo_stats
        min_sign_p = 2 * 0.5 ** stats["n_folds"]

        add(f"## 보유 {horizon}봉 — 폴드 {len(result.folds)}개\n")
        add("### 조합 수준\n")
        add(f"- {result.combo_verdict}")
        add(f"- 참고(상관 무시한 집계): {stats['pooled_survived']}/{stats['pooled_selected']}건\n")
        add(_md_table(result.combo_by_fold[["label", "n_selected", "n_survived", "survival_rate"]].round(3)))
        add("")

        add("### 필터 수준\n")
        summary = result.filter_summary.copy()
        # 부호검정은 방향을 가리지 않는다 — 12/12 양수도, 0/12 양수도 p 가 똑같이 작다.
        # 후자는 "일관되게 해로운" 필터이므로 도움이 되는 필터와 반드시 구분해야 한다.
        significant = summary["sign_p"] < threshold_p
        summary["판정"] = np.where(
            significant & (summary["consistency"] > 0.5), "도움(일관)",
            np.where(significant & (summary["consistency"] < 0.5), "해로움(일관)", "-"),
        )
        for name in summary[summary["판정"] == "도움(일관)"].index:
            passed.setdefault(name, []).append(horizon)
        for name in summary[summary["판정"] == "해로움(일관)"].index:
            harmful.setdefault(name, []).append(horizon)
        cols = ["axis", "n_folds", "folds_positive", "consistency", "median_lift",
                "mean_improve_rate", "sign_p", "판정"]
        add(_md_table(summary[cols].round(4)))
        add(f"\n부호검정 최소 가능 p = {min_sign_p:.5f} (폴드 {stats['n_folds']}개). "
            f"보정 문턱 {threshold_p:.4f} {'보다 작으므로 통과 가능' if min_sign_p < threshold_p else '보다 크므로 통과 불가 — 검정력 부족'}.\n")

    add("## 결론\n")
    if passed:
        add("### 일관되게 **도움이 된** 필터 (보정 문턱 통과 + 과반 폴드에서 양수)\n")
        add("| 필터 | 통과한 보유기간 |")
        add("| --- | --- |")
        for name, hs in sorted(passed.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            add(f"| `{name}` | {', '.join(f'{h}봉' for h in hs)} ({len(hs)}/{len(horizons)}) |")
        add("")
        add("이것이 이 프로젝트에서 **다중검정 보정과 롤링 워크포워드를 모두 통과한 "
            "최초의 결과**다. 다만 필터는 매매 신호가 아니라 조건이다. "
            "실제 채택 전에 거래 시뮬로 비용 차감 후 성적을 확인해야 한다.\n")
    else:
        add("일관되게 도움이 된 필터가 없다.\n")

    if harmful:
        add("### 일관되게 **해로웠던** 필터\n")
        add("부호검정은 방향을 가리지 않는다 — 모든 폴드에서 양수인 것과 모든 폴드에서 "
            "음수인 것의 p 는 똑같이 작다. 아래는 후자다. 그 자체로도 정보다: "
            "이 조건을 거는 대신 **반대 조건**을 거는 편이 나았다는 뜻이다.\n")
        add("| 필터 | 해로웠던 보유기간 |")
        add("| --- | --- |")
        for name, hs in sorted(harmful.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            add(f"| `{name}` | {', '.join(f'{h}봉' for h in hs)} ({len(hs)}/{len(horizons)}) |")
        add("")

    add("## 한계\n")
    add("- 폴드들은 학습창을 겹쳐 쓴다. 검증창은 겹치지 않지만 완전히 독립은 아니다.")
    add("- 이 기간(약 5년)이 특정 레짐일 수 있다. 다른 시장 국면에서 재확인이 필요하다.")
    add("- 현재 상장 종목만 담겨 **생존 편향**이 남아 있다. 유니버스를 넓혀도 "
        "상장폐지 종목이 없으면 이 편향은 사라지지 않는다.")
    add("- 초과수익(edge) 기준이며 매매비용 차감 전이다. 보유가 짧을수록 비용 "
        "비중이 커진다 — 왕복 28bp 는 3봉 보유 lift 를 대부분 상쇄한다.")
    add("- 이 리포트는 매매 권유가 아니다.\n")

    return "\n".join(out)


def write_walkforward(
    candles_by_code: dict[str, pd.DataFrame], path: str | Path, **kwargs
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_walkforward(candles_by_code, **kwargs), encoding="utf-8")
    return target


# =====================================================================
# 팩터 리포트 — 이진 필터가 아니라 연속 변수로
# =====================================================================

FACTOR_DEFAULT = ("atrp_14", "ret_120", "ema60_gap", "ret_20", "ret_5", "rsi_14", "rangepos_20")


def build_factor(
    candles_by_code: dict[str, pd.DataFrame],
    *,
    interval: Interval = Interval.D1,
    horizons: tuple[int, ...] = (5, 20),
    factors: tuple[str, ...] = FACTOR_DEFAULT,
    n_buckets: int = 10,
    control: str = "ret_120",
    alpha: float = 0.05,
    flow_by_code: dict[str, pd.DataFrame] | None = None,
) -> str:
    """횡단면 팩터 리포트.

    flow_by_code 를 주면 투자자별 수급 팩터(기관·외국인 순매수 비중)가 함께 들어간다.
    """
    from .factor import (
        TRADE_FLOW_FACTORS, build_factor_panel, dose_response, double_sort,
        factor_correlations, market_regression,
    )
    from .validation import deflated_threshold

    if flow_by_code:
        factors = tuple(factors) + TRADE_FLOW_FACTORS

    out: list[str] = []
    add = out.append
    threshold = deflated_threshold(len(factors), alpha)

    add(f"# 횡단면 팩터 리포트 — {len(candles_by_code)}종목 ({interval.value})\n")
    add(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 팩터 {len(factors)}개 동시 검정 → 보정 문턱 |t| ≥ {threshold:.2f}\n")

    add("## 0. 이 리포트가 거르는 네 가지\n")
    add("| 함정 | 증상 | 대응 |")
    add("| --- | --- | --- |")
    add("| 시장 드리프트 | 보유기간을 늘리면 아무거나 유의해짐 | 같은 날 전 종목 평균을 뺀 초과수익으로 검정 |")
    add("| 횡단면 상관 | 같은 날 100종목이 동시에 신호 → 표본 100개로 셈 | 날짜 군집 보정 |")
    add("| **전방수익률 겹침** | 20일 수익률을 매일 계산하면 19/20을 공유 | **겹치지 않는 표본으로 t 재계산** |")
    add("| **베타** | 횡단면 평균을 빼도 고베타는 상승장에서 더 오름 | **시장수익률에 회귀해 알파/베타 분리** |")
    add("")

    for horizon in horizons:
        panel = build_factor_panel(
            candles_by_code, interval=interval, horizon=horizon, flow_by_code=flow_by_code
        )
        add(f"## 보유 {horizon}봉\n")
        add(f"표본 {len(panel.frame):,}행 · {panel.codes}종목 · {panel.days}일\n")

        rows = []
        for factor in factors:
            meta = dose_response(panel, factor, n_buckets=n_buckets).attrs
            reg = market_regression(panel, factor, n_buckets=n_buckets)
            rows.append({
                "factor": factor,
                "구분": "수급" if factor in TRADE_FLOW_FACTORS else "가격",
                "단조성ρ": meta["monotone_rho"],
                "스프레드%": abs(meta["spread_mean"]) * 100,
                "겹침t": abs(meta["spread_t_overlap"]),
                "비겹침t": abs(meta["spread_t"]),
                "알파%": reg["alpha"] * 100,
                "알파t": abs(reg["alpha_t_nonoverlap"]),
                "베타": reg["beta"],
                "시장R2": reg["market_r2"],
                "판정": "채택후보" if abs(reg["alpha_t_nonoverlap"]) >= threshold else "기각",
            })
        table = pd.DataFrame(rows).set_index("factor").sort_values("알파t", ascending=False)
        add(_md_table(table.round(3)))

        worst = table["겹침t"].idxmax()
        add(f"\n> 겹침 보정만으로 `{worst}` 의 t 가 "
            f"{table.loc[worst, '겹침t']:.2f} → {table.loc[worst, '비겹침t']:.2f} 로 내려간다. "
            f"베타까지 걷어내면 알파 t 는 {table.loc[worst, '알파t']:.2f} 다.\n")

    add("## 팩터 간 순위상관\n")
    add("상관이 높으면 같은 것을 다르게 부르고 있을 뿐이다. "
        "여러 팩터가 '유의'해 보여도 실제로는 하나를 여러 번 센 것일 수 있다.\n")
    panel = build_factor_panel(
        candles_by_code, interval=interval, horizon=horizons[-1], flow_by_code=flow_by_code
    )
    add(_md_table(factor_correlations(panel, factors)))
    add("")

    add(f"## 이중정렬 — `{control}` 을 통제하면\n")
    add(f"어떤 팩터가 사실은 `{control}` 의 다른 이름일 수 있다. "
        f"`{control}` 분위 **안에서** 그 팩터의 스프레드가 여전히 벌어지는지 본다. "
        "평평해지면 새로 발견한 것이 아니라 같은 것을 재발견한 것이다.\n")
    for factor in ("ema60_gap", "atrp_14"):
        if factor == control or factor not in factors:
            continue
        table = double_sort(panel, factor, control, n_buckets=5)
        add(f"### `{factor}` × `{control}` (셀 값 = 평균 초과수익 %)\n")
        add(_md_table((table * 100).round(3)))
        spreads = table.attrs["spreads"]
        add(f"\n통제 분위별 스프레드 t: "
            f"{', '.join(f'{v:+.2f}' for v in spreads['t'])} — "
            f"{'방향이 일정하지 않다' if spreads['t'].gt(0).nunique() > 1 else '방향이 일정하다'}\n")

    add("## 한계\n")
    if flow_by_code:
        add("- 수급은 **기관·외국인 순매매량 / 그날 거래량** 비중이다. 개인은 잔차이므로 "
            "따로 넣지 않았다 (기관+외국인의 부호를 뒤집은 것과 거의 같다).")
    add("- 현재 상장 종목만 담겨 **생존 편향**이 남아 있다.")
    add("- 매매비용 차감 전이다. 분위 재구성 회전율까지 반영하면 더 나빠진다.")
    add("- 이 표본 기간이 특정 레짐일 수 있다.")
    add("- 이 리포트는 매매 권유가 아니다.\n")

    return "\n".join(out)


def write_factor(candles_by_code: dict[str, pd.DataFrame], path: str | Path, **kwargs) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_factor(candles_by_code, **kwargs), encoding="utf-8")
    return target


# =====================================================================
# 패턴 리포트 — 캘린더-타임 포트폴리오 기반
# =====================================================================

def build_pattern(
    candles_by_code: dict[str, pd.DataFrame],
    *,
    holdings: tuple[int, ...] = (5, 10, 20, 60),
    cost_bps: float = 28.0,
    n_periods: int = 4,
    alpha: float = 0.05,
    fdr_alpha: float = 0.10,
    min_events: int = 100,
) -> str:
    """차트 형태 패턴 리포트 — 여러 패턴을 같은 잣대로 줄 세운다."""
    from dataclasses import replace

    from ..signals.patterns import (
        PATTERNS, CupHandleParams, cup_with_handle, cup_with_handle_loose,
    )
    from .eventstudy import calendar_time_portfolio, compare_holdings
    from .validation import benjamini_hochberg, deflated_threshold, p_from_t

    variants: dict[str, object] = dict(PATTERNS)
    variants["완화컵(대조군)"] = cup_with_handle_loose

    out: list[str] = []
    add = out.append

    add(f"# 차트 패턴 리포트 — {len(variants)}개 패턴 · {len(candles_by_code)}종목\n")
    add(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 왕복 매매비용 {cost_bps:.0f}bp 차감\n")

    add("## 0. 방법\n")
    add("**캘린더-타임 포트폴리오.** \"신호 후 60일 수익률\"을 이벤트마다 계산하면 창이 겹친다. "
        "겹침을 피하려고 이벤트를 띄워 고르면 표본의 대부분을 버린다 "
        "(실측: 1,900여 개 → 약 20개). 그래서 관점을 뒤집어 **매일의 포트폴리오 수익률**을 본다 — "
        "지난 h일 안에 신호가 뜬 종목을 동일가중으로 담고 그날의 초과수익을 기록한다. "
        "일별 계열은 겹치지 않으므로 t검정이 유효하고 표본도 버리지 않는다.\n")
    add("- t 는 **Newey-West** 보정값이다 (포트폴리오가 매일 거의 같은 종목을 들고 있으므로).")
    add("- 벤치마크는 **유니버스 전체**의 그날 평균이다 (이벤트 종목만으로 내면 왜곡된다).")
    add("- 파라미터는 원저자 기준을 그대로 옮겼고 이 데이터로 튜닝하지 않았다.\n")

    events = {}
    for name, fn in variants.items():
        events[name] = {code: fn(candles) for code, candles in candles_by_code.items()}

    rows = []
    for name, evs in events.items():
        total = sum(int(series.sum()) for series in evs.values())
        if total < min_events:
            continue
        table = compare_holdings(evs, candles_by_code, holdings=holdings, cost_bps=cost_bps)
        for hold, row in table.iterrows():
            rows.append({"패턴": name, "보유": hold, "이벤트": int(row["이벤트"]),
                         "평균보유종목": row["평균보유종목"],
                         "연환산%": row["연환산%"], "t": row["t"]})

    frame = pd.DataFrame(rows)
    threshold = deflated_threshold(len(frame), alpha)
    frame["p"] = p_from_t(frame["t"].to_numpy(), None)
    frame["BH통과"] = benjamini_hochberg(frame["p"].to_numpy(), fdr_alpha) & (frame["t"] > 0)
    frame["Šidák통과"] = frame["t"] >= threshold

    add("## 1. 전체 결과\n")
    add(f"가설 {len(frame)}개(패턴 × 보유기간) 동시검정.\n")
    add(f"- **Šidák** 문턱 |t| ≥ {threshold:.2f} — 하나라도 틀리면 안 된다(FWER)를 통제. 보수적.")
    add(f"- **BH-FDR**(α={fdr_alpha:.0%}) — 채택한 것 중 몇 %가 가짜여도 되는가를 통제. "
        "스크리닝에 적합.\n")
    ranked = frame.sort_values("t", ascending=False).set_index(["패턴", "보유"])
    add(_md_table(ranked[["이벤트", "평균보유종목", "연환산%", "t", "p", "BH통과", "Šidák통과"]].round(4)))
    survivors = ranked[ranked["BH통과"]]
    add(f"\nBH-FDR 통과: **{', '.join(f'{a}({b}일)' for a, b in survivors.index) or '없음'}**\n")

    add("## 2. 기간 안정성\n")
    add("전체 기간 하나로는 한 구간이 다 벌어준 것을 구분할 수 없다.\n")
    all_days = sorted(set().union(*[c.index for c in candles_by_code.values()]))
    cuts = [all_days[i * len(all_days) // n_periods] for i in range(n_periods)] + [all_days[-1]]
    best_hold = holdings[-1]

    stability = []
    for name, evs in events.items():
        record: dict[str, object] = {"패턴": name}
        positive = 0
        for i in range(n_periods):
            window = {
                code: candles[(candles.index >= cuts[i]) & (candles.index <= cuts[i + 1])]
                for code, candles in candles_by_code.items()
            }
            window = {code: c for code, c in window.items() if len(c) > 60}
            sliced = {code: evs[code].reindex(c.index).fillna(False) for code, c in window.items()}
            result = calendar_time_portfolio(sliced, window, holding_days=best_hold, cost_bps=cost_bps)
            value = result.annualized * 100 if not result.daily.empty else float("nan")
            record[f"{cuts[i]:%y-%m}~"] = value
            positive += int(np.isfinite(value) and value > 0)
        record["양수구간"] = f"{positive}/{n_periods}"
        stability.append(record)
    add(f"보유 {best_hold}일 · 연환산 %\n")
    add(_md_table(pd.DataFrame(stability).set_index("패턴").round(1)))
    add("")

    add("## 3. 파라미터 민감도 — 컵앤핸들\n")
    add("기준값을 조금 바꿔도 결과가 유지되는지, 그리고 **어떤 조건을 풀었을 때 무너지는가**를 본다. "
        "후자가 그 조건의 중요도를 말해준다.\n")
    base = CupHandleParams()
    perturbations = {
        "기준(오닐)": base,
        "컵 최소 25봉": replace(base, cup_min=25),
        "컵 최소 50봉": replace(base, cup_min=50),
        "컵깊이 ≥8%": replace(base, cup_depth_min=0.08),
        "컵깊이 ≥18%": replace(base, cup_depth_min=0.18),
        "회복률 88%": replace(base, rim_recovery=0.88),
        "회복률 97%": replace(base, rim_recovery=0.97),
        "핸들 3~15봉": replace(base, handle_min=3, handle_max=15),
        "핸들 8~30봉": replace(base, handle_min=8, handle_max=30),
        "돌파거래량 1.0배": replace(base, breakout_volume=1.0),
        "돌파거래량 2.0배": replace(base, breakout_volume=2.0),
        "선행상승 조건 해제": replace(base, prior_gain=0.0),
        "선행상승 40%": replace(base, prior_gain=0.40),
        "U자 조건 해제": replace(base, trough_center=0.9),
        "핸들위치 조건 해제": replace(base, handle_upper_half=0.0),
        "봉우리 조건 해제": replace(base, rim_is_peak=0.0, rim_position=1.0),
    }
    sensitivity = []
    for label, params in perturbations.items():
        evs = {code: cup_with_handle(c, params) for code, c in candles_by_code.items()}
        result = calendar_time_portfolio(evs, candles_by_code, holding_days=best_hold, cost_bps=cost_bps)
        sensitivity.append({"변형": label, "이벤트": result.n_events,
                            "연환산%": result.annualized * 100, "t": result.t_stat})
    table = pd.DataFrame(sensitivity).set_index("변형")
    add(_md_table(table.round(2)))
    add(f"\n연환산 양수 {int((table['연환산%'] > 0).sum())}/{len(table)} · "
        f"t>2 {int((table['t'] > 2).sum())}/{len(table)}\n")
    weakest = table["t"].idxmin()
    add(f"> 가장 크게 무너지는 변형은 `{weakest}` (t={table.loc[weakest, 't']:.2f}, "
        f"연환산 {table.loc[weakest, '연환산%']:.1f}%). 그 조건이 이 패턴의 핵심이다.\n")

    add("## 4. 한계\n")
    add(f"- 살아남은 것이 있다면 **{best_hold}일 보유에서만**이다. 짧은 보유는 유의하지 않다 — "
        "스윙이지 단타가 아니다.")
    add("- 패턴을 더 시험할수록 문턱이 올라간다. 같은 결과라도 나중에 잰 가설이 많으면 기각될 수 있다.")
    add("- 개별 구간의 t 는 대개 유의 수준에 못 미친다. 전체를 합쳐야 문턱을 넘는다.")
    add("- 이벤트가 겹쳐 수십 종목을 동시 보유한다. 그만한 자금과 분산이 전제다.")
    add("- 현재 상장 종목만 담겨 **생존 편향**이 남아 있다.")
    add("- 이 리포트는 매매 권유가 아니다.\n")

    return "\n".join(out)


def write_pattern(candles_by_code: dict[str, pd.DataFrame], path: str | Path, **kwargs) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_pattern(candles_by_code, **kwargs), encoding="utf-8")
    return target
