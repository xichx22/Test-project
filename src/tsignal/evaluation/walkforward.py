"""롤링 워크포워드 — IS/OOS 분할을 한 번이 아니라 여러 번 한다.

분할을 한 번만 하면 결론이 "어느 날짜에 잘랐는가"에 걸린다. 특히 변동성·레짐
필터는 IS 구간과 OOS 구간의 시장 성격 차이를 그대로 반영해버린다.

창을 밀어가며 같은 검증을 반복하고, **폴드를 넘어 방향이 유지되는지**를 본다.
한두 폴드에서 좋은 것은 우연이고, 대부분의 폴드에서 같은 방향이면 신호다.

두 가지 방식
  rolling  : 학습창이 고정 길이로 따라 움직인다 (오래된 레짐을 잊는다)
  anchored : 학습창이 처음부터 누적된다 (표본이 계속 늘어난다)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from ..datasource.base import Interval
from .combine import (
    CombinationLab,
    SymbolPanel,
    build_panels,
    filter_contribution,
    select_and_validate,
)

Scheme = Literal["rolling", "anchored"]


@dataclass(frozen=True)
class Fold:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def label(self) -> str:
        return f"F{self.idx}: 학습 {self.train_start:%y-%m}~{self.train_end:%y-%m} / 검증 {self.test_start:%y-%m}~{self.test_end:%y-%m}"


def make_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int | None = None,
    scheme: Scheme = "rolling",
) -> list[Fold]:
    """날짜 기준으로 폴드를 만든다.

    행 번호가 아니라 **날짜**로 자른다. 종목마다 상장일이 달라 행 수가 다르므로,
    행으로 자르면 폴드마다 다른 기간을 보게 된다.
    """
    step = step_months or test_months
    folds: list[Fold] = []
    train_start = pd.Timestamp(start)
    idx = 1

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        folds.append(Fold(
            idx=idx,
            train_start=pd.Timestamp(start) if scheme == "anchored" else train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
        ))
        train_start = train_start + pd.DateOffset(months=step)
        idx += 1
    return folds


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    combo_by_fold: pd.DataFrame       # 폴드별 선정/생존 수
    filter_by_fold: pd.DataFrame      # 폴드 × 필터 검증구간 성적
    filter_summary: pd.DataFrame      # 필터별 폴드 통과 집계
    scheme: Scheme
    horizon: int
    meta: dict = field(default_factory=dict)

    NULL_SURVIVAL_RATE = 0.16     # 엣지가 없을 때 기대되는 생존률 (t>1 의 단측 확률)

    @property
    def combo_stats(self) -> dict[str, float]:
        """폴드를 독립 단위로 본 집계.

        **조합을 단위로 세면 안 된다.** 한 폴드 안의 조합 20개는 표본이 크게 겹쳐
        서로 독립이 아니다 — 실제로 한 폴드에서 1/20 이 살고 다른 폴드에서 19/20 이
        사는 식으로 함께 움직인다. 조합 120건을 독립 시행으로 놓고 이항검정을 하면
        p 가 가짜로 작아진다.

        그래서 폴드별 생존률이 귀무 생존률(0.16)을 넘는지를 세고, 폴드 사이에서만
        부호검정을 한다. 검정력은 낮아지지만 이쪽이 정직하다.
        """
        from .combine import _binomial_two_sided

        rates = self.combo_by_fold["survival_rate"].dropna()
        above = int((rates > self.NULL_SURVIVAL_RATE).sum())
        return {
            "n_folds": len(rates),
            "folds_above_null": above,
            "median_rate": float(rates.median()) if len(rates) else np.nan,
            "pooled_selected": int(self.combo_by_fold["n_selected"].sum()),
            "pooled_survived": int(self.combo_by_fold["n_survived"].sum()),
            "sign_p": _binomial_two_sided(above, len(rates), 0.5) if len(rates) else np.nan,
        }

    @property
    def combo_verdict(self) -> str:
        stats = self.combo_stats
        if not stats["n_folds"]:
            return "선정된 조합 없음"
        head = (f"폴드 {stats['n_folds']}개 중 {stats['folds_above_null']}개에서 "
                f"생존률이 귀무값 {self.NULL_SURVIVAL_RATE:.0%} 를 상회 "
                f"(중앙값 {stats['median_rate']:.0%}, 부호검정 p={stats['sign_p']:.3f})")
        if stats["sign_p"] < 0.05 and stats["folds_above_null"] > stats["n_folds"] / 2:
            return head + " — 우연으로 보기 어려움"
        return head + " — 우연 범위"


def run(
    candles_by_code: Mapping[str, pd.DataFrame] | None = None,
    *,
    interval: Interval = Interval.D1,
    horizon: int = 5,
    exclude_tags: tuple[str, ...] = (),
    train_months: int = 24,
    test_months: int = 6,
    step_months: int | None = None,
    scheme: Scheme = "rolling",
    top_k: int = 20,
    max_filters: int = 2,
    min_events: int = 100,
    min_test_events: int = 30,
    baseline: str = "cross_sectional",
    panels: Sequence[SymbolPanel] | None = None,
    panel_columns: tuple[list[str], list[str]] | None = None,
) -> WalkForwardResult:
    """워크포워드 실행.

    panels 를 넘기면 지표를 다시 계산하지 않는다 — 보유기간별로 여러 번 돌릴 때
    이 재사용이 없으면 대부분의 시간을 같은 계산에 쓴다.
    """
    if panels is None:
        if candles_by_code is None:
            raise ValueError("candles_by_code 또는 panels 가 필요합니다.")
        panels, trig_cols, filt_cols = build_panels(
            candles_by_code, interval=interval, horizon=horizon, exclude_tags=exclude_tags
        )
    else:
        if panel_columns is None:
            raise ValueError("panels 를 넘길 때는 panel_columns 도 필요합니다.")
        trig_cols, filt_cols = panel_columns

    span_start = min(p.index[0] for p in panels)
    span_end = max(p.index[-1] for p in panels)
    folds = make_folds(span_start, span_end, train_months=train_months,
                       test_months=test_months, step_months=step_months, scheme=scheme)
    if not folds:
        raise ValueError(
            f"폴드를 만들 수 없습니다. 데이터 기간 {span_start:%Y-%m}~{span_end:%Y-%m} 에 비해 "
            f"train_months={train_months} + test_months={test_months} 가 깁니다."
        )

    combo_rows, filter_rows = [], []
    for fold in folds:
        train = CombinationLab.from_panels(
            panels, (trig_cols, filt_cols), horizon=horizon,
            start=fold.train_start, end=fold.train_end, baseline=baseline,
        )
        test = CombinationLab.from_panels(
            panels, (trig_cols, filt_cols), horizon=horizon,
            start=fold.test_start, end=fold.test_end, baseline=baseline,
        )

        selection = select_and_validate(
            train, test, top_k=top_k, max_filters=max_filters,
            min_events=min_events, min_oos_events=min_test_events,
        )
        combo_rows.append({
            "fold": fold.idx, "label": fold.label(),
            "n_train": len(train.excess), "n_test": len(test.excess),
            "n_selected": selection.n_selected, "n_survived": selection.n_survived,
            "survival_rate": (selection.n_survived / selection.n_selected
                              if selection.n_selected else np.nan),
            "binomial_p": selection.binomial_p,
        })

        contrib = filter_contribution(test, min_events=min_test_events)
        for name, row in contrib.iterrows():
            filter_rows.append({
                "fold": fold.idx, "filter": name, "axis": row["axis"],
                "n_pairs": row["n_pairs"], "improve_rate": row["improve_rate"],
                "median_lift": row["median_lift"],
            })

    combo_by_fold = pd.DataFrame(combo_rows).set_index("fold")
    filter_by_fold = pd.DataFrame(filter_rows)

    summary = (
        filter_by_fold.groupby(["filter", "axis"])
        .agg(
            n_folds=("median_lift", "size"),
            folds_positive=("median_lift", lambda s: int((s > 0).sum())),
            median_lift=("median_lift", "median"),
            mean_improve_rate=("improve_rate", "mean"),
            worst_lift=("median_lift", "min"),
            best_lift=("median_lift", "max"),
        )
        .reset_index()
        .set_index("filter")
    )
    summary["consistency"] = summary["folds_positive"] / summary["n_folds"]
    from .combine import _binomial_two_sided

    summary["sign_p"] = [
        _binomial_two_sided(int(r.folds_positive), int(r.n_folds), 0.5)
        for r in summary.itertuples()
    ]
    summary = summary.sort_values(["consistency", "median_lift"], ascending=False)

    return WalkForwardResult(
        folds=folds, combo_by_fold=combo_by_fold, filter_by_fold=filter_by_fold,
        filter_summary=summary, scheme=scheme, horizon=horizon,
        meta={"n_codes": len(panels), "train_months": train_months,
              "test_months": test_months, "span": (span_start, span_end),
              "baseline": baseline},
    )
