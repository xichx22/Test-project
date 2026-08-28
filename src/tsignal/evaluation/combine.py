"""신호 결합 탐색 — "트리거 × 상태필터" 조합에서 살아남는 것 찾기.

단독 트리거가 전부 기각된 뒤의 다음 가설: 트리거는 시장 상태를 구분하지 않아서
서로 다른 사건을 한 통에 넣고 세고 있었다. 상태 필터로 조건을 좁히면
엣지가 드러날 수 있다.

동시에 이 탐색은 **과최적화 기계**다. 조합 2,800개를 훑으면 엣지가 전혀 없어도
t>3 짜리가 수십 개 나온다. 그래서 프로토콜을 이렇게 고정한다.

  1. 탐색 공간을 먼저 줄인다 — 같은 축의 필터끼리는 결합하지 않고, 필터는 최대 2개
  2. 실제로 시험한 조합 수만큼 문턱을 올린다 (Šidák) + BH-FDR 을 나란히 본다
  3. **IS 에서 고르고 OOS 에서 확인한다.** IS 상위 K개가 OOS 에서 몇 개나
     살아남는지를, 우연히 기대되는 개수와 비교한다 ← 이게 최종 판정
  4. 단독 트리거 대비 **증분(lift)** 을 함께 본다. 필터가 트리거를 개선하지
     못하면 그 조합은 의미가 없다
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import indicators as ind
from .. import signals as sig
from ..datasource.base import Interval
from ..signals import filters as filt
from . import metrics, validation
from .forward import forward_returns

# 필터 없는 조합을 나타내는 표기
BARE = "(필터없음)"


@dataclass
class Combo:
    trigger: str
    filters: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.trigger} + {' & '.join(self.filters)}" if self.filters else f"{self.trigger} + {BARE}"


@dataclass
class SymbolPanel:
    """한 종목의 트리거·필터·전방수익률을 미리 계산해 담아 둔 판.

    워크포워드는 구간을 옮겨가며 같은 계산을 반복한다. 지표를 폴드마다 다시
    계산하면 199종목 × 6폴드 = 몇 분씩 걸린다. 종목당 한 번만 계산해 두고
    구간을 잘라 쓰면 초 단위로 끝난다.

    구간을 잘라 써도 미래참조가 생기지 않는다 — 지표는 인과적이라 t 시점 값이
    t 이전 정보만 쓰기 때문이다. 오히려 창 시작 이전의 이력까지 반영되므로
    실제 트레이더가 보는 값에 더 가깝다 (워밍업 구간 왜곡이 없다).
    """

    code: str
    index: pd.DatetimeIndex
    trig: np.ndarray          # (N, T) bool
    filt: np.ndarray          # (N, F) bool
    fwd: np.ndarray           # (N,) float — NaN 포함
    day: np.ndarray           # (N,) int64 — 에폭 기준 일수. 날짜 군집 보정에 쓴다


def build_panels(
    candles_by_code: Mapping[str, pd.DataFrame],
    *,
    interval: Interval,
    horizon: int = 5,
    entry: str = "next_open",
    exclude_tags: tuple[str, ...] = (),
    trigger_names: Sequence[str] | None = None,
    filter_names: Sequence[str] | None = None,
) -> tuple[list[SymbolPanel], list[str], list[str]]:
    """종목별 판을 한 번에 만든다. 반환 (panels, trigger_names, filter_names)."""
    panels: list[SymbolPanel] = []
    trig_cols: list[str] = []
    filt_cols: list[str] = []

    for code, candles in candles_by_code.items():
        features = ind.compute_all(candles, interval=interval)
        triggers = sig.evaluate_all(
            candles, features, kind="entry", names=trigger_names, exclude_tags=exclude_tags
        )
        states = filt.evaluate_all(candles, features, names=filter_names)
        fwd = forward_returns(candles, (horizon,), entry=entry)[f"fwd_{horizon}"]

        trig_cols, filt_cols = list(triggers.columns), list(states.columns)
        panels.append(SymbolPanel(
            code=code, index=candles.index,
            trig=triggers.to_numpy(bool), filt=states.to_numpy(bool),
            fwd=fwd.to_numpy(float),
            day=candles.index.tz_localize(None).to_numpy().astype("datetime64[D]").astype(np.int64),
        ))
    return panels, trig_cols, filt_cols


class CombinationLab:
    """유니버스 전체를 하나의 평평한 배열로 눌러 담아 조합을 빠르게 평가한다.

    종목마다 지표·트리거·필터를 한 번만 계산해 두고, 조합 평가는 boolean AND 로만
    한다. 조합 수천 개를 훑어도 몇 초면 끝난다.
    """

    def __init__(
        self,
        candles_by_code: Mapping[str, pd.DataFrame] | None = None,
        *,
        interval: Interval | None = None,
        horizon: int = 5,
        entry: str = "next_open",
        exclude_tags: tuple[str, ...] = (),
        trigger_names: Sequence[str] | None = None,
        filter_names: Sequence[str] | None = None,
        panels: Sequence[SymbolPanel] | None = None,
        panel_columns: tuple[list[str], list[str]] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> None:
        self.horizon = horizon
        self.interval = interval

        if panels is None:
            if candles_by_code is None or interval is None:
                raise ValueError("candles_by_code 와 interval, 또는 panels 가 필요합니다.")
            panels, trig_cols, filt_cols = build_panels(
                candles_by_code, interval=interval, horizon=horizon, entry=entry,
                exclude_tags=exclude_tags, trigger_names=trigger_names, filter_names=filter_names,
            )
        else:
            if panel_columns is None:
                raise ValueError("panels 를 넘길 때는 panel_columns 도 함께 넘겨야 합니다.")
            trig_cols, filt_cols = panel_columns

        self.trigger_names, self.filter_names = trig_cols, filt_cols
        self._assemble(panels, start, end)

        self._trig_col = {n: i for i, n in enumerate(self.trigger_names)}
        self._filt_col = {n: i for i, n in enumerate(self.filter_names)}
        self._axis_of = {name: spec.axis for name, spec in filt.REGISTRY.items()}

    @classmethod
    def from_panels(
        cls,
        panels: Sequence[SymbolPanel],
        panel_columns: tuple[list[str], list[str]],
        *,
        horizon: int = 5,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> "CombinationLab":
        """미리 만든 판에서 특정 기간만 잘라 Lab 을 만든다 (워크포워드용)."""
        return cls(panels=panels, panel_columns=panel_columns, horizon=horizon,
                   start=start, end=end)

    def _assemble(
        self, panels: Sequence[SymbolPanel],
        start: pd.Timestamp | None, end: pd.Timestamp | None,
    ) -> None:
        trig_parts, filt_parts, ret_parts, exc_parts, code_parts, day_parts = [], [], [], [], [], []
        self.codes: list[str] = []

        for panel in panels:
            window = np.ones(len(panel.index), dtype=bool)
            if start is not None:
                window &= panel.index >= start
            if end is not None:
                window &= panel.index <= end
            keep = window & np.isfinite(panel.fwd)
            if keep.sum() < 2:
                continue

            fwd = panel.fwd[keep]
            # 기준선은 **그 창 안에서** 계산해야 한다. 전체 기간 평균을 쓰면
            # 창 밖의 정보가 초과수익에 섞여 들어간다.
            baseline = float(fwd.mean())

            self.codes.append(panel.code)
            code_id = len(self.codes) - 1
            trig_parts.append(panel.trig[keep])
            filt_parts.append(panel.filt[keep])
            ret_parts.append(fwd)
            exc_parts.append(fwd - baseline)
            code_parts.append(np.full(int(keep.sum()), code_id, dtype=np.int32))
            day_parts.append(panel.day[keep])

        if not self.codes:
            raise ValueError("해당 구간에 유효한 종목이 없습니다.")

        self.trig = np.concatenate(trig_parts)
        self.filt = np.concatenate(filt_parts)
        self.ret = np.concatenate(ret_parts)
        self.excess = np.concatenate(exc_parts)
        self.code_idx = np.concatenate(code_parts)
        self.day = np.concatenate(day_parts)
        self.n_codes = len(self.codes)

    # --- 조합 평가 -------------------------------------------------------
    def mask(self, combo: Combo) -> np.ndarray:
        out = self.trig[:, self._trig_col[combo.trigger]].copy()
        for name in combo.filters:
            out &= self.filt[:, self._filt_col[name]]
        return out

    def stats(self, combo: Combo) -> dict[str, float | str | int]:
        mask = self.mask(combo)
        n = int(mask.sum())
        row: dict[str, float | str | int] = {
            "combo": combo.name, "trigger": combo.trigger,
            "filters": " & ".join(combo.filters) or BARE, "n_filters": len(combo.filters), "n": n,
        }
        if n < 3:
            row.update({k: np.nan for k in
                        ("expectancy", "edge", "win_rate", "t_edge", "t_naive",
                         "breadth", "n_codes", "n_days")})
            return row

        exc, ret, days = self.excess[mask], self.ret[mask], self.day[mask]
        std = exc.std(ddof=1)
        counts = np.bincount(self.code_idx[mask], minlength=self.n_codes)
        sums = np.bincount(self.code_idx[mask], weights=exc, minlength=self.n_codes)
        seen = counts > 0
        code_means = sums[seen] / counts[seen]

        row.update({
            "expectancy": float(ret.mean()),
            "edge": float(exc.mean()),
            "win_rate": float((ret > 0).mean()),
            # t_edge 는 날짜 군집 보정값이다. 판정은 이것으로만 한다.
            # t_naive 는 보정하지 않았을 때 얼마나 부풀려지는지 보여주기 위해 남긴다.
            "t_edge": metrics.clustered_t_stat(exc, days),
            "t_naive": float(exc.mean() / (std / np.sqrt(n))) if std > 0 else np.nan,
            "breadth": float((code_means > 0).mean()),
            "n_codes": int(seen.sum()),
            "n_days": int(len(np.unique(days))),
        })
        return row

    # --- 탐색 ------------------------------------------------------------
    def dedupe(self, combos: Sequence[Combo]) -> list[Combo]:
        """선택하는 봉 집합이 완전히 같은 조합을 하나로 합친다.

        느슨한 필터(예: `not_overbought` 는 90% 구간에서 참)를 붙이면
        원래 조합과 **글자 그대로 같은 봉 집합**이 나온다. 이걸 별개 조합으로
        세면 두 가지가 깨진다.

          - 다중검정 문턱: 독립이 아닌 가설을 개수에 넣어 문턱이 잘못 계산된다
          - IS→OOS 생존 검정: 같은 조합이 중복 선정돼 생존 수가 이중 계산되고,
            이항검정 p값이 가짜로 낮아진다 (실제로 이 버그를 겪었다)

        같은 마스크끼리는 필터가 가장 적은 것 = 가장 단순한 표현만 남긴다.
        """
        seen: dict[bytes, Combo] = {}
        for combo in combos:
            key = hashlib.blake2b(np.packbits(self.mask(combo)).tobytes(), digest_size=16).digest()
            current = seen.get(key)
            if current is None or (len(combo.filters), combo.name) < (len(current.filters), current.name):
                seen[key] = combo
        return sorted(seen.values(), key=lambda c: (c.trigger, len(c.filters), c.name))

    def enumerate_combos(self, *, max_filters: int = 2) -> list[Combo]:
        """같은 축의 필터끼리는 묶지 않는다 — 공허하거나 중복인 조합을 미리 뺀다."""
        filter_sets: list[tuple[str, ...]] = [()]
        for k in range(1, max_filters + 1):
            for group in combinations(self.filter_names, k):
                if len({self._axis_of[name] for name in group}) == len(group):
                    filter_sets.append(group)
        return [Combo(t, fs) for t in self.trigger_names for fs in filter_sets]

    def search(
        self,
        *,
        max_filters: int = 2,
        min_events: int = 100,
        min_days: int = 60,
        alpha: float = 0.05,
        fdr_alpha: float = 0.10,
        combos: Iterable[Combo] | None = None,
        dedupe: bool = True,
    ) -> pd.DataFrame:
        """모든 조합을 평가하고 두 가지 다중검정 잣대를 붙인다.

        dedupe=True 면 봉 집합이 동일한 조합을 하나로 합친 뒤 검정한다.
        중복을 남기면 문턱과 p값이 모두 왜곡된다 (dedupe() 독스트링 참고).
        """
        combos = list(combos) if combos is not None else self.enumerate_combos(max_filters=max_filters)
        n_raw = len(combos)
        if dedupe:
            combos = self.dedupe(combos)
        table = pd.DataFrame([self.stats(c) for c in combos]).set_index("combo")

        # 단독 트리거 대비 증분. 필터가 트리거를 개선하지 못하면 의미 없는 조합이다.
        bare = table[table["n_filters"] == 0].set_index("trigger")["edge"]
        table["edge_bare"] = table["trigger"].map(bare)
        table["lift"] = table["edge"] - table["edge_bare"]

        # 유효 표본은 신호 발생 횟수가 아니라 **발생한 날짜 수**다.
        # 같은 날 100종목에서 떴어도 독립 관측 100개가 아니다.
        eligible = (table["n"] >= min_events) & (table["n_days"] >= min_days)
        n_trials = int(eligible.sum())
        threshold = validation.deflated_threshold(max(1, n_trials), alpha)

        pvals = np.where(eligible, validation.p_from_t(table["t_edge"].to_numpy(), table["n"].to_numpy()), np.nan)
        table["p_value"] = pvals
        table["pass_sidak"] = eligible & (table["t_edge"] >= threshold)
        table["pass_fdr"] = False
        table.loc[eligible, "pass_fdr"] = validation.benjamini_hochberg(pvals[eligible.to_numpy()], fdr_alpha) & (
            table.loc[eligible, "t_edge"] > 0
        )

        table.attrs.update({
            "n_combos": len(table), "n_combos_raw": n_raw,
            "n_trials": n_trials, "threshold": threshold, "min_days": min_days,
            "alpha": alpha, "fdr_alpha": fdr_alpha, "min_events": min_events,
            "horizon": self.horizon, "n_codes": self.n_codes,
        })
        return table.sort_values("t_edge", ascending=False)


def split_labs(
    candles_by_code: Mapping[str, pd.DataFrame],
    *,
    train_ratio: float = 0.6,
    **lab_kwargs,
) -> tuple[CombinationLab, CombinationLab]:
    """종목마다 같은 비율로 잘라 IS/OOS 두 개의 Lab 을 만든다."""
    is_parts, oos_parts = {}, {}
    for code, candles in candles_by_code.items():
        cut = int(len(candles) * train_ratio)
        is_parts[code], oos_parts[code] = candles.iloc[:cut], candles.iloc[cut:]
    return CombinationLab(is_parts, **lab_kwargs), CombinationLab(oos_parts, **lab_kwargs)


@dataclass
class SelectionResult:
    """IS 에서 고르고 OOS 에서 확인한 결과 — 이 탐색의 최종 판정."""

    table: pd.DataFrame        # 상위 K개의 IS/OOS 나란히
    n_selected: int
    n_survived: int            # OOS 에서 edge>0 이고 t_edge>1 인 개수
    expected_by_chance: float  # 엣지가 없다면 기대되는 생존 개수
    binomial_p: float          # 관측 생존 수가 우연일 확률

    @property
    def verdict(self) -> str:
        if self.n_selected == 0:
            return "선정된 조합 없음"
        if self.binomial_p < 0.05 and self.n_survived > self.expected_by_chance:
            return "OOS 생존이 우연으로 보기 어려움 — 추가 검증 가치 있음"
        return "OOS 생존이 우연 범위 — 조합 탐색으로 얻은 엣지 없음"


def select_and_validate(
    is_lab: CombinationLab,
    oos_lab: CombinationLab,
    *,
    top_k: int = 20,
    max_filters: int = 2,
    min_events: int = 100,
    min_days: int = 60,
    min_oos_events: int = 30,
    min_oos_days: int = 20,
) -> SelectionResult:
    """정직한 프로토콜: IS 성적으로만 고르고, 그 선택을 OOS 로 채점한다.

    "IS 상위 K개 중 OOS 에서 살아남은 개수"를, 엣지가 전혀 없을 때 기대되는
    개수(K/2 — 부호가 반반이므로)와 이항검정으로 비교한다.
    """
    # 중복 제거는 필수다. 같은 봉 집합을 두 번 세면 생존 수가 이중 계산돼
    # 이항검정 p값이 가짜로 낮아진다.
    is_table = is_lab.search(max_filters=max_filters, min_events=min_events,
                             min_days=min_days, dedupe=True)
    picked = is_table[(is_table["n"] >= min_events) & (is_table["n_days"] >= min_days)].head(top_k)

    combos = [Combo(row["trigger"], tuple(f for f in str(row["filters"]).split(" & ") if f != BARE))
              for _, row in picked.iterrows()]
    oos_rows = pd.DataFrame([oos_lab.stats(c) for c in combos]).set_index("combo")

    merged = picked[["n", "n_days", "edge", "t_edge", "breadth", "lift"]].add_suffix("_is").join(
        oos_rows[["n", "n_days", "edge", "t_edge", "breadth"]].add_suffix("_oos")
    )
    merged["survived"] = (
        (merged["n_oos"] >= min_oos_events) & (merged["n_days_oos"] >= min_oos_days)
        & (merged["edge_oos"] > 0) & (merged["t_edge_oos"] > 1.0)
    )

    n_selected = len(merged)
    n_survived = int(merged["survived"].sum())
    # 엣지가 없다면 OOS edge 부호는 반반, t>1 조건까지 더하면 생존확률 ≈ 0.16.
    p_null = 0.16
    expected = n_selected * p_null
    binomial_p = _binomial_tail(n_survived, n_selected, p_null)

    return SelectionResult(merged, n_selected, n_survived, expected, binomial_p)


def _binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p). 조합 수가 작아 정확 계산으로 충분하다."""
    from math import comb

    if n == 0:
        return np.nan
    return float(sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


# =====================================================================
# 필터 단위 분석 — 조합 하나를 고르는 것보다 훨씬 잘 뒷받침되는 질문
# =====================================================================

def filter_contribution(
    lab: CombinationLab,
    *,
    min_events: int = 80,
) -> pd.DataFrame:
    """필터별 기여도: 27개 트리거 전부에 그 필터를 걸었을 때 lift 가 어떻게 되는가.

    조합 2,835개 중 최고를 고르는 것은 가설을 2,835개 세우는 짓이다.
    "필터 X 는 트리거 종류와 무관하게 도움이 되는가"는 가설이 15개뿐이라
    같은 데이터로도 훨씬 강한 결론을 낼 수 있다.

    검정은 **부호검정(sign test)** 을 쓴다. 트리거별 lift 는 표본이 겹쳐
    서로 독립이 아니므로 t검정은 유의성을 부풀린다. "몇 개 트리거에서
    개선됐는가"는 그 상관에 훨씬 덜 휘둘린다.

    컬럼
      n_pairs      : lift 를 계산할 수 있었던 트리거 수
      n_improved   : 그중 lift > 0 인 개수
      median_lift  : 트리거별 lift 의 중앙값
      mean_lift    : 평균 (이상치에 민감하므로 중앙값과 같이 볼 것)
      sign_p       : 부호검정 p값 (귀무: 개선 확률 0.5)
      pooled_n     : 그 필터가 걸린 조합들의 총 표본 수
    """
    bare = {t: lab.stats(Combo(t, ())) for t in lab.trigger_names}
    rows = []
    for name in lab.filter_names:
        lifts, pooled = [], 0
        for trigger in lab.trigger_names:
            base = bare[trigger]
            with_filter = lab.stats(Combo(trigger, (name,)))
            if (base["n"] < min_events or with_filter["n"] < min_events
                    or not np.isfinite(base["edge"]) or not np.isfinite(with_filter["edge"])):
                continue
            lifts.append(float(with_filter["edge"]) - float(base["edge"]))
            pooled += int(with_filter["n"])

        if not lifts:
            continue
        arr = np.asarray(lifts)
        improved = int((arr > 0).sum())
        rows.append({
            "filter": name,
            "axis": filt.REGISTRY[name].axis,
            "n_pairs": len(arr),
            "n_improved": improved,
            "improve_rate": improved / len(arr),
            "median_lift": float(np.median(arr)),
            "mean_lift": float(arr.mean()),
            "sign_p": _binomial_two_sided(improved, len(arr), 0.5),
            "pooled_n": pooled,
        })

    out = pd.DataFrame(rows).set_index("filter")
    return out.sort_values("median_lift", ascending=False)


def _binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """양측 이항검정 p값."""
    from math import comb

    if n == 0:
        return np.nan
    pmf = [comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    observed = pmf[k]
    return float(sum(v for v in pmf if v <= observed + 1e-12))


def compare_filter_contribution(
    is_lab: CombinationLab, oos_lab: CombinationLab, *, min_events: int = 80
) -> pd.DataFrame:
    """필터 기여도를 IS/OOS 로 나눠 비교한다. 두 구간에서 같은 방향이어야 믿을 수 있다."""
    left = filter_contribution(is_lab, min_events=min_events)
    right = filter_contribution(oos_lab, min_events=max(20, min_events // 3))
    cols = ["n_pairs", "improve_rate", "median_lift", "sign_p"]
    merged = left[["axis"] + cols].join(right[cols].add_suffix("_oos"), how="left")
    merged["consistent"] = (
        (np.sign(merged["median_lift"]) == np.sign(merged["median_lift_oos"]))
        & (merged["improve_rate"] > 0.5) & (merged["improve_rate_oos"] > 0.5)
    )
    return merged.sort_values("median_lift", ascending=False)
