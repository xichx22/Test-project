"""팩터 분석 — 이진 필터를 연속 변수로 바꿔서 본다.

`trend_down`(종가 < 60EMA)이 12폴드 전부에서 도움이 됐다. 하지만 이진 분할은
정보를 버린다. 60EMA 를 1% 밑도는 것과 30% 밑도는 것이 같은 취급을 받는다.

여기서 묻는 것은 둘이다.

  1. **용량-반응**: 이탈이 깊을수록 단조적으로 좋아지는가?
     진짜 효과라면 단조성이 나와야 한다. 우연히 만들어진 이진 분할은
     보통 단조적이지 않다. 임의의 문턱(60EMA)에 기대지 않는 검사이기도 하다.

  2. **이미 알려진 팩터인가**: "최근 많이 빠진 종목이 반등한다"는 단기 반전
     (short-term reversal)은 학계에 잘 알려져 있다. `trend_down` 이 그걸
     다시 발견한 것뿐이라면 그렇게 말해야 한다. 과거 수익률을 통제한 뒤에도
     남는지를 이중 정렬로 확인한다.

횡단면(cross-sectional) 순위로 본다
-----------------------------------
"60EMA 대비 -5%" 의 의미는 시장 상황에 따라 다르다. 전 종목이 빠진 날의 -5%와
혼자 빠진 날의 -5%는 다른 사건이다. 그래서 **매일 199종목을 순위 매겨**
분위수로 나눈다. 이게 일봉 전략의 자연스러운 형태이기도 하다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..datasource.base import Interval
from . import metrics
from .forward import forward_returns


def self_horizon(panel: "FactorPanel") -> int:
    return panel.horizon


@dataclass
class FactorPanel:
    """종목 × 날짜로 쌓은 팩터 값과 전방수익률."""

    frame: pd.DataFrame       # code, day, factor 컬럼들, fwd
    horizon: int = 5          # 전방수익률의 보유기간. 겹침 보정에 필요하다.

    @property
    def days(self) -> int:
        return int(self.frame["day"].nunique())

    @property
    def codes(self) -> int:
        return int(self.frame["code"].nunique())


FACTOR_SPECS: dict[str, str] = {
    "ema60_gap": "종가/60EMA − 1. 음수일수록 추세선 아래로 깊이 이탈",
    "ret_5": "최근 5봉 수익률 — 단기 반전 팩터의 표준 정의",
    "ret_20": "최근 20봉 수익률 — 중기 반전/모멘텀",
    "ret_120": "최근 120봉 수익률 — 중장기 모멘텀",
    "rsi_14": "RSI(14)",
    "atrp_14": "ATR(14)/종가 ×100 — 변동성",
    "rangepos_20": "20봉 레인지 내 위치 (0=하단, 1=상단)",
    # --- 수급 (extras 가 있을 때만) ---
    "foreign_rate": "외국인 지분율 수준 (%)",
    "foreign_flow_1": "외국인 지분율 1일 변화 (%p) — 하루치 순매수 프록시",
    "foreign_flow_5": "외국인 지분율 5일 변화 (%p)",
    "foreign_flow_20": "외국인 지분율 20일 변화 (%p)",
    "foreign_flow_z": "20일 순매수의 z-score (자기 종목 기준 60일)",
}

FLOW_FACTORS = ("foreign_rate", "foreign_flow_1", "foreign_flow_5",
                "foreign_flow_20", "foreign_flow_z")

# 투자자별 순매매량 기반 팩터 (data/flow 가 있을 때)
FACTOR_SPECS.update({
    "inst_ratio": "기관 순매수 / 그날 거래량 — 하루치",
    "foreign_ratio": "외국인 순매수 / 그날 거래량 — 하루치",
    "combined_ratio": "기관+외국인 순매수 / 거래량",
    "inst_ratio_5": "기관 순매수 비중 5일 평균",
    "foreign_ratio_5": "외국인 순매수 비중 5일 평균",
    "combined_ratio_5": "기관+외국인 순매수 비중 5일 평균",
    "inst_ratio_20": "기관 순매수 비중 20일 평균",
    "foreign_ratio_20": "외국인 순매수 비중 20일 평균",
    "combined_ratio_20": "기관+외국인 순매수 비중 20일 평균",
})

TRADE_FLOW_FACTORS = (
    "inst_ratio", "foreign_ratio", "combined_ratio",
    "inst_ratio_5", "foreign_ratio_5", "combined_ratio_5",
    "inst_ratio_20", "foreign_ratio_20", "combined_ratio_20",
)


def build_factor_panel(
    candles_by_code: Mapping[str, pd.DataFrame],
    *,
    interval: Interval = Interval.D1,
    horizon: int = 5,
    entry: str = "next_open",
    extras_by_code: Mapping[str, pd.DataFrame] | None = None,
    flow_by_code: Mapping[str, pd.DataFrame] | None = None,
) -> FactorPanel:
    """종목별로 팩터와 전방수익률을 계산해 하나의 긴 표로 쌓는다.

    extras_by_code 를 주면 수급 팩터가 함께 만들어진다.
    `foreign_rate`(외국인 지분율)의 변화가 외국인 순매수의 프록시다.
    """
    extras_by_code = extras_by_code or {}
    flow_by_code = flow_by_code or {}
    rows = []
    for code, candles in candles_by_code.items():
        features = ind.compute_all(candles, interval=interval)
        close = candles["close"]
        fwd = forward_returns(candles, (horizon,), entry=entry)[f"fwd_{horizon}"]

        frame = pd.DataFrame({
            "code": code,
            "day": candles.index.tz_localize(None).normalize(),
            "ema60_gap": close / features["ema_60"] - 1,
            "ret_5": close / close.shift(5) - 1,
            "ret_20": close / close.shift(20) - 1,
            "ret_120": close / close.shift(120) - 1,
            "rsi_14": features["rsi_14"],
            "atrp_14": features["atrp_14"],
            "rangepos_20": features["rangepos_20"],
            "fwd": fwd,
        })

        extras = extras_by_code.get(code)
        if extras is not None and "foreign_rate" in extras.columns:
            rate = extras["foreign_rate"].reindex(candles.index)
            flow20 = rate - rate.shift(20)
            frame["foreign_rate"] = rate
            frame["foreign_flow_1"] = rate.diff()
            frame["foreign_flow_5"] = rate - rate.shift(5)
            frame["foreign_flow_20"] = flow20
            # 종목마다 지분율 변동 폭이 다르므로 자기 기준으로 표준화한다.
            frame["foreign_flow_z"] = (
                (flow20 - flow20.rolling(60, min_periods=60).mean())
                / flow20.rolling(60, min_periods=60).std(ddof=0).replace(0, np.nan)
            )

        flow = flow_by_code.get(code)
        if flow is not None and not flow.empty:
            from ..datasource.naver_flow import flow_features

            features_flow = flow_features(flow, candles["volume"])
            for column in features_flow.columns:
                if column == "foreign_rate" and "foreign_rate" in frame.columns:
                    continue      # extras 쪽 값을 우선한다 (같은 값이다)
                frame[column] = features_flow[column].reindex(candles.index)

        rows.append(frame.dropna())

    panel = pd.concat(rows, ignore_index=True)
    # 초과수익 = 그날 전 종목 평균을 뺀 값. 시장 전체가 오른 날의 상승은 신호가 아니다.
    panel["excess"] = panel["fwd"] - panel.groupby("day")["fwd"].transform("mean")
    return FactorPanel(panel, horizon=horizon)


def _spearman(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    """순위 상관. scipy 없이 순위에 대한 피어슨 상관으로 계산한다."""
    left = pd.Series(np.asarray(a, dtype=float)).rank()
    right = pd.Series(np.asarray(b, dtype=float)).rank()
    if len(left) < 3 or left.std(ddof=0) == 0 or right.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _cross_sectional_bucket(panel: pd.DataFrame, factor: str, n_buckets: int) -> pd.Series:
    """날짜마다 전 종목을 순위 매겨 분위수로 나눈다.

    날짜별로 나누는 것이 핵심이다. 전체 표본에 대해 한 번에 나누면
    "시장 전체가 빠진 시기"가 통째로 하위 분위에 들어가 시점 효과와 섞인다.
    """
    def rank_within_day(series: pd.Series) -> pd.Series:
        if series.notna().sum() < n_buckets:
            return pd.Series(np.nan, index=series.index)
        return pd.qcut(series.rank(method="first"), n_buckets, labels=False, duplicates="drop")

    return panel.groupby("day")[factor].transform(rank_within_day)


def dose_response(
    panel: FactorPanel,
    factor: str = "ema60_gap",
    *,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """분위수별 초과수익. 단조성이 있는지 보는 표.

    t 는 날짜 군집 보정값이다 — 같은 날 20종목이 같은 분위에 들어가도
    독립 관측 20개가 아니다.
    """
    frame = panel.frame.copy()
    frame["bucket"] = _cross_sectional_bucket(frame, factor, n_buckets)
    frame = frame.dropna(subset=["bucket"])

    rows = []
    for bucket, group in frame.groupby("bucket"):
        daily_mean = group.groupby("day")["excess"].mean().sort_index()
        rows.append({
            "bucket": int(bucket),
            "n": len(group),
            "n_days": int(group["day"].nunique()),
            f"{factor}_median": float(group[factor].median()),
            "excess": float(group["excess"].mean()),
            "win_rate": float((group["excess"] > 0).mean()),
            "t_overlap": metrics.clustered_t_stat(
                group["excess"].to_numpy(), group["day"].to_numpy()
            ),
            "t_edge": metrics.non_overlapping_t_stat(daily_mean.to_numpy(), panel.horizon),
        })
    out = pd.DataFrame(rows).set_index("bucket")

    # 롱숏 스프레드: 최하위 분위 − 최상위 분위. 실제로 거래 가능한 형태다.
    daily = frame.groupby(["day", "bucket"])["excess"].mean().unstack()
    lo, hi = daily.columns.min(), daily.columns.max()
    spread = (daily[lo] - daily[hi]).dropna()
    out.attrs["spread_mean"] = float(spread.mean())
    # 겹침 t 는 참고용이다. 전방수익률이 매일 겹치므로 √h 배 부풀려져 있다.
    out.attrs["spread_t_overlap"] = metrics.clustered_t_stat(
        spread.to_numpy(), spread.index.to_numpy()
    )
    # 판정은 겹치지 않는 표본으로 한다.
    out.attrs["spread_t"] = metrics.non_overlapping_t_stat(spread.to_numpy(), self_horizon(panel))
    out.attrs["spread_days"] = len(spread)
    out.attrs["monotone_rho"] = _spearman(out.index.to_numpy(), out["excess"].to_numpy())
    out.attrs["factor"] = factor
    return out


def double_sort(
    panel: FactorPanel,
    factor: str = "ema60_gap",
    control: str = "ret_5",
    *,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """통제 변수 안에서 팩터가 여전히 작동하는지 본다.

    `ema60_gap` 이 사실은 `ret_5`(단기 반전)의 다른 이름일 수 있다.
    ret_5 분위 **안에서** ema60_gap 분위별 초과수익이 여전히 벌어진다면
    독립적인 정보를 담고 있는 것이고, 평평해지면 같은 것을 재발견한 것이다.
    """
    frame = panel.frame.copy()
    frame["f_bucket"] = _cross_sectional_bucket(frame, factor, n_buckets)
    frame["c_bucket"] = _cross_sectional_bucket(frame, control, n_buckets)
    frame = frame.dropna(subset=["f_bucket", "c_bucket"])

    table = frame.pivot_table(index="c_bucket", columns="f_bucket", values="excess", aggfunc="mean")
    table.index.name = f"{control} 분위"
    table.columns.name = f"{factor} 분위"

    # 통제 분위마다 (최저 팩터 − 최고 팩터) 스프레드
    spreads = []
    for c_bucket, group in frame.groupby("c_bucket"):
        daily = group.groupby(["day", "f_bucket"])["excess"].mean().unstack()
        if daily.shape[1] < 2:
            continue
        lo, hi = daily.columns.min(), daily.columns.max()
        series = (daily[lo] - daily[hi]).dropna()
        spreads.append({
            "c_bucket": int(c_bucket),
            "spread": float(series.mean()),
            "t": metrics.clustered_t_stat(series.to_numpy(), series.index.to_numpy()),
            "n_days": len(series),
        })
    table.attrs["spreads"] = pd.DataFrame(spreads).set_index("c_bucket")
    return table


def market_regression(panel: FactorPanel, factor: str, *, n_buckets: int = 10) -> dict[str, float]:
    """롱숏 스프레드를 시장수익률에 회귀해 알파와 베타를 분리한다.

    횡단면 평균을 빼도 **베타 노출은 남는다.** 고변동성 종목은 시장이 오를 때
    평균보다 더 오른다. 상승장 표본에서는 그것만으로 스프레드가 커진다.
    총 스프레드가 커도 알파가 0 근처면 시장 방향에 올라탄 것이다.
    """
    frame = panel.frame.copy()
    frame["bucket"] = _cross_sectional_bucket(frame, factor, n_buckets)
    frame = frame.dropna(subset=["bucket"])

    daily = frame.groupby(["day", "bucket"])["excess"].mean().unstack()
    lo, hi = daily.columns.min(), daily.columns.max()
    spread = (daily[hi] - daily[lo]).dropna()
    market = panel.frame.groupby("day")["fwd"].mean().reindex(spread.index)

    design = np.column_stack([np.ones(len(market)), market.to_numpy()])
    coef, *_ = np.linalg.lstsq(design, spread.to_numpy(), rcond=None)
    resid = spread.to_numpy() - design @ coef
    return {
        "spread": float(spread.mean()),
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "market_r2": float(1 - np.var(resid) / np.var(spread)) if np.var(spread) > 0 else np.nan,
        "alpha_t_nonoverlap": metrics.non_overlapping_t_stat(
            spread.to_numpy() - coef[1] * market.to_numpy(), panel.horizon
        ),
        "n_days": len(spread),
    }


def factor_correlations(panel: FactorPanel, factors: Sequence[str] | None = None) -> pd.DataFrame:
    """팩터끼리 얼마나 겹치는지. 상관이 높으면 같은 것을 다르게 부르고 있을 뿐이다."""
    cols = list(factors or FACTOR_SPECS)
    frame = panel.frame[cols]
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in cols:
        for j in cols:
            out.loc[i, j] = 1.0 if i == j else _spearman(frame[i], frame[j])
    return out.round(3)
