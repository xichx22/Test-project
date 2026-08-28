"""지표 레지스트리.

지표를 하나 추가한다는 것은 = 함수를 쓰고 `register()` 한 줄을 더하는 것.
그러면 `compute_all()` 의 피처 행렬에도, 검증 리포트에도 자동으로 따라 들어간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd

from ..datasource.base import Interval
from . import momentum, trend, volatility, volume
from ._util import cross_down, cross_up, moving_average

IndicatorFunc = Callable[..., pd.Series | pd.DataFrame]


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    func: IndicatorFunc
    category: str
    params: dict[str, Any] = field(default_factory=dict)
    doc: str = ""

    def compute(self, df: pd.DataFrame, **overrides: Any) -> pd.DataFrame:
        out = self.func(df, **{**self.params, **overrides})
        if isinstance(out, pd.Series):
            out = out.to_frame(out.name or self.name)
        return out


REGISTRY: dict[str, IndicatorSpec] = {}


def register(name: str, func: IndicatorFunc, category: str, **params: Any) -> IndicatorSpec:
    spec = IndicatorSpec(name, func, category, params, (func.__doc__ or "").strip().split("\n")[0])
    REGISTRY[name] = spec
    return spec


# --- 추세 ---------------------------------------------------------------
for _span in (5, 10, 20, 60, 120):
    register(f"ema{_span}", trend.ma, "trend", n=_span, kind="ema")
for _span in (5, 20, 60):
    register(f"sma{_span}", trend.ma, "trend", n=_span, kind="sma")
    register(f"dema{_span}", trend.ma, "trend", n=_span, kind="dema")
    register(f"tema{_span}", trend.ma, "trend", n=_span, kind="tema")
    register(f"hma{_span}", trend.ma, "trend", n=_span, kind="hma")
register("macd", trend.macd, "trend")
register("ppo", trend.ppo, "trend")
register("adx", trend.adx, "trend")
register("aroon", trend.aroon, "trend")
register("supertrend", trend.supertrend, "trend")
register("ichimoku", trend.ichimoku, "trend")
register("psar", trend.psar, "trend")
register("ma_slope", trend.ma_slope, "trend")
register("ribbon", trend.ma_ribbon_align, "trend")

# --- 모멘텀 -------------------------------------------------------------
register("rsi", momentum.rsi, "momentum", n=14)
register("rsi7", momentum.rsi, "momentum", n=7)
register("stoch", momentum.stoch, "momentum")
register("stoch_rsi", momentum.stoch_rsi, "momentum")
register("williams_r", momentum.williams_r, "momentum", n=14)
register("cci", momentum.cci, "momentum")
register("roc", momentum.roc, "momentum")
register("tsi", momentum.tsi, "momentum")
register("cmo", momentum.cmo, "momentum")
register("ultimate", momentum.ultimate, "momentum")
register("awesome", momentum.awesome, "momentum")
register("rsi_divergence", momentum.rsi_divergence, "momentum")
register("atr_percent", momentum.atr_percent, "momentum")

# --- 변동성 -------------------------------------------------------------
register("envelope", volatility.envelope, "volatility", n=20, pct=2.0)
register("envelope_atr", volatility.envelope_atr, "volatility")
register("bollinger", volatility.bollinger, "volatility")
register("keltner", volatility.keltner, "volatility")
register("donchian", volatility.donchian, "volatility")
register("squeeze", volatility.squeeze, "volatility")
register("atr", volatility.atr_indicator, "volatility")
register("realized_vol", volatility.realized_vol, "volatility")
register("chaikin_vol", volatility.chaikin_volatility, "volatility")
register("range_position", volatility.range_position, "volatility")

# --- 거래량 -------------------------------------------------------------
register("obv", volume.obv, "volume")
register("ad", volume.ad_line, "volume")
register("cmf", volume.cmf, "volume")
register("mfi", volume.mfi, "volume")
register("vwma", volume.vwma, "volume")
register("vwap", volume.vwap, "volume")
register("force_index", volume.force_index, "volume")
register("eom", volume.ease_of_movement, "volume")
register("pvt", volume.pvt, "volume")
register("volume_z", volume.volume_zscore, "volume")
register("rvol", volume.relative_volume, "volume")
register("vr", volume.volume_ratio, "volume")


BARS_PER_YEAR = {
    Interval.M1: 252 * 390, Interval.M3: 252 * 130, Interval.M5: 252 * 78,
    Interval.M15: 252 * 26, Interval.M30: 252 * 13, Interval.H1: 252 * 7, Interval.D1: 252,
}


def compute_all(
    df: pd.DataFrame,
    *,
    names: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    interval: Interval | None = None,
) -> pd.DataFrame:
    """등록된 지표를 모두 계산해 하나의 피처 행렬로 합친다.

    타임프레임에 따라 달라져야 하는 파라미터(연율화 계수)는 interval 로 주입한다.
    """
    specs = [REGISTRY[n] for n in names] if names else list(REGISTRY.values())
    if categories:
        wanted = set(categories)
        specs = [s for s in specs if s.category in wanted]

    frames: list[pd.DataFrame] = []
    for spec in specs:
        overrides: dict[str, Any] = {}
        if spec.name == "realized_vol" and interval is not None:
            overrides["bars_per_year"] = BARS_PER_YEAR[interval]
        out = spec.compute(df, **overrides)
        # 같은 지표를 파라미터만 바꿔 여러 번 등록해도 컬럼이 충돌하지 않게 한다.
        out = out.rename(columns={c: c if c not in _seen(frames) else f"{spec.name}_{c}" for c in out.columns})
        frames.append(out)

    features = pd.concat(frames, axis=1)
    return features.loc[:, ~features.columns.duplicated()]


def _seen(frames: list[pd.DataFrame]) -> set[str]:
    return {c for f in frames for c in f.columns}


def catalog() -> pd.DataFrame:
    """등록된 지표 목록 — 문서/리포트에 그대로 싣는다."""
    rows = [
        {"name": s.name, "category": s.category, "params": s.params or "", "doc": s.doc}
        for s in REGISTRY.values()
    ]
    return pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)


__all__ = ["IndicatorSpec", "REGISTRY", "register", "compute_all", "catalog",
           "cross_up", "cross_down", "moving_average",
           "trend", "momentum", "volatility", "volume"]
