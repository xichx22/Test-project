"""상태 필터 — 트리거와 결합해 "언제 그 신호를 믿을 것인가"를 좁힌다.

신호(signal)와 필터(filter)는 성격이 다르다.

  트리거(event) : 특정 봉에서만 True. "지금 사라" — 예: MACD 골든크로스
  필터(state)   : 구간 내내 True/False. "지금은 살 만한 상황인가" — 예: 종가 > 60EMA

단독 트리거가 실데이터에서 전부 기각된 이유 중 하나는, 트리거가 시장 상태를
구분하지 않기 때문이라는 가설이 있다. 상승추세에서의 RSI 반등과 하락추세에서의
RSI 반등은 다른 사건인데 같은 신호로 세고 있었다.

필터는 축(axis)별로 묶여 있다. 같은 축의 필터끼리는 결합하지 않는다 —
`추세상승 AND 추세하락` 같은 공허한 조합과, 사실상 같은 조건을 두 번 거는
중복 조합을 탐색 공간에서 미리 빼기 위해서다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

FilterFunc = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class FilterSpec:
    name: str
    func: FilterFunc
    axis: str              # trend / regime / volatility / volume / position
    rationale: str = ""

    def evaluate(self, candles: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        out = self.func(candles, features)
        return out.fillna(False).astype(bool).rename(self.name)


REGISTRY: dict[str, FilterSpec] = {}


def filter_(name: str, axis: str, *, rationale: str = "") -> Callable[[FilterFunc], FilterFunc]:
    def wrap(func: FilterFunc) -> FilterFunc:
        REGISTRY[name] = FilterSpec(name, func, axis, rationale)
        return func

    return wrap


def _f(features: pd.DataFrame, col: str) -> pd.Series:
    if col not in features.columns:
        raise KeyError(f"피처 '{col}' 가 없습니다.")
    return features[col]


# --- 추세 축 -----------------------------------------------------------
@filter_("trend_up", "trend", rationale="상위 추세가 살아 있을 때만 롱을 잡는다.")
def trend_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return c["close"] > _f(f, "ema_60")


@filter_("trend_down", "trend", rationale="하락추세에서의 반등 신호는 다른 사건이라는 가설의 대조군.")
def trend_down(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return c["close"] < _f(f, "ema_60")


@filter_("ribbon_up", "trend", rationale="단기·중기·장기 이평 정배열 — 추세의 가장 엄격한 정의.")
def ribbon_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "ribbon_align") == 1


@filter_("supertrend_up", "trend", rationale="ATR 스톱 기준 상승 국면.")
def supertrend_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "supertrend_dir") == 1


# --- 레짐 축 (추세성 vs 횡보) -------------------------------------------
@filter_("adx_strong", "regime", rationale="ADX>25 = 추세장. 돌파 신호가 먹힐 조건이라는 가설.")
def adx_strong(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "adx") > 25


@filter_("adx_weak", "regime", rationale="ADX<20 = 횡보장. 평균회귀 신호가 먹힐 조건이라는 가설.")
def adx_weak(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "adx") < 20


# --- 변동성 축 ---------------------------------------------------------
@filter_("vol_high", "volatility", rationale="자기 종목 기준 변동성 상위 국면. 기대수익이 비용을 넘길 여지가 크다.")
def vol_high(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    atrp = _f(f, "atrp_14")
    return atrp > atrp.rolling(100, min_periods=100).median()


@filter_("vol_low", "volatility", rationale="변동성 하위 국면. 수축 후 확장을 노리는 조건.")
def vol_low(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    atrp = _f(f, "atrp_14")
    return atrp < atrp.rolling(100, min_periods=100).median()


@filter_("squeeze_on", "volatility", rationale="볼린저가 켈트너 안에 든 변동성 수축 구간.")
def squeeze_on(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "squeeze_on") == 1


# --- 거래량 축 ---------------------------------------------------------
@filter_("volume_above_avg", "volume", rationale="평균 이상 거래량 — 신호에 참여자가 붙었는가.")
def volume_above_avg(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "rvol_20") > 1.2


@filter_("volume_surge", "volume", rationale="평균 2배 이상 거래량 — 더 엄격한 확인.")
def volume_surge(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "rvol_20") > 2.0


@filter_("money_flow_positive", "volume", rationale="자금이 순유입 중인 구간.")
def money_flow_positive(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "cmf_20") > 0


# --- 위치 축 -----------------------------------------------------------
@filter_("near_range_low", "position", rationale="20봉 레인지 하단부 — 되돌림 진입의 위치 조건.")
def near_range_low(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "rangepos_20") < 0.3


@filter_("near_range_high", "position", rationale="20봉 레인지 상단부 — 돌파 진입의 위치 조건.")
def near_range_high(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "rangepos_20") > 0.7


@filter_("not_overbought", "position", rationale="RSI 70 미만 — 과열 구간 추격매수를 뺀다.")
def not_overbought(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "rsi_14") < 70


def evaluate_all(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    *,
    names: Iterable[str] | None = None,
) -> pd.DataFrame:
    specs = [REGISTRY[n] for n in names] if names else list(REGISTRY.values())
    return pd.DataFrame({s.name: s.evaluate(candles, features) for s in specs}, index=candles.index)


def axes() -> dict[str, list[str]]:
    """축 → 필터명 목록."""
    out: dict[str, list[str]] = {}
    for spec in REGISTRY.values():
        out.setdefault(spec.axis, []).append(spec.name)
    return out


def catalog() -> pd.DataFrame:
    rows = [{"name": s.name, "axis": s.axis, "rationale": s.rationale} for s in REGISTRY.values()]
    return pd.DataFrame(rows).sort_values(["axis", "name"]).reset_index(drop=True)
