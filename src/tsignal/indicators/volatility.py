"""변동성/채널 지표."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import atr, ema, moving_average, sma, true_range


def envelope(df: pd.DataFrame, n: int = 20, pct: float = 2.0, kind: str = "sma") -> pd.DataFrame:
    """이동평균 엔벌로프 — 중심선의 고정 비율 밴드.

    볼린저와 달리 밴드폭이 변동성에 반응하지 않는다. 그래서
      - 변동성이 낮은 구간: 밴드가 상대적으로 넓어 신호가 안 뜬다
      - 변동성이 급등한 구간: 밴드를 계속 뚫어 과매수/과매도 신호가 남발된다
    단타에서 엔벌로프를 쓸 거면 pct 를 ATR% 로 적응시키거나(`envelope_atr`)
    변동성 레짐 필터와 반드시 같이 써야 한다.
    """
    mid = moving_average(df["close"], n, kind)
    band = mid * (pct / 100.0)
    upper, lower = mid + band, mid - band
    return pd.DataFrame({
        "env_mid": mid,
        "env_upper": upper,
        "env_lower": lower,
        # 밴드 내 상대 위치. 0=하단, 1=상단, 밖으로 나가면 0 미만/1 초과.
        "env_pctb": (df["close"] - lower) / (upper - lower).replace(0, np.nan),
    })


def envelope_atr(df: pd.DataFrame, n: int = 20, mult: float = 2.0, kind: str = "ema") -> pd.DataFrame:
    """엔벌로프의 적응형 변형 — 폭을 ATR 로 잡는다. 고정 % 판의 약점을 덮는다."""
    mid = moving_average(df["close"], n, kind)
    band = mult * atr(df, n)
    upper, lower = mid + band, mid - band
    return pd.DataFrame({
        "enva_mid": mid, "enva_upper": upper, "enva_lower": lower,
        "enva_pctb": (df["close"] - lower) / (upper - lower).replace(0, np.nan),
    })


def bollinger(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(df["close"], n)
    std = df["close"].rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * std, mid - k * std
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "bb_width": (upper - lower) / mid.replace(0, np.nan) * 100,
        "bb_pctb": (df["close"] - lower) / (upper - lower).replace(0, np.nan),
    })


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0, atr_n: int = 10) -> pd.DataFrame:
    mid = ema(df["close"], n)
    band = mult * atr(df, atr_n)
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + band, "kc_lower": mid - band})


def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(n, min_periods=n).max()
    lower = df["low"].rolling(n, min_periods=n).min()
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_mid": (upper + lower) / 2})


def squeeze(df: pd.DataFrame, n: int = 20, bb_k: float = 2.0, kc_mult: float = 1.5) -> pd.DataFrame:
    """TTM Squeeze. 볼린저가 켈트너 안으로 들어가면(=1) 변동성 수축 → 돌파 대기 구간."""
    bb = bollinger(df, n, bb_k)
    kc = keltner(df, n, kc_mult, n)
    on = ((bb["bb_upper"] < kc["kc_upper"]) & (bb["bb_lower"] > kc["kc_lower"])).astype(float)
    on = on.where(bb["bb_upper"].notna() & kc["kc_upper"].notna())
    return pd.DataFrame({"squeeze_on": on, "squeeze_release": ((on.shift(1) == 1) & (on == 0)).astype(float)})


def atr_indicator(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    a = atr(df, n)
    return pd.DataFrame({"atr": a, "natr": a / df["close"].replace(0, np.nan) * 100, "tr": true_range(df)})


def realized_vol(df: pd.DataFrame, n: int = 20, bars_per_year: int = 252 * 78) -> pd.Series:
    """로그수익률 기반 실현변동성(연율 %). bars_per_year 는 타임프레임에 맞춰 넘길 것."""
    ret = np.log(df["close"]).diff()
    return (ret.rolling(n, min_periods=n).std(ddof=0) * np.sqrt(bars_per_year) * 100).rename(f"rv_{n}")


def chaikin_volatility(df: pd.DataFrame, n: int = 10, roc_n: int = 10) -> pd.Series:
    spread = ema(df["high"] - df["low"], n)
    return ((spread / spread.shift(roc_n) - 1) * 100).rename("chaikin_vol")


def range_position(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """n봉 레인지 안에서 종가의 위치(0~1). 돌파/눌림 판정의 공통 축."""
    high = df["high"].rolling(n, min_periods=n).max()
    low = df["low"].rolling(n, min_periods=n).min()
    return ((df["close"] - low) / (high - low).replace(0, np.nan)).rename(f"rangepos_{n}")


__all__ = ["envelope", "envelope_atr", "bollinger", "keltner", "donchian", "squeeze",
           "atr_indicator", "realized_vol", "chaikin_volatility", "range_position"]
