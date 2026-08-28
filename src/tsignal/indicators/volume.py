"""거래량/자금흐름 지표.

단타에서 가격 신호 단독은 신뢰도가 낮다. 거래량 확인이 붙었을 때 기대수익이
어떻게 바뀌는지가 이 프로젝트 검증의 핵심 축 중 하나다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ohlcv import sessions
from ._util import ema, rma, sma, typical_price, zscore


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum().rename("obv")


def ad_line(df: pd.DataFrame) -> pd.Series:
    """누적 매집/분산선."""
    span = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span
    return (mfm.fillna(0.0) * df["volume"]).cumsum().rename("ad")


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    span = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span
    mfv = (mfm.fillna(0.0) * df["volume"]).rolling(n, min_periods=n).sum()
    return (mfv / df["volume"].rolling(n, min_periods=n).sum().replace(0, np.nan)).rename(f"cmf_{n}")


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """자금흐름지수 — 거래량 가중 RSI."""
    tp = typical_price(df)
    raw = tp * df["volume"]
    up = raw.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    down = raw.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    ratio = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).rename(f"mfi_{n}")


def vwma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    num = (df["close"] * df["volume"]).rolling(n, min_periods=n).sum()
    den = df["volume"].rolling(n, min_periods=n).sum().replace(0, np.nan)
    return (num / den).rename(f"vwma_{n}")


def vwap(df: pd.DataFrame, *, session_reset: bool = True) -> pd.DataFrame:
    """VWAP. 분봉에서는 반드시 세션(거래일) 단위로 리셋해야 의미가 있다.

    session_reset=False 로 두면 누적 VWAP 이 되어, 장 시작 시점의 기준선
    역할을 하지 못한다. 단타 기준선으로 쓸 거면 True 를 유지할 것.
    """
    tp = typical_price(df)
    pv = tp * df["volume"]
    if session_reset:
        key = sessions(df)
        cum_pv = pv.groupby(key).cumsum()
        cum_v = df["volume"].groupby(key).cumsum()
        counts = df["volume"].groupby(key).cumcount()
        var = (tp.pow(2) * df["volume"]).groupby(key).cumsum() / cum_v.replace(0, np.nan)
    else:
        cum_pv, cum_v = pv.cumsum(), df["volume"].cumsum()
        counts = pd.Series(np.arange(len(df)), index=df.index)
        var = (tp.pow(2) * df["volume"]).cumsum() / cum_v.replace(0, np.nan)

    line = cum_pv / cum_v.replace(0, np.nan)
    band = np.sqrt((var - line.pow(2)).clip(lower=0))
    return pd.DataFrame({
        "vwap": line,
        "vwap_upper": line + band,
        "vwap_lower": line - band,
        "vwap_dev_pct": (df["close"] / line - 1) * 100,
        "session_bar": counts.astype(float),   # 세션 내 몇 번째 봉인지 (개장 직후 필터용)
    })


def force_index(df: pd.DataFrame, n: int = 13) -> pd.Series:
    return ema(df["close"].diff() * df["volume"], n).rename(f"force_{n}")


def ease_of_movement(df: pd.DataFrame, n: int = 14, scale: float = 1e8) -> pd.Series:
    distance = ((df["high"] + df["low"]) / 2).diff()
    box = (df["volume"] / scale) / (df["high"] - df["low"]).replace(0, np.nan)
    return sma(distance / box.replace(0, np.nan), n).rename(f"eom_{n}")


def pvt(df: pd.DataFrame) -> pd.Series:
    return (df["close"].pct_change().fillna(0.0) * df["volume"]).cumsum().rename("pvt")


def volume_zscore(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """거래량 급증 정도. 돌파 신호의 확인(confirmation) 축으로 쓴다."""
    return zscore(df["volume"], n).rename(f"volz_{n}")


def relative_volume(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """평균 대비 거래량 배수(RVOL)."""
    return (df["volume"] / sma(df["volume"], n).replace(0, np.nan)).rename(f"rvol_{n}")


def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """국내 HTS 의 거래량비율(VR). 상승일 거래량 / 하락일 거래량 * 100."""
    change = df["close"].diff()
    up = df["volume"].where(change > 0, 0.0).rolling(n, min_periods=n).sum()
    down = df["volume"].where(change < 0, 0.0).rolling(n, min_periods=n).sum()
    flat = df["volume"].where(change == 0, 0.0).rolling(n, min_periods=n).sum()
    return (100 * (up + flat / 2) / (down + flat / 2).replace(0, np.nan)).rename(f"vr_{n}")


__all__ = ["obv", "ad_line", "cmf", "mfi", "vwma", "vwap", "force_index",
           "ease_of_movement", "pvt", "volume_zscore", "relative_volume", "volume_ratio"]
