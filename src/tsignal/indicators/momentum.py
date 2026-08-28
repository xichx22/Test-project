"""모멘텀/오실레이터 지표."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import atr, ema, rma, sma, typical_price


def rsi(df: pd.DataFrame, n: int = 14, column: str = "close") -> pd.Series:
    delta = df[column].diff()
    gain = rma(delta.clip(lower=0), n)
    loss = rma((-delta).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # 손실이 0인 구간은 RS 가 무한대 → RSI 100.
    return out.mask(loss.eq(0) & gain.gt(0), 100.0).rename(f"rsi_{n}")


def stoch(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    """느린 스토캐스틱. %K 는 smooth 로 한 번 평활한 값."""
    low = df["low"].rolling(k, min_periods=k).min()
    high = df["high"].rolling(k, min_periods=k).max()
    raw = 100 * (df["close"] - low) / (high - low).replace(0, np.nan)
    k_line = sma(raw, smooth)
    return pd.DataFrame({"stoch_k": k_line, "stoch_d": sma(k_line, d), "stoch_raw": raw})


def stoch_rsi(df: pd.DataFrame, n: int = 14, k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    """RSI 에 스토캐스틱을 씌운 것. RSI 보다 훨씬 예민해 단타 트리거로 많이 쓴다."""
    r = rsi(df, n)
    low = r.rolling(k, min_periods=k).min()
    high = r.rolling(k, min_periods=k).max()
    raw = 100 * (r - low) / (high - low).replace(0, np.nan)
    k_line = sma(raw, smooth)
    return pd.DataFrame({"stochrsi_k": k_line, "stochrsi_d": sma(k_line, d)})


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Williams %R. 범위 -100(최저) ~ 0(최고).

    통상 -80 이하 과매도 / -20 이상 과매수로 읽지만, 강한 추세에서는
    %R 이 -20 위에 눌러앉는다 → 단독 역추세 진입 근거로 쓰면 안 된다.
    """
    high = df["high"].rolling(n, min_periods=n).max()
    low = df["low"].rolling(n, min_periods=n).min()
    return (-100 * (high - df["close"]) / (high - low).replace(0, np.nan)).rename(f"williams_r_{n}")


def cci(df: pd.DataFrame, n: int = 20, constant: float = 0.015) -> pd.Series:
    tp = typical_price(df)
    mean = sma(tp, n)
    mad = tp.rolling(n, min_periods=n).apply(lambda x: float(np.abs(x - x.mean()).mean()), raw=True)
    return ((tp - mean) / (constant * mad.replace(0, np.nan))).rename(f"cci_{n}")


def roc(df: pd.DataFrame, n: int = 10, column: str = "close") -> pd.Series:
    return ((df[column] / df[column].shift(n) - 1) * 100).rename(f"roc_{n}")


def momentum(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return (df["close"] - df["close"].shift(n)).rename(f"mom_{n}")


def tsi(df: pd.DataFrame, long: int = 25, short: int = 13, signal: int = 7) -> pd.DataFrame:
    """True Strength Index. 이중 평활이라 노이즈에 강하다."""
    diff = df["close"].diff()
    num = ema(ema(diff, long), short)
    den = ema(ema(diff.abs(), long), short)
    line = 100 * num / den.replace(0, np.nan)
    return pd.DataFrame({"tsi": line, "tsi_signal": ema(line, signal)})


def cmo(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Chande Momentum Oscillator. -100~+100."""
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(n, min_periods=n).sum()
    down = (-delta).clip(lower=0).rolling(n, min_periods=n).sum()
    return (100 * (up - down) / (up + down).replace(0, np.nan)).rename(f"cmo_{n}")


def ultimate(df: pd.DataFrame, s: int = 7, m: int = 14, l: int = 28) -> pd.Series:
    """Ultimate Oscillator. 세 기간을 가중 합성해 단일 기간 오실레이터의 다이버전스 오탐을 줄인다."""
    prev_close = df["close"].shift(1)
    buying = df["close"] - pd.concat([df["low"], prev_close], axis=1).min(axis=1)
    tr = pd.concat([df["high"], prev_close], axis=1).max(axis=1) - pd.concat(
        [df["low"], prev_close], axis=1
    ).min(axis=1)

    def avg(n: int) -> pd.Series:
        return buying.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum().replace(0, np.nan)

    return (100 * (4 * avg(s) + 2 * avg(m) + avg(l)) / 7).rename("ultimate")


def awesome(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2
    return (sma(hl2, fast) - sma(hl2, slow)).rename("awesome")


def rsi_divergence(df: pd.DataFrame, n: int = 14, lookback: int = 20) -> pd.Series:
    """정규 다이버전스 탐지. +1 강세(가격 저점 하락 / RSI 저점 상승), -1 약세.

    lookback 창 안에서 확정된 값만 쓰므로 미래참조가 없다.
    """
    r = rsi(df, n)
    price_low = df["low"].rolling(lookback, min_periods=lookback).min()
    price_high = df["high"].rolling(lookback, min_periods=lookback).max()
    rsi_at_low = r.rolling(lookback, min_periods=lookback).min()
    rsi_at_high = r.rolling(lookback, min_periods=lookback).max()

    bull = (df["low"] <= price_low) & (r > rsi_at_low.shift(lookback))
    bear = (df["high"] >= price_high) & (r < rsi_at_high.shift(lookback))
    return (bull.astype(int) - bear.astype(int)).rename("rsi_divergence")


def atr_percent(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """가격 대비 ATR(%). 종목 간 변동성 비교와 손절폭 산정의 기준."""
    return (atr(df, n) / df["close"] * 100).rename(f"atrp_{n}")


__all__ = ["rsi", "stoch", "stoch_rsi", "williams_r", "cci", "roc", "momentum", "tsi",
           "cmo", "ultimate", "awesome", "rsi_divergence", "atr_percent"]
