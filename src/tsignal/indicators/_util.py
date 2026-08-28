"""지표 구현에 공통으로 쓰는 이동평균/변동성 원시 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    # adjust=False → 실시간 갱신식과 동일. 백테스트/실전 값이 어긋나지 않게 한다.
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder 평활. RSI/ATR/ADX 의 표준 평활법."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def dema(s: pd.Series, n: int) -> pd.Series:
    """Double EMA — EMA 의 지연을 한 겹 걷어낸 값."""
    e1 = ema(s, n)
    return 2 * e1 - ema(e1, n)


def tema(s: pd.Series, n: int) -> pd.Series:
    e1 = ema(s, n)
    e2 = ema(e1, n)
    return 3 * e1 - 3 * e2 + ema(e2, n)


def hma(s: pd.Series, n: int) -> pd.Series:
    half, root = max(1, n // 2), max(1, int(np.sqrt(n)))
    return wma(2 * wma(s, half) - wma(s, n), root)


def kama(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman 적응형 이동평균. 추세 효율이 낮으면 스스로 둔해진다."""
    change = (s - s.shift(n)).abs()
    volatility = s.diff().abs().rolling(n, min_periods=n).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2

    values = s.to_numpy(dtype=float)
    alpha = sc.to_numpy(dtype=float)
    out = np.full(len(s), np.nan)
    if len(s) <= n:
        return pd.Series(out, index=s.index)
    out[n] = values[n]
    for i in range(n + 1, len(s)):
        out[i] = out[i - 1] + alpha[i] * (values[i] - out[i - 1])
    return pd.Series(out, index=s.index)


MA_FUNCS = {"sma": sma, "ema": ema, "wma": wma, "dema": dema, "tema": tema, "hma": hma, "rma": rma}


def moving_average(s: pd.Series, n: int, kind: str = "ema") -> pd.Series:
    if kind not in MA_FUNCS:
        raise KeyError(f"지원하지 않는 이동평균 '{kind}'. 사용 가능: {sorted(MA_FUNCS)}")
    return MA_FUNCS[kind](s, n)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)


def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    """a 가 b 를 상향 돌파한 봉에서 True. 두 계열 모두 유효한 구간만."""
    prev = (a.shift(1) <= b.shift(1)) & a.shift(1).notna() & b.shift(1).notna()
    return prev & (a > b)


def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    prev = (a.shift(1) >= b.shift(1)) & a.shift(1).notna() & b.shift(1).notna()
    return prev & (a < b)
