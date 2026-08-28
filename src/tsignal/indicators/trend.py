"""추세계 지표."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import atr, cross_down, cross_up, ema, moving_average, rma, true_range


def ma(df: pd.DataFrame, n: int = 20, kind: str = "ema", column: str = "close") -> pd.Series:
    return moving_average(df[column], n, kind).rename(f"{kind}_{n}")


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(df["close"], fast) - ema(df["close"], slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def ppo(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 의 백분율판. 가격 수준이 다른 종목 간 비교가 가능해진다."""
    slow_ema = ema(df["close"], slow)
    line = (ema(df["close"], fast) - slow_ema) / slow_ema * 100
    sig = ema(line, signal)
    return pd.DataFrame({"ppo": line, "ppo_signal": sig, "ppo_hist": line - sig})


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """방향성 지수. ADX 는 추세의 '세기'만 말하고 방향은 DI 가 말한다."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr_n = rma(true_range(df), n).replace(0, np.nan)
    plus_di = 100 * rma(plus_dm, n) / atr_n
    minus_di = 100 * rma(minus_dm, n) / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": rma(dx, n)})


def aroon(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    hi = df["high"].rolling(n + 1, min_periods=n + 1).apply(lambda x: float(np.argmax(x)), raw=True)
    lo = df["low"].rolling(n + 1, min_periods=n + 1).apply(lambda x: float(np.argmin(x)), raw=True)
    up = 100 * hi / n
    down = 100 * lo / n
    return pd.DataFrame({"aroon_up": up, "aroon_down": down, "aroon_osc": up - down})


def supertrend(df: pd.DataFrame, n: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """ATR 밴드를 추세 방향으로만 조이는 추세추종 스톱.

    dir = +1 상승추세(밴드가 지지), -1 하락추세(밴드가 저항).
    """
    hl2 = (df["high"] + df["low"]) / 2
    band = multiplier * atr(df, n)
    upper_raw, lower_raw = hl2 + band, hl2 - band

    close = df["close"].to_numpy(dtype=float)
    ub, lb = upper_raw.to_numpy(dtype=float), lower_raw.to_numpy(dtype=float)
    n_rows = len(df)
    upper = np.full(n_rows, np.nan)
    lower = np.full(n_rows, np.nan)
    direction = np.full(n_rows, np.nan)

    start = int(np.argmax(~np.isnan(ub))) if (~np.isnan(ub)).any() else n_rows
    if start < n_rows:
        upper[start], lower[start], direction[start] = ub[start], lb[start], 1.0
    for i in range(start + 1, n_rows):
        upper[i] = ub[i] if (ub[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = lb[i] if (lb[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
        if close[i] > upper[i - 1]:
            direction[i] = 1.0
        elif close[i] < lower[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]

    line = np.where(direction > 0, lower, upper)
    return pd.DataFrame(
        {"supertrend": line, "supertrend_dir": direction,
         "supertrend_upper": upper, "supertrend_lower": lower},
        index=df.index,
    )


def ichimoku(df: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52) -> pd.DataFrame:
    """일목균형표.

    선행스팬은 base 만큼 앞으로 민 값이므로, 현재 봉에서 '지금 보이는 구름'은
    과거에 확정된 값이다 → 미래참조가 없다.
    """
    def mid(n: int) -> pd.Series:
        return (df["high"].rolling(n, min_periods=n).max()
                + df["low"].rolling(n, min_periods=n).min()) / 2

    tenkan, kijun = mid(conv), mid(base)
    return pd.DataFrame({
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": ((tenkan + kijun) / 2).shift(base),
        "senkou_b": mid(span_b).shift(base),
        "chikou": df["close"].shift(base),   # 표시용. 신호에 쓰면 미래참조가 된다.
    })


def psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    """Parabolic SAR. dir=+1 롱 스톱, -1 숏 스톱."""
    high, low = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    sar = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n < 2:
        return pd.DataFrame({"psar": sar, "psar_dir": direction}, index=df.index)

    bull = True
    af = step
    ep = high[0]
    sar[0], direction[0] = low[0], 1.0
    for i in range(1, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)
        if bull:
            cur = min(cur, low[i - 1], low[max(0, i - 2)])
            if low[i] < cur:
                bull, cur, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            cur = max(cur, high[i - 1], high[max(0, i - 2)])
            if high[i] > cur:
                bull, cur, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        sar[i], direction[i] = cur, 1.0 if bull else -1.0
    return pd.DataFrame({"psar": sar, "psar_dir": direction}, index=df.index)


def ma_slope(df: pd.DataFrame, n: int = 20, kind: str = "ema", lookback: int = 5) -> pd.Series:
    """이동평균의 기울기(%). 추세 필터로 쓰기 좋다."""
    line = moving_average(df["close"], n, kind)
    return ((line / line.shift(lookback) - 1) * 100).rename(f"{kind}{n}_slope{lookback}")


def ma_ribbon_align(df: pd.DataFrame, spans: tuple[int, ...] = (5, 10, 20, 60), kind: str = "ema") -> pd.Series:
    """정배열(+1)/역배열(-1)/혼조(0)."""
    mas = [moving_average(df["close"], s, kind) for s in spans]
    up = pd.concat([mas[i] > mas[i + 1] for i in range(len(mas) - 1)], axis=1).all(axis=1)
    down = pd.concat([mas[i] < mas[i + 1] for i in range(len(mas) - 1)], axis=1).all(axis=1)
    return (up.astype(int) - down.astype(int)).where(mas[-1].notna()).rename("ribbon_align")


__all__ = ["ma", "macd", "ppo", "adx", "aroon", "supertrend", "ichimoku", "psar",
           "ma_slope", "ma_ribbon_align", "cross_up", "cross_down"]
