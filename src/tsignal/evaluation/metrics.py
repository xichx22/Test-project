"""수익률 계열 → 성과 지표.

모든 함수는 '거래 단위 수익률(fraction, 0.01 = +1%)' Series 를 받는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(r: pd.Series) -> pd.Series:
    return pd.Series(r, dtype="float64").dropna()


def win_rate(r: pd.Series) -> float:
    r = _clean(r)
    return float((r > 0).mean()) if len(r) else np.nan


def expectancy(r: pd.Series) -> float:
    """1회 거래당 기대수익률. 단타에서 가장 중요한 단일 숫자."""
    r = _clean(r)
    return float(r.mean()) if len(r) else np.nan


def payoff_ratio(r: pd.Series) -> float:
    """평균이익 / 평균손실. 승률과 짝지어 봐야 의미가 있다."""
    r = _clean(r)
    wins, losses = r[r > 0], r[r < 0]
    if not len(wins) or not len(losses):
        return np.nan
    return float(wins.mean() / abs(losses.mean()))


def profit_factor(r: pd.Series) -> float:
    r = _clean(r)
    gain = r[r > 0].sum()
    loss = abs(r[r < 0].sum())
    return float(gain / loss) if loss > 0 else np.nan


def sharpe(r: pd.Series, periods_per_year: float = 252.0) -> float:
    """거래 단위 샤프. periods_per_year 는 '연간 거래 횟수' 로 넣는다."""
    r = _clean(r)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(r: pd.Series) -> float:
    """거래를 순서대로 복리 누적했을 때의 최대낙폭(음수)."""
    r = _clean(r)
    if not len(r):
        return np.nan
    equity = (1 + r).cumprod()
    return float((equity / equity.cummax() - 1).min())


def t_stat(r: pd.Series) -> float:
    """평균이 0이라는 귀무가설에 대한 t 통계량.

    |t| < 2 면 '표본에서 우연히 나온 양의 기대값'과 구분되지 않는다 —
    이 프로젝트에서 신호를 채택/기각하는 1차 기준.
    """
    r = _clean(r)
    if len(r) < 3 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def bootstrap_ci(r: pd.Series, *, n: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    """기대수익률의 부트스트랩 신뢰구간. 하한이 0 위면 신호가 살아남았다고 본다."""
    r = _clean(r).to_numpy()
    if len(r) < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n, len(r)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize(r: pd.Series, *, periods_per_year: float = 252.0) -> dict[str, float]:
    r = _clean(r)
    lo, hi = bootstrap_ci(r)
    return {
        "n": int(len(r)),
        "expectancy": expectancy(r),
        "win_rate": win_rate(r),
        "payoff": payoff_ratio(r),
        "profit_factor": profit_factor(r),
        "median": float(r.median()) if len(r) else np.nan,
        "std": float(r.std(ddof=1)) if len(r) > 1 else np.nan,
        "sharpe": sharpe(r, periods_per_year),
        "max_dd": max_drawdown(r),
        "t_stat": t_stat(r),
        "ci_low": lo,
        "ci_high": hi,
        "total_return": float((1 + r).prod() - 1) if len(r) else np.nan,
    }
