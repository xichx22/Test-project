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


def clustered_t_stat(values: np.ndarray, cluster: np.ndarray) -> float:
    """군집(날짜) 보정 t 통계량.

    왜 필요한가
    -----------
    한국 주식 199개는 같은 날 같이 움직인다. 시장이 빠진 날에는 수십~백 종목에서
    `bollinger_lower_reclaim` 이 동시에 뜬다. 이걸 독립 관측 100개로 세면
    표준오차가 √100 만큼 작아지고 t 가 그만큼 부풀려진다.

    실측(199종목·5년 일봉)에서 이 인플레이션은 최대 **10배**였다.
      bollinger_lower_reclaim : 순진한 t 5.93 → 군집 보정 t 0.58
      donchian_breakout       : 순진한 t 3.56 → 군집 보정 t -0.79 (부호까지 뒤집힘)

    방법
    ----
    같은 날짜의 관측을 먼저 평균 내어 날짜당 하나의 값으로 만들고, 날짜 사이에서
    t 를 낸다 (Fama-MacBeth 방식). 유효 표본 수는 신호 발생 횟수가 아니라
    **신호가 발생한 날짜 수**다.
    """
    vals = np.asarray(values, dtype=float)
    keys = np.asarray(cluster)
    if len(vals) < 2:
        return np.nan

    uniq, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(uniq))
    sums = np.bincount(inverse, weights=vals, minlength=len(uniq))
    means = sums / counts

    if len(means) < 3:
        return np.nan
    sd = means.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(means.mean() / (sd / np.sqrt(len(means))))


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
