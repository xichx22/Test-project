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


def non_overlapping_t_stat(values: np.ndarray, horizon: int) -> float:
    """겹치지 않는 표본으로 계산한 t.

    왜 필요한가
    -----------
    보유 20일 전방수익률을 매일 계산하면, 인접한 날의 관측은 20일 중 19일을
    공유한다. 독립 관측이 아닌데 독립으로 세면 t 가 대략 √h 배 부풀려진다.
    날짜 군집 보정은 이걸 잡지 못한다 — 같은 날 안의 상관은 잡지만 날짜 사이의
    겹침은 그대로 두기 때문이다.

    실측(199종목·5년 일봉, 보유 20일 롱숏 스프레드):
      ema60_gap : 겹침 t 6.12 → 비겹침 t 1.36
      ret_120   : 겹침 t 8.62 → 비겹침 t 1.95
      atrp_14   : 겹침 t 13.50 → 비겹침 t 3.03

    방법
    ----
    h일 간격으로 뽑아 겹치지 않는 부분표본을 만든다. 시작점을 어디로 잡느냐에
    따라 h개의 부분표본이 나오므로, 전부 계산해 평균을 낸다 (한 시작점만 쓰면
    그 선택 자체가 또 하나의 자유도가 된다).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if horizon < 1:
        raise ValueError("horizon 은 1 이상이어야 합니다.")
    if len(arr) < horizon * 3:
        return np.nan

    stats = []
    for offset in range(horizon):
        sub = arr[offset::horizon]
        if len(sub) < 3:
            continue
        sd = sub.std(ddof=1)
        if sd > 0:
            stats.append(sub.mean() / (sd / np.sqrt(len(sub))))
    return float(np.mean(stats)) if stats else np.nan


def spaced_t_stat(days: np.ndarray, values: np.ndarray, horizon: int) -> float:
    """드문 이벤트용 겹침 보정 t.

    `non_overlapping_t_stat` 은 매 봉 값이 있는 연속 계열을 가정한다.
    차트 패턴처럼 드문드문 발생하는 이벤트는 날짜가 띄엄띄엄하므로,
    **서로 horizon 일 이상 떨어진 이벤트만** 남겨 겹침을 없앤다.

    시작점에 따라 남는 집합이 달라지므로 여러 시작점으로 반복해 평균한다.
    """
    day = np.asarray(days, dtype="datetime64[D]").astype(np.int64)
    val = np.asarray(values, dtype=float)
    order = np.argsort(day)
    day, val = day[order], val[order]

    # 같은 날짜의 이벤트는 먼저 평균 낸다 (횡단면 상관 제거).
    uniq, inverse = np.unique(day, return_inverse=True)
    means = np.bincount(inverse, weights=val) / np.bincount(inverse)
    if len(uniq) < 3:
        return np.nan

    stats = []
    for start in range(min(horizon, len(uniq))):
        picked, last = [], -(10**9)
        for i in range(start, len(uniq)):
            if uniq[i] - last >= horizon:
                picked.append(means[i])
                last = uniq[i]
        if len(picked) < 3:
            continue
        arr = np.asarray(picked)
        sd = arr.std(ddof=1)
        if sd > 0:
            stats.append(arr.mean() / (sd / np.sqrt(len(arr))))
    return float(np.mean(stats)) if stats else np.nan


def newey_west_t_stat(values: np.ndarray, lags: int | None = None) -> float:
    """자기상관을 감안한 t (Newey-West, Bartlett 커널).

    캘린더-타임 포트폴리오는 매일 거의 같은 종목을 들고 있으므로 일별 수익률이
    서로 독립이 아니다. 평범한 t 는 이 지속성을 무시해 부풀려질 수 있다.
    lags 를 주지 않으면 Newey-West 의 표준 규칙 4*(n/100)^(2/9) 을 쓴다.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 10:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(0, min(lags, n - 2))

    resid = arr - arr.mean()
    gamma0 = float(resid @ resid) / n
    variance = gamma0
    for lag in range(1, lags + 1):
        cov = float(resid[lag:] @ resid[:-lag]) / n
        variance += 2 * (1 - lag / (lags + 1)) * cov
    if variance <= 0:
        return np.nan
    return float(arr.mean() / np.sqrt(variance / n))


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
