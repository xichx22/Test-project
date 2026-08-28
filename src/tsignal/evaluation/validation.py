"""과최적화 방어 장치.

지표 60개 × 신호 39개를 한 데이터에 던지면, 아무 엣지가 없어도 t>2 인 신호가
확률적으로 몇 개는 나온다. 그 착시를 걷어내기 위한 세 가지 검사:

  1. split_sample   — 앞뒤로 나눠 IS 에서 좋았던 게 OOS 에서도 좋은가
  2. stability      — 기간을 잘게 쪼개 부호가 일관되는가 (한 구간이 다 벌어준 건 아닌가)
  3. deflated_t     — 여러 신호를 동시에 시험한 만큼 t 문턱을 올린다
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def split_sample(
    candles: pd.DataFrame,
    signal_frame: pd.DataFrame,
    *,
    horizon: int = 5,
    train_ratio: float = 0.6,
    entry: str = "next_open",
) -> pd.DataFrame:
    """시계열 앞 train_ratio 를 IS, 나머지를 OOS 로 두고 같은 통계를 각각 낸다.

    IS 에서 t 가 높은데 OOS 에서 부호가 뒤집히면 그 신호는 우연이었다고 본다.
    """
    from .forward import forward_returns

    fwd = forward_returns(candles, (horizon,), entry=entry)[f"fwd_{horizon}"]
    cut = int(len(candles) * train_ratio)
    is_idx, oos_idx = candles.index[:cut], candles.index[cut:]

    rows = []
    for name in signal_frame.columns:
        mask = signal_frame[name].astype(bool)
        r_is = fwd.loc[is_idx][mask.loc[is_idx]].dropna()
        r_oos = fwd.loc[oos_idx][mask.loc[oos_idx]].dropna()
        rows.append({
            "signal": name,
            "n_is": len(r_is), "exp_is": r_is.mean() if len(r_is) else np.nan,
            "t_is": metrics.t_stat(r_is),
            "n_oos": len(r_oos), "exp_oos": r_oos.mean() if len(r_oos) else np.nan,
            "t_oos": metrics.t_stat(r_oos),
        })
    out = pd.DataFrame(rows).set_index("signal")
    out["sign_agree"] = np.sign(out["exp_is"]) == np.sign(out["exp_oos"])
    return out.sort_values("t_oos", ascending=False)


def stability(
    candles: pd.DataFrame,
    events: pd.Series,
    *,
    horizon: int = 5,
    folds: int = 5,
    entry: str = "next_open",
) -> pd.DataFrame:
    """기간을 folds 등분해 구간별 기대값을 낸다. 한 구간에 성과가 몰렸는지 본다."""
    from .forward import forward_returns

    fwd = forward_returns(candles, (horizon,), entry=entry)[f"fwd_{horizon}"]
    mask = events.reindex(candles.index).fillna(False).astype(bool)
    bounds = np.linspace(0, len(candles), folds + 1, dtype=int)

    rows = []
    for i in range(folds):
        window = candles.index[bounds[i]:bounds[i + 1]]
        r = fwd.loc[window][mask.loc[window]].dropna()
        rows.append({
            "fold": i + 1,
            "from": window[0] if len(window) else pd.NaT,
            "to": window[-1] if len(window) else pd.NaT,
            "n": len(r),
            "expectancy": float(r.mean()) if len(r) else np.nan,
            "win_rate": metrics.win_rate(r),
        })
    out = pd.DataFrame(rows).set_index("fold")
    out.attrs["positive_folds"] = int((out["expectancy"] > 0).sum())
    return out


def deflated_threshold(n_trials: int, alpha: float = 0.05) -> float:
    """다중검정 보정된 t 문턱 (Šidák).

    신호 39개를 동시에 시험하면 5% 유의수준의 문턱은 t≈2.0 이 아니라 t≈3.3 이다.
    이 프로젝트가 신호를 '채택'하는 기준은 보정 후 문턱이다.
    """
    if n_trials < 1:
        raise ValueError("n_trials 는 1 이상이어야 합니다.")
    per_trial = 1 - (1 - alpha) ** (1 / n_trials)
    return abs(_norm_ppf(per_trial / 2))


def _norm_ppf(p: float) -> float:
    """표준정규 분위수 (Acklam 근사). scipy 의존을 피하기 위한 자체 구현."""
    if not 0 < p < 1:
        raise ValueError("p 는 (0,1) 범위여야 합니다.")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def p_from_t(t: float | np.ndarray, n: int | np.ndarray) -> np.ndarray:
    """양측 p값. 표본이 100을 넘는 구간만 쓰므로 정규근사로 충분하다.

    (자유도 100에서 t분포와 정규분포의 양측 p값 차이는 유의수준 근처에서 3% 미만이다.)
    """
    z = np.abs(np.asarray(t, dtype=float))
    return 2 * (1 - _norm_cdf(z))


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    from math import erf, sqrt

    vec = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))
    return vec(z)


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """BH-FDR. 기대 오발견 비율을 alpha 이하로 통제한다.

    Šidák/Bonferroni 는 '하나라도 틀리면 안 된다'(FWER)를 통제해서,
    조합 수천 개를 훑는 스크리닝에서는 지나치게 보수적이다.
    BH 는 '채택한 것 중 몇 %가 가짜여도 되는가'를 통제하므로 탐색 단계에 맞다.
    두 잣대를 나란히 보여주고, 최종 판단은 OOS 로 한다.

    반환: 각 가설의 기각 여부(bool 배열)
    """
    p = np.asarray(pvalues, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    if not ok.any():
        return out

    idx = np.argsort(p[ok])
    ranked = p[ok][idx]
    m = len(ranked)
    thresholds = alpha * np.arange(1, m + 1) / m
    passed = ranked <= thresholds
    if not passed.any():
        return out
    cutoff = ranked[np.max(np.flatnonzero(passed))]
    out[ok] = p[ok] <= cutoff
    return out


def verdict(row: pd.Series, *, t_threshold: float, min_events: int = 30) -> str:
    """스크리너 한 줄 → 채택/보류/기각 판정."""
    n = row.get("n", 0)
    if n < min_events:
        return "표본부족"
    t = row.get("t_oos", row.get("t_5", np.nan))
    if not np.isfinite(t):
        return "판정불가"
    if abs(t) >= t_threshold and t > 0:
        return "채택후보"
    if t > 1.0:
        return "보류"
    return "기각"
