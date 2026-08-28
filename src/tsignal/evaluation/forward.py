"""이벤트 스터디 — "이 신호가 뜬 다음에 실제로 무슨 일이 일어났는가".

전략을 짜기 전에 먼저 던져야 할 질문이다.
신호 시점 t 이후 k봉 동안의 수익률을, 같은 기간 전체의 무조건부 분포와 비교한다.
차이(엣지)가 없으면 그 신호로는 어떤 규칙을 만들어도 돈이 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics

DEFAULT_HORIZONS = (1, 3, 5, 10, 20, 40)


def forward_returns(
    candles: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    *,
    entry: str = "next_open",
) -> pd.DataFrame:
    """t 봉에서 신호가 떴을 때 얻게 될 k봉 후 수익률.

    entry="next_open": t+1 시가 진입 → t+1+k 시가 청산. 실제 체결 가능한 정의다.
    entry="close":     t 종가 진입 → t+k 종가 청산. 낙관적이라 상한선 참고용.
    """
    if entry == "next_open":
        buy = candles["open"].shift(-1)
        base = candles["open"]
        out = {f"fwd_{k}": (base.shift(-(1 + k)) / buy - 1) for k in horizons}
    elif entry == "close":
        buy = candles["close"]
        out = {f"fwd_{k}": (candles["close"].shift(-k) / buy - 1) for k in horizons}
    else:
        raise ValueError(f"알 수 없는 entry='{entry}'")
    return pd.DataFrame(out, index=candles.index)


def excursions(candles: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """MFE/MAE — 신호 이후 horizon 봉 동안 최대 유리/불리 움직임(%).

    익절/손절 폭을 '감'이 아니라 분포에서 정하기 위한 재료.
    """
    buy = candles["open"].shift(-1)
    # t+1 ~ t+horizon 구간의 고가/저가. 진입 이후 구간만 보도록 shift(-1) 후 롤링.
    fwd_high = candles["high"].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    fwd_low = candles["low"].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    return pd.DataFrame({"mfe": fwd_high / buy - 1, "mae": fwd_low / buy - 1}, index=candles.index)


@dataclass
class EventStudyResult:
    signal: str
    table: pd.DataFrame       # horizon 별 통계
    baseline: pd.DataFrame    # 같은 horizon 의 무조건부 분포
    n_events: int

    @property
    def edge(self) -> pd.DataFrame:
        """신호 기대값 - 무조건부 기대값. 이게 진짜 엣지다."""
        cols = ["expectancy", "win_rate"]
        return (self.table[cols] - self.baseline[cols]).add_prefix("edge_")


def event_study(
    candles: pd.DataFrame,
    events: pd.Series,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    name: str = "signal",
    entry: str = "next_open",
) -> EventStudyResult:
    fwd = forward_returns(candles, horizons, entry=entry)
    mask = events.reindex(candles.index).fillna(False).astype(bool)

    rows, base_rows = [], []
    for k in horizons:
        col = f"fwd_{k}"
        rows.append({"horizon": k, **metrics.summarize(fwd.loc[mask, col])})
        base_rows.append({"horizon": k, **metrics.summarize(fwd[col])})

    table = pd.DataFrame(rows).set_index("horizon")
    baseline = pd.DataFrame(base_rows).set_index("horizon")
    return EventStudyResult(name, table, baseline, int(mask.sum()))


def screen_signals(
    candles: pd.DataFrame,
    signal_frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    entry: str = "next_open",
    min_events: int = 30,
) -> pd.DataFrame:
    """모든 신호를 한 번에 훑어 '엣지가 있는 후보'만 남기는 1차 스크리너.

    반환 컬럼
      n            : 신호 발생 횟수 (적으면 통계가 무의미하다)
      exp_{k}      : k봉 후 기대수익률
      edge_{k}     : 무조건부 대비 초과 기대수익률
      t_{k}        : 기대값이 0이라는 귀무가설의 t 통계량
      win_{k}      : 승률
    """
    fwd = forward_returns(candles, horizons, entry=entry)
    unconditional = {k: float(fwd[f"fwd_{k}"].mean()) for k in horizons}

    rows = []
    for name in signal_frame.columns:
        mask = signal_frame[name].astype(bool)
        n = int(mask.sum())
        row: dict[str, object] = {"signal": name, "n": n}
        for k in horizons:
            r = fwd.loc[mask, f"fwd_{k}"].dropna()
            row[f"exp_{k}"] = float(r.mean()) if len(r) else np.nan
            row[f"edge_{k}"] = row[f"exp_{k}"] - unconditional[k] if len(r) else np.nan
            row[f"t_{k}"] = metrics.t_stat(r)
            row[f"win_{k}"] = metrics.win_rate(r)
        rows.append(row)

    out = pd.DataFrame(rows).set_index("signal")
    out.attrs["unconditional"] = unconditional
    out.attrs["min_events"] = min_events
    return out.sort_values(f"t_{horizons[len(horizons) // 2]}", ascending=False)
