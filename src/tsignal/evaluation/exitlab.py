"""청산 규칙을 **전 표본**으로 잰다.

왜 다시 만드나
--------------
`pathprofile.exit_timing` 은 청산 조건이 **켜진 신호만** 모아 그때의 수익
중앙값을 냈다. 그러면 "오르면 판다" 류의 조건이 항상 이긴다 — 안 오른
종목은 애초에 과매수 구간에 닿지 못해 표본에서 빠지기 때문이다. 실측에서
`%R > -5` 는 68.5% 만 발동했고, 나머지 31.5% 는 계산에 들어가지 않았다.

여기서는 **모든 신호가 반드시 청산가를 갖는다.** 조건이 켜지면 거기서,
끝까지 안 켜지면 만기에 시간 청산. 그래야 규칙끼리 비교가 성립한다.

체결 가정 (프로젝트 전체와 동일)
--------------------------------
- 진입: 신호 t봉 종가 확정 → t+1봉 **시가**
- 손절: 장중 저가가 손절가에 닿으면 그 가격. 시가가 이미 아래면 **시가**
  (갭하락에서 손절가로 나갈 수 있다고 가정하면 성과가 부풀려진다)
- 익절: 장중 고가가 목표가에 닿으면 그 가격. 시가가 이미 위면 시가
- 지표 청산: 그 봉 **종가**로 판정 → 다음 봉 **시가** 체결
- 시간 청산: 만기 봉 종가
- 같은 봉에서 손절과 익절이 둘 다 닿으면 **손절을 먼저** 잡는다.
  봉 안의 순서는 일봉으로 알 수 없으므로 불리한 쪽으로 가정한다.

기준선을 반드시 같이 낸다
-------------------------
손절은 어떤 진입에도 분포를 바꾼다. 신호 집합에서만 재면 "손절이 좋다" 가
신호의 성질인지 손절 자체의 성질인지 갈리지 않는다. `baseline_events` 로
같은 종목·같은 기간의 아무 날이나 뽑아 **같은 규칙**을 걸어 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

STOCK_ROUND_TRIP_BPS = 28.0

Flagger = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Exit:
    """청산 규칙 하나. 여러 장치를 동시에 걸 수 있고, 먼저 닿는 것이 이긴다."""

    name: str
    stop_loss: float | None = None      # 진입가 대비 하락률
    take_profit: float | None = None    # 진입가 대비 상승률
    trail: float | None = None          # 최고가 대비 하락률
    atr_stop: float | None = None       # 진입일 ATR 의 배수만큼 아래
    flag: Flagger | None = None         # 지표 조건 (t봉 종가 판정)
    horizon: int = 60                   # 아무것도 안 걸리면 여기서 시간 청산


def _resolve_one(
    i: int,
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    flag: np.ndarray | None, atr: np.ndarray | None, rule: Exit,
) -> tuple[float, int, str] | None:
    """신호 i 하나의 청산가·보유봉수·사유. 반드시 하나를 돌려준다."""
    n = len(close)
    if i + 1 >= n:
        return None
    entry = open_[i + 1]
    if not np.isfinite(entry) or entry <= 0:
        return None

    fixed_stop = entry * (1 - rule.stop_loss) if rule.stop_loss else None
    if rule.atr_stop is not None and atr is not None and np.isfinite(atr[i]):
        level = entry - rule.atr_stop * atr[i]
        fixed_stop = level if fixed_stop is None else max(fixed_stop, level)
    target = entry * (1 + rule.take_profit) if rule.take_profit else None

    last = min(i + rule.horizon, n - 1)
    # 트레일링 고점은 **직전 봉까지**로 잡는다. 같은 봉의 고가를 쓰면
    # "고가가 저가보다 먼저 나왔다" 를 가정하게 되는데, 일봉으로는 알 수 없다.
    peak = entry
    for j in range(i + 1, last + 1):
        stop = fixed_stop
        if rule.trail is not None and np.isfinite(peak):
            trail_level = peak * (1 - rule.trail)
            stop = trail_level if stop is None else max(stop, trail_level)

        if stop is not None and np.isfinite(low[j]) and low[j] <= stop:
            price = min(open_[j], stop) if np.isfinite(open_[j]) else stop
            return float(price), j - i, "손절"

        if target is not None and np.isfinite(high[j]) and high[j] >= target:
            price = max(open_[j], target) if np.isfinite(open_[j]) else target
            return float(price), j - i, "익절"

        if flag is not None and j < last and flag[j] and j + 1 < n \
                and np.isfinite(open_[j + 1]):
            return float(open_[j + 1]), j + 1 - i, "지표"

        if np.isfinite(high[j]):
            peak = max(peak, high[j])

    return float(close[last]), last - i, "만기"


def resolve(
    events: Mapping[str, pd.Series],
    candles: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame] | None = None,
    *,
    rule: Exit,
    cost_bps: float = STOCK_ROUND_TRIP_BPS,
) -> pd.DataFrame:
    """규칙 하나를 전 신호에 걸어 매매 목록을 만든다.

    신호 하나당 정확히 한 줄. 발동하지 않은 신호도 만기 청산으로 남는다 —
    이것이 `pathprofile.exit_timing` 과의 유일하고 결정적인 차이다.
    """
    rows = []
    for code, series in events.items():
        frame = candles.get(code)
        if frame is None or not series.any():
            continue
        open_ = frame["open"].to_numpy(float)
        high = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        close = frame["close"].to_numpy(float)

        flag = None
        if rule.flag is not None:
            feat = (features or {}).get(code)
            if feat is None:
                continue
            flag = rule.flag(frame, feat).fillna(False).to_numpy(bool)

        atr = None
        if rule.atr_stop is not None:
            tr = pd.concat([
                frame["high"] - frame["low"],
                (frame["high"] - frame["close"].shift()).abs(),
                (frame["low"] - frame["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=14).mean().to_numpy(float)

        pos = {ts: k for k, ts in enumerate(frame.index)}
        for stamp in series[series].index:
            i = pos.get(stamp)
            if i is None:
                continue
            got = _resolve_one(i, open_, high, low, close, flag, atr, rule)
            if got is None:
                continue
            price, bars, reason = got
            entry = open_[i + 1]
            rows.append({
                "code": code, "신호일": stamp, "진입": entry, "청산": price,
                "보유봉": bars, "사유": reason,
                "수익": price / entry - 1 - cost_bps / 10_000,
            })
    return pd.DataFrame(rows)


def summarise(trades: pd.DataFrame, *, label: str = "") -> dict:
    """매매 목록 하나를 한 줄로. 사유별 비중을 반드시 같이 낸다."""
    if trades.empty:
        return {"규칙": label, "표본": 0}
    ret = trades["수익"].to_numpy(float)
    counts = trades["사유"].value_counts(normalize=True)
    return {
        "규칙": label,
        "표본": len(trades),
        "평균": float(ret.mean()),
        "중앙값": float(np.median(ret)),
        "승률": float((ret > 0).mean()),
        "보유봉": float(trades["보유봉"].median()),
        "손절%": float(counts.get("손절", 0.0)),
        "익절%": float(counts.get("익절", 0.0)),
        "지표%": float(counts.get("지표", 0.0)),
        "만기%": float(counts.get("만기", 0.0)),
        "하위5%": float(np.quantile(ret, 0.05)),
        "최악": float(ret.min()),
    }


def compare(
    events: Mapping[str, pd.Series],
    candles: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame] | None,
    rules: Sequence[Exit],
    *,
    cost_bps: float = STOCK_ROUND_TRIP_BPS,
    baseline: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """규칙들을 같은 신호 집합에 걸어 나란히 놓는다.

    `baseline` 을 주면 같은 규칙을 아무 날 진입에도 걸어 **차이**를 낸다.
    손절이 신호와 무관하게 좋은 것인지 가려내는 유일한 방법이다.
    """
    rows = []
    for rule in rules:
        got = summarise(resolve(events, candles, features, rule=rule,
                                cost_bps=cost_bps), label=rule.name)
        if baseline is not None and got.get("표본"):
            base = summarise(resolve(baseline, candles, features, rule=rule,
                                     cost_bps=cost_bps), label=rule.name)
            if base.get("표본"):
                got["기준선 평균"] = base["평균"]
                got["차이"] = got["평균"] - base["평균"]
                got["기준선 승률"] = base["승률"]
        rows.append(got)
    return pd.DataFrame(rows)


def baseline_events(
    candles: Mapping[str, pd.DataFrame],
    *,
    step: int = 40,
    seed: int = 0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, pd.Series]:
    """같은 종목·같은 기간의 '아무 날' 진입 집합."""
    rng = np.random.default_rng(seed)
    out: dict[str, pd.Series] = {}
    for code, frame in candles.items():
        if len(frame) < step + 2:
            continue
        hit = pd.Series(False, index=frame.index)
        offset = int(rng.integers(0, step))
        hit.iloc[offset::step] = True
        if start is not None:
            hit &= frame.index >= start
        if end is not None:
            hit &= frame.index <= end
        if hit.any():
            out[code] = hit
    return out


def by_year(
    events: Mapping[str, pd.Series],
    candles: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame] | None,
    rule: Exit,
    *,
    cost_bps: float = STOCK_ROUND_TRIP_BPS,
    baseline: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """해마다 다시. 전체 평균 하나로는 한 해가 전부를 만든 경우를 못 거른다."""
    trades = resolve(events, candles, features, rule=rule, cost_bps=cost_bps)
    if trades.empty:
        return pd.DataFrame()
    trades["연도"] = pd.DatetimeIndex(trades["신호일"]).year
    base = None
    if baseline is not None:
        base = resolve(baseline, candles, features, rule=rule, cost_bps=cost_bps)
        if not base.empty:
            base["연도"] = pd.DatetimeIndex(base["신호일"]).year

    rows = []
    for year, group in trades.groupby("연도"):
        row = {"연도": int(year), "표본": len(group),
               "평균": float(group["수익"].mean()),
               "승률": float((group["수익"] > 0).mean())}
        if base is not None:
            sub = base[base["연도"] == year]
            if len(sub) >= 30:
                row["기준선"] = float(sub["수익"].mean())
                row["차이"] = row["평균"] - row["기준선"]
        rows.append(row)
    return pd.DataFrame(rows)
