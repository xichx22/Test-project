"""신호 이후의 경로를 기술한다 — 전략이 아니라 관찰.

무엇이 다른가
-------------
이 프로젝트의 다른 모든 검증은 "이 신호가 벤치마크를 이기는가" 를 물었고
답은 거의 항상 아니오였다. 여기서는 다른 것을 묻는다 —
**신호가 뜬 뒤 차트가 실제로 어떤 모양으로 움직였는가.**

이기지 못하는 신호도 경로에는 구조가 있을 수 있다. 예를 들어 "평균적으로는
본전이지만 5일 안에 최고점을 찍고 그 뒤 흘러내린다" 는 사실은 초과수익이
0이어도 참일 수 있고, 그건 청산 시점을 정할 때 쓸 수 있는 정보다.

주의: 여기 나오는 숫자는 **기술통계**이지 검정이 아니다. "신호 후 중앙값이
+2%" 라는 문장은 "그 신호를 사면 2% 번다" 를 뜻하지 않는다. 같은 기간 아무
종목이나 사도 얼마였는지(기준선)를 항상 같이 봐야 한다.

측정하는 것
-----------
`forward_paths`   신호일 다음 봉 시가 진입 기준의 누적 경로 (N × horizon)
`path_shape`      중앙 경로, 최고·최저 도달일, MFE/MAE 분포
`exit_timing`     여러 청산 조건이 각각 언제 켜지고 그때 수익이 얼마인가
`turn_signature`  고점 직전 며칠 동안 어떤 지표가 먼저 꺾이는가
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

STOCK_ROUND_TRIP_BPS = 28.0


# =====================================================================
# 경로 추출
# =====================================================================

def forward_paths(
    events: Mapping[str, pd.Series],
    candles: Mapping[str, pd.DataFrame],
    *,
    horizon: int = 60,
    entry: str = "next_open",
) -> np.ndarray:
    """신호마다 앞으로 `horizon` 봉의 누적수익률 경로를 뽑는다.

    진입은 신호 다음 봉 **시가**다 (이 프로젝트 전체 규약). 종가 기준으로
    재면 사기 전에 벌어진 갭이 성과에 들어간다.

    반환 (N, horizon) — 행 하나가 신호 하나, 열 j 는 진입 후 j+1 봉째 수익률.
    구간이 모자란 신호는 NaN 으로 남긴다 (버리면 최근 신호가 통째로 빠진다).
    """
    rows: list[np.ndarray] = []
    for code, series in events.items():
        frame = candles.get(code)
        if frame is None or not series.any():
            continue
        close = frame["close"].to_numpy(float)
        open_ = frame["open"].to_numpy(float)
        pos = {ts: i for i, ts in enumerate(frame.index)}
        for stamp in series[series].index:
            i = pos.get(stamp)
            if i is None or i + 1 >= len(close):
                continue
            base = open_[i + 1] if entry == "next_open" else close[i]
            if not np.isfinite(base) or base <= 0:
                continue
            path = np.full(horizon, np.nan)
            end = min(i + 1 + horizon, len(close))
            path[: end - (i + 1)] = close[i + 1: end] / base - 1
            rows.append(path)
    return np.vstack(rows) if rows else np.empty((0, horizon))


def baseline_paths(
    candles: Mapping[str, pd.DataFrame],
    *,
    horizon: int = 60,
    step: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """같은 종목·같은 기간에서 **아무 날에나** 산 경로.

    신호 경로를 이것과 나란히 놓지 않으면 "신호 후 +3%" 가 신호 덕인지
    그냥 시장이 오른 것인지 알 수 없다.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for frame in candles.values():
        close = frame["close"].to_numpy(float)
        open_ = frame["open"].to_numpy(float)
        if len(close) < horizon + 2:
            continue
        offset = int(rng.integers(0, step))
        for i in range(offset, len(close) - 1, step):
            base = open_[i + 1]
            if not np.isfinite(base) or base <= 0:
                continue
            path = np.full(horizon, np.nan)
            end = min(i + 1 + horizon, len(close))
            path[: end - (i + 1)] = close[i + 1: end] / base - 1
            rows.append(path)
    return np.vstack(rows) if rows else np.empty((0, horizon))


# =====================================================================
# 경로 모양
# =====================================================================

@dataclass
class PathShape:
    n: int
    median: np.ndarray            # 중앙 경로
    q25: np.ndarray
    q75: np.ndarray
    days_to_peak: np.ndarray      # 신호별 최고점 도달일 (1부터)
    days_to_trough: np.ndarray
    mfe: np.ndarray               # 최대 상승폭
    mae: np.ndarray               # 최대 하락폭
    positive_rate: np.ndarray     # 봉마다 양수 비율

    def summary(self, horizon_marks: Sequence[int] = (1, 3, 5, 10, 20, 40, 60)) -> pd.DataFrame:
        rows = []
        for d in horizon_marks:
            if d > len(self.median):
                continue
            rows.append({
                "경과일": d,
                "중앙값": self.median[d - 1],
                "하위25%": self.q25[d - 1],
                "상위25%": self.q75[d - 1],
                "양수율": self.positive_rate[d - 1],
            })
        return pd.DataFrame(rows)

    def peak_summary(self) -> dict:
        return {
            "표본": self.n,
            "최고점 도달일 중앙값": float(np.nanmedian(self.days_to_peak)),
            "최저점 도달일 중앙값": float(np.nanmedian(self.days_to_trough)),
            "MFE 중앙값": float(np.nanmedian(self.mfe)),
            "MAE 중앙값": float(np.nanmedian(self.mae)),
            "최고점이 먼저인 비율": float(np.nanmean(
                self.days_to_peak < self.days_to_trough)),
        }


def path_shape(paths: np.ndarray) -> PathShape:
    """경로 행렬에서 모양을 요약한다."""
    if paths.size == 0:
        empty = np.array([])
        return PathShape(0, empty, empty, empty, empty, empty, empty, empty, empty)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(paths, axis=0)
        q25 = np.nanquantile(paths, 0.25, axis=0)
        q75 = np.nanquantile(paths, 0.75, axis=0)
        positive = np.nanmean(paths > 0, axis=0)
        # 전부 NaN 인 행이 있으면 argmax 가 터진다 — 마스크로 걸러 낸다
        usable = ~np.all(np.isnan(paths), axis=1)
        sub = paths[usable]
        peak = np.nanargmax(sub, axis=1) + 1
        trough = np.nanargmin(sub, axis=1) + 1
        mfe = np.nanmax(sub, axis=1)
        mae = np.nanmin(sub, axis=1)
    return PathShape(len(sub), median, q25, q75, peak, trough, mfe, mae, positive)


# =====================================================================
# 청산 조건이 언제 켜지는가
# =====================================================================

ExitRule = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


def default_exit_rules() -> dict[str, ExitRule]:
    """널리 쓰이는 청산 조건들. 전부 t 봉 종가로 판정된다.

    켜지는 봉의 **다음 봉 시가**에 판다고 가정한다 — 진입과 같은 규약이다.
    """
    def macd_down(c, f):
        return f["macd_hist"] < 0

    def stoch_cross_down(c, f):
        return (f["stoch_k"] < f["stoch_d"]) & (f["stoch_k"].shift(1) >= f["stoch_d"].shift(1))

    def below_ema20(c, f):
        return c["close"] < f["ema_20"]

    def below_ema5(c, f):
        return c["close"] < f["ema_5"]

    def cloud_break(c, f):
        floor = pd.concat([f["senkou_a"], f["senkou_b"]], axis=1).min(axis=1)
        return c["close"] < floor

    def williams_overbought(c, f):
        return f["williams_r_14"] > -20

    def volume_down_bar(c, f):
        # 거래량이 20일 평균의 2배인데 음봉 — 분산(distribution) 의 전형
        avg = c["volume"].rolling(20, min_periods=20).mean()
        return (c["volume"] > 2 * avg) & (c["close"] < c["open"])

    def kijun_break(c, f):
        return c["close"] < f["kijun"]

    return {
        "MACD 히스토그램 음전": macd_down,
        "스토캐스틱 %K가 %D 하향돌파": stoch_cross_down,
        "종가가 5일선 아래": below_ema5,
        "종가가 20일선 아래": below_ema20,
        "일목 기준선 아래": kijun_break,
        "일목 구름대 이탈": cloud_break,
        "윌리엄스 과매수(-20 위)": williams_overbought,
        "거래량 2배 + 음봉": volume_down_bar,
    }


def exit_timing(
    events: Mapping[str, pd.Series],
    candles: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame],
    *,
    rules: Mapping[str, ExitRule] | None = None,
    horizon: int = 60,
    cost_bps: float = STOCK_ROUND_TRIP_BPS,
) -> pd.DataFrame:
    """각 청산 조건이 신호 후 언제 켜지고, 그때 팔면 얼마인가.

    비교 대상은 두 개다 — **만기까지 들고 있었을 때**와 **그 조건에 팔았을 때**.
    조건에 판 쪽이 나으면 그 조건이 정보를 담고 있다는 뜻이고,
    나쁘면 그냥 일찍 잘라 상승분을 버린 것이다.
    """
    rule_set = dict(rules or default_exit_rules())
    records: dict[str, list[tuple[int, float]]] = {k: [] for k in rule_set}
    held: list[float] = []

    for code, series in events.items():
        frame, feat = candles.get(code), features.get(code)
        if frame is None or feat is None or not series.any():
            continue
        close = frame["close"].to_numpy(float)
        open_ = frame["open"].to_numpy(float)
        pos = {ts: i for i, ts in enumerate(frame.index)}
        flags = {name: fn(frame, feat).fillna(False).to_numpy(bool)
                 for name, fn in rule_set.items()}

        for stamp in series[series].index:
            i = pos.get(stamp)
            if i is None or i + 2 >= len(close):
                continue
            base = open_[i + 1]
            if not np.isfinite(base) or base <= 0:
                continue
            end = min(i + 1 + horizon, len(close))
            held.append(close[end - 1] / base - 1 - cost_bps / 10_000)
            for name, flag in flags.items():
                window = flag[i + 1: end]
                where = np.flatnonzero(window)
                if where.size == 0:
                    continue
                hit = i + 1 + int(where[0])
                if hit + 1 >= len(open_):
                    continue
                ret = open_[hit + 1] / base - 1 - cost_bps / 10_000
                records[name].append((int(where[0]) + 1, float(ret)))

    rows = []
    held_median = float(np.median(held)) if held else np.nan
    for name, hits in records.items():
        if not hits:
            continue
        days = np.array([d for d, _ in hits], dtype=float)
        rets = np.array([r for _, r in hits], dtype=float)
        rows.append({
            "청산 조건": name,
            "발동률": len(hits) / max(len(held), 1),
            "발동일 중앙값": float(np.median(days)),
            "청산 수익 중앙값": float(np.median(rets)),
            "만기보유 대비": float(np.median(rets)) - held_median,
            "표본": len(hits),
        })
    columns = ["청산 조건", "발동률", "발동일 중앙값", "청산 수익 중앙값",
               "만기보유 대비", "표본"]
    # 어떤 조건도 켜지지 않으면 rows 가 비고, 그러면 정렬할 컬럼이 없다.
    out = (pd.DataFrame(rows).sort_values("만기보유 대비", ascending=False)
           if rows else pd.DataFrame(columns=columns))
    out.attrs["held_median"] = held_median
    out.attrs["n_events"] = len(held)
    return out


# =====================================================================
# 고점 직전에 무엇이 보이는가
# =====================================================================

def turn_signature(
    events: Mapping[str, pd.Series],
    candles: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame],
    *,
    horizon: int = 60,
    lookback: int = 3,
    columns: Sequence[str] = ("macd_hist", "stoch_k", "williams_r_14",
                              "rsi_14", "cmf_20", "mfi_14"),
) -> pd.DataFrame:
    """신호 후 최고점 부근의 지표 상태를, 신호 시점과 비교한다.

    "무엇이 먼저 꺾이는가" 를 보려면 고점 **직전**을 봐야 한다. 고점 당일은
    이미 늦고, 그 값은 사후에만 알 수 있다. 여기서 나오는 것은 예측 규칙이
    아니라 **정황**이다 — 그 정황이 하락 전에만 나타나는지는 따로 재야 한다.
    """
    rows = []
    for code, series in events.items():
        frame, feat = candles.get(code), features.get(code)
        if frame is None or feat is None or not series.any():
            continue
        close = frame["close"].to_numpy(float)
        volume = frame["volume"].to_numpy(float)
        avg_vol = frame["volume"].rolling(20, min_periods=20).mean().to_numpy()
        pos = {ts: i for i, ts in enumerate(frame.index)}
        cols = [c for c in columns if c in feat.columns]
        values = {c: feat[c].to_numpy(float) for c in cols}

        for stamp in series[series].index:
            i = pos.get(stamp)
            if i is None or i + 2 >= len(close):
                continue
            end = min(i + 1 + horizon, len(close))
            seg = close[i + 1: end]
            if len(seg) < lookback + 2:
                continue
            peak = i + 1 + int(np.nanargmax(seg))
            pre = max(i + 1, peak - lookback)
            row = {"code": code, "days_to_peak": peak - i,
                   "peak_gain": close[peak] / close[i] - 1}
            for c in cols:
                row[f"{c}_신호일"] = values[c][i]
                row[f"{c}_고점직전"] = np.nanmean(values[c][pre: peak + 1])
            if np.isfinite(avg_vol[peak]) and avg_vol[peak] > 0:
                row["거래량비_고점"] = volume[peak] / avg_vol[peak]
            if np.isfinite(avg_vol[i]) and avg_vol[i] > 0:
                row["거래량비_신호일"] = volume[i] / avg_vol[i]
            rows.append(row)
    return pd.DataFrame(rows)
