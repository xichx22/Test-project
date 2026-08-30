"""신호에 조건을 얹으면 정말 나아지는가.

왜 따로 만드나
--------------
"컵앤핸들에 거래량 조건을 걸면 좋아진다" 같은 주장은 검증하기 까다롭다.
조건을 걸면 표본이 바뀌므로, 좋아진 것이 **신호 덕인지 조건 자체가 좋은 날을
고른 것인지** 구분되지 않기 때문이다. 시장이 200일선 위일 때만 사면 성적이
좋아지는데, 그건 그 기간에 아무 종목이나 사도 좋았기 때문일 수 있다.

여기서 쓰는 방법
----------------
모든 종목의 **모든 봉**에 같은 정황 컬럼을 붙인 표를 만든다. 그러면

    신호 = 그 표의 부분집합 (불리언 마스크)
    기준선 = 같은 조건을 만족하는 **모든 날**

이 되고, 조건 A 를 건 신호는 **조건 A 를 건 기준선하고만** 비교된다.
"조건을 걸었더니 좋아졌다" 가 신호 덕인지 조건 덕인지 바로 갈린다.

실측(2020~2025, 1,063종목, 신호 28종 × 조건 23개 = 624칸)에서
조건 23개의 **평균 효과가 전부 음수**였다. 필터를 거는 것 자체는 도움이
되지 않았고, 이긴 칸들은 조건이 아니라 신호(RSI·MFI 과매도 반등)가 끌었다.

주의: 창이 겹치므로 표본 수는 독립 표본 수보다 훨씬 적다. t 값을 내지 말고
**중앙값 차이와 해마다의 방향**을 보라.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

Condition = Callable[[pd.DataFrame], pd.Series]


def context_table(
    candles: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame],
    *,
    horizons: Sequence[int] = (20, 60),
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    market: pd.Series | None = None,
) -> pd.DataFrame:
    """전 종목 · 전 봉의 정황 + 앞으로의 결과를 한 표로.

    정황 컬럼은 전부 **그 봉까지의 정보**로만 만든다. 결과 컬럼(fwd*)만
    미래를 본다 — 진입은 다음 봉 시가다 (이 프로젝트 전체 규약).

    `market` 은 지수 종가. 넘기면 200일 이동평균 위/아래를 `mkt_up` 으로 붙인다.
    """
    frames = []
    for code, frame in candles.items():
        feat = features.get(code)
        if feat is None or frame.empty:
            continue
        close, open_, high = frame["close"], frame["open"], frame["high"]
        volume = frame["volume"]
        avg20 = volume.rolling(20, min_periods=20).mean()
        true_range = pd.concat([
            frame["high"] - frame["low"],
            (frame["high"] - close.shift()).abs(),
            (frame["low"] - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        width = (feat["bb_upper"] - feat["bb_lower"]).replace(0, np.nan)
        cloud = pd.concat([feat["senkou_a"], feat["senkou_b"]], axis=1).max(axis=1)

        table = pd.DataFrame({
            "종가>60일선": close > feat["ema_60"],
            "20일선>60일선": feat["ema_20"] > feat["ema_60"],
            "52주고점대비": close / high.rolling(250, min_periods=100).max(),
            "3개월수익률": close / close.shift(60) - 1,
            "12-1모멘텀": close.shift(20) / close.shift(250) - 1,
            "거래량배수": volume / avg20,
            "거래대금20일": (close * volume).rolling(20, min_periods=20).mean(),
            "RSI": feat["rsi_14"],
            "%R": feat["williams_r_14"],
            "MACD히스토": feat["macd_hist"],
            "스토캐스틱차": feat["stoch_k"] - feat["stoch_d"],
            "CMF": feat["cmf_20"],
            "MFI": feat["mfi_14"],
            "변동성": atr / close,
            "종가>구름대": close > cloud,
            "몸통": (close - open_) / open_,
            "볼린저위치": (close - feat["bb_lower"]) / width,
        }, index=frame.index)
        entry = open_.shift(-1)
        for h in horizons:
            table[f"fwd{h}"] = close.shift(-(h + 1)) / entry - 1
        table["code"] = code
        frames.append(table)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    out["연도"] = out.index.year
    if market is not None:
        up = (market > market.rolling(200, min_periods=200).mean()).rename("mkt_up")
        out = out.join(up, how="left")
        out["mkt_up"] = out["mkt_up"].fillna(False)
    return out


def signal_mask(table: pd.DataFrame, events: Mapping[str, pd.Series]) -> np.ndarray:
    """`events` (종목 → 불리언 시리즈) 를 정황 표 위의 마스크로 옮긴다."""
    pairs = [(code, stamp)
             for code, series in events.items()
             for stamp in series[series].index]
    if not pairs:
        return np.zeros(len(table), dtype=bool)
    key = pd.MultiIndex.from_arrays([table["code"].to_numpy(), table.index])
    return key.isin(pd.MultiIndex.from_tuples(pairs))


def cell(
    table: pd.DataFrame,
    signal: np.ndarray,
    condition: np.ndarray,
    *,
    horizon: int = 20,
    min_signals: int = 100,
    min_year_signals: int = 30,
    min_year_baseline: int = 200,
) -> dict | None:
    """신호∩조건 을, **조건만 건 기준선**과 비교한 칸 하나.

    `연도승` 은 해마다 신호가 기준선을 이겼는지 센 것이다. 전체 중앙값 하나로는
    한 해가 전부를 만든 경우를 걸러 낼 수 없다.
    """
    both = signal & condition
    n = int(both.sum())
    if n < min_signals:
        return None
    fwd = table[f"fwd{horizon}"].to_numpy()
    years = table["연도"].to_numpy()
    sig_med = float(np.nanmedian(fwd[both]))
    base_med = float(np.nanmedian(fwd[condition]))

    wins = tested = 0
    for year in np.unique(years):
        in_year = years == year
        s, b = both & in_year, condition & in_year
        if s.sum() < min_year_signals or b.sum() < min_year_baseline:
            continue
        tested += 1
        wins += int(np.nanmedian(fwd[s]) > np.nanmedian(fwd[b]))
    return {
        "신호 수": n,
        "신호": sig_med,
        "기준선": base_med,
        "차이": sig_med - base_med,
        "오른비율": float(np.nanmean(fwd[both] > 0)),
        "기준오른비율": float(np.nanmean(fwd[condition] > 0)),
        "연도승": wins,
        "연도수": tested,
    }


def grid(
    table: pd.DataFrame,
    signals: Mapping[str, np.ndarray],
    conditions: Mapping[str, np.ndarray],
    *,
    horizon: int = 20,
    **kwargs,
) -> pd.DataFrame:
    """신호 × 조건 전체 격자. 표본이 모자란 칸은 버린다."""
    rows = []
    for sig_name, sig in signals.items():
        for cond_name, cond in conditions.items():
            got = cell(table, sig, cond, horizon=horizon, **kwargs)
            if got:
                rows.append({"신호": sig_name, "조건": cond_name, **got})
    columns = ["신호", "조건", "신호 수", "신호", "기준선", "차이",
               "오른비율", "기준오른비율", "연도승", "연도수"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("차이", ascending=False)


def consistent(frame: pd.DataFrame, *, allow_losses: int = 1,
               min_years: int = 5) -> pd.DataFrame:
    """해마다 거의 항상 이긴 칸만. 한 해가 전부를 만든 칸을 걸러 낸다."""
    if frame.empty:
        return frame
    return frame[(frame["연도수"] >= min_years)
                 & (frame["연도승"] >= frame["연도수"] - allow_losses)
                 & (frame["차이"] > 0)]
