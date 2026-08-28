"""OHLCV 데이터 계약(contract).

이 프로젝트의 모든 데이터 소스는 아래 규격의 DataFrame 하나로 수렴한다.
지표/신호/검증 코드는 데이터가 토스에서 왔는지 CSV에서 왔는지 알 필요가 없다.

규격
----
- index: tz-aware DatetimeIndex (Asia/Seoul), 오름차순, 중복 없음
- columns: open, high, low, close, volume (float64)
- 결측 봉은 행 자체가 없다 (0으로 채우지 않는다)
"""

from __future__ import annotations

import pandas as pd

KST = "Asia/Seoul"
COLUMNS = ["open", "high", "low", "close", "volume"]


class OhlcvError(ValueError):
    """OHLCV 규격 위반."""


def normalize(df: pd.DataFrame, *, tz: str = KST) -> pd.DataFrame:
    """임의의 OHLCV 유사 DataFrame을 규격에 맞게 정규화한다."""
    out = df.copy()

    lower = {c: str(c).lower() for c in out.columns}
    out = out.rename(columns=lower)

    alias = {
        "date": "dt", "datetime": "dt", "time": "dt", "timestamp": "dt",
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "vol": "volume",
    }
    out = out.rename(columns={k: v for k, v in alias.items() if k in out.columns})

    if not isinstance(out.index, pd.DatetimeIndex):
        if "dt" not in out.columns:
            raise OhlcvError("DatetimeIndex 또는 dt/date/datetime 컬럼이 필요합니다.")
        out = out.set_index("dt")

    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    out.index = idx.tz_localize(tz) if idx.tz is None else idx.tz_convert(tz)
    out.index.name = "dt"

    missing = [c for c in COLUMNS if c not in out.columns]
    if missing:
        raise OhlcvError(f"필수 컬럼 누락: {missing}")

    out = out[COLUMNS].astype("float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """규격 + 값의 정합성을 검사한다. 통과하면 원본을 그대로 돌려준다."""
    if list(df.columns) != COLUMNS:
        raise OhlcvError(f"컬럼이 {COLUMNS} 와 다릅니다: {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise OhlcvError("tz-aware DatetimeIndex 가 아닙니다.")
    if not df.index.is_monotonic_increasing:
        raise OhlcvError("인덱스가 오름차순이 아닙니다.")
    if df.index.has_duplicates:
        raise OhlcvError("중복된 타임스탬프가 있습니다.")

    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    bad = (df["high"] < body_hi - 1e-9) | (df["low"] > body_lo + 1e-9) | (df["high"] < df["low"])
    if bad.any():
        raise OhlcvError(f"고가/저가 정합성 위반 {int(bad.sum())}건 (예: {df.index[bad][0]})")
    if (df["volume"] < 0).any():
        raise OhlcvError("음수 거래량이 있습니다.")
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """분봉을 상위 타임프레임으로 집계한다. rule 예: '5min', '15min', '1D'."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    return validate(out[COLUMNS])


def sessions(df: pd.DataFrame) -> pd.Series:
    """각 봉이 속한 거래일(날짜). 분봉 지표의 세션 경계 처리에 쓴다."""
    return pd.Series(df.index.tz_convert(KST).date, index=df.index, name="session")
