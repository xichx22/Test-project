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

import numpy as np
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


# KRX 주식 호가 단위 (2023년 개편 기준). 반올림 오차의 자연스러운 크기다.
TICK_BANDS = (
    (2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
    (200_000, 100), (500_000, 500), (float("inf"), 1_000),
)


def tick_size(price: pd.Series) -> pd.Series:
    """가격대별 호가 단위. 수정주가라 실제 틱과 정확히 맞지는 않지만,
    "반올림 오차가 가격에 비례하지 않는다"는 성질을 담기에는 충분하다."""
    # 결측 가격에는 틱이 없다. 기본값 1원을 남기면 "1원짜리 틱"이 섞여
    # 통계를 왜곡한다 (실측: 36,500원 종목의 중앙 틱이 1원으로 보고됐다).
    out = pd.Series(np.nan, index=price.index, dtype="float64")
    lower = 0.0
    for upper, tick in TICK_BANDS:
        out = out.mask((price >= lower) & (price < upper), float(tick))
        lower = upper
    return out


def repair(
    df: pd.DataFrame,
    *,
    tolerance: float = 0.005,
    max_ticks: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """수정주가 반올림으로 생긴 미세한 정합성 위반을 보정한다.

    국내 시세 제공자는 액면분할·무상증자 등을 소급 반영한 수정주가를 주는데,
    개별 가격을 따로 반올림하는 탓에 `종가 = 고가 + 1원` 같은 행이 나온다.
    (예: 삼성SDI 2022-11-01 고가 744,064 / 종가 744,065)

    실제 관측 오차가 아니라 계산 아티팩트이므로, 오차가 tolerance(기본 0.5%)
    이내면 고가/저가를 시가·종가를 포함하도록 넓혀 보정한다.
    그보다 큰 위반은 손대지 않는다 — 그건 진짜 잘못된 데이터이고,
    validate() 가 잡아야 한다.

    허용치가 0.5% 인 이유: 기간이 길수록 오래된 저가 구간의 반올림 오차
    비율이 커진다. 실측에서 대한전선 0.28%, 에코프로 0.12% 까지 나왔다.
    반면 진짜 깨진 데이터(시가/고가/저가가 0 인 거래정지 행)는 오차가 100% 라
    이 문턱으로 충분히 갈린다.

    비율만으로는 부족하다 — `absolute` 를 함께 두는 이유
    ----------------------------------------------------
    반올림 오차는 **가격에 비례하지 않는다**. 언제나 호가 한 틱 남짓이다.
    그래서 주가가 낮을수록 같은 1원이 큰 비율이 된다. 20년치를 받으면
    2004년 구간의 수정주가가 세 자릿수까지 내려가고, 1원 차이가 0.6% 가 된다.
    실측(파미셀 2004년): 최대 위반 1.183%, 249봉 중 37봉. 비율 문턱만 쓰면
    이런 종목이 통째로 버려지는데, 하필 **가장 오래된 종목들만** 골라서
    빠진다. 표본에서 옛날 구간이 계통적으로 사라지는 것이다.

    그래서 "비율이 작거나, **호가 몇 틱 이내면**" 보정한다. 틱으로 재는 이유는
    반올림 오차의 자연스러운 단위가 틱이기 때문이다. 실측에서 남은 위반은
    전부 3~6원이었고 (유진테크 545원에 3원, 코맥스 714원에 5원) 그 가격대의
    호가 단위는 1원이다. 즉 몇 틱짜리 오차다.

    다만 왜 1틱이 아니라 3~6틱인지는 설명하지 못했다. 수정 계수를 필드마다
    따로 적용한 흔적으로 보이지만 확인할 방법이 없다. 그래서 이 문턱은
    **경험적**이고, 그만큼 결론이 여기에 의존하면 안 된다 —
    보정 대상 종목을 넣고 뺀 두 결과를 같이 봐야 한다.

    거래정지 행(가격 0)은 오차가 가격 전체 크기라 여전히 걸러진다.

    반환 (보정된 df, 보정 내역)
    """
    out = df.copy()
    body_hi = out[["open", "close"]].max(axis=1)
    body_lo = out[["open", "close"]].min(axis=1)

    over = (body_hi - out["high"]).clip(lower=0)
    under = (out["low"] - body_lo).clip(lower=0)
    scale = out["close"].abs().replace(0, np.nan)
    small_ratio = ((over / scale) <= tolerance) & ((under / scale) <= tolerance)
    allowance = tick_size(scale) * max_ticks
    small_tick = (over <= allowance) & (under <= allowance)
    fixable = small_ratio | small_tick
    touched = fixable & ((over > 0) | (under > 0))

    log = pd.DataFrame({
        "high_before": out.loc[touched, "high"],
        "low_before": out.loc[touched, "low"],
        "high_gap": over[touched],
        "low_gap": under[touched],
    })

    out.loc[touched, "high"] = body_hi[touched]
    out.loc[touched, "low"] = body_lo[touched]
    return out, log


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """분봉을 상위 타임프레임으로 집계한다. rule 예: '5min', '15min', '1D'."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    return validate(out[COLUMNS])


def sessions(df: pd.DataFrame) -> pd.Series:
    """각 봉이 속한 거래일(날짜). 분봉 지표의 세션 경계 처리에 쓴다."""
    return pd.Series(df.index.tz_convert(KST).date, index=df.index, name="session")
