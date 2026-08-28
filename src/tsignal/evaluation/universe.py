"""여러 종목을 묶어서 검증한다.

단일 종목 결과는 근거가 못 된다. 종목 하나에서 t=3 이 나와도 그 종목의
그 기간 성질을 외운 것일 수 있다. 같은 신호를 N개 종목에 던져
  - 풀링(pooled) 기대값이 문턱을 넘는가
  - 종목별 부호가 몇 개나 일치하는가 (breadth)
를 함께 봐야 "이 신호로 진입할 근거가 있다"고 말할 수 있다.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .. import indicators as ind
from .. import signals as sig
from ..datasource.base import Interval
from . import metrics, validation
from .forward import forward_returns


def signal_returns(
    candles_by_code: Mapping[str, pd.DataFrame],
    *,
    interval: Interval,
    horizon: int = 5,
    entry: str = "next_open",
    exclude_tags: tuple[str, ...] = (),
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """종목별로 (신호 → 수익률 표본)을 만든다.

    반환 (per_signal, unconditional)
      per_signal[신호명] = DataFrame[code, ret]
      unconditional      = 종목별 무조건부 평균 (엣지 계산의 기준선)
    """
    per_signal: dict[str, list[pd.DataFrame]] = {}
    base_means: dict[str, float] = {}

    for code, candles in candles_by_code.items():
        features = ind.compute_all(candles, interval=interval)
        entries = sig.evaluate_all(candles, features, kind="entry", exclude_tags=exclude_tags)
        fwd = forward_returns(candles, (horizon,), entry=entry)[f"fwd_{horizon}"]
        base_means[code] = float(fwd.mean())

        for name in entries.columns:
            r = fwd[entries[name].astype(bool)].dropna()
            if r.empty:
                continue
            per_signal.setdefault(name, []).append(
                # excess = 그 종목의 같은 기간 무조건부 평균을 뺀 값.
                # 신호를 검정할 때는 이쪽을 쓴다 — 아래 주석 참고.
                pd.DataFrame({
                    "code": code, "ret": r.to_numpy(),
                    "excess": r.to_numpy() - base_means[code],
                    # 날짜 군집 보정을 위해 발생 날짜를 함께 들고 간다.
                    "day": r.index.tz_localize(None).to_numpy().astype("datetime64[D]"),
                })
            )

    pooled = {name: pd.concat(parts, ignore_index=True) for name, parts in per_signal.items()}
    return pooled, pd.Series(base_means, name="unconditional")


def screen_universe(
    candles_by_code: Mapping[str, pd.DataFrame],
    *,
    interval: Interval,
    horizon: int = 5,
    entry: str = "next_open",
    exclude_tags: tuple[str, ...] = (),
    min_events: int = 100,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """유니버스 전체에 대한 신호 스크리닝.

    컬럼
      n_codes      : 신호가 한 번이라도 발생한 종목 수
      n            : 풀링 표본 수
      expectancy   : 풀링 기대수익률 (원시)
      edge         : 무조건부(종목별 평균) 대비 초과분 = 검정 대상
      t_raw        : 원시 수익률의 t — **판정에 쓰지 않는다**
      t_naive      : 초과수익의 t, 군집 보정 없음 — **판정에 쓰지 않는다**
      t_edge       : 초과수익의 **날짜 군집 보정** t — 판정 기준
      n_days       : 신호가 발생한 날짜 수 = 유효 표본 수
      breadth      : 종목별 초과수익 평균이 양(+)인 비율
      t_by_code_med: 종목별 초과수익 t 의 중앙값

    왜 t_raw 로 판정하지 않는가
    --------------------------
    보유기간을 늘리면 어떤 신호든 t_raw 가 커진다. 시장이 우상향하면
    "아무 때나 사서 10일 들고 있기"의 기대값이 양수이기 때문이다.
    실제로 이 유니버스(24종목·5년)에서 horizon 10일로 t_raw 를 쓰면
    edge 가 음수인 신호까지 '채택후보'로 올라온다.
    신호가 기여한 몫만 보려면 종목별 무조건부 평균을 뺀 초과수익을 검정해야 한다.
    """
    pooled, base = signal_returns(
        candles_by_code, interval=interval, horizon=horizon, entry=entry, exclude_tags=exclude_tags
    )
    baseline = float(base.mean())

    rows = []
    for name, frame in pooled.items():
        by_code = frame.groupby("code")["excess"]
        code_means = by_code.mean()
        code_t = by_code.apply(lambda r: metrics.t_stat(pd.Series(r.to_numpy())))
        rows.append({
            "signal": name,
            "n_codes": int(frame["code"].nunique()),
            "n": len(frame),
            "expectancy": float(frame["ret"].mean()),
            "edge": float(frame["excess"].mean()),
            "win_rate": metrics.win_rate(frame["ret"]),
            # t_raw 는 보정 없는 값 — 얼마나 부풀려지는지 보여주기 위해서만 남긴다.
            "t_raw": metrics.t_stat(frame["ret"]),
            "t_naive": metrics.t_stat(frame["excess"]),
            # 판정은 날짜 군집 보정된 t 로만 한다.
            "t_edge": metrics.clustered_t_stat(frame["excess"].to_numpy(), frame["day"].to_numpy()),
            "n_days": int(frame["day"].nunique()),
            "breadth": float((code_means > 0).mean()),
            # 표본이 3개 미만인 종목은 t 가 정의되지 않는다 → 빼고 중앙값을 낸다.
            "t_by_code_med": float(code_t.dropna().median()) if code_t.notna().any() else np.nan,
        })

    out = pd.DataFrame(rows).set_index("signal")
    threshold = validation.deflated_threshold(max(1, len(out)), alpha)
    out["verdict"] = [
        _verdict(row, threshold=threshold, min_events=min_events) for _, row in out.iterrows()
    ]
    out.attrs["baseline"] = baseline
    out.attrs["threshold"] = threshold
    out.attrs["n_codes"] = len(candles_by_code)
    return out.sort_values("t_edge", ascending=False)


def _verdict(row: pd.Series, *, threshold: float, min_events: int) -> str:
    if row["n"] < min_events or row.get("n_days", np.inf) < 40:
        return "표본부족"
    t = row["t_edge"]
    if not np.isfinite(t):
        return "판정불가"
    if t >= threshold and row["breadth"] >= 0.6:
        return "채택후보"
    if t >= threshold:
        return "쏠림주의"          # 통계는 유의하나 소수 종목에 몰림
    if t > 1.0:
        return "보류"
    return "기각"


def split_universe(
    candles_by_code: Mapping[str, pd.DataFrame],
    *,
    interval: Interval,
    horizon: int = 5,
    train_ratio: float = 0.6,
    exclude_tags: tuple[str, ...] = (),
) -> pd.DataFrame:
    """유니버스 전체에 대한 IS/OOS 분할 검증. 종목마다 같은 비율로 자른다."""
    is_parts: dict[str, pd.DataFrame] = {}
    oos_parts: dict[str, pd.DataFrame] = {}
    for code, candles in candles_by_code.items():
        cut = int(len(candles) * train_ratio)
        is_parts[code] = candles.iloc[:cut]
        oos_parts[code] = candles.iloc[cut:]

    kwargs = dict(interval=interval, horizon=horizon, exclude_tags=exclude_tags, min_events=1)
    cols = ["n", "n_days", "edge", "t_edge", "breadth"]
    left = screen_universe(is_parts, **kwargs)[cols]
    right = screen_universe(oos_parts, **kwargs)[cols]
    merged = left.add_suffix("_is").join(right.add_suffix("_oos"), how="outer")
    merged["sign_agree"] = np.sign(merged["edge_is"]) == np.sign(merged["edge_oos"])
    return merged.sort_values("t_edge_oos", ascending=False)
