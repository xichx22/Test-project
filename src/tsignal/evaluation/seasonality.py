"""계절성 — 달력 위치에 따른 수익률 차이.

지금까지 시험한 것들과 성격이 또 다르다. 가격·거래량에서 신호를 뽑는 게 아니라
**달력만 본다.** 예측이 전혀 없다 — 오늘이 며칠인지만 알면 된다.

문헌에서 반복 보고된 것들
  월말·월초 효과(turn-of-the-month) : 월말 마지막 며칠과 월초 며칠에 수익이 몰린다
                                     (Ariel 1987, Lakonishok & Smidt 1988)
  요일 효과                          : 월요일이 나쁘고 금요일이 좋다는 보고
  월별 효과                          : 1월 효과, "Sell in May"

계절성이 매력적인 이유는 **시장 노출 시간이 짧다**는 것이다. 월말·월초 4~5일만
들고 있으면 한 달의 20% 만 위험에 노출된다. 그 20% 동안 한 달 수익의 대부분이
나온다면, 나머지 80% 는 현금으로 두면서 같은 수익을 얻는 셈이다.

주의: 계절성은 과최적화가 특히 쉽다. 달력 위치 조합이 수십 가지라
아무거나 골라 보면 하나는 좋아 보인다. 그래서 여기서도
**문헌에 나온 정의를 그대로** 쓰고, 시험한 가설 수만큼 문턱을 올린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def calendar_frame(prices: pd.Series) -> pd.DataFrame:
    """가격 계열에 달력 위치를 붙인다.

    `tom_position`: 월말 기준 상대 위치. 그 달 마지막 거래일이 -1,
    그 전날이 -2 … 다음 달 첫 거래일이 +1, 둘째 날이 +2 …
    이렇게 잡아야 월말·월초가 연속된 축 위에 놓인다.
    """
    frame = pd.DataFrame({"close": prices})
    frame["ret"] = prices.pct_change()
    frame["month"] = frame.index.month
    frame["weekday"] = frame.index.weekday          # 0=월 … 4=금

    period = frame.index.to_period("M")
    order_in_month = frame.groupby(period).cumcount()
    size = frame.groupby(period)["close"].transform("size")

    # 월초에서 센 위치(+1부터)와 월말에서 센 위치(-1부터)를 하나의 축으로 합친다.
    from_start = order_in_month + 1
    from_end = order_in_month - size                # 마지막 날이 -1
    frame["day_from_start"] = from_start
    frame["day_from_end"] = from_end
    frame["tom_position"] = np.where(from_start <= 10, from_start, from_end)
    return frame.dropna(subset=["ret"])


def turn_of_month_profile(prices: pd.Series, span: int = 8) -> pd.DataFrame:
    """월말 -span일 ~ 월초 +span일 구간의 일평균 수익률."""
    frame = calendar_frame(prices)
    rows = []
    for position in list(range(-span, 0)) + list(range(1, span + 1)):
        subset = frame[frame["tom_position"] == position]["ret"]
        if len(subset) < 20:
            continue
        rows.append({
            "위치": position,
            "관측": len(subset),
            "일평균%": subset.mean() * 100,
            "승률": float((subset > 0).mean()),
            "t": metrics.t_stat(subset),
        })
    return pd.DataFrame(rows).set_index("위치")


def group_profile(prices: pd.Series, column: str) -> pd.DataFrame:
    """월별/요일별 일평균 수익률."""
    frame = calendar_frame(prices)
    rows = []
    for key, group in frame.groupby(column):
        rows.append({
            column: key,
            "관측": len(group),
            "일평균%": group["ret"].mean() * 100,
            "승률": float((group["ret"] > 0).mean()),
            "t": metrics.t_stat(group["ret"]),
        })
    return pd.DataFrame(rows).set_index(column)


def seasonal_strategy(
    prices: pd.Series,
    *,
    enter_before_end: int = 4,
    exit_after_start: int = 3,
    cash_rate: float = 0.02,
    one_way_bps: float = 6.5,
):
    """월말 enter_before_end일 전부터 월초 exit_after_start일까지만 보유.

    `allocation.BacktestResult` 를 그대로 돌려주므로 다른 전략들과 같은 잣대로
    비교할 수 있다 (CAGR·샤프·MDD·노출).
    """
    from .allocation import _run

    frame = calendar_frame(prices)
    position = frame["tom_position"]
    in_window = ((position <= -1) & (position >= -enter_before_end)) | (
        (position >= 1) & (position <= exit_after_start)
    )
    # 그날 수익을 얻으려면 전날 종가에 이미 들고 있어야 한다.
    weight = in_window.astype(float).reindex(prices.index).fillna(0.0)
    name = f"월말{enter_before_end}일~월초{exit_after_start}일"
    return _run(prices, weight, name=name, cash_rate=cash_rate, one_way_bps=one_way_bps)
