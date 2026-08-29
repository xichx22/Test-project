"""시가총액 순위 기반 매수법 — "상위 N종목을 들고 주기적으로 갈아탄다".

왜 이건 다른가
--------------
지금까지 기각한 것들은 전부 **예측**을 했다. "이 신호가 켜지면 오른다".
시총 상위 N 보유는 예측을 하지 않는다. 시장이 이미 매긴 순위를 그대로
받아들이고, 순위가 바뀌면 따라가기만 한다. 예측이 없으므로 틀릴 자리도
그만큼 적다 — 자산배분이 이 프로젝트에서 유일하게 통과한 것과 같은 성질이다.

과거 시총을 어떻게 복원하는가
-----------------------------
과거 시점의 상장주식수 이력이 없다. 대신 **오늘의 주식수 × 그날의 수정주가**
로 근사한다. 액면분할은 수정주가와 현재 주식수가 서로 상쇄하므로 정확하다.
오차는 유상증자·자사주 소각·합병에서만 생긴다.

읽을 때 반드시 같이 봐야 할 한계
--------------------------------
유니버스가 **오늘 시총 상위**로 뽑혀 있다. 2016년에 시총 20위였다가
지금은 400위 밖으로 밀려난 종목은 데이터에 아예 없다. 그런 종목을
피해간 것처럼 계산되므로 결과가 위로 편향된다. 그리고 이 편향은
하필 **이 전략이 다루는 바로 그 위험**(상위 종목의 몰락)을 지운다.
그래서 여기 나오는 절대 수익률은 믿으면 안 되고, 같은 편향을 공유하는
벤치마크와의 **상대 비교**만 유효하다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .allocation import BacktestResult, _rebalance_marks

# 국내주식 왕복: 수수료 1.5bp×2 + 증권거래세 15bp + 슬리피지 5bp×2
STOCK_ROUND_TRIP_BPS = 28.0


def market_caps(
    closes: pd.DataFrame,
    shares: pd.Series,
) -> pd.DataFrame:
    """수정주가 × 오늘 상장주식수 = 과거 시가총액 근사.

    액면분할은 수정주가와 현재 주식수가 상쇄해 정확하다.
    유상증자·소각·합병만 오차로 남는다.
    """
    common = [c for c in closes.columns if c in shares.index]
    return closes[common] * shares[common]


def top_n_portfolio(
    caps: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    top_n: int = 20,
    rebalance: str = "QE",
    weighting: str = "equal",
    cost_bps: float = STOCK_ROUND_TRIP_BPS,
    risk_free: float = 0.02,
    exclude: tuple[str, ...] = (),
) -> BacktestResult:
    """시총 상위 `top_n` 을 보유하고 주기마다 순위대로 갈아탄다.

    체결 규약은 이 프로젝트 전체와 같다 — 순위는 리밸런싱 봉의 **종가**로
    확정하고, 매매는 **다음 봉 시가**에서 일어난다. 그래서 편입 첫날의
    수익률은 종가/시가로 잰다 (종가/전일종가를 쓰면 사기 전의 갭을 먹는다).

    `weighting`
      equal  동일가중 — 20종목에 5%씩
      cap    시총가중 — 지수와 같은 방식. 삼성전자 비중이 커진다
    """
    codes = [c for c in caps.columns if c not in exclude]
    caps, opens, closes = caps[codes], opens[codes], closes[codes]
    index = caps.index

    ret = closes.pct_change(fill_method=None).fillna(0.0).to_numpy()
    entry_ret = (closes / opens - 1).fillna(0.0).to_numpy()
    listed = closes.notna().to_numpy()
    cap_values = caps.to_numpy()

    marks = _rebalance_marks(index, rebalance)
    cost = cost_bps / 10_000.0

    n_d, n_c = ret.shape
    weights = np.zeros(n_c)
    pending: np.ndarray | None = None       # 다음 봉 시가에 실행할 목표 비중
    daily, trades = [], 0

    for d in range(n_d):
        entered = np.zeros(n_c, dtype=bool)
        if pending is not None:
            # 시가에 갈아탄다: 회전율만큼 비용을 내고 목표 비중으로 바꾼다
            turnover = float(np.abs(pending - weights).sum()) / 2
            entered = (pending > 0) & (weights <= 0)
            weights = pending
            pending = None
            trades += 1
            bar = np.where(entered, entry_ret[d], ret[d])
            value = float((weights * bar).sum()) - turnover * 2 * cost
        else:
            value = float((weights * ret[d]).sum())
        daily.append(value)

        # 비중은 수익률만큼 표류한다
        grown = weights * (1 + np.where(entered, entry_ret[d], ret[d]))
        total = grown.sum()
        if total > 0:
            weights = grown / total

        if index[d] in marks:
            usable = listed[d] & np.isfinite(cap_values[d])
            if usable.sum() >= top_n:
                ranked = np.argsort(np.where(usable, -cap_values[d], np.inf))
                picks = ranked[:top_n]
                target = np.zeros(n_c)
                if weighting == "cap":
                    chosen = cap_values[d][picks]
                    target[picks] = chosen / chosen.sum()
                else:
                    target[picks] = 1.0 / top_n
                pending = target

    series = pd.Series(daily, index=index)
    equity = (1 + series).cumprod()
    weight = pd.Series(1.0, index=index)
    label = f"시총 상위{top_n} {weighting} {rebalance}"
    return BacktestResult(label, equity, weight, series, trades, risk_free=risk_free)


def turnover_report(
    caps: pd.DataFrame,
    *,
    top_n: int = 20,
    rebalance: str = "QE",
) -> pd.DataFrame:
    """상위 N 명단이 얼마나 자주 바뀌는가.

    갈아타기가 잦으면 비용과 세금이 커지고, 전략의 성격도 달라진다.
    "시총 상위를 들고 있는다"가 실제로 몇 번의 매매인지 먼저 알아야 한다.
    """
    marks = sorted(_rebalance_marks(caps.index, rebalance))
    rows, previous = [], None
    for stamp in marks:
        row = caps.loc[stamp].dropna()
        if len(row) < top_n:
            continue
        names = set(row.nlargest(top_n).index)
        if previous is not None:
            rows.append({
                "시점": stamp.date(),
                "신규편입": len(names - previous),
                "탈락": len(previous - names),
                "유지": len(names & previous),
            })
        previous = names
    return pd.DataFrame(rows)


def common_start(closes: pd.DataFrame, *, min_listed: int) -> pd.Timestamp:
    """`min_listed` 종목 이상이 상장돼 있는 첫 날.

    종목마다 상장일이 다르므로 합집합 인덱스의 첫 날에는 몇 종목밖에 없다.
    거기서부터 재면 "상위 20"이 성립하지 않는 구간이 앞에 붙어 결과가
    통째로 왜곡된다. 실측: 합집합 첫날 2015-08-17 에 상장 종목이 **2개**였다.
    """
    listed = closes.notna().sum(axis=1)
    enough = listed[listed >= min_listed]
    if enough.empty:
        raise ValueError(f"{min_listed}종목이 동시에 상장된 날이 없습니다")
    return enough.index[0]


def survivor_note(caps: pd.DataFrame, *, top_n: int = 20) -> dict:
    """생존편향의 크기를 가늠할 수 있는 만큼만 재서 돌려준다.

    유니버스에서 사라진 종목은 잴 수 없다 — 데이터에 없기 때문이다.
    대신 **첫 시점 상위 N 중 몇 개가 마지막 시점에도 상위 N 인가**를 센다.
    이 값이 낮을수록 순위 교체가 활발했다는 뜻이고, 유니버스 밖으로
    완전히 사라진 종목까지 있었다면 실제 편향은 이보다 크다.
    """
    usable = caps.dropna(axis=0, thresh=top_n)
    if usable.empty:
        raise ValueError(f"상위 {top_n} 을 뽑을 수 있는 날이 없습니다")
    first = set(usable.iloc[0].dropna().nlargest(top_n).index)
    last = set(usable.iloc[-1].dropna().nlargest(top_n).index)
    return {
        "첫 시점 상위N": top_n,
        "끝까지 상위N 유지": len(first & last),
        "교체율": 1 - len(first & last) / top_n,
    }
