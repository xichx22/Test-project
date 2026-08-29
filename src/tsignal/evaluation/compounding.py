"""복리로 무엇이 남는가 — 매매 규칙을 최종 자산으로 줄 세운다.

왜 연수익이 아니라 복리인가
---------------------------
연평균수익 12% 인 A 와 10% 인 B 가 있으면 A 를 고르는 게 맞아 보인다.
그런데 A 가 중간에 −50% 를 맞았다면, 그 구멍을 메우는 데 +100% 가 필요하다.
복리는 낙폭을 제곱으로 벌준다. 그래서 "연수익이 가장 큰 규칙"과
"돈이 가장 많이 불어난 규칙"은 자주 다르다.

여기서 재는 규칙은 전부 **종목을 고르지 않는다**. 자산군 ETF 몇 개를 놓고
언제 무엇을 얼마나 들지만 정한다. 종목 선택이 빠지면 예측해야 할 것이
크게 줄고, 그만큼 틀릴 여지도 줄어든다.

적립식을 따로 재는 이유
-----------------------
목돈을 한 번 넣고 두는 것과 매달 넣는 것은 **다른 전략이다**. 매달 넣으면
하락 구간에서 싸게 사 모으므로 낙폭의 의미가 달라진다. 실제로 월급에서
떼어 넣는 사람에게는 거치식 백테스트가 답하는 질문이 아니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .allocation import (
    ETF_ONE_WAY_BPS,
    BacktestResult,
    _rebalance_marks,
    static_mix,
)


def _result(name: str, daily: pd.Series, trades: int,
            risk_free: float = 0.02) -> BacktestResult:
    equity = (1 + daily).cumprod()
    weight = pd.Series(1.0, index=daily.index)
    return BacktestResult(name, equity, weight, daily, trades, risk_free=risk_free)


def momentum_rotation(
    assets: dict[str, pd.Series],
    *,
    top_n: int = 3,
    lookback: int = 252,
    rebalance: str = "ME",
    one_way_bps: float = ETF_ONE_WAY_BPS,
    cash_rate: float = 0.02,
    absolute: bool = True,
) -> BacktestResult:
    """최근 `lookback`일 수익률 상위 `top_n` 자산만 동일가중으로 보유.

    종목을 고르는 게 아니라 **자산군을 고른다**. 후보가 6개뿐이라
    과최적화 여지가 개별 종목 선택보다 훨씬 작다.

    `absolute=True` 면 최근 수익률이 음수인 자산은 상위권이어도 사지 않고
    현금으로 둔다 (Antonacci 듀얼모멘텀의 절대모멘텀 다리).
    """
    frame = pd.DataFrame(assets).dropna()
    returns = frame.pct_change().fillna(0.0)
    momentum = frame / frame.shift(lookback) - 1
    marks = _rebalance_marks(frame.index, rebalance)
    cash_daily = (1 + cash_rate) ** (1 / 252) - 1
    cost = one_way_bps / 10_000.0

    codes = list(frame.columns)
    weights = pd.Series(0.0, index=codes)
    daily, trades = [], 0
    for timestamp, row in returns.iterrows():
        gross = float((weights * row).sum())
        cash_weight = 1.0 - float(weights.sum())
        value = gross + cash_weight * cash_daily
        if timestamp in marks:
            scores = momentum.loc[timestamp]
            if scores.notna().sum() >= top_n:
                picks = scores.nlargest(top_n)
                if absolute:
                    picks = picks[picks > 0]
                target = pd.Series(0.0, index=codes)
                if len(picks):
                    target[picks.index] = 1.0 / len(picks)
                turnover = float((target - weights).abs().sum()) / 2
                value -= turnover * 2 * cost
                trades += int(turnover > 1e-9)
                weights = target
        daily.append(value)

    label = f"모멘텀 상위{top_n} 로테이션 ({lookback}일)"
    return _result(label, pd.Series(daily, index=returns.index), trades)


def trend_filtered_mix(
    assets: dict[str, pd.Series],
    *,
    window: int = 200,
    rebalance: str = "ME",
    one_way_bps: float = ETF_ONE_WAY_BPS,
    cash_rate: float = 0.02,
) -> BacktestResult:
    """자산마다 자기 이동평균 위에 있을 때만 보유, 아니면 그 몫은 현금.

    자산배분과 추세추종을 합친 형태다. 예측을 하지 않는다는 점은 같고,
    "지금 이 자산에 위험을 질 때인가"만 자산별로 따로 묻는다.
    """
    frame = pd.DataFrame(assets).dropna()
    returns = frame.pct_change().fillna(0.0)
    above = frame > frame.rolling(window).mean()
    marks = _rebalance_marks(frame.index, rebalance)
    share = 1.0 / frame.shape[1]
    cash_daily = (1 + cash_rate) ** (1 / 252) - 1
    cost = one_way_bps / 10_000.0

    weights = pd.Series(0.0, index=frame.columns)
    daily, trades = [], 0
    for timestamp, row in returns.iterrows():
        value = float((weights * row).sum()) + (1.0 - float(weights.sum())) * cash_daily
        if timestamp in marks:
            target = above.loc[timestamp].fillna(False).astype(float) * share
            turnover = float((target - weights).abs().sum()) / 2
            value -= turnover * 2 * cost
            trades += int(turnover > 1e-9)
            weights = target
        daily.append(value)

    label = f"자산별 {window}일 추세필터 + 동일가중"
    return _result(label, pd.Series(daily, index=returns.index), trades)


def band_rebalance(
    assets: dict[str, pd.Series],
    weights: dict[str, float],
    *,
    band: float = 0.05,
    one_way_bps: float = ETF_ONE_WAY_BPS,
) -> BacktestResult:
    """비중이 목표에서 `band` 이상 벌어졌을 때만 되돌린다.

    달력이 아니라 **상태**로 매매 시점을 정한다. 매매 횟수가 줄어 비용과
    (일반계좌라면) 세금이 준다. 분기 리밸런싱과 비교해서 실익이 있는지 본다.
    """
    frame = pd.DataFrame(assets).dropna()
    returns = frame.pct_change().fillna(0.0)
    codes = list(frame.columns)
    target = np.array([weights[c] for c in codes], dtype=float)
    target = target / target.sum()
    cost = one_way_bps / 10_000.0

    holding = target.copy()
    daily, trades = [], 0
    for _, row in returns.iterrows():
        grown = holding * (1 + row.to_numpy())
        total = grown.sum()
        value = total - 1
        holding = grown / total if total > 0 else target.copy()
        if np.abs(holding - target).max() >= band:
            turnover = float(np.abs(target - holding).sum()) / 2
            value -= turnover * 2 * cost
            holding = target.copy()
            trades += 1
        daily.append(value)

    label = f"{band:.0%} 밴드 리밸런싱"
    return _result(label, pd.Series(daily, index=returns.index), trades)


def dca(
    result: BacktestResult,
    *,
    monthly: float = 500_000,
    initial: float = 0.0,
) -> dict[str, float]:
    """적립식 결과 — 매달 같은 금액을 넣었을 때 최종 자산과 원금.

    거치식 CAGR 로는 답할 수 없는 질문이다. 매달 넣으면 하락장에서 싸게
    사 모으므로, 같은 전략이라도 "언제 하락이 왔는가"에 결과가 달라진다.

    반환하는 `수익배수` 는 최종자산÷원금이다. 이건 CAGR 이 아니다 —
    나중에 넣은 돈은 굴러간 기간이 짧으므로, 같은 CAGR 이어도 배수는 낮다.
    `연환산수익률` 은 현금흐름을 맞추는 내부수익률(IRR)로 따로 낸다.
    """
    equity = result.equity
    months = equity.resample("ME").last().dropna()
    if len(months) < 12:
        raise ValueError("적립식을 재려면 최소 12개월이 필요합니다")

    units = initial / float(equity.iloc[0])
    flows, times = [], []
    if initial:
        flows.append(-initial)
        times.append(equity.index[0])
    for timestamp, level in months.items():
        units += monthly / float(level)
        flows.append(-monthly)
        times.append(timestamp)

    final = units * float(equity.iloc[-1])
    principal = initial + monthly * len(months)
    flows.append(final)
    times.append(equity.index[-1])

    years = np.array([(t - times[0]).days / 365.25 for t in times])
    values = np.array(flows, dtype=float)

    def npv(rate: float) -> float:
        return float((values / (1 + rate) ** years).sum())

    low, high = -0.95, 5.0
    for _ in range(200):                    # 이분법 — 항상 수렴한다
        mid = (low + high) / 2
        if npv(mid) > 0:
            low = mid
        else:
            high = mid
    irr = (low + high) / 2

    return {
        "원금": principal,
        "최종자산": final,
        "수익배수": final / principal,
        "연환산수익률": irr,
        "납입월수": len(months),
    }


def compare_rules(
    assets: dict[str, pd.Series],
    *,
    rebalance: str = "QE",
    monthly: float = 500_000,
) -> pd.DataFrame:
    """종목을 고르지 않는 매매 규칙들을 복리 잣대로 한 표에 놓는다."""
    weights = {code: 1.0 for code in assets}
    rules = [
        static_mix(assets, weights, rebalance=rebalance, name="동일가중 분기리밸"),
        static_mix(assets, weights, rebalance="ME", name="동일가중 월리밸"),
        static_mix(assets, weights, rebalance=None, name="동일가중 무리밸(방치)"),
        band_rebalance(assets, weights, band=0.05),
        momentum_rotation(assets, top_n=3, lookback=252),
        momentum_rotation(assets, top_n=2, lookback=252),
        momentum_rotation(assets, top_n=3, lookback=126),
        trend_filtered_mix(assets, window=200),
    ]
    rows = []
    for rule in rules:
        plan = dca(rule, monthly=monthly)
        rows.append({
            "규칙": rule.name,
            "연수익": rule.cagr,
            "MDD": rule.max_drawdown,
            "궤양": rule.ulcer_index,
            "양수율12M": rule.rolling_positive(12),
            "최악12M": rule.worst_rolling(12),
            "거치 1000만원": 10_000_000 * float(rule.equity.iloc[-1]),
            "적립 최종자산": plan["최종자산"],
            "적립 IRR": plan["연환산수익률"],
            "매매횟수": rule.trades,
        })
    return pd.DataFrame(rows).sort_values("적립 최종자산", ascending=False)


def sign_test(wins: int, losses: int) -> float:
    """동점을 제외한 양측 부호검정 (Wilcoxon 표준).

    동점을 패로 세면 p 가 가짜로 커진다. 5승0패2무(p=0.062)와
    5승2패(p=0.453)는 전혀 다른 증거다.
    """
    from math import comb

    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def subperiod_test(
    assets: dict[str, pd.Series],
    *,
    periods: int = 5,
    baseline: str = "분기리밸",
    monthly: float = 500_000,
) -> pd.DataFrame:
    """전 구간 1등이 구간마다도 1등인가.

    전 구간 성과 하나로 규칙을 고르면 안 된다. 한 구간에서 크게 번 것이
    전체 평균을 끌어올려 1등처럼 보이는 일이 흔하다. 구간을 쪼개서 기준
    규칙과 매번 비교하고, 승패를 부호검정에 넣는다.

    구간이 5개면 부호검정의 최소 p 는 0.0625 다 — 즉 **전승해도 0.05 를
    넘길 수 없다**. 그러니 여기서 "유의하다"는 말은 쓰지 않는다.
    방향과 일관성만 본다.
    """
    frame = pd.DataFrame(assets).dropna()
    weights = {code: 1.0 for code in frame.columns}

    def build(sub: dict[str, pd.Series]) -> dict[str, BacktestResult]:
        w = {code: 1.0 for code in sub}
        return {
            "분기리밸": static_mix(sub, w, rebalance="QE"),
            "5%밴드": band_rebalance(sub, w, band=0.05),
            "모멘텀상위2": momentum_rotation(sub, top_n=2, lookback=252),
            "모멘텀상위3": momentum_rotation(sub, top_n=3, lookback=252),
            "추세필터": trend_filtered_mix(sub, window=200),
        }

    segments = np.array_split(np.arange(len(frame)), periods)
    log: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        window = {c: frame[c].iloc[segment[0]: segment[-1] + 1] for c in frame}
        for name, result in build(window).items():
            log.setdefault(name, []).append(
                (result.cagr, result.ulcer_index))

    base_cagr = [x[0] for x in log[baseline]]
    base_ulcer = [x[1] for x in log[baseline]]
    rows = []
    for name, series in log.items():
        if name == baseline:
            continue
        cagr = [x[0] for x in series]
        ulcer = [x[1] for x in series]
        win_c = sum(a > b for a, b in zip(cagr, base_cagr))
        lose_c = sum(a < b for a, b in zip(cagr, base_cagr))
        win_u = sum(a < b for a, b in zip(ulcer, base_ulcer))   # 낮을수록 좋다
        lose_u = sum(a > b for a, b in zip(ulcer, base_ulcer))
        rows.append({
            "규칙": name,
            "수익 승패": f"{win_c}승{lose_c}패",
            "수익 p": sign_test(win_c, lose_c),
            "궤양 승패": f"{win_u}승{lose_u}패",
            "궤양 p": sign_test(win_u, lose_u),
        })
    out = pd.DataFrame(rows)
    out.attrs["periods"] = periods
    out.attrs["min_p"] = sign_test(periods, 0)
    out.attrs["segments"] = [
        (frame.index[s[0]].date(), frame.index[s[-1]].date()) for s in segments
    ]
    return out
