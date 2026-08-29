"""전술적 자산배분 / 추세추종 검증.

왜 여기로 오는가
----------------
"어떤 종목이 오를지 맞히기"는 금융에서 가장 어려운 문제다. 이 프로젝트가
지표·조합·팩터·수급·차트패턴을 전부 훑고도 아무것도 못 찾은 것은 놀랄 일이 아니다.

추세추종은 **훨씬 약한 주장**을 한다. "무엇이 오를지"가 아니라 "지금 위험을
질 때인가"만 판단한다. 그리고 목표도 다르다 — 수익률을 높이는 게 아니라
**하락을 피하는 것**이다. 그래서 실증 증거가 훨씬 강하고 오래 살아남았다.

여기서 재는 규칙들은 전부 발표된 문헌 그대로다. 이 데이터로 튜닝하지 않았다.

  Faber(2007) 10개월 이동평균  월말 종가 > 10개월 SMA 면 주식, 아니면 현금
  200일 이동평균               같은 규칙의 일간판 (가장 널리 쓰이는 형태)
  절대 모멘텀(12개월)          Antonacci 듀얼모멘텀의 절대모멘텀 다리
  변동성 타겟팅                실현변동성이 목표를 넘으면 비중을 줄인다

평가 기준이 다르다
------------------
개별 신호를 잴 때는 초과수익의 t 를 봤다. 자산배분은 **위험 조정 수익**이
목적이므로 CAGR 하나로 판단하면 안 된다. 최대낙폭(MDD)과 샤프를 함께 본다.
상승장에서는 추세추종이 매수후보유보다 수익률이 낮은 것이 정상이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# ETF 는 증권거래세가 없다. 수수료 1.5bp + 슬리피지 5bp, 편도 기준.
ETF_ONE_WAY_BPS = 6.5


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series          # 누적 수익 곡선 (1.0 시작)
    weight: pd.Series          # 일별 주식 비중
    daily: pd.Series           # 일별 수익률 (비용 차감)
    trades: int

    @property
    def years(self) -> float:
        return (self.equity.index[-1] - self.equity.index[0]).days / 365.25

    @property
    def cagr(self) -> float:
        return float(self.equity.iloc[-1] ** (1 / self.years) - 1)

    @property
    def volatility(self) -> float:
        return float(self.daily.std(ddof=1) * np.sqrt(252))

    @property
    def sharpe(self) -> float:
        return float(self.cagr / self.volatility) if self.volatility > 0 else np.nan

    @property
    def max_drawdown(self) -> float:
        return float((self.equity / self.equity.cummax() - 1).min())

    @property
    def calmar(self) -> float:
        mdd = abs(self.max_drawdown)
        return float(self.cagr / mdd) if mdd > 0 else np.nan

    @property
    def exposure(self) -> float:
        """시장에 노출된 시간 비율."""
        return float(self.weight.mean())

    @property
    def worst_year(self) -> float:
        yearly = self.equity.resample("YE").last().pct_change().dropna()
        return float(yearly.min()) if len(yearly) else np.nan

    def summary(self) -> dict[str, float]:
        return {
            "CAGR%": self.cagr * 100,
            "변동성%": self.volatility * 100,
            "샤프": self.sharpe,
            "MDD%": self.max_drawdown * 100,
            "칼마": self.calmar,
            "최악의해%": self.worst_year * 100,
            "노출%": self.exposure * 100,
            "매매횟수": self.trades,
        }


def _run(
    prices: pd.Series,
    weight: pd.Series,
    *,
    name: str,
    cash_rate: float = 0.02,
    one_way_bps: float = ETF_ONE_WAY_BPS,
) -> BacktestResult:
    """비중 계열 → 백테스트 결과.

    비중은 **전일 종가 기준으로 결정된 것**이어야 한다. 호출부에서 shift(1) 을
    끝낸 상태로 넘긴다. 여기서 또 밀지 않는다.
    """
    asset_return = prices.pct_change().fillna(0.0)
    cash_daily = (1 + cash_rate) ** (1 / 252) - 1

    weight = weight.reindex(prices.index).fillna(0.0).clip(0.0, 1.0)
    turnover = weight.diff().abs().fillna(weight.iloc[0])
    cost = turnover * (one_way_bps / 10_000.0)

    daily = weight * asset_return + (1 - weight) * cash_daily - cost
    equity = (1 + daily).cumprod()
    trades = int((turnover > 1e-9).sum())
    return BacktestResult(name, equity, weight, daily, trades)


# --- 규칙들 -------------------------------------------------------------

def buy_and_hold(prices: pd.Series, **kwargs) -> BacktestResult:
    return _run(prices, pd.Series(1.0, index=prices.index), name="매수후보유", **kwargs)


def monthly_sma(prices: pd.Series, months: int = 10, **kwargs) -> BacktestResult:
    """Faber(2007). 월말에만 판단하고 한 달 유지한다.

    월말 판단이라 매매가 드물다 — 비용과 세금 면에서 유리하고,
    일간 잡음에 흔들리지 않는다.
    """
    monthly = prices.resample("ME").last()
    signal = (monthly > monthly.rolling(months, min_periods=months).mean()).astype(float)
    # 월말 종가로 판단 → 다음 달부터 적용
    weight = signal.shift(1).reindex(prices.index, method="ffill")
    return _run(prices, weight, name=f"Faber {months}개월 SMA", **kwargs)


def daily_sma(prices: pd.Series, window: int = 200, **kwargs) -> BacktestResult:
    """일간 이동평균. 가장 널리 알려진 형태."""
    signal = (prices > prices.rolling(window, min_periods=window).mean()).astype(float)
    return _run(prices, signal.shift(1), name=f"{window}일 SMA", **kwargs)


def absolute_momentum(prices: pd.Series, months: int = 12, **kwargs) -> BacktestResult:
    """Antonacci 듀얼모멘텀의 절대모멘텀 다리. 최근 수익률이 양수면 보유."""
    monthly = prices.resample("ME").last()
    signal = (monthly / monthly.shift(months) - 1 > 0).astype(float)
    weight = signal.shift(1).reindex(prices.index, method="ffill")
    return _run(prices, weight, name=f"절대모멘텀 {months}개월", **kwargs)


def volatility_target(
    prices: pd.Series,
    target: float = 0.10,
    window: int = 60,
    cap: float = 1.0,
    band: float = 0.0,
    **kwargs,
) -> BacktestResult:
    """실현변동성이 목표를 넘으면 비중을 줄인다. **방향 예측이 전혀 없다.**

    추세 신호는 가격이 기준선을 뚫을 때까지 기다려야 한다. 강한 상승 뒤의 급락에서는
    기준선이 한참 아래에 있어 대응이 늦다. 변동성은 폭락 첫날부터 튀므로 즉시 줄인다.

    실측(KODEX 200, 2002~2026):
      2008 느린 약세장 — 200일 SMA -18.2%, 변동성타겟 -22.3% (추세가 낫다)
      2026 빠른 폭락   — 200일 SMA -40.8%, 변동성타겟  -7.1% (변동성이 압도)

    band 를 주면 비중 변화가 그 폭보다 작을 때 리밸런싱을 건너뛴다.
    매일 조정하면 매매가 수천 번 발생하므로 현실성을 위해 필요하다.
    """
    realized = prices.pct_change().rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252)
    raw = (target / realized.replace(0, np.nan)).clip(upper=cap)

    if band > 0:
        values = raw.to_numpy(dtype=float)
        held = np.full(len(values), np.nan)
        current = np.nan
        for i, value in enumerate(values):
            if not np.isfinite(value):
                continue
            if not np.isfinite(current) or abs(value - current) >= band:
                current = value
            held[i] = current
        raw = pd.Series(held, index=raw.index)

    label = f"변동성타겟 {target:.0%}" + (f" 밴드{band:.0%}" if band else "")
    if cap != 1.0:
        label += f" 최대{cap:.0%}"
    return _run(prices, raw.shift(1), name=label, **kwargs)


def trend_and_vol(
    prices: pd.Series, target: float = 0.10, window: int = 200, band: float = 0.10, **kwargs
) -> BacktestResult:
    """추세와 변동성을 곱한다. 둘 다 우호적일 때만 온전히 싣는다.

    2008 형 하락은 추세가, 2026 형 급락은 변동성이 막는다는 관찰에서 나온 조합.
    """
    trend = (prices > prices.rolling(window, min_periods=window).mean()).astype(float)
    vol = volatility_target(prices, target=target, band=band, **kwargs).weight
    combined = (trend.shift(1) * vol).fillna(0.0)
    return _run(prices, combined, name=f"추세×변동성 {target:.0%}", **kwargs)


def dual_momentum(
    risky: pd.Series, safe: pd.Series, months: int = 12, **kwargs
) -> BacktestResult:
    """듀얼 모멘텀: 위험자산의 12개월 수익률이 안전자산보다 높으면 위험자산.

    안전자산 수익률을 현금 대신 실제 채권 ETF 로 쓴다.
    """
    index = risky.index.intersection(safe.index)
    risky, safe = risky.reindex(index), safe.reindex(index)
    m_risky, m_safe = risky.resample("ME").last(), safe.resample("ME").last()
    signal = ((m_risky / m_risky.shift(months)) > (m_safe / m_safe.shift(months))).astype(float)
    weight = signal.shift(1).reindex(index, method="ffill").fillna(0.0)

    # 위험자산에서 빠지면 안전자산을 든다 (현금이 아니라)
    asset_return = risky.pct_change().fillna(0.0)
    safe_return = safe.pct_change().fillna(0.0)
    turnover = weight.diff().abs().fillna(weight.iloc[0])
    cost = turnover * (kwargs.get("one_way_bps", ETF_ONE_WAY_BPS) / 10_000.0)
    daily = weight * asset_return + (1 - weight) * safe_return - cost
    equity = (1 + daily).cumprod()
    return BacktestResult(f"듀얼모멘텀 {months}개월", equity, weight, daily,
                          int((turnover > 1e-9).sum()))


STRATEGIES: dict[str, Callable[..., BacktestResult]] = {
    "매수후보유": buy_and_hold,
    "Faber 10개월": monthly_sma,
    "200일 SMA": daily_sma,
    "절대모멘텀 12개월": absolute_momentum,
    "변동성타겟 10%": volatility_target,
    "변동성타겟 10% 밴드10%": lambda p, **kw: volatility_target(p, band=0.10, **kw),
    "추세×변동성": trend_and_vol,
}


def compare(prices: pd.Series, **kwargs) -> pd.DataFrame:
    rows = []
    for name, fn in STRATEGIES.items():
        result = fn(prices, **kwargs)
        rows.append({"전략": name, **result.summary()})
    return pd.DataFrame(rows).set_index("전략")


def by_period(prices: pd.Series, freq: str = "YE", **kwargs) -> pd.DataFrame:
    """연도별 수익률 비교. 어느 해에 이기고 어느 해에 지는지가 성격을 말해준다."""
    out = {}
    for name, fn in STRATEGIES.items():
        equity = fn(prices, **kwargs).equity
        out[name] = equity.resample(freq).last().pct_change().dropna() * 100
    return pd.DataFrame(out).round(1)
