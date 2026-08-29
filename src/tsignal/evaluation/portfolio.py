"""실행 계층 — 어떤 ETF 로, 어떤 계좌에서 굴리는가.

왜 별도 모듈인가
----------------
`allocation.py` 는 "자산군을 어떤 비중으로 섞을 것인가"를 잰다. 거기서
6자산 동일가중이 꾸준함 잣대를 통과했다. 하지만 그 결론은 **자산군 수준**의
결론이고, 실제로 사는 것은 자산군이 아니라 종목 코드가 붙은 ETF 다.

자산군에서 ETF 로 내려오는 순간 새 위험이 셋 붙는다.

  유동성   같은 자산군을 추종해도 거래대금이 100배 차이 난다. 거래대금이
           작은 ETF 는 호가 스프레드가 벌어져서 백테스트가 가정한 왕복
           13bp 가 실제로는 몇 배가 된다.
  환노출   같은 "금"이라도 환헤지판과 환노출판은 다른 자산이다. 원/달러가
           포트폴리오 안에서 어떤 역할을 하느냐에 따라 결론이 바뀐다.
  세금     연금계좌는 운용 중 과세가 없고 일반계좌는 리밸런싱마다 실현이익에
           15.4% 가 붙는다. 그 차이가 얼마인지는 재봐야 안다.

이 모듈은 그 셋을 각각 수치로 잰다. 셋 다 백테스트 밖의 문제라
`BacktestResult` 만 봐서는 절대 보이지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .allocation import ETF_ONE_WAY_BPS, BacktestResult, _rebalance_marks

# 국내주식형 ETF 는 매매차익 비과세(분배금만 15.4%). 그 외(해외지수·채권·
# 원자재·통화)는 매매차익에 배당소득세 15.4% 가 붙는다.
KOREA_TAX_RATE = 0.0
OVERSEAS_TAX_RATE = 0.154


@dataclass(frozen=True)
class EtfSpec:
    """ETF 한 종목의 실행 관점 속성."""

    code: str
    name: str
    asset: str                  # 자산군 이름 (같은 자산군끼리 후보)
    taxable: bool = True        # False = 국내주식형(매매차익 비과세)
    fx_hedged: bool | None = None   # None = 해당 없음(원화 자산)
    note: str = ""


# 자산군별 후보. 같은 자산군 안의 종목은 서로 대체재이고,
# 어느 쪽을 고르느냐가 유동성·환노출·세금에서 갈린다.
ETF_CATALOG: tuple[EtfSpec, ...] = (
    EtfSpec("069500", "KODEX 200", "한국주식", taxable=False),
    EtfSpec("102110", "TIGER 200", "한국주식", taxable=False),
    EtfSpec("278530", "KODEX 200TR", "한국주식", taxable=False,
            note="분배금을 재투자하는 TR 형. 일반계좌에서 분배금 과세를 늦춘다"),
    EtfSpec("152100", "PLUS 200", "한국주식", taxable=False),
    EtfSpec("069660", "KIWOOM 200", "한국주식", taxable=False),
    EtfSpec("148070", "KIWOOM 국고채10년", "국고채10년"),
    EtfSpec("152380", "KODEX 국채선물10년", "국고채10년"),
    EtfSpec("114260", "KODEX 국고채3년", "국고채3년"),
    EtfSpec("114100", "RISE 국고채3년", "국고채3년"),
    EtfSpec("133690", "TIGER 미국나스닥100", "미국주식", fx_hedged=False),
    EtfSpec("379810", "KODEX 미국나스닥100", "미국주식", fx_hedged=False),
    EtfSpec("360750", "TIGER 미국S&P500", "미국주식", fx_hedged=False),
    EtfSpec("132030", "KODEX 골드선물(H)", "금", fx_hedged=True),
    EtfSpec("411060", "ACE KRX금현물", "금", fx_hedged=False),
    EtfSpec("261240", "KODEX 미국달러선물", "달러"),
    EtfSpec("138230", "KIWOOM 미국달러선물", "달러"),
    EtfSpec("305080", "TIGER 미국채10년선물", "미국채10년", fx_hedged=False),
)


def spec_for(code: str) -> EtfSpec | None:
    for spec in ETF_CATALOG:
        if spec.code == code:
            return spec
    return None


def liquidity(
    candles: dict[str, pd.DataFrame],
    *,
    days: int = 250,
) -> pd.DataFrame:
    """최근 `days` 거래일의 일평균 거래대금(원)을 종목별로 낸다.

    평균이 아니라 **중앙값**을 쓴다. 거래대금 분포는 한쪽으로 심하게 치우쳐서
    (테마가 붙은 며칠이 전체를 끌어올린다) 평균은 평소 체결 난이도를 과대평가한다.

    왜 이걸 재는가: 백테스트는 왕복 13bp 를 가정했다. 그 가정은 호가 한두 칸
    안에서 체결된다는 뜻인데, 일 거래대금이 억 단위 아래면 성립하지 않는다.
    """
    rows = []
    for code, frame in candles.items():
        tail = frame.tail(days)
        turnover = (tail["close"] * tail["volume"]).astype(float)
        spec = spec_for(code)
        rows.append(
            {
                "code": code,
                "name": spec.name if spec else code,
                "asset": spec.asset if spec else "",
                "median_turnover": float(turnover.median()),
                "min_turnover": float(turnover.min()),
                "days": int(len(tail)),
            }
        )
    out = pd.DataFrame(rows).set_index("code")
    return out.sort_values(["asset", "median_turnover"], ascending=[True, False])


@dataclass
class TaxedResult:
    """세후 백테스트 결과. `BacktestResult` 와 같은 지표를 쓰되 세금을 뺀다."""

    name: str
    equity: pd.Series
    daily: pd.Series
    tax_paid: float             # 누적 납부세액 (초기자본 1.0 기준)
    rebalances: int
    realized_gain: float = 0.0

    @property
    def years(self) -> float:
        return (self.equity.index[-1] - self.equity.index[0]).days / 365.25

    @property
    def cagr(self) -> float:
        return float(self.equity.iloc[-1] ** (1 / self.years) - 1)

    def to_backtest(self, *, risk_free: float = 0.02) -> BacktestResult:
        """꾸준함 지표(양수율·궤양지수 등)를 그대로 재사용하기 위한 변환."""
        weight = pd.Series(1.0, index=self.daily.index)
        return BacktestResult(
            self.name, self.equity, weight, self.daily, self.rebalances,
            risk_free=risk_free,
        )


def after_tax_mix(
    assets: dict[str, pd.Series],
    weights: dict[str, float],
    *,
    rebalance: str = "QE",
    one_way_bps: float = ETF_ONE_WAY_BPS,
    tax_rates: dict[str, float] | None = None,
    name: str | None = None,
) -> TaxedResult:
    """리밸런싱마다 실현이익에 과세하는 정적 배분 — 일반계좌 모형.

    `tax_rates` 를 전부 0 으로 주면 연금계좌(운용 중 비과세)가 된다.
    같은 코드로 두 계좌를 돌려야 세금 항목만 차이로 남는다.

    과세 모형
    ---------
    자산별로 취득원가를 평균단가로 들고 간다. 리밸런싱에서 목표 비중보다
    많아진 자산을 파는데, 그 매도분에 실려 있던 평가이익만 실현이익이 된다.
    세금은 그 자리에서 현금으로 빠진다.

    일부러 넣지 않은 것
    -------------------
    - 손실 이월·손익통산: 넣으면 세금이 줄어드는 방향이므로, 빼는 쪽이
      일반계좌에 불리한 보수적 가정이다.
    - 분배금 과세: 여기 쓰는 가격은 수정주가라 분배금이 가격에 녹아 있다.
      실제로는 받는 시점에 15.4% 가 원천징수되므로 일반계좌가 조금 더 불리하다.
    - 금융소득종합과세(연 2천만원 초과): 원금이 커지면 세율이 올라간다.
    세 항목 모두 일반계좌를 불리하게 만드는 방향이므로, 아래 결과는
    **일반계좌 쪽 최선의 경우**로 읽어야 한다.
    """
    frame = pd.DataFrame(assets).dropna()
    codes = list(frame.columns)
    rates = tax_rates or {}
    rate = np.array([rates.get(c, OVERSEAS_TAX_RATE) for c in codes], dtype=float)
    target = np.array([weights[c] for c in codes], dtype=float)
    target = target / target.sum()

    returns = frame.pct_change().fillna(0.0)
    marks = _rebalance_marks(frame.index, rebalance)

    value = target.copy()          # 자산별 평가액 (총 1.0 에서 시작)
    basis = target.copy()          # 자산별 취득원가
    cost_bps = one_way_bps / 10_000.0

    daily, tax_total, gain_total, count = [], 0.0, 0.0, 0
    prev_total = 1.0
    for timestamp, row in returns.iterrows():
        value = value * (1 + row.to_numpy())
        total = float(value.sum())
        if timestamp in marks and total > 0:
            desired = target * total
            sell = np.maximum(value - desired, 0.0)
            # 매도분에 실려 있던 평가이익만 실현된다.
            safe = np.where(value > 0, value, 1.0)
            unrealized = np.clip(1 - basis / safe, 0.0, None)
            gain = sell * unrealized
            tax = float((gain * rate).sum())
            traded = float(sell.sum())
            fee = traded * 2 * cost_bps

            total = total - tax - fee
            held = target * total          # 세금·수수료를 낸 뒤의 최종 보유액
            # 취득원가 갱신: 줄어든 자산은 비례해서 깎고, 늘어난 자산은 더한다.
            ratio = np.clip(held / safe, 0.0, 1.0)
            basis = np.where(held < value, basis * ratio,
                             basis + np.maximum(held - value, 0.0))
            value = held
            tax_total += tax
            gain_total += float(gain.sum())
            count += 1
        daily.append(total / prev_total - 1)
        prev_total = total

    series = pd.Series(daily, index=returns.index)
    equity = (1 + series).cumprod()
    label = name or ("세후 " + "+".join(f"{c}{w:.0%}" for c, w in zip(codes, target)))
    return TaxedResult(label, equity, series, tax_total, count, gain_total)


def account_comparison(
    assets: dict[str, pd.Series],
    weights: dict[str, float],
    *,
    rebalance: str = "QE",
    domestic_equity: tuple[str, ...] = (),
) -> pd.DataFrame:
    """연금계좌 vs 일반계좌 — 세금 항목만 다른 두 번의 실행.

    연금계좌의 진짜 이점(납입 세액공제 13.2~16.5%)은 여기 들어 있지 않다.
    그건 운용 성과가 아니라 납입 시점에 받는 환급이라 백테스트로 잴 대상이
    아니다. 여기서 재는 것은 **운용 중 과세이연 효과 하나뿐**이다.
    """
    pension = after_tax_mix(
        assets, weights, rebalance=rebalance,
        tax_rates={c: 0.0 for c in assets}, name="연금계좌 (운용 중 비과세)",
    )
    taxed_rates = {
        c: (KOREA_TAX_RATE if c in domestic_equity else OVERSEAS_TAX_RATE)
        for c in assets
    }
    taxable = after_tax_mix(
        assets, weights, rebalance=rebalance,
        tax_rates=taxed_rates, name="일반계좌 (리밸런싱마다 과세)",
    )
    rows = []
    for result in (pension, taxable):
        rows.append(
            {
                "계좌": result.name,
                "연수익": result.cagr,
                "최종배수": float(result.equity.iloc[-1]),
                "납부세액": result.tax_paid,
                "리밸런싱": result.rebalances,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["drag"] = float(pension.cagr - taxable.cagr)
    return out


def horizon_gap(drag: float, base_cagr: float, years: tuple[int, ...] = (10, 20, 30),
                principal: float = 10_000_000) -> pd.DataFrame:
    """연 `drag` 만큼의 수익률 차이가 기간이 길어지면 얼마가 되는가.

    연 0.1%p 는 작아 보이지만 복리로 30년이면 눈에 보이는 금액이 된다.
    반대로 "세금 때문에 일반계좌는 안 된다"는 통념이 과장인지도 여기서 갈린다.
    """
    rows = []
    for year in years:
        high = principal * (1 + base_cagr) ** year
        low = principal * (1 + base_cagr - drag) ** year
        rows.append({"기간": f"{year}년", "연금계좌": high, "일반계좌": low,
                     "차이": high - low})
    return pd.DataFrame(rows)


# 연금계좌 세액공제 (2026년 기준)
PENSION_LIMIT = 9_000_000          # 연금저축 600 + IRP 300
DEDUCTION_LOW = 0.165              # 총급여 5,500만원 이하 (지방소득세 포함)
DEDUCTION_HIGH = 0.132             # 초과
PENSION_EXIT_TAX = 0.055           # 연금소득세 3.3~5.5%, 보수적으로 상단


def contribution_plan(
    *,
    annual_return: float,
    years: int,
    yearly: float = PENSION_LIMIT,
    deduction: float = DEDUCTION_LOW,
    exit_tax: float = PENSION_EXIT_TAX,
    taxable_drag: float = 0.0044,
) -> pd.DataFrame:
    """연금계좌 vs 일반계좌 — 매년 같은 금액을 넣었을 때 최종 수령액.

    왜 이 계산이 백테스트보다 중요한가
    ----------------------------------
    이 프로젝트가 찾아낸 어떤 매매 규칙도 연 몇 %p 를 다투는 수준이었다.
    세액공제는 납입한 해에 **확정적으로 16.5%** 를 돌려준다. 시장이 어떻든,
    전략이 맞든 틀리든 받는다. 크기가 다른 항목이다.

    모형
    ----
    연금계좌: 매년 `yearly` 납입 → 그 해 `deduction` 만큼 환급받아 **재투자**.
              운용 중 과세 없음. 인출 시 전액에 `exit_tax`.
    일반계좌: 매년 `yearly` 납입. 리밸런싱 과세로 수익률이 `taxable_drag` 만큼
              낮다 (실측 연 0.44%p). 인출 시 추가 과세 없음
              (국내주식형 비과세분이 섞여 있어 보수적으로 0 으로 둔다).

    일부러 뺀 것: 연금계좌의 55세 인출 제한과 중도해지 시 기타소득세 16.5%.
    돈이 묶이는 비용은 숫자로 잴 수 없고 사람마다 다르다.
    """
    rows = []
    pension = taxable = 0.0
    for year in range(1, years + 1):
        # 연금: 납입 + 작년치 환급 재투자
        refund = yearly * deduction
        pension = (pension + yearly + refund) * (1 + annual_return)
        taxable = (taxable + yearly) * (1 + annual_return - taxable_drag)
        rows.append({
            "연차": year,
            "누적납입": yearly * year,
            "연금계좌(세전)": pension,
            "연금계좌(인출후)": pension * (1 - exit_tax),
            "일반계좌": taxable,
        })
    frame = pd.DataFrame(rows)
    frame["차이"] = frame["연금계좌(인출후)"] - frame["일반계좌"]
    frame.attrs["deduction"] = deduction
    frame.attrs["exit_tax"] = exit_tax
    return frame


def edge_ranking(annual_return: float = 0.0972) -> pd.DataFrame:
    """이 프로젝트가 잰 것들을 **연 수익률 기여도**로 한 줄에 세운다.

    매매 규칙을 찾느라 쓴 시간과, 계좌를 바꾸는 데 드는 시간의 크기 차이를
    보이기 위한 표다. 세액공제는 납입액 기준이라 다른 항목과 단위가 다르므로,
    900만원을 넣었을 때 그 900만원에 대한 1년 수익률로 환산해 비교한다.
    """
    rows = [
        {"항목": "연금계좌 세액공제 (총급여 5,500만원 이하)",
         "연 기여": DEDUCTION_LOW, "성격": "확정", "근거": "세법"},
        {"항목": "연금계좌 세액공제 (초과)",
         "연 기여": DEDUCTION_HIGH, "성격": "확정", "근거": "세법"},
        {"항목": "6자산 동일가중 분기리밸 (실측)",
         "연 기여": annual_return, "성격": "시장", "근거": "9.7년 백테스트"},
        {"항목": "리밸런싱 자체의 기여 (분기 vs 방치)",
         "연 기여": -0.0166, "성격": "시장", "근거": "10.72% vs 12.38%"},
        {"항목": "일반계좌 세금 드래그",
         "연 기여": -0.0044, "성격": "확정", "근거": "실측"},
        {"항목": "플랫베이스 패턴 매매 (편향 보정 후)",
         "연 기여": 0.0198, "성격": "미검증", "근거": "p=0.73"},
        {"항목": "달러 ETF 유동성 잘못 고른 비용",
         "연 기여": -0.005, "성격": "확정", "근거": "호가 스프레드 추정"},
    ]
    return pd.DataFrame(rows).sort_values("연 기여", ascending=False)
