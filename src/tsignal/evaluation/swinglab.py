"""스윙 전략 순위표 — "연평균수익이 가장 큰 모델"을 데이터로 찾는다.

앞의 검증과 무엇이 다른가
--------------------------
`combine.py` 는 "이 신호에 **통계적으로 유의한** 초과수익이 있는가"를 물었다.
답은 전부 아니오였다. 여기서는 질문이 다르다 — "실제로 돈을 굴렸을 때
**연평균수익**이 가장 큰 규칙은 무엇인가".

같은 데이터를 다른 잣대로 보는 것이므로 결과도 다르게 나온다. 그리고 그
차이가 정확히 위험한 지점이다: 수천 개를 줄 세워 1등을 뽑으면, 그 1등은
**실력이 아니라 운으로도 나온다**. 그래서 이 모듈은 1등을 뽑는 것과 동시에
그 1등이 다음 구간에서도 1등인지를 반드시 같이 낸다.

계산 구조
---------
종목별로 지표를 다시 계산하면 조합 수천 개를 못 돈다. 그래서 (날짜 × 종목)
행렬을 트리거·필터별로 한 번만 만들어 두고, 조합 평가는 boolean AND 와
누적합으로만 한다. 보유 여부는 `h`일 롤링 OR 인데, 누적합의 차분으로
한 번에 낸다 — `cs[d] - cs[d-h] > 0`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..datasource.base import Interval
from ..signals import filters as filt
from ..signals import library as _lib  # noqa: F401  (신호 등록을 위한 임포트)
from ..signals import base as sig
from .allocation import BacktestResult

# 국내주식 왕복 비용: 수수료 1.5bp×2 + 증권거래세 15bp + 슬리피지 5bp×2
STOCK_ROUND_TRIP_BPS = 28.0

BARE = "(필터없음)"


@dataclass
class SwingResult:
    """한 규칙의 실행 결과. 순위표 한 줄이 된다."""

    trigger: str
    filter_name: str
    holding: int
    max_positions: int | None
    result: BacktestResult

    @property
    def name(self) -> str:
        cap = f" 상위{self.max_positions}" if self.max_positions else ""
        return f"{self.trigger} + {self.filter_name} / {self.holding}일{cap}"

    def row(self) -> dict:
        r = self.result
        return {
            "트리거": self.trigger,
            "필터": self.filter_name,
            "보유일": self.holding,
            "동시보유": self.max_positions or 0,
            "연수익": r.cagr,
            "변동성": r.volatility,
            "샤프": r.sharpe,
            "MDD": r.max_drawdown,
            "양수율12M": r.rolling_positive(12),
            "최악12M": r.worst_rolling(12),
            "궤양": r.ulcer_index,
            "투자비중": r.exposure,
            "매매횟수": r.trades,
        }


class SwingLab:
    """(날짜 × 종목) 행렬 위에서 스윙 규칙 수천 개를 돌린다."""

    def __init__(
        self,
        candles_by_code: Mapping[str, pd.DataFrame],
        *,
        interval: Interval = Interval.D1,
        trigger_names: Sequence[str] | None = None,
        filter_names: Sequence[str] | None = None,
        exclude_tags: tuple[str, ...] = (),
        _prebuilt: dict | None = None,
    ) -> None:
        if _prebuilt is not None:
            self.__dict__.update(_prebuilt)
            return

        codes = sorted(candles_by_code)
        index = None
        for code in codes:
            idx = candles_by_code[code].index
            index = idx if index is None else index.union(idx)
        self.dates: pd.DatetimeIndex = index
        self.codes: list[str] = codes

        n_d, n_c = len(self.dates), len(codes)
        self.ret = np.zeros((n_d, n_c), dtype=np.float64)
        # 체결일에는 종가-종가가 아니라 **시가→종가**를 써야 한다.
        # 종가-종가를 쓰면 사기 전에 벌어진 전일종가→시가 갭을 먹는다.
        self.entry_ret = np.zeros((n_d, n_c), dtype=np.float64)
        self.valid = np.zeros((n_d, n_c), dtype=bool)

        trig_stack: dict[str, np.ndarray] = {}
        filt_stack: dict[str, np.ndarray] = {}

        for j, code in enumerate(codes):
            candles = candles_by_code[code]
            features = ind.compute_all(candles, interval=interval)
            triggers = sig.evaluate_all(
                candles, features, kind="entry",
                names=trigger_names, exclude_tags=exclude_tags,
            )
            states = filt.evaluate_all(candles, features, names=filter_names)
            rows = self.dates.get_indexer(candles.index)

            # 수익률은 종가 대비 종가. 첫 봉은 0 (진입 자체가 다음 날이라 무해).
            self.ret[rows, j] = candles["close"].pct_change().fillna(0.0).to_numpy()
            self.entry_ret[rows, j] = (
                candles["close"] / candles["open"] - 1).fillna(0.0).to_numpy()
            self.valid[rows, j] = True

            for name in triggers.columns:
                block = trig_stack.setdefault(
                    name, np.zeros((n_d, n_c), dtype=bool))
                block[rows, j] = triggers[name].to_numpy(bool)
            for name in states.columns:
                block = filt_stack.setdefault(
                    name, np.zeros((n_d, n_c), dtype=bool))
                block[rows, j] = states[name].to_numpy(bool)

        self.trig = trig_stack
        self.filt = filt_stack

    # ------------------------------------------------------------------ 조회
    @property
    def trigger_names(self) -> list[str]:
        return sorted(self.trig)

    @property
    def filter_names(self) -> list[str]:
        return sorted(self.filt)

    def slice(self, start=None, end=None) -> "SwingLab":
        """기간을 잘라 같은 판을 재사용한다 (지표 재계산 없음).

        구간을 잘라도 미래참조가 생기지 않는다 — 지표가 인과적이라 t 시점 값이
        t 이전만 쓰기 때문이다. 오히려 워밍업 왜곡이 없어 실제에 더 가깝다.
        """
        def _at(value) -> pd.Timestamp:
            stamp = pd.Timestamp(value)
            # 이미 tz 가 붙은 값에 tz= 를 다시 주면 pandas 가 거부한다.
            return (stamp.tz_convert(self.dates.tz) if stamp.tzinfo
                    else stamp.tz_localize(self.dates.tz))

        mask = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            mask &= self.dates >= _at(start)
        if end is not None:
            mask &= self.dates <= _at(end)
        return SwingLab(
            {}, _prebuilt={
                "dates": self.dates[mask], "codes": self.codes,
                "ret": self.ret[mask], "entry_ret": self.entry_ret[mask],
                "valid": self.valid[mask],
                "trig": {k: v[mask] for k, v in self.trig.items()},
                "filt": {k: v[mask] for k, v in self.filt.items()},
            },
        )

    # ------------------------------------------------------------------ 실행
    def _held(self, entry: np.ndarray, holding: int) -> np.ndarray:
        """진입 신호 → 보유 여부. 체결은 신호 다음 날, `holding`일 보유."""
        enter = np.zeros_like(entry)
        enter[1:] = entry[:-1]                      # t+1 체결
        enter &= self.valid
        cs = np.cumsum(enter, axis=0, dtype=np.int32)
        past = np.zeros_like(cs)
        if holding < len(cs):
            past[holding:] = cs[:-holding]
        return ((cs - past) > 0) & self.valid, enter

    def run(
        self,
        trigger: str,
        filter_name: str = BARE,
        *,
        holding: int = 20,
        cost_bps: float = STOCK_ROUND_TRIP_BPS,
        cash_rate: float = 0.02,
        max_positions: int | None = None,
        risk_free: float = 0.02,
    ) -> SwingResult:
        entry = self.trig[trigger].copy()
        if filter_name != BARE:
            entry &= self.filt[filter_name]

        held, enter = self._held(entry, holding)
        if max_positions:
            # 한도를 넘으면 먼저 진입한 종목 우선 (컬럼 순서로 결정론적)
            held = held & (np.cumsum(held, axis=1) <= max_positions)
            enter = enter & held

        count = held.sum(axis=1)
        invested = count > 0
        denom = np.maximum(count, 1)
        # 체결일은 시가→종가, 그 뒤로는 종가→종가.
        per_bar = np.where(enter, self.entry_ret, self.ret)
        gross = np.where(held, per_bar, 0.0).sum(axis=1) / denom
        cash_daily = (1 + cash_rate) ** (1 / 252) - 1
        daily = np.where(invested, gross, cash_daily)
        # 그날 새로 편입된 비중만큼 왕복 비용을 한 번에 뺀다 (보수적)
        daily = daily - (enter.sum(axis=1) / denom) * (cost_bps / 10_000.0)

        series = pd.Series(daily, index=self.dates)
        equity = (1 + series).cumprod()
        weight = pd.Series(invested.astype(float), index=self.dates)
        name = f"{trigger}+{filter_name}/{holding}일"
        result = BacktestResult(name, equity, weight, series,
                                int(enter.sum()), risk_free=risk_free)
        return SwingResult(trigger, filter_name, holding, max_positions, result)

    def benchmark(self, *, cash_rate: float = 0.02,
                  risk_free: float = 0.02) -> BacktestResult:
        """유니버스 동일가중 매수후보유 — 모든 규칙이 넘어야 할 선.

        규칙이 이걸 못 이기면 "아무 것도 안 하고 전 종목을 들고 있는 것"보다
        못하다는 뜻이다. 신호를 쓰는 이유가 없어진다.
        """
        count = self.valid.sum(axis=1)
        denom = np.maximum(count, 1)
        first = self.valid & ~np.vstack(
            [np.zeros((1, self.valid.shape[1]), bool), self.valid[:-1]])
        per_bar = np.where(first, self.entry_ret, self.ret)
        daily = np.where(self.valid, per_bar, 0.0).sum(axis=1) / denom
        series = pd.Series(daily, index=self.dates)
        equity = (1 + series).cumprod()
        weight = pd.Series(1.0, index=self.dates)
        return BacktestResult("유니버스 동일가중 매수후보유", equity, weight,
                              series, 1, risk_free=risk_free)

    # ------------------------------------------------------------------ 탐색
    def leaderboard(
        self,
        *,
        holdings: Sequence[int] = (5, 10, 20, 60),
        max_positions: Sequence[int | None] = (None,),
        cost_bps: float = STOCK_ROUND_TRIP_BPS,
        min_trades: int = 100,
        min_exposure: float = 0.05,
        triggers: Sequence[str] | None = None,
        filters: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """모든 (트리거 × 필터 × 보유일 × 동시보유) 를 돌려 연수익으로 줄 세운다.

        `min_trades` / `min_exposure` 로 표본이 너무 얇은 규칙을 뺀다.
        매매 10번으로 나온 연 30% 는 규칙의 성질이 아니라 우연이다.
        """
        trig_list = list(triggers) if triggers is not None else self.trigger_names
        filt_list = [BARE] + (list(filters) if filters is not None
                              else self.filter_names)
        rows = []
        for trigger in trig_list:
            for filter_name in filt_list:
                for holding in holdings:
                    for cap in max_positions:
                        out = self.run(trigger, filter_name, holding=holding,
                                       cost_bps=cost_bps, max_positions=cap)
                        if out.result.trades < min_trades:
                            continue
                        if out.result.exposure < min_exposure:
                            continue
                        rows.append(out.row())
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("연수익", ascending=False).reset_index(drop=True)


def split_test(
    lab: SwingLab,
    board: pd.DataFrame,
    *,
    top: int = 20,
    frac: float = 0.5,
    cost_bps: float = STOCK_ROUND_TRIP_BPS,
) -> pd.DataFrame:
    """전반부 1등이 후반부에서도 1등인가.

    이 표가 이 모듈의 존재 이유다. 순위표 1등은 수천 개를 줄 세운 결과이므로
    **운으로도 1등이 된다**. 전반부에서 뽑고 후반부에서 채점해야 실력인지
    아닌지 갈린다. 순위 상관(스피어만)이 0 근처면 순위표는 잡음이다.
    """
    cut = lab.dates[int(len(lab.dates) * frac)]
    first, second = lab.slice(end=cut), lab.slice(start=cut)
    rows = []
    for _, row in board.head(top).iterrows():
        cap = int(row["동시보유"]) or None
        args = dict(holding=int(row["보유일"]), cost_bps=cost_bps, max_positions=cap)
        a = first.run(row["트리거"], row["필터"], **args).result
        b = second.run(row["트리거"], row["필터"], **args).result
        rows.append({
            "트리거": row["트리거"], "필터": row["필터"], "보유일": int(row["보유일"]),
            "전반부": a.cagr, "후반부": b.cagr, "차이": b.cagr - a.cagr,
            "전반매매": a.trades, "후반매매": b.trades,
        })
    out = pd.DataFrame(rows)
    if len(out) > 2:
        out.attrs["spearman"] = float(
            out["전반부"].rank().corr(out["후반부"].rank(), method="pearson"))
        out.attrs["bench_first"] = first.benchmark().cagr
        out.attrs["bench_second"] = second.benchmark().cagr
    return out
