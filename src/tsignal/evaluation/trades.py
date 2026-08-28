"""진입 신호 → 실제 체결 가능한 거래 시뮬레이션.

이벤트 스터디가 "엣지가 있는가"를 답한다면, 여기는
"익절/손절/청산 규칙을 붙였을 때 손에 남는가"를 답한다.

체결 가정 (보수적으로 잡았다)
  - 진입: 신호 봉 t 의 **다음 봉 시가**. 종가 체결은 실전에서 불가능하다.
  - 한 봉 안에서 익절선과 손절선이 모두 닿으면 **손절이 먼저** 닿았다고 본다.
  - 수수료/세금/슬리피지를 왕복 모두 차감한다.
  - 기본은 동시 1포지션. 신호가 겹쳐 나와도 중복 진입하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators._util import atr


@dataclass(frozen=True)
class CostModel:
    """국내 주식 매매비용 기본값 (2026년 기준, 실제 계좌 조건으로 바꿔 쓸 것)."""

    fee_bps: float = 1.5          # 위탁수수료, 편도 (0.015%)
    tax_bps: float = 15.0         # 증권거래세+농특세, 매도 시에만 (0.15%)
    slippage_bps: float = 5.0     # 호가 스프레드/체결 지연, 편도

    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.fee_bps + self.slippage_bps) + self.tax_bps

    def apply(self, gross_return: pd.Series | np.ndarray) -> np.ndarray:
        return np.asarray(gross_return, dtype=float) - self.round_trip_bps / 10_000.0


ZERO_COST = CostModel(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ExitPolicy:
    """청산 규칙. 이 프로젝트에서 '어떤 신호에 팔 것인가'가 구현되는 자리.

    우선순위: 손절 → 익절 → 청산신호 → 최대보유 → 세션마감
    """

    stop_atr: float | None = 1.0        # 손절 = 진입가 - stop_atr * ATR
    target_atr: float | None = 2.0      # 익절 = 진입가 + target_atr * ATR
    stop_pct: float | None = None       # ATR 대신 고정 %로 쓰고 싶을 때
    target_pct: float | None = None
    max_bars: int = 40                  # 시간 손절
    exit_signal: str | None = None      # signals 레지스트리의 exit 신호 이름
    close_at_session_end: bool = True   # 오버나이트 금지
    atr_period: int = 14

    def describe(self) -> str:
        parts = []
        if self.stop_pct is not None:
            parts.append(f"손절 -{self.stop_pct:.2f}%")
        elif self.stop_atr is not None:
            parts.append(f"손절 -{self.stop_atr:g}ATR")
        if self.target_pct is not None:
            parts.append(f"익절 +{self.target_pct:.2f}%")
        elif self.target_atr is not None:
            parts.append(f"익절 +{self.target_atr:g}ATR")
        parts.append(f"최대 {self.max_bars}봉")
        if self.exit_signal:
            parts.append(f"청산신호 {self.exit_signal}")
        if self.close_at_session_end:
            parts.append("당일청산")
        return " / ".join(parts)


TRADE_COLUMNS = [
    "entry_time", "entry_price", "exit_time", "exit_price", "bars_held",
    "ret_gross", "ret_net", "mfe", "mae", "exit_reason",
]


def simulate(
    candles: pd.DataFrame,
    entries: pd.Series,
    *,
    policy: ExitPolicy = ExitPolicy(),
    costs: CostModel = CostModel(),
    exit_events: pd.Series | None = None,
    allow_overlap: bool = False,
) -> pd.DataFrame:
    """진입 신호를 거래 리스트로 바꾼다."""
    idx = candles.index
    n = len(candles)
    open_, high, low = (candles[c].to_numpy(float) for c in ("open", "high", "low"))

    atr_arr = atr(candles, policy.atr_period).to_numpy(float)
    entry_mask = entries.reindex(idx).fillna(False).to_numpy(bool)
    exit_mask = (
        exit_events.reindex(idx).fillna(False).to_numpy(bool)
        if exit_events is not None else np.zeros(n, dtype=bool)
    )
    session = np.asarray([d.date() for d in idx.to_pydatetime()])

    trades: list[dict] = []
    blocked_until = -1

    for t in range(n - 1):
        if not entry_mask[t] or (not allow_overlap and t < blocked_until):
            continue
        entry_i = t + 1                       # 다음 봉 시가 진입
        entry_price = open_[entry_i]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        unit = atr_arr[t]
        if policy.stop_pct is not None:
            stop = entry_price * (1 - policy.stop_pct / 100)
        elif policy.stop_atr is not None and np.isfinite(unit):
            stop = entry_price - policy.stop_atr * unit
        else:
            stop = -np.inf

        if policy.target_pct is not None:
            target = entry_price * (1 + policy.target_pct / 100)
        elif policy.target_atr is not None and np.isfinite(unit):
            target = entry_price + policy.target_atr * unit
        else:
            target = np.inf

        last_i = min(entry_i + policy.max_bars - 1, n - 1)
        exit_i, exit_price, reason = last_i, None, "max_bars"
        run_high, run_low = -np.inf, np.inf

        for j in range(entry_i, last_i + 1):
            run_high, run_low = max(run_high, high[j]), min(run_low, low[j])

            if low[j] <= stop:                      # 보수적: 손절 우선
                exit_i, exit_price, reason = j, stop, "stop"
                break
            if high[j] >= target:
                exit_i, exit_price, reason = j, target, "target"
                break
            if policy.close_at_session_end and (j == n - 1 or session[j + 1] != session[j]):
                exit_i, exit_price, reason = j, candles["close"].iat[j], "session_end"
                break
            if exit_mask[j] and j > entry_i:
                # 청산 신호는 그 봉 종가 확정 후에만 알 수 있다 → 다음 봉 시가에 나간다.
                nxt = min(j + 1, n - 1)
                exit_i, exit_price, reason = nxt, open_[nxt], "exit_signal"
                break
        else:
            exit_price = candles["close"].iat[last_i]

        if exit_price is None or not np.isfinite(exit_price):
            continue

        gross = exit_price / entry_price - 1
        trades.append({
            "entry_time": idx[entry_i], "entry_price": entry_price,
            "exit_time": idx[exit_i], "exit_price": exit_price,
            "bars_held": exit_i - entry_i + 1,
            "ret_gross": gross,
            "ret_net": gross - costs.round_trip_bps / 10_000.0,
            "mfe": run_high / entry_price - 1,
            "mae": run_low / entry_price - 1,
            "exit_reason": reason,
        })
        blocked_until = exit_i + 1

    if not trades:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame(trades)[TRADE_COLUMNS]


def exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """청산 사유별 성적. '왜 팔았는가'가 성과에 어떻게 기여하는지 본다."""
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("exit_reason")["ret_net"]
    out = pd.DataFrame({
        "n": g.size(),
        "share": g.size() / len(trades),
        "mean_ret": g.mean(),
        "total_ret": g.sum(),
        "win_rate": g.apply(lambda r: float((r > 0).mean())),
    })
    return out.sort_values("n", ascending=False)


def suggest_barriers(
    candles: pd.DataFrame,
    entries: pd.Series,
    *,
    horizon: int = 40,
    quantiles: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    """MFE/MAE 분포로 익절/손절 폭 후보를 뽑는다.

    "익절 2%, 손절 1%" 를 감으로 정하지 않기 위한 근거표.
    ATR 배수로도 같이 준다 — 종목/기간이 달라져도 옮겨 쓸 수 있는 단위이기 때문.
    """
    from .forward import excursions

    ex = excursions(candles, horizon)
    unit = (atr(candles, 14) / candles["open"].shift(-1)).rename("atr_unit")
    mask = entries.reindex(candles.index).fillna(False).astype(bool)
    sub = ex.loc[mask].join(unit.loc[mask]).dropna()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for q in quantiles:
        mfe_q, mae_q = sub["mfe"].quantile(q), sub["mae"].quantile(1 - q)
        rows.append({
            "quantile": q,
            "mfe_pct": mfe_q * 100,
            "mae_pct": mae_q * 100,
            "mfe_atr": mfe_q / sub["atr_unit"].median(),
            "mae_atr": mae_q / sub["atr_unit"].median(),
        })
    return pd.DataFrame(rows).set_index("quantile")
