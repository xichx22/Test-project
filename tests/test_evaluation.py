import numpy as np
import pandas as pd
import pytest

from tsignal import indicators as ind
from tsignal import signals as sig
from tsignal.datasource import Interval
from tsignal.evaluation import metrics, validation
from tsignal.evaluation.forward import forward_returns, screen_signals
from tsignal.evaluation.trades import CostModel, ExitPolicy, ZERO_COST, simulate


def _bars(rows) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:00", periods=len(rows), freq="5min", tz="Asia/Seoul")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx).astype(float)


def test_forward_return_uses_next_open():
    bars = _bars([[100, 101, 99, 100, 1], [110, 111, 109, 110, 1], [120, 121, 119, 120, 1]])
    fwd = forward_returns(bars, (1,), entry="next_open")
    # 0번 봉 신호 → 1번 봉 시가(110) 진입 → 2번 봉 시가(120) 청산
    assert fwd["fwd_1"].iloc[0] == pytest.approx(120 / 110 - 1)
    assert np.isnan(fwd["fwd_1"].iloc[-1])


def test_simulate_enters_at_next_open_not_signal_close():
    bars = _bars([[100, 100, 100, 100, 1], [105, 106, 104, 105, 1], [105, 105, 105, 105, 1]])
    entries = pd.Series([True, False, False], index=bars.index)
    trades = simulate(bars, entries, policy=ExitPolicy(stop_atr=None, target_atr=None, max_bars=2,
                                                       close_at_session_end=False), costs=ZERO_COST)
    assert len(trades) == 1
    assert trades["entry_price"].iloc[0] == 105     # 신호 봉 종가 100 이 아니라 다음 봉 시가 105
    assert trades["entry_time"].iloc[0] == bars.index[1]


def test_simulate_prefers_stop_when_both_barriers_hit_in_one_bar():
    """한 봉에 익절선과 손절선이 모두 닿으면 보수적으로 손절 처리한다."""
    bars = _bars([[100, 100, 100, 100, 1], [100, 130, 70, 100, 1], [100, 100, 100, 100, 1]])
    entries = pd.Series([True, False, False], index=bars.index)
    trades = simulate(bars, entries,
                      policy=ExitPolicy(stop_atr=None, target_atr=None, stop_pct=10, target_pct=10,
                                        max_bars=2, close_at_session_end=False),
                      costs=ZERO_COST)
    assert trades["exit_reason"].iloc[0] == "stop"
    assert trades["ret_gross"].iloc[0] == pytest.approx(-0.10)


def test_costs_reduce_every_trade_by_round_trip(candles_5m):
    features = ind.compute_all(candles_5m, interval=Interval.M5)
    entries = sig.evaluate_all(candles_5m, features)["ema_pullback"]
    costs = CostModel()
    trades = simulate(candles_5m, entries, costs=costs)
    assert not trades.empty
    diff = trades["ret_gross"] - trades["ret_net"]
    assert np.allclose(diff, costs.round_trip_bps / 10_000)


def test_no_overlapping_positions(candles_5m):
    features = ind.compute_all(candles_5m, interval=Interval.M5)
    entries = sig.evaluate_all(candles_5m, features)["williams_oversold_turn"]
    trades = simulate(candles_5m, entries)
    assert (trades["entry_time"].iloc[1:].to_numpy() > trades["exit_time"].iloc[:-1].to_numpy()).all()


def test_deflated_threshold_rises_with_trial_count():
    assert validation.deflated_threshold(1) == pytest.approx(1.96, abs=0.01)
    assert validation.deflated_threshold(40) > validation.deflated_threshold(10) > validation.deflated_threshold(1)


def test_metrics_on_known_series():
    r = pd.Series([0.02, -0.01, 0.03, -0.01])
    assert metrics.win_rate(r) == 0.5
    assert metrics.expectancy(r) == pytest.approx(0.00750)
    assert metrics.payoff_ratio(r) == pytest.approx(2.5)
    assert metrics.profit_factor(r) == pytest.approx(2.5)


def test_random_walk_produces_no_significant_edge(candles_5m):
    """합성 랜덤워크에서는 어떤 신호도 보정된 문턱을 넘으면 안 된다.

    넘는다면 검증 코드가 없는 엣지를 만들어내고 있다는 뜻이다.
    """
    features = ind.compute_all(candles_5m, interval=Interval.M5)
    entries = sig.evaluate_all(candles_5m, features, kind="entry")
    screen = screen_signals(candles_5m, entries)
    threshold = validation.deflated_threshold(len(entries.columns))
    survivors = screen[(screen["n"] >= 30) & (screen["t_5"].abs() >= threshold)]
    assert survivors.empty, f"랜덤워크에서 유의한 신호가 나왔다: {list(survivors.index)}"
