"""운용 규칙 시뮬레이터 — 체결·손절·자리 제한이 실제로 지켜지는가."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.playbook import Plan, run_plan, summarize


def _bars(close, high=None, low=None, open_=None, volume=1e9, start="2020-01-01"):
    close = np.asarray(close, float)
    n = len(close)
    index = pd.date_range(start, periods=n, freq="B", tz="Asia/Seoul")
    return pd.DataFrame(
        {"open": close if open_ is None else np.asarray(open_, float),
         "high": close * 1.01 if high is None else np.asarray(high, float),
         "low": close * 0.99 if low is None else np.asarray(low, float),
         "close": close,
         "volume": np.full(n, float(volume))},
        index=index,
    )


def _signal(frame, positions):
    hit = pd.Series(False, index=frame.index)
    for p in positions:
        hit.iloc[p] = True
    return hit


def test_entry_happens_on_the_next_bar_open():
    """신호 봉 종가가 아니라 다음 봉 시가에 산다."""
    close = np.full(40, 100.0)
    open_ = np.full(40, 100.0)
    open_[11] = 105.0                       # 다음 봉 시가만 다르게
    frame = _bars(close, open_=open_)
    result, log = run_plan({"A": _signal(frame, [10])}, {"A": frame},
                           Plan(max_positions=1, holding_days=5, stop_loss=None))
    assert len(log) == 1
    assert log.iloc[0]["entry"] == frame.index[11]


def test_stop_loss_exits_at_the_stop_price():
    close = np.r_[np.full(12, 100.0), np.full(28, 100.0)]
    low = close * 0.99
    low[15] = 80.0                          # 장중에 손절가(-8%=92)를 관통
    frame = _bars(close, low=low)
    _, log = run_plan({"A": _signal(frame, [10])}, {"A": frame},
                      Plan(max_positions=1, holding_days=60, stop_loss=0.08))
    assert log.iloc[0]["reason"] == "stop"
    assert log.iloc[0]["수익률"] == pytest.approx(-0.08, abs=1e-9)


def test_gap_down_fills_at_the_open_not_the_stop():
    """갭하락이면 손절가로 못 나간다. 시가로 나가야 성과가 부풀지 않는다."""
    close = np.full(40, 100.0)
    open_ = np.full(40, 100.0)
    low = close * 0.99
    open_[15], low[15], close[15] = 70.0, 68.0, 69.0   # 손절가 92 아래로 갭
    frame = _bars(close, low=low, open_=open_)
    _, log = run_plan({"A": _signal(frame, [10])}, {"A": frame},
                      Plan(max_positions=1, holding_days=60, stop_loss=0.08))
    assert log.iloc[0]["reason"] == "stop"
    assert log.iloc[0]["수익률"] == pytest.approx(-0.30, abs=1e-9)


def test_time_exit_after_the_holding_period():
    frame = _bars(np.full(60, 100.0))
    _, log = run_plan({"A": _signal(frame, [10])}, {"A": frame},
                      Plan(max_positions=1, holding_days=20, stop_loss=None))
    assert log.iloc[0]["reason"] == "time"
    assert log.iloc[0]["bars"] == 20


def test_take_profit_caps_the_winner():
    close = np.r_[np.full(12, 100.0), np.linspace(100, 200, 28)]
    frame = _bars(close)
    _, log = run_plan({"A": _signal(frame, [10])}, {"A": frame},
                      Plan(max_positions=1, holding_days=60, stop_loss=None,
                           take_profit=0.20))
    assert log.iloc[0]["reason"] == "target"
    assert log.iloc[0]["수익률"] == pytest.approx(0.20, abs=1e-9)


def test_position_cap_is_never_exceeded():
    """자리 수를 넘겨 사면 안 된다 — 이게 개인이 실제로 지는 제약이다."""
    frames = {f"S{i}": _bars(np.full(80, 100.0)) for i in range(8)}
    events = {c: _signal(f, [10]) for c, f in frames.items()}
    result, log = run_plan(events, frames,
                           Plan(max_positions=3, holding_days=60, stop_loss=None))
    assert len(log) == 3, "자리가 3개인데 3건을 넘게 체결했다"


def test_freed_slot_is_reused():
    """앞 포지션이 청산되면 그 자리에 새로 들어가야 한다."""
    a = _bars(np.full(80, 100.0))
    b = _bars(np.full(80, 100.0))
    events = {"A": _signal(a, [10]), "B": _signal(b, [40])}
    _, log = run_plan(events, {"A": a, "B": b},
                      Plan(max_positions=1, holding_days=10, stop_loss=None))
    assert set(log["code"]) == {"A", "B"}


def test_costs_are_charged_on_both_sides():
    frame = _bars(np.full(40, 100.0))
    events = {"A": _signal(frame, [10])}
    free, _ = run_plan(events, {"A": frame},
                       Plan(max_positions=1, holding_days=10, stop_loss=None,
                            one_way_bps=0.0, cash_rate=0.0))
    paid, _ = run_plan(events, {"A": frame},
                       Plan(max_positions=1, holding_days=10, stop_loss=None,
                            one_way_bps=14.0, cash_rate=0.0))
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]


def test_illiquid_names_are_skipped():
    """거래대금 하한을 넘지 못하는 종목은 후보에서 빠진다."""
    frame = _bars(np.full(40, 100.0), volume=1.0)     # 거래대금 100원
    _, log = run_plan({"A": _signal(frame, [10])}, {"A": frame},
                      Plan(max_positions=1, min_turnover=3e8))
    assert log.empty


def test_ranking_prefers_the_more_liquid_name_when_slots_are_short():
    thin = _bars(np.full(40, 100.0), volume=1e7)
    thick = _bars(np.full(40, 100.0), volume=1e9)
    events = {"THIN": _signal(thin, [10]), "THICK": _signal(thick, [10])}
    _, log = run_plan(events, {"THIN": thin, "THICK": thick},
                      Plan(max_positions=1, holding_days=10, rank_by="liquidity",
                           stop_loss=None))
    assert list(log["code"]) == ["THICK"]


def test_unclosed_positions_are_still_logged():
    """끝까지 안 팔린 포지션이 로그에서 빠지면 승률이 가짜로 좋아진다."""
    frame = _bars(np.full(40, 100.0))
    _, log = run_plan({"A": _signal(frame, [10])}, {"A": frame},
                      Plan(max_positions=1, holding_days=200, stop_loss=None))
    assert len(log) == 1
    assert log.iloc[0]["reason"] == "open"


def test_summarize_reports_win_rate_and_payoff():
    log = pd.DataFrame({"code": list("abcd"), "bars": [10] * 4,
                        "reason": ["time"] * 4,
                        "수익률": [0.30, -0.08, -0.08, -0.08]})
    out = summarize(log)
    assert out["승률"] == pytest.approx(0.25)
    assert out["손익비"] == pytest.approx(0.30 / 0.08, abs=1e-6)


def test_summarize_handles_an_empty_log():
    assert summarize(pd.DataFrame()) == {}
