"""청산 규칙은 발동한 표본만 세면 안 된다 — 전 표본 회계."""

import numpy as np
import pandas as pd
import pytest

from tsignal.evaluation.exitlab import (
    Exit, baseline_events, by_year, compare, insurance_cost, portfolio,
    resolve, split_by_regime, summarise,
)


def _frame(rows):
    """rows: (open, high, low, close) 목록."""
    index = pd.date_range("2020-01-01", periods=len(rows), freq="B", tz="Asia/Seoul")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)}, index=index)


def _flat(n, price=100.0):
    return _frame([(price, price, price, price)] * n)


def _signal(frame, positions):
    hit = pd.Series(False, index=frame.index)
    for p in positions:
        hit.iloc[p] = True
    return hit


def test_every_signal_gets_an_exit():
    """발동하지 않아도 만기 청산으로 남아야 한다. 이것이 핵심이다."""
    frame = _flat(40)
    events = {"A": _signal(frame, [0, 5, 10])}
    never = Exit("절대 안 켜짐", flag=lambda c, f: pd.Series(False, index=c.index),
                 horizon=10)
    trades = resolve(events, {"A": frame}, {"A": frame}, rule=never, cost_bps=0)
    assert len(trades) == 3
    assert set(trades["사유"]) == {"만기"}


def test_partial_trigger_rule_still_counts_the_whole_sample():
    """일부만 발동하는 규칙도 표본 수가 줄면 안 된다.

    옛 exit_timing 은 발동한 것만 세서 '오르면 판다' 가 항상 이겼다.
    """
    up = _frame([(100, 100, 100, 100)] * 5 + [(120, 120, 120, 120)] * 20)
    down = _frame([(100, 100, 100, 100)] * 5 + [(80, 80, 80, 80)] * 20)
    candles = {"UP": up, "DOWN": down}
    events = {"UP": _signal(up, [0]), "DOWN": _signal(down, [0])}
    # '20% 이상 오르면 판다' — DOWN 에서는 영영 안 켜진다
    rule = Exit("오르면 판다", take_profit=0.20, horizon=15)
    trades = resolve(events, candles, None, rule=rule, cost_bps=0)
    assert len(trades) == 2                      # 두 건 모두 남는다
    assert set(trades["사유"]) == {"익절", "만기"}
    assert trades["수익"].mean() == pytest.approx(0.0, abs=1e-9)   # +20%, −20%


def test_gap_down_fills_at_open_not_at_the_stop():
    """갭하락에서 손절가로 나갈 수 있다고 가정하면 성과가 부풀려진다."""
    frame = _frame([(100, 100, 100, 100), (100, 100, 100, 100),
                    (85, 86, 84, 85)] + [(85, 85, 85, 85)] * 10)
    events = {"A": _signal(frame, [0])}
    rule = Exit("8% 손절", stop_loss=0.08, horizon=10)
    trades = resolve(events, {"A": frame}, None, rule=rule, cost_bps=0)
    assert trades["사유"].iloc[0] == "손절"
    assert trades["청산"].iloc[0] == pytest.approx(85.0)   # 92 가 아니다


def test_stop_wins_over_target_on_the_same_bar():
    """봉 안의 순서는 일봉으로 알 수 없다 — 불리한 쪽으로 가정한다."""
    frame = _frame([(100, 100, 100, 100), (100, 100, 100, 100),
                    (100, 120, 88, 100)] + [(100, 100, 100, 100)] * 10)
    events = {"A": _signal(frame, [0])}
    rule = Exit("양쪽", stop_loss=0.10, take_profit=0.10, horizon=10)
    trades = resolve(events, {"A": frame}, None, rule=rule, cost_bps=0)
    assert trades["사유"].iloc[0] == "손절"


def test_trailing_stop_follows_the_peak():
    prices = [100] * 2 + [120] + [110] + [100] * 8
    frame = _frame([(p, p, p, p) for p in prices])
    events = {"A": _signal(frame, [0])}
    rule = Exit("고점 대비 10%", trail=0.10, horizon=10)
    trades = resolve(events, {"A": frame}, None, rule=rule, cost_bps=0)
    assert trades["사유"].iloc[0] == "손절"
    # 고점 120 → 손절선 108. 110 봉은 살아남고 그 다음 100 봉에서 나간다.
    assert trades["청산"].iloc[0] == pytest.approx(100.0)
    assert trades["보유봉"].iloc[0] == 4


def test_trailing_peak_uses_only_bars_before_the_current_one():
    """같은 봉의 고가로 손절선을 올리면 '고가가 먼저' 를 가정하게 된다.

    고가 120 · 저가 100 인 한 봉에서, 직전 고점이 100 이면 손절선은 90 이므로
    이 봉에서는 나가지 않아야 한다.
    """
    frame = _frame([(100, 100, 100, 100), (100, 100, 100, 100),
                    (100, 120, 100, 100)] + [(100, 100, 100, 100)] * 8)
    events = {"A": _signal(frame, [0])}
    trades = resolve(events, {"A": frame}, None,
                     rule=Exit("t", trail=0.10, horizon=8), cost_bps=0)
    assert trades["사유"].iloc[0] == "손절"
    assert trades["보유봉"].iloc[0] == 3        # 다음 봉(저가 100 ≤ 108)에서 나간다


def test_indicator_exit_fills_at_the_next_open():
    """지표는 종가로 판정하고 다음 봉 시가에 판다."""
    frame = _frame([(100, 100, 100, 100)] * 3 + [(130, 130, 130, 105)]
                   + [(100, 100, 100, 100)] * 8)
    events = {"A": _signal(frame, [0])}
    flags = pd.Series(False, index=frame.index)
    flags.iloc[2] = True                      # 2봉 종가 판정 → 3봉 시가 체결
    rule = Exit("지표", flag=lambda c, f: flags, horizon=10)
    trades = resolve(events, {"A": frame}, {"A": frame}, rule=rule, cost_bps=0)
    assert trades["사유"].iloc[0] == "지표"
    assert trades["청산"].iloc[0] == pytest.approx(130.0)


def test_cost_is_charged_once_round_trip():
    frame = _flat(20)
    events = {"A": _signal(frame, [0])}
    rule = Exit("만기", horizon=10)
    trades = resolve(events, {"A": frame}, None, rule=rule, cost_bps=28.0)
    assert trades["수익"].iloc[0] == pytest.approx(-0.0028)


def test_summarise_reports_reason_mix():
    frame = _frame([(100, 100, 100, 100)] * 2 + [(90, 90, 88, 90)] * 10)
    events = {"A": _signal(frame, [0])}
    got = summarise(resolve(events, {"A": frame}, None,
                            rule=Exit("s", stop_loss=0.05, horizon=8), cost_bps=0))
    assert got["손절%"] == pytest.approx(1.0)
    assert got["만기%"] == pytest.approx(0.0)


def test_compare_puts_the_same_rule_on_a_baseline():
    """손절이 신호와 무관하게 좋은 것인지 가리려면 기준선에도 걸어야 한다."""
    rng = np.random.default_rng(0)
    price = 100 * np.cumprod(1 + rng.normal(0, 0.02, 400))
    frame = _frame([(p, p * 1.02, p * 0.98, p) for p in price])
    candles = {"A": frame}
    events = {"A": _signal(frame, list(range(0, 300, 30)))}
    base = baseline_events(candles, step=25, seed=1)
    out = compare(events, candles, None, [Exit("만기", horizon=20)], baseline=base)
    assert "차이" in out.columns and "기준선 평균" in out.columns
    assert out["표본"].iloc[0] == 10


def test_by_year_splits_the_record():
    index = pd.date_range("2020-01-01", periods=800, freq="B", tz="Asia/Seoul")
    frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                          "close": 100.0, "volume": 1.0}, index=index)
    hit = pd.Series(False, index=index)
    hit.iloc[::20] = True
    out = by_year({"A": hit}, {"A": frame}, None, Exit("만기", horizon=10))
    assert len(out) >= 2 and set(out.columns) >= {"연도", "표본", "평균", "승률"}


def test_split_by_regime_uses_the_entry_date():
    index = pd.date_range("2020-01-01", periods=200, freq="B", tz="Asia/Seoul")
    frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                          "close": 100.0, "volume": 1.0}, index=index)
    hit = pd.Series(False, index=index)
    hit.iloc[[10, 100]] = True
    trades = resolve({"A": hit}, {"A": frame}, None, rule=Exit("t", horizon=10))
    regime = pd.Series(False, index=index)
    regime.iloc[:50] = True
    out = split_by_regime(trades, regime)
    assert list(out["표본"]) == [1, 1]


def test_insurance_breaks_even_where_premium_equals_payout():
    """보험료와 보험금이 같으면 손익분기는 50% 여야 한다."""
    index = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-0%d" % d, tz="Asia/Seoul") for d in (1, 2, 3, 6)])
    up = pd.Series([True, True, False, False], index=index)
    hold = pd.DataFrame({"신호일": index, "수익": [0.10, 0.10, -0.10, -0.10]})
    rule = pd.DataFrame({"신호일": index, "수익": [0.05, 0.05, -0.05, -0.05]})
    out = insurance_cost({"없음": hold, "규칙": rule}, up, hold_label="없음")
    row = out.iloc[0]
    assert row["보험료"] == pytest.approx(-0.05)
    assert row["보험금"] == pytest.approx(+0.05)
    assert row["손익분기 하락비중"] == pytest.approx(0.5)
    assert bool(row["보험 성립"])
    assert out.attrs["실제 하락비중"] == pytest.approx(0.5)


def test_a_rule_that_loses_in_both_regimes_is_never_insurance():
    """하락 구간에서도 더 나쁘면 보험금이 음수 — 손익분기가 1을 넘는다."""
    index = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-0%d" % d, tz="Asia/Seoul") for d in (1, 2, 3, 6)])
    up = pd.Series([True, True, False, False], index=index)
    hold = pd.DataFrame({"신호일": index, "수익": [0.10, 0.10, -0.10, -0.10]})
    rule = pd.DataFrame({"신호일": index, "수익": [0.05, 0.05, -0.20, -0.20]})
    out = insurance_cost({"없음": hold, "규칙": rule}, up, hold_label="없음")
    assert out.iloc[0]["보험금"] < 0
    assert not out.iloc[0]["보험 성립"]
    assert np.isnan(out.iloc[0]["손익분기 하락비중"])


def _panel(n=300, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B", tz="Asia/Seoul")
    out = {}
    for code in ("A", "B", "C"):
        price = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
        out[code] = pd.DataFrame(
            {"open": price, "high": price * 1.02, "low": price * 0.98,
             "close": price, "volume": 1000.0}, index=index)
    return out, index


def test_portfolio_respects_the_slot_limit():
    """자리가 1개면 같은 날 뜬 신호 3개 중 1개만 산다."""
    candles, index = _panel()
    events = {}
    for code in candles:
        hit = pd.Series(False, index=index)
        hit.iloc[10] = True
        events[code] = hit
    trades = resolve(events, candles, None, rule=Exit("t", horizon=20))
    _, stats = portfolio(trades, candles, calendar=index, max_positions=1)
    assert stats["매매 수"] == 1
    assert stats["자리없어 못산 신호"] == 2


def test_portfolio_rank_decides_who_gets_the_slot():
    candles, index = _panel(seed=5)
    events = {c: pd.Series(False, index=index) for c in candles}
    for c in events:
        events[c].iloc[10] = True
    trades = resolve(events, candles, None, rule=Exit("t", horizon=20))
    entry = trades["진입일"].iloc[0]
    rank = pd.Series(
        {("A", entry): 1.0, ("B", entry): 9.0, ("C", entry): 5.0})
    rank.index = pd.MultiIndex.from_tuples(rank.index)
    eq_b, _ = portfolio(trades, candles, calendar=index, max_positions=1, rank=rank)
    eq_a, _ = portfolio(trades, candles, calendar=index, max_positions=1)
    # B 가 1순위이므로 B 의 결과를 따라간다 — 순서만 따랐을 때와 달라야 한다
    assert not np.allclose(eq_a.to_numpy(), eq_b.to_numpy())


def test_portfolio_marks_open_positions_at_market():
    """진입가로만 평가하면 최대낙폭이 실제보다 작게 나온다."""
    index = pd.date_range("2020-01-01", periods=60, freq="B", tz="Asia/Seoul")
    price = np.r_[np.full(5, 100.0), np.linspace(100, 50, 30), np.full(25, 50.0)]
    frame = pd.DataFrame({"open": price, "high": price, "low": price,
                          "close": price, "volume": 1.0}, index=index)
    hit = pd.Series(False, index=index)
    hit.iloc[2] = True
    trades = resolve({"A": hit}, {"A": frame}, None, rule=Exit("t", horizon=50))
    _, stats = portfolio(trades, {"A": frame}, calendar=index, max_positions=1)
    assert stats["최대낙폭"] < -0.30      # 보유 중 반토막이 잔고에 보여야 한다


def test_portfolio_frees_the_slot_after_the_exit():
    candles, index = _panel(seed=9)
    hit = pd.Series(False, index=index)
    hit.iloc[[10, 40]] = True
    trades = resolve({"A": hit}, candles, None, rule=Exit("t", horizon=20))
    _, stats = portfolio(trades, candles, calendar=index, max_positions=1)
    assert stats["매매 수"] == 2 and stats["자리없어 못산 신호"] == 0
