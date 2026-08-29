"""스윙 순위표 엔진 검증 — 특히 체결 가정."""

import numpy as np
import pandas as pd
import pytest

from tsignal.datasource.base import Interval
from tsignal.evaluation.swinglab import BARE, SwingLab


def _candles(seed: int, n: int = 400, gap: float = 0.0) -> pd.DataFrame:
    """일봉 생성. `gap` 을 주면 매일 전일종가 대비 시가가 그만큼 뛴다."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B", tz="Asia/Seoul")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    prev = np.r_[close[0], close[:-1]]
    open_ = prev * (1 + gap)
    high = np.maximum(close, open_) * 1.01
    low = np.minimum(close, open_) * 0.99
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(1_000, 50_000, n).astype(float)},
        index=index,
    )


@pytest.fixture(scope="module")
def lab() -> SwingLab:
    data = {f"{i:06d}": _candles(i) for i in range(1, 7)}
    return SwingLab(data, interval=Interval.D1)


def test_panel_shapes_line_up(lab):
    n_d, n_c = len(lab.dates), len(lab.codes)
    assert lab.ret.shape == (n_d, n_c)
    assert lab.entry_ret.shape == (n_d, n_c)
    assert lab.valid.shape == (n_d, n_c)
    assert lab.trigger_names and lab.filter_names
    for block in lab.trig.values():
        assert block.shape == (n_d, n_c)


def test_entry_day_uses_open_not_previous_close():
    """체결일 수익률은 시가→종가여야 한다.

    종가→종가를 쓰면 신호 다음날 **시가 갭까지** 먹는다. 그건 사기 전에
    벌어진 움직임이라 실제로는 얻을 수 없다. 매일 +2% 갭이 나도록 만든
    데이터에서 그 갭이 성과에 들어오면 안 된다.
    """
    data = {f"{i:06d}": _candles(i, gap=0.02) for i in range(1, 5)}
    lab = SwingLab(data, interval=Interval.D1)
    # 시가→종가 수익률에는 갭이 들어 있지 않아야 한다.
    assert lab.entry_ret[lab.valid].mean() < lab.ret[lab.valid].mean()
    trigger = lab.trigger_names[0]
    out = lab.run(trigger, holding=5, cost_bps=0.0)
    # 갭을 먹었다면 종가-종가만 쓴 판보다 성과가 낮아야 정상이다.
    naive = lab.entry_ret.copy()
    lab.entry_ret = lab.ret.copy()           # 옛 (틀린) 가정 재현
    try:
        wrong = lab.run(trigger, holding=5, cost_bps=0.0)
    finally:
        lab.entry_ret = naive
    assert out.result.cagr < wrong.result.cagr


def test_entry_is_delayed_by_one_bar(lab):
    """신호 봉 당일에는 보유가 아니어야 한다 (미래참조 방지)."""
    trigger = lab.trigger_names[0]
    entry = lab.trig[trigger]
    held, enter = lab._held(entry, holding=1)
    assert not (enter[0]).any(), "첫 봉에 진입이 있으면 하루를 당겨온 것이다"
    assert np.array_equal(enter[1:], entry[:-1] & lab.valid[1:])


def test_holding_window_length_is_respected(lab):
    """h일 보유면 진입 후 정확히 h봉 동안만 보유여야 한다."""
    entry = np.zeros_like(lab.valid)
    entry[10, 0] = True
    held, _ = lab._held(entry, holding=4)
    column = held[:, 0]
    assert column[:11].sum() == 0
    assert column[11:15].all()
    assert column[15:].sum() == 0


def test_cost_reduces_return(lab):
    trigger = lab.trigger_names[0]
    free = lab.run(trigger, holding=10, cost_bps=0.0).result
    paid = lab.run(trigger, holding=10, cost_bps=28.0).result
    assert paid.cagr < free.cagr


def test_max_positions_caps_concurrent_holdings(lab):
    trigger = lab.trigger_names[0]
    capped = lab.run(trigger, holding=60, max_positions=2).result
    uncapped = lab.run(trigger, holding=60).result
    assert capped.trades <= uncapped.trades


def test_slice_keeps_every_block_aligned(lab):
    cut = lab.dates[len(lab.dates) // 2]
    first = lab.slice(end=cut)
    assert len(first.dates) < len(lab.dates)
    assert first.ret.shape[0] == len(first.dates)
    assert first.entry_ret.shape[0] == len(first.dates)
    for block in first.trig.values():
        assert block.shape[0] == len(first.dates)


def test_slice_accepts_a_tz_aware_timestamp(lab):
    """이미 tz 가 붙은 Timestamp 를 넘겨도 터지지 않아야 한다."""
    cut = lab.dates[10]
    assert cut.tzinfo is not None
    assert len(lab.slice(start=cut).dates) == len(lab.dates) - 10


def test_leaderboard_is_sorted_and_filtered(lab):
    board = lab.leaderboard(holdings=(10,), min_trades=1, min_exposure=0.0,
                            triggers=lab.trigger_names[:3])
    assert not board.empty
    assert board["연수익"].is_monotonic_decreasing
    assert (board["매매횟수"] >= 1).all()
    assert BARE in set(board["필터"])


def test_leaderboard_drops_thin_rules(lab):
    """매매가 거의 없는 규칙은 순위표에 올라오면 안 된다."""
    board = lab.leaderboard(holdings=(10,), min_trades=10_000_000,
                            min_exposure=0.0, triggers=lab.trigger_names[:3])
    assert board.empty


def test_benchmark_holds_everything(lab):
    bench = lab.benchmark()
    assert bench.exposure == pytest.approx(1.0)
    assert len(bench.equity) == len(lab.dates)
