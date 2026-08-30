"""매매 데스크 — 수량·차단기·게이트·기록·리뷰."""

from datetime import date, timedelta

import pytest

from tsignal.desk import (
    Answers, Fill, Ledger, Thresholds,
    evaluate_checklist, evaluate_guard, review, size_order,
)
from tsignal.desk.checklist import BLOCK, PASS, REVIEW
from tsignal.desk.guard import ALLOWED, HALTED
from tsignal.desk.sizing import round_to_tick


# --------------------------------------------------------------------- 수량
def test_stop_price_lands_on_a_tick():
    """호가 사이 가격으로 주문하면 체결되지 않는다."""
    out = size_order(10_000_000, 107_500, stop_loss=0.08)
    assert out.stop_price % 100 == 0        # 5만~20만원대 호가 단위 100원
    assert out.stop_price < 107_500


def test_stop_rounds_down_not_up():
    """올림하면 의도보다 일찍 잘린다."""
    assert round_to_tick(98_912, mode="down") == 98_900
    assert round_to_tick(98_912, mode="up") == 99_000


def test_slot_and_risk_are_both_checked():
    """손절이 아주 넓으면 자리가 아니라 위험 한도가 수량을 정해야 한다."""
    tight = size_order(10_000_000, 50_000, stop_loss=0.05, max_risk_pct=0.02)
    wide = size_order(10_000_000, 50_000, stop_loss=0.50, max_risk_pct=0.02)
    assert tight.binding == "자리"
    assert wide.binding == "손절"
    assert wide.shares < tight.shares


def test_more_slots_means_smaller_orders():
    five = size_order(10_000_000, 50_000, max_positions=5)
    twenty = size_order(10_000_000, 50_000, max_positions=20)
    assert five.shares > twenty.shares


def test_sizing_rejects_nonsense():
    with pytest.raises(ValueError):
        size_order(0, 1000)
    with pytest.raises(ValueError):
        size_order(1_000_000, 1000, stop_loss=1.5)
    with pytest.raises(ValueError):
        size_order(1_000_000, 1000, max_positions=0)


# ------------------------------------------------------------------- 차단기
def _loss(pnl_ratio: float, account: float, when: date) -> Fill:
    """지정한 손실률이 나오는 청산 건 하나."""
    entry, shares = 10_000.0, int(account * abs(pnl_ratio) / (10_000.0 * 0.10))
    return Fill(code="000000", entry_date=str(when), entry_price=entry, shares=shares,
                exit_date=str(when), exit_price=entry * (1 + (-0.10 if pnl_ratio < 0 else 0.10)),
                exit_reason="stop")


def test_guard_allows_when_quiet():
    out = evaluate_guard([], 10_000_000)
    assert out.state == ALLOWED and out.can_trade


def test_guard_halts_on_daily_limit():
    today = date(2026, 9, 15)
    fills = [_loss(-0.06, 10_000_000, today)]
    out = evaluate_guard(fills, 10_000_000, as_of=today)
    assert out.state == HALTED
    assert any("일간" in r for r in out.reasons)


def test_guard_ignores_a_long_losing_streak():
    """연속 손실은 고장이 아니라 설계다 — 원본 기본값을 그대로 쓰면 안 된다.

    이 전략은 승률 34%, 최장 연속 손실 32회다. '연속 2패 쿨다운'을 걸면
    21년 실측에서 45번 발동해 사실상 영구 정지가 된다.
    """
    today = date(2026, 9, 15)
    fills = [Fill(code=f"{i:06d}", entry_date="2026-01-02", entry_price=10_000,
                  shares=1, exit_date="2026-01-05", exit_price=9_200,
                  exit_reason="stop") for i in range(20)]
    out = evaluate_guard(fills, 10_000_000, as_of=today)
    assert out.can_trade, "연속 손실만으로 차단되면 이 전략은 돌 수 없다"


def test_guard_halts_on_rule_breaks():
    """성과와 무관하게 규칙 위반 2회면 중단이다."""
    fills = [Fill(code="A", rule_breaks=["손절 미룸"]),
             Fill(code="B", rule_breaks=["자리 초과"])]
    out = evaluate_guard(fills, 10_000_000)
    assert out.state == HALTED
    assert any("규칙 위반" in r for r in out.reasons)


def test_guard_only_counts_realized_pnl():
    """평가손익으로 차단하면 버티는 것이 전부인 규칙과 충돌한다."""
    holding = [Fill(code="A", entry_date="2026-09-01", entry_price=10_000, shares=100)]
    assert evaluate_guard(holding, 10_000_000).can_trade


def test_guard_thresholds_are_configurable():
    today = date(2026, 9, 15)
    fills = [_loss(-0.03, 10_000_000, today)]
    assert evaluate_guard(fills, 10_000_000, as_of=today).can_trade
    strict = Thresholds(daily=0.02)
    assert not evaluate_guard(fills, 10_000_000, as_of=today, thresholds=strict).can_trade


# ------------------------------------------------------------------- 게이트
def _full_answers(**kw):
    base = dict(written_plan=True, stop_defined=True, size_within_plan=True,
                gate_open=True, slots_free=3, already_held=False,
                planned_risk=80_000.0, actual_risk=77_000.0)
    base.update(kw)
    return Answers(**base)


def test_checklist_passes_when_everything_answered():
    assert evaluate_checklist(_full_answers()).decision == PASS


def test_unanswered_is_review_not_pass():
    """확인할 수 없으면 통과가 아니라 차단이다 (fail-closed)."""
    out = evaluate_checklist(Answers())
    assert out.decision == REVIEW
    assert not out.ok


@pytest.mark.parametrize("field,value,word", [
    ("written_plan", False, "계획"),
    ("stop_defined", False, "손절"),
    ("gate_open", False, "게이트"),
    ("already_held", True, "중복"),
    ("slots_free", 0, "자리"),
])
def test_checklist_blocks_each_violation(field, value, word):
    out = evaluate_checklist(_full_answers(**{field: value}))
    assert out.decision == BLOCK
    assert any(word in r for r in out.reasons)


def test_checklist_blocks_oversized_risk():
    out = evaluate_checklist(_full_answers(actual_risk=120_000.0))
    assert out.decision == BLOCK
    assert any("위험" in r for r in out.reasons)


def test_checklist_respects_the_guard():
    guard = evaluate_guard([Fill(code="A", rule_breaks=["x", "y"])], 10_000_000)
    out = evaluate_checklist(_full_answers(), guard)
    assert out.decision == BLOCK


# --------------------------------------------------------------------- 기록
def test_ledger_round_trip(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.add(Fill(code="105560", name="KB금융", entry_date="2026-09-01",
                 entry_price=107_500, shares=9, stop_price=98_900))
    assert len(led.open_positions()) == 1
    led.close("105560", exit_date="2026-09-20", exit_price=120_000, exit_reason="time")
    done = led.closed()[0]
    assert done.gross_return == pytest.approx(120_000 / 107_500 - 1)
    assert done.net_return() < done.gross_return


def test_ledger_blocks_duplicate_entry():
    """같은 종목을 두 자리에 담으면 분산이 깨진다 — 보통은 주문 실수다."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(f"{d}/l.jsonl")
        led.add(Fill(code="105560", entry_price=100, shares=1))
        with pytest.raises(ValueError, match="이미 보유"):
            led.add(Fill(code="105560", entry_price=100, shares=1))


def test_ledger_reports_the_broken_line(tmp_path):
    """한 줄이 깨졌을 때 조용히 넘어가면 안 된다."""
    path = tmp_path / "l.jsonl"
    path.write_text('{"code": "A", "entry_price": 1}\n{ broken\n')
    with pytest.raises(ValueError, match="2"):
        Ledger(path).load()


def test_closing_an_unopened_position_raises(tmp_path):
    with pytest.raises(ValueError):
        Ledger(tmp_path / "l.jsonl").close("999999", exit_date="2026-09-01",
                                           exit_price=1, exit_reason="stop")


# --------------------------------------------------------------------- 리뷰
def _closed(ret: float, day: str = "2026-09-10") -> Fill:
    return Fill(code="A", entry_date="2026-09-01", entry_price=10_000, shares=1,
                exit_date=day, exit_price=10_000 * (1 + ret), exit_reason="stop")


def test_review_flags_a_small_sample():
    out = review([_closed(0.2), _closed(-0.08)])
    assert out.n_trades == 2
    assert any("표본" in f for f in out.flags)


def test_review_counts_rule_breaks_even_on_open_positions():
    fills = [Fill(code="A", rule_breaks=["손절 미룸"]), _closed(-0.08)]
    assert review(fills).rule_breaks == 1


def test_review_measures_the_losing_streak():
    fills = [_closed(-0.08, f"2026-09-{d:02d}") for d in range(1, 6)]
    fills.append(_closed(0.2, "2026-09-06"))
    assert review(fills).longest_loss_streak == 5


def test_review_handles_no_closed_trades():
    out = review([Fill(code="A", entry_price=1, shares=1)])
    assert out.n_trades == 0
    assert "청산된 매매 없음" in out.report()
