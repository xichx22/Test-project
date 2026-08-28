import pandas as pd
import pytest

from tsignal.datasource.naver import NaverDataError, parse_sise
from tsignal.ohlcv import OhlcvError, repair, validate

SAMPLE = """
 [['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'],
["20250102", 52700, 53600, 52300, 53400, 16630538, 50.45],
["20250103", 52800, 55100, 52800, 54400, 19318046, 50.47],
["20250106", 0, 0, 0, 0, 0, 50.5]]
"""


def test_parse_sise_reads_naver_literal_format():
    df = parse_sise(SAMPLE)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2                      # 거래정지(전부 0)인 행은 버린다
    assert df["close"].iloc[0] == 53400
    assert str(df.index.tz) == "Asia/Seoul"


def test_parse_sise_rejects_garbage():
    with pytest.raises(NaverDataError):
        parse_sise("<html>error</html>")


def _frame(rows):
    idx = pd.date_range("2024-01-02", periods=len(rows), freq="D", tz="Asia/Seoul")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx).astype(float)


def test_repair_fixes_rounding_artifact():
    """수정주가 반올림으로 종가가 고가를 1원 넘긴 실제 사례(삼성SDI 2022-11-01)."""
    df = _frame([[719589, 744064, 716652, 744065, 416088]])
    fixed, log = repair(df)
    assert len(log) == 1
    assert fixed["high"].iloc[0] == 744065
    validate(fixed)                          # 보정 후에는 검증을 통과해야 한다


def test_repair_leaves_large_violations_for_validate_to_catch():
    """오차가 크면 계산 아티팩트가 아니라 잘못된 데이터다 — 손대지 않는다."""
    df = _frame([[100, 105, 95, 130, 1000]])
    fixed, log = repair(df, tolerance=0.001)
    assert log.empty
    assert fixed["high"].iloc[0] == 105
    with pytest.raises(OhlcvError):
        validate(fixed)
