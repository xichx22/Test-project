"""미래참조 검사 — 이 프로젝트에서 가장 중요한 테스트.

검증 결과가 아무리 좋아도 미래를 한 칸이라도 훔쳐봤다면 전부 무의미하다.
두 방향으로 막는다.

  1. 기계적 검사: 데이터를 t 봉에서 잘라 계산한 값이, 전체 데이터로 계산한
     t 봉 값과 같아야 한다. 다르면 그 지표/신호는 미래를 보고 있다.
  2. 정적 검사: 지표/신호 소스에 음수 shift 가 있으면 실패시킨다.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsignal import indicators as ind
from tsignal import signals as sig
from tsignal.datasource import Interval

CUTS = (900, 1400, 1900)


def _last_rows(frame: pd.DataFrame, at: int) -> pd.Series:
    return frame.iloc[at - 1]


def test_indicators_are_causal(candles_5m):
    full = ind.compute_all(candles_5m, interval=Interval.M5)
    for cut in CUTS:
        truncated = ind.compute_all(candles_5m.iloc[:cut], interval=Interval.M5)
        a, b = _last_rows(full, cut), _last_rows(truncated, cut)
        offenders = [
            col for col in full.columns
            if not (pd.isna(a[col]) and pd.isna(b[col]))
            and not np.isclose(a[col], b[col], rtol=1e-9, atol=1e-9, equal_nan=True)
        ]
        assert not offenders, f"cut={cut} 에서 미래참조 의심 지표: {offenders}"


def test_signals_are_causal(candles_5m):
    full_feat = ind.compute_all(candles_5m, interval=Interval.M5)
    full = sig.evaluate_all(candles_5m, full_feat)
    for cut in CUTS:
        sub = candles_5m.iloc[:cut]
        truncated = sig.evaluate_all(sub, ind.compute_all(sub, interval=Interval.M5))
        a, b = _last_rows(full, cut), _last_rows(truncated, cut)
        offenders = [col for col in full.columns if a[col] != b[col]]
        assert not offenders, f"cut={cut} 에서 미래참조 의심 신호: {offenders}"


@pytest.mark.parametrize("package", ["indicators", "signals"])
def test_no_negative_shift_in_source(package):
    """지표/신호 코드에는 음수 shift 가 있으면 안 된다.

    (레이블을 만드는 evaluation/forward.py 는 의도적으로 음수 shift 를 쓰므로 제외한다.)
    """
    root = Path(__file__).resolve().parents[1] / "src" / "tsignal" / package
    pattern = re.compile(r"shift\(\s*-")
    offenders = [
        f"{path.name}:{line_no}"
        for path in root.rglob("*.py")
        for line_no, code in _code_lines(path)
        if pattern.search(code)
    ]
    assert not offenders, f"음수 shift 발견: {offenders}"


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """주석과 문자열/독스트링을 걷어낸 실행 코드만 돌려준다."""
    source = path.read_text(encoding="utf-8")
    drop = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT}
    lines: dict[int, list[str]] = {}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in drop:
            continue
        lines.setdefault(tok.start[0], []).append(tok.string)
    return [(no, "".join(parts)) for no, parts in sorted(lines.items())]
