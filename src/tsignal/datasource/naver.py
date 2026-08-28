"""네이버 금융 공개 시세 어댑터 (일/주/월봉).

`pykrx` 1.2.x 는 KRX 로그인 자격증명을 요구하도록 바뀌어 익명 수집이 막혔다.
네이버 금융의 `siseJson.naver` 는 인증 없이 일봉 OHLCV + 외국인소진율을 준다.

    GET https://api.finance.naver.com/siseJson.naver
        ?symbol=005930&requestType=1&startTime=20250101&endTime=20250120&timeframe=day

응답은 순수 JSON 이 아니라 파이썬/JS 리터럴에 가까운 형태다(헤더 행이 홑따옴표).
`ast.literal_eval` 로 안전하게 파싱한다 — `eval` 은 쓰지 않는다.

분봉은 이 엔드포인트가 제공하지 않는다. 분봉이 필요해지면 별도 어댑터를 붙인다.
"""

from __future__ import annotations

import ast
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from ..ohlcv import KST, repair, validate
from .base import DataSource, Interval, Symbol

BASE_URL = "https://api.finance.naver.com/siseJson.naver"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}

_TIMEFRAME = {Interval.D1: "day"}


class NaverDataError(RuntimeError):
    pass


def parse_sise(text: str) -> pd.DataFrame:
    """siseJson 응답 → OHLCV 규격."""
    try:
        rows = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError) as exc:
        raise NaverDataError(f"응답을 파싱하지 못했습니다: {text[:200]}") from exc
    if not isinstance(rows, list) or len(rows) < 2:
        raise NaverDataError(f"데이터 행이 없습니다: {text[:200]}")

    header, *body = rows
    frame = pd.DataFrame(body, columns=list(header))
    frame = frame.rename(columns={
        "날짜": "dt", "시가": "open", "고가": "high", "저가": "low",
        "종가": "close", "거래량": "volume", "외국인소진율": "foreign_rate",
    })
    frame["dt"] = pd.to_datetime(frame["dt"], format="%Y%m%d").dt.tz_localize(KST)
    frame = frame.set_index("dt").sort_index()

    ohlcv = frame[["open", "high", "low", "close", "volume"]].astype("float64")
    # 거래정지일은 시가/고가/저가가 0으로 내려오므로 버린다.
    ohlcv = ohlcv[(ohlcv[["open", "high", "low", "close"]] > 0).all(axis=1)]
    ohlcv.index.name = "dt"
    # 수정주가 반올림으로 종가가 고가를 1원 넘기는 행이 섞여 나온다 → 보정 후 검증.
    ohlcv, fixed = repair(ohlcv)
    if not fixed.empty:
        ohlcv.attrs["repaired_rows"] = len(fixed)
    return validate(ohlcv)


class NaverDataSource(DataSource):
    """네이버 금융 일봉 소스."""

    name = "naver"

    def __init__(self, *, timeout: float = 15.0, retries: int = 3, min_interval_sec: float = 0.3) -> None:
        self.timeout = timeout
        self.retries = retries
        self.min_interval_sec = min_interval_sec
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    def supports(self, interval: Interval) -> bool:
        return interval in _TIMEFRAME

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval_sec:
            time.sleep(self.min_interval_sec - gap)
        self._last_call = time.monotonic()

    def candles(
        self,
        code: str,
        interval: Interval,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        if not self.supports(interval):
            raise NaverDataError(
                f"네이버 소스는 {interval.value} 를 제공하지 않습니다 (일봉만 지원). "
                "분봉이 필요하면 다른 어댑터를 쓰세요."
            )
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
        if start is not None:
            start_ts = pd.Timestamp(start)
        else:
            # 영업일은 달력일의 약 68% → 여유 있게 잡아 받고 뒤에서 잘라낸다.
            days = int((count or 1000) * 1.6) + 30
            start_ts = end_ts - timedelta(days=days)

        params = {
            "symbol": code.lstrip("Aa") if code[:1].upper() == "A" else code,
            "requestType": 1,
            "startTime": start_ts.strftime("%Y%m%d"),
            "endTime": end_ts.strftime("%Y%m%d"),
            "timeframe": _TIMEFRAME[interval],
        }

        last: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self._session.get(BASE_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                df = parse_sise(resp.text)
                return df.tail(count) if count else df
            except (requests.RequestException, NaverDataError) as exc:
                last = exc
                time.sleep(2**attempt)
        raise NaverDataError(f"{self.retries}회 재시도 실패: {code}") from last

    def symbol(self, code: str) -> Symbol:
        return Symbol(code=code)
