"""네이버 금융 투자자별 매매동향 (기관·외국인 순매매량).

    https://finance.naver.com/item/frgn.naver?code=005930&page=1

한 페이지에 20영업일. 컬럼은
    날짜 · 종가 · 전일비 · 등락률 · 거래량 ·
    **기관 순매매량** · **외국인 순매매량** · 외국인 보유주수 · 외국인 보유율

siseJson 의 `외국인소진율` 은 소수점 2자리로 반올림돼 있어 하루치 변화에
반올림 잡음이 크다. 여기서는 **실제 순매매 주식 수**를 받으므로 훨씬 깨끗하다.

HTML 이지만 `lxml` 의존을 더하지 않으려고 정규식으로 파싱한다. 표 구조가
단순하고(한 행에 td 9개), 날짜 패턴으로 데이터 행을 특정할 수 있어 충분하다.
구조가 바뀌면 `parse_flow_page()` 하나만 고치면 된다.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests

from ..ohlcv import KST
from .base import Interval
from .naver import HEADERS

FLOW_URL = "https://finance.naver.com/item/frgn.naver"
ROWS_PER_PAGE = 20

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")
_DATE = re.compile(r"\d{4}\.\d{2}\.\d{2}")

COLUMNS = ["inst_net", "foreign_net", "foreign_shares", "foreign_rate"]


class NaverFlowError(RuntimeError):
    pass


def _number(text: str) -> float:
    cleaned = text.replace(",", "").replace("%", "").replace("+", "").strip()
    if not cleaned or cleaned == "-":
        return float("nan")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def parse_flow_page(html: str) -> pd.DataFrame:
    """frgn.naver 한 페이지 → DataFrame[inst_net, foreign_net, foreign_shares, foreign_rate]."""
    records = []
    for row in _ROW.findall(html):
        if not _DATE.search(row):
            continue
        cells = [_TAG.sub("", c).replace("\xa0", " ").strip() for c in _CELL.findall(row)]
        if len(cells) < 9:
            continue
        records.append({
            "dt": pd.to_datetime(cells[0], format="%Y.%m.%d"),
            "inst_net": _number(cells[5]),
            "foreign_net": _number(cells[6]),
            "foreign_shares": _number(cells[7]),
            "foreign_rate": _number(cells[8]),
        })
    if not records:
        return pd.DataFrame(columns=["dt", *COLUMNS]).set_index("dt")

    frame = pd.DataFrame(records).set_index("dt").sort_index()
    frame.index = frame.index.tz_localize(KST)
    return frame


class NaverFlowSource:
    """투자자별 매매동향 수집기."""

    def __init__(self, *, timeout: float = 20.0, retries: int = 3,
                 min_interval_sec: float = 0.5) -> None:
        self.timeout = timeout
        self.retries = retries
        self.min_interval_sec = min_interval_sec
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval_sec:
            time.sleep(self.min_interval_sec - gap)
        self._last_call = time.monotonic()

    def page(self, code: str, page: int) -> pd.DataFrame:
        last: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self._session.get(
                    FLOW_URL, params={"code": code, "page": page}, timeout=self.timeout
                )
                resp.raise_for_status()
                return parse_flow_page(resp.content.decode("euc-kr", errors="replace"))
            except requests.RequestException as exc:
                last = exc
                time.sleep(2**attempt)
        raise NaverFlowError(f"{code} page={page} 실패") from last

    def flow(self, code: str, *, count: int = 1200) -> pd.DataFrame:
        """최근 count 영업일치 수급. 페이지를 이어 받다가 빈 페이지가 나오면 멈춘다."""
        pages = -(-count // ROWS_PER_PAGE)
        frames = []
        for page in range(1, pages + 1):
            frame = self.page(code, page)
            if frame.empty:
                break
            frames.append(frame)
        if not frames:
            raise NaverFlowError(f"{code}: 수급 데이터가 없습니다.")
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="first")].sort_index()
        return out.tail(count)


def flow_features(flow: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
    """순매매량 → 비교 가능한 팩터.

    순매매 '주식 수' 는 종목마다 스케일이 달라 횡단면 비교가 안 된다.
    그날 거래량으로 나눠 **거래량 대비 순매수 비중**으로 만든다 (-1 ~ +1).
    """
    # 거래정지일은 거래량 0 → 0으로 나누지 않도록 NaN 으로 돌린다.
    # (pd.NA 를 쓰면 object dtype 이 되어 float 연산이 깨진다.)
    volume = volume.reindex(flow.index).astype("float64").replace(0.0, np.nan)
    inst = flow["inst_net"].astype("float64") / volume
    foreign = flow["foreign_net"].astype("float64") / volume

    out = pd.DataFrame(index=flow.index)
    out["inst_ratio"] = inst
    out["foreign_ratio"] = foreign
    out["combined_ratio"] = inst + foreign
    for span in (5, 20):
        out[f"inst_ratio_{span}"] = inst.rolling(span, min_periods=span).mean()
        out[f"foreign_ratio_{span}"] = foreign.rolling(span, min_periods=span).mean()
        out[f"combined_ratio_{span}"] = (inst + foreign).rolling(span, min_periods=span).mean()
    out["foreign_rate"] = flow["foreign_rate"]
    return out
