"""유니버스 구성 — 시가총액 상위 종목 목록을 가져온다.

네이버 금융 시가총액 페이지를 읽는다 (인증 불필요, 페이지당 50종목).
    https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page=N
    sosok=0 코스피 / sosok=1 코스닥

생존 편향에 대하여
-----------------
이 목록은 **현재 상장된 종목만** 담는다. 상장폐지된 종목은 애초에 페이지에
나오지 않으므로, 유니버스를 아무리 넓혀도 생존 편향은 남는다.
넓히면 표본과 검정력이 늘어날 뿐, 편향이 사라지지는 않는다.
편향까지 없애려면 상장폐지 종목의 과거 시세가 필요하고, 그건 공개
엔드포인트로는 얻기 어렵다 — 유료 벤더나 KRX 정보데이터시스템이 필요하다.
"""

from __future__ import annotations

import re
import time

import pandas as pd
import requests

from .base import Symbol
from .naver import HEADERS

LISTING_URL = "https://finance.naver.com/sise/sise_market_sum.naver"
MARKETS = {"KOSPI": 0, "KOSDAQ": 1}

_ROW = re.compile(r'<a href="/item/main\.naver\?code=(\d{6})"[^>]*class="tltle"[^>]*>([^<]+)</a>')

# 우선주·스팩·리츠는 보통주와 성격이 달라 기술적 신호 검증 대상에서 뺀다.
_EXCLUDE_NAME = re.compile(r"스팩|리츠|우B$|우C$")


class UniverseError(RuntimeError):
    pass


def fetch_listing(
    market: str = "KOSPI",
    *,
    top_n: int = 200,
    include_preferred: bool = False,
    min_interval_sec: float = 0.4,
    timeout: float = 20.0,
) -> list[Symbol]:
    """시가총액 상위 top_n 종목. 순서는 시총 내림차순."""
    if market not in MARKETS:
        raise UniverseError(f"알 수 없는 시장 '{market}'. 사용 가능: {sorted(MARKETS)}")

    session = requests.Session()
    session.headers.update(HEADERS)
    out: list[Symbol] = []
    seen: set[str] = set()

    for page in range(1, (top_n // 50) * 2 + 4):     # 필터로 빠지는 만큼 여유 있게 더 읽는다
        if len(out) >= top_n:
            break
        resp = session.get(
            LISTING_URL, params={"sosok": MARKETS[market], "page": page}, timeout=timeout
        )
        if resp.status_code != 200:
            raise UniverseError(f"HTTP {resp.status_code} (page={page})")
        html = resp.content.decode("euc-kr", errors="replace")

        matches = _ROW.findall(html)
        if not matches:
            break                                     # 마지막 페이지를 넘어섰다
        for code, name in matches:
            name = name.strip()
            if code in seen:
                continue
            # 보통주 코드는 끝자리가 0. 5/7/9 로 끝나면 우선주다.
            if not include_preferred and not code.endswith("0"):
                continue
            if _EXCLUDE_NAME.search(name):
                continue
            seen.add(code)
            out.append(Symbol(code=code, name=name, market=market))
            if len(out) >= top_n:
                break
        time.sleep(min_interval_sec)

    return out


def fetch_universe(
    *, kospi: int = 150, kosdaq: int = 50, **kwargs
) -> list[Symbol]:
    """코스피 + 코스닥 시총 상위를 합친 유니버스."""
    symbols: list[Symbol] = []
    if kospi:
        symbols += fetch_listing("KOSPI", top_n=kospi, **kwargs)
    if kosdaq:
        symbols += fetch_listing("KOSDAQ", top_n=kosdaq, **kwargs)
    return symbols


def to_frame(symbols: list[Symbol]) -> pd.DataFrame:
    return pd.DataFrame([{"code": s.code, "name": s.name, "market": s.market} for s in symbols])
