"""토스증권 웹(WTS) 엔드포인트 어댑터.

주의 — 실측 결과(2026-08 기준)
----------------------------
토스증권은 공개 문서화된 REST API를 제공하지 않는다. 아래는 웹 클라이언트가
쓰는 내부 엔드포인트를 직접 확인한 결과다.

  익명 호출로 동작 확인됨
    GET /api/v2/stock-infos/{A코드}     종목 메타 (이름/시장/ISIN/로고 등)
    GET /api/v2/stock-prices/{A코드}    현재가 스냅샷 (시/고/저/종/거래량/52주)

  경로는 존재하나 익명 호출은 400 으로 거절됨
    GET /api/v1/c-chart/kr-stock/{A코드}/{period}    과거 캔들
      - 존재하지 않는 경로는 404 를 주는데 이 경로는 400 을 준다 →
        라우트는 맞고 인증/필수 파라미터가 빠진 상태.
      - period 후보(day/1D/minute/min:1 …)와 파라미터 후보
        (count/to/from/size/interval/…)를 조합 검증했으나 전부 400.
      - 결론: 브라우저 세션 헤더(쿠키/디바이스 ID)가 필요할 가능성이 높다.

그래서 이 어댑터는
  1) 검증된 엔드포인트는 그대로 제공하고,
  2) 캔들은 `session_headers` 로 브라우저 헤더를 주입할 수 있게 열어두고,
  3) 파라미터 재발견용 `probe_candle_endpoint()` 를 함께 둔다.
스키마가 바뀌어도 여기 파일 하나만 고치면 된다.

비공식 엔드포인트다. 서비스 약관/robots 를 확인하고 본인 책임으로 쓰되,
호출 간격(`min_interval_sec`)을 지켜 서버에 부담을 주지 말 것.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from ..ohlcv import normalize, validate
from .base import DataSource, Interval, Symbol

BASE_URL = "https://wts-info-api.tossinvest.com"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://tossinvest.com/",
    "Origin": "https://tossinvest.com",
}

# c-chart 경로가 쓰는 기간 토큰 추정치. 확정되면 여기만 고친다.
_PERIOD_TOKEN = {
    Interval.M1: "minute:1",
    Interval.M3: "minute:3",
    Interval.M5: "minute:5",
    Interval.M15: "minute:15",
    Interval.M30: "minute:30",
    Interval.H1: "minute:60",
    Interval.D1: "day",
}


class TossApiError(RuntimeError):
    pass


class TossCandlesUnavailable(TossApiError):
    """캔들 엔드포인트가 익명 호출을 거절함 — 세션 헤더 주입이 필요하다."""


def _to_a_code(code: str) -> str:
    return code if code.upper().startswith("A") else f"A{code}"


def load_session(path: str | Path) -> dict[str, Any]:
    """저장해 둔 세션(헤더/쿠키) 파일을 읽는다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"session_headers": data.get("headers", {}), "cookies": data.get("cookies", {})}


def save_session(path: str | Path, *, headers: dict[str, str], cookies: dict[str, str],
                 endpoint: dict[str, Any] | None = None) -> Path:
    """확인된 세션과 엔드포인트 정보를 파일로 남긴다.

    쿠키에는 계정 세션 토큰이 들어갈 수 있다. 이 파일은 커밋하지 말 것
    (`.gitignore` 에 `*.session.json` 이 등록돼 있다).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"headers": headers, "cookies": cookies, "endpoint": endpoint or {}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


class TossClient:
    """얇은 HTTP 래퍼. 재시도 + 호출 간격만 책임진다."""

    def __init__(
        self,
        *,
        session_headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        timeout: float = 15.0,
        retries: int = 3,
        min_interval_sec: float = 0.25,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.min_interval_sec = min_interval_sec
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        if session_headers:
            self._session.headers.update(session_headers)
        if cookies:
            self._session.cookies.update(cookies)

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval_sec:
            time.sleep(self.min_interval_sec - gap)
        self._last_call = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{BASE_URL}{path}"
        last: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last = exc
                time.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                last = TossApiError(f"{resp.status_code} {url}")
                time.sleep(2**attempt)
                continue
            raise TossApiError(f"HTTP {resp.status_code} {url} :: {resp.text[:200]}")
        raise TossApiError(f"{self.retries}회 재시도 실패: {url}") from last

    # --- 검증된 엔드포인트 -------------------------------------------------
    def stock_info(self, code: str) -> dict[str, Any]:
        return self.get(f"/api/v2/stock-infos/{_to_a_code(code)}")["result"]

    def price_snapshot(self, code: str) -> dict[str, Any]:
        return self.get(f"/api/v2/stock-prices/{_to_a_code(code)}")["result"]

    # --- 미확정 엔드포인트 -------------------------------------------------
    def raw_candles(self, code: str, period: str, **params: Any) -> Any:
        return self.get(f"/api/v1/c-chart/kr-stock/{_to_a_code(code)}/{period}", params)

    def replay(self, request: Any) -> tuple[int, str]:
        """cURL 에서 파싱한 요청을 그대로 재현한다."""
        self._throttle()
        resp = self._session.request(
            request.method, request.base_url(), params=request.params,
            headers=request.headers, cookies=request.cookies,
            data=request.data, timeout=self.timeout,
        )
        return resp.status_code, resp.text

    def minimal_params(self, request: Any) -> dict[str, Any]:
        """파라미터를 하나씩 빼보며 정말 필요한 것만 추린다.

        브라우저는 쓰지도 않는 파라미터를 딸려 보내는 경우가 많다.
        무엇이 필수인지 알아야 어댑터를 안정적으로 짤 수 있다.
        """
        required: dict[str, Any] = {}
        params = dict(request.params)
        for key in list(params):
            trimmed = {k: v for k, v in params.items() if k != key}
            self._throttle()
            resp = self._session.request(
                request.method, request.base_url(), params=trimmed,
                headers=request.headers, cookies=request.cookies, timeout=self.timeout,
            )
            if resp.status_code != 200:
                required[key] = params[key]
        return required

    def probe_candle_endpoint(
        self,
        code: str = "005930",
        periods: Iterable[str] = ("day", "1D", "minute:1", "minute"),
        param_sets: Iterable[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """period/파라미터 조합을 훑어 200을 주는 조합을 찾는다.

        헤더를 바꿔가며 이걸 돌리면 스키마 확정 없이도 재발견이 가능하다.
        반환값의 각 항목: {period, params, status, body_head}
        """
        if param_sets is None:
            param_sets = [
                {"count": 10},
                {"count": 10, "to": datetime.now().strftime("%Y-%m-%d")},
                {"size": 10},
                {},
            ]
        results: list[dict[str, Any]] = []
        for period in periods:
            for params in param_sets:
                url = f"{BASE_URL}/api/v1/c-chart/kr-stock/{_to_a_code(code)}/{period}"
                self._throttle()
                try:
                    resp = self._session.get(url, params=params, timeout=self.timeout)
                    body, status = resp.text[:300], resp.status_code
                except requests.RequestException as exc:
                    body, status = str(exc)[:300], -1
                results.append(
                    {"period": period, "params": params, "status": status, "body_head": body}
                )
        return results


def parse_candles(payload: Any) -> pd.DataFrame:
    """토스 캔들 응답 → OHLCV 규격.

    응답 스키마가 확정되지 않았으므로 흔한 키 이름을 폭넓게 받아들인다.
    실제 응답을 확보하면 여기를 좁히면 된다.
    """
    rows = payload
    for key in ("result", "candles", "data", "prices"):
        if isinstance(rows, dict) and key in rows:
            rows = rows[key]
    if isinstance(rows, dict):
        rows = next((v for v in rows.values() if isinstance(v, list)), [])
    if not isinstance(rows, list) or not rows:
        raise TossApiError(f"캔들 배열을 찾지 못했습니다: {str(payload)[:200]}")

    frame = pd.DataFrame(rows)
    keymap = {
        "dt": "dt", "dtBase": "dt", "dtOriginal": "dt", "baseDateTime": "dt",
        "tradeDateTime": "dt", "date": "dt", "time": "dt",
        "open": "open", "openPrice": "open",
        "high": "high", "highPrice": "high",
        "low": "low", "lowPrice": "low",
        "close": "close", "closePrice": "close", "price": "close",
        "volume": "volume", "accVolume": "volume", "tradeVolume": "volume",
    }
    frame = frame.rename(columns={k: v for k, v in keymap.items() if k in frame.columns})
    return validate(normalize(frame))


class TossDataSource(DataSource):
    name = "toss"

    def __init__(
        self,
        client: TossClient | None = None,
        *,
        session_file: str | Path | None = None,
        **client_kwargs: Any,
    ) -> None:
        """session_file 을 주면 `probe-toss --curl` 로 저장한 헤더/쿠키를 쓴다."""
        if session_file is not None:
            client_kwargs = {**load_session(session_file), **client_kwargs}
        self.client = client or TossClient(**client_kwargs)

    def symbol(self, code: str) -> Symbol:
        info = self.client.stock_info(code)
        return Symbol(
            code=info.get("symbol", code),
            name=info.get("name", ""),
            market=(info.get("market") or {}).get("displayName", ""),
        )

    def snapshot(self, code: str) -> dict[str, Any]:
        """현재가 스냅샷. 실시간 스캐너의 1차 필터로 쓸 수 있다."""
        return self.client.price_snapshot(code)

    def candles(
        self,
        code: str,
        interval: Interval,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        period = _PERIOD_TOKEN[interval]
        params: dict[str, Any] = {}
        if count:
            params["count"] = count
        if end is not None:
            params["to"] = str(end)
        try:
            payload = self.client.raw_candles(code, period, **params)
        except TossApiError as exc:
            if " 400 " in f" {exc} " or "HTTP 400" in str(exc):
                raise TossCandlesUnavailable(
                    "토스 캔들 엔드포인트가 익명 호출을 거절했습니다(400). "
                    "브라우저 개발자도구 Network 탭에서 c-chart 요청의 헤더/쿠키를 복사해 "
                    "TossDataSource(session_headers=..., cookies=...) 로 주입한 뒤 "
                    "TossClient.probe_candle_endpoint() 로 파라미터를 확정하세요."
                ) from exc
            raise
        df = parse_candles(payload)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]
        return df
