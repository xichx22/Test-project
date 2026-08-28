"""브라우저에서 복사한 cURL 명령을 파싱한다.

토스 캔들 엔드포인트는 익명 호출을 거절한다(400). 실제 요청이 어떤 경로·파라미터·
헤더를 쓰는지는 브라우저가 보내는 요청을 그대로 보는 게 가장 확실하다.

개발자도구에서 요청을 우클릭 → Copy → **Copy as cURL** 하면 URL·헤더·쿠키가
전부 담긴 한 줄이 나온다. 그걸 파일로 저장해 그대로 넘기면 된다.
헤더를 하나씩 JSON 으로 옮겨 적는 것보다 실수가 적다.

Chrome(bash/cmd), Firefox, Safari 형식을 모두 받는다.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlsplit


@dataclass
class CurlRequest:
    url: str
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)
    data: str | None = None

    @property
    def path(self) -> str:
        return urlsplit(self.url).path

    @property
    def host(self) -> str:
        return urlsplit(self.url).netloc

    @property
    def params(self) -> dict[str, str]:
        return dict(parse_qsl(urlsplit(self.url).query, keep_blank_values=True))

    def base_url(self) -> str:
        parts = urlsplit(self.url)
        return f"{parts.scheme}://{parts.netloc}{parts.path}"


class CurlParseError(ValueError):
    pass


def _split_cookie_header(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in value.split(";"):
        if "=" in chunk:
            key, _, val = chunk.partition("=")
            out[key.strip()] = val.strip()
    return out


def parse_curl(text: str) -> CurlRequest:
    """cURL 명령 문자열 → CurlRequest."""
    cleaned = text.strip()
    if not cleaned:
        raise CurlParseError("빈 입력입니다.")

    # Windows cmd 형식(^ 줄바꿈)과 PowerShell 백틱을 정리한다.
    cleaned = re.sub(r"\^\s*\n", " ", cleaned)
    cleaned = re.sub(r"`\s*\n", " ", cleaned)
    cleaned = re.sub(r"\\\s*\n", " ", cleaned)
    cleaned = cleaned.replace("^\"", "\"")

    try:
        tokens = shlex.split(cleaned)
    except ValueError as exc:
        raise CurlParseError(f"명령을 토큰으로 나누지 못했습니다: {exc}") from exc

    if not tokens or tokens[0] != "curl":
        raise CurlParseError("'curl' 로 시작하는 명령이어야 합니다.")

    request = CurlRequest(url="")
    method_set = False
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token in ("-H", "--header"):
            i += 1
            raw = tokens[i].strip()
            if raw.startswith(":"):          # HTTP/2 의사 헤더(:authority 등)는 버린다
                i += 1
                continue
            key, _, value = raw.partition(":")
            key, value = key.strip(), value.strip()
            if key.lower() == "cookie":
                request.cookies.update(_split_cookie_header(value))
            else:
                request.headers[key] = value
        elif token in ("-b", "--cookie"):
            i += 1
            request.cookies.update(_split_cookie_header(tokens[i]))
        elif token in ("-X", "--request"):
            i += 1
            request.method = tokens[i].upper()
            method_set = True
        elif token in ("-d", "--data", "--data-raw", "--data-binary"):
            i += 1
            request.data = tokens[i]
            if not method_set:
                request.method = "POST"
        elif token.startswith("-"):
            pass                              # --compressed, -k 등 부가 옵션은 무시
        elif not request.url:
            request.url = token
        i += 1

    if not request.url:
        raise CurlParseError("URL 을 찾지 못했습니다.")
    return request


def parse_curl_file(path: str) -> CurlRequest:
    with open(path, encoding="utf-8") as handle:
        return parse_curl(handle.read())


def summarize(request: CurlRequest) -> str:
    lines = [
        f"{request.method} {request.base_url()}",
        f"  호스트   : {request.host}",
        f"  경로     : {request.path}",
        f"  파라미터 : {request.params or '(없음)'}",
        f"  헤더     : {len(request.headers)}개  {', '.join(sorted(request.headers)[:8])}",
        f"  쿠키     : {len(request.cookies)}개  {', '.join(sorted(request.cookies)[:8])}",
    ]
    return "\n".join(lines)
