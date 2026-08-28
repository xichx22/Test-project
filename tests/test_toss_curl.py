"""브라우저 cURL 파싱 테스트.

토스 캔들 엔드포인트는 브라우저 요청을 그대로 재현해야 뚫린다.
사용자가 손으로 옮겨 적지 않고 'Copy as cURL' 을 그대로 쓸 수 있어야 한다.
"""

import pytest

from tsignal.datasource.toss_curl import CurlParseError, parse_curl

CHROME = """curl 'https://wts-info-api.tossinvest.com/api/v1/c-chart/kr-stock/A005930/day?count=100&to=2026-08-28' \\
  -H 'accept: application/json' \\
  -H 'referer: https://tossinvest.com/' \\
  -H 'x-device-id: dev-abc' \\
  -b 'x-toss-session=deadbeef; _ga=GA1.1.5' \\
  --compressed"""

FIREFOX = ("""curl "https://wts-info-api.tossinvest.com/api/v1/c-chart/kr-stock/A005930/day?count=50" """
           """-X GET -H "Accept: application/json" -H "Cookie: a=1; b=2" """)


def test_parses_chrome_style():
    req = parse_curl(CHROME)
    assert req.method == "GET"
    assert req.host == "wts-info-api.tossinvest.com"
    assert req.path == "/api/v1/c-chart/kr-stock/A005930/day"
    assert req.params == {"count": "100", "to": "2026-08-28"}
    assert req.headers["x-device-id"] == "dev-abc"
    assert req.cookies == {"x-toss-session": "deadbeef", "_ga": "GA1.1.5"}


def test_parses_firefox_style_with_cookie_header():
    req = parse_curl(FIREFOX)
    assert req.params == {"count": "50"}
    assert req.cookies == {"a": "1", "b": "2"}
    # Cookie 헤더는 쿠키로 옮겨가고 헤더에는 남지 않아야 한다.
    assert "Cookie" not in req.headers


def test_cookie_header_is_not_duplicated_into_headers():
    req = parse_curl(CHROME)
    assert not any(k.lower() == "cookie" for k in req.headers)


def test_http2_pseudo_headers_are_dropped():
    req = parse_curl("""curl 'https://x.example/a' -H ':authority: x.example' -H 'accept: */*'""")
    assert list(req.headers) == ["accept"]


def test_post_body_sets_method():
    req = parse_curl("""curl 'https://x.example/a' --data-raw '{"k":1}'""")
    assert req.method == "POST"
    assert req.data == '{"k":1}'


def test_windows_cmd_line_continuations():
    req = parse_curl('curl ^\n "https://x.example/a?b=1" ^\n -H "accept: application/json"')
    assert req.params == {"b": "1"}
    assert req.headers["accept"] == "application/json"


def test_rejects_non_curl_input():
    with pytest.raises(CurlParseError):
        parse_curl("wget https://x.example/a")
    with pytest.raises(CurlParseError):
        parse_curl("")


def test_base_url_strips_query():
    req = parse_curl(CHROME)
    assert req.base_url().endswith("/day")
    assert "?" not in req.base_url()
