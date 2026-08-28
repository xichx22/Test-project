# 토스증권 웹(WTS) 엔드포인트 실측 기록

측정일: 2026-08-28 / 측정 도구: `curl` (익명, 브라우저 세션 없음)
호스트: `https://wts-info-api.tossinvest.com`

토스증권은 공개 문서화된 REST API를 제공하지 않는다. 아래는 웹 클라이언트가
사용하는 내부 엔드포인트를 직접 호출해 확인한 결과이며, **언제든 예고 없이
바뀔 수 있다**. 어댑터를 `src/tsignal/datasource/toss.py` 한 파일로 격리해 둔
이유가 이것이다.

## 1. 동작 확인된 엔드포인트

### 종목 메타

```
GET /api/v2/stock-infos/A005930
→ 200
{"result":{"code":"A005930","symbol":"005930","isinCode":"KR7005930003",
  "name":"삼성전자","englishName":"SamsungElec",
  "market":{"code":"KSP","displayName":"코스피"},
  "group":{"code":"ST","displayName":"주권"}, ...}}
```

종목 코드는 `A` 접두사가 붙은 형태(`A005930`)를 쓴다.

### 현재가 스냅샷

```
GET /api/v2/stock-prices/A005930
→ 200
{"result":{"code":"A005930","tradeDateTime":"...","open":262500,"high":266000,
  "low":258000,"close":258500,"volume":14521176,"value":3797432925000,
  "base":266000,"changeType":"DOWN","currency":"KRW",
  "high52w":380000,"low52w":67500, ...}}
```

일중 OHLCV 요약이라 **실시간 스캐너의 1차 필터**로는 쓸 수 있지만,
과거 시계열이 아니므로 지표 계산에는 부족하다.

## 2. 막힌 엔드포인트 — 과거 캔들

```
GET /api/v1/c-chart/kr-stock/A005930/{period}
→ 400 {"error":{"statusCode":400,"code":"400","message":null}}
```

**라우트 자체는 존재한다.** 근거:

| 요청 | 응답 |
| --- | --- |
| `/api/v1/c-chart/zzz/yyy` (없는 리소스) | `404 page not found` |
| `/api/v1/c-chart/kr-etf?count=10` (없는 리소스) | `404` |
| `/api/v1/c-chart/kr-stock/A005930/day` | `400` (JSON 에러 바디) |

404가 아니라 400이라는 것은 라우팅은 성공했고 요청 검증에서 걸렸다는 뜻이다.

### 시도한 조합 (전부 400)

- **period 세그먼트**: `day`, `1D`, `1d`, `D`, `DAY`, `days`, `daily`,
  `day:1`, `1:day`, `minute`, `min:1`, `week`, `month`, `year`
- **쿼리 파라미터**: `count`, `size`, `to`, `from`, `dt`, `endDateTime`,
  `interval`, `period`, `type`, `productCode`, `sessionType`, `adjusted`,
  `useAdjustedRate`, `session`, `lastDt`, `priceType` (단독 및 조합)
- **API 버전**: `/api/v1/`, `/api/v2/`
- **메서드**: GET, POST(JSON 바디)
- **헤더**: `User-Agent`, `Referer: https://tossinvest.com/`,
  `Origin: https://tossinvest.com`, `Accept: application/json`

### 추정 원인

브라우저 세션 컨텍스트(쿠키 / 디바이스 식별 헤더 / 내부 토큰)가 필요할 가능성이
가장 높다. SPA 번들에서 파라미터 이름을 직접 확인하려 했으나, 이 환경의 프록시
IP 에서는 `tossinvest.com` 의 페이지 라우트가 404를 반환해 청크 다운로드까지
가지 못했다.

## 3. 재탐색 방법

브라우저에서 토스증권 차트를 연 뒤, 개발자도구 → Network → `c-chart` 요청의
**Request Headers 를 JSON 파일로 저장**하고:

```bash
python -m tsignal probe-toss --code 005930 --headers headers.json
```

`TossClient.probe_candle_endpoint()` 가 period × 파라미터 조합을 훑어
200을 주는 조합을 찾아 출력한다. 찾으면 고칠 곳은 두 군데뿐이다.

1. `toss.py` 의 `_PERIOD_TOKEN` — 기간 토큰 매핑
2. `toss.py` 의 `parse_candles()` — 응답 키 매핑 (이미 흔한 키 이름들을 폭넓게 받는다)

## 4. 대안 데이터 소스

| 소스 | 분봉 | 인증 | 비고 |
| --- | --- | --- | --- |
| 토스 WTS (비공식) | ? | 세션 헤더 추정 | 약관 확인 필요, 스키마 불안정 |
| 한국투자증권 Open API | 당일 위주 | 앱키/앱시크릿 | 문서화됨, 모의투자 주문까지 가능 |
| 키움 OpenAPI+ | 과거분봉 O | 계좌+OCX | Windows 32bit 의존 |
| pykrx (KRX 공개데이터) | ✗ (일봉만) | 불필요 | 일봉 검증용으로는 충분 |

**권장**: 재현 가능한 검증이 목적이므로 공식 API(한국투자증권)를 축으로 두고,
토스는 실시간 스냅샷 보조로 쓰는 구성. 어댑터를 하나 더 쓰면 되고,
지표·신호·검증 코드는 한 줄도 바뀌지 않는다.

## 5. 주의

- 비공식 엔드포인트다. 서비스 약관과 `robots.txt` 를 확인하고 본인 책임으로 사용할 것.
- `TossClient(min_interval_sec=...)` 로 호출 간격을 두고 있다. 낮추지 말 것.
- 수집한 데이터는 CSV 로 떨어뜨려 두고(`tsignal fetch`), 실험은 `csv` 소스로
  돌리는 것을 기본으로 한다. 같은 입력에서 같은 결과가 나와야 검증이 의미를 갖는다.
