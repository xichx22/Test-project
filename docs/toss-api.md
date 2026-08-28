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

## 2-b. 서버가 라우트 패턴을 알려준다 (탐지기)

응답 헤더에 라우트 패턴이 그대로 실려 온다.

```
tossinvest-path-pattern: /api/v1/c-chart/{product}/{code}/{stepUnit}
```

이걸 **존재하는 경로를 찾는 탐지기**로 쓸 수 있다. 이 헤더가 오면 라우트가
매칭된 것이고, 안 오면 그런 경로가 없는 것이다.

확인된 사실:

- 세 번째 세그먼트의 이름은 `period` 가 아니라 **`stepUnit`** 이다
- `/api/v1`, `/api/v2`, `/api/v3` 모두 같은 패턴으로 존재한다
- `chart`, `charts`, `candles`, `candle`, `price-chart`, `stock-chart`,
  `stock-prices/{code}/period|history|chart|candles` 는 **전부 라우트 없음**
  → **`c-chart` 가 캔들 경로가 맞다**

그런데도 400 이다. 아래를 전부 시도했고 예외 없이 400 이었다.

- **stepUnit 값 46종**: `day/week/month/year/minute/hour/tick`, 대문자판,
  `D/W/M/Y`, `1d/1w/1M/5m/15m/30m/1h`, `DAY1/DAY_1/MINUTE1/ONE_DAY`,
  `daily/weekly/monthly`, 숫자 등 — **의미 없는 값과 응답이 똑같다**
  → stepUnit 파싱 단계까지 가지도 못한다는 뜻
- **쿼리 파라미터**: `count/step/size/limit/to/from/dt/timestamp/order/
  direction/baseDateTime/endDateTime/startDateTime/useAdjustedRate/session/
  sessionType/priceType` 단독 및 전체 조합
- **헤더**: `x-device-id`, `browser-tab-id` (CORS 프리플라이트가 허용한다고
  응답한 두 개), 그리고 `sec-ch-ua`/`sec-fetch-*`/`accept-language` 까지 갖춘
  완전한 브라우저 헤더 세트
- **메서드**: GET, POST(JSON 바디)

같은 헤더 세트로 `/api/v2/stock-prices/A005930` 은 **200** 이 온다.
연결 자체가 막힌 것은 아니다.

에러 바디의 `message` 가 `null` 이라 서버가 무엇을 문제 삼는지 알려주지 않는다.
남은 후보는 **인증 쿠키** 또는 **이 환경의 출구 IP**(미국 Google Cloud) 다.
사용자 PC 에서는 동작한다는 보고가 있어 후자일 가능성이 있다.

## 3. SPA 번들에서 찾아보기 — 실패

파라미터 이름을 코드에서 직접 확인하려고 `tossinvest.com/stocks/A005930` 의
Next.js 청크 39개(1.3MB)를 받아 뒤졌으나, `c-chart`·`candle`·`wts-info-api`
어느 것도 나오지 않았다. 페이지 자체에 종목명(`삼성전자`)도 `<title>` 도 없다 —
서버가 이 IP 에 **내용 없는 껍데기**를 내려주고 있고, 차트 코드는 동적 임포트라
초기 HTML 의 청크 목록에도 없다.

즉 **브라우저 없이는 확정할 수 없다.** 아래 절차가 유일한 경로다.

## 4. 확정 절차 — "Copy as cURL"

헤더를 손으로 옮겨 적을 필요 없다. 브라우저가 보내는 요청을 통째로 복사하면 된다.

1. 브라우저에서 <https://tossinvest.com/stocks/A005930> 을 열고 **차트** 탭으로 간다
2. **F12** → **Network** 탭 → 필터 입력창에 `c-chart` 입력
3. 차트의 **기간 버튼(1일/1주/1개월 등)을 눌러** 요청이 새로 뜨게 한다
4. 목록에 뜬 요청을 **우클릭 → Copy → Copy as cURL**
5. 아무 파일에나 붙여넣어 저장하고:

```bash
python -m tsignal probe-toss --curl 저장한파일.txt
```

도구가 하는 일:

- cURL 을 파싱해 **경로·쿼리 파라미터·헤더·쿠키**를 분해해 보여준다
  (Chrome/Firefox/Safari, Windows cmd 줄바꿈까지 처리)
- 그 요청을 **그대로 재현**해 200 이 나오는지 확인한다
- 파라미터를 **하나씩 빼보며 진짜 필수인 것만** 추린다
  (브라우저는 쓰지도 않는 파라미터를 딸려 보내는 경우가 많다)
- 응답을 캔들로 파싱해 봉 수와 기간을 출력한다
- 확인된 헤더/쿠키/엔드포인트를 `toss.session.json` 으로 저장한다

이후 수집은 저장된 세션으로 한다:

```python
TossDataSource(session_file="toss.session.json")
```

확정되면 고칠 곳은 두 군데뿐이다.

1. `toss.py` 의 `_PERIOD_TOKEN` — 기간 토큰 매핑
2. `toss.py` 의 `parse_candles()` — 응답 키 매핑 (이미 흔한 키 이름을 폭넓게 받는다)

### 주의 — 쿠키

복사한 cURL 에는 **로그인 세션 쿠키가 들어 있을 수 있다.** 남에게 공유하지 말 것.
`toss.session.json` 은 `.gitignore` 에 등록돼 있어 커밋되지 않는다.

## 5. 대안 데이터 소스

| 소스 | 분봉 | 인증 | 비고 |
| --- | --- | --- | --- |
| 토스 WTS (비공식) | ? | 세션 헤더 추정 | 약관 확인 필요, 스키마 불안정 |
| 한국투자증권 Open API | 당일 위주 | 앱키/앱시크릿 | 문서화됨, 모의투자 주문까지 가능 |
| 키움 OpenAPI+ | 과거분봉 O | 계좌+OCX | Windows 32bit 의존 |
| pykrx (KRX 공개데이터) | ✗ (일봉만) | 불필요 | 일봉 검증용으로는 충분 |

**권장**: 재현 가능한 검증이 목적이므로 공식 API(한국투자증권)를 축으로 두고,
토스는 실시간 스냅샷 보조로 쓰는 구성. 어댑터를 하나 더 쓰면 되고,
지표·신호·검증 코드는 한 줄도 바뀌지 않는다.

## 6. 주의

- 비공식 엔드포인트다. 서비스 약관과 `robots.txt` 를 확인하고 본인 책임으로 사용할 것.
- `TossClient(min_interval_sec=...)` 로 호출 간격을 두고 있다. 낮추지 말 것.
- 수집한 데이터는 CSV 로 떨어뜨려 두고(`tsignal fetch`), 실험은 `csv` 소스로
  돌리는 것을 기본으로 한다. 같은 입력에서 같은 결과가 나와야 검증이 의미를 갖는다.
