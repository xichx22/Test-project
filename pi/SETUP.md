# 라즈베리파이 이식

장 마감 뒤 신호를 뽑아 알림을 보내고, 집 안에서 볼 수 있는 대시보드를
띄운다. 매수·매도는 사람이 한다 — 이 파이는 주문을 내지 않는다.

## 하드웨어

실측 부하는 다음과 같다 (Xeon 2.8GHz 기준).

| 항목 | 값 |
|---|---|
| 신호 조회 1회 | 22초 (CPU 16초) |
| 최대 메모리 | 84MB |
| 데이터 | 53MB (1,064종목 × 760봉) |
| 저장소 전체 | 약 100MB |

| 모델 | 예상 조회 시간 | 판정 |
|---|---|---|
| 파이 5 (4GB) | 40~60초 | 넉넉하다 |
| 파이 4 (2GB) | 70~90초 | 충분하다 |
| 파이 제로 2 W (512MB) | 3~5분 | 된다. 메모리도 84MB라 여유 |

SD카드는 16GB 이상이면 된다. 매일 쓰기가 적어(하루 수십 KB) 수명 걱정은 없다.

## 설치

```bash
sudo timedatectl set-timezone Asia/Seoul        # 이걸 안 하면 타이머가 엉뚱한 때 돈다
sudo apt update && sudo apt install -y git python3-venv

git clone https://github.com/xichx22/Test-project.git ~/Test-project
cd ~/Test-project
git checkout claude/toss-api-signal-analysis-wfppcm

python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .
```

`data_live/1d`(53MB)는 저장소에 들어 있으므로 시세를 처음부터 받을 필요가 없다.
clone 직후 빠진 날짜만 받으면 된다:

```bash
.venv/bin/python scripts/refresh.py --live
```

## 손으로 한 번 돌려보기

```bash
.venv/bin/python scripts/today.py 3 --live          # 사람이 읽는 표
.venv/bin/python scripts/today.py 3 --live --json   # 기계가 읽는 형태
./pi/daily.sh                                        # 전체 파이프라인
```

`pi/public/index.html` 이 생기면 성공이다.

## 알림 설정

`pi/notify.conf` 를 만든다. **이 파일은 커밋하지 않는다** (.gitignore 에 있다).

가장 쉬운 방법 — ntfy.sh (계정 없이 됨):

```
NTFY_TOPIC=아무거나-겹치지-않을-이름-1234
```

휴대폰에 ntfy 앱을 깔고 같은 이름을 구독하면 알림이 온다.
**주제 이름을 아는 사람은 누구나 볼 수 있으니** 길고 추측하기 어렵게 짓는다.

텔레그램을 쓰려면:

```
TELEGRAM_TOKEN=123456:AA...
TELEGRAM_CHAT=987654321
```

설정이 없으면 화면에만 찍고 조용히 넘어간다.

## 자동 실행

```bash
sudo cp pi/systemd/*.service pi/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stocksignal.timer     # 평일 16:00 신호 조회
sudo systemctl enable --now stockdash.service     # 대시보드 웹서버

systemctl list-timers stocksignal.timer           # 다음 실행 시각 확인
sudo systemctl start stocksignal.service          # 지금 바로 한 번 돌리기
journalctl -u stocksignal -n 50                   # 로그
```

유닛 파일은 `User=pi`, 경로 `/home/pi/Test-project` 로 되어 있다.
사용자 이름이나 경로가 다르면 세 파일 모두 고칠 것.

대시보드는 집 안에서 `http://<파이IP>:8800` 으로 열린다.
**공유기 포트포워딩으로 바깥에 열지 말 것** — 인증이 없다.

## 하는 일

```
16:00  타이머
  ↓  refresh.py --live    네이버에서 그날 일봉 (1,064종목, 약 5분)
  ↓  today.py --json      컵앤핸들 검출 → pi/signals.json
  ↓  render.py            pi/public/index.html 다시 그림 + 보유 종목 현재가 갱신
  ↓  notify.py            신호·익절·만기 있으면 시끄럽게, 없으면 조용히
```

## 규칙 (C안 혼합)

| | |
|---|---|
| ⑦ 과매도 후 컵앤핸들 | 계좌 총자산의 **1/3**. 현금이 모자라면 ⑤ 중 청산 목표일이 가까운 순서로 필요한 만큼만 판다. ⑦ 자리는 최대 3개 |
| ⑤ 컵앤핸들 + 20일 신고가 | **33만원** |
| 같은 날 둘 다 | **⑦ 우선** |
| 매수 | 신호일 **다음 날 시가** |
| 매도 | **+20%** 또는 **60거래일**, 먼저 오는 쪽. 손절 없음 |
| 제외 | 20일 평균 거래대금 3억원 미만 |

**① 컵앤핸들 단독은 매수 대상이 아니다.** `today.py` 표에는 나오지만
`--json` 과 대시보드에는 ⑤·⑦만 실린다.

## 매매 기록

`pi/state.json` 이 원장이다. 매수·매도하면 여기에 적는다.

```json
{"id":"t20260902","type":"⑤","code":"036800","name":"나이스정보통신",
 "sigdate":"2026-09-01","buydate":"2026-09-01","price":7900,"qty":42,
 "pivot":7650,"turnover":"11.8","due":"2026-12-01","now":7700,
 "checks":{"c1":true,"c2":true,"c3":true,"c4":true},
 "memo":"","exitdate":"","exitPrice":0,"reason":""}
```

- `due` 는 체결일 + 90일(달력) — 60거래일에 해당한다
- 청산하면 `exitPrice` · `exitdate` · `reason` 을 채운다
- `checks` 는 규칙을 지켰는지다. 하나라도 false 면 대시보드에 "규칙 밖"으로 표시된다

고친 뒤 `.venv/bin/python pi/render.py` 를 돌리면 대시보드에 반영된다.

## 안 되는 것 · 주의

- **주문을 내지 않는다.** 신호만 알린다. 매수·매도는 증권사 앱에서 직접 한다.
- **네이버가 막히면 어제 데이터로 돈다.** 그때는 "일봉 갱신 실패" 알림이 먼저 간다.
  이 알림을 받으면 그날 신호는 믿지 않는다.
- **파이가 꺼져 있던 날**은 켜질 때 한 번 돈다(`Persistent=true`). 다만 그날
  종가 기준 신호이므로 매수 시점(다음 날 시가)은 이미 지났을 수 있다.
- 검출기가 보는 최대 길이는 종목당 385봉(컵 325 + 선행 60)이다. 760봉을
  들고 있으므로 여유가 있지만, `data_live` 를 더 줄이면 신호가 달라진다.
