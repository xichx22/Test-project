#!/usr/bin/env bash
# 매일 장 마감 후 한 번 도는 파이프라인. systemd timer 가 부른다.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG="$ROOT/pi/log/$(date +%Y-%m).log"
mkdir -p "$ROOT/pi/log" "$ROOT/pi/public"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 두 개가 동시에 돌면 CSV 가 깨진다. 손으로 돌리는 동안 타이머가 떠도 안전하게.
exec 9>/tmp/stocksignal.lock
if ! flock -n 9; then
  say "이미 돌고 있다. 이번은 건너뛴다"
  exit 0
fi

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

say "=== 시작 ==="

# 1) 일봉 갱신 — 네트워크가 죽어도 이전 데이터로 계속 간다
if ! "$PY" scripts/refresh.py --live >>"$LOG" 2>&1; then
  say "경고: 일봉 갱신 실패. 이전 데이터로 진행한다"
  "$ROOT/pi/notify.py" --title "일봉 갱신 실패" \
    --body "네이버에서 시세를 못 받았다. 오늘 신호는 어제 데이터 기준이라 믿을 수 없다." \
    --level high >>"$LOG" 2>&1
fi

# 2) 신호 조회
if ! "$PY" scripts/today.py 3 --live --json > "$ROOT/pi/signals.json" 2>>"$LOG"; then
  say "오류: 신호 조회 실패"
  "$ROOT/pi/notify.py" --title "신호 조회 실패" \
    --body "today.py 가 죽었다. pi/log 를 봐라." --level high >>"$LOG" 2>&1
  exit 1
fi

ASOF=$("$PY" -c "import json;print(json.load(open('pi/signals.json'))['asof'])")
N=$("$PY" -c "import json;print(len(json.load(open('pi/signals.json'))['signals']))")
say "기준일 $ASOF · 신호 $N건"

# 3) 대시보드 다시 그리기 (보유 종목 현재가도 여기서 갱신된다)
"$PY" pi/render.py >>"$LOG" 2>&1 || say "경고: 대시보드 생성 실패"

# 4) 알림 — 신호가 있거나 청산 대상이 있을 때만 시끄럽게, 아니면 조용히 한 줄
"$PY" pi/notify.py --from-state >>"$LOG" 2>&1

say "=== 끝 ==="
