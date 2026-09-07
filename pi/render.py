#!/usr/bin/env python3
"""로컬 대시보드를 다시 그린다.

`docs/pages/trade-journal.html` 을 틀로 쓰고, 그 안의 상태 JSON 만
`pi/state.json` + `pi/signals.json` 으로 갈아끼워 `pi/public/index.html`
로 내보낸다. 페이지 자체는 아티팩트에 올린 것과 같은 파일이라
디자인이 두 벌로 갈리지 않는다.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "docs/pages/trade-journal.html"
STATE = ROOT / "pi/state.json"
SIGNALS = ROOT / "pi/signals.json"
OUT = ROOT / "pi/public/index.html"
LIVE = ROOT / "data_live/1d"


def last_close(code: str) -> float | None:
    f = LIVE / f"{code}.csv"
    if not f.exists():
        return None
    try:
        line = f.read_text().rstrip().rsplit("\n", 1)[-1]
        return float(line.split(",")[4])          # dt,open,high,low,close,volume
    except (ValueError, IndexError):
        return None


def main() -> int:
    if not TPL.exists():
        print(f"틀이 없다: {TPL}", file=sys.stderr); return 1
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
        "settings": {"start": "2026-09-01", "seed": 1000000,
                     "monthly": 350000, "depositDay": 21},
        "asof": "", "signals": [], "trades": []}

    if SIGNALS.exists():
        sig = json.loads(SIGNALS.read_text(encoding="utf-8"))
        state["asof"] = sig.get("asof", state.get("asof", ""))
        # ⑤·⑦ 만 싣는다. ①(컵앤핸들 단독)은 매수 규칙이 아니다.
        state["signals"] = [s for s in sig.get("signals", []) if s.get("type") in ("⑤", "⑦")]

    # 보유 종목 현재가 갱신
    for t in state.get("trades", []):
        if t.get("exitPrice"):
            continue
        px = last_close(t.get("code", ""))
        if px:
            t["now"] = px

    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    html = TPL.read_text(encoding="utf-8")
    blob = json.dumps({k: state[k] for k in ("settings", "asof", "signals", "trades")},
                      ensure_ascii=False).replace("<", "\\u003c")
    new, n = re.subn(r'(<script type="application/json" id="state">).*?(</script>)',
                     lambda m: m.group(1) + blob + m.group(2), html, count=1, flags=re.S)
    if n != 1:
        print("상태 블록을 못 찾았다", file=sys.stderr); return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(new, encoding="utf-8")
    open_n = sum(1 for t in state["trades"] if not t.get("exitPrice"))
    print(f"대시보드 생성: {OUT}  (신호 {len(state['signals'])}건 · 보유 {open_n}자리)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
