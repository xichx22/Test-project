#!/usr/bin/env python3
"""알림 — ntfy.sh(계정 불필요) 또는 텔레그램.

  pi/notify.conf 에 아래 중 하나를 쓴다 (없으면 화면에만 찍는다):
      NTFY_TOPIC=아무거나-겹치지-않는-이름
      TELEGRAM_TOKEN=...
      TELEGRAM_CHAT=...

  --from-state : state.json + signals.json 을 읽어 오늘 보낼 내용을 스스로 만든다
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import date, datetime
from pathlib import Path
import urllib.parse, urllib.request

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "pi/notify.conf"
COST, TARGET = 0.0028, 1.20


def conf() -> dict[str, str]:
    out = dict(os.environ)
    if CONF.exists():
        for line in CONF.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def send(title: str, body: str, level: str = "default") -> bool:
    c = conf()
    ok = False
    if c.get("NTFY_TOPIC"):
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{c['NTFY_TOPIC']}", data=body.encode("utf-8"),
                headers={"Title": urllib.parse.quote(title),
                         "Priority": {"high": "high", "low": "low"}.get(level, "default"),
                         "Tags": "chart_with_upwards_trend"})
            urllib.request.urlopen(req, timeout=15).read()
            ok = True
        except Exception as e:                      # 알림 실패가 파이프라인을 죽이면 안 된다
            print(f"ntfy 실패: {e}", file=sys.stderr)
    if c.get("TELEGRAM_TOKEN") and c.get("TELEGRAM_CHAT"):
        try:
            data = urllib.parse.urlencode(
                {"chat_id": c["TELEGRAM_CHAT"], "text": f"{title}\n\n{body}"}).encode()
            urllib.request.urlopen(
                f"https://api.telegram.org/bot{c['TELEGRAM_TOKEN']}/sendMessage",
                data=data, timeout=15).read()
            ok = True
        except Exception as e:
            print(f"텔레그램 실패: {e}", file=sys.stderr)
    print(f"[{level}] {title}\n{body}")
    return ok


def from_state() -> tuple[str, str, str]:
    st = json.loads((ROOT / "pi/state.json").read_text(encoding="utf-8"))
    sig = st.get("signals", [])
    asof = st.get("asof", str(date.today()))
    lines, urgent = [], False

    if sig:
        urgent = True
        lines.append(f"■ 매수 신호 {len(sig)}건 — 내일 시가에 산다")
        for s in sig:
            amt = "자산÷3" if s["type"] == "⑦" else "33만원"
            lines.append(f"  {s['type']} {s['name']}({s['code']}) "
                         f"종가 {s['close']:,} · 돌파선 {s['pivot']:,} · "
                         f"거래량 {s['volMult']}배 · {s['turnover']}억 → {amt}")
    else:
        lines.append("■ 매수 신호 없음")

    hold = [t for t in st.get("trades", []) if not t.get("exitPrice")]
    if hold:
        lines.append(f"\n■ 보유 {len(hold)}자리")
        for t in hold:
            now = t.get("now") or t["price"]
            r = (now / t["price"] - 1 - COST) * 100
            tag = ""
            if now >= t["price"] * TARGET:
                tag = "  ← 익절 대상"; urgent = True
            else:
                try:
                    left = (datetime.strptime(t["due"], "%Y-%m-%d").date() - date.today()).days
                    if left <= 7:
                        tag = f"  ← 만기 {left}일"; urgent = True
                except (ValueError, KeyError):
                    pass
            lines.append(f"  {t['name']} {now:,}원 ({r:+.1f}%) "
                         f"목표 {round(t['price']*TARGET):,}{tag}")
    else:
        lines.append("\n■ 보유 없음")

    head = f"매매 {asof}" + (f" · 신호 {len(sig)}건" if sig else " · 신호 없음")
    return head, "\n".join(lines), ("high" if urgent else "low")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title"); ap.add_argument("--body")
    ap.add_argument("--level", default="default")
    ap.add_argument("--from-state", action="store_true")
    a = ap.parse_args()
    if a.from_state:
        t, b, lv = from_state()
    else:
        t, b, lv = a.title or "알림", a.body or "", a.level
    send(t, b, lv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
