"""⑥ 과 ⑦ 을 나란히 — 최근 신호가 언제였고, 연도별 성적이 어떤가."""
import pickle, numpy as np, pandas as pd
from tsignal.evaluation.exitlab import Exit, resolve
S="/tmp/claude-0/-home-user-Test-project/e2567955-0c39-5157-a915-38387e9cd240/scratchpad"
parts=pickle.load(open(f"{S}/parts16.pkl","rb"))
blob=pickle.load(open(f"{S}/cache16.pkl","rb")); data,feats=blob["data"],blob["feats"]
uni=pd.read_csv("data_wide/universe.csv",dtype=str).set_index("code")
cup,mfi=parts["컵앤핸들"],parts["MFI과매도"]
LAST=max(d.index[-1] for d in data.values())
print(f"데이터 마지막 거래일: {LAST.date()}\n")

def build(kind):
    ev={}
    for code,sc in cup.items():
        sm=mfi.get(code)
        if sm is None: continue
        if kind=="O":   # ⑦ MFI 가 먼저, 돌파일에만
            h=sc & sm.rolling(21,min_periods=1).max().astype(bool)
        else:           # ⑥ 순서 무관 — 둘 중 나중에 뜬 날
            h=(sc & sm.rolling(21,min_periods=1).max().astype(bool)) | \
              (sm & sc.rolling(21,min_periods=1).max().astype(bool))
        h=h.fillna(False)
        if h.any(): ev[code]=h
    return ev

RULE=Exit("익절20%",take_profit=0.20,horizon=60)
res={}
for name,kind in [("⑦ MFI→컵앤핸들","O"),("⑥ 컵∩MFI 순서무관","I")]:
    ev=build(kind); tr=resolve(ev,data,feats,rule=RULE)
    tr["진입일"]=pd.DatetimeIndex(tr["진입일"]); res[name]=tr
    print(f"=== {name} ===  전체 매매 {len(tr):,}건")
    for yr in (2024,2025,2026):
        t=tr[tr["진입일"].dt.year==yr]
        if t.empty: print(f"  {yr}: 없음"); continue
        r=(t["청산"]*(1-0.0014)/t["진입"]-1)
        print(f"  {yr}  {len(t):3d}건  평균 {r.mean():+6.2%}  중앙 {r.median():+6.2%}  "
              f"승률 {(r>0).mean():4.0%}  최악 {r.min():+6.1%}")
    # 최근 신호
    allsig=sorted({ts for c,s in ev.items() for ts in s.index[s.to_numpy()]})
    idx=pd.DatetimeIndex(sorted({t for d in data.values() for t in d.index}))
    ago=lambda ts: len(idx)-1-idx.get_loc(ts)
    print(f"  최근 신호일: {allsig[-1].date()} ({ago(allsig[-1])}거래일 전)")
    live=[ts for ts in allsig if ago(ts)<60]
    print(f"  아직 60일 만기 전인 신호일 수: {len(live)}")
    if live:
        rows=[]
        for ts in live:
            for c,s in ev.items():
                if ts in s.index and bool(s.loc[ts]):
                    d=data[c]; i=d.index.get_loc(ts)
                    to=float((d["close"]*d["volume"]).rolling(20).mean().iloc[i])/1e8
                    rows.append(dict(신호일=ts.date(), code=c, 종목=uni["name"].get(c,"?"),
                        체결가=float(d["open"].iloc[i+1]) if i+1<len(d) else np.nan,
                        현재가=float(d["close"].iloc[-1]), 거래대금억=to, 남은일=60-ago(ts)))
        r=pd.DataFrame(rows).sort_values("신호일")
        r["목표가"]=r["체결가"]*1.2; r["손익%"]=(r["현재가"]/r["체결가"]-1)*100
        print(r.to_string(index=False,formatters={"체결가":"{:,.0f}".format,"현재가":"{:,.0f}".format,
            "목표가":"{:,.0f}".format,"손익%":"{:+.1f}".format,"거래대금억":"{:.1f}".format}))
    print()

print("=== 신호 공백이 얼마나 흔한가 (2010~2026) ===")
idx=pd.DatetimeIndex(sorted({t for d in data.values() for t in d.index}))
idx=idx[idx>=pd.Timestamp("2010-01-01",tz=idx.tz)]
pos={ts:i for i,ts in enumerate(idx)}
for name,kind in [("⑦","O"),("⑥","I")]:
    ev=build(kind)
    days=sorted({ts for c,s in ev.items() for ts in s.index[s.to_numpy()] if ts in pos})
    g=np.diff([pos[t] for t in days])
    cur = len(idx)-1-pos[days[-1]]
    print(f"{name}  신호가 뜬 날 {len(days)}일 · 공백 중앙 {np.median(g):.0f}거래일 · "
          f"상위10% {np.percentile(g,90):.0f}일 · 최장 {g.max()}일")
    print(f"    지금 공백 {cur}거래일 → 과거 공백들 중 {(g<cur).mean():.0%} 보다 길다")
