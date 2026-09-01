"""⑤ 컵앤핸들 + 20일 신고가 — 연도별 성적, 최근 신호, 오늘 후보."""
import pickle, numpy as np, pandas as pd
from tsignal.evaluation.exitlab import Exit, resolve
S="/tmp/claude-0/-home-user-Test-project/e2567955-0c39-5157-a915-38387e9cd240/scratchpad"
parts=pickle.load(open(f"{S}/parts16.pkl","rb"))
blob=pickle.load(open(f"{S}/cache16.pkl","rb")); data,feats=blob["data"],blob["feats"]
uni=pd.read_csv("data_wide/universe.csv",dtype=str).set_index("code")
cup,hi20=parts["컵앤핸들"],parts["20일신고가"]
idxcal=pd.DatetimeIndex(sorted({t for d in data.values() for t in d.index}))

def build(names):
    a,b=parts[names[0]],parts[names[1]]; out={}
    for c in sorted(set(a)&set(b)):
        h=(a[c]&b[c]).fillna(False)
        if h.any(): out[c]=h
    return out
ev5=build(["컵앤핸들","20일신고가"])
ev1={c:s for c,s in cup.items() if s.any()}
RULE=Exit("익절20%",take_profit=0.20,horizon=60)
for nm,ev in [("⑤ 컵앤핸들+20일신고가",ev5),("① 컵앤핸들만",ev1)]:
    tr=resolve(ev,data,feats,rule=RULE); tr["진입일"]=pd.DatetimeIndex(tr["진입일"])
    r=(tr["청산"]*(1-0.0014)/tr["진입"]-1)
    print(f"=== {nm} ===  전체 {len(tr):,}건  평균 {r.mean():+.2%}  중앙 {r.median():+.2%}  "
          f"승률 {(r>0).mean():.0%}")
    print("   사유:", {k:int(v) for k,v in tr['사유'].value_counts().items()},
          f" 평균 보유 {tr['보유봉'].mean():.0f}봉")
    yr=tr.groupby(tr["진입일"].dt.year).apply(
        lambda t: pd.Series({"건수":len(t),
            "평균":(t["청산"]*(1-0.0014)/t["진입"]-1).mean(),
            "승률":((t["청산"]*(1-0.0014)/t["진입"]-1)>0).mean()}),include_groups=False)
    print("   연도별:", "  ".join(f"{int(i)} {int(v['건수']):3d}건 {v['평균']:+.1%}" for i,v in yr.iterrows()))
    print()
# 20일신고가 필터가 걸러낸 것은 무엇인가
only1=[]; both=[]
for c,s in cup.items():
    h5=ev5.get(c)
    for ts in s.index[s.to_numpy()]:
        (both if (h5 is not None and ts in h5.index and bool(h5.loc[ts])) else only1).append((c,ts))
print(f"컵앤핸들 신호 {len(only1)+len(both):,}건 중 20일신고가를 겸한 것 {len(both):,}건 "
      f"({len(both)/(len(only1)+len(both))*100:.0f}%)\n")
# 오늘 후보
LAST=idxcal[-1]
print(f"=== 오늘({LAST.date()}) 기준 ⑤ 신호 ===")
rows=[]
for c,s in ev5.items():
    for ts in s.index[s.to_numpy()][-3:]:
        ago=len(idxcal)-1-idxcal.get_loc(ts)
        if ago>=60: continue
        d=data[c]; i=d.index.get_loc(ts)
        to=float((d["close"]*d["volume"]).rolling(20).mean().iloc[i])/1e8
        if to<3: continue
        rows.append(dict(신호일=ts.date(),code=c,종목=uni["name"].get(c,"?"),
            체결가=float(d["open"].iloc[i+1]) if i+1<len(d) else np.nan,
            현재가=float(d["close"].iloc[-1]),거래대금억=to,남은일=60-ago))
if rows:
    r=pd.DataFrame(rows).sort_values("신호일")
    r["목표가"]=r["체결가"]*1.2; r["손익%"]=(r["현재가"]/r["체결가"]-1)*100
    print(r.to_string(index=False,formatters={"체결가":"{:,.0f}".format,"현재가":"{:,.0f}".format,
        "목표가":"{:,.0f}".format,"손익%":"{:+.1f}".format,"거래대금억":"{:.1f}".format}))
else:
    allsig=sorted({ts for c,s in ev5.items() for ts in s.index[s.to_numpy()]})
    print(f"  없음. 마지막 신호 {allsig[-1].date()} ({len(idxcal)-1-idxcal.get_loc(allsig[-1])}거래일 전)")
