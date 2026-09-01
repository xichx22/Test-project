"""v1 을 10년 굴리면 — 10년 창을 한 달씩 밀어가며 전부 굴린다."""
import pickle, numpy as np, pandas as pd
from tsignal.evaluation.exitlab import Exit, resolve
from tsignal.datasource.csv_source import CsvDataSource
from tsignal.datasource.base import Interval
S="/tmp/claude-0/-home-user-Test-project/e2567955-0c39-5157-a915-38387e9cd240/scratchpad"
parts=pickle.load(open(f"{S}/parts16.pkl","rb"))
blob=pickle.load(open(f"{S}/cache16.pkl","rb")); data,feats=blob["data"],blob["feats"]
TZ=next(iter(data.values())).index.tz
FULL=pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
LO=pd.Timestamp("2010-01-01",tz=TZ); HI=pd.Timestamp("2026-08-28",tz=TZ)
FULL=FULL[(FULL>=LO)&(FULL<=HI)]
CLOSE=pd.DataFrame({c:d["close"] for c,d in data.items()}).reindex(FULL).ffill()
CODEIX={c:i for i,c in enumerate(CLOSE.columns)}; DAYIX={t:i for i,t in enumerate(FULL)}
CLOSEV=CLOSE.to_numpy(float)
idx=CsvDataSource("data_asset").candles("069500",Interval.D1)["close"].reindex(FULL).ffill()
START,MONTHLY,SLOT,CASH=1_000_000.,400_000.,1_000_000/3,0.025

def deposit_days(cal):
    days,seen=set(),set()
    for ts in cal:
        k=(ts.year,ts.month)
        if ts.day>=21 and k not in seen: seen.add(k); days.add(ts)
    return days-{min(days)} if days else set()
def irr(flows):
    t0=flows[0][0]; yrs=np.array([(d-t0).days/365.25 for d,_ in flows]); am=np.array([a for _,a in flows])
    f=lambda r: float((am/((1+r)**yrs)).sum()); lo,hi=-.95,5.
    if f(lo)*f(hi)>0: return np.nan
    for _ in range(200):
        m=(lo+hi)/2
        if f(lo)*f(m)<=0: hi=m
        else: lo=m
    return (lo+hi)/2
def dca(by,cal,rk):
    dep=deposit_days(cal); cash,pos,flows=START,[],[(cal[0],-START)]
    d1=(1+CASH)**(1/252)-1; curve=[]; invested=[]
    for ts in cal:
        cash*=1+d1
        if ts in dep: cash+=MONTHLY; flows.append((ts,-MONTHLY))
        keep=[]
        for p in pos:
            if p[2]<=ts: cash+=p[1]*p[3]
            else: keep.append(p)
        pos=keep
        rows=by.get(ts)
        if rows:
            for row in sorted(rows,key=lambda r:-rk.get((r[0],ts),0.)):
                if cash<SLOT: continue
                code,entry,xd,xp=row
                if not np.isfinite(entry) or entry<=0: continue
                cash-=SLOT; pos.append((code,SLOT/entry,xd,xp))
        r=DAYIX[ts]; mv=0.
        for p in pos:
            v=CLOSEV[r,CODEIX[p[0]]]; mv+=p[1]*(v if np.isfinite(v) else p[3])
        curve.append(cash+mv); invested.append(mv/(cash+mv) if cash+mv>0 else 0)
    eq=np.array(curve); flows.append((cal[-1],float(eq[-1])))
    dd=float((eq/np.maximum.accumulate(eq)-1).min())
    return irr(flows),dd,float(eq[-1]),float(np.mean(invested)),len(flows)-2
def bench(cal,ser):
    dep=deposit_days(cal); px=ser.reindex(cal).ffill(); u=START/px.iloc[0]
    flows=[(cal[0],-START)]; last=0.
    for ts in cal:
        if ts in dep: u+=MONTHLY/px.loc[ts]; flows.append((ts,-MONTHLY))
        last=u*px.loc[ts]
    flows.append((cal[-1],float(last))); return irr(flows),float(last)


def combine(kind,names,win):
    if kind=="U":
        out={}
        for n in names:
            for c,x in parts[n].items(): out[c]= x if c not in out else (out[c]|x)
        return out
    a,b=parts[names[0]],parts[names[1]]; out={}
    for c in sorted(set(a)&set(b)):
        sa,sb=a[c],b[c]
        if kind=="I":
            hit=(sa&sb) if win==0 else ((sa & sb.rolling(win+1,min_periods=1).max().astype(bool))
                                        |(sb & sa.rolling(win+1,min_periods=1).max().astype(bool)))
        else:
            hit=sa & sb.rolling(win+1,min_periods=1).max().astype(bool)
        hit=hit.fillna(False)
        if hit.any(): out[c]=hit
    return out

CASES={
 "⑦ MFI→컵앤핸들":     ("O",["컵앤핸들","MFI과매도"],20),
 "⑥ 컵∩MFI(20일)":    ("I",["컵앤핸들","MFI과매도"],20),
 "③ 컵∩거래량급등":      ("I",["컵앤핸들","거래량급등"],0),
 "④ 컵∩상승삼각형":      ("I",["컵앤핸들","상승삼각형"],0),
 "⑤ 컵∩20일신고가":     ("I",["컵앤핸들","20일신고가"],0),
 "① 컵앤핸들 단독":      ("U",["컵앤핸들"],0),
 "② 컵∪MFI":         ("U",["컵앤핸들","MFI과매도"],0),
 "⑧ MFI 단독":        ("U",["MFI과매도"],0),
 "⑪ MFI∪RSI∪윌리엄스":  ("U",["MFI과매도","RSI과매도","윌리엄스과매도"],0),
}
starts=[]; seen=set()
for ts in FULL:
    k=(ts.year,ts.month)
    if k not in seen: seen.add(k); starts.append(ts)
WIN=[]
for s0 in starts:
    e0=s0+pd.DateOffset(years=10); cal=FULL[(FULL>=s0)&(FULL<=e0)]
    if len(cal)>=2400: WIN.append(cal)
print(f"10년 창 {len(WIN)}개  ({WIN[0][0].date()}~ 부터 {WIN[-1][0].date()}~ 까지)\n")

print(f"10년 창 {len(WIN)}개  ({WIN[0][0].date()}~ 부터 {WIN[-1][0].date()}~ 까지)")
B=[]
for cal in WIN:
    bi,fi=bench(cal,idx)
    bc,fc=bench(cal,pd.Series(np.cumprod(np.full(len(cal),(1+CASH)**(1/252))),index=cal))
    B.append((bi,fi,bc,fc))
bi=np.array([x[0] for x in B]); bc=np.array([x[2] for x in B])
print(f"지수 적립 IRR 중앙 {np.median(bi):+.2%} · 현금 {np.median(bc):+.2%} · "
      f"넣은 돈 4,860만원\n", flush=True)
rows=[]
for cname,spec in CASES.items():
    ev=combine(*spec)
    tr=resolve(ev,data,feats,rule=Exit("익절20%",take_profit=0.20,horizon=60))
    ent=pd.DatetimeIndex(tr["진입일"]); rng=np.random.default_rng(0)
    rk={(c,t):float(v) for c,t,v in zip(tr["code"],ent,rng.random(len(tr)))}
    by_all={}
    for code,e,x,ep,xp in zip(tr["code"],ent,pd.DatetimeIndex(tr["청산일"]),tr["진입"],tr["청산"]):
        by_all.setdefault(e,[]).append((code,ep,x,xp*(1-0.0014)))
    R=[]
    for cal in WIN:
        by={k:v for k,v in by_all.items() if cal[0]<=k<=cal[-1]}
        t3=[]
        for seed in range(3):
            r2=np.random.default_rng(seed)
            rk2={k:float(r2.random()) for k in rk} if seed else rk
            t3.append(dca(by,cal,rk2))
        R.append((np.mean([x[0] for x in t3]),np.mean([x[1] for x in t3]),
                  np.mean([x[2] for x in t3]),np.mean([x[3] for x in t3]),
                  max(x[0] for x in t3)-min(x[0] for x in t3)))
    a=np.array([x[0] for x in R]); fin=np.array([x[2] for x in R])
    inv=np.array([x[3] for x in R]); dd=np.array([x[1] for x in R])
    sp=np.array([x[4] for x in R])
    row=dict(방식=cname,신호=len(tr),IRR중앙=np.median(a),IRR최악=a.min(),
             최종중앙=np.median(fin),투자비중=np.median(inv),낙폭중앙=np.median(dd),
             낙폭최악=dd.min(),현금이김=(a>bc).mean(),지수이김=(a>bi).mean(),폭=np.median(sp))
    rows.append(row)
    print(f"{cname:18s} 신호{len(tr):6,}  IRR중앙 {row['IRR중앙']:+6.2%}  최악 {row['IRR최악']:+6.2%}  "
          f"최종 {row['최종중앙']/1e4:6,.0f}만  투자비중 {row['투자비중']:5.1%}  "
          f"낙폭 {row['낙폭중앙']:6.1%}/{row['낙폭최악']:6.1%}  현금 {row['현금이김']:4.0%}  "
          f"지수 {row['지수이김']:4.0%}  폭 {row['폭']*100:4.1f}%p", flush=True)
pd.DataFrame(rows).to_pickle(f"{S}/tenyr2.pkl")
print("\n저장 완료", flush=True)
