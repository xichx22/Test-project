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

# ⑦ 신호
cup,mfi=parts["컵앤핸들"],parts["MFI과매도"]
ev={}
for code,sc in cup.items():
    sm=mfi.get(code)
    if sm is None: continue
    h=(sc & sm.rolling(21,min_periods=1).max().astype(bool)).fillna(False)
    if h.any(): ev[code]=h
tr=resolve(ev,data,feats,rule=Exit("익절20%",take_profit=0.20,horizon=60))
ent=pd.DatetimeIndex(tr["진입일"])
rng=np.random.default_rng(0)
rk={(c,t):float(v) for c,t,v in zip(tr["code"],ent,rng.random(len(tr)))}
by_all={}
for code,e,x,ep,xp in zip(tr["code"],ent,pd.DatetimeIndex(tr["청산일"]),tr["진입"],tr["청산"]):
    by_all.setdefault(e,[]).append((code,ep,x,xp*(1-0.0014)))

starts=[]; seen=set()
for ts in FULL:
    k=(ts.year,ts.month)
    if k not in seen: seen.add(k); starts.append(ts)
WIN=[]
for s0 in starts:
    e0=s0+pd.DateOffset(years=10); cal=FULL[(FULL>=s0)&(FULL<=e0)]
    if len(cal)>=2400: WIN.append(cal)
print(f"10년 창 {len(WIN)}개  ({WIN[0][0].date()}~ 부터 {WIN[-1][0].date()}~ 까지)\n")
out=[]
for cal in WIN:
    lo_,hi_=cal[0],cal[-1]
    by={k:v for k,v in by_all.items() if lo_<=k<=hi_}
    ir3=[]
    for seed in range(3):
        r2=np.random.default_rng(seed)
        rk2={k:float(r2.random()) for k in rk} if seed else rk
        ir3.append(dca(by,cal,rk2))
    a=np.mean([x[0] for x in ir3]); fin=np.mean([x[2] for x in ir3])
    dd=np.mean([x[1] for x in ir3]); inv=np.mean([x[3] for x in ir3])
    ndep=ir3[0][4]; paid=START+MONTHLY*ndep
    bi,fi=bench(cal,idx)
    bc,fc=bench(cal,pd.Series(np.cumprod(np.full(len(cal),(1+CASH)**(1/252))),index=cal))
    nsig=sum(len(v) for v in by.values())
    out.append(dict(시작=lo_.date(),끝=hi_.date(),넣은돈=paid,최종=fin,IRR=a,낙폭=dd,
        투자비중=inv,신호=nsig,지수IRR=bi,지수최종=fi,현금IRR=bc,현금최종=fc,
        폭=max(x[0] for x in ir3)-min(x[0] for x in ir3)))
d=pd.DataFrame(out)
pd.to_pickle(d,f"{S}/tenyr.pkl")
M=1e4
print(f"넣은 돈 (중앙) {d['넣은돈'].median()/M:,.0f}만원   ·  매수 신호 (중앙) {d['신호'].median():.0f}건\n")
print("           IRR중앙    IRR최악    IRR최고   최종중앙   최종최악   최종최고")
for nm,ic,fc_ in [("v1 ⑦","IRR","최종"),("지수 적립","지수IRR","지수최종"),("현금","현금IRR","현금최종")]:
    print(f"{nm:9s}{d[ic].median():>8.2%}{d[ic].min():>10.2%}{d[ic].max():>10.2%}"
          f"{d[fc_].median()/M:>10,.0f}만{d[fc_].min()/M:>9,.0f}만{d[fc_].max()/M:>9,.0f}만")
print(f"\n지수를 이긴 창 {(d['IRR']>d['지수IRR']).mean():.0%}   현금을 이긴 창 {(d['IRR']>d['현금IRR']).mean():.0%}")
print(f"최대 낙폭 (중앙) {d['낙폭'].median():.1%}  최악 {d['낙폭'].min():.1%}")
print(f"돈이 실제로 주식에 들어가 있던 비중 (중앙) {d['투자비중'].median():.1%}")
print(f"뽑기 폭 (중앙) {d['폭'].median()*100:.1f}%p")
