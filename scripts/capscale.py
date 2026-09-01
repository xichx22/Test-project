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
CAPS=[1_000_000.,10_000_000.,30_000_000.,50_000_000.]

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
 "⑦ 과매도 후 컵앤핸들": ("O",["컵앤핸들","MFI과매도"],20),
 "⑤ 컵앤핸들+20일신고가":("I",["컵앤핸들","20일신고가"],0),
 "① 컵앤핸들만":        ("U",["컵앤핸들"],0),
 "② 컵앤핸들 또는 과매도":("U",["컵앤핸들","MFI과매도"],0),
}
starts=[]; seen=set()
for ts in FULL:
    k=(ts.year,ts.month)
    if k not in seen: seen.add(k); starts.append(ts)
WIN=[]
for s0 in starts:
    e0=s0+pd.DateOffset(years=1); cal=FULL[(FULL>=s0)&(FULL<=e0)]
    if len(cal)>=200: WIN.append(cal)
print(f"1년 창 {len(WIN)}개  ({WIN[0][0].date()} ~ {WIN[-1][0].date()} 시작)", flush=True)
B=[]
for cal in WIN:
    B.append((bench(cal,idx)[0],
              bench(cal,pd.Series(np.cumprod(np.full(len(cal),(1+CASH)**(1/252))),index=cal))[0]))
bi=np.array([x[0] for x in B]); bc=np.array([x[1] for x in B])
print(f"지수 적립 IRR 중앙 {np.median(bi):+.2%} · 내린 창 {(bi<0).mean():.0%} · 현금 {np.median(bc):+.2%}\n", flush=True)
import tsignal.evaluation.exitlab as _x
TR={}
for cname,spec in CASES.items():
    ev=combine(*spec); tr=resolve(ev,data,feats,rule=Exit("익절20%",take_profit=0.20,horizon=60))
    ent=pd.DatetimeIndex(tr["진입일"]); rng=np.random.default_rng(0)
    rk={(c,t):float(v) for c,t,v in zip(tr["code"],ent,rng.random(len(tr)))}
    ba={}
    for code,e,x,ep,xp in zip(tr["code"],ent,pd.DatetimeIndex(tr["청산일"]),tr["진입"],tr["청산"]):
        ba.setdefault(e,[]).append((code,ep,x,xp*(1-0.0014)))
    TR[cname]=(len(tr),rk,ba)
    print(f"{cname}: 매매 {len(tr):,}건 준비", flush=True)
print(flush=True)
rows=[]
for cap in CAPS:
    globals()["START"]=cap
    print(f"===== 시작 자금 {cap/1e4:,.0f}만원 (매월 40만 추가 · 한 자리 33.3만) =====", flush=True)
    for cname,(nsig,rk,ba) in TR.items():
        R=[]
        for cal in WIN:
            by={k:v for k,v in ba.items() if cal[0]<=k<=cal[-1]}
            t3=[dca(by,cal,({k:float(np.random.default_rng(sd).random()) for k in rk} if sd else rk))
                for sd in range(3)]
            R.append((np.mean([x[0] for x in t3]),np.mean([x[3] for x in t3]),
                      max(x[0] for x in t3)-min(x[0] for x in t3),
                      sum(len(v) for v in by.values())))
        a=np.array([x[0] for x in R]); inv=np.array([x[1] for x in R]); sp=np.array([x[2] for x in R])
        ok=np.isfinite(a); down=ok&(bi<0)
        row=dict(자금=cap,방식=cname,신호=nsig,IRR중앙=np.median(a[ok]),
                 비중=np.median(inv[ok]),현금이김=(a[ok]>bc[ok]).mean(),
                 지수이김=(a[ok]>bi[ok]).mean(),
                 내린창=(a[down]>bc[down]).mean(),내린창IRR=np.median(a[down]),
                 최악창=a[ok].min(),폭=np.median(sp[ok]))
        rows.append(row)
        print(f"  {cname:20s} IRR중앙 {row['IRR중앙']:+7.2%}  비중 {row['비중']:5.1%}  "
              f"현금 {row['현금이김']:4.0%}  지수 {row['지수이김']:4.0%}  "
              f"내린창 {row['내린창']:4.0%} ({row['내린창IRR']:+6.2%})  "
              f"최악창 {row['최악창']:+7.2%}  폭 {row['폭']*100:4.1f}%p", flush=True)
    print(flush=True)
pd.DataFrame(rows).to_pickle(f"{S}/capscale.pkl")
print("저장 완료", flush=True)
