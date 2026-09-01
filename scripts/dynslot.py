"""자리 수를 고정하고 종목당 금액을 계좌에 비례시킨다.

  · 자리 수 N 고정 (⑦ 의 동시보유 분포 기준)
  · 종목당 금액 = 그 시점 계좌 총자산 ÷ N  → 계좌가 크면 종목당 금액도 크다
  · 매월 21일 35만원 입금
  · 익절 +20% / 60거래일 만기 / 손절 없음
"""
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
MONTHLY, CASH = 350_000., 0.025
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
def run(by,cal,rk,N,START):
    dep=deposit_days(cal); cash,pos,flows=START,[],[(cal[0],-START)]
    d1=(1+CASH)**(1/252)-1; curve=[]; invested=[]; bought=0; skipped=0
    for ts in cal:
        cash*=1+d1
        if ts in dep: cash+=MONTHLY; flows.append((ts,-MONTHLY))
        keep=[]
        for p in pos:
            if p[2]<=ts: cash+=p[1]*p[3]
            else: keep.append(p)
        pos=keep
        r=DAYIX[ts]
        mv=sum(p[1]*(CLOSEV[r,CODEIX[p[0]]] if np.isfinite(CLOSEV[r,CODEIX[p[0]]]) else p[3]) for p in pos)
        rows=by.get(ts)
        if rows:
            for row in sorted(rows,key=lambda x:-rk.get((x[0],ts),0.)):
                if len(pos)>=N: skipped+=1; continue
                slot=(cash+mv)/N                      # 종목당 금액 = 총자산 ÷ 자리수
                if cash<slot: skipped+=1; continue
                code,entry,xd,xp=row
                if not np.isfinite(entry) or entry<=0: continue
                cash-=slot; pos.append((code,slot/entry,xd,xp)); bought+=1
                mv=sum(p[1]*(CLOSEV[r,CODEIX[p[0]]] if np.isfinite(CLOSEV[r,CODEIX[p[0]]]) else p[3]) for p in pos)
        mv=sum(p[1]*(CLOSEV[r,CODEIX[p[0]]] if np.isfinite(CLOSEV[r,CODEIX[p[0]]]) else p[3]) for p in pos)
        curve.append(cash+mv); invested.append(mv/(cash+mv) if cash+mv>0 else 0)
    eq=np.array(curve); flows.append((cal[-1],float(eq[-1])))
    dd=float((eq/np.maximum.accumulate(eq)-1).min())
    return irr(flows),dd,float(eq[-1]),float(np.mean(invested)),bought,skipped
def bench(cal,ser,START):
    dep=deposit_days(cal); px=ser.reindex(cal).ffill(); u=START/px.iloc[0]
    flows=[(cal[0],-START)]; last=0.
    for ts in cal:
        if ts in dep: u+=MONTHLY/px.loc[ts]; flows.append((ts,-MONTHLY))
        last=u*px.loc[ts]
    flows.append((cal[-1],float(last))); return irr(flows)
cup,mfi,hi20=parts["컵앤핸들"],parts["MFI과매도"],parts["20일신고가"]
def build(kind):
    out={}
    for c,sc in cup.items():
        if kind=="⑦":
            sm=mfi.get(c)
            if sm is None: continue
            h=sc & sm.rolling(21,min_periods=1).max().astype(bool)
        elif kind=="⑤":
            sh=hi20.get(c)
            if sh is None: continue
            h=sc & sh
        else: h=sc
        h=h.fillna(False)
        if h.any(): out[c]=h
    return out
RULE=Exit("익절20%",take_profit=0.20,horizon=60)
TR={}
for k in ("⑦","⑤","①"):
    tr=resolve(build(k),data,feats,rule=RULE)
    ent=pd.DatetimeIndex(tr["진입일"]); rng=np.random.default_rng(0)
    rk={(c,t):float(v) for c,t,v in zip(tr["code"],ent,rng.random(len(tr)))}
    ba={}
    for code,e,x,ep,xp in zip(tr["code"],ent,pd.DatetimeIndex(tr["청산일"]),tr["진입"],tr["청산"]):
        ba.setdefault(e,[]).append((code,ep,x,xp*(1-0.0014)))
    TR[k]=(len(tr),rk,ba)
starts=[]; seen=set()
for ts in FULL:
    key=(ts.year,ts.month)
    if key not in seen: seen.add(key); starts.append(ts)
WIN=[]
for s0 in starts:
    e0=s0+pd.DateOffset(years=1); c=FULL[(FULL>=s0)&(FULL<=e0)]
    if len(c)>=200: WIN.append(c)
print(f"1년 창 {len(WIN)}개 · 매월 21일 35만원 입금 · 종목당 금액 = 총자산 ÷ 자리수\n",flush=True)
out=[]
for START in (1_000_000.,10_000_000.,30_000_000.):
    bi=np.array([bench(c,idx,START) for c in WIN])
    bc=np.array([bench(c,pd.Series(np.cumprod(np.full(len(c),(1+CASH)**(1/252))),index=c),START) for c in WIN])
    print(f"===== 시작 {START/1e4:,.0f}만원 · 지수 {np.median(bi):+.2%} · 현금 {np.median(bc):+.2%} =====",flush=True)
    for k in ("⑦","⑤","①"):
        nsig,rk,ba=TR[k]
        for N in (3,5,10,15,20,30):
            R=[]
            for cal in WIN:
                by={t:v for t,v in ba.items() if cal[0]<=t<=cal[-1]}
                t3=[run(by,cal,({kk:float(np.random.default_rng(sd).random()) for kk in rk} if sd else rk),N,START)
                    for sd in range(3)]
                R.append((np.mean([x[0] for x in t3]),np.mean([x[1] for x in t3]),
                          np.mean([x[3] for x in t3]),max(x[0] for x in t3)-min(x[0] for x in t3),
                          np.mean([x[4] for x in t3]),np.mean([x[5] for x in t3])))
            a=np.array([x[0] for x in R]); ok=np.isfinite(a); down=ok&(bi<0)
            dd=np.array([x[1] for x in R]); inv=np.array([x[2] for x in R])
            sp=np.array([x[3] for x in R]); bg=np.array([x[4] for x in R]); sk=np.array([x[5] for x in R])
            take=bg.sum()/(bg.sum()+sk.sum()) if (bg.sum()+sk.sum()) else 1
            row=dict(자금=START,규칙=k,자리=N,IRR=np.median(a[ok]),비중=np.median(inv[ok]),
                     낙폭=np.median(dd[ok]),최악=a[ok].min(),현금이김=(a[ok]>bc[ok]).mean(),
                     지수이김=(a[ok]>bi[ok]).mean(),내린창=(a[down]>bc[down]).mean(),
                     폭=np.median(sp[ok]),수용률=take,매수=bg.mean())
            out.append(row)
            print(f"  {k} {N:>2d}자리  IRR {row['IRR']:+7.2%}  비중 {row['비중']:5.1%}  "
                  f"낙폭 {row['낙폭']:6.1%}  최악 {row['최악']:+7.2%}  현금 {row['현금이김']:4.0%}  "
                  f"지수 {row['지수이김']:4.0%}  내린창 {row['내린창']:4.0%}  폭 {row['폭']*100:4.1f}%p  "
                  f"수용 {take:4.0%}  연매수 {row['매수']:.0f}건",flush=True)
        print(flush=True)
    pd.DataFrame(out).to_pickle(f"{S}/dynslot.pkl")
print("저장 완료",flush=True)
