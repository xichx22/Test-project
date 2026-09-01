"""⑤ + ⑦ 혼합 — 사용자 설계.

  · ⑤ 신호  → 종목당 33만원 고정
  · ⑦ 신호  → 그날 보유현금의 1/3
  · 같은 날 둘 다 뜨면 ⑦ 이 먼저
  · A안: ⑦ 신호가 뜨면 ⑤ 보유분을 전부 팔아 현금을 만든 뒤 ⑦ 매수
    B안: 강제 청산 없음. 있는 현금의 1/3 로만 ⑦ 매수
    C안: 목표를 자산의 1/3 으로 잡고, 현금이 모자라면 ⑤ 중에서
         청산 목표일이 가까운 순서로 필요한 만큼만 판다
  · 청산은 양쪽 다 +20% / 60거래일
  · 비교군: ⑤ 단독(33만 고정), ⑦ 단독(자산÷3)
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
FULL=FULL[(FULL>=pd.Timestamp("2010-01-01",tz=TZ))&(FULL<=pd.Timestamp("2026-08-28",tz=TZ))]
CLOSE=pd.DataFrame({c:d["close"] for c,d in data.items()}).reindex(FULL).ffill()
OPEN=pd.DataFrame({c:d["open"] for c,d in data.items()}).reindex(FULL).ffill()
CODEIX={c:i for i,c in enumerate(CLOSE.columns)}; DAYIX={t:i for i,t in enumerate(FULL)}
CV=CLOSE.to_numpy(float); OV=OPEN.to_numpy(float)
idx=CsvDataSource("data_asset").candles("069500",Interval.D1)["close"].reindex(FULL).ffill()
MONTHLY,CASH,SLOT5=350_000.,0.025,330_000.
def dep_days(cal):
    d,seen=set(),set()
    for ts in cal:
        k=(ts.year,ts.month)
        if ts.day>=21 and k not in seen: seen.add(k); d.add(ts)
    return d-{min(d)} if d else set()
def irr(fl):
    t0=fl[0][0]; y=np.array([(d-t0).days/365.25 for d,_ in fl]); a=np.array([x for _,x in fl])
    f=lambda r: float((a/((1+r)**y)).sum()); lo,hi=-.95,5.
    if f(lo)*f(hi)>0: return np.nan
    for _ in range(200):
        m=(lo+hi)/2
        if f(lo)*f(m)<=0: hi=m
        else: lo=m
    return (lo+hi)/2
cup,mfi,hi20=parts["컵앤핸들"],parts["MFI과매도"],parts["20일신고가"]
def build(kind):
    out={}
    for c,sc in cup.items():
        if kind=="⑦":
            sm=mfi.get(c)
            if sm is None: continue
            h=sc & sm.rolling(21,min_periods=1).max().astype(bool)
        else:
            sh=hi20.get(c)
            if sh is None: continue
            h=sc & sh
        h=h.fillna(False)
        if h.any(): out[c]=h
    return out
RULE=Exit("익절20%",take_profit=0.20,horizon=60)
BY={}
for k in ("⑦","⑤"):
    tr=resolve(build(k),data,feats,rule=RULE)
    ent=pd.DatetimeIndex(tr["진입일"]); rng=np.random.default_rng(0)
    rk={(c,t):float(v) for c,t,v in zip(tr["code"],ent,rng.random(len(tr)))}
    ba={}
    for code,e,x,ep,xp in zip(tr["code"],ent,pd.DatetimeIndex(tr["청산일"]),tr["진입"],tr["청산"]):
        ba.setdefault(e,[]).append((code,ep,x,xp*(1-0.0014)))
    BY[k]=(ba,rk,len(tr))

def sim(cal,mode,START,rk5,rk7,N7=3):
    """mode: 'A' 강제청산 · 'B' 강제청산없음 · '⑤' 단독 · '⑦' 단독(자산÷3)"""
    dep=dep_days(cal); cash,pos,fl=START,[],[(cal[0],-START)]
    d1=(1+CASH)**(1/252)-1; curve=[]; inv=[]; n5=n7=0; n5cut=0
    b5,b7=BY["⑤"][0],BY["⑦"][0]
    for ts in cal:
        cash*=1+d1
        if ts in dep: cash+=MONTHLY; fl.append((ts,-MONTHLY))
        keep=[]
        for p in pos:
            if p[2]<=ts: cash+=p[1]*p[3]
            else: keep.append(p)
        pos=keep
        r=DAYIX[ts]
        mv=lambda: sum(p[1]*(CV[r,CODEIX[p[0]]] if np.isfinite(CV[r,CODEIX[p[0]]]) else p[3]) for p in pos)
        s7=b7.get(ts) if mode in ("A","B","C","⑦") else None
        s5=b5.get(ts) if mode in ("A","B","C","⑤") else None
        if s7:
            if mode=="A":                       # ⑦ 뜨면 ⑤ 보유분 시가 청산
                keep=[]
                for p in pos:
                    if p[4]=="⑤":
                        px=OV[r,CODEIX[p[0]]]
                        cash+=p[1]*(px if np.isfinite(px) and px>0 else CV[r,CODEIX[p[0]]])*(1-0.0014)
                    else: keep.append(p)
                pos=keep
            for row in sorted(s7,key=lambda x:-rk7.get((x[0],ts),0.)):
                if mode=="⑦":
                    slot=(cash+mv())/N7
                    if len(pos)>=N7: continue
                elif mode=="C":
                    if sum(1 for p in pos if p[4]=="⑦")>=N7: continue
                    slot=(cash+mv())/N7
                    if cash<slot:
                        five=sorted([p for p in pos if p[4]=="⑤"],key=lambda p:p[2])
                        for p in five:
                            if cash>=slot: break
                            px=OV[r,CODEIX[p[0]]]
                            px=px if np.isfinite(px) and px>0 else CV[r,CODEIX[p[0]]]
                            if not np.isfinite(px) or px<=0: continue
                            cash+=p[1]*px*(1-0.0014); pos.remove(p); n5cut+=1
                        slot=min(slot,cash)
                else:
                    slot=cash/3
                if cash<slot or slot<=0: continue
                code,e,xd,xp=row
                if not np.isfinite(e) or e<=0: continue
                cash-=slot; pos.append((code,slot/e,xd,xp,"⑦")); n7+=1
        if s5:
            for row in sorted(s5,key=lambda x:-rk5.get((x[0],ts),0.)):
                if cash<SLOT5: continue
                code,e,xd,xp=row
                if not np.isfinite(e) or e<=0: continue
                cash-=SLOT5; pos.append((code,SLOT5/e,xd,xp,"⑤")); n5+=1
        m=mv(); curve.append(cash+m); inv.append(m/(cash+m) if cash+m>0 else 0)
    eq=np.array(curve); fl.append((cal[-1],float(eq[-1])))
    return irr(fl),float((eq/np.maximum.accumulate(eq)-1).min()),float(eq[-1]),float(np.mean(inv)),n5,n7,n5cut
def bench(cal,ser,START):
    dep=dep_days(cal); px=ser.reindex(cal).ffill(); u=START/px.iloc[0]
    fl=[(cal[0],-START)]; last=0.
    for ts in cal:
        if ts in dep: u+=MONTHLY/px.loc[ts]; fl.append((ts,-MONTHLY))
        last=u*px.loc[ts]
    fl.append((cal[-1],float(last))); return irr(fl)
starts=[]; seen=set()
for ts in FULL:
    k=(ts.year,ts.month)
    if k not in seen: seen.add(k); starts.append(ts)
WIN=[c for c in (FULL[(FULL>=s)&(FULL<=s+pd.DateOffset(years=1))] for s in starts) if len(c)>=200]
print(f"1년 창 {len(WIN)}개 · 매월 35만 입금 · ⑤=33만 고정 · ⑦=현금÷3\n",flush=True)
_,rk5,_=BY["⑤"]; _,rk7,_=BY["⑦"]
for START in (1_000_000.,10_000_000.):
    bi=np.array([bench(c,idx,START) for c in WIN])
    bc=np.array([bench(c,pd.Series(np.cumprod(np.full(len(c),(1+CASH)**(1/252))),index=c),START) for c in WIN])
    print(f"===== 시작 {START/1e4:,.0f}만원 · 지수 {np.median(bi):+.2%} · 현금 {np.median(bc):+.2%} =====",flush=True)
    for mode,lab in [("C","C안 혼합(만기임박순 필요분만 매도)"),
                     ("B","B안 혼합(강제청산 없음)"),
                     ("A","A안 혼합(⑦뜨면 ⑤전부 청산)"),
                     ("⑤","⑤ 단독 33만 고정"),("⑦","⑦ 단독 자산÷3")]:
        R=[]
        for cal in WIN:
            t3=[]
            for sd in range(3):
                a5=({k:float(np.random.default_rng(sd).random()) for k in rk5} if sd else rk5)
                a7=({k:float(np.random.default_rng(sd+100).random()) for k in rk7} if sd else rk7)
                t3.append(sim(cal,mode,START,a5,a7))
            R.append((np.mean([x[0] for x in t3]),np.mean([x[1] for x in t3]),
                      np.mean([x[3] for x in t3]),max(x[0] for x in t3)-min(x[0] for x in t3),
                      np.mean([x[4] for x in t3]),np.mean([x[5] for x in t3]),
                      np.mean([x[6] for x in t3])))
        a=np.array([x[0] for x in R]); ok=np.isfinite(a); down=ok&(bi<0)
        dd=np.array([x[1] for x in R]); iv=np.array([x[2] for x in R]); sp=np.array([x[3] for x in R])
        print(f"  {lab:24s} IRR {np.median(a[ok]):+7.2%}  비중 {np.median(iv[ok]):5.1%}  "
              f"낙폭 {np.median(dd[ok]):6.1%}  최악 {a[ok].min():+7.2%}  현금 {(a[ok]>bc[ok]).mean():4.0%}  "
              f"지수 {(a[ok]>bi[ok]).mean():4.0%}  내린창 {(a[down]>bc[down]).mean():4.0%}  "
              f"폭 {np.median(sp[ok])*100:4.1f}%p  연매수 ⑤{np.mean([x[4] for x in R]):.0f}/⑦{np.mean([x[5] for x in R]):.0f}"
              f"  ⑤중도매도 {np.mean([x[6] for x in R]):.0f}건",flush=True)
    print(flush=True)
print("저장 완료",flush=True)
