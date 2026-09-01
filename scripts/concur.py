"""신호를 하나도 안 놓치려면 동시에 몇 자리가 필요한가."""
import pickle, numpy as np, pandas as pd
from tsignal.evaluation.exitlab import Exit, resolve
S="/tmp/claude-0/-home-user-Test-project/e2567955-0c39-5157-a915-38387e9cd240/scratchpad"
parts=pickle.load(open(f"{S}/parts16.pkl","rb"))
blob=pickle.load(open(f"{S}/cache16.pkl","rb")); data,feats=blob["data"],blob["feats"]
cal=pd.DatetimeIndex(sorted({t for d in data.values() for t in d.index}))
cal=cal[cal>=pd.Timestamp("2010-01-01",tz=cal.tz)]
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
pos={t:i for i,t in enumerate(cal)}
print("신호를 전부 받아내려면 동시에 몇 자리가 필요한가 (2010~2026)\n")
print(f"{'규칙':6s}{'총신호':>8s}{'연평균':>7s}{'평균보유':>8s}{'동시보유 평균':>13s}{'중앙':>6s}{'90%':>6s}{'99%':>6s}{'최대':>6s}")
res={}
for kind in ("⑦","⑤","①"):
    ev=build(kind)
    tr=resolve(ev,data,feats,rule=RULE)
    ent=pd.DatetimeIndex(tr["진입일"]); ext=pd.DatetimeIndex(tr["청산일"])
    open_ct=np.zeros(len(cal),dtype=int)
    for e,x in zip(ent,ext):
        if e not in pos: continue
        a=pos[e]; b=pos.get(x,len(cal)-1)
        open_ct[a:max(a+1,b)]+=1
    yrs=(cal[-1]-cal[0]).days/365.25
    res[kind]=(len(tr),open_ct,tr["보유봉"].mean())
    print(f"{kind:6s}{len(tr):>8,}{len(tr)/yrs:>7.0f}{tr['보유봉'].mean():>8.0f}"
          f"{open_ct.mean():>13.1f}{np.median(open_ct):>6.0f}"
          f"{np.percentile(open_ct,90):>6.0f}{np.percentile(open_ct,99):>6.0f}{open_ct.max():>6.0f}")
print()
print("자리 수를 제한하면 신호를 몇 % 나 받는가")
print(f"{'자리':>5s}" + "".join(f"{k:>10s}" for k in ("⑦","⑤","①")))
for slots in (3,5,8,10,15,20,30,50,80):
    row=f"{slots:>5d}"
    for kind in ("⑦","⑤","①"):
        n,oc,_=res[kind]
        # 동시보유가 자리수를 넘는 날의 초과분만큼 놓친다고 근사
        taken=np.minimum(oc,slots).sum()/oc.sum() if oc.sum() else 1.0
        row+=f"{taken*100:>9.0f}%"
    print(row)
print()
for kind in ("⑦","⑤","①"):
    n,oc,_=res[kind]
    for cover in (0.90,0.95,0.99):
        need=int(np.ceil(np.percentile(oc,cover*100)))
        if cover==0.90: line=f"{kind}: "
        line+=f"{cover:.0%} 받으려면 {need}자리   "
    print(line)
