"""2026년에 실제로 ⑦ 규칙을 굴렸다면 — 청산까지 끝낸 결과."""
import pickle, numpy as np, pandas as pd
from tsignal.evaluation.exitlab import Exit, resolve
S="/tmp/claude-0/-home-user-Test-project/e2567955-0c39-5157-a915-38387e9cd240/scratchpad"
parts=pickle.load(open(f"{S}/parts16.pkl","rb"))
blob=pickle.load(open(f"{S}/cache16.pkl","rb")); data,feats=blob["data"],blob["feats"]
uni=pd.read_csv("data_wide/universe.csv",dtype=str).set_index("code")
cup,mfi=parts["컵앤핸들"],parts["MFI과매도"]
ev={}
for code,sc in cup.items():
    sm=mfi.get(code)
    if sm is None: continue
    h=(sc & sm.rolling(21,min_periods=1).max().astype(bool)).fillna(False)
    if h.any(): ev[code]=h
tr=resolve(ev,data,feats,rule=Exit("익절20%",take_profit=0.20,horizon=60))
tr["진입일"]=pd.DatetimeIndex(tr["진입일"])
for yr in (2024,2025,2026):
    t=tr[tr["진입일"].dt.year==yr]
    if t.empty: print(f"{yr}: 신호 없음"); continue
    r=(t["청산"]*(1-0.0014)/t["진입"]-1)
    print(f"{yr}년  매매 {len(t):3d}건   평균 {r.mean():+6.2%}  중앙 {r.median():+6.2%}  "
          f"승률 {(r>0).mean():4.0%}  최고 {r.max():+6.1%}  최악 {r.min():+6.1%}")
    print("        청산 사유:", dict(t["사유"].value_counts()))
t=tr[tr["진입일"].dt.year==2026].copy()
t["수익%"]=(t["청산"]*(1-0.0014)/t["진입"]-1)*100
t["종목"]=[uni["name"].get(c,"?") for c in t["code"]]
print("\n=== 2026년 ⑦ 매매 전체 ===")
print(t[["code","종목","진입일","청산일","진입","청산","보유봉","사유","수익%"]].to_string(
    index=False,formatters={"진입":"{:,.0f}".format,"청산":"{:,.0f}".format,"수익%":"{:+.1f}".format}))
