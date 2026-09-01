"""오늘 기준 ⑦ 조건 후보 — MFI 과매도 탈출 후 20거래일 안에 컵앤핸들 돌파."""
import pickle, numpy as np, pandas as pd
S="/tmp/claude-0/-home-user-Test-project/e2567955-0c39-5157-a915-38387e9cd240/scratchpad"
parts=pickle.load(open(f"{S}/parts16.pkl","rb"))
blob=pickle.load(open(f"{S}/cache16.pkl","rb")); data,feats=blob["data"],blob["feats"]
uni=pd.read_csv("data_wide/universe.csv",dtype=str).set_index("code")
cup, mfi = parts["컵앤핸들"], parts["MFI과매도"]
LAST=max(d.index[-1] for d in data.values())
print(f"데이터 마지막 거래일: {LAST.date()}\n")

rows=[]
for code, sc in cup.items():
    sm = mfi.get(code)
    if sm is None: continue
    hit = sc & sm.rolling(21, min_periods=1).max().astype(bool)   # 컵 돌파일, 직전 20일 내 MFI
    hit = hit.fillna(False)
    for ts in sc.index[hit.to_numpy()][-6:]:
        d=data[code]; f=feats[code]
        i=d.index.get_loc(ts)
        ago=len(d.index)-1-i                      # 며칠 전 신호인가
        # MFI 가 언제 떴는지
        w=sm.iloc[max(0,i-20):i+1]; mday=w.index[w.to_numpy()][-1] if w.any() else None
        turn=float((d["close"]*d["volume"]).rolling(20).mean().iloc[i])
        entry=float(d["open"].iloc[i+1]) if i+1<len(d) else np.nan   # 다음 봉 시가 = 체결가
        now=float(d["close"].iloc[-1])
        vol=float(d["volume"].iloc[i]/d["volume"].rolling(20).mean().iloc[i])
        rows.append(dict(code=code, 종목=uni["name"].get(code,"?"), 시장=uni["market"].get(code,"?"),
            돌파일=ts.date(), 며칠전=ago, MFI일=mday.date() if mday is not None else None,
            간격=(i-d.index.get_loc(mday)) if mday is not None else None,
            돌파종가=float(d["close"].iloc[i]), 체결가=entry, 현재가=now,
            거래대금억=turn/1e8, 돌파거래량배=vol))
r=pd.DataFrame(rows)
if r.empty:
    print("최근 신호 없음"); raise SystemExit
r=r.sort_values("며칠전").head(15)
r["목표가"]=r["체결가"]*1.20
r["만기일차"]=60-r["며칠전"]
r["현재손익%"]=(r["현재가"]/r["체결가"]-1)*100
pd.set_option("display.width",250,"display.max_columns",30)
print("=== 가장 최근 ⑦ 신호 15건 ===")
print(r[["code","종목","시장","돌파일","며칠전","MFI일","간격","체결가","현재가","현재손익%",
         "목표가","만기일차","거래대금억","돌파거래량배"]].to_string(index=False,
    formatters={"체결가":"{:,.0f}".format,"현재가":"{:,.0f}".format,"목표가":"{:,.0f}".format,
                "현재손익%":"{:+.1f}".format,"거래대금억":"{:.1f}".format,"돌파거래량배":"{:.1f}".format}))
print(f"\n유동성 3억 미만 제외 후: {(r['거래대금억']>=3).sum()}건")
