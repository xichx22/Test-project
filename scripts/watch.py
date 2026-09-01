"""⑦ 감시 목록 — 아직 돌파 안 했지만 컵·핸들 모양이 이미 완성된 종목.

검출기(cup_with_handle)의 기하 조건을 그대로 쓰되, 마지막 봉 다음을
'가상의 돌파봉' 으로 놓고 두 가지만 뺀다:
  · 종가 > 핸들 고점  (아직 안 뚫었으니까 — 대신 얼마나 남았는지 잰다)
  · 돌파봉 거래량 1.4배 (돌파하는 날 정해지는 값이라 미리 알 수 없다)
나머지(컵 깊이·좌우 테두리 비율·U자·핸들 위치·핸들 거래량·선행 상승)는
실제 신호와 완전히 같은 문턱이다.
"""
import pickle, numpy as np, pandas as pd
from tsignal.signals.patterns import CupHandleParams
S="/tmp/claude-0/-home-user-Test-project/e2567955-0c39-5157-a915-38387e9cd240/scratchpad"
blob=pickle.load(open(f"{S}/cache16.pkl","rb")); data,feats=blob["data"],blob["feats"]
parts=pickle.load(open(f"{S}/parts16.pkl","rb"))
uni=pd.read_csv("data_wide/universe.csv",dtype=str).set_index("code")
P=CupHandleParams()
LAST=max(d.index[-1] for d in data.values())
print(f"기준일 {LAST.date()}  ·  컵앤핸들 기하 조건은 실제 검출기와 동일\n")

def pending(high,low,close,volume,n):
    """t=n 을 가상 돌파봉으로 놓고 성립하는 컵·핸들을 찾는다."""
    best=None
    for handle_len in range(P.handle_min,P.handle_max+1):
        h_start=n-handle_len
        if h_start<=0: break
        handle_high=high[h_start:n].max(); handle_low=low[h_start:n].min()
        if handle_high<=0: continue
        if close[n-1]>handle_high: continue          # 이미 뚫었으면 감시 대상 아님
        handle_depth=(handle_high-handle_low)/handle_high
        if handle_depth>P.handle_depth_max: continue
        for cup_len in range(P.cup_min,P.cup_max+1,5):
            c_end=h_start; c_start=c_end-cup_len
            if c_start-P.prior_window<0: break
            third=max(1,cup_len//3)
            left_seg=high[c_start:c_start+third]; left_rim=left_seg.max()
            if int(np.argmax(left_seg))>P.rim_position*third: continue
            right_rim=high[c_end-third:c_end].max()
            ti=int(np.argmin(low[c_start:c_end]))+c_start; trough=low[ti]
            if left_rim<=0 or trough<=0: continue
            cup_depth=(left_rim-trough)/left_rim
            if not (P.cup_depth_min<=cup_depth<=P.cup_depth_max): continue
            if not (P.rim_recovery<=right_rim/left_rim<=P.rim_overshoot): continue
            if handle_high<right_rim*P.handle_at_rim: continue
            pf=max(0,c_start-P.rim_lookback)
            if left_rim<P.rim_is_peak*high[pf:c_start+third].max(): continue
            margin=(1-P.trough_center)/2; pos=(ti-c_start)/cup_len
            if not (margin<=pos<=1-margin): continue
            if handle_low<trough+P.handle_upper_half*(left_rim-trough): continue
            if handle_depth>P.handle_vs_cup*cup_depth: continue
            cv=volume[c_start:c_end].mean()
            if cv<=0: continue
            if volume[h_start:n].mean()>P.handle_volume_max*cv: continue
            prior=close[c_start-P.prior_window]
            if prior<=0 or close[c_start]/prior-1<P.prior_gain: continue
            best=dict(피벗=handle_high, 컵봉=cup_len, 손잡이봉=handle_len,
                      컵깊이=cup_depth, 좌림=left_rim, 우림=right_rim, 저점=trough)
            break
        if best: break
    return best

mfi=parts["MFI과매도"]
rows=[]
for code,d in data.items():
    if len(d)<400: continue
    n=len(d)
    hi=d["high"].to_numpy(float); lo=d["low"].to_numpy(float)
    cl=d["close"].to_numpy(float); vo=d["volume"].to_numpy(float)
    if not np.isfinite(cl[-1]) or cl[-1]<=0: continue
    b=pending(hi,lo,cl,vo,n)
    if not b: continue
    turn=float((d["close"]*d["volume"]).rolling(20).mean().iloc[-1])/1e8
    sm=mfi.get(code)
    mday=None; mago=None
    if sm is not None:
        w=sm.iloc[-21:]
        if w.any(): mday=w.index[w.to_numpy()][-1]; mago=n-1-d.index.get_loc(mday)
    v50=np.nanmean(vo[-50:])
    rows.append(dict(code=code, 종목=uni["name"].get(code,"?"), 시장=uni["market"].get(code,"?"),
        현재가=cl[-1], 피벗=b["피벗"], 남은=(b["피벗"]/cl[-1]-1)*100,
        컵봉=b["컵봉"], 손잡이=b["손잡이봉"], 컵깊이=b["컵깊이"]*100,
        우림비=b["우림"]/b["좌림"], 거래대금억=turn,
        MFI일=mday.date() if mday is not None else None, MFI전=mago,
        필요거래량=P.breakout_volume*v50))
r=pd.DataFrame(rows)
if r.empty:
    print("모양이 완성된 종목 자체가 없다."); raise SystemExit
r=r.sort_values("남은")
pd.set_option("display.width",260,"display.max_columns",30)
fmt={"현재가":"{:,.0f}".format,"피벗":"{:,.0f}".format,"남은":"{:+.1f}".format,
     "컵깊이":"{:.0f}".format,"우림비":"{:.2f}".format,"거래대금억":"{:.1f}".format,
     "필요거래량":"{:,.0f}".format}
cols=["code","종목","시장","현재가","피벗","남은","컵봉","손잡이","컵깊이","우림비","거래대금억","MFI일","MFI전"]
print(f"=== 컵·핸들 모양이 완성된 종목: {len(r)}개 ===\n")
print("[A] ⑦ 조건 충족 대기 — MFI 과매도 탈출이 최근 20일 안에 있다 (돌파만 하면 매수 신호)")
a=r[r["MFI전"].notna() & (r["거래대금억"]>=3)]
print(a[cols].to_string(index=False,formatters=fmt) if len(a) else "  없음")
print(f"\n[B] 모양만 완성 — MFI 가 아직 안 왔다 (컵앤핸들 단독 신호는 되지만 ⑦ 은 아님)")
b=r[r["MFI전"].isna() & (r["거래대금억"]>=3)].head(20)
print(b[cols].to_string(index=False,formatters=fmt) if len(b) else "  없음")
print(f"\n유동성 3억 미만 제외: {(r['거래대금억']<3).sum()}개")
