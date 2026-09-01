"""오늘 기준 매수 신호 — 최신 CSV 를 직접 읽어 마지막 며칠만 검사한다.

검출기 전체를 다시 돌리지 않고, 마지막 N 봉이 돌파봉인지만 확인한다.
기하 조건·거래량 조건은 cup_with_handle 과 완전히 동일하다.
"""
import glob, os, sys, numpy as np, pandas as pd
from tsignal.signals.patterns import CupHandleParams
from tsignal.indicators.volume import mfi
P=CupHandleParams()
ARGS=[a for a in sys.argv[1:] if not a.startswith("--")]
DAYS=int(ARGS[0]) if ARGS else 3
ROOT="data_live/1d" if "--live" in sys.argv else "data_wide/1d"
uni=pd.read_csv("data_wide/universe.csv",dtype=str).set_index("code")

def fired(high,low,close,volume,t,avgv):
    """t 봉이 컵앤핸들 돌파봉인가 — 검출기와 같은 문턱."""
    if not np.isfinite(avgv[t]) or avgv[t]<=0: return None
    if volume[t] < P.breakout_volume*avgv[t]: return None
    for hl in range(P.handle_min,P.handle_max+1):
        hs=t-hl
        if hs<=0: break
        hh=high[hs:t].max(); hlw=low[hs:t].min()
        if not (close[t]>hh and close[t-1]<=hh): continue
        if hh<=0: continue
        hd=(hh-hlw)/hh
        if hd>P.handle_depth_max: continue
        for cl_ in range(P.cup_min,P.cup_max+1,5):
            ce=hs; cs=ce-cl_
            if cs-P.prior_window<0: break
            th=max(1,cl_//3); ls=high[cs:cs+th]; lr=ls.max()
            if int(np.argmax(ls))>P.rim_position*th: continue
            rr=high[ce-th:ce].max(); ti=int(np.argmin(low[cs:ce]))+cs; tro=low[ti]
            if lr<=0 or tro<=0: continue
            cd=(lr-tro)/lr
            if not (P.cup_depth_min<=cd<=P.cup_depth_max): continue
            if not (P.rim_recovery<=rr/lr<=P.rim_overshoot): continue
            if hh<rr*P.handle_at_rim: continue
            pf=max(0,cs-P.rim_lookback)
            if lr<P.rim_is_peak*high[pf:cs+th].max(): continue
            mg=(1-P.trough_center)/2; pos=(ti-cs)/cl_
            if not (mg<=pos<=1-mg): continue
            if hlw<tro+P.handle_upper_half*(lr-tro): continue
            if hd>P.handle_vs_cup*cd: continue
            cv=volume[cs:ce].mean()
            if cv<=0 or volume[hs:t].mean()>P.handle_volume_max*cv: continue
            pr=close[cs-P.prior_window]
            if pr<=0 or close[cs]/pr-1<P.prior_gain: continue
            return dict(피벗=hh,컵봉=cl_,손잡이=hl,깊이=cd,거래량배=volume[t]/avgv[t])
    return None

rows=[]; last_seen=None
for path in sorted(glob.glob(f"{ROOT}/*.csv")):
    code=os.path.basename(path)[:-4]
    d=pd.read_csv(path,index_col=0,parse_dates=[0])
    if len(d)<400: continue
    if d.index.tz is None: d.index=d.index.tz_localize("Asia/Seoul")
    last_seen=max(last_seen,d.index[-1]) if last_seen is not None else d.index[-1]
    hi=d["high"].to_numpy(float); lo=d["low"].to_numpy(float)
    cl=d["close"].to_numpy(float); vo=d["volume"].to_numpy(float)
    n=len(d)
    avgv=pd.Series(vo).rolling(P.volume_window,min_periods=P.volume_window).mean().to_numpy()
    turn=(d["close"]*d["volume"]).rolling(20).mean().to_numpy()
    m=mfi(d,14).to_numpy()
    for t in range(n-DAYS,n):
        if t<1: continue
        b=fired(hi,lo,cl,vo,t,avgv)
        if not b: continue
        hi20=float(np.nanmax(hi[t-20:t]))
        mfi_ok=False; mday=None
        for k in range(max(1,t-20),t+1):
            if np.isfinite(m[k-1]) and np.isfinite(m[k]) and m[k-1]<=20 and m[k]>20:
                mfi_ok=True; mday=d.index[k].date()
        rows.append(dict(신호일=d.index[t].date(),code=code,종목=uni["name"].get(code,"?"),
            시장=uni["market"].get(code,"?"),종가=cl[t],피벗=b["피벗"],
            거래량배=b["거래량배"],거래대금억=turn[t]/1e8,
            신고가겸함="예" if cl[t]>hi20 else "아니오",
            MFI선행="예" if mfi_ok else "아니오",MFI일=mday,
            컵봉=b["컵봉"],손잡이=b["손잡이"],깊이=b["깊이"]*100,
            다음날시가=cl[t+1] if False else np.nan))
print(f"데이터 마지막 거래일: {last_seen.date()}  ·  최근 {DAYS}거래일 검사\n")
if not rows:
    print("신호 없음."); raise SystemExit
r=pd.DataFrame(rows); r=r[r["거래대금억"]>=3].sort_values(["신호일","거래대금억"],ascending=[True,False])
f={"종가":"{:,.0f}".format,"피벗":"{:,.0f}".format,"거래량배":"{:.1f}".format,
   "거래대금억":"{:.1f}".format,"깊이":"{:.0f}".format}
cols=["신호일","code","종목","시장","종가","피벗","거래량배","거래대금억","컵봉","손잡이","깊이","신고가겸함","MFI선행"]
print(f"=== ① 컵앤핸들 돌파 (유동성 3억↑): {len(r)}건 ===")
print(r[cols].to_string(index=False,formatters=f))
r5=r[r["신고가겸함"]=="예"]; r7=r[r["MFI선행"]=="예"]
print(f"\n=== ⑤ 컵앤핸들 + 20일 신고가: {len(r5)}건 ===")
print(r5[cols].to_string(index=False,formatters=f) if len(r5) else "  없음")
print(f"\n=== ⑦ 과매도 후 컵앤핸들: {len(r7)}건 ===")
print(r7[cols].to_string(index=False,formatters=f) if len(r7) else "  없음")
