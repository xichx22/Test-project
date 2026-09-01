"""data_wide/1d 의 일봉을 오늘까지 갱신한다. 이미 있는 날은 새 값으로 덮는다."""
import sys, glob, os, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tsignal.datasource.naver import NaverDataSource
from tsignal.datasource.base import Interval
TODAY=pd.Timestamp.now(tz="Asia/Seoul").normalize()
ROOT="data_live/1d" if "--live" in sys.argv else "data_wide/1d"
KEEP=760          # --live 는 최근 3년치만 유지한다
files=sorted(glob.glob(f"{ROOT}/*.csv"))
print(f"대상 {len(files)}종목 · {ROOT} · 오늘 {TODAY.date()}", flush=True)
COLS=["open","high","low","close","volume"]
def one(path):
    code=os.path.basename(path)[:-4]
    try:
        old=pd.read_csv(path,index_col=0,parse_dates=[0])
        if old.empty: return code,0,"빈 파일"
        last=old.index[-1]
        if last.tzinfo is None: old.index=old.index.tz_localize("Asia/Seoul"); last=old.index[-1]
        if last>=TODAY: return code,0,"이미 최신"
        src=NaverDataSource()
        new=src.candles(code,Interval.D1,start=(last-pd.Timedelta(days=7)),end=TODAY)
        if new is None or new.empty: return code,0,"응답 없음"
        new=new[COLS]
        merged=pd.concat([old[COLS],new])
        merged=merged[~merged.index.duplicated(keep="last")].sort_index()
        added=len(merged)-len(old)
        if ROOT.startswith("data_live"): merged=merged.tail(KEEP)
        merged.to_csv(path)
        return code,added,None
    except Exception as e:
        return code,0,f"{type(e).__name__}: {e}"
ok=add=fail=0; errs={}
with ThreadPoolExecutor(max_workers=6) as ex:
    for i,(code,n,err) in enumerate(ex.map(one,files),1):
        if err and err not in ("이미 최신","응답 없음"): fail+=1; errs[err]=errs.get(err,0)+1
        else: ok+=1; add+=n
        if i%200==0: print(f"  {i}/{len(files)}  추가된 봉 {add:,}  실패 {fail}", flush=True)
print(f"\n완료: 성공 {ok} · 실패 {fail} · 새로 추가된 봉 {add:,}", flush=True)
for e,c in sorted(errs.items(),key=lambda x:-x[1])[:5]: print(f"  {c}회  {e}", flush=True)
