"""명령줄 진입점.

    python -m tsignal catalog                       지표/신호 목록
    python -m tsignal fetch --code 005930 ...       캔들 수집 → CSV
    python -m tsignal report --code 005930 ...      검증 리포트 생성
    python -m tsignal probe-toss                    토스 캔들 엔드포인트 재탐색
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import indicators as ind
from . import signals as sig
from .datasource import CsvDataSource, Interval, get_source
from .datasource.toss import TossClient
from .evaluation.report import (
    ReportConfig, write, write_combination, write_factor, write_pattern,
    write_universe, write_walkforward,
)
from .evaluation.trades import CostModel, ExitPolicy


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="synthetic", choices=["naver", "toss", "csv", "synthetic"])
    p.add_argument("--root", default="data", help="csv 소스의 루트 디렉터리")
    p.add_argument("--code", default="005930")
    p.add_argument("--interval", default="5m", choices=[i.value for i in Interval])
    p.add_argument("--count", type=int, default=8000)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, Interval]:
    interval = Interval(args.interval)
    kwargs = {"root": args.root} if args.source == "csv" else {}
    source = get_source(args.source, **kwargs)
    candles = source.candles(
        args.code, interval, start=args.start, end=args.end, count=args.count
    )
    return candles, interval


def cmd_catalog(args: argparse.Namespace) -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_colwidth", 70)
    if args.what in ("indicators", "all"):
        print("=== 지표 ===")
        print(ind.catalog().to_string(index=False))
    if args.what in ("signals", "all"):
        print("\n=== 신호 (트리거/청산) ===")
        print(sig.catalog().to_string(index=False))
    if args.what in ("filters", "all"):
        print("\n=== 상태 필터 ===")
        print(sig.filters.catalog().to_string(index=False))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    candles, interval = _load(args)
    path = CsvDataSource(args.root).save(candles, args.code, interval)
    print(f"{len(candles):,}봉 저장 → {path}")
    print(f"구간 {candles.index[0]} ~ {candles.index[-1]}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    candles, interval = _load(args)
    config = ReportConfig(
        code=args.code,
        interval=interval,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        min_events=args.min_events,
        top_k=args.top_k,
        costs=CostModel(fee_bps=args.fee_bps, tax_bps=args.tax_bps, slippage_bps=args.slippage_bps),
        policy=ExitPolicy(
            stop_atr=args.stop_atr, target_atr=args.target_atr,
            max_bars=args.max_bars, exit_signal=args.exit_signal,
        ),
    )
    out = args.out or f"reports/{args.code}_{interval.value}.md"
    path = write(candles, config, out)
    print(f"리포트 생성 → {path}")
    return 0


DEFAULT_UNIVERSE = {
    "005930": "삼성전자", "000660": "SK하이닉스", "373220": "LG에너지솔루션",
    "207940": "삼성바이오로직스", "005380": "현대차", "000270": "기아",
    "068270": "셀트리온", "105560": "KB금융", "055550": "신한지주",
    "035420": "NAVER", "035720": "카카오", "012330": "현대모비스",
    "051910": "LG화학", "006400": "삼성SDI", "028260": "삼성물산",
    "034730": "SK", "015760": "한국전력", "032830": "삼성생명",
    "003670": "포스코퓨처엠", "086790": "하나금융지주", "247540": "에코프로비엠",
    "066570": "LG전자", "010130": "고려아연", "009540": "HD한국조선해양",
}


def cmd_universe(args: argparse.Namespace) -> int:
    data, interval = _load_universe(args)
    if not data:
        print("수집된 종목이 없습니다.", file=sys.stderr)
        return 1

    out = args.out or f"reports/universe_{interval.value}.md"
    exclude = ("session",) if interval is Interval.D1 else ()
    path = write_universe(
        data, out, interval=interval, exclude_tags=exclude,
        train_ratio=args.train_ratio, min_events=args.min_events,
        costs=CostModel(fee_bps=args.fee_bps, tax_bps=args.tax_bps, slippage_bps=args.slippage_bps),
        names=_universe_names(args.root),
    )
    print(f"{len(data)}종목 리포트 생성 → {path}")
    return 0


def cmd_combine(args: argparse.Namespace) -> int:
    data, interval = _load_universe(args)
    if not data:
        print("수집된 종목이 없습니다.", file=sys.stderr)
        return 1

    out = args.out or f"reports/combination_{interval.value}.md"
    path = write_combination(
        data, out, interval=interval,
        horizons=tuple(int(h) for h in args.horizons.split(",")),
        exclude_tags=("session",) if interval is Interval.D1 else (),
        train_ratio=args.train_ratio, max_filters=args.max_filters,
        min_events=args.min_events, top_k=args.top_k, names=DEFAULT_UNIVERSE,
    )
    print(f"{len(data)}종목 조합 탐색 리포트 생성 → {path}")
    return 0


def _universe_names(root: str) -> dict[str, str]:
    """data/universe.csv 가 있으면 종목명을 읽는다. 없으면 기본 목록."""
    path = Path(root) / "universe.csv"
    if not path.exists():
        return dict(DEFAULT_UNIVERSE)
    frame = pd.read_csv(path, dtype={"code": str})
    return dict(zip(frame["code"], frame["name"]))


def _load_universe(args: argparse.Namespace) -> tuple[dict, Interval]:
    interval = Interval(args.interval)
    kwargs = {"root": args.root} if args.source == "csv" else {}
    source = get_source(args.source, **kwargs)

    if args.codes:
        codes = args.codes.split(",")
    elif args.source == "csv":
        codes = [s.code for s in source.symbols(interval)]
    else:
        codes = list(DEFAULT_UNIVERSE)

    data, failed = {}, []
    for code in codes:
        try:
            data[code] = source.candles(code, interval, count=args.count)
        except Exception as exc:                       # noqa: BLE001
            failed.append(f"{code}: {type(exc).__name__} {str(exc)[:60]}")
    if failed:
        print(f"수집 실패 {len(failed)}건: {failed[0]} ...", file=sys.stderr)
    return data, interval


def cmd_fetch_flow(args: argparse.Namespace) -> int:
    """투자자별 수급(기관·외국인 순매매량)을 종목별로 저장한다."""
    from .datasource.naver_flow import NaverFlowSource

    interval = Interval(args.interval)
    sink = CsvDataSource(args.root)
    codes = args.codes.split(",") if args.codes else [s.code for s in sink.symbols(interval)]
    source = NaverFlowSource(min_interval_sec=args.interval_sec)

    saved, failed = 0, []
    for code in codes:
        path = sink.flow_path(code, interval)
        if path.exists() and not args.overwrite:
            continue
        try:
            frame = source.flow(code, count=args.count)
        except Exception as exc:                       # noqa: BLE001
            failed.append(f"{code}: {type(exc).__name__}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index_label="dt")
        saved += 1
        if saved % 20 == 0:
            print(f"  {saved}종목 완료", flush=True)
    print(f"수급 저장 {saved}종목 / 실패 {len(failed)}")
    if failed:
        print(f"  {', '.join(failed[:5])}", file=sys.stderr)
    return 0 if saved or not failed else 1


def cmd_fetch_universe(args: argparse.Namespace) -> int:
    """시총 상위 종목 목록을 받아 캔들을 통째로 저장한다."""
    from .datasource.universe_list import fetch_universe, to_frame

    symbols = fetch_universe(kospi=args.kospi, kosdaq=args.kosdaq)
    frame = to_frame(symbols)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "universe.csv", index=False)
    print(f"유니버스 {len(frame)}종목 → {root / 'universe.csv'}")

    interval = Interval(args.interval)
    source, sink = get_source("naver"), CsvDataSource(args.root)
    saved, short, failed = 0, [], []
    for symbol in symbols:
        try:
            candles = source.candles(symbol.code, interval, count=args.count)
        except Exception as exc:                       # noqa: BLE001
            failed.append(f"{symbol.code} {symbol.name}: {type(exc).__name__}")
            continue
        if len(candles) < args.min_bars:
            short.append(f"{symbol.code} {symbol.name}({len(candles)}봉)")
            continue
        sink.save(candles, symbol.code, interval)
        if args.extras:
            # 외국인소진율은 같은 응답에 딸려 오지만 한 번 더 받아야 한다
            # (candles() 가 OHLCV 규격만 돌려주기 때문). 요청 1회 추가.
            try:
                sink.save_extras(source.extras(symbol.code, interval, count=args.count),
                                 symbol.code, interval)
            except Exception as exc:                   # noqa: BLE001
                failed.append(f"{symbol.code} extras: {type(exc).__name__}")
        saved += 1
    print(f"저장 {saved}종목 / 봉부족 {len(short)} / 실패 {len(failed)}")
    if short:
        print(f"  봉부족: {', '.join(short[:5])}")
    if failed:
        print(f"  실패: {', '.join(failed[:5])}", file=sys.stderr)
    return 0 if saved else 1


def cmd_walkforward(args: argparse.Namespace) -> int:
    data, interval = _load_universe(args)
    if not data:
        print("수집된 종목이 없습니다.", file=sys.stderr)
        return 1
    out = args.out or f"reports/walkforward_{interval.value}.md"
    path = write_walkforward(
        data, out, interval=interval,
        horizons=tuple(int(h) for h in args.horizons.split(",")),
        exclude_tags=("session",) if interval is Interval.D1 else (),
        train_months=args.train_months, test_months=args.test_months,
        step_months=args.step_months, scheme=args.scheme,
        top_k=args.top_k, min_events=args.min_events,
    )
    print(f"{len(data)}종목 워크포워드 리포트 생성 → {path}")
    return 0


def cmd_factor(args: argparse.Namespace) -> int:
    data, interval = _load_universe(args)
    if not data:
        print("수집된 종목이 없습니다.", file=sys.stderr)
        return 1
    flow = None
    if args.flow and args.source == "csv":
        flow = CsvDataSource(args.root).load_all_flow(interval)
        print(f"수급 데이터 {len(flow)}종목")
    out = args.out or f"reports/factor_{interval.value}.md"
    path = write_factor(
        data, out, interval=interval,
        horizons=tuple(int(h) for h in args.horizons.split(",")),
        n_buckets=args.buckets, control=args.control, flow_by_code=flow,
    )
    print(f"{len(data)}종목 팩터 리포트 생성 → {path}")
    return 0


def cmd_pattern(args: argparse.Namespace) -> int:
    data, interval = _load_universe(args)
    if not data:
        print("수집된 종목이 없습니다.", file=sys.stderr)
        return 1
    out = args.out or f"reports/pattern_{interval.value}.md"
    path = write_pattern(
        data, out,
        holdings=tuple(int(h) for h in args.holdings.split(",")),
        cost_bps=args.cost_bps, n_periods=args.periods,
    )
    print(f"{len(data)}종목 패턴 리포트 생성 → {path}")
    return 0


def cmd_probe_toss(args: argparse.Namespace) -> int:
    """브라우저에서 복사한 요청으로 토스 캔들 엔드포인트를 확정한다."""
    from .datasource.toss import TossApiError, TossClient, parse_candles, save_session
    from .datasource.toss_curl import parse_curl_file, summarize

    if not args.curl and not args.headers:
        print(
            "브라우저 요청이 필요합니다.\n"
            "  1) tossinvest.com 에서 아무 종목의 차트를 엽니다\n"
            "  2) 개발자도구(F12) → Network 탭 → 필터에 'c-chart' 입력\n"
            "  3) 차트 기간을 바꿔 요청이 뜨게 한 뒤, 그 요청을 우클릭\n"
            "     → Copy → Copy as cURL\n"
            "  4) 파일로 저장하고: tsignal probe-toss --curl 저장한파일\n",
            file=sys.stderr,
        )
        return 2

    if args.curl:
        request = parse_curl_file(args.curl)
        print(summarize(request), end="\n\n")
        client = TossClient(session_headers=request.headers, cookies=request.cookies)

        status, body = client.replay(request)
        print(f"재현 결과: HTTP {status}")
        if status != 200:
            print(f"  {body[:300]}")
            print("\n요청이 재현되지 않았습니다. 쿠키가 만료됐을 수 있으니 "
                  "브라우저에서 다시 복사해 보세요.", file=sys.stderr)
            return 1

        print(f"  응답 {len(body):,}바이트\n  {body[:300]}\n")

        required = client.minimal_params(request)
        print(f"필수 파라미터: {required or '(없음 — 전부 선택적)'}")
        optional = sorted(set(request.params) - set(required))
        print(f"생략 가능    : {optional or '(없음)'}\n")

        try:
            candles = parse_candles(json.loads(body))
        except (TossApiError, ValueError) as exc:
            print(f"캔들 파싱 실패: {exc}", file=sys.stderr)
            print("→ toss.py 의 parse_candles() 키 매핑을 응답 구조에 맞게 넓히세요.",
                  file=sys.stderr)
            return 1

        print(f"캔들 파싱 성공: {len(candles)}봉  "
              f"{candles.index[0]:%Y-%m-%d %H:%M} ~ {candles.index[-1]:%Y-%m-%d %H:%M}")
        print(candles.head(3).to_string())

        path = save_session(
            args.out, headers=request.headers, cookies=request.cookies,
            endpoint={"path": request.path, "params": request.params, "required": required},
        )
        print(f"\n세션 저장 → {path}")
        print("  (쿠키가 들어 있으니 커밋하지 마세요. .gitignore 에 등록돼 있습니다.)")
        print(f"\n확인된 경로: {request.path}")
        print("→ 이 경로의 기간 토큰을 toss.py 의 _PERIOD_TOKEN 에 반영하면 수집이 됩니다.")
        return 0

    headers = json.loads(Path(args.headers).read_text(encoding="utf-8"))
    client = TossClient(session_headers=headers)
    print(f"메타: {client.stock_info(args.code)['name']}")
    hits = 0
    for row in client.probe_candle_endpoint(args.code):
        flag = "OK " if row["status"] == 200 else "   "
        print(f"{flag}[{row['status']}] {row['period']:<12} {row['params']}")
        if row["status"] == 200:
            hits += 1
            print(f"     {row['body_head'][:200]}")
    print(f"\n200 응답 {hits}건.")
    return 0 if hits else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsignal", description="단타 신호 검증 파이프라인")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("catalog", help="등록된 지표/신호 목록")
    p.add_argument("--what", default="all", choices=["indicators", "signals", "filters", "all"])
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("fetch", help="캔들 수집 후 CSV 저장")
    _add_data_args(p)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("report", help="검증 리포트 생성")
    _add_data_args(p)
    p.add_argument("--out", default=None)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--min-events", type=int, default=30)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=1.5)
    p.add_argument("--tax-bps", type=float, default=15.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--stop-atr", type=float, default=1.0)
    p.add_argument("--target-atr", type=float, default=2.0)
    p.add_argument("--max-bars", type=int, default=40)
    p.add_argument("--exit-signal", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("universe", help="여러 종목을 묶어 검증 (풀링 + breadth + OOS)")
    p.add_argument("--source", default="csv", choices=["naver", "csv", "synthetic"])
    p.add_argument("--root", default="data")
    p.add_argument("--codes", default=None, help="쉼표 구분 종목코드. 생략하면 기본 유니버스")
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--count", type=int, default=1200)
    p.add_argument("--out", default=None)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--min-events", type=int, default=100)
    p.add_argument("--fee-bps", type=float, default=1.5)
    p.add_argument("--tax-bps", type=float, default=15.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.set_defaults(func=cmd_universe)

    p = sub.add_parser("combine", help="트리거 × 상태필터 조합 탐색 (IS 선정 → OOS 채점)")
    p.add_argument("--source", default="csv", choices=["naver", "csv", "synthetic"])
    p.add_argument("--root", default="data")
    p.add_argument("--codes", default=None)
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--count", type=int, default=1200)
    p.add_argument("--out", default=None)
    p.add_argument("--horizons", default="3,5,10")
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--max-filters", type=int, default=2)
    p.add_argument("--min-events", type=int, default=80)
    p.add_argument("--top-k", type=int, default=20)
    p.set_defaults(func=cmd_combine)

    p = sub.add_parser("fetch-universe", help="시총 상위 종목 목록 + 캔들 일괄 수집")
    p.add_argument("--root", default="data")
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--kospi", type=int, default=150)
    p.add_argument("--kosdaq", type=int, default=50)
    p.add_argument("--count", type=int, default=1200)
    p.add_argument("--min-bars", type=int, default=400)
    p.add_argument("--extras", action="store_true", help="외국인소진율 등 부가 데이터도 저장")
    p.set_defaults(func=cmd_fetch_universe)

    p = sub.add_parser("fetch-flow", help="투자자별 수급 수집 (기관·외국인 순매매량)")
    p.add_argument("--root", default="data")
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--codes", default=None)
    p.add_argument("--count", type=int, default=1200)
    p.add_argument("--interval-sec", type=float, default=0.5, help="요청 간격 (서버 부담 고려)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_fetch_flow)

    p = sub.add_parser("walkforward", help="롤링 워크포워드 (분할을 여러 번)")
    p.add_argument("--source", default="csv", choices=["naver", "csv", "synthetic"])
    p.add_argument("--root", default="data")
    p.add_argument("--codes", default=None)
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--count", type=int, default=1200)
    p.add_argument("--out", default=None)
    p.add_argument("--horizons", default="3,5,10")
    p.add_argument("--train-months", type=int, default=24)
    p.add_argument("--test-months", type=int, default=3)
    p.add_argument("--step-months", type=int, default=None)
    p.add_argument("--scheme", default="rolling", choices=["rolling", "anchored"])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-events", type=int, default=120)
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("factor", help="횡단면 팩터 분석 (겹침·베타 보정 포함)")
    p.add_argument("--source", default="csv", choices=["naver", "csv", "synthetic"])
    p.add_argument("--root", default="data")
    p.add_argument("--codes", default=None)
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--count", type=int, default=1200)
    p.add_argument("--out", default=None)
    p.add_argument("--horizons", default="5,20")
    p.add_argument("--buckets", type=int, default=10)
    p.add_argument("--control", default="ret_120")
    p.add_argument("--flow", action="store_true", help="투자자별 수급 팩터 포함 (csv 소스)")
    p.set_defaults(func=cmd_factor)

    p = sub.add_parser("pattern", help="차트 형태 패턴 (컵앤핸들) 검증")
    p.add_argument("--source", default="csv", choices=["naver", "csv", "synthetic"])
    p.add_argument("--root", default="data")
    p.add_argument("--codes", default=None)
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--count", type=int, default=1200)
    p.add_argument("--out", default=None)
    p.add_argument("--holdings", default="5,10,20,60")
    p.add_argument("--cost-bps", type=float, default=28.0)
    p.add_argument("--periods", type=int, default=4)
    p.set_defaults(func=cmd_pattern)

    p = sub.add_parser("probe-toss", help="브라우저 요청으로 토스 캔들 엔드포인트 확정")
    p.add_argument("--curl", default=None,
                   help="개발자도구에서 'Copy as cURL' 한 내용을 저장한 파일 (권장)")
    p.add_argument("--headers", default=None, help="헤더만 담은 JSON 파일 (구식 경로)")
    p.add_argument("--code", default="005930")
    p.add_argument("--out", default="toss.session.json", help="확인된 세션을 저장할 경로")
    p.set_defaults(func=cmd_probe_toss)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
