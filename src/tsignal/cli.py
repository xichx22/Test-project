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
from .evaluation.report import ReportConfig, write, write_universe
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
        print("\n=== 신호 ===")
        print(sig.catalog().to_string(index=False))
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
    interval = Interval(args.interval)
    codes = args.codes.split(",") if args.codes else list(DEFAULT_UNIVERSE)
    kwargs = {"root": args.root} if args.source == "csv" else {}
    source = get_source(args.source, **kwargs)

    data, failed = {}, []
    for code in codes:
        try:
            data[code] = source.candles(code, interval, count=args.count)
        except Exception as exc:                       # noqa: BLE001 — 한 종목 실패로 전체를 멈추지 않는다
            failed.append(f"{code}: {type(exc).__name__} {str(exc)[:60]}")
    if failed:
        print("수집 실패:\n  " + "\n  ".join(failed), file=sys.stderr)
    if not data:
        print("수집된 종목이 없습니다.", file=sys.stderr)
        return 1

    out = args.out or f"reports/universe_{interval.value}.md"
    exclude = ("session",) if interval is Interval.D1 else ()
    path = write_universe(
        data, out, interval=interval, exclude_tags=exclude,
        train_ratio=args.train_ratio, min_events=args.min_events,
        costs=CostModel(fee_bps=args.fee_bps, tax_bps=args.tax_bps, slippage_bps=args.slippage_bps),
        names=DEFAULT_UNIVERSE,
    )
    print(f"{len(data)}종목 리포트 생성 → {path}")
    return 0


def cmd_probe_toss(args: argparse.Namespace) -> int:
    headers = json.loads(Path(args.headers).read_text()) if args.headers else None
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
    if not hits:
        print("모두 실패했습니다. 브라우저 개발자도구 > Network 에서 c-chart 요청의 "
              "요청 헤더를 JSON 파일로 저장한 뒤 --headers 로 넘겨보세요.")
    return 0 if hits else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsignal", description="단타 신호 검증 파이프라인")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("catalog", help="등록된 지표/신호 목록")
    p.add_argument("--what", default="all", choices=["indicators", "signals", "all"])
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

    p = sub.add_parser("probe-toss", help="토스 캔들 엔드포인트 파라미터 재탐색")
    p.add_argument("--code", default="005930")
    p.add_argument("--headers", default=None, help="브라우저 요청 헤더 JSON 파일 경로")
    p.set_defaults(func=cmd_probe_toss)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
