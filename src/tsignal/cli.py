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


def cmd_portfolio(args: argparse.Namespace) -> int:
    """자산군 결론을 실제 ETF·계좌로 내려보낸다 — 유동성·환노출·세금."""
    import pandas as pd

    from .evaluation.allocation import buy_and_hold, static_mix
    from .evaluation.portfolio import (
        ETF_CATALOG, account_comparison, horizon_gap, liquidity,
    )

    source = CsvDataSource(args.root)
    interval = Interval(args.interval)

    def close(code: str) -> pd.Series:
        series = source.candles(code, interval)["close"]
        if args.start:
            series = series[series.index >= pd.Timestamp(args.start, tz=series.index.tz)]
        return series

    print("== 유동성 (최근 250거래일 일평균 거래대금 중앙값) ==")
    candles = {}
    for spec in ETF_CATALOG:
        try:
            candles[spec.code] = source.candles(spec.code, interval)
        except FileNotFoundError:
            continue
    if candles:
        table = liquidity(candles, days=args.liquidity_days)
        table["중앙값(억)"] = (table.pop("median_turnover") / 1e8).round(1)
        table["최소(억)"] = (table.pop("min_turnover") / 1e8).round(2)
        print(table.drop(columns=["days"]).to_string())
    else:
        print(f"  {args.root} 에 ETF 캔들이 없습니다.")
        return 1

    picks = dict(pair.split("=") for pair in args.mix.split(","))
    assets = {name: close(code) for name, code in picks.items()}
    weights = {name: 1.0 for name in assets}

    print("\n== 배분 성과 (동일가중, 분기 리밸런싱) ==")
    mix = static_mix(assets, weights, rebalance=args.rebalance, name="동일가중")
    frame = pd.DataFrame(assets).dropna()
    rows = [mix]
    if args.benchmark in picks:
        bench = buy_and_hold(frame[args.benchmark])
        bench.name = f"{args.benchmark} 100%"
        rows.append(bench)
    print(f"기간 {frame.index[0].date()} ~ {frame.index[-1].date()} ({mix.years:.1f}년)")
    print(pd.DataFrame([{
        "": r.name, "연수익": f"{r.cagr:.2%}",
        "양수율12M": f"{r.rolling_positive(12):.3f}",
        "최악12M": f"{r.worst_rolling(12):.2%}", "최악해": f"{r.worst_year:.2%}",
        "무저점일수": r.longest_underwater_days, "궤양": f"{r.ulcer_index:.2f}",
        "MDD": f"{r.max_drawdown:.2%}", "리밸": r.trades,
    } for r in rows]).to_string(index=False))

    print("\n== 계좌별 세금 (연금 vs 일반) ==")
    by_code = {code: series for code, series in
               zip(picks.values(), assets.values())}
    domestic = tuple(c for c in by_code if _is_domestic_equity(c))
    accounts = account_comparison(by_code, {c: 1.0 for c in by_code},
                                  rebalance=args.rebalance, domestic_equity=domestic)
    shown = accounts.copy()
    shown["연수익"] = shown["연수익"].map("{:.2%}".format)
    shown["최종배수"] = shown["최종배수"].map("{:.4f}".format)
    shown["납부세액"] = shown["납부세액"].map(lambda v: f"{v * 100:.2f}%")
    print(shown.to_string(index=False))
    drag = accounts.attrs["drag"]
    print(f"세금 드래그: 연 {drag:.3%}p  (매매차익 비과세: {', '.join(domestic) or '없음'})")
    gap = horizon_gap(drag, accounts.loc[0, "연수익"], principal=args.principal)
    for column in ("연금계좌", "일반계좌", "차이"):
        gap[column] = gap[column].map(lambda v: f"{v:,.0f}원")
    print(gap.to_string(index=False))
    return 0


def _is_domestic_equity(code: str) -> bool:
    from .evaluation.portfolio import spec_for
    spec = spec_for(code)
    return bool(spec and not spec.taxable)


def cmd_swing(args: argparse.Namespace) -> int:
    """스윙 규칙 수천 개를 연수익으로 줄 세우고, 그 순위가 진짜인지 확인한다."""
    import pandas as pd

    from .evaluation.swinglab import SwingLab, split_test

    source = CsvDataSource(args.root)
    interval = Interval(args.interval)
    codes = args.codes.split(",") if args.codes else [
        s.code for s in source.symbols(interval)]
    data = {code: source.candles(code, interval) for code in codes}
    if not data:
        print(f"{args.root} 에 캔들이 없습니다.", file=sys.stderr)
        return 1

    lab = SwingLab(data, interval=interval)
    bench = lab.benchmark()
    print(f"[벤치] 유니버스 동일가중 매수후보유 {bench.years:.1f}년  "
          f"연 {bench.cagr:.2%}  MDD {bench.max_drawdown:.2%}  "
          f"궤양 {bench.ulcer_index:.2f}")
    print(f"종목 {len(lab.codes)}  트리거 {len(lab.trigger_names)}  "
          f"필터 {len(lab.filter_names)}")

    holdings = tuple(int(h) for h in args.holdings.split(","))
    board = lab.leaderboard(holdings=holdings, cost_bps=args.cost_bps,
                            min_trades=args.min_trades)
    if board.empty:
        print("조건을 만족하는 규칙이 없습니다.", file=sys.stderr)
        return 1
    pd.set_option("display.width", 250)
    shown = board.head(args.top).copy()
    for column in ("연수익", "변동성", "MDD", "최악12M", "투자비중"):
        shown[column] = shown[column].map("{:.2%}".format)
    for column in ("샤프", "궤양"):
        shown[column] = shown[column].map("{:.2f}".format)
    shown["양수율12M"] = shown["양수율12M"].map("{:.3f}".format)
    print(f"\n== 연수익 상위 {args.top} (조합 {len(board)}개 중) ==")
    print(shown.to_string(index=False))
    beat = int((board["연수익"] > bench.cagr).sum())
    print(f"\n벤치마크를 이긴 조합 {beat} / {len(board)}")

    print("\n== 전반부에서 뽑아 후반부에서 채점 ==")
    split = split_test(lab, board, top=args.top, cost_bps=args.cost_bps)
    view = split.copy()
    for column in ("전반부", "후반부", "차이"):
        view[column] = view[column].map("{:.2%}".format)
    print(view.to_string(index=False))
    if "spearman" in split.attrs:
        print(f"순위 상관 {split.attrs['spearman']:.3f}  "
              f"(벤치 전반 {split.attrs['bench_first']:.2%} / "
              f"후반 {split.attrs['bench_second']:.2%})")
        print("순위 상관이 0 근처면 순위표는 잡음이다 — 1등을 골라도 소용이 없다.")
    return 0


def cmd_pattern_lab(args: argparse.Namespace) -> int:
    """차트 패턴 전종 검증 — 검출 · 채점 · 구간 분할을 한 번에."""
    import pandas as pd

    from .evaluation.patternlab import detect, score, subperiods
    from .signals.patterns import FAMOUS_PATTERNS, PATTERNS

    source = CsvDataSource(args.root)
    interval = Interval(args.interval)
    data = {}
    for symbol in source.symbols(interval):
        candles = source.candles(symbol.code, interval)
        if args.end:
            candles = candles[candles.index <= pd.Timestamp(args.end,
                                                            tz=candles.index.tz)]
        if len(candles) >= args.min_bars:
            data[symbol.code] = candles
    if not data:
        print(f"{args.root} 에 쓸 만한 캔들이 없습니다.", file=sys.stderr)
        return 1

    first = min(d.index[0] for d in data.values())
    last = max(d.index[-1] for d in data.values())
    print(f"종목 {len(data)}  {first.date()} ~ {last.date()} "
          f"({(last - first).days / 365.25:.1f}년)  비용 {args.cost_bps}bp")

    found = detect(data, PATTERNS)
    labels = {n: ("유명" if n in FAMOUS_PATTERNS else "덜알려짐") for n in PATTERNS}
    table = score(found, data, holding_days=args.holding, cost_bps=args.cost_bps,
                  labels=labels)
    pd.set_option("display.width", 200)
    view = table.copy()
    view["연초과"] = view["연초과"].map(lambda v: "-" if pd.isna(v) else f"{v:+.2%}")
    view["t"] = view["t"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    print(f"\n패턴 {table.attrs['n_hypotheses']}종 동시 검정 "
          f"→ 필요 |t| >= {table.attrs['threshold']:.2f}")
    print(view.to_string(index=False))
    print(f"통과 {(table['판정'] == '통과').sum()} / "
          f"{(table['판정'] != '표본부족').sum()}")

    print(f"\n== {args.periods}구간 분할 ==")
    ranked = [r["패턴"] for _, r in table.iterrows() if r["판정"] != "표본부족"]
    for name in ranked[:args.top]:
        out = subperiods(found[name], data, periods=args.periods,
                         holding_days=args.holding, cost_bps=args.cost_bps)
        cells = " ".join("  n/a " if pd.isna(v) else f"{v:+6.1%}"
                         for v in out["표"]["연초과"])
        print(f"  {name:28s} {cells}   {out['승']}승{out['패']}패  p={out['p']:.4f}")
        if name == ranked[0]:
            print(f"  (구간 {args.periods}개의 최소 가능 p = {out['min_p']:.4f} — "
                  f"이보다 작은 p 는 나올 수 없다)")
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

    p = sub.add_parser("portfolio", help="ETF 선택·계좌 세금까지 내려본 실행 리포트")
    p.add_argument("--root", default="data_asset")
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--mix", default="한국주식=069500,미국주식=133690,국고채3년=114260,"
                                    "국고채10년=148070,금=132030,달러=261240",
                   help="이름=종목코드 를 쉼표로 나열")
    p.add_argument("--benchmark", default="한국주식")
    p.add_argument("--rebalance", default="QE", choices=["ME", "QE", "YE"])
    p.add_argument("--start", default=None)
    p.add_argument("--liquidity-days", type=int, default=250)
    p.add_argument("--principal", type=float, default=10_000_000)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("swing", help="스윙 규칙 순위표 + 순위가 유지되는지 검정")
    p.add_argument("--root", default="data_big")
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--codes", default=None)
    p.add_argument("--holdings", default="5,10,20,60")
    p.add_argument("--cost-bps", type=float, default=28.0)
    p.add_argument("--min-trades", type=int, default=200)
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(func=cmd_swing)

    p = sub.add_parser("pattern-lab", help="차트 패턴 전종 검증 (다중검정 + 구간 분할)")
    p.add_argument("--root", default="data_wide")
    p.add_argument("--interval", default="1d", choices=[i.value for i in Interval])
    p.add_argument("--end", default="2025-12-31", help="이 날짜까지만 (기본: 2026년 제외)")
    p.add_argument("--holding", type=int, default=60)
    p.add_argument("--cost-bps", type=float, default=28.0)
    p.add_argument("--min-bars", type=int, default=600)
    p.add_argument("--periods", type=int, default=8)
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=cmd_pattern_lab)

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
