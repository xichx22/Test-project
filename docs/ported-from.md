# 이식 기록 — tradermonty/claude-trading-skills

## 출처

    https://github.com/tradermonty/claude-trading-skills
    MIT License · Copyright (c) 2026 TraderMonty

MIT 라이선스이므로 사용·수정·재배포가 자유롭고, 저작권 표시를 유지한다.
이 문서가 그 표시이자 무엇을 어떻게 바꿨는지의 기록이다.

## 왜 이 저장소인가

스킬 73종의 트레이딩 워크플로 모음이다. 구조가 이 프로젝트의 결론과 거의 같다.

> "The goal is not to outsource buy/sell decisions to AI."
> "The core loop is not `Ask → Signal → Trade`. It is `Plan → Trade → Record → Review → Improve`."
> "long-term investing, ETFs as their **core**, disciplined swing trading as a **satellite**"

코어(연금 6자산) + 위성(스윙)이라는 두 계좌 구조, 그리고 "신호를 파는 것이
아니라 과정을 만든다"는 태도가 이 프로젝트가 21년 데이터로 도달한 지점과 같다.

## 무엇을 가져왔는가

원본 4종의 **개념과 구조**를 `src/tsignal/desk/` 로 옮겼다. 코드를 복사하지
않았다 — 원본은 미국 시장·유료 API(FMP·Alpaca·FINVIZ) 전제이고, 이쪽은
한국 시장·네이버 공개 데이터·연금계좌 전제라 재작성이 필요했다.

| 원본 스킬 | 이식 | 바뀐 것 |
|---|---|---|
| `position-sizer` | `desk/sizing.py` | 호가 단위 반올림, 정수 주, 왕복 28bp, 자리·손절 이중 검증 |
| `drawdown-circuit-breaker` | `desk/guard.py` | **문턱 전면 재유도** (아래 참조), 한국 거래일, 실현손익만 사용 |
| `pre-trade-discipline-gate` | `desk/checklist.py` | 시장 게이트·자리 수·중복 진입 항목 추가, fail-closed 유지 |
| `trader-memory-core` | `desk/ledger.py` | YAML 상태 트리 → JSON Lines 한 줄 한 건으로 단순화 |
| `signal-postmortem` + `weekly-performance-digest` | `desk/review.py` | 백테스트 참조 분포와 나란히 출력 |

## 문턱은 왜 그대로 못 쓰는가 — 이식에서 가장 중요한 부분

원본 `drawdown-circuit-breaker` 의 기본값:

```
Max daily loss           2.0%  → HALTED
Losing streak cooldown   2건   → COOLDOWN
Weekly drawdown          5.0%  → HALTED
Monthly drawdown         8.0%  → HALTED
```

이 값을 이 프로젝트의 스윙 규칙(10자리·120일·손절 −8%·월말 게이트)에
그대로 적용하면, 21년 552건 실측에서 이렇게 된다.

```
연속 2패 쿨다운   45번 발동   ← 사실상 영구 정지
일 손실 2% 정지  113일 발동   (전체 거래일의 2.1%)
주간 5% 정지      13주 발동
월간 8% 정지       2개월 발동
```

이 규칙은 **승률 34%, 최장 연속 손실 32회**다. 연속 손실은 고장이 아니라
설계다. 원본 기본값은 승률이 높고 연속 손실이 짧은 전략을 전제한 값이고,
그런 전략에는 맞지만 이 전략에는 맞지 않는다.

그래서 연속 패배 문턱은 **쓰지 않고**, 나머지는 실측 분포의 꼬리에서 유도했다.

```
                실측 분포                        채택 문턱
일간    하위 0.1% −3.98%, 최악 −8.09%    →  −5%
주간    하위 1%   −5.14%, 최악 −10.18%   →  −8%
월간    하위 1%   −7.73%, 최악 −11.13%   →  −12%
12개월  최악      −23.5%                  →  −25%
```

원칙은 하나다 — **정상 범위의 나쁜 구간에서는 울리지 않고, 백테스트가
설명하지 못하는 구간에서만 울린다.** 자주 울리는 차단기는 전략을 죽인다.

`tests/test_desk.py::test_guard_ignores_a_long_losing_streak` 가 이 결정을
회귀로 고정한다.

## 무엇을 안 가져왔는가

73종 중 대부분은 이 프로젝트에 쓸 수 없다. 이유별로 적는다.

**미국 시장 전용 (약 40종)** — `canslim-screener`, `vcp-screener`,
`finviz-screener`, `stockbee-*` 5종, `pead-screener`, `ftd-detector`,
`ibd-distribution-day-monitor`, `us-market-bubble-detector` 등.
한국 종목·데이터에 맞지 않고, 옮기려면 재작성이 아니라 새로 만드는 것이 된다.

**유료 API 전제** — FMP 11곳, Alpaca 5곳, FINVIZ 4곳이 인덱스에 등장한다.

**이 프로젝트 범위 밖** — 옵션(`options-strategy-advisor`),
선물(`futures-position-sizer`), MT5(`mt5-robot-tester`),
일본 세무(`kanchi-dividend-us-tax-accounting`).

**이미 더 강한 것이 있음** — `backtest-expert`, `residual-edge-analyzer` 는
좋은 방법론 문서지만, 이 프로젝트의 `docs/methodology.md` 가 같은 주제를
실측 사례와 함께 더 깊게 다룬다 (다중검정 Šidák 문턱, 날짜 군집 보정,
겹침 보정, 체결일 시가 규약, 호가 한 틱 검사 등).

## 원본에 없어서 이쪽에서 더한 것

- **호가 단위 반올림** — 한국 시장은 가격대별 호가 단위가 있어 손절가를
  그 격자에 맞춰야 한다. 맞추지 않으면 주문이 안 들어간다.
- **중복 진입 차단** — 같은 종목을 두 자리에 담는 것을 기록 단계에서 막는다.
- **백테스트 참조 분포** — 리뷰가 "이번 분기 −8%" 가 아니라
  "백테스트 최악이 −23.5% 였는데 지금 −8%" 로 나오게 한다. 앞의 문장은
  판단할 수 없고 뒤의 문장은 판단할 수 있다.
- **표본 부족 경고** — 청산 20건 미만이면 성과로 판단하지 말라고 표시한다.

## 검증 상태에 대한 참고

원본 저장소의 `skills-index.yaml` 은 스킬마다 `verification` 블록을 두고
있는데, **73종 전부 `empirical_validation: not_verified`**(또는 `not_applicable`)
이고 45종은 어떤 검증 항목도 통과 표시가 없다.

이것은 비판이 아니다. 저장소가 스스로 "not a signal service or a promise of
profitability" 라고 명시하고, 메타데이터로 그 상태를 공개하고 있다.
다만 **이 스킬들을 쓴다고 해서 전략이 검증되는 것은 아니다**는 점은
분명히 해 둔다. 이식한 것은 성과가 아니라 규율이다.
