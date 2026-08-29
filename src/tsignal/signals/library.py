"""신호 라이브러리.

각 신호에는 `rationale` — 왜 이게 의미를 가질 수 있는지에 대한 가설 — 을 붙인다.
가설이 없는 신호는 검증할 수 없다. 검증 리포트는 이 가설이 데이터에서
살아남는지를 종목/기간별로 확인해주는 장치다.
"""

from __future__ import annotations

import pandas as pd

from ..indicators._util import cross_down, cross_up
from .base import LONG, SHORT, signal


def _f(features: pd.DataFrame, col: str) -> pd.Series:
    if col not in features.columns:
        raise KeyError(f"피처 '{col}' 가 없습니다. compute_all() 로 전체 지표를 계산했는지 확인하세요.")
    return features[col]


# =====================================================================
# 1. 추세 돌파 (breakout) — 가격이 기준선/레인지를 뚫는 순간
# =====================================================================

@signal("donchian_breakout", "breakout",
        rationale="20봉 신고가 돌파는 매도벽 소진을 뜻한다는 가설. 추세추종의 원형.",
        tags=("classic", "trend"))
def donchian_breakout(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    # 직전 봉까지의 채널을 기준으로 해야 자기 자신을 뚫는 자명한 신호가 되지 않는다.
    return c["close"] > _f(f, "dc_upper").shift(1)


@signal("donchian_breakout_vol", "breakout",
        rationale="같은 돌파라도 거래량이 실렸을 때만 유효하다는 가설. 거래량 확인의 순증분을 잰다.",
        tags=("confirmation", "volume"))
def donchian_breakout_vol(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return donchian_breakout(c, f) & (_f(f, "rvol_20") > 2.0)


@signal("squeeze_release_up", "volatility",
        rationale="변동성 수축(볼린저⊂켈트너) 뒤의 확장은 방향성 이동을 동반한다는 가설.",
        tags=("regime",))
def squeeze_release_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return (_f(f, "squeeze_release") == 1) & (c["close"] > _f(f, "bb_mid"))


@signal("vwap_reclaim", "breakout",
        rationale="장중 평균단가 회복은 매수세 우위 전환의 신호라는 가설. 단타의 핵심 기준선.",
        tags=("intraday", "session"))
def vwap_reclaim(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    # 개장 직후 VWAP 은 표본이 얇아 노이즈다 → 최소 10봉 경과 후만 인정.
    return cross_up(c["close"], _f(f, "vwap")) & (_f(f, "session_bar") >= 10)


@signal("opening_range_breakout", "breakout",
        rationale="개장 30분 레인지 상단 돌파가 그날의 방향을 결정한다는 ORB 가설.",
        tags=("intraday", "session", "classic"))
def opening_range_breakout(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    bar = _f(f, "session_bar")
    session = pd.Series(c.index.date, index=c.index)
    opening = c["high"].where(bar < 6)
    or_high = opening.groupby(session).cummax().groupby(session).ffill()
    return (bar >= 6) & (c["close"] > or_high) & (c["close"].shift(1) <= or_high.shift(1))


@signal("bollinger_upper_break", "breakout",
        rationale="변동성 조정 상단 이탈은 추세 가속 구간이라는 가설.",
        tags=("classic",))
def bollinger_upper_break(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(c["close"], _f(f, "bb_upper"))


# =====================================================================
# 2. 추세 눌림목 (pullback) — 추세는 살아있고 가격만 되돌린 지점
# =====================================================================

@signal("ema_pullback", "pullback",
        rationale="정배열 추세에서 20EMA 되돌림은 저위험 재진입 지점이라는 가설.",
        tags=("trend",))
def ema_pullback(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    uptrend = (_f(f, "ribbon_align") == 1) & (_f(f, "ema_20") > _f(f, "ema_60"))
    touched = c["low"] <= _f(f, "ema_20")
    return uptrend & touched & (c["close"] > _f(f, "ema_20"))


@signal("supertrend_flip_up", "pullback",
        rationale="ATR 스톱의 방향 전환은 추세 전환의 객관적 정의라는 가설.",
        tags=("trend",))
def supertrend_flip_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    d = _f(f, "supertrend_dir")
    return (d == 1) & (d.shift(1) == -1)


@signal("adx_trend_start", "pullback",
        rationale="ADX 가 20을 상향 돌파하며 +DI 가 우위면 추세 초입이라는 가설.",
        tags=("filter", "trend"))
def adx_trend_start(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(_f(f, "adx"), pd.Series(20.0, index=c.index)) & (_f(f, "plus_di") > _f(f, "minus_di"))


@signal("vwap_pullback_hold", "pullback",
        rationale="VWAP 위에서 눌림 후 지지받으면 장중 추세가 유지된다는 가설.",
        tags=("intraday", "session"))
def vwap_pullback_hold(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    dev = _f(f, "vwap_dev_pct")
    return (dev > 0) & (dev.shift(1) < 0.1) & (c["close"] > c["open"]) & (_f(f, "session_bar") >= 10)


# =====================================================================
# 3. 평균회귀 (reversion) — 과도한 이탈의 되돌림
# =====================================================================

@signal("envelope_lower_touch", "reversion",
        rationale="20SMA -2% 엔벌로프 하단 이탈은 단기 과매도라는 고전 가설. "
                  "고정 % 밴드라 변동성 레짐에 따라 유효성이 갈릴 것으로 예상 — 그 차이를 재는 게 목적.",
        tags=("classic", "envelope"))
def envelope_lower_touch(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_down(c["close"], _f(f, "env_lower"))


@signal("envelope_lower_reclaim", "reversion",
        rationale="엔벌로프 하단을 이탈했다가 되돌아오는 순간이 진짜 반전이라는 가설. "
                  "단순 이탈(envelope_lower_touch) 대비 얼마나 개선되는지가 관전 포인트.",
        tags=("envelope",))
def envelope_lower_reclaim(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    below = c["close"] < _f(f, "env_lower")
    return (~below) & below.shift(1, fill_value=False)


@signal("envelope_atr_lower_reclaim", "reversion",
        rationale="엔벌로프 폭을 ATR 로 적응시키면 고정 % 판의 변동성 편향이 사라진다는 가설.",
        tags=("envelope", "adaptive"))
def envelope_atr_lower_reclaim(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    below = c["close"] < _f(f, "enva_lower")
    return (~below) & below.shift(1, fill_value=False)


@signal("williams_oversold_turn", "reversion",
        rationale="%R 이 -80 아래에서 위로 돌아서는 순간이 매수 타이밍이라는 가설. "
                  "강한 하락추세에서는 %R 이 눌러앉으므로, 추세 필터 유무의 차이를 함께 잰다.",
        tags=("classic", "williams"))
def williams_oversold_turn(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    wr = _f(f, "williams_r_14")
    return (wr.shift(1) <= -80) & (wr > -80)


@signal("williams_oversold_turn_trend", "reversion",
        rationale="같은 %R 반전이라도 상위 추세(종가>60EMA)에서만 잡으면 승률이 오른다는 가설.",
        tags=("williams", "filter"))
def williams_oversold_turn_trend(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return williams_oversold_turn(c, f) & (c["close"] > _f(f, "ema_60"))


@signal("rsi_oversold_turn", "reversion",
        rationale="RSI 30 이탈 후 회복은 매도 소진 신호라는 가설.",
        tags=("classic",))
def rsi_oversold_turn(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    r = _f(f, "rsi_14")
    return (r.shift(1) <= 30) & (r > 30)


@signal("stochrsi_oversold_cross", "reversion",
        rationale="StochRSI 는 RSI 보다 예민해 분봉 단타 트리거로 적합하다는 가설.",
        tags=("intraday",))
def stochrsi_oversold_cross(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    k, d = _f(f, "stochrsi_k"), _f(f, "stochrsi_d")
    return cross_up(k, d) & (k < 20)


@signal("bollinger_lower_reclaim", "reversion",
        rationale="변동성 조정 하단 회복은 통계적 되돌림 지점이라는 가설.",
        tags=("classic",))
def bollinger_lower_reclaim(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    below = c["close"] < _f(f, "bb_lower")
    return (~below) & below.shift(1, fill_value=False)


@signal("cci_oversold_turn", "reversion",
        rationale="CCI -100 회복은 평균 이탈의 되돌림 시작이라는 가설.")
def cci_oversold_turn(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    x = _f(f, "cci_20")
    return (x.shift(1) <= -100) & (x > -100)


@signal("rsi_bullish_divergence", "reversion",
        rationale="가격은 저점을 낮추는데 모멘텀은 낮추지 않으면 하락이 소진됐다는 가설.")
def rsi_bullish_divergence(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "rsi_divergence") == 1


# =====================================================================
# 4. 모멘텀 교차 (momentum)
# =====================================================================

@signal("macd_cross_up", "momentum",
        rationale="MACD 시그널 상향 교차는 단기 모멘텀 우위 전환이라는 가설.",
        tags=("classic",))
def macd_cross_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(_f(f, "macd"), _f(f, "macd_signal"))


@signal("macd_cross_up_zero", "momentum",
        rationale="0선 위에서의 MACD 교차만 취하면 추세 역행 신호가 걸러진다는 가설.",
        tags=("filter",))
def macd_cross_up_zero(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return macd_cross_up(c, f) & (_f(f, "macd") > 0)


@signal("dema_golden_cross", "momentum",
        rationale="DEMA 는 EMA 보다 지연이 작아 단타 교차 신호가 더 빠르다는 가설. "
                  "다만 노이즈도 같이 커지므로, 같은 파라미터의 EMA 교차와 직접 비교한다.",
        tags=("dema",))
def dema_golden_cross(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(_f(f, "dema_5"), _f(f, "dema_20"))


@signal("ema_golden_cross", "momentum",
        rationale="DEMA 교차의 비교 기준선(baseline). 지연이 큰 대신 노이즈가 적다는 가설.",
        tags=("baseline",))
def ema_golden_cross(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(_f(f, "ema_5"), _f(f, "ema_20"))


@signal("tema_golden_cross", "momentum",
        rationale="TEMA 는 지연을 한 겹 더 걷어낸다 — 지연↓/노이즈↑ 트레이드오프의 극단값.",
        tags=("tema",))
def tema_golden_cross(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(_f(f, "tema_5"), _f(f, "tema_20"))


@signal("tsi_cross_up", "momentum",
        rationale="이중 평활 모멘텀의 교차는 오탐이 적다는 가설.")
def tsi_cross_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(_f(f, "tsi"), _f(f, "tsi_signal"))


@signal("stoch_cross_up", "momentum",
        rationale="느린 스토캐스틱 %K/%D 상향 교차는 단기 반전 트리거라는 가설.",
        tags=("classic",))
def stoch_cross_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(_f(f, "stoch_k"), _f(f, "stoch_d")) & (_f(f, "stoch_k") < 30)


# =====================================================================
# 5. 거래량 (volume)
# =====================================================================

@signal("volume_spike_up", "volume",
        rationale="거래량 급증 + 양봉은 신규 자금 유입이라는 가설. 가장 단순한 이벤트 드리븐 신호.",
        tags=("event",))
def volume_spike_up(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return (_f(f, "volz_20") > 3.0) & (c["close"] > c["open"])


@signal("mfi_oversold_turn", "volume",
        rationale="거래량 가중 RSI 의 반전은 가격만 보는 RSI 보다 신뢰도가 높다는 가설.")
def mfi_oversold_turn(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    x = _f(f, "mfi_14")
    return (x.shift(1) <= 20) & (x > 20)


@signal("cmf_turn_positive", "volume",
        rationale="자금흐름이 순매수로 전환되면 가격이 따라온다는 가설.")
def cmf_turn_positive(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(_f(f, "cmf_20"), pd.Series(0.0, index=c.index))


# =====================================================================
# 7. 차트 형태 패턴
# =====================================================================

@signal("cup_with_handle", "pattern",
        rationale="오닐 CANSLIM 의 컵앤핸들. 오르던 종목이 U자로 조정한 뒤 얕은 "
                  "핸들을 만들고 거래량 실린 돌파를 내면 상승이 이어진다는 가설. "
                  "지표 교차가 아니라 여러 봉에 걸친 '모양'을 본다.",
        tags=("pattern", "oneil", "swing"))
def cup_with_handle_signal(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    from .patterns import cup_with_handle

    return cup_with_handle(c)


@signal("cup_with_handle_loose", "pattern",
        rationale="컵앤핸들의 완화판 대조군. 기준을 느슨하게 하면 '패턴다움'이 사라져 "
                  "성과도 사라지는지 확인한다 — 원 기준의 구체성이 우연이 아님을 보이는 장치.",
        tags=("pattern", "control"))
def cup_with_handle_loose_signal(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    from .patterns import cup_with_handle_loose

    return cup_with_handle_loose(c)


@signal("inverse_head_and_shoulders", "pattern",
        rationale="가장 널리 알려진 반전 패턴. 왼어깨-머리-오른어깨 세 저점을 만든 뒤 "
                  "넥라인을 거래량과 함께 뚫으면 하락 추세가 끝났다는 가설.",
        tags=("pattern", "famous", "reversal", "swing"))
def inverse_head_and_shoulders_signal(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    from .patterns import inverse_head_and_shoulders

    return inverse_head_and_shoulders(c)


@signal("falling_wedge", "pattern",
        rationale="고점도 저점도 내려오는데 폭이 좁아지는 형태. 매도 압력이 "
                  "소진되는 중이라 상단을 뚫으면 방향이 바뀐다는 가설.",
        tags=("pattern", "famous", "reversal", "swing"))
def falling_wedge_signal(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    from .patterns import falling_wedge

    return falling_wedge(c)


@signal("volatility_contraction", "pattern",
        rationale="미너비니 VCP. 조정이 갈수록 얕아지고 거래량이 마르는 것은 "
                  "매도 물량이 소진되는 흔적이고, 그 뒤 돌파가 잘 간다는 가설.",
        tags=("pattern", "lesser_known", "minervini", "swing"))
def volatility_contraction_signal(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    from .patterns import volatility_contraction

    return volatility_contraction(c)


@signal("high_tight_flag", "pattern",
        rationale="오닐이 '가장 드물지만 가장 강하다'고 한 형태. 짧은 기간 급등한 뒤 "
                  "아주 얕게만 쉬면 매물이 없다는 뜻이라는 가설. 표본이 적다.",
        tags=("pattern", "lesser_known", "oneil", "swing"))
def high_tight_flag_signal(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    from .patterns import high_tight_flag

    return high_tight_flag(c)


@signal("nr7_breakout", "pattern",
        rationale="크라벨 NR7. 변동성은 수축과 확장을 반복하므로, 최근 7봉 중 가장 "
                  "좁았던 봉 다음의 확장 방향을 따라간다. 방향을 예측하지 않는다.",
        tags=("pattern", "lesser_known", "crabel", "volatility"))
def nr7_breakout_signal(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    from .patterns import nr7_breakout

    return nr7_breakout(c)


@signal("pocket_pivot", "pattern",
        rationale="모랄레스·캐처의 파워 매수. 상승 봉 거래량이 최근 하락 봉들의 최대 "
                  "거래량을 넘으면 기관이 매집 중이라는 가설. 돌파 전에 들어간다.",
        tags=("pattern", "lesser_known", "volume", "swing"))
def pocket_pivot_signal(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    from .patterns import pocket_pivot

    return pocket_pivot(c)


# =====================================================================
# 6. 청산 신호 (exit) — "언제 팔 것인가"의 후보들
# =====================================================================

@signal("exit_macd_cross_down", "momentum", side=SHORT, kind="exit",
        rationale="진입 근거였던 모멘텀 우위가 사라지면 나간다.")
def exit_macd_cross_down(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_down(_f(f, "macd"), _f(f, "macd_signal"))


@signal("exit_supertrend_flip_down", "pullback", side=SHORT, kind="exit",
        rationale="ATR 추세 스톱이 뒤집히면 추세추종 포지션의 전제가 깨진 것.")
def exit_supertrend_flip_down(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    d = _f(f, "supertrend_dir")
    return (d == -1) & (d.shift(1) == 1)


@signal("exit_envelope_upper", "reversion", side=SHORT, kind="exit",
        rationale="평균회귀 진입은 반대편 밴드(또는 중심선)에서 목표 달성으로 본다.")
def exit_envelope_upper(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return c["close"] >= _f(f, "env_upper")


@signal("exit_envelope_mid", "reversion", side=SHORT, kind="exit",
        rationale="상단까지 기다리지 않고 중심선에서 끊는 보수적 청산. 상단 청산과 기대값을 비교한다.")
def exit_envelope_mid(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_up(c["close"], _f(f, "env_mid"))


@signal("exit_williams_overbought", "reversion", side=SHORT, kind="exit",
        rationale="%R 이 -20 위로 올라오면 단기 과매수 — 되돌림 트레이드의 목표 도달.")
def exit_williams_overbought(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "williams_r_14") > -20


@signal("exit_rsi_overbought", "reversion", side=SHORT, kind="exit",
        rationale="RSI 70 도달은 단기 과열 — 분할 청산의 고전적 기준.")
def exit_rsi_overbought(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return _f(f, "rsi_14") > 70


@signal("exit_vwap_lose", "breakout", side=SHORT, kind="exit",
        rationale="장중 기준선을 내주면 그날의 매수 우위가 끝난 것.",
        tags=("intraday", "session"))
def exit_vwap_lose(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_down(c["close"], _f(f, "vwap"))


@signal("exit_ema20_lose", "pullback", side=SHORT, kind="exit",
        rationale="추세 진입의 최소 조건인 20EMA 이탈.")
def exit_ema20_lose(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return cross_down(c["close"], _f(f, "ema_20"))


@signal("exit_session_close", "breakout", side=SHORT, kind="exit",
        rationale="단타는 오버나이트 갭 리스크를 지지 않는다 — 장 마감 전 강제 청산.",
        tags=("intraday", "session", "risk"))
def exit_session_close(c: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    # 거래소 일정은 사전에 알려진 정보이므로 시각 비교는 미래참조가 아니다.
    # (shift(-1) 로 '다음 봉이 있는지' 보는 방식은 미래참조가 되므로 쓰지 않는다.)
    minutes = c.index.hour * 60 + c.index.minute
    return pd.Series(minutes >= 15 * 60 + 10, index=c.index)
