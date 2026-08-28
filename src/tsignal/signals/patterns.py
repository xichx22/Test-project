"""차트 형태 패턴.

지표 교차와 달리 **모양**을 본다. 여러 봉에 걸친 구조를 찾아야 하므로
롤링 연산만으로는 안 되고 후보 구간을 직접 훑는다.

컵앤핸들 (Cup with Handle)
--------------------------
William O'Neil 의 CANSLIM 에서 온 패턴. 기준은 그의 책과 IBD 자료에서
통용되는 값을 따랐다.

    선행 상승 ─┐                        ┌── 핸들
               │   좌측 고점      우측 고점 ╲
               │      ╲            ╱      ╲_╱  ← 여기 돌파에서 매수
               │       ╲__      __╱
               │          ╲____╱
               │           (컵 바닥, U자)

  1. 선행 상승 : 컵 시작 전 60봉 동안 +20% 이상 (패턴은 상승 종목의 조정이다)
  2. 컵 기간   : 35~325봉 (7~65주)
  3. 컵 깊이   : 좌측 고점 대비 12~50% 하락
  4. U자 바닥  : 저점이 컵 구간의 **가운데 절반** 안에 위치 (V자 반등 제외)
  5. 우측 회복 : 우측 고점이 좌측 고점의 93% 이상까지 회복
  5-1. 좌측 고점이 **진짜 봉우리**여야 하고, **컵 시작부에 있어야** 한다.
       컵 길이를 여러 개 시도하므로, 이 조건이 없으면 창을 선행 상승 구간까지
       뒤로 늘려 저점을 인위적으로 가운데에 놓을 수 있다 — 그러면 V자 반등도
       컵으로 통과한다.
  6. 핸들 기간 : 5~20봉 (1~4주)
  7. 핸들 깊이 : 12% 이내 **그리고** 컵 깊이의 절반 이내
  8. 핸들 위치 : 핸들 저점이 컵 상단 절반 안에 (바닥 근처 조정은 핸들이 아니다)
  9. 핸들 거래량 : 컵 구간 평균보다 적어야 한다 (매물 소진)
 10. 돌파      : 종가가 핸들 고점을 상향 돌파 + 거래량 50봉 평균의 1.4배 이상

기준값은 전부 인자로 뺐다. 원 기준이 임의적이라는 점 자체가 검증 대상이므로,
느슨한 판과 엄격한 판을 나란히 재기 위해서다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CupHandleParams:
    """오닐 기준의 기본값. 완화판을 쓰려면 인자를 바꿔 넣는다."""

    cup_min: int = 35              # 컵 최소 길이 (봉)
    cup_max: int = 325             # 컵 최대 길이
    cup_depth_min: float = 0.12    # 컵 깊이 하한
    cup_depth_max: float = 0.50    # 컵 깊이 상한
    rim_recovery: float = 0.93     # 우측 고점 / 좌측 고점
    rim_is_peak: float = 0.98      # 좌측 고점이 직전 고점 대비 이 비율 이상 (진짜 봉우리인가)
    rim_lookback: int = 20         # 좌측 고점의 '직전' 을 몇 봉으로 볼지
    rim_position: float = 0.34     # 좌측 고점이 컵 앞쪽 1/3 구간의 이 비율 안에 있어야 한다
    trough_center: float = 0.50    # 저점이 들어가야 할 가운데 구간 비율 (U자 판정)
                                   # 0.50 = 컵의 가운데 절반. 컵 길이를 여러 개
                                   # 시도하므로 느슨하면 창을 늘려 V자도 통과한다.
    handle_min: int = 5            # 핸들 최소 길이
    handle_max: int = 20           # 핸들 최대 길이
    handle_depth_max: float = 0.12
    handle_vs_cup: float = 0.50    # 핸들 깊이 / 컵 깊이 상한
    handle_upper_half: float = 0.50  # 핸들 저점이 있어야 할 컵 상단 비율
    handle_volume_max: float = 1.0   # 핸들 평균거래량 / 컵 평균거래량 상한
    prior_gain: float = 0.20       # 컵 이전 60봉 상승률 하한
    prior_window: int = 60
    breakout_volume: float = 1.40  # 돌파봉 거래량 / 50봉 평균
    volume_window: int = 50


def cup_with_handle(
    candles: pd.DataFrame,
    params: CupHandleParams = CupHandleParams(),
    *,
    return_details: bool = False,
) -> pd.Series | pd.DataFrame:
    """컵앤핸들 돌파 봉에서 True.

    t 봉에서 판정할 때 t 이전 데이터와 t 봉의 종가·거래량만 쓴다 (미래참조 없음).
    체결은 검증 코드가 t+1 시가로 잡는다.
    """
    high = candles["high"].to_numpy(float)
    low = candles["low"].to_numpy(float)
    close = candles["close"].to_numpy(float)
    volume = candles["volume"].to_numpy(float)
    n = len(candles)

    hit = np.zeros(n, dtype=bool)
    details: list[dict] = []

    avg_volume = pd.Series(volume).rolling(
        params.volume_window, min_periods=params.volume_window
    ).mean().to_numpy()

    earliest = params.prior_window + params.cup_min + params.handle_min
    for t in range(earliest, n):
        if not np.isfinite(avg_volume[t]) or avg_volume[t] <= 0:
            continue
        if volume[t] < params.breakout_volume * avg_volume[t]:
            continue

        # --- 핸들: t 직전 구간에서 길이를 바꿔가며 찾는다 --------------------
        found = None
        for handle_len in range(params.handle_min, params.handle_max + 1):
            h_start = t - handle_len
            if h_start <= 0:
                break
            handle_high = high[h_start:t].max()
            handle_low = low[h_start:t].min()

            # 돌파: 직전까지는 핸들 고점 아래, 이번 봉에 위로
            if not (close[t] > handle_high and close[t - 1] <= handle_high):
                continue
            if handle_high <= 0:
                continue
            handle_depth = (handle_high - handle_low) / handle_high
            if handle_depth > params.handle_depth_max:
                continue

            # --- 컵: 핸들 직전 구간에서 길이를 바꿔가며 찾는다 ---------------
            for cup_len in range(params.cup_min, params.cup_max + 1, 5):
                c_end = h_start
                c_start = c_end - cup_len
                if c_start - params.prior_window < 0:
                    break

                third = max(1, cup_len // 3)
                left_seg = high[c_start:c_start + third]
                left_rim = left_seg.max()
                # 좌측 고점은 컵 시작부에 있어야 한다. 앞쪽 1/3 의 끝자락에 있다면
                # 그 창은 하락이 시작되기 전 구간까지 끌어온 것이다.
                if int(np.argmax(left_seg)) > params.rim_position * third:
                    continue
                right_rim = high[c_end - third:c_end].max()
                trough_idx = int(np.argmin(low[c_start:c_end])) + c_start
                trough = low[trough_idx]
                if left_rim <= 0 or trough <= 0:
                    continue

                cup_depth = (left_rim - trough) / left_rim
                if not (params.cup_depth_min <= cup_depth <= params.cup_depth_max):
                    continue
                if right_rim < left_rim * params.rim_recovery:
                    continue

                # 좌측 고점이 진짜 봉우리인가. 하락 도중의 한 점을 좌측 고점으로
                # 잡으면 V자 반등도 컵으로 통과해버린다.
                peak_from = max(0, c_start - params.rim_lookback)
                if left_rim < params.rim_is_peak * high[peak_from:c_start + third].max():
                    continue

                # U자: 저점이 가운데 구간에 있어야 한다 (V자 반등 제외)
                margin = (1 - params.trough_center) / 2
                position = (trough_idx - c_start) / cup_len
                if not (margin <= position <= 1 - margin):
                    continue

                # 핸들은 컵 상단 절반에서 형성돼야 한다
                if handle_low < trough + params.handle_upper_half * (left_rim - trough):
                    continue
                if handle_depth > params.handle_vs_cup * cup_depth:
                    continue

                # 핸들 거래량은 컵 평균보다 적어야 한다 (매물 소진)
                cup_volume = volume[c_start:c_end].mean()
                if cup_volume <= 0:
                    continue
                if volume[h_start:t].mean() > params.handle_volume_max * cup_volume:
                    continue

                # 선행 상승: 컵은 오르던 종목의 조정이어야 한다
                prior = close[c_start - params.prior_window]
                if prior <= 0 or close[c_start] / prior - 1 < params.prior_gain:
                    continue

                found = {
                    "cup_len": cup_len, "handle_len": handle_len,
                    "cup_depth": cup_depth, "handle_depth": handle_depth,
                    "left_rim": left_rim, "right_rim": right_rim, "trough": trough,
                    "pivot": handle_high,
                }
                break
            if found:
                break

        if found:
            hit[t] = True
            if return_details:
                details.append({"dt": candles.index[t], **found})

    series = pd.Series(hit, index=candles.index, name="cup_with_handle")
    if not return_details:
        return series
    frame = pd.DataFrame(details)
    return frame.set_index("dt") if not frame.empty else frame


# =====================================================================
# 공통 헬퍼 — 패턴들이 함께 쓰는 조건
# =====================================================================

def _prior_uptrend(close: np.ndarray, start: int, window: int, gain: float) -> bool:
    """베이스 시작 전에 상승이 있었는가.

    컵앤핸들 실측에서 이 조건을 빼자 연환산 초과수익이 +15.9% → +0.6% 로
    사라졌다. 베이스 패턴은 전부 '오르던 종목의 조정'이라는 전제 위에 있으므로
    모든 패턴에 같은 조건을 건다.
    """
    prior_idx = start - window
    if prior_idx < 0 or close[prior_idx] <= 0:
        return False
    return close[start] / close[prior_idx] - 1 >= gain


def _breakout(close: np.ndarray, volume: np.ndarray, avg_volume: np.ndarray,
              t: int, pivot: float, volume_mult: float) -> bool:
    """직전까지 pivot 아래였다가 이번 봉에 위로, 거래량이 실렸는가."""
    if not np.isfinite(avg_volume[t]) or avg_volume[t] <= 0:
        return False
    if volume[t] < volume_mult * avg_volume[t]:
        return False
    return close[t] > pivot >= close[t - 1]


def _arrays(candles: pd.DataFrame, volume_window: int):
    high = candles["high"].to_numpy(float)
    low = candles["low"].to_numpy(float)
    close = candles["close"].to_numpy(float)
    volume = candles["volume"].to_numpy(float)
    avg = pd.Series(volume).rolling(volume_window, min_periods=volume_window).mean().to_numpy()
    return high, low, close, volume, avg


# =====================================================================
# 쌍바닥 (Double Bottom, W 형)
# =====================================================================

@dataclass(frozen=True)
class DoubleBottomParams:
    """오닐의 double bottom base.

        좌측 고점 ╲          ┌─ 중간 고점(피벗)          ╱ 돌파
                   ╲        ╱          ╲              ╱
                    ╲      ╱            ╲            ╱
                     ╲____╱              ╲__________╱
                      1차 저점            2차 저점
    """

    base_min: int = 35
    base_max: int = 260
    depth_min: float = 0.12
    depth_max: float = 0.50
    low_tolerance: float = 0.07     # 2차 저점이 1차 저점 대비 ±이 비율 안
    peak_min_rise: float = 0.05     # 중간 고점이 저점보다 이만큼은 높아야 W 가 된다
    peak_below_rim: float = 0.97    # 중간 고점은 좌측 고점보다 낮아야 한다 (베이스 안)
    leg_min: int = 8                # 각 다리의 최소 길이
    prior_gain: float = 0.20
    prior_window: int = 60
    breakout_volume: float = 1.40
    volume_window: int = 50


def double_bottom(candles: pd.DataFrame, params: DoubleBottomParams = DoubleBottomParams()) -> pd.Series:
    """쌍바닥 돌파 봉에서 True. 피벗은 두 저점 사이의 중간 고점."""
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(candles)
    hit = np.zeros(n, dtype=bool)

    for t in range(params.prior_window + params.base_min, n):
        if not np.isfinite(avg[t]) or avg[t] <= 0 or volume[t] < params.breakout_volume * avg[t]:
            continue

        for base_len in range(params.base_min, params.base_max + 1, 5):
            b_start = t - base_len
            if b_start - params.prior_window < 0:
                break

            left_rim = high[b_start:b_start + max(1, base_len // 6)].max()
            if left_rim <= 0:
                continue

            # 1차 저점: 베이스 앞 절반, 2차 저점: 뒤 절반
            mid = b_start + base_len // 2
            i1 = int(np.argmin(low[b_start:mid])) + b_start
            i2 = int(np.argmin(low[mid:t])) + mid
            if i2 - i1 < params.leg_min * 2:
                continue

            l1, l2 = low[i1], low[i2]
            if l1 <= 0 or abs(l2 / l1 - 1) > params.low_tolerance:
                continue

            depth = (left_rim - min(l1, l2)) / left_rim
            if not (params.depth_min <= depth <= params.depth_max):
                continue

            # 중간 고점 = 피벗
            peak_idx = int(np.argmax(high[i1:i2])) + i1
            pivot = high[peak_idx]
            if peak_idx - i1 < params.leg_min or i2 - peak_idx < params.leg_min:
                continue
            if pivot / max(l1, l2) - 1 < params.peak_min_rise:
                continue
            if pivot > left_rim * params.peak_below_rim:
                continue

            if not _breakout(close, volume, avg, t, pivot, params.breakout_volume):
                continue
            if not _prior_uptrend(close, b_start, params.prior_window, params.prior_gain):
                continue

            hit[t] = True
            break

    return pd.Series(hit, index=candles.index, name="double_bottom")


# =====================================================================
# 플랫 베이스 (Flat Base)
# =====================================================================

@dataclass(frozen=True)
class FlatBaseParams:
    """오닐의 flat base — 상승 후 좁은 횡보. 깊이가 얕은 것이 특징이다."""

    base_min: int = 25              # 최소 5주
    base_max: int = 90
    depth_max: float = 0.15         # 베이스 전체 깊이 상한
    drift_max: float = 0.10         # 베이스 시작가 대비 종료가 이동폭 상한 (평평함)
    prior_gain: float = 0.20
    prior_window: int = 60
    breakout_volume: float = 1.40
    volume_window: int = 50


def flat_base(candles: pd.DataFrame, params: FlatBaseParams = FlatBaseParams()) -> pd.Series:
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(candles)
    hit = np.zeros(n, dtype=bool)

    for t in range(params.prior_window + params.base_min, n):
        if not np.isfinite(avg[t]) or avg[t] <= 0 or volume[t] < params.breakout_volume * avg[t]:
            continue

        for base_len in range(params.base_min, params.base_max + 1, 5):
            b_start = t - base_len
            if b_start - params.prior_window < 0:
                break

            top = high[b_start:t].max()
            bottom = low[b_start:t].min()
            if top <= 0 or (top - bottom) / top > params.depth_max:
                continue
            # 평평함: 시작과 끝이 크게 벌어지지 않아야 한다 (기울어진 채널 제외)
            if abs(close[t - 1] / close[b_start] - 1) > params.drift_max:
                continue
            if not _breakout(close, volume, avg, t, top, params.breakout_volume):
                continue
            if not _prior_uptrend(close, b_start, params.prior_window, params.prior_gain):
                continue

            hit[t] = True
            break

    return pd.Series(hit, index=candles.index, name="flat_base")


# =====================================================================
# 상승 깃발 (Bull Flag)
# =====================================================================

@dataclass(frozen=True)
class BullFlagParams:
    """급등(깃대) 후 얕고 짧은 조정(깃발), 그리고 재돌파."""

    pole_min: int = 5
    pole_max: int = 30
    pole_gain: float = 0.20         # 깃대 상승률 하한
    flag_min: int = 5
    flag_max: int = 25
    flag_retrace_max: float = 0.50  # 깃대 상승분의 절반 이상 되돌리면 깃발이 아니다
    flag_depth_max: float = 0.15
    flag_volume_max: float = 0.90   # 깃발 거래량 / 깃대 거래량
    flag_starts_at_top: float = 0.97  # 깃발 첫 봉 고가 / 깃대 고점.
                                      # 깃발은 깃대 꼭대기에서 시작해야 한다.
                                      # 없으면 하락 도중의 반등 구간을 깃발로 잡는다.
    breakout_volume: float = 1.40
    volume_window: int = 50


def bull_flag(candles: pd.DataFrame, params: BullFlagParams = BullFlagParams()) -> pd.Series:
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(candles)
    hit = np.zeros(n, dtype=bool)

    for t in range(params.volume_window + params.pole_max + params.flag_max, n):
        if not np.isfinite(avg[t]) or avg[t] <= 0 or volume[t] < params.breakout_volume * avg[t]:
            continue

        for flag_len in range(params.flag_min, params.flag_max + 1):
            f_start = t - flag_len
            flag_high = high[f_start:t].max()
            flag_low = low[f_start:t].min()
            if flag_high <= 0 or (flag_high - flag_low) / flag_high > params.flag_depth_max:
                continue
            if not _breakout(close, volume, avg, t, flag_high, params.breakout_volume):
                continue

            for pole_len in range(params.pole_min, params.pole_max + 1):
                p_start = f_start - pole_len
                if p_start < 0:
                    break
                pole_low = low[p_start]
                pole_high = high[p_start:f_start].max()
                if pole_low <= 0 or pole_high / pole_low - 1 < params.pole_gain:
                    continue
                # 깃발은 깃대 꼭대기에서 시작해야 한다
                if high[f_start] < pole_high * params.flag_starts_at_top:
                    continue
                # 되돌림이 깃대 상승분의 절반을 넘으면 깃발이 아니다
                if (pole_high - flag_low) / (pole_high - pole_low) > params.flag_retrace_max:
                    continue
                pole_volume = volume[p_start:f_start].mean()
                if pole_volume <= 0:
                    continue
                if volume[f_start:t].mean() > params.flag_volume_max * pole_volume:
                    continue

                hit[t] = True
                break
            if hit[t]:
                break

    return pd.Series(hit, index=candles.index, name="bull_flag")


# =====================================================================
# 상승 삼각형 (Ascending Triangle)
# =====================================================================

@dataclass(frozen=True)
class AscendingTriangleParams:
    """수평 저항 + 높아지는 저점. 매물이 소진되며 위로 밀린다는 형태."""

    base_min: int = 25
    base_max: int = 120
    resistance_band: float = 0.04   # 고점들이 이 폭 안에 모여야 수평 저항
    min_touches: int = 3            # 저항선 접촉 횟수
    slope_min: float = 0.0002       # 저점 회귀 기울기 하한 (봉당 비율)
    rising_low_min: float = 0.03    # 뒤 1/3 저점이 앞 1/3 저점보다 이만큼 높아야 한다
                                    # (회귀 기울기만 보면 창을 짧게 잡아 상승 구간만
                                    #  담는 식으로 우회된다)
    depth_max: float = 0.30
    prior_gain: float = 0.10
    prior_window: int = 60
    breakout_volume: float = 1.40
    volume_window: int = 50


def ascending_triangle(
    candles: pd.DataFrame, params: AscendingTriangleParams = AscendingTriangleParams()
) -> pd.Series:
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(candles)
    hit = np.zeros(n, dtype=bool)

    for t in range(params.prior_window + params.base_min, n):
        if not np.isfinite(avg[t]) or avg[t] <= 0 or volume[t] < params.breakout_volume * avg[t]:
            continue

        for base_len in range(params.base_min, params.base_max + 1, 5):
            b_start = t - base_len
            if b_start - params.prior_window < 0:
                break

            seg_high = high[b_start:t]
            seg_low = low[b_start:t]
            resistance = seg_high.max()
            if resistance <= 0:
                continue
            if (resistance - seg_low.min()) / resistance > params.depth_max:
                continue

            # 수평 저항: 고점들이 좁은 띠 안에 여러 번 닿아야 한다
            touches = int((seg_high >= resistance * (1 - params.resistance_band)).sum())
            if touches < params.min_touches:
                continue

            # 저점은 우상향해야 한다. 회귀 기울기만 쓰면 창을 짧게 잡아
            # 상승 구간만 담는 식으로 우회되므로, 앞/뒤 1/3 저점도 함께 비교한다.
            x = np.arange(len(seg_low), dtype=float)
            slope = np.polyfit(x, seg_low, 1)[0] / max(seg_low.mean(), 1e-9)
            if slope < params.slope_min:
                continue
            third = max(1, len(seg_low) // 3)
            early_low, late_low = seg_low[:third].min(), seg_low[-third:].min()
            if early_low <= 0 or late_low / early_low - 1 < params.rising_low_min:
                continue

            if not _breakout(close, volume, avg, t, resistance, params.breakout_volume):
                continue
            if not _prior_uptrend(close, b_start, params.prior_window, params.prior_gain):
                continue

            hit[t] = True
            break

    return pd.Series(hit, index=candles.index, name="ascending_triangle")


PATTERNS = {
    "cup_with_handle": lambda c: cup_with_handle(c),
    "double_bottom": double_bottom,
    "flat_base": flat_base,
    "bull_flag": bull_flag,
    "ascending_triangle": ascending_triangle,
}


def cup_with_handle_loose(candles: pd.DataFrame) -> pd.Series:
    """완화판. 원 기준이 임의적이라는 점을 확인하기 위한 대조군.

    컵을 짧게(20봉부터), 깊이·회복률·거래량 조건을 느슨하게 잡는다.
    엄격판보다 표본이 늘어나는 대신 '패턴다움'은 옅어진다.
    """
    return cup_with_handle(candles, CupHandleParams(
        cup_min=20, cup_depth_min=0.08, rim_recovery=0.85, rim_is_peak=0.90,
        trough_center=0.85, handle_depth_max=0.18, handle_vs_cup=0.80,
        handle_upper_half=0.30, handle_volume_max=1.3,
        prior_gain=0.05, breakout_volume=1.10,
    ))
