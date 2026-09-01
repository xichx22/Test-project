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
    rim_recovery: float = 0.93     # 우측 고점 / 좌측 고점 (하한)
    rim_overshoot: float = 1.15    # 우측 고점 / 좌측 고점 (상한). 컵은 좌우 테두리가
                                   # 비슷한 높이여야 한다. 상한이 없으면 좌측 고점을
                                   # 훨씬 뚫고 올라간 급등도 '컵' 으로 통과한다 —
                                   # 실측에서 전체 신호의 32.6% 가 비율 1.20 초과였고
                                   # 최대 10.2배였다.
    handle_at_rim: float = 0.88    # 핸들 고점 / 우측 고점. 핸들은 컵의 오른쪽 입술
                                   # 근처에서 만들어진다. 우측 고점 한참 아래에서
                                   # 생긴 눌림은 핸들이 아니라 붕괴 도중의 반등이다.
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
                if not (params.rim_recovery <= right_rim / left_rim
                        <= params.rim_overshoot):
                    continue
                # 핸들은 컵 오른쪽 입술 근처에서 만들어져야 한다
                if handle_high < right_rim * params.handle_at_rim:
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


def _envelope_lines(seg_high: np.ndarray, seg_low: np.ndarray, blocks: int = 6):
    """구간을 blocks 조각으로 나눠 각 조각의 고점/저점을 잇는 두 추세선.

    모든 봉에 최소제곱을 걸면 안 된다. 지그재그의 **한가운데**를 지나가서
    상단선과 하단선이 거의 같아지고, 실제로 수렴하는 쐐기도 폭 비율이
    0.9 로 나온다 (실측). 차트에서 사람이 그리듯 봉우리끼리·골끼리 이어야
    포락선이 된다.

    반환: (고점선 기울기, 고점선 절편, 저점선 기울기, 저점선 절편) — x 는 봉 인덱스.
    """
    n = len(seg_high)
    if n < blocks * 2:
        return None
    edges = np.linspace(0, n, blocks + 1).astype(int)
    xs, tops, bottoms = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xs.append((a + b - 1) / 2)
        tops.append(seg_high[a:b].max())
        bottoms.append(seg_low[a:b].min())
    if len(xs) < 3:
        return None
    xs = np.asarray(xs, dtype=float)
    hi_slope, hi_base = np.polyfit(xs, np.asarray(tops, float), 1)
    lo_slope, lo_base = np.polyfit(xs, np.asarray(bottoms, float), 1)
    return hi_slope, hi_base, lo_slope, lo_base


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


# 아래에 정의된 패턴들은 파일 끝에서 PATTERNS 에 추가된다 (정의 순서 때문).
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


# =====================================================================
# 역헤드앤숄더 (Inverse Head and Shoulders) — 가장 유명한 반전 패턴
# =====================================================================

@dataclass(frozen=True)
class InverseHeadShouldersParams:
    """왼어깨 – 머리 – 오른어깨, 그리고 넥라인 돌파.

    조건은 교과서(Edwards & Magee, Bulkowski)를 그대로 옮겼다. 이 데이터로
    맞춘 값이 아니다. 특히 어깨 대칭 허용치와 머리 깊이는 문헌에서 흔히
    쓰는 범위의 중간값을 골랐다.
    """

    window: int = 120           # 패턴 전체를 찾을 창
    head_deeper: float = 0.03   # 머리가 두 어깨보다 최소 3% 낮아야
    shoulder_gap: float = 0.15  # 두 어깨 저점 차이가 15% 안
    min_separation: int = 8     # 저점끼리 최소 간격 (봉)
    neckline_slope: float = 0.10  # 넥라인이 15% 넘게 기울면 형태가 아니다
    volume_mult: float = 1.3
    volume_window: int = 20
    prior_window: int = 60
    prior_gain: float = 0.05    # 반전 패턴이므로 '앞선 하락'을 본다 (음수 방향)


def inverse_head_and_shoulders(
    candles: pd.DataFrame,
    params: InverseHeadShouldersParams = InverseHeadShouldersParams(),
) -> pd.Series:
    """역헤드앤숄더 완성 봉(넥라인 돌파)에 True.

    구조 조건을 느슨하게 두면 아무 W 나 통과한다. 세 저점이 실제로
    저점인지(양옆보다 낮은지), 머리가 가운데 있는지를 강제한다.
    """
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < params.window + params.prior_window + 2:
        return pd.Series(out, index=candles.index)

    for t in range(params.prior_window + params.window, n):
        start = t - params.window
        seg_low = low[start:t]
        if len(seg_low) < 3 * params.min_separation:
            continue
        # 거래량이 안 실렸거나 이번 봉이 상승이 아니면 돌파일 수 없다.
        # 세 저점을 찾는 계산 전에 걸러 낸다 (결과는 같다).
        if not np.isfinite(avg[t]) or volume[t] < params.volume_mult * avg[t]:
            continue
        if close[t] <= close[t - 1]:
            continue

        # 세 구간으로 나눠 각 구간의 최저점을 어깨/머리 후보로 삼는다
        third = len(seg_low) // 3
        left_i = start + int(np.argmin(seg_low[:third]))
        head_i = start + third + int(np.argmin(seg_low[third:2 * third]))
        right_i = start + 2 * third + int(np.argmin(seg_low[2 * third:]))

        if head_i - left_i < params.min_separation:
            continue
        if right_i - head_i < params.min_separation:
            continue

        left, head, right = low[left_i], low[head_i], low[right_i]
        if not (head > 0 and left > 0 and right > 0):
            continue
        # 머리가 두 어깨보다 확실히 낮아야 한다
        if head > min(left, right) * (1 - params.head_deeper):
            continue
        # 두 어깨는 서로 비슷한 높이여야 한다
        if abs(left - right) / max(left, right) > params.shoulder_gap:
            continue

        # 넥라인 = 머리 양옆 반등 고점을 잇는 선
        peak_left = high[left_i:head_i].max() if head_i > left_i else np.nan
        peak_right = high[head_i:right_i].max() if right_i > head_i else np.nan
        if not (np.isfinite(peak_left) and np.isfinite(peak_right)):
            continue
        if abs(peak_left - peak_right) / max(peak_left, peak_right) > params.neckline_slope:
            continue
        neckline = max(peak_left, peak_right)

        # 반전 패턴이므로 앞선 하락을 확인한다
        prior = start - params.prior_window
        if prior < 0 or close[prior] <= 0:
            continue
        if close[start] / close[prior] - 1 > -params.prior_gain:
            continue

        if _breakout(close, volume, avg, t, neckline, params.volume_mult):
            out[t] = True
    return pd.Series(out, index=candles.index, name="inverse_head_and_shoulders")


# =====================================================================
# 하락쐐기 (Falling Wedge) — 고점도 저점도 내려오는데 수렴한다
# =====================================================================

@dataclass(frozen=True)
class FallingWedgeParams:
    window: int = 60
    min_contraction: float = 0.4   # 뒤쪽 폭이 앞쪽의 40% 이하로 좁아져야
    both_falling: float = 0.0      # 고점·저점 모두 하락 (기울기 < 0)
    volume_mult: float = 1.3
    volume_window: int = 20
    prior_window: int = 60
    prior_gain: float = 0.10


def falling_wedge(
    candles: pd.DataFrame,
    params: FallingWedgeParams = FallingWedgeParams(),
) -> pd.Series:
    """하락쐐기 상단 돌파에 True.

    삼각수렴과의 차이는 **두 선이 모두 아래로** 기운다는 것이다. 그 조건을
    빼면 그냥 수렴 패턴이 되므로 반드시 강제한다.
    """
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < params.window + params.prior_window + 2:
        return pd.Series(out, index=candles.index)

    x = np.arange(params.window, dtype=float)
    for t in range(params.prior_window + params.window, n):
        start = t - params.window
        seg_high, seg_low = high[start:t], low[start:t]
        if len(seg_high) < params.window:
            continue

        # 돌파부터 본다. 구조 판정(추세선 회귀 2회)은 봉마다 하면 너무 비싸고,
        # 돌파가 없으면 어차피 신호가 아니다. 순서만 바꾼 것이라 결과는 같다.
        half = len(seg_high) // 2
        pivot = seg_high[half:].max()
        if not _breakout(close, volume, avg, t, pivot, params.volume_mult):
            continue
        if not _prior_uptrend(close, start, params.prior_window, params.prior_gain):
            continue

        lines = _envelope_lines(seg_high, seg_low)
        if lines is None:
            continue
        hi_slope, hi_base, lo_slope, lo_base = lines
        span = x[:len(seg_high)]
        if hi_slope >= params.both_falling or lo_slope >= params.both_falling:
            continue
        # 고점이 저점보다 빨리 내려와야 수렴한다
        if hi_slope >= lo_slope:
            continue

        # 수축은 **두 추세선 사이 폭**으로 잰다. 구간 최고-최저로 재면
        # 하락 드리프트가 폭에 섞여 들어가, 실제로 수렴하는 쐐기도
        # 비율이 0.5 아래로 안 내려간다 (실측: 정석 모양에서 0.527).
        width_start = hi_base - lo_base
        width_end = (hi_base + hi_slope * span[-1]) - (lo_base + lo_slope * span[-1])
        if width_start <= 0 or width_end <= 0:
            continue
        if width_end / width_start > params.min_contraction:
            continue
        out[t] = True
    return pd.Series(out, index=candles.index, name="falling_wedge")


# =====================================================================
# VCP — 변동성 수축 패턴 (Mark Minervini)
# =====================================================================

@dataclass(frozen=True)
class VcpParams:
    """조정이 갈수록 얕아지고 거래량이 마르다가 터진다.

    미너비니가 대중화한 형태다. 핵심은 '조정 깊이가 단조 감소'라는 것 하나이고,
    나머지는 그것을 재는 방식일 뿐이다.
    """

    window: int = 80
    legs: int = 3               # 수축 구간을 몇 개로 나눠 볼 것인가
    shrink: float = 0.75        # 다음 조정은 직전의 75% 이하여야
    max_first_depth: float = 0.35
    dry_up: float = 0.85        # 마지막 구간 거래량이 앞선 평균의 85% 이하
    volume_mult: float = 1.5    # 돌파는 거래량이 크게 실려야
    volume_window: int = 50
    prior_window: int = 60
    prior_gain: float = 0.15


def volatility_contraction(
    candles: pd.DataFrame,
    params: VcpParams = VcpParams(),
) -> pd.Series:
    """VCP 돌파 봉에 True — 조정 깊이가 단조 감소 + 거래량 고갈 후 돌파."""
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < params.window + params.prior_window + 2:
        return pd.Series(out, index=candles.index)

    for t in range(params.prior_window + params.window, n):
        start = t - params.window
        size = params.window // params.legs
        depths, volumes = [], []
        for leg in range(params.legs):
            a = start + leg * size
            b = a + size
            seg_high, seg_low = high[a:b], low[a:b]
            if len(seg_high) == 0 or seg_high.max() <= 0:
                break
            depths.append(1 - seg_low.min() / seg_high.max())
            volumes.append(volume[a:b].mean())
        if len(depths) < params.legs:
            continue
        if depths[0] > params.max_first_depth:
            continue
        # 조정이 갈수록 얕아져야 한다
        if any(depths[i + 1] > depths[i] * params.shrink
               for i in range(params.legs - 1)):
            continue
        # 거래량이 말라야 한다
        if volumes[-1] > np.mean(volumes[:-1]) * params.dry_up:
            continue

        if not _prior_uptrend(close, start, params.prior_window, params.prior_gain):
            continue
        pivot = high[start:t].max()
        if _breakout(close, volume, avg, t, pivot, params.volume_mult):
            out[t] = True
    return pd.Series(out, index=candles.index, name="volatility_contraction")


# =====================================================================
# 하이 타이트 플래그 (High Tight Flag) — 오닐이 가장 강하다고 한 형태
# =====================================================================

@dataclass(frozen=True)
class HighTightFlagParams:
    """짧은 기간에 급등한 뒤 아주 얕게만 쉬는 형태.

    오닐은 '가장 드물지만 가장 강한' 패턴이라고 했다. 드물다는 건
    표본이 적다는 뜻이므로, 검정력 부족을 결과로 오독하지 않도록 주의한다.
    """

    run_window: int = 40        # 급등을 볼 창
    min_run: float = 0.90       # 그 기간에 90% 이상 상승
    flag_window: int = 20       # 이후 조정 구간
    max_pullback: float = 0.25  # 조정이 25% 안
    volume_mult: float = 1.3
    volume_window: int = 20


def high_tight_flag(
    candles: pd.DataFrame,
    params: HighTightFlagParams = HighTightFlagParams(),
) -> pd.Series:
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(close)
    out = np.zeros(n, dtype=bool)
    span = params.run_window + params.flag_window
    if n < span + 2:
        return pd.Series(out, index=candles.index)

    for t in range(span, n):
        run_start = t - span
        run_end = run_start + params.run_window
        if close[run_start] <= 0:
            continue
        peak = high[run_start:run_end].max()
        if peak / close[run_start] - 1 < params.min_run:
            continue
        flag_low = low[run_end:t].min()
        if peak <= 0 or 1 - flag_low / peak > params.max_pullback:
            continue
        if _breakout(close, volume, avg, t, peak, params.volume_mult):
            out[t] = True
    return pd.Series(out, index=candles.index, name="high_tight_flag")


# =====================================================================
# NR7 확장 돌파 (Toby Crabel) — 덜 알려졌지만 오래된 규칙
# =====================================================================

@dataclass(frozen=True)
class Nr7Params:
    """최근 7봉 중 일간 변동폭이 가장 좁은 봉 다음의 상방 돌파.

    "변동성은 수축과 확장을 반복한다"는 것만 쓰는 규칙이다. 방향을
    예측하지 않고 확장이 시작되는 쪽을 따라간다.
    """

    lookback: int = 7
    volume_mult: float = 1.2
    volume_window: int = 20
    trend_window: int = 50      # 추세 위에서만 롱을 잡는다


def nr7_breakout(candles: pd.DataFrame, params: Nr7Params = Nr7Params()) -> pd.Series:
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(close)
    out = np.zeros(n, dtype=bool)
    rng = high - low
    sma = pd.Series(close).rolling(params.trend_window,
                                   min_periods=params.trend_window).mean().to_numpy()
    for t in range(max(params.lookback, params.trend_window) + 1, n):
        window = rng[t - params.lookback: t]
        if len(window) < params.lookback or not np.isfinite(sma[t]):
            continue
        # 직전 봉이 최근 7봉 중 가장 좁았는가
        if rng[t - 1] > window.min():
            continue
        if close[t] <= sma[t]:
            continue
        if _breakout(close, volume, avg, t, high[t - 1], params.volume_mult):
            out[t] = True
    return pd.Series(out, index=candles.index, name="nr7_breakout")


# =====================================================================
# 파워 매수 (Pocket Pivot) — Morales & Kacher, 소문은 났지만 덜 검증된 것
# =====================================================================

@dataclass(frozen=True)
class PocketPivotParams:
    """상승 봉의 거래량이 '최근 하락 봉들의 최대 거래량'보다 큰 날.

    돌파를 기다리지 않고 베이스 안에서 먼저 들어가려는 규칙이다.
    기관 매집의 흔적을 거래량 비대칭으로 잡는다는 발상.
    """

    lookback: int = 10          # 최근 하락 봉을 볼 창
    trend_window: int = 50
    max_extension: float = 0.10  # 이동평균에서 10% 넘게 떨어져 있으면 늦었다


def pocket_pivot(candles: pd.DataFrame,
                 params: PocketPivotParams = PocketPivotParams()) -> pd.Series:
    high, low, close, volume, _ = _arrays(candles, params.trend_window)
    n = len(close)
    out = np.zeros(n, dtype=bool)
    sma = pd.Series(close).rolling(params.trend_window,
                                   min_periods=params.trend_window).mean().to_numpy()
    for t in range(params.trend_window + params.lookback, n):
        if close[t] <= close[t - 1]:
            continue
        if not np.isfinite(sma[t]) or sma[t] <= 0:
            continue
        if close[t] < sma[t]:
            continue
        if close[t] / sma[t] - 1 > params.max_extension:
            continue
        window = slice(t - params.lookback, t)
        down = volume[window][close[window] < close[t - params.lookback - 1: t - 1]]
        if len(down) == 0:
            continue
        if volume[t] > down.max():
            out[t] = True
    return pd.Series(out, index=candles.index, name="pocket_pivot")


# 널리 알려진 것 / 덜 알려진 것을 나눠 둔다. 유명세와 성과는 별개라는 것이
# 이 프로젝트에서 확인하려는 것 중 하나다.
FAMOUS_PATTERNS = ("cup_with_handle", "double_bottom", "flat_base", "bull_flag",
                   "ascending_triangle", "inverse_head_and_shoulders",
                   "falling_wedge")
LESSER_KNOWN_PATTERNS = ("volatility_contraction", "high_tight_flag",
                         "nr7_breakout", "pocket_pivot")

PATTERNS.update({
    "inverse_head_and_shoulders": inverse_head_and_shoulders,
    "falling_wedge": falling_wedge,
    "volatility_contraction": volatility_contraction,
    "high_tight_flag": high_tight_flag,
    "nr7_breakout": nr7_breakout,
    "pocket_pivot": pocket_pivot,
})


# =====================================================================
# 삼중바닥 (Triple Bottom) — 같은 높이의 저점 셋
# =====================================================================

@dataclass(frozen=True)
class TripleBottomParams:
    """세 저점이 비슷한 높이여야 하고, 사이의 반등이 충분해야 한다.

    쌍바닥과의 차이는 저점이 하나 더 있다는 것뿐이지만, 조건이 하나 늘면
    표본이 급격히 줄어든다. 문헌에서 '더 강한 신호'라고 말하는 이유가
    희소성 때문인지 형태 때문인지는 이 데이터로 갈리지 않는다.
    """

    window: int = 120
    level_gap: float = 0.06      # 세 저점의 높이 차이가 6% 안
    min_separation: int = 10     # 저점끼리 최소 간격
    min_bounce: float = 0.05     # 저점 사이 반등이 5% 이상
    volume_mult: float = 1.3
    volume_window: int = 20
    prior_window: int = 60
    prior_drop: float = 0.10     # 반전 패턴이므로 앞선 하락을 본다


def triple_bottom(
    candles: pd.DataFrame,
    params: TripleBottomParams = TripleBottomParams(),
) -> pd.Series:
    """삼중바닥 완성 봉(넥라인 돌파)에 True."""
    high, low, close, volume, avg = _arrays(candles, params.volume_window)
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < params.window + params.prior_window + 2:
        return pd.Series(out, index=candles.index)

    for t in range(params.prior_window + params.window, n):
        if not np.isfinite(avg[t]) or volume[t] < params.volume_mult * avg[t]:
            continue
        if close[t] <= close[t - 1]:
            continue

        start = t - params.window
        seg_low = low[start:t]
        third = len(seg_low) // 3
        if third < params.min_separation:
            continue
        idx = [start + int(np.argmin(seg_low[:third])),
               start + third + int(np.argmin(seg_low[third:2 * third])),
               start + 2 * third + int(np.argmin(seg_low[2 * third:]))]
        if idx[1] - idx[0] < params.min_separation:
            continue
        if idx[2] - idx[1] < params.min_separation:
            continue

        lows = [low[i] for i in idx]
        if min(lows) <= 0:
            continue
        # 세 저점이 같은 높이여야 한다 — 이게 삼중바닥의 정의다
        if (max(lows) - min(lows)) / min(lows) > params.level_gap:
            continue

        # 저점 사이에 실제 반등이 있어야 한다 (붙어 있는 저점은 하나로 본다)
        peaks = [high[idx[0]:idx[1]].max(), high[idx[1]:idx[2]].max()]
        if any(p / min(lows) - 1 < params.min_bounce for p in peaks):
            continue

        prior = start - params.prior_window
        if prior < 0 or close[prior] <= 0:
            continue
        if close[start] / close[prior] - 1 > -params.prior_drop:
            continue

        neckline = max(peaks)
        if _breakout(close, volume, avg, t, neckline, params.volume_mult):
            out[t] = True
    return pd.Series(out, index=candles.index, name="triple_bottom")


PATTERNS["triple_bottom"] = triple_bottom
FAMOUS_PATTERNS = FAMOUS_PATTERNS + ("triple_bottom",)
