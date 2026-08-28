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
