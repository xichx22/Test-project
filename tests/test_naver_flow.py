"""투자자별 수급 파싱 테스트.

`lxml` 의존을 피해 정규식으로 파싱하므로, 표 구조 가정이 깨지면 조용히
빈 결과가 나올 수 있다. 실제 응답 형태를 고정해 둔다.
"""

import numpy as np
import pandas as pd
import pytest

from tsignal.datasource.naver_flow import COLUMNS, flow_features, parse_flow_page

# 실제 응답에서 그대로 가져온 구조 (한 행에 td 9개)
PAGE = """
<table>
<tr><th>날짜</th><th>종가</th><th>전일비</th><th>등락률</th><th>거래량</th>
    <th>기관</th><th>외국인</th><th>보유주수</th><th>보유율</th></tr>
<tr onmouseover="mouseOver(this)">
  <td class="tc">2026.08.28</td><td class="num">257,000</td>
  <td class="num"><span class="tah p11 nv01">하락
        9,000</span></td>
  <td class="num">-3.38%</td><td class="num">14,698,877</td>
  <td class="num">-1,652,437</td><td class="num">-1,927,155</td>
  <td class="num">2,731,215,894</td><td class="num">46.72%</td>
</tr>
<tr onmouseover="mouseOver(this)">
  <td class="tc">2026.08.27</td><td class="num">266,000</td>
  <td class="num">상승 4,500</td><td class="num">+1.72%</td><td class="num">16,829,395</td>
  <td class="num">-97,433</td><td class="num">+1,381,786</td>
  <td class="num">2,733,143,049</td><td class="num">46.75%</td>
</tr>
<tr><td colspan="9">&nbsp;</td></tr>
</table>
"""


def test_parses_rows_and_signs():
    frame = parse_flow_page(PAGE)
    assert list(frame.columns) == COLUMNS
    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing
    assert str(frame.index.tz) == "Asia/Seoul"

    latest = frame.iloc[-1]
    assert latest["inst_net"] == -1_652_437       # 쉼표와 음수 부호를 살려야 한다
    assert latest["foreign_net"] == -1_927_155
    assert latest["foreign_rate"] == 46.72        # % 기호 제거
    assert frame.iloc[0]["foreign_net"] == 1_381_786   # +부호도 숫자로


def test_ignores_rows_without_a_date():
    frame = parse_flow_page(PAGE)
    assert len(frame) == 2                        # 헤더와 빈 행은 걸러진다


def test_empty_page_returns_empty_frame():
    frame = parse_flow_page("<table><tr><td>데이터가 없습니다</td></tr></table>")
    assert frame.empty
    assert list(frame.columns) == COLUMNS


def test_flow_features_normalize_by_volume():
    """순매매 '주식 수' 는 종목마다 스케일이 달라 거래량으로 나눠야 비교된다."""
    frame = parse_flow_page(PAGE)
    volume = pd.Series([16_829_395.0, 14_698_877.0], index=frame.index)
    out = flow_features(frame, volume)

    assert out["foreign_ratio"].iloc[0] == pytest.approx(1_381_786 / 16_829_395)
    assert out["inst_ratio"].iloc[-1] == pytest.approx(-1_652_437 / 14_698_877)
    assert out["combined_ratio"].iloc[-1] == pytest.approx(
        out["inst_ratio"].iloc[-1] + out["foreign_ratio"].iloc[-1]
    )
    # 비중이므로 대략 -1~+1 범위 안에 들어와야 한다.
    assert out["combined_ratio"].abs().max() < 2


def test_flow_features_survive_zero_volume():
    """거래정지일(거래량 0)에 0으로 나누지 않는다."""
    frame = parse_flow_page(PAGE)
    volume = pd.Series([0.0, 14_698_877.0], index=frame.index)
    out = flow_features(frame, volume)
    assert np.isnan(out["inst_ratio"].iloc[0])
    assert np.isfinite(out["inst_ratio"].iloc[-1])
    assert out["inst_ratio"].dtype == np.float64      # object dtype 으로 새지 않아야 한다
