"""tsignal — 국내주식 단타 신호의 통계적 근거를 만들기 위한 파이프라인.

데이터소스 → 지표 → 신호 → 검증 → 리포트
"""

from . import datasource, evaluation, indicators, ohlcv, signals

__version__ = "0.1.0"
__all__ = ["datasource", "indicators", "signals", "evaluation", "ohlcv"]
