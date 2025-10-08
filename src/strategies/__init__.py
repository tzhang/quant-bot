"""
策略模块
包含各种交易策略的实现
"""

from .base_strategy import BaseStrategy
from .templates import (
    MomentumStrategy, 
    MeanReversionStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    MACDStrategy,
    PairsTradingStrategy,
    VolatilityBreakoutStrategy
)
from .strategy_tester import StrategyTester

# 实时交易策略
try:
    from .live_strategy import LiveTradingStrategy, StrategyConfig, StrategyManager
    LIVE_STRATEGY_AVAILABLE = True
except ImportError:
    LIVE_STRATEGY_AVAILABLE = False

__all__ = [
    'BaseStrategy',
    'MomentumStrategy', 
    'MeanReversionStrategy',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'MACDStrategy',
    'PairsTradingStrategy',
    'VolatilityBreakoutStrategy',
    'StrategyTester'
]

if LIVE_STRATEGY_AVAILABLE:
    __all__.extend(['LiveTradingStrategy', 'StrategyConfig', 'StrategyManager'])