"""
回测引擎模块

提供增强的回测功能
"""

from .enhanced_backtest_engine import (
    EventType,
    OrderType,
    OrderSide,
    OrderStatus,
    Order,
    Fill,
    Position,
    Event,
    EventHandler,
    MarketDataHandler,
    SignalHandler,
    OrderHandler,
    FillHandler,
    BaseStrategy,
    SimpleMovingAverageStrategy,
    Portfolio,
    EnhancedBacktestEngine,
    MultiStrategyBacktestManager
)

__all__ = [
    "EventType",
    "OrderType", 
    "OrderSide",
    "OrderStatus",
    "Order",
    "Fill",
    "Position",
    "Event",
    "EventHandler",
    "MarketDataHandler",
    "SignalHandler",
    "OrderHandler",
    "FillHandler",
    "BaseStrategy",
    "SimpleMovingAverageStrategy",
    "Portfolio",
    "EnhancedBacktestEngine",
    "MultiStrategyBacktestManager"
]