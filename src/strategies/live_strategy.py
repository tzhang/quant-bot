"""
实时交易策略模块
实现动量策略和均值回归策略的实时交易逻辑
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ..trading.ib_trading_manager import TradingSignal, TradeOrder


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_type: str  # 'momentum' 或 'mean_reversion'
    lookback_period: int = 20  # 回看周期
    signal_threshold: float = 0.02  # 信号阈值
    position_size: float = 0.1  # 仓位大小（占总资金比例）
    stop_loss_pct: float = 0.05  # 止损百分比
    take_profit_pct: float = 0.10  # 止盈百分比
    max_positions: int = 5  # 最大持仓数量


class LiveTradingStrategy:
    """实时交易策略基类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.price_history: Dict[str, List[float]] = {}
        self.last_signals: Dict[str, TradingSignal] = {}
        
    def update_price(self, symbol: str, price: float, timestamp: datetime = None):
        """更新价格数据"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append(price)
        
        # 保持历史数据在合理范围内
        max_history = self.config.lookback_period * 2
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """生成交易信号"""
        if symbol not in self.price_history:
            return None
            
        prices = self.price_history[symbol]
        if len(prices) < self.config.lookback_period:
            return None
            
        # 根据策略类型生成信号
        if self.config.strategy_type == 'momentum':
            return self._momentum_signal(symbol, prices)
        elif self.config.strategy_type == 'mean_reversion':
            return self._mean_reversion_signal(symbol, prices)
        else:
            self.logger.warning(f"未知策略类型: {self.config.strategy_type}")
            return None
    
    def _momentum_signal(self, symbol: str, prices: List[float]) -> Optional[TradingSignal]:
        """动量策略信号"""
        if len(prices) < self.config.lookback_period:
            return None
            
        # 计算短期和长期移动平均
        short_ma = np.mean(prices[-5:])  # 5日均线
        long_ma = np.mean(prices[-self.config.lookback_period:])  # 长期均线
        current_price = prices[-1]
        
        # 计算动量指标
        momentum = (short_ma - long_ma) / long_ma
        
        signal_type = None
        confidence = abs(momentum)
        
        if momentum > self.config.signal_threshold:
            signal_type = 'BUY'
        elif momentum < -self.config.signal_threshold:
            signal_type = 'SELL'
        
        if signal_type:
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=min(confidence, 1.0),
                price=current_price,
                timestamp=datetime.now(),
                strategy='momentum',
                metadata={
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'momentum': momentum
                }
            )
        
        return None
    
    def _mean_reversion_signal(self, symbol: str, prices: List[float]) -> Optional[TradingSignal]:
        """均值回归策略信号"""
        if len(prices) < self.config.lookback_period:
            return None
            
        # 计算统计指标
        price_array = np.array(prices[-self.config.lookback_period:])
        mean_price = np.mean(price_array)
        std_price = np.std(price_array)
        current_price = prices[-1]
        
        # 计算Z-score
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
        
        signal_type = None
        confidence = abs(z_score) / 2.0  # 标准化置信度
        
        # 均值回归信号（价格偏离均值时反向操作）
        if z_score > self.config.signal_threshold * 2:  # 价格过高，卖出
            signal_type = 'SELL'
        elif z_score < -self.config.signal_threshold * 2:  # 价格过低，买入
            signal_type = 'BUY'
        
        if signal_type:
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=min(confidence, 1.0),
                price=current_price,
                timestamp=datetime.now(),
                strategy='mean_reversion',
                metadata={
                    'mean_price': mean_price,
                    'std_price': std_price,
                    'z_score': z_score
                }
            )
        
        return None
    
    def create_order(self, signal: TradingSignal, account_value: float, 
                    current_positions: Dict[str, float]) -> Optional[TradeOrder]:
        """根据信号创建订单"""
        if not signal:
            return None
            
        # 检查是否已有相同方向的信号
        last_signal = self.last_signals.get(signal.symbol)
        if (last_signal and 
            last_signal.signal_type == signal.signal_type and
            (datetime.now() - last_signal.timestamp).seconds < 300):  # 5分钟内不重复信号
            return None
        
        # 计算仓位大小
        position_value = account_value * self.config.position_size
        quantity = int(position_value / signal.price)
        
        if quantity <= 0:
            return None
        
        # 检查最大持仓限制
        if len(current_positions) >= self.config.max_positions:
            self.logger.warning(f"已达到最大持仓数量限制: {self.config.max_positions}")
            return None
        
        # 调整数量（卖出时考虑当前持仓）
        if signal.signal_type == 'SELL':
            current_qty = current_positions.get(signal.symbol, 0)
            if current_qty <= 0:
                return None  # 没有持仓，无法卖出
            quantity = min(quantity, abs(current_qty))
        
        # 创建订单
        order = TradeOrder(
            symbol=signal.symbol,
            action=signal.signal_type,
            quantity=quantity,
            order_type='MKT',  # 市价单
            price=signal.price,
            stop_loss=signal.price * (1 - self.config.stop_loss_pct) if signal.signal_type == 'BUY' 
                     else signal.price * (1 + self.config.stop_loss_pct),
            take_profit=signal.price * (1 + self.config.take_profit_pct) if signal.signal_type == 'BUY'
                       else signal.price * (1 - self.config.take_profit_pct),
            timestamp=datetime.now(),
            strategy=signal.strategy,
            metadata={
                'signal_confidence': signal.confidence,
                'signal_metadata': signal.metadata
            }
        )
        
        # 记录信号
        self.last_signals[signal.symbol] = signal
        
        return order
    
    def should_close_position(self, symbol: str, entry_price: float, 
                            current_price: float, position_qty: float) -> bool:
        """判断是否应该平仓"""
        if position_qty == 0:
            return False
            
        # 计算盈亏百分比
        if position_qty > 0:  # 多头持仓
            pnl_pct = (current_price - entry_price) / entry_price
            # 止损或止盈
            return (pnl_pct <= -self.config.stop_loss_pct or 
                   pnl_pct >= self.config.take_profit_pct)
        else:  # 空头持仓
            pnl_pct = (entry_price - current_price) / entry_price
            # 止损或止盈
            return (pnl_pct <= -self.config.stop_loss_pct or 
                   pnl_pct >= self.config.take_profit_pct)
    
    def get_strategy_status(self) -> Dict:
        """获取策略状态"""
        return {
            'strategy_type': self.config.strategy_type,
            'tracked_symbols': list(self.price_history.keys()),
            'last_signals': {k: v.signal_type for k, v in self.last_signals.items()},
            'config': {
                'lookback_period': self.config.lookback_period,
                'signal_threshold': self.config.signal_threshold,
                'position_size': self.config.position_size,
                'max_positions': self.config.max_positions
            }
        }


class StrategyManager:
    """策略管理器"""
    
    def __init__(self):
        self.strategies: Dict[str, LiveTradingStrategy] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_strategy(self, name: str, strategy: LiveTradingStrategy):
        """添加策略"""
        self.strategies[name] = strategy
        self.logger.info(f"添加策略: {name} ({strategy.config.strategy_type})")
    
    def update_prices(self, price_data: Dict[str, float]):
        """更新所有策略的价格数据"""
        for symbol, price in price_data.items():
            for strategy in self.strategies.values():
                strategy.update_price(symbol, price)
    
    def generate_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """生成所有策略的交易信号"""
        signals = []
        for strategy in self.strategies.values():
            for symbol in symbols:
                signal = strategy.generate_signal(symbol)
                if signal:
                    signals.append(signal)
        return signals
    
    def create_orders(self, signals: List[TradingSignal], account_value: float,
                     current_positions: Dict[str, float]) -> List[TradeOrder]:
        """根据信号创建订单"""
        orders = []
        for signal in signals:
            # 找到生成该信号的策略
            for strategy in self.strategies.values():
                if signal.strategy == strategy.config.strategy_type:
                    order = strategy.create_order(signal, account_value, current_positions)
                    if order:
                        orders.append(order)
                    break
        return orders
    
    def get_all_status(self) -> Dict:
        """获取所有策略状态"""
        return {name: strategy.get_strategy_status() 
                for name, strategy in self.strategies.items()}