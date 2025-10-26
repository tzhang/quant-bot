#!/usr/bin/env python3
"""
增强的交易执行引擎
整合策略信号生成、风险管理、订单执行和监控功能
支持模拟交易和实盘交易的无缝切换
"""

import sys
import os
import time
import threading
import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """交易模式"""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

class OrderType(Enum):
    """订单类型"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAILING_STOP = "TRAIL"

class OrderAction(Enum):
    """订单动作"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    action: OrderAction
    quantity: int
    signal_strength: float
    confidence: str
    strategy_type: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TradeOrder:
    """交易订单"""
    order_id: str
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = None
    parent_signal: Optional[TradingSignal] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()
        self.update_market_value()
    
    def update_market_value(self):
        """更新市值和未实现盈亏"""
        if self.current_price > 0:
            self.market_value = self.quantity * self.current_price
            self.unrealized_pnl = (self.current_price - self.avg_cost) * self.quantity

@dataclass
class RiskLimits:
    """风险限制"""
    max_position_value: float = 50000.0
    max_daily_loss: float = 5000.0
    max_symbol_exposure: float = 20000.0
    max_daily_trades: int = 100
    max_portfolio_risk: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_leverage: float = 2.0

class EnhancedTradingEngine:
    """增强的交易执行引擎"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 trading_mode: TradingMode = TradingMode.LIVE,
                 risk_limits: Optional[RiskLimits] = None):
        """
        初始化交易引擎
        
        Args:
            initial_capital: 初始资金
            trading_mode: 交易模式
            risk_limits: 风险限制
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trading_mode = trading_mode
        self.risk_limits = risk_limits or RiskLimits()
        
        # 交易状态
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, TradeOrder] = {}
        self.pending_orders: Dict[str, TradeOrder] = {}
        self.filled_orders: Dict[str, TradeOrder] = {}
        
        # 交易历史
        self.trade_history: List[TradeOrder] = []
        self.pnl_history: List[Dict] = []
        
        # 风险管理
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
        # 市场数据
        self.market_data: Dict[str, Dict] = {}
        self.price_history: Dict[str, List] = defaultdict(list)
        
        # 策略信号队列
        self.signal_queue = queue.Queue()
        self.execution_queue = queue.Queue()
        
        # 线程控制
        self.running = False
        self.threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        
        # 回调函数
        self.callbacks = {
            'on_order_filled': [],
            'on_position_update': [],
            'on_risk_alert': [],
            'on_pnl_update': []
        }
        
        # 性能统计
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        logger.info(f"交易引擎初始化完成 - 模式: {trading_mode.value}, 初始资金: ${initial_capital:,.2f}")
    
    def start(self):
        """启动交易引擎"""
        if self.running:
            logger.warning("交易引擎已在运行中")
            return
        
        self.running = True
        
        # 启动信号处理线程
        signal_thread = threading.Thread(target=self._signal_processor, daemon=True)
        signal_thread.start()
        self.threads.append(signal_thread)
        
        # 启动订单执行线程
        execution_thread = threading.Thread(target=self._order_executor, daemon=True)
        execution_thread.start()
        self.threads.append(execution_thread)
        
        # 启动风险监控线程
        risk_thread = threading.Thread(target=self._risk_monitor, daemon=True)
        risk_thread.start()
        self.threads.append(risk_thread)
        
        logger.info("交易引擎启动成功")
    
    def stop(self):
        """停止交易引擎"""
        self.running = False
        
        # 等待所有线程结束
        for thread in self.threads:
            thread.join(timeout=5)
        
        self.threads.clear()
        logger.info("交易引擎已停止")
    
    def add_signal(self, signal: TradingSignal):
        """添加交易信号"""
        try:
            self.signal_queue.put(signal, timeout=1)
            logger.debug(f"添加交易信号: {signal.symbol} {signal.action.value} {signal.quantity}")
        except queue.Full:
            logger.warning("信号队列已满，丢弃信号")
    
    def _signal_processor(self):
        """信号处理线程"""
        while self.running:
            try:
                signal = self.signal_queue.get(timeout=1)
                
                # 风险检查
                if self._validate_signal(signal):
                    # 创建订单
                    order = self._create_order_from_signal(signal)
                    if order:
                        self.execution_queue.put(order)
                        logger.info(f"信号转换为订单: {order.symbol} {order.action.value} {order.quantity}")
                else:
                    logger.warning(f"信号未通过风险检查: {signal.symbol}")
                
                self.signal_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"信号处理错误: {e}")
    
    def _order_executor(self):
        """订单执行线程"""
        while self.running:
            try:
                order = self.execution_queue.get(timeout=1)
                
                # 执行订单
                success = self._execute_order(order)
                if success:
                    logger.info(f"订单执行成功: {order.order_id}")
                else:
                    logger.error(f"订单执行失败: {order.order_id}")
                
                self.execution_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"订单执行错误: {e}")
    
    def _risk_monitor(self):
        """风险监控线程"""
        while self.running:
            try:
                # 检查风险指标
                self._check_risk_limits()
                
                # 更新统计数据
                self._update_statistics()
                
                # 检查止损止盈
                self._check_stop_orders()
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"风险监控错误: {e}")
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """验证交易信号"""
        with self._lock:
            # 检查日交易次数限制
            if self.daily_trades >= self.risk_limits.max_daily_trades:
                logger.warning("已达到日交易次数限制")
                return False
            
            # 检查日损失限制
            if self.daily_pnl < -self.risk_limits.max_daily_loss:
                logger.warning("已达到日损失限制")
                return False
            
            # 检查单个标的暴露限制
            current_price = self._get_current_price(signal.symbol)
            if current_price is None:
                logger.warning(f"无法获取 {signal.symbol} 的当前价格")
                return False
            
            order_value = signal.quantity * current_price
            if order_value > self.risk_limits.max_symbol_exposure:
                logger.warning(f"订单价值超过单标的暴露限制: ${order_value:,.2f}")
                return False
            
            # 检查资金充足性
            if signal.action == OrderAction.BUY:
                if order_value > self.current_capital:
                    logger.warning(f"资金不足: 需要 ${order_value:,.2f}, 可用 ${self.current_capital:,.2f}")
                    return False
            
            return True
    
    def _create_order_from_signal(self, signal: TradingSignal) -> Optional[TradeOrder]:
        """从信号创建订单"""
        try:
            order_id = f"ORD_{int(time.time() * 1000)}"
            
            # 确定订单类型和价格
            order_type = OrderType.MARKET
            price = None
            
            if signal.target_price is not None:
                order_type = OrderType.LIMIT
                price = signal.target_price
            
            order = TradeOrder(
                order_id=order_id,
                symbol=signal.symbol,
                action=signal.action,
                quantity=signal.quantity,
                order_type=order_type,
                price=price,
                parent_signal=signal
            )
            
            # 添加止损止盈订单
            if signal.stop_loss is not None:
                order.stop_price = signal.stop_loss
            
            return order
            
        except Exception as e:
            logger.error(f"创建订单失败: {e}")
            return None
    
    def _execute_order(self, order: TradeOrder) -> bool:
        """执行订单"""
        try:
            with self._lock:
                # 获取当前价格
                current_price = self._get_current_price(order.symbol)
                if current_price is None:
                    order.status = OrderStatus.REJECTED
                    return False
                
                # 模拟订单执行
                if self.trading_mode == TradingMode.PAPER or self.trading_mode == TradingMode.BACKTEST:
                    fill_price = current_price
                    
                    # 限价单检查
                    if order.order_type == OrderType.LIMIT:
                        if order.action == OrderAction.BUY and current_price > order.price:
                            order.status = OrderStatus.PENDING
                            self.pending_orders[order.order_id] = order
                            return True
                        elif order.action == OrderAction.SELL and current_price < order.price:
                            order.status = OrderStatus.PENDING
                            self.pending_orders[order.order_id] = order
                            return True
                        fill_price = order.price
                    
                    # 执行成交
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.avg_fill_price = fill_price
                    order.commission = self._calculate_commission(order)
                    
                    # 更新持仓
                    self._update_position(order)
                    
                    # 更新资金
                    if order.action == OrderAction.BUY:
                        self.current_capital -= (order.filled_quantity * fill_price + order.commission)
                    else:
                        self.current_capital += (order.filled_quantity * fill_price - order.commission)
                    
                    # 记录交易
                    self.orders[order.order_id] = order
                    self.filled_orders[order.order_id] = order
                    self.trade_history.append(order)
                    self.daily_trades += 1
                    
                    # 触发回调
                    self._trigger_callbacks('on_order_filled', order)
                    
                    logger.info(f"订单成交: {order.symbol} {order.action.value} {order.filled_quantity}@${fill_price:.2f}")
                    return True
                
                else:
                    # 实盘交易 - 这里需要集成实际的交易API
                    logger.warning("实盘交易功能待实现")
                    return False
                    
        except Exception as e:
            logger.error(f"订单执行错误: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    def _update_position(self, order: TradeOrder):
        """更新持仓"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_cost=0.0
            )
        
        position = self.positions[symbol]
        
        if order.action == OrderAction.BUY:
            # 买入
            total_cost = position.quantity * position.avg_cost + order.filled_quantity * order.avg_fill_price
            total_quantity = position.quantity + order.filled_quantity
            
            if total_quantity > 0:
                position.avg_cost = total_cost / total_quantity
                position.quantity = total_quantity
        else:
            # 卖出
            position.quantity -= order.filled_quantity
            
            # 计算已实现盈亏
            realized_pnl = (order.avg_fill_price - position.avg_cost) * order.filled_quantity
            position.realized_pnl += realized_pnl
            self.daily_pnl += realized_pnl
        
        # 更新当前价格和市值
        position.current_price = order.avg_fill_price
        position.update_market_value()
        position.last_update = datetime.now()
        
        # 如果持仓为0，移除记录
        if position.quantity == 0:
            del self.positions[symbol]
        
        # 触发回调
        self._trigger_callbacks('on_position_update', position)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 这里应该从市场数据源获取实时价格
        # 目前使用模拟价格
        if symbol in self.market_data:
            return self.market_data[symbol].get('price', 100.0)
        
        # 生成模拟价格
        base_price = 100.0
        if symbol in self.price_history and self.price_history[symbol]:
            base_price = self.price_history[symbol][-1]
        
        # 添加随机波动
        volatility = 0.02
        price_change = np.random.normal(0, volatility)
        new_price = base_price * (1 + price_change)
        
        # 记录价格历史
        self.price_history[symbol].append(new_price)
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
        
        return new_price
    
    def _calculate_commission(self, order: TradeOrder) -> float:
        """计算佣金"""
        # 简化的佣金计算
        return max(1.0, order.filled_quantity * 0.005)
    
    def _check_risk_limits(self):
        """检查风险限制"""
        with self._lock:
            # 检查最大回撤
            current_equity = self.get_total_equity()
            if current_equity > self.peak_capital:
                self.peak_capital = current_equity
            
            drawdown = (self.peak_capital - current_equity) / self.peak_capital
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
            
            # 检查风险警报
            if drawdown > 0.1:  # 10%回撤警报
                self._trigger_callbacks('on_risk_alert', {
                    'type': 'drawdown',
                    'value': drawdown,
                    'message': f'当前回撤: {drawdown:.2%}'
                })
    
    def _check_stop_orders(self):
        """检查止损止盈订单"""
        with self._lock:
            for symbol, position in list(self.positions.items()):
                current_price = self._get_current_price(symbol)
                if current_price is None:
                    continue
                
                position.current_price = current_price
                position.update_market_value()
                
                # 检查止损
                if position.quantity > 0:  # 多头持仓
                    loss_pct = (position.avg_cost - current_price) / position.avg_cost
                    if loss_pct > self.risk_limits.stop_loss_pct:
                        # 触发止损
                        self._create_stop_loss_order(symbol, position.quantity)
                        logger.warning(f"触发止损: {symbol}, 损失: {loss_pct:.2%}")
                
                # 检查止盈
                profit_pct = (current_price - position.avg_cost) / position.avg_cost
                if profit_pct > self.risk_limits.take_profit_pct:
                    # 触发止盈
                    self._create_take_profit_order(symbol, position.quantity)
                    logger.info(f"触发止盈: {symbol}, 盈利: {profit_pct:.2%}")
    
    def _create_stop_loss_order(self, symbol: str, quantity: int):
        """创建止损订单"""
        signal = TradingSignal(
            symbol=symbol,
            action=OrderAction.SELL,
            quantity=quantity,
            signal_strength=1.0,
            confidence="HIGH",
            strategy_type="STOP_LOSS"
        )
        self.add_signal(signal)
    
    def _create_take_profit_order(self, symbol: str, quantity: int):
        """创建止盈订单"""
        signal = TradingSignal(
            symbol=symbol,
            action=OrderAction.SELL,
            quantity=quantity,
            signal_strength=1.0,
            confidence="HIGH",
            strategy_type="TAKE_PROFIT"
        )
        self.add_signal(signal)
    
    def _update_statistics(self):
        """更新统计数据"""
        if not self.trade_history:
            return
        
        with self._lock:
            total_trades = len(self.trade_history)
            winning_trades = 0
            losing_trades = 0
            total_pnl = 0.0
            
            for order in self.trade_history:
                if order.action == OrderAction.SELL and order.symbol in self.positions:
                    # 计算该笔交易的盈亏
                    position = self.positions[order.symbol]
                    pnl = (order.avg_fill_price - position.avg_cost) * order.filled_quantity
                    total_pnl += pnl
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
            
            self.stats.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'win_rate': winning_trades / max(total_trades, 1),
                'max_drawdown': self.max_drawdown
            })
    
    def _trigger_callbacks(self, event_type: str, data: Any):
        """触发回调函数"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"回调函数执行错误: {e}")
    
    def add_callback(self, event_type: str, callback: callable):
        """添加回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def get_total_equity(self) -> float:
        """获取总权益"""
        total_position_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_capital + total_position_value
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        total_equity = self.get_total_equity()
        total_pnl = total_equity - self.initial_capital
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_equity': total_equity,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl / self.initial_capital,
            'positions_count': len(self.positions),
            'daily_trades': self.daily_trades,
            'max_drawdown': self.max_drawdown,
            'positions': {symbol: asdict(pos) for symbol, pos in self.positions.items()},
            'stats': self.stats
        }
    
    def update_market_data(self, symbol: str, price: float, volume: int = 0):
        """更新市场数据"""
        self.market_data[symbol] = {
            'price': price,
            'volume': volume,
            'timestamp': datetime.now()
        }
        
        # 更新持仓市值
        if symbol in self.positions:
            self.positions[symbol].current_price = price
            self.positions[symbol].update_market_value()

def main():
    """示例用法"""
    # 创建交易引擎
    engine = EnhancedTradingEngine(
        initial_capital=100000.0,
        trading_mode=TradingMode.PAPER
    )
    
    # 添加回调函数
    def on_order_filled(order):
        print(f"订单成交: {order.symbol} {order.action.value} {order.filled_quantity}@${order.avg_fill_price:.2f}")
    
    def on_position_update(position):
        print(f"持仓更新: {position.symbol} {position.quantity} 股, 未实现盈亏: ${position.unrealized_pnl:.2f}")
    
    engine.add_callback('on_order_filled', on_order_filled)
    engine.add_callback('on_position_update', on_position_update)
    
    # 启动引擎
    engine.start()
    
    try:
        # 模拟交易信号
        signals = [
            TradingSignal("AAPL", OrderAction.BUY, 100, 0.8, "HIGH", "MOMENTUM"),
            TradingSignal("GOOGL", OrderAction.BUY, 50, 0.7, "MEDIUM", "MEAN_REVERSION"),
            TradingSignal("MSFT", OrderAction.BUY, 75, 0.9, "HIGH", "BREAKOUT")
        ]
        
        # 添加信号
        for signal in signals:
            engine.add_signal(signal)
            time.sleep(1)
        
        # 运行一段时间
        time.sleep(10)
        
        # 获取投资组合摘要
        summary = engine.get_portfolio_summary()
        print("\n投资组合摘要:")
        print(f"总权益: ${summary['total_equity']:,.2f}")
        print(f"总盈亏: ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:.2%})")
        print(f"持仓数量: {summary['positions_count']}")
        print(f"日交易次数: {summary['daily_trades']}")
        
    finally:
        # 停止引擎
        engine.stop()

if __name__ == "__main__":
    main()