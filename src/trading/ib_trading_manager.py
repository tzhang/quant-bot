#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Brokers 实时交易管理器
集成IB API到量化交易系统的核心交易模块
"""

import sys
import os
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

# 添加examples目录到路径以导入IB相关模块
examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
sys.path.insert(0, examples_path)

try:
    from enhanced_ib_trading_system import EnhancedIBTradingSystem, TradingConfig, TradingMode
    from ib_adapter import IBAdapter, IBConfig
except ImportError as e:
    logging.warning(f"IB模块导入失败: {e}")
    # 提供备用实现
    class TradingConfig:
        def __init__(self, **kwargs):
            self.trading_mode = kwargs.get('trading_mode', 'paper')
    
    class EnhancedIBTradingSystem:
        def __init__(self, config):
            self.config = config

logger = logging.getLogger(__name__)

class TradingSignal(Enum):
    """交易信号枚举"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradeOrder:
    """交易订单"""
    symbol: str
    signal: TradingSignal
    quantity: int
    price: Optional[float] = None
    order_type: str = "MKT"
    strategy_name: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class RiskLimits:
    """风险限制"""
    max_position_value: float = 50000.0
    max_daily_loss: float = 5000.0
    max_symbol_exposure: float = 20000.0
    max_daily_trades: int = 100
    stop_loss_pct: float = 0.05  # 5%止损
    take_profit_pct: float = 0.10  # 10%止盈

class IBTradingManager:
    """IB实时交易管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化交易管理器"""
        self.config = config or {}
        
        # 交易配置
        self.trading_config = TradingConfig(
            host=self.config.get('host', '127.0.0.1'),
            paper_port=self.config.get('paper_port', 7497),
            live_port=self.config.get('live_port', 7496),
            client_id=self.config.get('client_id', 1),
            trading_mode=TradingMode.PAPER if self.config.get('paper_trading', True) else TradingMode.LIVE
        )
        
        # 风险限制
        self.risk_limits = RiskLimits(**self.config.get('risk_limits', {}))
        
        # IB交易系统
        self.ib_system = None
        self.connected = False
        
        # 交易状态
        self.active_orders: Dict[int, TradeOrder] = {}
        self.positions: Dict[str, Any] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # 实时数据
        self.market_data: Dict[str, Dict] = {}
        self.price_history: Dict[str, List] = {}
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'order_filled': [],
            'position_update': [],
            'risk_alert': [],
            'connection_status': []
        }
        
        # 线程控制
        self.trading_thread = None
        self.monitoring_thread = None
        self.running = False
        
        logger.info("IB交易管理器初始化完成")
    
    def connect(self) -> bool:
        """连接到IB TWS"""
        try:
            logger.info("正在连接到Interactive Brokers...")
            
            self.ib_system = EnhancedIBTradingSystem(self.trading_config)
            
            if self.ib_system.connect_to_ib():
                self.connected = True
                logger.info("✅ IB连接成功")
                
                # 设置回调
                self._setup_callbacks()
                
                # 启动监控线程
                self._start_monitoring()
                
                return True
            else:
                logger.error("❌ IB连接失败")
                return False
                
        except Exception as e:
            logger.error(f"连接IB时发生错误: {e}")
            return False
    
    def disconnect(self):
        """断开IB连接"""
        self.running = False
        
        if self.ib_system:
            self.ib_system.disconnect_from_ib()
            
        self.connected = False
        logger.info("IB连接已断开")
    
    def _setup_callbacks(self):
        """设置IB系统回调"""
        if not self.ib_system:
            return
            
        # 订单状态回调
        self.ib_system.add_callback('order_status', self._on_order_status)
        
        # 持仓更新回调
        self.ib_system.add_callback('position_update', self._on_position_update)
        
        # 市场数据回调
        self.ib_system.add_callback('market_data', self._on_market_data)
    
    def _start_monitoring(self):
        """启动实时监控"""
        self.running = True
        
        # 风险监控线程
        self.monitoring_thread = threading.Thread(target=self._risk_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("实时监控已启动")
    
    def _risk_monitoring_loop(self):
        """风险监控循环"""
        while self.running:
            try:
                self._check_risk_limits()
                self._check_stop_loss_take_profit()
                self._reset_daily_counters()
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"风险监控错误: {e}")
                time.sleep(5)
    
    def place_order(self, order: TradeOrder) -> Optional[int]:
        """下单"""
        if not self.connected or not self.ib_system:
            logger.error("IB未连接，无法下单")
            return None
        
        # 风险检查
        if not self._pre_trade_risk_check(order):
            logger.warning(f"风险检查未通过，拒绝订单: {order.symbol}")
            return None
        
        try:
            # 转换信号为IB订单参数
            action = "BUY" if order.signal == TradingSignal.BUY else "SELL"
            
            # 下单
            order_id = self.ib_system.place_order(
                symbol=order.symbol,
                action=action,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.price
            )
            
            if order_id:
                self.active_orders[order_id] = order
                self.daily_trades += 1
                logger.info(f"订单已提交: {order.symbol} {action} {order.quantity} (ID: {order_id})")
                return order_id
            else:
                logger.error(f"订单提交失败: {order.symbol}")
                return None
                
        except Exception as e:
            logger.error(f"下单时发生错误: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """取消订单"""
        if not self.connected or not self.ib_system:
            return False
        
        try:
            success = self.ib_system.cancel_order(order_id)
            if success and order_id in self.active_orders:
                del self.active_orders[order_id]
                logger.info(f"订单已取消: {order_id}")
            return success
            
        except Exception as e:
            logger.error(f"取消订单时发生错误: {e}")
            return False
    
    def subscribe_market_data(self, symbols: List[str]) -> bool:
        """订阅市场数据"""
        if not self.connected or not self.ib_system:
            return False
        
        try:
            for symbol in symbols:
                self.ib_system.subscribe_market_data(symbol)
                self.market_data[symbol] = {}
                self.price_history[symbol] = []
                
            logger.info(f"已订阅市场数据: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"订阅市场数据时发生错误: {e}")
            return False
    
    def get_positions(self) -> Dict[str, Any]:
        """获取当前持仓"""
        if self.ib_system:
            return self.ib_system.get_positions()
        return {}
    
    def get_account_summary(self) -> Dict[str, Any]:
        """获取账户摘要"""
        if self.ib_system:
            return self.ib_system.get_account_summary()
        return {}
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """获取市场数据"""
        return self.market_data.get(symbol)
    
    def _pre_trade_risk_check(self, order: TradeOrder) -> bool:
        """交易前风险检查"""
        try:
            # 检查日交易次数
            if self.daily_trades >= self.risk_limits.max_daily_trades:
                logger.warning("已达到日交易次数限制")
                return False
            
            # 检查日损失
            if self.daily_pnl <= -self.risk_limits.max_daily_loss:
                logger.warning("已达到日损失限制")
                return False
            
            # 检查单个标的暴露
            current_exposure = self._get_symbol_exposure(order.symbol)
            if current_exposure >= self.risk_limits.max_symbol_exposure:
                logger.warning(f"标的 {order.symbol} 暴露过大")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"风险检查时发生错误: {e}")
            return False
    
    def _check_risk_limits(self):
        """检查风险限制"""
        try:
            # 更新持仓信息
            self.positions = self.get_positions()
            
            # 检查总持仓价值
            total_position_value = sum(
                abs(pos.market_value) for pos in self.positions.values()
            )
            
            if total_position_value > self.risk_limits.max_position_value:
                self._trigger_risk_alert("总持仓价值超限", {
                    'current': total_position_value,
                    'limit': self.risk_limits.max_position_value
                })
            
            # 检查日损失
            account_summary = self.get_account_summary()
            if account_summary:
                current_pnl = account_summary.get('unrealized_pnl', 0)
                if current_pnl <= -self.risk_limits.max_daily_loss:
                    self._trigger_risk_alert("日损失超限", {
                        'current_pnl': current_pnl,
                        'limit': -self.risk_limits.max_daily_loss
                    })
                    
        except Exception as e:
            logger.error(f"风险检查时发生错误: {e}")
    
    def _check_stop_loss_take_profit(self):
        """检查止损止盈"""
        try:
            for symbol, position in self.positions.items():
                if position.quantity == 0:
                    continue
                
                current_price = self.market_data.get(symbol, {}).get('last', 0)
                if current_price == 0:
                    continue
                
                # 计算盈亏比例
                if position.quantity > 0:  # 多头持仓
                    pnl_pct = (current_price - position.avg_cost) / position.avg_cost
                else:  # 空头持仓
                    pnl_pct = (position.avg_cost - current_price) / position.avg_cost
                
                # 止损检查
                if pnl_pct <= -self.risk_limits.stop_loss_pct:
                    self._execute_stop_loss(symbol, position)
                
                # 止盈检查
                elif pnl_pct >= self.risk_limits.take_profit_pct:
                    self._execute_take_profit(symbol, position)
                    
        except Exception as e:
            logger.error(f"止损止盈检查时发生错误: {e}")
    
    def _execute_stop_loss(self, symbol: str, position: Any):
        """执行止损"""
        try:
            action = "SELL" if position.quantity > 0 else "BUY"
            quantity = abs(int(position.quantity))
            
            order_id = self.ib_system.place_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type="MKT"
            )
            
            if order_id:
                logger.warning(f"执行止损: {symbol} {action} {quantity} (订单ID: {order_id})")
                self._trigger_risk_alert("执行止损", {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'order_id': order_id
                })
                
        except Exception as e:
            logger.error(f"执行止损时发生错误: {e}")
    
    def _execute_take_profit(self, symbol: str, position: Any):
        """执行止盈"""
        try:
            action = "SELL" if position.quantity > 0 else "BUY"
            quantity = abs(int(position.quantity))
            
            order_id = self.ib_system.place_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type="MKT"
            )
            
            if order_id:
                logger.info(f"执行止盈: {symbol} {action} {quantity} (订单ID: {order_id})")
                
        except Exception as e:
            logger.error(f"执行止盈时发生错误: {e}")
    
    def _get_symbol_exposure(self, symbol: str) -> float:
        """获取单个标的暴露"""
        position = self.positions.get(symbol)
        if position:
            return abs(position.market_value)
        return 0.0
    
    def _reset_daily_counters(self):
        """重置日计数器"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logger.info("日计数器已重置")
    
    def _trigger_risk_alert(self, alert_type: str, data: Dict):
        """触发风险警报"""
        logger.warning(f"风险警报: {alert_type} - {data}")
        
        for callback in self.callbacks.get('risk_alert', []):
            try:
                callback(alert_type, data)
            except Exception as e:
                logger.error(f"风险警报回调错误: {e}")
    
    def _on_order_status(self, order_id: int, status: str, filled_qty: float, avg_price: float):
        """订单状态回调"""
        logger.info(f"订单状态更新: {order_id} - {status}")
        
        for callback in self.callbacks.get('order_filled', []):
            try:
                callback(order_id, status, filled_qty, avg_price)
            except Exception as e:
                logger.error(f"订单状态回调错误: {e}")
    
    def _on_position_update(self, symbol: str, position: Any):
        """持仓更新回调"""
        self.positions[symbol] = position
        
        for callback in self.callbacks.get('position_update', []):
            try:
                callback(symbol, position)
            except Exception as e:
                logger.error(f"持仓更新回调错误: {e}")
    
    def _on_market_data(self, symbol: str, data: Dict):
        """市场数据回调"""
        self.market_data[symbol] = data
        
        # 保存价格历史
        if 'last' in data and data['last'] > 0:
            self.price_history[symbol].append({
                'timestamp': datetime.now(),
                'price': data['last']
            })
            
            # 只保留最近1000个数据点
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
        
        for callback in self.callbacks.get('market_data', []):
            try:
                callback(symbol, data)
            except Exception as e:
                logger.error(f"市场数据回调错误: {e}")
    
    def add_callback(self, event_type: str, callback: Callable):
        """添加回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """移除回调函数"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def get_trading_status(self) -> Dict[str, Any]:
        """获取交易状态"""
        return {
            'connected': self.connected,
            'trading_mode': self.trading_config.trading_mode.value if hasattr(self.trading_config.trading_mode, 'value') else str(self.trading_config.trading_mode),
            'active_orders': len(self.active_orders),
            'positions': len(self.positions),
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'subscribed_symbols': list(self.market_data.keys())
        }