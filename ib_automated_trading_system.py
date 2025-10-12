#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Interactive Brokers API的程序化交易系统
替代Firstrade，使用IB API进行自动化交易

作者: AI Assistant
日期: 2025年1月
版本: v3.1.0
"""

import sys
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import json

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'examples'))

# 导入IB相关模块
try:
    from src.trading.ib_trading_manager import IBTradingManager, TradingSignal, TradeOrder, RiskLimits
    from examples.enhanced_ib_trading_system import EnhancedIBTradingSystem, TradingConfig, TradingMode
    from examples.ib_risk_manager import IBRiskManager, RiskLimit, RiskLevel
    from examples.ib_order_manager import IBOrderManager, OrderRequest, OrderType
    IB_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"IB模块导入失败: {e}")
    IB_MODULES_AVAILABLE = False

# 导入策略模块
try:
    from src.strategies import BaseStrategy, StrategyManager
    from src.strategies.live_strategy import LiveTradingStrategy
    STRATEGY_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"策略模块导入失败: {e}")
    STRATEGY_MODULES_AVAILABLE = False

# 导入数据模块
try:
    from src.data.ib_data_provider import IBDataProvider
    from src.data.stock_data_dao import StockDataDAO
    DATA_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"数据模块导入失败: {e}")
    DATA_MODULES_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ib_automated_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemStatus(Enum):
    """交易系统状态"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"

@dataclass
class SystemConfig:
    """系统配置"""
    # IB连接配置
    ib_host: str = "127.0.0.1"
    ib_paper_port: int = 7497
    ib_live_port: int = 7496
    ib_client_id: int = 1
    paper_trading: bool = True
    
    # 交易配置
    initial_capital: float = 100000.0
    max_position_value: float = 50000.0
    max_daily_loss: float = 5000.0
    max_symbol_exposure: float = 20000.0
    max_daily_trades: int = 100
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    
    # 策略配置
    strategy_types: List[str] = None
    rebalance_frequency: str = "1H"  # 1小时重新平衡
    signal_threshold: float = 0.6
    
    # 监控配置
    monitoring_interval: int = 30  # 秒
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.strategy_types is None:
            self.strategy_types = ["momentum", "mean_reversion", "rsi"]

class IBAutomatedTradingSystem:
    """基于IB API的程序化交易系统"""
    
    def __init__(self, config: SystemConfig = None):
        """初始化交易系统"""
        self.config = config or SystemConfig()
        self.status = TradingSystemStatus.STOPPED
        
        # 核心组件
        self.ib_trading_manager = None
        self.data_provider = None
        self.strategy_manager = None
        self.risk_manager = None
        
        # 交易状态
        self.active_positions: Dict[str, Any] = {}
        self.active_orders: Dict[int, Any] = {}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trades = 0
        
        # 策略和信号
        self.active_strategies: List[BaseStrategy] = []
        self.latest_signals: Dict[str, TradingSignal] = {}
        
        # 线程控制
        self.trading_thread = None
        self.monitoring_thread = None
        self.running = False
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'on_trade_executed': [],
            'on_position_update': [],
            'on_risk_alert': [],
            'on_system_error': [],
            'on_status_change': []
        }
        
        # 性能统计
        self.stats = {
            'start_time': None,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        logger.info("IB程序化交易系统初始化完成")
    
    def initialize(self) -> bool:
        """初始化系统组件"""
        try:
            logger.info("正在初始化IB程序化交易系统...")
            
            # 检查模块可用性
            if not IB_MODULES_AVAILABLE:
                logger.error("IB模块不可用，无法启动交易系统")
                return False
            
            # 初始化IB交易管理器
            ib_config = {
                'host': self.config.ib_host,
                'paper_port': self.config.ib_paper_port,
                'live_port': self.config.ib_live_port,
                'client_id': self.config.ib_client_id,
                'paper_trading': self.config.paper_trading,
                'risk_limits': {
                    'max_position_value': self.config.max_position_value,
                    'max_daily_loss': self.config.max_daily_loss,
                    'max_symbol_exposure': self.config.max_symbol_exposure,
                    'max_daily_trades': self.config.max_daily_trades,
                    'stop_loss_pct': self.config.stop_loss_pct,
                    'take_profit_pct': self.config.take_profit_pct
                }
            }
            
            self.ib_trading_manager = IBTradingManager(ib_config)
            
            # 初始化数据提供者
            if DATA_MODULES_AVAILABLE:
                self.data_provider = IBDataProvider()
                logger.info("✅ IB数据提供者初始化成功")
            
            # 初始化策略管理器
            if STRATEGY_MODULES_AVAILABLE:
                self.strategy_manager = StrategyManager()
                self._load_strategies()
                logger.info("✅ 策略管理器初始化成功")
            
            # 设置回调函数
            self._setup_callbacks()
            
            logger.info("✅ 系统组件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            self.status = TradingSystemStatus.ERROR
            return False
    
    def _load_strategies(self):
        """加载交易策略"""
        try:
            for strategy_type in self.config.strategy_types:
                if strategy_type == "momentum":
                    strategy = self._create_momentum_strategy()
                elif strategy_type == "mean_reversion":
                    strategy = self._create_mean_reversion_strategy()
                elif strategy_type == "rsi":
                    strategy = self._create_rsi_strategy()
                else:
                    logger.warning(f"未知策略类型: {strategy_type}")
                    continue
                
                if strategy:
                    self.active_strategies.append(strategy)
                    logger.info(f"✅ 加载策略: {strategy_type}")
            
            logger.info(f"总共加载了 {len(self.active_strategies)} 个策略")
            
        except Exception as e:
            logger.error(f"策略加载失败: {e}")
    
    def _create_momentum_strategy(self) -> Optional[BaseStrategy]:
        """创建动量策略"""
        try:
            # 这里可以根据实际的策略类来创建
            # 暂时返回None，实际实现时需要根据具体的策略类来创建
            return None
        except Exception as e:
            logger.error(f"创建动量策略失败: {e}")
            return None
    
    def _create_mean_reversion_strategy(self) -> Optional[BaseStrategy]:
        """创建均值回归策略"""
        try:
            # 这里可以根据实际的策略类来创建
            return None
        except Exception as e:
            logger.error(f"创建均值回归策略失败: {e}")
            return None
    
    def _create_rsi_strategy(self) -> Optional[BaseStrategy]:
        """创建RSI策略"""
        try:
            # 这里可以根据实际的策略类来创建
            return None
        except Exception as e:
            logger.error(f"创建RSI策略失败: {e}")
            return None
    
    def _setup_callbacks(self):
        """设置回调函数"""
        if self.ib_trading_manager:
            # 设置交易管理器回调
            self.ib_trading_manager.add_callback('order_filled', self._on_order_filled)
            self.ib_trading_manager.add_callback('position_update', self._on_position_update)
            self.ib_trading_manager.add_callback('risk_alert', self._on_risk_alert)
            self.ib_trading_manager.add_callback('connection_status', self._on_connection_status)
    
    def start(self) -> bool:
        """启动程序化交易系统"""
        try:
            logger.info("🚀 启动IB程序化交易系统...")
            
            if self.status == TradingSystemStatus.RUNNING:
                logger.warning("系统已经在运行中")
                return True
            
            self.status = TradingSystemStatus.STARTING
            self._notify_status_change()
            
            # 连接到IB
            if not self.ib_trading_manager.connect():
                logger.error("❌ 无法连接到IB")
                self.status = TradingSystemStatus.ERROR
                return False
            
            # 启动交易线程
            self.running = True
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            
            self.trading_thread.start()
            self.monitoring_thread.start()
            
            self.status = TradingSystemStatus.RUNNING
            self.stats['start_time'] = datetime.now()
            self._notify_status_change()
            
            logger.info("✅ IB程序化交易系统启动成功")
            return True
            
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            self.status = TradingSystemStatus.ERROR
            return False
    
    def stop(self):
        """停止程序化交易系统"""
        try:
            logger.info("🛑 停止IB程序化交易系统...")
            
            self.running = False
            self.status = TradingSystemStatus.STOPPED
            
            # 等待线程结束
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5)
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # 断开IB连接
            if self.ib_trading_manager:
                self.ib_trading_manager.disconnect()
            
            self._notify_status_change()
            logger.info("✅ IB程序化交易系统已停止")
            
        except Exception as e:
            logger.error(f"系统停止时发生错误: {e}")
    
    def _trading_loop(self):
        """主交易循环"""
        logger.info("交易循环已启动")
        
        while self.running:
            try:
                # 生成交易信号
                self._generate_signals()
                
                # 执行交易决策
                self._execute_trading_decisions()
                
                # 更新持仓和风险
                self._update_positions_and_risk()
                
                # 等待下一个周期
                time.sleep(60)  # 1分钟周期
                
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                time.sleep(30)  # 错误后等待30秒
        
        logger.info("交易循环已结束")
    
    def _monitoring_loop(self):
        """监控循环"""
        logger.info("监控循环已启动")
        
        while self.running:
            try:
                # 监控系统状态
                self._monitor_system_health()
                
                # 监控风险指标
                self._monitor_risk_metrics()
                
                # 更新性能统计
                self._update_performance_stats()
                
                # 等待下一个监控周期
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(10)
        
        logger.info("监控循环已结束")
    
    def _generate_signals(self):
        """生成交易信号"""
        try:
            if not self.active_strategies:
                return
            
            # 获取市场数据
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]  # 示例股票
            
            for symbol in symbols:
                # 获取最新数据
                if self.data_provider:
                    data = self.data_provider.get_stock_data(symbol, period='1d')
                    if data is not None and not data.empty:
                        # 为每个策略生成信号
                        for strategy in self.active_strategies:
                            try:
                                signal = strategy.generate_signal(data)
                                if signal and signal != TradingSignal.HOLD:
                                    self.latest_signals[f"{symbol}_{strategy.__class__.__name__}"] = signal
                            except Exception as e:
                                logger.error(f"策略 {strategy.__class__.__name__} 生成信号失败: {e}")
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
    
    def _execute_trading_decisions(self):
        """执行交易决策"""
        try:
            for signal_key, signal in self.latest_signals.items():
                symbol = signal_key.split('_')[0]
                
                # 检查是否已有该股票的持仓
                current_position = self.active_positions.get(symbol, 0)
                
                # 根据信号执行交易
                if signal == TradingSignal.BUY and current_position <= 0:
                    self._place_buy_order(symbol)
                elif signal == TradingSignal.SELL and current_position > 0:
                    self._place_sell_order(symbol)
            
            # 清空已处理的信号
            self.latest_signals.clear()
            
        except Exception as e:
            logger.error(f"执行交易决策失败: {e}")
    
    def _place_buy_order(self, symbol: str):
        """下买单"""
        try:
            # 计算订单数量（基于风险管理）
            quantity = self._calculate_order_quantity(symbol, "BUY")
            
            if quantity > 0:
                order = TradeOrder(
                    symbol=symbol,
                    action="BUY",
                    quantity=quantity,
                    order_type="MKT"
                )
                
                order_id = self.ib_trading_manager.place_order(order)
                if order_id:
                    self.active_orders[order_id] = order
                    logger.info(f"✅ 下买单成功: {symbol} x {quantity}")
                    self.daily_trades += 1
                    self.stats['total_trades'] += 1
                
        except Exception as e:
            logger.error(f"下买单失败 {symbol}: {e}")
            self.stats['failed_trades'] += 1
    
    def _place_sell_order(self, symbol: str):
        """下卖单"""
        try:
            # 获取当前持仓数量
            current_position = self.active_positions.get(symbol, 0)
            
            if current_position > 0:
                order = TradeOrder(
                    symbol=symbol,
                    action="SELL",
                    quantity=current_position,
                    order_type="MKT"
                )
                
                order_id = self.ib_trading_manager.place_order(order)
                if order_id:
                    self.active_orders[order_id] = order
                    logger.info(f"✅ 下卖单成功: {symbol} x {current_position}")
                    self.daily_trades += 1
                    self.stats['total_trades'] += 1
                
        except Exception as e:
            logger.error(f"下卖单失败 {symbol}: {e}")
            self.stats['failed_trades'] += 1
    
    def _calculate_order_quantity(self, symbol: str, action: str) -> int:
        """计算订单数量"""
        try:
            # 基于风险管理计算订单数量
            max_position_value = self.config.max_symbol_exposure
            
            # 获取当前价格（模拟）
            current_price = 150.0  # 这里应该从市场数据获取实际价格
            
            # 计算最大可买数量
            max_quantity = int(max_position_value / current_price)
            
            # 应用额外的风险控制
            return min(max_quantity, 100)  # 最多100股
            
        except Exception as e:
            logger.error(f"计算订单数量失败: {e}")
            return 0
    
    def _update_positions_and_risk(self):
        """更新持仓和风险"""
        try:
            if self.ib_trading_manager:
                # 获取最新持仓
                positions = self.ib_trading_manager.get_positions()
                if positions:
                    self.active_positions = positions
                
                # 获取账户信息
                account_info = self.ib_trading_manager.get_account_info()
                if account_info:
                    self.daily_pnl = account_info.get('daily_pnl', 0.0)
                    self.total_pnl = account_info.get('total_pnl', 0.0)
            
        except Exception as e:
            logger.error(f"更新持仓和风险失败: {e}")
    
    def _monitor_system_health(self):
        """监控系统健康状态"""
        try:
            # 检查IB连接状态
            if self.ib_trading_manager and not self.ib_trading_manager.is_connected():
                logger.warning("⚠️ IB连接断开，尝试重连...")
                self.ib_trading_manager.reconnect()
            
            # 检查内存使用
            # 检查线程状态
            # 其他健康检查...
            
        except Exception as e:
            logger.error(f"系统健康监控失败: {e}")
    
    def _monitor_risk_metrics(self):
        """监控风险指标"""
        try:
            # 检查日内亏损
            if abs(self.daily_pnl) > self.config.max_daily_loss:
                logger.warning(f"⚠️ 日内亏损超限: {self.daily_pnl}")
                self._trigger_risk_alert("daily_loss_exceeded")
            
            # 检查交易次数
            if self.daily_trades > self.config.max_daily_trades:
                logger.warning(f"⚠️ 日内交易次数超限: {self.daily_trades}")
                self._trigger_risk_alert("daily_trades_exceeded")
            
        except Exception as e:
            logger.error(f"风险监控失败: {e}")
    
    def _update_performance_stats(self):
        """更新性能统计"""
        try:
            if self.stats['total_trades'] > 0:
                self.stats['win_rate'] = self.stats['successful_trades'] / self.stats['total_trades']
            
            # 计算其他性能指标...
            
        except Exception as e:
            logger.error(f"性能统计更新失败: {e}")
    
    def _trigger_risk_alert(self, alert_type: str):
        """触发风险警报"""
        logger.warning(f"🚨 风险警报: {alert_type}")
        
        # 执行风险控制措施
        if alert_type in ["daily_loss_exceeded", "daily_trades_exceeded"]:
            logger.info("暂停交易以控制风险")
            self.status = TradingSystemStatus.PAUSED
    
    # 回调函数
    def _on_order_filled(self, order_info):
        """订单成交回调"""
        logger.info(f"📈 订单成交: {order_info}")
        self.stats['successful_trades'] += 1
        
        # 通知回调
        for callback in self.callbacks['on_trade_executed']:
            try:
                callback(order_info)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
    
    def _on_position_update(self, position_info):
        """持仓更新回调"""
        logger.info(f"📊 持仓更新: {position_info}")
        
        # 通知回调
        for callback in self.callbacks['on_position_update']:
            try:
                callback(position_info)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
    
    def _on_risk_alert(self, alert_info):
        """风险警报回调"""
        logger.warning(f"🚨 风险警报: {alert_info}")
        
        # 通知回调
        for callback in self.callbacks['on_risk_alert']:
            try:
                callback(alert_info)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
    
    def _on_connection_status(self, status):
        """连接状态回调"""
        logger.info(f"🔗 连接状态: {status}")
    
    def _notify_status_change(self):
        """通知状态变化"""
        for callback in self.callbacks['on_status_change']:
            try:
                callback(self.status)
            except Exception as e:
                logger.error(f"状态变化回调失败: {e}")
    
    # 公共接口
    def add_callback(self, event_type: str, callback: Callable):
        """添加回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def get_status(self) -> TradingSystemStatus:
        """获取系统状态"""
        return self.status
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def get_positions(self) -> Dict:
        """获取当前持仓"""
        return self.active_positions.copy()
    
    def get_daily_pnl(self) -> float:
        """获取日内盈亏"""
        return self.daily_pnl

def run_automated_trading():
    """运行程序化交易系统"""
    logger.info("🚀 启动基于IB API的程序化交易系统")
    
    # 创建系统配置
    config = SystemConfig(
        paper_trading=True,  # 使用模拟交易
        initial_capital=100000.0,
        max_position_value=50000.0,
        max_daily_loss=5000.0,
        strategy_types=["momentum", "mean_reversion", "rsi"]
    )
    
    # 创建交易系统
    trading_system = IBAutomatedTradingSystem(config)
    
    try:
        # 初始化系统
        if not trading_system.initialize():
            logger.error("❌ 系统初始化失败")
            return False
        
        # 启动系统
        if not trading_system.start():
            logger.error("❌ 系统启动失败")
            return False
        
        logger.info("✅ IB程序化交易系统运行中...")
        logger.info("按 Ctrl+C 停止系统")
        
        # 保持运行
        try:
            while trading_system.get_status() == TradingSystemStatus.RUNNING:
                time.sleep(10)
                
                # 打印状态信息
                stats = trading_system.get_stats()
                positions = trading_system.get_positions()
                daily_pnl = trading_system.get_daily_pnl()
                
                logger.info(f"📊 状态报告 - 交易次数: {stats['total_trades']}, "
                          f"持仓数: {len(positions)}, 日内盈亏: ${daily_pnl:.2f}")
                
        except KeyboardInterrupt:
            logger.info("收到停止信号...")
        
        # 停止系统
        trading_system.stop()
        logger.info("✅ IB程序化交易系统已安全停止")
        return True
        
    except Exception as e:
        logger.error(f"系统运行错误: {e}")
        trading_system.stop()
        return False

if __name__ == "__main__":
    # 运行程序化交易系统
    success = run_automated_trading()
    
    if success:
        logger.info("🎉 程序化交易系统运行完成")
    else:
        logger.error("❌ 程序化交易系统运行失败")
        sys.exit(1)