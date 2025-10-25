#!/usr/bin/env python3
"""
真实数据收集器 - 连接主交易系统获取真实交易数据
替换监控面板中的模拟数据生成器
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.ib_trading_manager import IBTradingManager
from src.data.data_manager import DataManager

logger = logging.getLogger(__name__)

@dataclass
class RealSystemMetrics:
    """真实系统指标"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    error_count: int
    warning_count: int
    timestamp: str

@dataclass
class RealTradingMetrics:
    """真实交易指标"""
    total_trades: int
    successful_trades: int
    win_rate: float
    total_pnl: float
    portfolio_value: float
    active_positions: int
    timestamp: str

@dataclass
class RealMarketData:
    """真实市场数据"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str

@dataclass
class RealPositionInfo:
    """真实持仓信息"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    timestamp: str

class RealDataCollector:
    """真实数据收集器 - 连接主交易系统获取真实数据"""
    
    def __init__(self, ib_trading_manager: Optional[IBTradingManager] = None):
        self.ib_manager = ib_trading_manager
        self.data_manager = DataManager()
        self.is_running = False  # 添加is_running属性
        self.collection_thread = None
        self.data_cache = {
            'trading_metrics': {},
            'positions': [],
            'market_data': [],
            'system_metrics': {}
        }
        self._lock = threading.Lock()
        logger.info("真实数据收集器初始化完成")
    
    def start_collection(self):
        """开始数据收集"""
        if not self.is_running:
            self.is_running = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            logger.info("开始真实数据收集")
    
    def stop_collection(self):
        """停止数据收集"""
        if self.is_running:
            self.is_running = False
            if self.collection_thread:
                self.collection_thread.join(timeout=5)
            logger.info("停止真实数据收集")
    
    def _collection_loop(self):
        """数据收集循环"""
        while self.is_running:
            try:
                self._collect_all_data()
                time.sleep(5)  # 每5秒收集一次数据
            except Exception as e:
                logger.error(f"数据收集出错: {e}")
                time.sleep(10)  # 出错时等待更长时间
    
    def _collect_all_data(self):
        """收集所有数据"""
        try:
            # 更新缓存数据
            self.data_cache['trading_metrics'] = self._get_real_trading_metrics()
            self.data_cache['positions'] = self._get_real_positions()
            self.data_cache['market_data'] = self._get_real_market_data()
            self.data_cache['system_metrics'] = self.get_system_metrics()
        except Exception as e:
            logger.error(f"收集数据时出错: {e}")
    
    def _get_real_trading_metrics(self) -> Dict[str, Any]:
        """获取真实交易指标"""
        try:
            if self.ib_manager and self.ib_manager.is_connected():
                # 从IB交易管理器获取真实数据
                account_summary = self.ib_manager.get_account_summary()
                positions = self.ib_manager.get_positions()
                
                # 计算交易统计
                total_trades = len(getattr(self.ib_manager, 'trade_history', []))
                successful_trades = len([t for t in getattr(self.ib_manager, 'trade_history', []) 
                                       if getattr(t, 'pnl', 0) > 0])
                win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
                
                return {
                    'total_trades': total_trades,
                    'successful_trades': successful_trades,
                    'win_rate': win_rate,
                    'total_pnl': account_summary.get('daily_pnl', 0),
                    'portfolio_value': account_summary.get('net_liquidation', 0),
                    'active_positions': len([p for p in positions.values() if p.quantity != 0]),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # 如果IB未连接，返回默认值
                return {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'portfolio_value': 0.0,
                    'active_positions': 0,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"获取真实交易指标失败: {e}")
            return {
                'total_trades': 0,
                'successful_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'portfolio_value': 0.0,
                'active_positions': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_real_positions(self) -> List[Dict[str, Any]]:
        """获取真实持仓信息"""
        try:
            if self.ib_manager and self.ib_manager.is_connected():
                positions = self.ib_manager.get_positions()
                real_positions = []
                
                for symbol, position in positions.items():
                    if position.quantity != 0:  # 只返回有持仓的股票
                        # 获取当前价格
                        market_data = self.ib_manager.get_market_data(symbol)
                        current_price = market_data.last if market_data else position.avg_cost
                        
                        # 计算市值和盈亏
                        market_value = current_price * abs(position.quantity)
                        cost_basis = position.avg_cost * abs(position.quantity)
                        unrealized_pnl = market_value - cost_basis
                        unrealized_pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
                        
                        real_positions.append({
                            'symbol': symbol,
                            'quantity': int(position.quantity),
                            'avg_cost': float(position.avg_cost),
                            'current_price': float(current_price),
                            'market_value': float(market_value),
                            'unrealized_pnl': float(unrealized_pnl),
                            'unrealized_pnl_percent': float(unrealized_pnl_percent),
                            'timestamp': datetime.now().isoformat()
                        })
                
                return real_positions
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取真实持仓信息失败: {e}")
            return []
    
    def _get_real_market_data(self) -> List[Dict[str, Any]]:
        """获取真实市场数据"""
        try:
            if self.ib_manager and self.ib_manager.is_connected():
                # 获取监控的股票列表
                symbols = getattr(self.ib_manager, 'symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'])
                market_data_list = []
                
                for symbol in symbols:
                    market_data = self.ib_manager.get_market_data(symbol)
                    if market_data:
                        # 计算变化
                        change = market_data.last - market_data.close if market_data.close > 0 else 0
                        change_percent = (change / market_data.close * 100) if market_data.close > 0 else 0
                        
                        market_data_list.append({
                            'symbol': symbol,
                            'price': float(market_data.last),
                            'change': float(change),
                            'change_percent': float(change_percent),
                            'volume': int(market_data.volume) if market_data.volume else 0,
                            'timestamp': datetime.now().isoformat()
                        })
                
                return market_data_list
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取真实市场数据失败: {e}")
            return []
    
    # 公共接口方法
    def get_trading_metrics(self) -> Dict[str, Any]:
        """获取交易指标"""
        return self.data_cache.get('trading_metrics', {})
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        return self.data_cache.get('positions', [])
    
    def get_market_data(self) -> List[Dict[str, Any]]:
        """获取市场数据"""
        return self.data_cache.get('market_data', [])
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标（使用psutil获取真实系统数据）"""
        try:
            import psutil
            
            return {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'active_connections': len(psutil.net_connections()),
                'error_count': 0,  # 可以从日志系统获取
                'warning_count': 0,  # 可以从日志系统获取
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0,
                'active_connections': 0,
                'error_count': 0,
                'warning_count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_order_info(self) -> List[Dict[str, Any]]:
        """获取订单信息"""
        return []  # 返回空列表，避免前端forEach错误
    
    def get_trading_info(self) -> List[Dict[str, Any]]:
        """获取交易信息"""
        return []  # 返回空列表
    
    def get_risk_events(self) -> List[Dict[str, Any]]:
        """获取风险事件"""
        return []  # 返回空列表
    
    def get_market_events(self) -> List[Dict[str, Any]]:
        """获取市场事件"""
        return []  # 返回空列表