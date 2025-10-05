#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
券商工厂类 - 统一管理多个券商API
"""

import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod

from config import config
from firstrade_trading_system import FirstradeTradingSystem
from alpaca_trading_system import AlpacaTradingSystem

logger = logging.getLogger(__name__)

class TradingSystemInterface(ABC):
    """交易系统接口"""
    
    @abstractmethod
    def connect(self) -> bool:
        """连接到券商API"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass
    
    @abstractmethod
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """获取当前持仓"""
        pass
    
    @abstractmethod
    def get_performance(self) -> Dict[str, Any]:
        """获取交易表现"""
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, quantity: int, side: str, **kwargs) -> Dict[str, Any]:
        """下单"""
        pass
    
    @abstractmethod
    def get_detailed_positions(self) -> List[Dict[str, Any]]:
        """获取详细持仓信息"""
        pass
    
    @abstractmethod
    def calculate_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        """计算投资组合表现"""
        pass

class FirstradeAdapter(TradingSystemInterface):
    """Firstrade适配器"""
    
    def __init__(self, username: str, password: str, pin: str = "", dry_run: bool = True):
        self.system = FirstradeTradingSystem(username, password, pin, dry_run)
        self.broker_name = "Firstrade"
    
    def connect(self) -> bool:
        return self.system.connect()
    
    def disconnect(self):
        self.system.disconnect()
    
    def is_connected(self) -> bool:
        return self.system.is_connected()
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        return self.system.get_portfolio_status()
    
    def get_positions(self) -> List[Dict[str, Any]]:
        return self.system.get_positions()
    
    def get_performance(self) -> Dict[str, Any]:
        return self.system.get_performance()
    
    def place_order(self, symbol: str, quantity: int, side: str, **kwargs) -> Dict[str, Any]:
        order_type = kwargs.get('order_type', 'market')
        price = kwargs.get('price')
        return self.system.place_order(symbol, quantity, side, order_type, price)
    
    def get_detailed_positions(self) -> List[Dict[str, Any]]:
        return self.system.get_detailed_positions()
    
    def calculate_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        return self.system.calculate_portfolio_performance(days)

class AlpacaAdapter(TradingSystemInterface):
    """Alpaca适配器"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets", dry_run: bool = True):
        self.system = AlpacaTradingSystem(api_key, secret_key, base_url, dry_run)
        self.broker_name = "Alpaca"
    
    def connect(self) -> bool:
        return self.system.connect()
    
    def disconnect(self):
        self.system.disconnect()
    
    def is_connected(self) -> bool:
        return self.system.is_connected()
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        return self.system.get_portfolio_status()
    
    def get_positions(self) -> List[Dict[str, Any]]:
        return self.system.get_positions()
    
    def get_performance(self) -> Dict[str, Any]:
        return self.system.get_performance()
    
    def place_order(self, symbol: str, quantity: int, side: str, **kwargs) -> Dict[str, Any]:
        order_type = kwargs.get('order_type', 'market')
        time_in_force = kwargs.get('time_in_force', 'day')
        limit_price = kwargs.get('limit_price')
        return self.system.place_order(symbol, quantity, side, order_type, time_in_force, limit_price)
    
    def get_detailed_positions(self) -> List[Dict[str, Any]]:
        # Alpaca的get_positions已经提供了详细信息，直接返回
        return self.system.get_positions()
    
    def calculate_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        # 使用get_performance方法作为基础，添加时间范围参数
        performance = self.system.get_performance()
        performance['days'] = days
        return performance

class MockTradingSystem(TradingSystemInterface):
    """模拟交易系统 - 用于演示和测试"""
    
    def __init__(self, broker_name: str = "Mock"):
        self.broker_name = broker_name
        self.connected = False
        self.mock_portfolio_value = 100000.0
        self.mock_positions = [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'market_value': 15000.0,
                'cost_basis': 14500.0,
                'unrealized_pl': 500.0,
                'unrealized_plpc': 3.45,
                'current_price': 150.0,
                'side': 'long'
            },
            {
                'symbol': 'GOOGL',
                'quantity': 50,
                'market_value': 12500.0,
                'cost_basis': 12000.0,
                'unrealized_pl': 500.0,
                'unrealized_plpc': 4.17,
                'current_price': 250.0,
                'side': 'long'
            }
         ]
    
    def calculate_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        """计算投资组合表现"""
        return {
            'total_return': 15.75,
            'total_return_percent': 1.75,
            'daily_return': 0.05,
            'volatility': 0.12,
            'sharpe_ratio': 1.25,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'profit_factor': 1.85,
            'days': days,
            'start_date': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def connect(self) -> bool:
        self.connected = True
        logger.info(f"{self.broker_name} 模拟连接成功")
        return True
    
    def disconnect(self):
        self.connected = False
        logger.info(f"{self.broker_name} 模拟连接已断开")
    
    def is_connected(self) -> bool:
        return self.connected
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        return {
            'total_value': self.mock_portfolio_value,
            'cash': 72500.0,
            'buying_power': 145000.0,
            'positions_count': len(self.mock_positions),
            'day_change': 1000.0,
            'day_change_percent': 1.0,
            'account_status': 'active'
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        return self.mock_positions.copy()
    
    def get_performance(self) -> Dict[str, Any]:
        return {
            'total_trades': 25,
            'successful_trades': 20,
            'success_rate': 80.0,
            'total_volume': 250000.0,
            'total_profit': 5000.0,
            'avg_profit_per_trade': 200.0,
            'portfolio_value': self.mock_portfolio_value,
            'day_change': 1000.0
        }
    
    def place_order(self, symbol: str, quantity: int, side: str, **kwargs) -> Dict[str, Any]:
        import time
        order_id = f"mock_{int(time.time())}"
        logger.info(f"{self.broker_name} 模拟下单: {side} {quantity} {symbol}")
        return {
            'id': order_id,
            'status': 'filled',
            'symbol': symbol,
            'qty': quantity,
            'side': side,
            'order_type': kwargs.get('order_type', 'market')
        }
    
    def get_detailed_positions(self) -> List[Dict[str, Any]]:
        """获取详细持仓信息"""
        return [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'avg_cost': 150.0,
                'current_price': 155.0,
                'market_value': 15500.0,
                'unrealized_pnl': 500.0,
                'unrealized_pnl_percent': 3.33,
                'side': 'long'
            },
            {
                'symbol': 'GOOGL',
                'quantity': 50,
                'avg_cost': 2800.0,
                'current_price': 2850.0,
                'market_value': 142500.0,
                'unrealized_pnl': 2500.0,
                'unrealized_pnl_percent': 1.79,
                'side': 'long'
            }
        ]

class BrokerFactory:
    """券商工厂类"""
    
    def __init__(self):
        self.brokers: Dict[str, TradingSystemInterface] = {}
        self.primary_broker: Optional[str] = None
    
    def initialize_brokers(self) -> Dict[str, bool]:
        """初始化所有配置的券商"""
        results = {}
        
        # 初始化Firstrade
        if config.firstrade.enabled and config.firstrade.username and config.firstrade.password:
            try:
                firstrade = FirstradeAdapter(
                    config.firstrade.username,
                    config.firstrade.password,
                    config.firstrade.pin,
                    config.firstrade.dry_run
                )
                if firstrade.connect():
                    self.brokers['firstrade'] = firstrade
                    results['firstrade'] = True
                    if not self.primary_broker:
                        self.primary_broker = 'firstrade'
                    logger.info("Firstrade 初始化成功")
                else:
                    results['firstrade'] = False
                    logger.error("Firstrade 连接失败")
            except Exception as e:
                results['firstrade'] = False
                logger.error(f"Firstrade 初始化失败: {e}")
        
        # 初始化Alpaca
        if config.alpaca.enabled and config.alpaca.api_key and config.alpaca.secret_key:
            try:
                alpaca = AlpacaAdapter(
                    config.alpaca.api_key,
                    config.alpaca.secret_key,
                    config.alpaca.base_url,
                    config.alpaca.dry_run
                )
                if alpaca.connect():
                    self.brokers['alpaca'] = alpaca
                    results['alpaca'] = True
                    if not self.primary_broker:
                        self.primary_broker = 'alpaca'
                    logger.info("Alpaca 初始化成功")
                else:
                    results['alpaca'] = False
                    logger.error("Alpaca 连接失败")
            except Exception as e:
                results['alpaca'] = False
                logger.error(f"Alpaca 初始化失败: {e}")
        
        # 如果没有真实券商可用，使用模拟券商
        if not self.brokers:
            logger.warning("没有可用的真实券商，使用模拟券商")
            mock_broker = MockTradingSystem("Mock Broker")
            mock_broker.connect()
            self.brokers['mock'] = mock_broker
            self.primary_broker = 'mock'
            results['mock'] = True
        
        return results
    
    def get_broker(self, broker_name: Optional[str] = None) -> Optional[TradingSystemInterface]:
        """获取指定券商或主要券商"""
        if broker_name:
            return self.brokers.get(broker_name)
        elif self.primary_broker:
            return self.brokers.get(self.primary_broker)
        else:
            return None
    
    def get_all_brokers(self) -> Dict[str, TradingSystemInterface]:
        """获取所有券商"""
        return self.brokers.copy()
    
    def get_primary_broker(self) -> Optional[TradingSystemInterface]:
        """获取主要券商"""
        return self.get_broker()
    
    def get_broker_names(self) -> List[str]:
        """获取所有券商名称"""
        return list(self.brokers.keys())
    
    def is_any_broker_connected(self) -> bool:
        """检查是否有任何券商连接"""
        return any(broker.is_connected() for broker in self.brokers.values())
    
    def disconnect_all(self):
        """断开所有券商连接"""
        for broker in self.brokers.values():
            try:
                broker.disconnect()
            except Exception as e:
                logger.error(f"断开券商连接时发生错误: {e}")
        
        self.brokers.clear()
        self.primary_broker = None
        logger.info("所有券商连接已断开")
    
    def get_aggregated_portfolio_status(self) -> Dict[str, Any]:
        """获取聚合的投资组合状态"""
        if not self.brokers:
            return {}
        
        total_value = 0
        total_cash = 0
        total_positions = 0
        total_day_change = 0
        
        for broker_name, broker in self.brokers.items():
            try:
                status = broker.get_portfolio_status()
                if status:
                    total_value += status.get('total_value', 0)
                    total_cash += status.get('cash', 0)
                    total_positions += status.get('positions_count', 0)
                    total_day_change += status.get('day_change', 0)
            except Exception as e:
                logger.error(f"获取 {broker_name} 投资组合状态失败: {e}")
        
        return {
            'total_value': total_value,
            'cash': total_cash,
            'positions_count': total_positions,
            'day_change': total_day_change,
            'day_change_percent': (total_day_change / total_value * 100) if total_value > 0 else 0,
            'brokers_count': len(self.brokers)
        }
    
    def get_aggregated_positions(self) -> List[Dict[str, Any]]:
        """获取聚合的持仓信息"""
        all_positions = []
        
        for broker_name, broker in self.brokers.items():
            try:
                positions = broker.get_positions()
                for pos in positions:
                    pos['broker'] = broker_name  # 添加券商标识
                    all_positions.append(pos)
            except Exception as e:
                logger.error(f"获取 {broker_name} 持仓信息失败: {e}")
        
        return all_positions
    
    def get_aggregated_performance(self) -> Dict[str, Any]:
        """获取聚合的交易表现"""
        if not self.brokers:
            return {}
        
        total_trades = 0
        successful_trades = 0
        total_volume = 0
        total_profit = 0
        total_portfolio_value = 0
        total_day_change = 0
        
        for broker_name, broker in self.brokers.items():
            try:
                performance = broker.get_performance()
                if performance:
                    total_trades += performance.get('total_trades', 0)
                    successful_trades += performance.get('successful_trades', 0)
                    total_volume += performance.get('total_volume', 0)
                    total_profit += performance.get('total_profit', 0)
                    total_portfolio_value += performance.get('portfolio_value', 0)
                    total_day_change += performance.get('day_change', 0)
            except Exception as e:
                logger.error(f"获取 {broker_name} 交易表现失败: {e}")
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'success_rate': (successful_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_volume': total_volume,
            'total_profit': total_profit,
            'avg_profit_per_trade': (total_profit / total_trades) if total_trades > 0 else 0,
            'portfolio_value': total_portfolio_value,
            'day_change': total_day_change
        }

# 全局券商工厂实例
broker_factory = BrokerFactory()

def test_broker_factory():
    """测试券商工厂"""
    print("测试券商工厂...")
    
    # 初始化券商
    results = broker_factory.initialize_brokers()
    print(f"券商初始化结果: {results}")
    
    # 获取主要券商
    primary = broker_factory.get_primary_broker()
    if primary:
        print(f"主要券商: {primary.broker_name}")
        
        # 测试投资组合状态
        portfolio = primary.get_portfolio_status()
        print(f"投资组合价值: ${portfolio.get('total_value', 0):,.2f}")
        
        # 测试持仓
        positions = primary.get_positions()
        print(f"持仓数量: {len(positions)}")
        
        # 测试交易表现
        performance = primary.get_performance()
        print(f"总交易次数: {performance.get('total_trades', 0)}")
    
    # 测试聚合数据
    agg_portfolio = broker_factory.get_aggregated_portfolio_status()
    print(f"聚合投资组合价值: ${agg_portfolio.get('total_value', 0):,.2f}")
    
    # 断开连接
    broker_factory.disconnect_all()

if __name__ == "__main__":
    test_broker_factory()