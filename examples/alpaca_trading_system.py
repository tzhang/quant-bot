#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpaca交易系统实现
提供与Alpaca Markets API的集成
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
import json

logger = logging.getLogger(__name__)

class AlpacaTradingSystem:
    """Alpaca交易系统"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets", dry_run: bool = True):
        """
        初始化Alpaca交易系统
        
        Args:
            api_key: Alpaca API密钥
            secret_key: Alpaca秘密密钥
            base_url: API基础URL（默认为纸上交易）
            dry_run: 是否为模拟模式
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')
        self.dry_run = dry_run
        
        # API端点
        self.data_url = "https://data.alpaca.markets"
        
        # 请求头
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        
        # 连接状态
        self.connected = False
        self.account_info = {}
        
        logger.info(f"AlpacaTradingSystem 初始化完成 (dry_run={dry_run})")
    
    def connect(self) -> bool:
        """连接到Alpaca API"""
        try:
            # 测试连接并获取账户信息
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.account_info = response.json()
                self.connected = True
                logger.info("成功连接到Alpaca API")
                logger.info(f"账户状态: {self.account_info.get('status', 'unknown')}")
                return True
            else:
                logger.error(f"连接Alpaca API失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"连接Alpaca API时发生错误: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.connected = False
        logger.info("已断开Alpaca API连接")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected
    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        if not self.connected:
            return {}
        
        try:
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"获取账户信息失败: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"获取账户信息时发生错误: {e}")
            return {}
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        account_info = self.get_account_info()
        if not account_info:
            return {}
        
        try:
            # 获取持仓信息
            positions_response = requests.get(
                f"{self.base_url}/v2/positions",
                headers=self.headers,
                timeout=10
            )
            
            positions = []
            if positions_response.status_code == 200:
                positions = positions_response.json()
            
            # 计算投资组合价值
            portfolio_value = float(account_info.get('portfolio_value', 0))
            cash = float(account_info.get('cash', 0))
            buying_power = float(account_info.get('buying_power', 0))
            
            return {
                'total_value': portfolio_value,
                'cash': cash,
                'buying_power': buying_power,
                'positions_count': len(positions),
                'day_change': float(account_info.get('unrealized_pl', 0)),
                'day_change_percent': (float(account_info.get('unrealized_pl', 0)) / portfolio_value * 100) if portfolio_value > 0 else 0,
                'account_status': account_info.get('status', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"获取投资组合状态时发生错误: {e}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """获取当前持仓"""
        if not self.connected:
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                positions = response.json()
                formatted_positions = []
                
                for pos in positions:
                    formatted_positions.append({
                        'symbol': pos.get('symbol', ''),
                        'quantity': float(pos.get('qty', 0)),
                        'market_value': float(pos.get('market_value', 0)),
                        'cost_basis': float(pos.get('cost_basis', 0)),
                        'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                        'unrealized_plpc': float(pos.get('unrealized_plpc', 0)),
                        'current_price': float(pos.get('current_price', 0)),
                        'side': pos.get('side', 'long')
                    })
                
                return formatted_positions
            else:
                logger.error(f"获取持仓失败: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"获取持仓时发生错误: {e}")
            return []
    
    def get_performance(self) -> Dict[str, Any]:
        """获取交易表现"""
        if not self.connected:
            return {}
        
        try:
            # 获取账户信息
            account_info = self.get_account_info()
            if not account_info:
                return {}
            
            # 获取最近的订单历史
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            orders_response = requests.get(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                params={
                    'status': 'all',
                    'limit': 500,
                    'after': start_date.isoformat(),
                    'until': end_date.isoformat()
                },
                timeout=10
            )
            
            orders = []
            if orders_response.status_code == 200:
                orders = orders_response.json()
            
            # 统计交易数据
            total_trades = len([o for o in orders if o.get('status') == 'filled'])
            successful_trades = len([o for o in orders if o.get('status') == 'filled' and float(o.get('filled_qty', 0)) > 0])
            
            # 计算总交易量
            total_volume = sum(float(o.get('filled_qty', 0)) * float(o.get('filled_avg_price', 0)) 
                             for o in orders if o.get('status') == 'filled')
            
            # 获取总盈亏
            total_profit = float(account_info.get('unrealized_pl', 0))
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'success_rate': (successful_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_volume': total_volume,
                'total_profit': total_profit,
                'avg_profit_per_trade': (total_profit / total_trades) if total_trades > 0 else 0,
                'portfolio_value': float(account_info.get('portfolio_value', 0)),
                'day_change': float(account_info.get('unrealized_pl', 0))
            }
            
        except Exception as e:
            logger.error(f"获取交易表现时发生错误: {e}")
            return {}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据"""
        try:
            # 获取最新报价
            response = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                quote = data.get('quote', {})
                
                return {
                    'symbol': symbol,
                    'price': float(quote.get('ap', 0)),  # ask price
                    'bid': float(quote.get('bp', 0)),   # bid price
                    'ask': float(quote.get('ap', 0)),   # ask price
                    'volume': int(quote.get('as', 0)),  # ask size
                    'timestamp': quote.get('t', '')
                }
            else:
                logger.error(f"获取市场数据失败: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"获取市场数据时发生错误: {e}")
            return {}
    
    def place_order(self, symbol: str, quantity: int, side: str, order_type: str = "market", 
                   time_in_force: str = "day", limit_price: Optional[float] = None) -> Dict[str, Any]:
        """下单"""
        if self.dry_run:
            logger.info(f"模拟下单: {side} {quantity} {symbol} @ {order_type}")
            return {
                'id': f'dry_run_{int(time.time())}',
                'status': 'filled',
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'order_type': order_type
            }
        
        if not self.connected:
            logger.error("未连接到Alpaca API")
            return {}
        
        try:
            order_data = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            if order_type == 'limit' and limit_price:
                order_data['limit_price'] = str(limit_price)
            
            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=10
            )
            
            if response.status_code == 201:
                order = response.json()
                logger.info(f"订单提交成功: {order.get('id')}")
                return order
            else:
                logger.error(f"下单失败: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"下单时发生错误: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if self.dry_run:
            logger.info(f"模拟取消订单: {order_id}")
            return True
        
        if not self.connected:
            logger.error("未连接到Alpaca API")
            return False
        
        try:
            response = requests.delete(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"订单取消成功: {order_id}")
                return True
            else:
                logger.error(f"取消订单失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"取消订单时发生错误: {e}")
            return False
    
    def get_orders(self, status: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        """获取订单列表"""
        if not self.connected:
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                params={
                    'status': status,
                    'limit': limit
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"获取订单列表失败: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"获取订单列表时发生错误: {e}")
            return []

def test_alpaca_connection():
    """测试Alpaca连接"""
    # 这里使用示例凭据，实际使用时需要真实凭据
    api_key = "your_api_key"
    secret_key = "your_secret_key"
    
    if api_key == "your_api_key":
        print("请在config.py中配置真实的Alpaca API凭据")
        return
    
    system = AlpacaTradingSystem(api_key, secret_key, dry_run=True)
    
    if system.connect():
        print("Alpaca连接成功!")
        
        # 测试获取账户信息
        account = system.get_account_info()
        print(f"账户状态: {account.get('status', 'unknown')}")
        
        # 测试获取投资组合
        portfolio = system.get_portfolio_status()
        print(f"投资组合价值: ${portfolio.get('total_value', 0):,.2f}")
        
        # 测试获取持仓
        positions = system.get_positions()
        print(f"持仓数量: {len(positions)}")
        
        system.disconnect()
    else:
        print("Alpaca连接失败")

if __name__ == "__main__":
    test_alpaca_connection()