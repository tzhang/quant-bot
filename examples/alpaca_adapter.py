"""
Alpaca 交易系统适配器
支持通过Alpaca API进行股票交易
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import requests
import json

# 尝试导入Alpaca API
try:
    import alpaca_trade_api as tradeapi
    HAS_ALPACA_API = True
except ImportError:
    HAS_ALPACA_API = False
    print("警告: Alpaca API未安装，将使用模拟模式")
    
    # 创建模拟类以避免导入错误
    class tradeapi:
        class REST:
            def __init__(self, *args, **kwargs):
                pass

@dataclass
class AlpacaPosition:
    """Alpaca持仓信息"""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    side: str  # long/short

@dataclass
class AlpacaOrder:
    """Alpaca订单信息"""
    order_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    order_type: str  # market/limit
    price: Optional[float] = None
    status: str = "new"
    filled_qty: float = 0

class AlpacaAdapter:
    """Alpaca交易系统适配器"""
    
    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = None):
        """
        初始化Alpaca适配器
        
        Args:
            api_key: Alpaca API密钥
            secret_key: Alpaca秘密密钥
            base_url: API基础URL (paper trading: https://paper-api.alpaca.markets)
        """
        self.api_key = api_key or "demo_key"
        self.secret_key = secret_key or "demo_secret"
        self.base_url = base_url or "https://paper-api.alpaca.markets"
        self.logger = logging.getLogger(__name__)
        self.api = None
        self.connected = False
        
    def connect(self) -> bool:
        """连接到Alpaca API"""
        try:
            if HAS_ALPACA_API:
                self.api = tradeapi.REST(
                    key_id=self.api_key,
                    secret_key=self.secret_key,
                    base_url=self.base_url,
                    api_version='v2'
                )
                
                # 测试连接
                account = self.api.get_account()
                if account:
                    self.connected = True
                    self.logger.info(f"成功连接到Alpaca账户: {account.id}")
                    return True
                else:
                    self.logger.error("无法获取Alpaca账户信息")
                    return False
            else:
                # 模拟模式
                self.connected = True
                self.logger.info("模拟模式：成功连接到Alpaca")
                return True
                
        except Exception as e:
            self.logger.error(f"连接Alpaca时发生错误: {str(e)}")
            return False
            
    def disconnect(self):
        """断开连接"""
        self.connected = False
        self.api = None
        self.logger.info("已断开与Alpaca的连接")
        
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        if not self.connected:
            return {"error": "未连接到Alpaca"}
            
        try:
            if HAS_ALPACA_API and self.api:
                account = self.api.get_account()
                return {
                    "account_id": account.id,
                    "buying_power": float(account.buying_power),
                    "cash": float(account.cash),
                    "portfolio_value": float(account.portfolio_value),
                    "equity": float(account.equity),
                    "day_trade_count": account.daytrade_count,
                    "pattern_day_trader": account.pattern_day_trader
                }
            else:
                # 模拟数据
                return {
                    "account_id": "ALPACA_DEMO_ACCOUNT",
                    "buying_power": 100000.0,
                    "cash": 50000.0,
                    "portfolio_value": 150000.0,
                    "equity": 150000.0,
                    "day_trade_count": 0,
                    "pattern_day_trader": False
                }
                
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {str(e)}")
            return {"error": str(e)}
            
    def get_positions(self) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        if not self.connected:
            return []
            
        try:
            positions = []
            if HAS_ALPACA_API and self.api:
                alpaca_positions = self.api.list_positions()
                for pos in alpaca_positions:
                    positions.append({
                        "symbol": pos.symbol,
                        "quantity": float(pos.qty),
                        "avg_cost": float(pos.avg_entry_price),
                        "market_value": float(pos.market_value),
                        "unrealized_pnl": float(pos.unrealized_pl),
                        "side": pos.side
                    })
            else:
                # 模拟持仓
                positions = [
                    {
                        "symbol": "AAPL",
                        "quantity": 50,
                        "avg_cost": 155.0,
                        "market_value": 7750.0,
                        "unrealized_pnl": 250.0,
                        "side": "long"
                    },
                    {
                        "symbol": "TSLA",
                        "quantity": 20,
                        "avg_cost": 200.0,
                        "market_value": 4000.0,
                        "unrealized_pnl": -200.0,
                        "side": "long"
                    }
                ]
                
            return positions
            
        except Exception as e:
            self.logger.error(f"获取持仓信息失败: {str(e)}")
            return []
            
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = "market", price: Optional[float] = None,
                   time_in_force: str = "day") -> Dict[str, Any]:
        """下单"""
        if not self.connected:
            return {"success": False, "error": "未连接到Alpaca"}
            
        try:
            if HAS_ALPACA_API and self.api:
                order_params = {
                    'symbol': symbol,
                    'qty': quantity,
                    'side': side.lower(),
                    'type': order_type.lower(),
                    'time_in_force': time_in_force
                }
                
                if price and order_type.lower() == 'limit':
                    order_params['limit_price'] = price
                    
                order = self.api.submit_order(**order_params)
                
                self.logger.info(f"已提交订单: {side} {quantity} {symbol}")
                return {
                    "success": True,
                    "order_id": order.id,
                    "status": order.status,
                    "message": f"订单已提交: {side} {quantity} {symbol}"
                }
            else:
                # 模拟下单
                order_id = f"alpaca_{int(time.time())}"
                self.logger.info(f"模拟下单: {side} {quantity} {symbol}")
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": "filled",
                    "message": f"模拟订单已提交: {side} {quantity} {symbol}"
                }
                
        except Exception as e:
            self.logger.error(f"下单失败: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """取消订单"""
        if not self.connected:
            return {"success": False, "error": "未连接到Alpaca"}
            
        try:
            if HAS_ALPACA_API and self.api:
                self.api.cancel_order(order_id)
                self.logger.info(f"已取消订单: {order_id}")
                return {"success": True, "message": f"订单 {order_id} 已取消"}
            else:
                self.logger.info(f"模拟取消订单: {order_id}")
                return {"success": True, "message": f"模拟订单 {order_id} 已取消"}
                
        except Exception as e:
            self.logger.error(f"取消订单失败: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def get_orders(self, status: str = "all") -> List[Dict[str, Any]]:
        """获取订单列表"""
        if not self.connected:
            return []
            
        try:
            orders = []
            if HAS_ALPACA_API and self.api:
                alpaca_orders = self.api.list_orders(status=status)
                for order in alpaca_orders:
                    orders.append({
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": float(order.qty),
                        "filled_qty": float(order.filled_qty or 0),
                        "order_type": order.order_type,
                        "status": order.status,
                        "submitted_at": order.submitted_at,
                        "filled_at": order.filled_at
                    })
            else:
                # 模拟订单
                orders = [
                    {
                        "order_id": "alpaca_demo_1",
                        "symbol": "AAPL",
                        "side": "buy",
                        "quantity": 10,
                        "filled_qty": 10,
                        "order_type": "market",
                        "status": "filled",
                        "submitted_at": datetime.now().isoformat(),
                        "filled_at": datetime.now().isoformat()
                    }
                ]
                
            return orders
            
        except Exception as e:
            self.logger.error(f"获取订单列表失败: {str(e)}")
            return []
            
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据"""
        if not self.connected:
            return {"error": "未连接到Alpaca"}
            
        try:
            if HAS_ALPACA_API and self.api:
                # 获取最新报价
                quote = self.api.get_latest_quote(symbol)
                return {
                    "symbol": symbol,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "timestamp": quote.timestamp.isoformat()
                }
            else:
                # 模拟数据
                base_price = 150.0 + hash(symbol) % 100
                return {
                    "symbol": symbol,
                    "bid": base_price - 0.05,
                    "ask": base_price + 0.05,
                    "bid_size": 100,
                    "ask_size": 100,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {str(e)}")
            return {"error": str(e)}
            
    def get_historical_data(self, symbol: str, timeframe: str = "1Day", 
                          start: str = None, end: str = None) -> List[Dict[str, Any]]:
        """获取历史数据"""
        if not self.connected:
            return []
            
        try:
            if HAS_ALPACA_API and self.api:
                bars = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start,
                    end=end
                ).df
                
                historical_data = []
                for index, row in bars.iterrows():
                    historical_data.append({
                        "timestamp": index.isoformat(),
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "volume": int(row['volume'])
                    })
                    
                return historical_data
            else:
                # 模拟历史数据
                return [
                    {
                        "timestamp": "2024-01-01T09:30:00",
                        "open": 150.0,
                        "high": 155.0,
                        "low": 148.0,
                        "close": 152.0,
                        "volume": 1000000
                    }
                ]
                
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {str(e)}")
            return []

def test_alpaca_connection():
    """测试Alpaca连接"""
    print("测试Alpaca连接...")
    
    # 创建适配器
    alpaca = AlpacaAdapter()
    
    # 连接测试
    if alpaca.connect():
        print("✅ 连接成功")
        
        # 获取账户信息
        account_info = alpaca.get_account_info()
        print(f"账户信息: {account_info}")
        
        # 获取持仓
        positions = alpaca.get_positions()
        print(f"持仓信息: {positions}")
        
        # 获取市场数据
        market_data = alpaca.get_market_data("AAPL")
        print(f"AAPL市场数据: {market_data}")
        
        # 获取订单
        orders = alpaca.get_orders()
        print(f"订单列表: {orders}")
        
        # 模拟下单测试
        order_result = alpaca.place_order("AAPL", "buy", 5, "market")
        print(f"下单结果: {order_result}")
        
        # 断开连接
        alpaca.disconnect()
        print("✅ 测试完成")
    else:
        print("❌ 连接失败")

if __name__ == "__main__":
    test_alpaca_connection()