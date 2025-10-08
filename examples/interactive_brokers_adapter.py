"""
Interactive Brokers (IB) 交易系统适配器
支持通过IB API进行股票交易
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading

# 尝试导入IB API
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import OrderId, TickerId
    HAS_IB_API = True
except ImportError:
    HAS_IB_API = False
    print("警告: Interactive Brokers API未安装，将使用模拟模式")
    
    # 创建模拟类以避免导入错误
    class EWrapper:
        pass
    
    class EClient:
        def __init__(self, wrapper):
            pass
    
    class Contract:
        def __init__(self):
            self.symbol = ""
            self.secType = ""
            self.exchange = ""
            self.currency = ""
    
    class Order:
        def __init__(self):
            self.action = ""
            self.totalQuantity = 0
            self.orderType = ""
    
    # 模拟类型别名
    OrderId = int
    TickerId = int

@dataclass
class IBPosition:
    """IB持仓信息"""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float

@dataclass
class IBOrder:
    """IB订单信息"""
    order_id: int
    symbol: str
    action: str  # BUY/SELL
    quantity: float
    order_type: str  # MKT/LMT
    price: Optional[float] = None
    status: str = "Submitted"

class IBTradingApp(EWrapper, EClient):
    """Interactive Brokers交易应用"""
    
    def __init__(self):
        if HAS_IB_API:
            EClient.__init__(self, self)
        self.next_order_id = 1
        self.positions = {}
        self.orders = {}
        self.account_info = {}
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    def nextValidId(self, orderId: OrderId):
        """接收下一个有效订单ID"""
        self.next_order_id = orderId
        self.logger.info(f"下一个有效订单ID: {orderId}")
        
    def connectAck(self):
        """连接确认"""
        self.connected = True
        self.logger.info("成功连接到IB TWS/Gateway")
        
    def connectionClosed(self):
        """连接关闭"""
        self.connected = False
        self.logger.info("与IB TWS/Gateway的连接已关闭")
        
    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        """错误处理"""
        self.logger.error(f"IB错误 - ID: {reqId}, 代码: {errorCode}, 信息: {errorString}")
        
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """持仓更新"""
        symbol = contract.symbol
        self.positions[symbol] = IBPosition(
            symbol=symbol,
            quantity=position,
            avg_cost=avgCost,
            market_value=position * avgCost,  # 简化计算
            unrealized_pnl=0  # 需要实时价格计算
        )
        
    def orderStatus(self, orderId: OrderId, status: str, filled: float, 
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        """订单状态更新"""
        if orderId in self.orders:
            self.orders[orderId].status = status
            self.logger.info(f"订单 {orderId} 状态更新: {status}")

class InteractiveBrokersAdapter:
    """Interactive Brokers交易系统适配器"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        初始化IB适配器
        
        Args:
            host: TWS/Gateway主机地址
            port: 端口号 (TWS: 7497, Gateway: 4001)
            client_id: 客户端ID
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.app = IBTradingApp()
        self.logger = logging.getLogger(__name__)
        self.connected = False
        
    def connect(self) -> bool:
        """连接到IB TWS/Gateway"""
        try:
            if HAS_IB_API:
                self.app.connect(self.host, self.port, self.client_id)
                # 等待连接建立
                time.sleep(2)
                if self.app.connected:
                    self.connected = True
                    self.logger.info("成功连接到Interactive Brokers")
                    # 请求账户信息
                    self.app.reqAccountUpdates(True, "")
                    self.app.reqPositions()
                    return True
                else:
                    self.logger.error("连接IB失败")
                    return False
            else:
                # 模拟模式
                self.connected = True
                self.logger.info("模拟模式：成功连接到Interactive Brokers")
                return True
                
        except Exception as e:
            self.logger.error(f"连接IB时发生错误: {str(e)}")
            return False
            
    def disconnect(self):
        """断开连接"""
        if HAS_IB_API and self.connected:
            self.app.disconnect()
        self.connected = False
        self.logger.info("已断开与Interactive Brokers的连接")
        
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        if not self.connected:
            return {"error": "未连接到IB"}
            
        if HAS_IB_API:
            return {
                "account_id": "IB_ACCOUNT",
                "buying_power": self.app.account_info.get("BuyingPower", 0),
                "total_cash": self.app.account_info.get("TotalCashValue", 0),
                "net_liquidation": self.app.account_info.get("NetLiquidation", 0)
            }
        else:
            # 模拟数据
            return {
                "account_id": "IB_PRODUCTION_ACCOUNT",
                "buying_power": 100000.0,
                "total_cash": 50000.0,
                "net_liquidation": 150000.0
            }
            
    def get_positions(self) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        if not self.connected:
            return []
            
        positions = []
        if HAS_IB_API:
            for symbol, pos in self.app.positions.items():
                positions.append({
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl
                })
        else:
            # 模拟持仓
            positions = [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "avg_cost": 150.0,
                    "market_value": 15000.0,
                    "unrealized_pnl": 500.0
                }
            ]
            
        return positions
        
    def create_contract(self, symbol: str, exchange: str = "SMART") -> Contract:
        """创建合约对象"""
        if HAS_IB_API:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = exchange
            contract.currency = "USD"
            return contract
        else:
            # 模拟合约
            return {"symbol": symbol, "exchange": exchange}
            
    def place_order(self, symbol: str, action: str, quantity: float, 
                   order_type: str = "MKT", price: Optional[float] = None) -> Dict[str, Any]:
        """下单"""
        if not self.connected:
            return {"success": False, "error": "未连接到IB"}
            
        try:
            if HAS_IB_API:
                contract = self.create_contract(symbol)
                order = Order()
                order.action = action.upper()
                order.totalQuantity = quantity
                order.orderType = order_type
                if price and order_type == "LMT":
                    order.lmtPrice = price
                    
                order_id = self.app.next_order_id
                self.app.placeOrder(order_id, contract, order)
                
                # 记录订单
                ib_order = IBOrder(
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    order_type=order_type,
                    price=price
                )
                self.app.orders[order_id] = ib_order
                
                self.logger.info(f"已提交订单: {action} {quantity} {symbol}")
                return {
                    "success": True,
                    "order_id": order_id,
                    "message": f"订单已提交: {action} {quantity} {symbol}"
                }
            else:
                # 模拟下单
                order_id = int(time.time())
                self.logger.info(f"模拟下单: {action} {quantity} {symbol}")
                return {
                    "success": True,
                    "order_id": order_id,
                    "message": f"模拟订单已提交: {action} {quantity} {symbol}"
                }
                
        except Exception as e:
            self.logger.error(f"下单失败: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """取消订单"""
        if not self.connected:
            return {"success": False, "error": "未连接到IB"}
            
        try:
            if HAS_IB_API:
                self.app.cancelOrder(order_id)
                self.logger.info(f"已取消订单: {order_id}")
                return {"success": True, "message": f"订单 {order_id} 已取消"}
            else:
                self.logger.info(f"模拟取消订单: {order_id}")
                return {"success": True, "message": f"模拟订单 {order_id} 已取消"}
                
        except Exception as e:
            self.logger.error(f"取消订单失败: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据"""
        if not self.connected:
            return {"error": "未连接到IB"}
            
        # 简化实现，返回模拟数据
        return {
            "symbol": symbol,
            "price": 150.0 + hash(symbol) % 100,
            "bid": 149.5,
            "ask": 150.5,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }

def test_ib_connection():
    """测试IB连接"""
    print("测试Interactive Brokers连接...")
    
    # 创建适配器
    ib = InteractiveBrokersAdapter()
    
    # 连接测试
    if ib.connect():
        print("✅ 连接成功")
        
        # 获取账户信息
        account_info = ib.get_account_info()
        print(f"账户信息: {account_info}")
        
        # 获取持仓
        positions = ib.get_positions()
        print(f"持仓信息: {positions}")
        
        # 获取市场数据
        market_data = ib.get_market_data("AAPL")
        print(f"AAPL市场数据: {market_data}")
        
        # 模拟下单测试
        order_result = ib.place_order("AAPL", "BUY", 10, "MKT")
        print(f"下单结果: {order_result}")
        
        # 断开连接
        ib.disconnect()
        print("✅ 测试完成")
    else:
        print("❌ 连接失败")

if __name__ == "__main__":
    test_ib_connection()