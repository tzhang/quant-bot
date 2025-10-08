#!/usr/bin/env python3
"""
Interactive Brokers 适配器
集成IB API到量化交易系统中
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, OrderId

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IBConfig:
    """IB配置类"""
    host: str = "127.0.0.1"
    port: int = 7497  # 模拟交易端口
    client_id: int = 1
    timeout: int = 10

@dataclass
class AccountInfo:
    """账户信息"""
    account_id: str = ""
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0
    currency: str = "USD"

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0

class IBAdapter(EWrapper, EClient):
    """Interactive Brokers 适配器"""
    
    def __init__(self, config: IBConfig = None):
        EClient.__init__(self, self)
        
        self.config = config or IBConfig()
        self.connected = False
        self.next_order_id = None
        self.account_info = AccountInfo()
        self.positions: Dict[str, Position] = {}
        self.market_data: Dict[str, Dict] = {}
        
        # 回调函数
        self.on_connection_callback: Optional[Callable] = None
        self.on_account_update_callback: Optional[Callable] = None
        self.on_position_update_callback: Optional[Callable] = None
        self.on_market_data_callback: Optional[Callable] = None
        
        # 请求ID管理
        self._req_id_counter = 1000
        
    def get_next_req_id(self) -> int:
        """获取下一个请求ID"""
        self._req_id_counter += 1
        return self._req_id_counter
    
    # ========== 连接管理 ==========
    
    def connect_to_ib(self) -> bool:
        """连接到IB TWS"""
        try:
            logger.info(f"正在连接到 IB TWS: {self.config.host}:{self.config.port}")
            self.connect(self.config.host, self.config.port, self.config.client_id)
            
            # 启动消息循环线程
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()
            
            # 等待连接建立
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < self.config.timeout:
                time.sleep(0.1)
                
            if self.connected:
                logger.info("✅ IB API 连接成功")
                self._request_account_info()
                return True
            else:
                logger.error("❌ IB API 连接超时")
                return False
                
        except Exception as e:
            logger.error(f"❌ IB API 连接失败: {e}")
            return False
    
    def disconnect_from_ib(self):
        """断开IB连接"""
        if self.isConnected():
            self.disconnect()
            self.connected = False
            logger.info("🔌 IB API 连接已断开")
    
    # ========== IB API 回调函数 ==========
    
    def connectAck(self):
        """连接确认"""
        logger.info("📡 IB 连接确认收到")
        
    def nextValidId(self, orderId: OrderId):
        """接收下一个有效订单ID"""
        self.connected = True
        self.next_order_id = orderId
        logger.info(f"✅ IB 连接成功，下一个订单ID: {orderId}")
        
        if self.on_connection_callback:
            self.on_connection_callback()
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """错误处理"""
        # 过滤信息性消息
        if errorCode in [2104, 2106, 2158]:  # 连接状态信息
            logger.debug(f"IB 信息: {errorString}")
        elif errorCode == 10089:  # 市场数据订阅
            logger.warning(f"市场数据需要订阅: {errorString}")
        else:
            logger.error(f"IB 错误 [{errorCode}]: {errorString}")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """账户摘要信息"""
        if tag == "NetLiquidation":
            self.account_info.net_liquidation = float(value)
        elif tag == "TotalCashValue":
            self.account_info.total_cash = float(value)
        elif tag == "BuyingPower":
            self.account_info.buying_power = float(value)
            
        self.account_info.account_id = account
        self.account_info.currency = currency
        
        if self.on_account_update_callback:
            self.on_account_update_callback(self.account_info)
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """持仓信息"""
        if position != 0:  # 只记录非零持仓
            pos = Position(
                symbol=contract.symbol,
                quantity=position,
                avg_cost=avgCost
            )
            self.positions[contract.symbol] = pos
            
            if self.on_position_update_callback:
                self.on_position_update_callback(pos)
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """市场数据价格"""
        symbol = self._get_symbol_by_req_id(reqId)
        if symbol:
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
                
            tick_types = {1: "bid", 2: "ask", 4: "last", 6: "high", 7: "low", 9: "close"}
            tick_name = tick_types.get(tickType)
            
            if tick_name:
                self.market_data[symbol][tick_name] = price
                
                if self.on_market_data_callback:
                    self.on_market_data_callback(symbol, tick_name, price)
    
    # ========== 公共接口 ==========
    
    def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        return self.account_info
    
    def get_positions(self) -> Dict[str, Position]:
        """获取持仓信息"""
        return self.positions.copy()
    
    def get_market_data(self, symbol: str) -> Dict:
        """获取市场数据"""
        return self.market_data.get(symbol, {})
    
    def subscribe_market_data(self, symbol: str, exchange: str = "SMART") -> bool:
        """订阅市场数据"""
        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = exchange
            contract.currency = "USD"
            
            req_id = self.get_next_req_id()
            self._symbol_req_map[req_id] = symbol
            
            self.reqMktData(req_id, contract, "", False, False, [])
            logger.info(f"📈 订阅市场数据: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 订阅市场数据失败 {symbol}: {e}")
            return False
    
    def place_order(self, symbol: str, quantity: int, order_type: str = "MKT", 
                   price: float = None) -> bool:
        """下单"""
        try:
            # 创建合约
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # 创建订单
            order = Order()
            order.action = "BUY" if quantity > 0 else "SELL"
            order.totalQuantity = abs(quantity)
            order.orderType = order_type
            
            if order_type == "LMT" and price:
                order.lmtPrice = price
            
            # 提交订单
            order_id = self.next_order_id
            self.placeOrder(order_id, contract, order)
            self.next_order_id += 1
            
            logger.info(f"📋 提交订单: {symbol} {quantity}股 @ {order_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下单失败 {symbol}: {e}")
            return False
    
    def _request_account_info(self):
        """请求账户信息"""
        self.reqAccountSummary(1, "All", "NetLiquidation,TotalCashValue,BuyingPower")
        self.reqPositions()
    
    def _get_symbol_by_req_id(self, req_id: int) -> Optional[str]:
        """根据请求ID获取股票代码"""
        return getattr(self, '_symbol_req_map', {}).get(req_id)
    
    def __init__(self, config: IBConfig = None):
        super().__init__()
        self.config = config or IBConfig()
        self.connected = False
        self.next_order_id = None
        self.account_info = AccountInfo()
        self.positions: Dict[str, Position] = {}
        self.market_data: Dict[str, Dict] = {}
        self._symbol_req_map: Dict[int, str] = {}
        
        # 回调函数
        self.on_connection_callback: Optional[Callable] = None
        self.on_account_update_callback: Optional[Callable] = None
        self.on_position_update_callback: Optional[Callable] = None
        self.on_market_data_callback: Optional[Callable] = None
        
        # 请求ID管理
        self._req_id_counter = 1000

# ========== 使用示例 ==========

def example_usage():
    """使用示例"""
    
    # 创建配置
    config = IBConfig(
        host="127.0.0.1",
        port=7497,  # 模拟交易端口
        client_id=1
    )
    
    # 创建适配器
    ib = IBAdapter(config)
    
    # 设置回调函数
    def on_connection():
        print("🎉 连接成功，开始交易...")
        
    def on_account_update(account: AccountInfo):
        print(f"💰 账户更新: 净值=${account.net_liquidation:,.2f}")
        
    def on_market_data(symbol: str, tick_type: str, price: float):
        print(f"📊 {symbol} {tick_type}: ${price}")
    
    ib.on_connection_callback = on_connection
    ib.on_account_update_callback = on_account_update
    ib.on_market_data_callback = on_market_data
    
    # 连接并测试
    if ib.connect_to_ib():
        print("✅ IB 适配器初始化成功")
        
        # 订阅市场数据
        ib.subscribe_market_data("AAPL")
        
        # 保持连接
        time.sleep(10)
        
        # 获取账户信息
        account = ib.get_account_info()
        print(f"账户净值: ${account.net_liquidation:,.2f}")
        
        # 断开连接
        ib.disconnect_from_ib()
    else:
        print("❌ IB 适配器初始化失败")

if __name__ == "__main__":
    example_usage()