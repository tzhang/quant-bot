#!/usr/bin/env python3
"""
增强的Interactive Brokers交易系统
支持模拟交易(Paper Trading)和实盘交易的无缝切换
优先使用IB API进行所有交易处理
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

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, OrderId
from ibapi.execution import Execution, ExecutionFilter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """交易模式枚举"""
    PAPER = "paper"  # 模拟交易
    LIVE = "live"    # 实盘交易

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "PendingSubmit"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"

@dataclass
class TradingConfig:
    """交易配置"""
    # 连接配置
    host: str = "127.0.0.1"
    paper_port: int = 4002  # IB Gateway 模拟交易端口
    live_port: int = 4001   # IB Gateway 实盘交易端口
    client_id: int = 1
    
    # 交易模式
    trading_mode: TradingMode = TradingMode.PAPER
    
    # 风险管理
    max_position_value: float = 50000.0
    max_daily_loss: float = 5000.0
    max_symbol_exposure: float = 20000.0
    max_daily_trades: int = 100
    
    # 订单配置
    default_order_type: str = "MKT"
    timeout_seconds: int = 30
    
    @property
    def port(self) -> int:
        """根据交易模式返回对应端口"""
        return self.paper_port if self.trading_mode == TradingMode.PAPER else self.live_port

@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class OrderInfo:
    """订单信息"""
    order_id: int
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    order_type: str
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TradeExecution:
    """交易执行记录"""
    execution_id: str
    order_id: int
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0

class EnhancedIBTradingSystem(EWrapper, EClient):
    """增强的IB交易系统"""
    
    def __init__(self, config: TradingConfig):
        EClient.__init__(self, self)
        
        self.config = config
        self.connected = False
        self.next_order_id = None
        
        # 数据存储
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[int, OrderInfo] = {}
        self.market_data: Dict[str, MarketData] = {}
        self.executions: List[TradeExecution] = []
        
        # 账户信息
        self.account_id = ""
        self.net_liquidation = 0.0
        self.total_cash = 0.0
        self.buying_power = 0.0
        
        # 风险管理
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.position_values: Dict[str, float] = {}
        
        # 请求ID管理
        self._req_id_counter = 1000
        self._symbol_req_map: Dict[int, str] = {}
        
        # 回调函数
        self.callbacks = {
            'on_connection': [],
            'on_order_status': [],
            'on_execution': [],
            'on_position_update': [],
            'on_market_data': [],
            'on_error': []
        }
        
        # 线程安全
        self._lock = threading.Lock()
        
    def get_next_req_id(self) -> int:
        """获取下一个请求ID"""
        self._req_id_counter += 1
        return self._req_id_counter
    
    # ========== 连接管理 ==========
    
    def connect_to_ib(self) -> bool:
        """连接到IB TWS"""
        try:
            mode_str = "模拟交易" if self.config.trading_mode == TradingMode.PAPER else "实盘交易"
            logger.info(f"正在连接到IB TWS ({mode_str}): {self.config.host}:{self.config.port}")
            
            self.connect(self.config.host, self.config.port, self.config.client_id)
            
            # 启动消息循环线程
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()
            
            # 等待连接建立
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < self.config.timeout_seconds:
                time.sleep(0.1)
                
            if self.connected:
                logger.info(f"✅ IB API 连接成功 ({mode_str})")
                self._request_initial_data()
                return True
            else:
                logger.error(f"❌ IB API 连接失败 ({mode_str})")
                return False
                
        except Exception as e:
            logger.error(f"连接异常: {e}")
            return False
    
    def disconnect_from_ib(self):
        """断开IB连接"""
        if self.connected:
            self.disconnect()
            self.connected = False
            logger.info("IB API 连接已断开")
    
    def switch_trading_mode(self, new_mode: TradingMode) -> bool:
        """切换交易模式"""
        if new_mode == self.config.trading_mode:
            logger.info(f"已经处于{new_mode.value}模式")
            return True
            
        logger.info(f"正在从{self.config.trading_mode.value}模式切换到{new_mode.value}模式")
        
        # 断开当前连接
        if self.connected:
            self.disconnect_from_ib()
            time.sleep(2)  # 等待断开完成
        
        # 更新配置
        self.config.trading_mode = new_mode
        
        # 重新连接
        return self.connect_to_ib()
    
    # ========== IB API 回调函数 ==========
    
    def connectAck(self):
        """连接确认"""
        logger.info("IB API 连接确认")
        
    def nextValidId(self, orderId: OrderId):
        """接收下一个有效订单ID"""
        self.connected = True
        self.next_order_id = orderId
        logger.info(f"连接成功，下一个订单ID: {orderId}")
        
        # 触发连接回调
        for callback in self.callbacks['on_connection']:
            try:
                callback(True)
            except Exception as e:
                logger.error(f"连接回调错误: {e}")
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """错误处理"""
        # 过滤信息性消息
        if errorCode in [2104, 2106, 2158]:  # 连接状态信息
            logger.debug(f"信息: {errorString}")
            return
            
        if errorCode >= 2000:  # 警告
            logger.warning(f"警告 (ID: {reqId}, 代码: {errorCode}): {errorString}")
        else:  # 错误
            logger.error(f"错误 (ID: {reqId}, 代码: {errorCode}): {errorString}")
            
        # 触发错误回调
        for callback in self.callbacks['on_error']:
            try:
                callback(reqId, errorCode, errorString)
            except Exception as e:
                logger.error(f"错误回调异常: {e}")
    
    def orderStatus(self, orderId: OrderId, status: str, filled: float, 
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        """订单状态更新"""
        with self._lock:
            if orderId in self.orders:
                order = self.orders[orderId]
                order.status = OrderStatus(status) if status in [s.value for s in OrderStatus] else OrderStatus.PENDING
                order.filled_qty = int(filled)
                order.avg_fill_price = avgFillPrice
                
                logger.info(f"订单状态更新: {orderId} - {status}, 已成交: {filled}, 均价: {avgFillPrice}")
                
                # 触发订单状态回调
                for callback in self.callbacks['on_order_status']:
                    try:
                        callback(order)
                    except Exception as e:
                        logger.error(f"订单状态回调错误: {e}")
    
    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        """执行详情"""
        trade_execution = TradeExecution(
            execution_id=execution.execId,
            order_id=execution.orderId,
            symbol=contract.symbol,
            side=execution.side,
            quantity=int(execution.shares),
            price=execution.price,
            timestamp=datetime.strptime(execution.time, "%Y%m%d  %H:%M:%S")
        )
        
        with self._lock:
            self.executions.append(trade_execution)
            self.daily_trades += 1
            
        logger.info(f"交易执行: {trade_execution.symbol} {trade_execution.side} {trade_execution.quantity}@{trade_execution.price}")
        
        # 触发执行回调
        for callback in self.callbacks['on_execution']:
            try:
                callback(trade_execution)
            except Exception as e:
                logger.error(f"执行回调错误: {e}")
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """持仓更新"""
        if position == 0:
            # 移除空仓位
            if contract.symbol in self.positions:
                del self.positions[contract.symbol]
        else:
            pos = Position(
                symbol=contract.symbol,
                quantity=position,
                avg_cost=avgCost,
                market_value=position * avgCost  # 临时值，等待市场数据更新
            )
            
            with self._lock:
                self.positions[contract.symbol] = pos
                
            logger.info(f"持仓更新: {contract.symbol} {position}@{avgCost}")
            
            # 触发持仓更新回调
            for callback in self.callbacks['on_position_update']:
                try:
                    callback(pos)
                except Exception as e:
                    logger.error(f"持仓更新回调错误: {e}")
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """市场数据价格更新"""
        symbol = self._symbol_req_map.get(reqId)
        if not symbol:
            return
            
        with self._lock:
            if symbol not in self.market_data:
                self.market_data[symbol] = MarketData(symbol=symbol)
                
            data = self.market_data[symbol]
            
            # 更新对应的价格类型
            if tickType == 1:  # 买价
                data.bid = price
            elif tickType == 2:  # 卖价
                data.ask = price
            elif tickType == 4:  # 最新价
                data.last = price
                
            data.timestamp = datetime.now()
            
            # 更新持仓市值
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.market_price = price
                pos.market_value = pos.quantity * price
                pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity
        
        # 触发市场数据回调
        for callback in self.callbacks['on_market_data']:
            try:
                callback(symbol, self.market_data[symbol])
            except Exception as e:
                logger.error(f"市场数据回调错误: {e}")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """账户摘要"""
        try:
            if tag == "NetLiquidation":
                self.net_liquidation = float(value)
            elif tag == "TotalCashValue":
                self.total_cash = float(value)
            elif tag == "BuyingPower":
                self.buying_power = float(value)
            elif tag == "AccountCode":
                self.account_id = value
                
            logger.debug(f"账户信息: {tag} = {value} {currency}")
        except ValueError:
            pass
    
    # ========== 交易功能 ==========
    
    def place_order(self, symbol: str, action: str, quantity: int, 
                   order_type: str = "MKT", limit_price: Optional[float] = None) -> Optional[int]:
        """下单"""
        if not self.connected or self.next_order_id is None:
            logger.error("未连接到IB API")
            return None
            
        # 风险检查
        if not self._check_risk_limits(symbol, action, quantity, limit_price or 0):
            logger.error("订单被风险管理拒绝")
            return None
        
        # 创建合约
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        # 创建订单
        order = Order()
        order.action = action.upper()
        order.totalQuantity = quantity
        order.orderType = order_type
        
        if order_type == "LMT" and limit_price:
            order.lmtPrice = limit_price
        
        # 获取订单ID
        order_id = self.next_order_id
        self.next_order_id += 1
        
        # 记录订单信息
        order_info = OrderInfo(
            order_id=order_id,
            symbol=symbol,
            action=action.upper(),
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price
        )
        
        with self._lock:
            self.orders[order_id] = order_info
        
        # 提交订单
        try:
            self.placeOrder(order_id, contract, order)
            logger.info(f"订单已提交: {order_id} - {symbol} {action} {quantity} @ {order_type}")
            return order_id
        except Exception as e:
            logger.error(f"下单失败: {e}")
            with self._lock:
                del self.orders[order_id]
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """取消订单"""
        try:
            self.cancelOrder(order_id)
            logger.info(f"取消订单: {order_id}")
            return True
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    def subscribe_market_data(self, symbol: str) -> bool:
        """订阅市场数据"""
        try:
            req_id = self.get_next_req_id()
            
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # 为特定股票设置主要交易所 - 这是关键修复
            if symbol == "AAPL":
                contract.primaryExchange = "NASDAQ"
            elif symbol == "GOOGL":
                contract.primaryExchange = "NASDAQ"
            elif symbol == "MSFT":
                contract.primaryExchange = "NASDAQ"
            elif symbol == "TSLA":
                contract.primaryExchange = "NASDAQ"
            elif symbol == "NVDA":
                contract.primaryExchange = "NASDAQ"
            
            self._symbol_req_map[req_id] = symbol
            self.reqMktData(req_id, contract, "", False, False, [])
            
            logger.info(f"已订阅市场数据: {symbol} (主要交易所: {contract.primaryExchange})")
            return True
        except Exception as e:
            logger.error(f"订阅市场数据失败: {e}")
            return False
    
    def unsubscribe_market_data(self, symbol: str):
        """取消订阅市场数据"""
        req_id = None
        for rid, sym in self._symbol_req_map.items():
            if sym == symbol:
                req_id = rid
                break
                
        if req_id:
            self.cancelMktData(req_id)
            del self._symbol_req_map[req_id]
            logger.info(f"已取消订阅: {symbol}")
    
    # ========== 风险管理 ==========
    
    def _check_risk_limits(self, symbol: str, action: str, quantity: int, price: float) -> bool:
        """检查风险限制"""
        # 检查日交易次数
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning(f"超过日交易限制: {self.daily_trades}/{self.config.max_daily_trades}")
            return False
        
        # 检查日损失限制
        if self.daily_pnl <= -self.config.max_daily_loss:
            logger.warning(f"超过日损失限制: {self.daily_pnl}")
            return False
        
        # 检查单个标的暴露
        position_value = quantity * price
        current_exposure = self.position_values.get(symbol, 0)
        
        if action.upper() == "BUY":
            new_exposure = current_exposure + position_value
        else:
            new_exposure = current_exposure - position_value
            
        if abs(new_exposure) > self.config.max_symbol_exposure:
            logger.warning(f"超过单标的暴露限制: {symbol} {new_exposure}")
            return False
        
        # 检查总仓位价值
        total_position_value = sum(abs(v) for v in self.position_values.values())
        if total_position_value + position_value > self.config.max_position_value:
            logger.warning(f"超过总仓位限制: {total_position_value + position_value}")
            return False
        
        return True
    
    def update_daily_pnl(self):
        """更新日盈亏"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(exec.price * exec.quantity for exec in self.executions 
                           if exec.timestamp.date() == datetime.now().date())
        self.daily_pnl = total_unrealized + total_realized
    
    # ========== 数据获取 ==========
    
    def get_positions(self) -> Dict[str, Position]:
        """获取持仓"""
        with self._lock:
            return self.positions.copy()
    
    def get_orders(self) -> Dict[int, OrderInfo]:
        """获取订单"""
        with self._lock:
            return self.orders.copy()
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """获取市场数据"""
        with self._lock:
            return self.market_data.get(symbol)
    
    def get_account_summary(self) -> Dict[str, Any]:
        """获取账户摘要"""
        return {
            'account_id': self.account_id,
            'net_liquidation': self.net_liquidation,
            'total_cash': self.total_cash,
            'buying_power': self.buying_power,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'trading_mode': self.config.trading_mode.value
        }
    
    # ========== 回调管理 ==========
    
    def add_callback(self, event_type: str, callback: callable):
        """添加回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: callable):
        """移除回调函数"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    # ========== 私有方法 ==========
    
    def _request_initial_data(self):
        """请求初始数据"""
        # 请求账户信息
        self.reqAccountSummary(9001, "All", "NetLiquidation,TotalCashValue,BuyingPower")
        
        # 请求持仓信息
        self.reqPositions()
        
        # 请求当日执行 - 使用ExecutionFilter对象
        execution_filter = ExecutionFilter()
        self.reqExecutions(9002, execution_filter)


def example_usage():
    """使用示例"""
    # 创建配置
    config = TradingConfig(
        trading_mode=TradingMode.PAPER,  # 使用模拟交易
        max_position_value=10000.0,
        max_daily_trades=50
    )
    
    # 创建交易系统
    trading_system = EnhancedIBTradingSystem(config)
    
    # 添加回调函数
    def on_connection(success):
        if success:
            print("✅ 连接成功")
            # 订阅市场数据
            trading_system.subscribe_market_data("AAPL")
            trading_system.subscribe_market_data("MSFT")
        else:
            print("❌ 连接失败")
    
    def on_market_data(symbol, data):
        print(f"📈 {symbol}: 买价={data.bid}, 卖价={data.ask}, 最新价={data.last}")
    
    def on_order_status(order):
        print(f"📋 订单状态: {order.order_id} - {order.status.value}")
    
    def on_execution(execution):
        print(f"✅ 交易执行: {execution.symbol} {execution.side} {execution.quantity}@{execution.price}")
    
    # 注册回调
    trading_system.add_callback('on_connection', on_connection)
    trading_system.add_callback('on_market_data', on_market_data)
    trading_system.add_callback('on_order_status', on_order_status)
    trading_system.add_callback('on_execution', on_execution)
    
    # 连接到IB
    if trading_system.connect_to_ib():
        print("🚀 交易系统启动成功")
        
        try:
            # 等待一段时间获取市场数据
            time.sleep(5)
            
            # 示例交易
            print("\n📊 当前账户信息:")
            account_info = trading_system.get_account_summary()
            for key, value in account_info.items():
                print(f"  {key}: {value}")
            
            # 下单示例（注意：这是真实的模拟交易订单）
            print("\n📋 下单示例:")
            order_id = trading_system.place_order("AAPL", "BUY", 10, "MKT")
            if order_id:
                print(f"订单已提交: {order_id}")
                
                # 等待订单执行
                time.sleep(3)
                
                # 查看订单状态
                orders = trading_system.get_orders()
                if order_id in orders:
                    order = orders[order_id]
                    print(f"订单状态: {order.status.value}")
            
            # 保持连接
            print("\n⏳ 系统运行中，按 Ctrl+C 停止...")
            while True:
                time.sleep(10)
                
                # 显示实时信息
                positions = trading_system.get_positions()
                if positions:
                    print(f"\n📊 当前持仓:")
                    for symbol, pos in positions.items():
                        print(f"  {symbol}: {pos.quantity}股 @ ${pos.avg_cost:.2f}, 市值: ${pos.market_value:.2f}")
                
        except KeyboardInterrupt:
            print("\n⚠️ 正在停止系统...")
        finally:
            trading_system.disconnect_from_ib()
            print("✅ 系统已停止")
    else:
        print("❌ 无法连接到IB TWS")


if __name__ == "__main__":
    example_usage()