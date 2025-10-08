#!/usr/bin/env python3
"""
Interactive Brokers 订单管理系统
提供完整的订单生命周期管理功能
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import json
import uuid

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.execution import Execution
from ibapi.commission_report import CommissionReport

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """订单类型"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAILING_STOP = "TRAIL"
    BRACKET = "BRACKET"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"

class TimeInForce(Enum):
    """订单有效期"""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # 高级订单参数
    parent_id: Optional[int] = None
    oca_group: Optional[str] = None  # One-Cancels-All group
    transmit: bool = True
    
    # 风险参数
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_amount: Optional[float] = None
    
    # 元数据
    strategy_id: Optional[str] = None
    notes: Optional[str] = None
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class OrderInfo:
    """订单信息"""
    order_id: int
    client_order_id: str
    symbol: str
    action: str
    quantity: int
    order_type: OrderType
    status: OrderStatus
    
    # 价格信息
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    avg_fill_price: Optional[float] = None
    
    # 执行信息
    filled_quantity: int = 0
    remaining_quantity: int = 0
    commission: float = 0.0
    
    # 时间信息
    submit_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    
    # 关联信息
    parent_id: Optional[int] = None
    child_orders: List[int] = field(default_factory=list)
    
    # 元数据
    strategy_id: Optional[str] = None
    notes: Optional[str] = None
    
    # 错误信息
    error_code: Optional[int] = None
    error_message: Optional[str] = None

@dataclass
class ExecutionInfo:
    """执行信息"""
    execution_id: str
    order_id: int
    symbol: str
    side: str
    quantity: int
    price: float
    time: datetime
    exchange: str
    commission: float = 0.0
    realized_pnl: float = 0.0

class IBOrderManager(EWrapper, EClient):
    """IB订单管理器"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        EClient.__init__(self, self)
        
        # 连接参数
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # 订单数据
        self.orders: Dict[int, OrderInfo] = {}
        self.executions: Dict[str, ExecutionInfo] = {}
        self.next_order_id = 1
        
        # 订单映射
        self.client_order_map: Dict[str, int] = {}  # client_order_id -> order_id
        self.strategy_orders: Dict[str, List[int]] = defaultdict(list)  # strategy_id -> order_ids
        
        # 回调函数
        self.order_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.execution_callbacks: List[Callable] = []
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 连接状态
        self.connected = False
        
        # 风险管理器（可选）
        self.risk_manager = None
        
    def connect_to_ib(self) -> bool:
        """连接到IB"""
        try:
            self.connect(self.host, self.port, self.client_id)
            
            # 等待连接
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                logger.info(f"成功连接到IB: {self.host}:{self.port}")
                return True
            else:
                logger.error("连接IB超时")
                return False
                
        except Exception as e:
            logger.error(f"连接IB失败: {e}")
            return False
    
    def disconnect_from_ib(self):
        """断开IB连接"""
        if self.connected:
            self.disconnect()
            self.connected = False
            logger.info("已断开IB连接")
    
    def set_risk_manager(self, risk_manager):
        """设置风险管理器"""
        self.risk_manager = risk_manager
    
    def submit_order(self, request: OrderRequest) -> Optional[int]:
        """提交订单"""
        try:
            # 风险检查
            if self.risk_manager:
                price = request.price or 0.0
                allow, alerts = self.risk_manager.check_order_risk(
                    request.symbol, request.action, request.quantity, price
                )
                
                if not allow:
                    logger.error(f"订单被风险管理器拒绝: {request.symbol}")
                    for alert in alerts:
                        logger.error(f"  {alert.message}")
                    return None
            
            # 创建合约
            contract = self._create_contract(request.symbol)
            
            # 创建订单
            order = self._create_order(request)
            
            # 获取订单ID
            order_id = self.next_order_id
            
            # 创建订单信息
            order_info = OrderInfo(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                action=request.action,
                quantity=request.quantity,
                order_type=request.order_type,
                status=OrderStatus.PENDING,
                limit_price=request.price,
                stop_price=request.stop_price,
                remaining_quantity=request.quantity,
                submit_time=datetime.now(),
                parent_id=request.parent_id,
                strategy_id=request.strategy_id,
                notes=request.notes
            )
            
            # 保存订单信息
            with self._lock:
                self.orders[order_id] = order_info
                self.client_order_map[request.client_order_id] = order_id
                
                if request.strategy_id:
                    self.strategy_orders[request.strategy_id].append(order_id)
            
            # 提交订单
            self.placeOrder(order_id, contract, order)
            
            logger.info(f"订单已提交: {order_id} - {request.symbol} {request.action} {request.quantity}")
            
            # 创建止损和止盈订单（如果指定）
            if request.stop_loss_price or request.take_profit_price:
                self._create_bracket_orders(order_id, request)
            
            return order_id
            
        except Exception as e:
            logger.error(f"提交订单失败: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """取消订单"""
        try:
            if order_id not in self.orders:
                logger.error(f"订单不存在: {order_id}")
                return False
            
            order_info = self.orders[order_id]
            if order_info.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"订单已完成或已取消: {order_id}")
                return False
            
            # 取消订单
            self.cancelOrder(order_id)
            
            logger.info(f"订单取消请求已发送: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """取消所有订单"""
        cancelled_count = 0
        
        with self._lock:
            active_orders = [
                order_id for order_id, order_info in self.orders.items()
                if order_info.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
            ]
        
        for order_id in active_orders:
            if self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info(f"已取消 {cancelled_count} 个订单")
        return cancelled_count
    
    def cancel_strategy_orders(self, strategy_id: str) -> int:
        """取消策略的所有订单"""
        cancelled_count = 0
        
        with self._lock:
            strategy_order_ids = self.strategy_orders.get(strategy_id, [])
        
        for order_id in strategy_order_ids:
            if order_id in self.orders:
                order_info = self.orders[order_id]
                if order_info.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                    if self.cancel_order(order_id):
                        cancelled_count += 1
        
        logger.info(f"已取消策略 {strategy_id} 的 {cancelled_count} 个订单")
        return cancelled_count
    
    def get_order_info(self, order_id: int) -> Optional[OrderInfo]:
        """获取订单信息"""
        return self.orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderInfo]:
        """根据客户端订单ID获取订单信息"""
        order_id = self.client_order_map.get(client_order_id)
        if order_id:
            return self.orders.get(order_id)
        return None
    
    def get_strategy_orders(self, strategy_id: str) -> List[OrderInfo]:
        """获取策略的所有订单"""
        order_ids = self.strategy_orders.get(strategy_id, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_active_orders(self) -> List[OrderInfo]:
        """获取活跃订单"""
        return [
            order_info for order_info in self.orders.values()
            if order_info.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
    
    def get_filled_orders(self, start_date: Optional[datetime] = None) -> List[OrderInfo]:
        """获取已成交订单"""
        filled_orders = [
            order_info for order_info in self.orders.values()
            if order_info.status == OrderStatus.FILLED
        ]
        
        if start_date:
            filled_orders = [
                order for order in filled_orders
                if order.fill_time and order.fill_time >= start_date
            ]
        
        return filled_orders
    
    def get_executions(self, symbol: Optional[str] = None, start_date: Optional[datetime] = None) -> List[ExecutionInfo]:
        """获取执行记录"""
        executions = list(self.executions.values())
        
        if symbol:
            executions = [exec for exec in executions if exec.symbol == symbol]
        
        if start_date:
            executions = [exec for exec in executions if exec.time >= start_date]
        
        return executions
    
    def add_order_callback(self, event_type: str, callback: Callable):
        """添加订单回调"""
        self.order_callbacks[event_type].append(callback)
    
    def add_execution_callback(self, callback: Callable):
        """添加执行回调"""
        self.execution_callbacks.append(callback)
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """获取订单统计"""
        with self._lock:
            total_orders = len(self.orders)
            filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
            cancelled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
            rejected_orders = len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED])
            active_orders = len(self.get_active_orders())
            
            total_volume = sum(o.filled_quantity for o in self.orders.values())
            total_commission = sum(o.commission for o in self.orders.values())
            
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'rejected_orders': rejected_orders,
            'active_orders': active_orders,
            'fill_rate': filled_orders / max(total_orders, 1),
            'total_volume': total_volume,
            'total_commission': total_commission,
            'avg_commission_per_order': total_commission / max(filled_orders, 1)
        }
    
    # ========== IB API 回调方法 ==========
    
    def connectAck(self):
        """连接确认"""
        self.connected = True
        logger.info("IB连接已确认")
    
    def nextValidId(self, orderId: int):
        """下一个有效订单ID"""
        self.next_order_id = orderId
        logger.info(f"下一个有效订单ID: {orderId}")
    
    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                   avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                   clientId: int, whyHeld: str, mktCapPrice: float):
        """订单状态更新"""
        if orderId in self.orders:
            order_info = self.orders[orderId]
            
            # 更新状态
            old_status = order_info.status
            if status == "Submitted":
                order_info.status = OrderStatus.SUBMITTED
            elif status == "Filled":
                order_info.status = OrderStatus.FILLED
                order_info.fill_time = datetime.now()
            elif status == "Cancelled":
                order_info.status = OrderStatus.CANCELLED
            elif status == "Rejected":
                order_info.status = OrderStatus.REJECTED
            elif status == "PreSubmitted":
                order_info.status = OrderStatus.PENDING
            elif filled > 0 and remaining > 0:
                order_info.status = OrderStatus.PARTIALLY_FILLED
            
            # 更新执行信息
            order_info.filled_quantity = int(filled)
            order_info.remaining_quantity = int(remaining)
            order_info.avg_fill_price = avgFillPrice if avgFillPrice > 0 else None
            
            logger.info(f"订单状态更新: {orderId} - {status} (成交: {filled}, 剩余: {remaining})")
            
            # 触发回调
            if old_status != order_info.status:
                self._trigger_order_callbacks("status_change", order_info)
    
    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState):
        """开放订单"""
        if orderId in self.orders:
            order_info = self.orders[orderId]
            
            # 更新订单信息
            if hasattr(orderState, 'status'):
                if orderState.status == "Submitted":
                    order_info.status = OrderStatus.SUBMITTED
            
            logger.debug(f"开放订单: {orderId} - {contract.symbol}")
    
    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        """执行详情"""
        exec_info = ExecutionInfo(
            execution_id=execution.execId,
            order_id=execution.orderId,
            symbol=contract.symbol,
            side=execution.side,
            quantity=int(execution.shares),
            price=execution.price,
            time=datetime.strptime(execution.time, "%Y%m%d  %H:%M:%S"),
            exchange=execution.exchange
        )
        
        self.executions[execution.execId] = exec_info
        
        logger.info(f"执行详情: {execution.execId} - {contract.symbol} {execution.side} {execution.shares}@{execution.price}")
        
        # 触发执行回调
        for callback in self.execution_callbacks:
            try:
                callback(exec_info)
            except Exception as e:
                logger.error(f"执行回调错误: {e}")
    
    def commissionReport(self, commissionReport: CommissionReport):
        """佣金报告"""
        exec_id = commissionReport.execId
        if exec_id in self.executions:
            exec_info = self.executions[exec_id]
            exec_info.commission = commissionReport.commission
            exec_info.realized_pnl = commissionReport.realizedPNL or 0.0
            
            # 更新订单佣金
            order_id = exec_info.order_id
            if order_id in self.orders:
                self.orders[order_id].commission += commissionReport.commission
            
            logger.info(f"佣金报告: {exec_id} - 佣金: ${commissionReport.commission:.2f}")
    
    def error(self, reqId: int, errorCode: int, errorString: str):
        """错误处理"""
        if reqId > 0 and reqId in self.orders:
            order_info = self.orders[reqId]
            order_info.error_code = errorCode
            order_info.error_message = errorString
            
            if errorCode in [201, 202]:  # 订单被拒绝
                order_info.status = OrderStatus.REJECTED
                self._trigger_order_callbacks("rejected", order_info)
        
        logger.error(f"IB错误 [{errorCode}]: {errorString}")
    
    # ========== 私有方法 ==========
    
    def _create_contract(self, symbol: str) -> Contract:
        """创建合约"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
    
    def _create_order(self, request: OrderRequest) -> Order:
        """创建订单"""
        order = Order()
        order.action = request.action
        order.totalQuantity = request.quantity
        order.orderType = request.order_type.value
        order.tif = request.time_in_force.value
        order.transmit = request.transmit
        
        if request.price:
            order.lmtPrice = request.price
        
        if request.stop_price:
            order.auxPrice = request.stop_price
        
        if request.parent_id:
            order.parentId = request.parent_id
        
        if request.oca_group:
            order.ocaGroup = request.oca_group
        
        if request.trailing_stop_amount:
            order.trailStopPrice = request.trailing_stop_amount
        
        return order
    
    def _create_bracket_orders(self, parent_order_id: int, request: OrderRequest):
        """创建括号订单（止损和止盈）"""
        parent_order = self.orders[parent_order_id]
        
        # 创建止损订单
        if request.stop_loss_price:
            stop_loss_request = OrderRequest(
                symbol=request.symbol,
                action="SELL" if request.action == "BUY" else "BUY",
                quantity=request.quantity,
                order_type=OrderType.STOP,
                stop_price=request.stop_loss_price,
                parent_id=parent_order_id,
                strategy_id=request.strategy_id,
                notes=f"止损订单 for {parent_order_id}",
                transmit=False
            )
            
            stop_loss_id = self.submit_order(stop_loss_request)
            if stop_loss_id:
                parent_order.child_orders.append(stop_loss_id)
        
        # 创建止盈订单
        if request.take_profit_price:
            take_profit_request = OrderRequest(
                symbol=request.symbol,
                action="SELL" if request.action == "BUY" else "BUY",
                quantity=request.quantity,
                order_type=OrderType.LIMIT,
                price=request.take_profit_price,
                parent_id=parent_order_id,
                strategy_id=request.strategy_id,
                notes=f"止盈订单 for {parent_order_id}",
                transmit=True
            )
            
            take_profit_id = self.submit_order(take_profit_request)
            if take_profit_id:
                parent_order.child_orders.append(take_profit_id)
    
    def _trigger_order_callbacks(self, event_type: str, order_info: OrderInfo):
        """触发订单回调"""
        for callback in self.order_callbacks[event_type]:
            try:
                callback(order_info)
            except Exception as e:
                logger.error(f"订单回调错误: {e}")


def example_usage():
    """使用示例"""
    # 创建订单管理器
    order_manager = IBOrderManager()
    
    # 连接到IB
    if not order_manager.connect_to_ib():
        print("连接失败")
        return
    
    try:
        # 添加订单状态回调
        def on_order_filled(order_info: OrderInfo):
            print(f"订单成交: {order_info.order_id} - {order_info.symbol}")
        
        order_manager.add_order_callback("status_change", on_order_filled)
        
        # 提交市价订单
        market_order = OrderRequest(
            symbol="AAPL",
            action="BUY",
            quantity=100,
            order_type=OrderType.MARKET,
            strategy_id="test_strategy"
        )
        
        order_id = order_manager.submit_order(market_order)
        if order_id:
            print(f"市价订单已提交: {order_id}")
        
        # 提交限价订单（带止损和止盈）
        limit_order = OrderRequest(
            symbol="MSFT",
            action="BUY",
            quantity=50,
            order_type=OrderType.LIMIT,
            price=300.0,
            stop_loss_price=290.0,
            take_profit_price=320.0,
            strategy_id="test_strategy"
        )
        
        order_id2 = order_manager.submit_order(limit_order)
        if order_id2:
            print(f"限价订单已提交: {order_id2}")
        
        # 等待一段时间
        time.sleep(10)
        
        # 获取订单统计
        stats = order_manager.get_order_statistics()
        print(f"\n订单统计:")
        print(f"  总订单数: {stats['total_orders']}")
        print(f"  成交订单数: {stats['filled_orders']}")
        print(f"  活跃订单数: {stats['active_orders']}")
        print(f"  成交率: {stats['fill_rate']:.2%}")
        
        # 获取活跃订单
        active_orders = order_manager.get_active_orders()
        print(f"\n活跃订单:")
        for order in active_orders:
            print(f"  {order.order_id}: {order.symbol} {order.action} {order.quantity} - {order.status.value}")
        
    finally:
        # 取消所有订单并断开连接
        order_manager.cancel_all_orders()
        time.sleep(2)
        order_manager.disconnect_from_ib()


if __name__ == "__main__":
    example_usage()