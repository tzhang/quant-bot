#!/usr/bin/env python3
"""
Interactive Brokers 市场数据实时获取模块
用于Citadel高频交易策略的实时数据流处理
"""

import time
import threading
import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
from collections import defaultdict, deque

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum

@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    bid_size: int = 0
    ask_size: int = 0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0

@dataclass
class BarData:
    """K线数据结构"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    wap: float = 0.0  # 加权平均价格
    count: int = 0    # 交易次数

class IBMarketDataStream(EWrapper, EClient):
    """IB市场数据流处理器"""
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        EClient.__init__(self, self)
        
        # 连接参数
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # 数据存储
        self.market_data: Dict[str, MarketData] = {}
        self.bar_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.tick_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # 订阅管理
        self.subscriptions: Dict[int, str] = {}  # req_id -> symbol
        self.contracts: Dict[str, Contract] = {}  # symbol -> contract
        self.req_id_counter = 1000
        
        # 回调函数
        self.data_callbacks: List[Callable] = []
        self.bar_callbacks: List[Callable] = []
        
        # 状态管理
        self.connected = False
        self.data_queue = queue.Queue()
        self.running = False
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
    def connect_to_ib(self) -> bool:
        """连接到IB TWS/Gateway"""
        try:
            self.logger.info(f"正在连接到IB TWS: {self.host}:{self.port} (客户端ID: {self.client_id})")
            self.connect(self.host, self.port, self.client_id)
            
            # 启动消息循环线程
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()
            
            # 等待连接建立
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if self.connected:
                self.logger.info(f"成功连接到IB TWS: {self.host}:{self.port}")
                return True
            else:
                self.logger.error("连接IB TWS超时")
                return False
                
        except Exception as e:
            self.logger.error(f"连接IB TWS失败: {e}")
            return False
    
    def disconnect_from_ib(self):
        """断开IB连接"""
        self.running = False
        if self.connected:
            self.disconnect()
            self.connected = False
            self.logger.info("已断开IB连接")
    
    # IB API 回调函数
    def connectAck(self):
        """连接确认回调"""
        self.logger.info("收到IB连接确认")
        
    def nextValidId(self, orderId):
        """接收下一个有效订单ID - 标志连接完全建立"""
        self.connected = True
        self.logger.info(f"IB市场数据流连接成功，下一个订单ID: {orderId}")
        
    def connectionClosed(self):
        """连接关闭回调"""
        self.connected = False
        self.logger.info("IB连接已关闭")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """错误处理"""
        if errorCode in [2104, 2106, 2158]:  # 信息性消息
            self.logger.info(f"IB信息: {errorString}")
        else:
            self.logger.error(f"IB错误 {errorCode}: {errorString}")
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """价格tick数据"""
        if reqId not in self.subscriptions:
            return
            
        symbol = self.subscriptions[reqId]
        timestamp = datetime.now()
        
        if symbol not in self.market_data:
            self.market_data[symbol] = MarketData(symbol=symbol, timestamp=timestamp)
        
        data = self.market_data[symbol]
        data.timestamp = timestamp
        
        # 更新价格数据
        if tickType == 1:  # BID
            data.bid = price
        elif tickType == 2:  # ASK
            data.ask = price
        elif tickType == 4:  # LAST
            data.last = price
        elif tickType == 6:  # HIGH
            data.high = price
        elif tickType == 7:  # LOW
            data.low = price
        elif tickType == 9:  # CLOSE
            data.close = price
        elif tickType == 14:  # OPEN
            data.open = price
            
        # 存储tick数据
        tick_info = {
            'timestamp': timestamp,
            'tick_type': TickTypeEnum.to_str(tickType),
            'price': price,
            'symbol': symbol
        }
        self.tick_data[symbol].append(tick_info)
        
        # 触发回调
        self._trigger_data_callbacks(symbol, data)
    
    def tickSize(self, reqId, tickType, size):
        """数量tick数据"""
        if reqId not in self.subscriptions:
            return
            
        symbol = self.subscriptions[reqId]
        timestamp = datetime.now()
        
        if symbol not in self.market_data:
            self.market_data[symbol] = MarketData(symbol=symbol, timestamp=timestamp)
        
        data = self.market_data[symbol]
        data.timestamp = timestamp
        
        # 更新数量数据
        if tickType == 0:  # BID_SIZE
            data.bid_size = size
        elif tickType == 3:  # ASK_SIZE
            data.ask_size = size
        elif tickType == 5:  # LAST_SIZE
            pass  # 可以添加last_size字段
        elif tickType == 8:  # VOLUME
            data.volume = size
    
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        """实时K线数据"""
        if reqId not in self.subscriptions:
            return
            
        symbol = self.subscriptions[reqId]
        timestamp = datetime.fromtimestamp(time)
        
        bar = BarData(
            symbol=symbol,
            timestamp=timestamp,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            wap=wap,
            count=count
        )
        
        # 存储K线数据
        self.bar_data[symbol].append(bar)
        
        # 触发K线回调
        self._trigger_bar_callbacks(symbol, bar)
    
    def subscribe_market_data(self, symbol: str, contract_type: str = 'STK', 
                            exchange: str = 'SMART', currency: str = 'USD') -> bool:
        """订阅市场数据"""
        try:
            # 创建合约
            contract = Contract()
            contract.symbol = symbol
            contract.secType = contract_type
            contract.exchange = exchange
            contract.currency = currency
            
            # 生成请求ID
            req_id = self.req_id_counter
            self.req_id_counter += 1
            
            # 存储订阅信息
            self.subscriptions[req_id] = symbol
            self.contracts[symbol] = contract
            
            # 订阅市场数据 - 使用延迟数据（15分钟延迟）
            # 先尝试请求延迟数据权限
            self.reqMarketDataType(3)  # 3 = 延迟数据
            
            # 订阅市场数据，使用空字符串让IB自动选择可用的数据类型
            self.reqMktData(req_id, contract, "", False, False, [])
            
            self.logger.info(f"已订阅 {symbol} 的市场数据")
            return True
            
        except Exception as e:
            self.logger.error(f"订阅市场数据失败 {symbol}: {e}")
            return False
    
    def subscribe_realtime_bars(self, symbol: str, bar_size: int = 5,
                              contract_type: str = 'STK', exchange: str = 'SMART', 
                              currency: str = 'USD') -> bool:
        """订阅实时K线数据"""
        try:
            if symbol not in self.contracts:
                # 创建合约
                contract = Contract()
                contract.symbol = symbol
                contract.secType = contract_type
                contract.exchange = exchange
                contract.currency = currency
                self.contracts[symbol] = contract
            else:
                contract = self.contracts[symbol]
            
            # 生成请求ID
            req_id = self.req_id_counter
            self.req_id_counter += 1
            
            # 存储订阅信息
            self.subscriptions[req_id] = symbol
            
            # 订阅实时K线 (5秒K线)
            self.reqRealTimeBars(req_id, contract, bar_size, "TRADES", True, [])
            
            self.logger.info(f"已订阅 {symbol} 的实时K线数据 ({bar_size}秒)")
            return True
            
        except Exception as e:
            self.logger.error(f"订阅实时K线失败 {symbol}: {e}")
            return False
    
    def unsubscribe_market_data(self, symbol: str):
        """取消订阅市场数据"""
        req_ids_to_remove = []
        for req_id, sub_symbol in self.subscriptions.items():
            if sub_symbol == symbol:
                self.cancelMktData(req_id)
                req_ids_to_remove.append(req_id)
        
        for req_id in req_ids_to_remove:
            del self.subscriptions[req_id]
        
        self.logger.info(f"已取消订阅 {symbol} 的市场数据")
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """获取最新市场数据"""
        return self.market_data.get(symbol)
    
    def get_bar_data(self, symbol: str, count: int = 100) -> List[BarData]:
        """获取K线数据"""
        if symbol in self.bar_data:
            bars = list(self.bar_data[symbol])
            return bars[-count:] if len(bars) > count else bars
        return []
    
    def get_tick_data(self, symbol: str, count: int = 1000) -> List[dict]:
        """获取tick数据"""
        if symbol in self.tick_data:
            ticks = list(self.tick_data[symbol])
            return ticks[-count:] if len(ticks) > count else ticks
        return []
    
    def add_data_callback(self, callback: Callable):
        """添加数据回调函数"""
        self.data_callbacks.append(callback)
    
    def add_bar_callback(self, callback: Callable):
        """添加K线回调函数"""
        self.bar_callbacks.append(callback)
    
    def _trigger_data_callbacks(self, symbol: str, data: MarketData):
        """触发数据回调"""
        for callback in self.data_callbacks:
            try:
                callback(symbol, data)
            except Exception as e:
                self.logger.error(f"数据回调错误: {e}")
    
    def _trigger_bar_callbacks(self, symbol: str, bar: BarData):
        """触发K线回调"""
        for callback in self.bar_callbacks:
            try:
                callback(symbol, bar)
            except Exception as e:
                self.logger.error(f"K线回调错误: {e}")
    
    def start_data_stream(self):
        """启动数据流"""
        self.running = True
        
        # 启动消息处理线程
        def run_loop():
            while self.running and self.connected:
                try:
                    self.run()
                except Exception as e:
                    self.logger.error(f"数据流处理错误: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        
        self.logger.info("数据流已启动")
    
    def stop_data_stream(self):
        """停止数据流"""
        self.running = False
        self.logger.info("数据流已停止")
    
    def get_market_summary(self) -> Dict:
        """获取市场数据摘要"""
        summary = {}
        for symbol, data in self.market_data.items():
            summary[symbol] = {
                'last_price': data.last,
                'bid': data.bid,
                'ask': data.ask,
                'spread': data.ask - data.bid if data.ask > 0 and data.bid > 0 else 0,
                'volume': data.volume,
                'timestamp': data.timestamp.isoformat(),
                'high': data.high,
                'low': data.low,
                'open': data.open,
                'close': data.close
            }
        return summary

# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建数据流实例
    data_stream = IBMarketDataStream()
    
    # 数据回调函数
    def on_market_data(symbol: str, data: MarketData):
        print(f"[{data.timestamp.strftime('%H:%M:%S')}] {symbol}: "
              f"Last={data.last:.2f}, Bid={data.bid:.2f}, Ask={data.ask:.2f}, "
              f"Volume={data.volume}")
    
    def on_bar_data(symbol: str, bar: BarData):
        print(f"[{bar.timestamp.strftime('%H:%M:%S')}] {symbol} Bar: "
              f"O={bar.open:.2f}, H={bar.high:.2f}, L={bar.low:.2f}, "
              f"C={bar.close:.2f}, V={bar.volume}")
    
    # 添加回调
    data_stream.add_data_callback(on_market_data)
    data_stream.add_bar_callback(on_bar_data)
    
    try:
        # 连接到IB
        if data_stream.connect_to_ib():
            # 启动数据流
            data_stream.start_data_stream()
            
            # 订阅数据
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
            for symbol in symbols:
                data_stream.subscribe_market_data(symbol)
                data_stream.subscribe_realtime_bars(symbol)
            
            print("数据流已启动，按Ctrl+C停止...")
            
            # 运行数据流
            while True:
                time.sleep(1)
                
                # 每10秒打印市场摘要
                if int(time.time()) % 10 == 0:
                    summary = data_stream.get_market_summary()
                    if summary:
                        print("\n=== 市场数据摘要 ===")
                        for symbol, info in summary.items():
                            print(f"{symbol}: Last={info['last_price']:.2f}, "
                                  f"Spread={info['spread']:.4f}, Volume={info['volume']}")
                        print("=" * 30)
                
    except KeyboardInterrupt:
        print("\n正在停止数据流...")
    finally:
        data_stream.stop_data_stream()
        data_stream.disconnect_from_ib()
        print("数据流已停止")