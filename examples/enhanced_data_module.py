"""
增强数据模块 - 提供高级数据处理、存储和分析功能
支持多种数据源、实时数据流处理、历史数据管理和高级分析功能
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path
import h5py
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketDataPoint:
    """市场数据点"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_size: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None

@dataclass
class BarData:
    """K线数据"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trade_count: Optional[int] = None

@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    size: int
    orders: int = 1

@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

class DataStorage:
    """数据存储管理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.data_dir / "tick").mkdir(exist_ok=True)
        (self.data_dir / "bars").mkdir(exist_ok=True)
        (self.data_dir / "orderbook").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
    def _init_database(self):
        """初始化SQLite数据库"""
        self.db_path = self.data_dir / "market_data.db"
        with sqlite3.connect(self.db_path) as conn:
            # 创建tick数据表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    bid REAL,
                    ask REAL,
                    bid_size INTEGER,
                    ask_size INTEGER,
                    last_size INTEGER,
                    high REAL,
                    low REAL,
                    open REAL,
                    close REAL,
                    UNIQUE(timestamp, symbol)
                )
            """)
            
            # 创建K线数据表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bar_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    vwap REAL,
                    trade_count INTEGER,
                    UNIQUE(timestamp, symbol, timeframe)
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tick_symbol_time ON tick_data(symbol, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bar_symbol_time ON bar_data(symbol, timestamp, timeframe)")
            
    def store_tick_data(self, data: MarketDataPoint):
        """存储tick数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tick_data 
                    (timestamp, symbol, price, volume, bid, ask, bid_size, ask_size, 
                     last_size, high, low, open, close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.timestamp.isoformat(),
                    data.symbol,
                    data.price,
                    data.volume,
                    data.bid,
                    data.ask,
                    data.bid_size,
                    data.ask_size,
                    data.last_size,
                    data.high,
                    data.low,
                    data.open,
                    data.close
                ))
        except Exception as e:
            self.logger.error(f"存储tick数据失败: {e}")
            
    def store_bar_data(self, data: BarData, timeframe: str = "1min"):
        """存储K线数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO bar_data 
                    (timestamp, symbol, timeframe, open, high, low, close, volume, vwap, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.timestamp.isoformat(),
                    data.symbol,
                    timeframe,
                    data.open,
                    data.high,
                    data.low,
                    data.close,
                    data.volume,
                    data.vwap,
                    data.trade_count
                ))
        except Exception as e:
            self.logger.error(f"存储K线数据失败: {e}")
            
    def get_tick_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """获取tick数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM tick_data 
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """
                df = pd.read_sql_query(query, conn, params=(
                    symbol, start_time.isoformat(), end_time.isoformat()
                ))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            self.logger.error(f"获取tick数据失败: {e}")
            return pd.DataFrame()
            
    def get_bar_data(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """获取K线数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM bar_data 
                    WHERE symbol = ? AND timeframe = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """
                df = pd.read_sql_query(query, conn, params=(
                    symbol, timeframe, start_time.isoformat(), end_time.isoformat()
                ))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()

class RealTimeDataProcessor:
    """实时数据处理器"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # 数据缓存
        self.tick_cache = defaultdict(lambda: deque(maxlen=1000))
        self.bar_cache = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        
        # 数据处理队列
        self.data_queue = queue.Queue(maxsize=10000)
        self.processing_thread = None
        self.running = False
        
        # 回调函数
        self.tick_callbacks = []
        self.bar_callbacks = []
        
        # 数据聚合器
        self.bar_aggregators = defaultdict(lambda: defaultdict(dict))
        
    def start(self):
        """启动数据处理"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        self.logger.info("实时数据处理器已启动")
        
    def stop(self):
        """停止数据处理"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.logger.info("实时数据处理器已停止")
        
    def add_tick_callback(self, callback: Callable[[MarketDataPoint], None]):
        """添加tick数据回调"""
        self.tick_callbacks.append(callback)
        
    def add_bar_callback(self, callback: Callable[[BarData, str], None]):
        """添加K线数据回调"""
        self.bar_callbacks.append(callback)
        
    def process_tick(self, data: MarketDataPoint):
        """处理tick数据"""
        try:
            self.data_queue.put(('tick', data), timeout=1)
        except queue.Full:
            self.logger.warning("数据队列已满，丢弃tick数据")
            
    def process_bar(self, data: BarData, timeframe: str = "1min"):
        """处理K线数据"""
        try:
            self.data_queue.put(('bar', data, timeframe), timeout=1)
        except queue.Full:
            self.logger.warning("数据队列已满，丢弃K线数据")
            
    def _processing_loop(self):
        """数据处理主循环"""
        while self.running:
            try:
                item = self.data_queue.get(timeout=1)
                
                if item[0] == 'tick':
                    self._process_tick_data(item[1])
                elif item[0] == 'bar':
                    self._process_bar_data(item[1], item[2])
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"数据处理错误: {e}")
                
    def _process_tick_data(self, data: MarketDataPoint):
        """处理单个tick数据"""
        # 存储到缓存
        self.tick_cache[data.symbol].append(data)
        
        # 存储到数据库
        self.storage.store_tick_data(data)
        
        # 聚合成K线
        self._aggregate_to_bars(data)
        
        # 调用回调函数
        for callback in self.tick_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Tick回调函数错误: {e}")
                
    def _process_bar_data(self, data: BarData, timeframe: str):
        """处理单个K线数据"""
        # 存储到缓存
        self.bar_cache[data.symbol][timeframe].append(data)
        
        # 存储到数据库
        self.storage.store_bar_data(data, timeframe)
        
        # 调用回调函数
        for callback in self.bar_callbacks:
            try:
                callback(data, timeframe)
            except Exception as e:
                self.logger.error(f"Bar回调函数错误: {e}")
                
    def _aggregate_to_bars(self, tick: MarketDataPoint):
        """将tick数据聚合成K线"""
        symbol = tick.symbol
        timestamp = tick.timestamp
        
        # 1分钟K线聚合
        minute_key = timestamp.replace(second=0, microsecond=0)
        
        if symbol not in self.bar_aggregators:
            self.bar_aggregators[symbol] = defaultdict(dict)
            
        if minute_key not in self.bar_aggregators[symbol]['1min']:
            # 新的1分钟K线
            self.bar_aggregators[symbol]['1min'][minute_key] = {
                'open': tick.price,
                'high': tick.price,
                'low': tick.price,
                'close': tick.price,
                'volume': tick.volume,
                'trade_count': 1,
                'vwap_sum': tick.price * tick.volume,
                'volume_sum': tick.volume
            }
        else:
            # 更新现有K线
            bar = self.bar_aggregators[symbol]['1min'][minute_key]
            bar['high'] = max(bar['high'], tick.price)
            bar['low'] = min(bar['low'], tick.price)
            bar['close'] = tick.price
            bar['volume'] += tick.volume
            bar['trade_count'] += 1
            bar['vwap_sum'] += tick.price * tick.volume
            bar['volume_sum'] += tick.volume
            
        # 检查是否需要完成K线
        current_minute = datetime.now().replace(second=0, microsecond=0)
        if minute_key < current_minute:
            self._finalize_bar(symbol, '1min', minute_key)
            
    def _finalize_bar(self, symbol: str, timeframe: str, timestamp: datetime):
        """完成K线数据"""
        if symbol in self.bar_aggregators and timeframe in self.bar_aggregators[symbol]:
            if timestamp in self.bar_aggregators[symbol][timeframe]:
                bar_data = self.bar_aggregators[symbol][timeframe][timestamp]
                
                # 计算VWAP
                vwap = bar_data['vwap_sum'] / bar_data['volume_sum'] if bar_data['volume_sum'] > 0 else bar_data['close']
                
                # 创建BarData对象
                bar = BarData(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=bar_data['open'],
                    high=bar_data['high'],
                    low=bar_data['low'],
                    close=bar_data['close'],
                    volume=bar_data['volume'],
                    vwap=vwap,
                    trade_count=bar_data['trade_count']
                )
                
                # 处理完成的K线
                self.process_bar(bar, timeframe)
                
                # 清理已完成的K线
                del self.bar_aggregators[symbol][timeframe][timestamp]
                
    def get_latest_tick(self, symbol: str) -> Optional[MarketDataPoint]:
        """获取最新tick数据"""
        if symbol in self.tick_cache and self.tick_cache[symbol]:
            return self.tick_cache[symbol][-1]
        return None
        
    def get_latest_bar(self, symbol: str, timeframe: str = "1min") -> Optional[BarData]:
        """获取最新K线数据"""
        if symbol in self.bar_cache and timeframe in self.bar_cache[symbol]:
            if self.bar_cache[symbol][timeframe]:
                return self.bar_cache[symbol][timeframe][-1]
        return None
        
    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[MarketDataPoint]:
        """获取最近的tick数据"""
        if symbol in self.tick_cache:
            return list(self.tick_cache[symbol])[-count:]
        return []
        
    def get_recent_bars(self, symbol: str, timeframe: str = "1min", count: int = 50) -> List[BarData]:
        """获取最近的K线数据"""
        if symbol in self.bar_cache and timeframe in self.bar_cache[symbol]:
            return list(self.bar_cache[symbol][timeframe])[-count:]
        return []

class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, processor: RealTimeDataProcessor):
        self.processor = processor
        self.logger = logging.getLogger(__name__)
        
    def calculate_technical_indicators(self, symbol: str, timeframe: str = "1min") -> Dict[str, Any]:
        """计算技术指标"""
        bars = self.processor.get_recent_bars(symbol, timeframe, 100)
        if len(bars) < 20:
            return {}
            
        # 转换为DataFrame
        df = pd.DataFrame([asdict(bar) for bar in bars])
        df.set_index('timestamp', inplace=True)
        
        indicators = {}
        
        try:
            # 移动平均线
            indicators['sma_5'] = df['close'].rolling(5).mean().iloc[-1]
            indicators['sma_10'] = df['close'].rolling(10).mean().iloc[-1]
            indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
            
            # 指数移动平均线
            indicators['ema_5'] = df['close'].ewm(span=5).mean().iloc[-1]
            indicators['ema_10'] = df['close'].ewm(span=10).mean().iloc[-1]
            indicators['ema_20'] = df['close'].ewm(span=20).mean().iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_histogram'] = (macd - signal).iloc[-1]
            
            # 布林带
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            indicators['bb_upper'] = (sma20 + 2 * std20).iloc[-1]
            indicators['bb_middle'] = sma20.iloc[-1]
            indicators['bb_lower'] = (sma20 - 2 * std20).iloc[-1]
            
            # 成交量指标
            indicators['volume_sma'] = df['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            
        return indicators
        
    def calculate_market_microstructure(self, symbol: str) -> Dict[str, Any]:
        """计算市场微观结构指标"""
        ticks = self.processor.get_recent_ticks(symbol, 1000)
        if len(ticks) < 100:
            return {}
            
        microstructure = {}
        
        try:
            # 价格序列
            prices = [tick.price for tick in ticks]
            volumes = [tick.volume for tick in ticks]
            
            # 价格波动率
            returns = np.diff(np.log(prices))
            microstructure['volatility'] = np.std(returns) * np.sqrt(252 * 24 * 60)  # 年化波动率
            
            # 买卖价差
            spreads = []
            for tick in ticks:
                if tick.bid and tick.ask:
                    spreads.append(tick.ask - tick.bid)
            if spreads:
                microstructure['avg_spread'] = np.mean(spreads)
                microstructure['spread_volatility'] = np.std(spreads)
            
            # 成交量加权平均价格偏差
            if len(ticks) >= 20:
                recent_prices = prices[-20:]
                recent_volumes = volumes[-20:]
                vwap = np.average(recent_prices, weights=recent_volumes)
                microstructure['price_vwap_deviation'] = (prices[-1] - vwap) / vwap
            
            # 价格冲击
            if len(returns) >= 10:
                microstructure['price_impact'] = np.mean(np.abs(returns[-10:]))
            
        except Exception as e:
            self.logger.error(f"计算市场微观结构指标失败: {e}")
            
        return microstructure

class EnhancedDataModule:
    """增强数据模块主类"""
    
    def __init__(self, data_dir: str = "data"):
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.storage = DataStorage(data_dir)
        self.processor = RealTimeDataProcessor(self.storage)
        self.analyzer = DataAnalyzer(self.processor)
        
        # 数据统计
        self.stats = {
            'ticks_processed': 0,
            'bars_processed': 0,
            'start_time': datetime.now()
        }
        
    def start(self):
        """启动数据模块"""
        self.processor.start()
        self.logger.info("增强数据模块已启动")
        
    def stop(self):
        """停止数据模块"""
        self.processor.stop()
        self.logger.info("增强数据模块已停止")
        
    def process_market_data(self, symbol: str, tick_data: Dict):
        """处理市场数据"""
        try:
            # 转换为MarketDataPoint
            data_point = MarketDataPoint(
                timestamp=datetime.now(),
                symbol=symbol,
                price=tick_data.get('price', 0.0),
                volume=tick_data.get('volume', 0),
                bid=tick_data.get('bid'),
                ask=tick_data.get('ask'),
                bid_size=tick_data.get('bid_size'),
                ask_size=tick_data.get('ask_size'),
                last_size=tick_data.get('last_size'),
                high=tick_data.get('high'),
                low=tick_data.get('low'),
                open=tick_data.get('open'),
                close=tick_data.get('close')
            )
            
            # 处理数据
            self.processor.process_tick(data_point)
            self.stats['ticks_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"处理市场数据失败: {e}")
            
    def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """获取市场摘要"""
        try:
            summary = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'latest_tick': None,
                'latest_bar': None,
                'technical_indicators': {},
                'microstructure': {}
            }
            
            # 最新数据
            latest_tick = self.processor.get_latest_tick(symbol)
            if latest_tick:
                summary['latest_tick'] = asdict(latest_tick)
                
            latest_bar = self.processor.get_latest_bar(symbol)
            if latest_bar:
                summary['latest_bar'] = asdict(latest_bar)
                
            # 技术指标
            summary['technical_indicators'] = self.analyzer.calculate_technical_indicators(symbol)
            
            # 市场微观结构
            summary['microstructure'] = self.analyzer.calculate_market_microstructure(symbol)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取市场摘要失败: {e}")
            return {}
            
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据模块统计信息"""
        runtime = datetime.now() - self.stats['start_time']
        return {
            'runtime_seconds': runtime.total_seconds(),
            'ticks_processed': self.stats['ticks_processed'],
            'bars_processed': self.stats['bars_processed'],
            'ticks_per_second': self.stats['ticks_processed'] / max(runtime.total_seconds(), 1),
            'bars_per_second': self.stats['bars_processed'] / max(runtime.total_seconds(), 1)
        }