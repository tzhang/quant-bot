"""
增强版回测引擎

支持多资产、多策略、事件驱动回测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
from functools import lru_cache
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from .adaptive_execution_strategy import AdaptiveExecutionStrategy, ExecutionStrategy, TaskMetrics

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存数据"""
        if len(self.cache) >= self.max_size:
            # 删除最久未访问的数据
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.cache = DataCache()
        
    def preprocess_market_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """预处理市场数据"""
        cache_key = f"{symbol}_{hash(str(data.index[0]))}_{hash(str(data.index[-1]))}"
        
        # 检查缓存
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # 数据预处理
        processed_data = data.copy()
        
        # 确保数据按时间排序
        processed_data = processed_data.sort_index()
        
        # 填充缺失值
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        
        # 添加常用技术指标（预计算以提高性能）
        processed_data = self._add_technical_indicators(processed_data)
        
        # 缓存结果
        self.cache.set(cache_key, processed_data)
        
        return processed_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # 移动平均线
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # 指数移动平均线
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # 成交量移动平均
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        
        return data
    
    def batch_preprocess(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """批量预处理市场数据"""
        processed_data = {}
        
        # 使用多线程并行处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for symbol, data in market_data.items():
                future = executor.submit(self.preprocess_market_data, data, symbol)
                futures[symbol] = future
            
            for symbol, future in futures.items():
                processed_data[symbol] = future.result()
        
        return processed_data


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class EventType(Enum):
    """事件类型"""
    MARKET_DATA = "market_data"
    ORDER = "order"
    FILL = "fill"
    SIGNAL = "signal"
    REBALANCE = "rebalance"
    RISK_CHECK = "risk_check"


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    strategy_id: Optional[str] = None
    
    @property
    def remaining_quantity(self) -> float:
        """剩余数量"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """是否完全成交"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """是否活跃订单"""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]


@dataclass
class Fill:
    """成交记录"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    strategy_id: Optional[str] = None


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """市值"""
        return self.quantity * self.last_price
    
    @property
    def unrealized_pnl(self) -> float:
        """未实现盈亏"""
        if self.quantity == 0:
            return 0.0
        return (self.last_price - self.avg_cost) * self.quantity
    
    def update_price(self, price: float):
        """更新价格"""
        self.last_price = price
    
    def add_fill(self, fill: Fill):
        """添加成交"""
        if fill.side == OrderSide.BUY:
            # 买入
            total_cost = self.quantity * self.avg_cost + fill.quantity * fill.price
            self.quantity += fill.quantity
            if self.quantity > 0:
                self.avg_cost = total_cost / self.quantity
        else:
            # 卖出
            if self.quantity > 0:
                self.realized_pnl += (fill.price - self.avg_cost) * fill.quantity
            self.quantity -= fill.quantity
            
            # 如果持仓为0，重置平均成本
            if abs(self.quantity) < 1e-8:
                self.quantity = 0.0
                self.avg_cost = 0.0


@dataclass
class Event:
    """事件"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 0  # 优先级，数字越小优先级越高
    
    def __lt__(self, other):
        """用于优先队列排序"""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority


class EventHandler(ABC):
    """事件处理器抽象类"""
    
    @abstractmethod
    def handle_event(self, event: Event) -> List[Event]:
        """处理事件，返回新产生的事件列表"""
        pass


class MarketDataHandler(EventHandler):
    """市场数据处理器"""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
    
    def handle_event(self, event: Event) -> List[Event]:
        """处理市场数据事件"""
        if event.event_type != EventType.MARKET_DATA:
            return []
        
        # 更新市场数据
        market_data = event.data
        self.backtest_engine.portfolio.update_market_data(market_data)
        
        # 生成信号事件
        signal_events = []
        for strategy in self.backtest_engine.strategies.values():
            signals = strategy.generate_signals(market_data, event.timestamp)
            for signal in signals:
                signal_event = Event(
                    event_type=EventType.SIGNAL,
                    timestamp=event.timestamp,
                    data=signal,
                    priority=1
                )
                signal_events.append(signal_event)
        
        return signal_events


class SignalHandler(EventHandler):
    """信号处理器"""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
    
    def handle_event(self, event: Event) -> List[Event]:
        """处理信号事件"""
        if event.event_type != EventType.SIGNAL:
            return []
        
        signal_data = event.data
        strategy_id = signal_data.get('strategy_id')
        
        # 生成订单
        orders = self.backtest_engine.generate_orders_from_signal(signal_data, event.timestamp)
        
        order_events = []
        for order in orders:
            order_event = Event(
                event_type=EventType.ORDER,
                timestamp=event.timestamp,
                data={'order': order},
                priority=2
            )
            order_events.append(order_event)
        
        return order_events


class OrderHandler(EventHandler):
    """订单处理器"""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
    
    def handle_event(self, event: Event) -> List[Event]:
        """处理订单事件"""
        if event.event_type != EventType.ORDER:
            return []
        
        order = event.data['order']
        
        # 执行订单
        fills = self.backtest_engine.execute_order(order, event.timestamp)
        
        fill_events = []
        for fill in fills:
            fill_event = Event(
                event_type=EventType.FILL,
                timestamp=event.timestamp,
                data={'fill': fill},
                priority=3
            )
            fill_events.append(fill_event)
        
        return fill_events


class FillHandler(EventHandler):
    """成交处理器"""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
    
    def handle_event(self, event: Event) -> List[Event]:
        """处理成交事件"""
        if event.event_type != EventType.FILL:
            return []
        
        fill = event.data['fill']
        
        # 更新持仓
        self.backtest_engine.update_position(fill)
        
        # 更新账户
        self.backtest_engine.update_account(fill)
        
        return []


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, strategy_id: str, name: str):
        self.strategy_id = strategy_id
        self.name = name
        self.logger = logger
        
        # 策略状态
        self.is_active = True
        self.positions: Dict[str, float] = {}
        self.signals_history: List[Dict] = []
        
        # 策略参数
        self.parameters: Dict[str, Any] = {}
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        timestamp: datetime) -> List[Dict[str, Any]]:
        """生成交易信号"""
        pass
    
    def set_parameter(self, key: str, value: Any):
        """设置策略参数"""
        self.parameters[key] = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """获取策略参数"""
        return self.parameters.get(key, default)
    
    def update_position(self, symbol: str, quantity: float):
        """更新策略持仓"""
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
    
    def get_position(self, symbol: str) -> float:
        """获取策略持仓"""
        return self.positions.get(symbol, 0)


class SimpleMovingAverageStrategy(BaseStrategy):
    """简单移动平均策略"""
    
    def __init__(self, strategy_id: str, short_window: int = 20, long_window: int = 50):
        super().__init__(strategy_id, "SimpleMovingAverage")
        self.short_window = short_window
        self.long_window = long_window
        
        # 历史数据缓存
        self.price_history: Dict[str, List[float]] = {}
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        timestamp: datetime) -> List[Dict[str, Any]]:
        """生成移动平均交叉信号"""
        signals = []
        
        for symbol, data in market_data.items():
            if data.empty:
                continue
            
            # 获取最新价格
            latest_price = data['close'].iloc[-1]
            
            # 更新价格历史
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(latest_price)
            
            # 保持历史数据长度
            if len(self.price_history[symbol]) > self.long_window:
                self.price_history[symbol] = self.price_history[symbol][-self.long_window:]
            
            # 计算移动平均
            if len(self.price_history[symbol]) >= self.long_window:
                short_ma = np.mean(self.price_history[symbol][-self.short_window:])
                long_ma = np.mean(self.price_history[symbol][-self.long_window:])
                
                # 生成信号
                current_position = self.get_position(symbol)
                
                if short_ma > long_ma and current_position <= 0:
                    # 金叉，买入信号
                    signals.append({
                        'strategy_id': self.strategy_id,
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': 100,  # 固定数量
                        'price': latest_price,
                        'signal_strength': (short_ma - long_ma) / long_ma
                    })
                
                elif short_ma < long_ma and current_position > 0:
                    # 死叉，卖出信号
                    signals.append({
                        'strategy_id': self.strategy_id,
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': abs(current_position),
                        'price': latest_price,
                        'signal_strength': (long_ma - short_ma) / long_ma
                    })
        
        return signals


class Portfolio:
    """投资组合"""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        
        # 账户统计
        self.total_commission = 0.0
        self.total_trades = 0
        
        # 历史记录
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdown_curve: List[Tuple[datetime, float]] = []
        
    @property
    def total_value(self) -> float:
        """总资产价值"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_pnl(self) -> float:
        """总盈亏"""
        return self.total_value - self.initial_capital
    
    @property
    def total_return(self) -> float:
        """总收益率"""
        return self.total_pnl / self.initial_capital
    
    def get_position(self, symbol: str) -> Position:
        """获取持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def update_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """更新市场数据"""
        for symbol, data in market_data.items():
            if not data.empty and symbol in self.positions:
                latest_price = data['close'].iloc[-1]
                self.positions[symbol].update_price(latest_price)
    
    def add_fill(self, fill: Fill):
        """添加成交记录"""
        self.fills.append(fill)
        
        # 更新持仓
        position = self.get_position(fill.symbol)
        position.add_fill(fill)
        
        # 更新现金
        if fill.side == OrderSide.BUY:
            self.cash -= fill.quantity * fill.price + fill.commission
        else:
            self.cash += fill.quantity * fill.price - fill.commission
        
        self.total_commission += fill.commission
        self.total_trades += 1
    
    def record_equity(self, timestamp: datetime):
        """记录净值"""
        equity = self.total_value
        self.equity_curve.append((timestamp, equity))
        
        # 计算回撤
        if len(self.equity_curve) > 1:
            peak = max(eq[1] for eq in self.equity_curve)
            drawdown = (equity - peak) / peak
            self.drawdown_curve.append((timestamp, drawdown))


class VectorizedSignalGenerator:
    """向量化信号生成器"""
    
    def __init__(self):
        self.signal_cache = {}
        
    def generate_batch_signals(self, strategies: Dict[str, BaseStrategy], 
                             market_data: Dict[str, pd.DataFrame],
                             time_index: pd.DatetimeIndex) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量生成信号
        
        Args:
            strategies: 策略字典
            market_data: 市场数据
            time_index: 时间索引
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 策略信号字典
        """
        all_signals = {}
        
        for strategy_id, strategy in strategies.items():
            signals = []
            
            # 为每个时间点生成信号
            for timestamp in time_index:
                # 获取当前时间点的数据
                current_data = {}
                for symbol, data in market_data.items():
                    if timestamp in data.index:
                        # 获取到当前时间点的历史数据
                        historical_data = data.loc[:timestamp]
                        current_data[symbol] = historical_data
                
                if current_data:
                    try:
                        strategy_signals = strategy.generate_signals(current_data, timestamp)
                        if strategy_signals:
                            for signal in strategy_signals:
                                signal['timestamp'] = timestamp
                                signal['strategy_id'] = strategy_id
                                signals.append(signal)
                    except Exception as e:
                        logger.warning(f"策略 {strategy_id} 在 {timestamp} 生成信号时出错: {e}")
            
            all_signals[strategy_id] = signals
        
        return all_signals


class OptimizedEventQueue:
    """优化的事件队列"""
    
    def __init__(self):
        self.events_by_type = {event_type: [] for event_type in EventType}
        self.events_by_timestamp = {}
        
    def add_event(self, event: Event):
        """添加事件"""
        # 按类型分类
        self.events_by_type[event.event_type].append(event)
        
        # 按时间戳分类
        timestamp_key = event.timestamp
        if timestamp_key not in self.events_by_timestamp:
            self.events_by_timestamp[timestamp_key] = []
        self.events_by_timestamp[timestamp_key].append(event)
    
    def get_events_by_timestamp(self, timestamp: datetime) -> List[Event]:
        """获取指定时间戳的所有事件"""
        return self.events_by_timestamp.get(timestamp, [])
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """获取指定类型的所有事件"""
        return self.events_by_type[event_type]
    
    def clear_processed_events(self, timestamp: datetime):
        """清理已处理的事件"""
        if timestamp in self.events_by_timestamp:
            events = self.events_by_timestamp[timestamp]
            for event in events:
                self.events_by_type[event.event_type].remove(event)
            del self.events_by_timestamp[timestamp]
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return all(len(events) == 0 for events in self.events_by_type.values())


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {
            'data_processing_time': 0.0,
            'signal_generation_time': 0.0,
            'order_execution_time': 0.0,
            'event_processing_time': 0.0,
            'total_events_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_usage': 0.0
        }
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """开始计时"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """结束计时"""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            if operation in self.metrics:
                self.metrics[operation] += elapsed
            del self.start_times[operation]
            return elapsed
        return 0.0
    
    def increment_counter(self, metric: str, value: int = 1):
        """增加计数器"""
        if metric in self.metrics:
            self.metrics[metric] += value
    
    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total == 0:
            return 0.0
        return self.metrics['cache_hits'] / total
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = self.metrics.copy()
        summary['cache_hit_rate'] = self.get_cache_hit_rate()
        return summary


class ParallelBacktestEngine:
    """
    并行回测引擎，支持多资产并行处理
    """
    
    def __init__(self, max_workers: int = None):
        """
        初始化并行回测引擎
        
        Args:
            max_workers: 最大工作进程数，默认为CPU核心数
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.engines: Dict[str, 'EnhancedBacktestEngine'] = {}
        self.logger = logger
        
        # 初始化自适应执行策略
        self.adaptive_strategy = AdaptiveExecutionStrategy()
    
    def add_engine(self, symbol: str, engine: 'EnhancedBacktestEngine'):
        """添加回测引擎"""
        self.engines[symbol] = engine
    
    def run_threaded_backtest(self, start_date: datetime, end_date: datetime,
                             frequency: str = 'D') -> Dict[str, Dict[str, Any]]:
        """
        使用线程池运行回测（适用于小任务）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            
        Returns:
            各个资产的回测结果
        """
        self.logger.info(f"开始线程池回测，使用 {min(len(self.engines), 4)} 个线程")
        
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(self.engines), 4)) as executor:
            # 提交任务
            future_to_symbol = {
                executor.submit(engine.run_backtest, start_date, end_date, frequency): symbol
                for symbol, engine in self.engines.items()
            }
            
            # 收集结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    self.logger.info(f"完成 {symbol} 的回测")
                except Exception as e:
                    self.logger.error(f"回测 {symbol} 时出错: {e}")
                    results[symbol] = {'error': str(e)}
        
        return results
    
    def run_parallel_backtest(self, start_date: datetime, end_date: datetime,
                            frequency: str = 'D') -> Dict[str, Dict[str, Any]]:
        """
        智能并行运行多个回测，自动选择最优执行策略
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            
        Returns:
            各个资产的回测结果
        """
        # 计算任务复杂度
        num_symbols = len(self.engines)
        if num_symbols == 0:
            return {}
        
        # 估算数据长度（简化计算）
        data_length = (end_date - start_date).days
        
        # 计算任务指标
        task_metrics = self.adaptive_strategy.calculate_task_complexity(
            num_symbols=num_symbols,
            data_length=data_length,
            num_strategies=1  # 假设每个引擎有一个策略
        )
        
        # 选择执行策略
        execution_strategy = self.adaptive_strategy.choose_execution_strategy(task_metrics)
        
        # 记录开始时间
        start_time = time.time()
        
        # 根据选择的策略执行回测
        if execution_strategy == ExecutionStrategy.SEQUENTIAL:
            results = self._run_sequential_backtest(start_date, end_date, frequency)
        elif execution_strategy == ExecutionStrategy.THREADED:
            results = self.run_threaded_backtest(start_date, end_date, frequency)
        elif execution_strategy == ExecutionStrategy.ASYNC:
            # 创建异步引擎并运行
            async_engine = AsyncBacktestEngine()
            for symbol, engine in self.engines.items():
                async_engine.add_engine(symbol, engine)
            
            import asyncio
            results = asyncio.run(async_engine.run_async_backtest(start_date, end_date, frequency))
        else:  # PARALLEL
            results = self._run_process_parallel_backtest(start_date, end_date, frequency)
        
        # 记录性能
        execution_time = time.time() - start_time
        self.adaptive_strategy.record_performance(execution_strategy, execution_time, task_metrics)
        
        self.logger.info(f"使用 {execution_strategy.value} 策略完成回测，耗时: {execution_time:.3f}秒")
        
        return results
    
    def _run_sequential_backtest(self, start_date: datetime, end_date: datetime,
                               frequency: str = 'D') -> Dict[str, Dict[str, Any]]:
        """顺序执行回测"""
        results = {}
        for symbol, engine in self.engines.items():
            try:
                self.logger.info(f"开始 {symbol} 的顺序回测")
                result = engine.run_backtest(start_date, end_date, frequency)
                results[symbol] = result
                self.logger.info(f"完成 {symbol} 的回测")
            except Exception as e:
                self.logger.error(f"回测 {symbol} 时发生错误: {str(e)}")
                results[symbol] = {'error': str(e)}
        return results
    
    def _run_process_parallel_backtest(self, start_date: datetime, end_date: datetime,
                                     frequency: str = 'D') -> Dict[str, Dict[str, Any]]:
        """使用进程池执行并行回测（原有逻辑）"""
        
        self.logger.info(f"开始并行回测，使用 {self.max_workers} 个进程")
        
        # 准备任务参数，减少序列化开销
        tasks = []
        for symbol, engine in self.engines.items():
            # 只传递必要的数据，减少进程间通信开销
            engine_data = {
                'market_data': engine.market_data,
                'strategies': engine.strategies,
                'initial_capital': engine.portfolio.initial_capital,
                'commission_rate': engine.commission_rate,
                'slippage_rate': engine.slippage_rate
            }
            tasks.append((symbol, engine_data, start_date, end_date, frequency))
        
        # 使用进程池执行并行回测
        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_symbol = {
                executor.submit(self._run_optimized_backtest, task): task[0] 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    self.logger.info(f"完成 {symbol} 的并行回测")
                except Exception as e:
                    self.logger.error(f"并行回测 {symbol} 时发生错误: {str(e)}")
                    results[symbol] = {'error': str(e)}
        
        return results
    
    @staticmethod
    def _run_optimized_backtest(task_params) -> Dict[str, Any]:
        """
        运行优化的单个回测任务（静态方法，用于多进程）
        减少对象序列化开销
        """
        symbol, engine_data, start_date, end_date, frequency = task_params
        try:
            # 在子进程中重建引擎，避免序列化整个对象
            engine = EnhancedBacktestEngine(
                initial_capital=engine_data['initial_capital'],
                commission_rate=engine_data['commission_rate'],
                slippage_rate=engine_data['slippage_rate']
            )
            
            # 设置市场数据和策略
            for sym, data in engine_data['market_data'].items():
                engine.set_market_data(sym, data)
            
            for strategy in engine_data['strategies'].values():
                engine.add_strategy(strategy)
            
            return engine.run_backtest(start_date, end_date, frequency)
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def _run_single_backtest(task_params) -> Dict[str, Any]:
        """
        运行单个回测任务（静态方法，用于多进程）
        """
        symbol, engine, start_date, end_date, frequency = task_params
        try:
            return engine.run_backtest(start_date, end_date, frequency)
        except Exception as e:
            return {'error': str(e)}


class AsyncBacktestEngine:
    """
    异步回测引擎，支持异步I/O操作
    """
    
    def __init__(self):
        self.engines: Dict[str, 'EnhancedBacktestEngine'] = {}
        self.logger = logger
    
    def add_engine(self, symbol: str, engine: 'EnhancedBacktestEngine'):
        """添加回测引擎"""
        self.engines[symbol] = engine
    
    async def run_async_backtest(self, start_date: datetime, end_date: datetime,
                               frequency: str = 'D') -> Dict[str, Dict[str, Any]]:
        """
        异步运行多个回测
        """
        self.logger.info("开始异步回测")
        
        # 创建异步任务
        tasks = []
        for symbol, engine in self.engines.items():
            task = asyncio.create_task(
                self._run_async_single_backtest(symbol, engine, start_date, end_date, frequency)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = {}
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (symbol, _) in enumerate(self.engines.items()):
            result = completed_tasks[i]
            if isinstance(result, Exception):
                self.logger.error(f"回测 {symbol} 时出错: {result}")
                results[symbol] = {'error': str(result)}
            else:
                results[symbol] = result
                self.logger.info(f"完成 {symbol} 的回测")
        
        return results
    
    async def _run_async_single_backtest(self, symbol: str, engine: 'EnhancedBacktestEngine',
                                       start_date: datetime, end_date: datetime,
                                       frequency: str) -> Dict[str, Any]:
        """
        异步运行单个回测
        """
        loop = asyncio.get_event_loop()
        # 在线程池中运行CPU密集型任务
        return await loop.run_in_executor(
            None, engine.run_backtest, start_date, end_date, frequency
        )


class EnhancedBacktestEngine:
    """增强版回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.001,
                 enable_caching: bool = True,
                 enable_preprocessing: bool = True):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            enable_caching: 是否启用缓存
            enable_preprocessing: 是否启用数据预处理
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.enable_caching = enable_caching
        self.enable_preprocessing = enable_preprocessing
        
        # 核心组件
        self.portfolio = Portfolio(initial_capital)
        self.strategies: Dict[str, BaseStrategy] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.preprocessed_data: Dict[str, pd.DataFrame] = {}
        
        # 性能优化组件
        if enable_preprocessing:
            self.data_preprocessor = DataPreprocessor()
        else:
            self.data_preprocessor = None
            
        self.signal_generator = VectorizedSignalGenerator()
        self.optimized_queue = OptimizedEventQueue()
        self.performance_monitor = PerformanceMonitor()
        
        # 事件系统
        self.event_queue = []
        self.event_handlers = {}
        self._initialize_event_handlers()
        
        # 回测状态
        self.is_running = False
        self.current_time = None
        self.start_date = None
        self.end_date = None
        
        # 性能统计
        self.performance_stats = {}
        self.order_id_counter = 0
        self.fill_id_counter = 0
        
        self.logger = logger
    
    def _initialize_event_handlers(self):
        """初始化事件处理器"""
        self.event_handlers[EventType.MARKET_DATA] = MarketDataHandler(self)
        self.event_handlers[EventType.SIGNAL] = SignalHandler(self)
        self.event_handlers[EventType.ORDER] = OrderHandler(self)
        self.event_handlers[EventType.FILL] = FillHandler(self)
    
    def add_strategy(self, strategy: BaseStrategy):
        """添加策略"""
        self.strategies[strategy.strategy_id] = strategy
        self.logger.info(f"已添加策略: {strategy.name} ({strategy.strategy_id})")
    
    def remove_strategy(self, strategy_id: str):
        """移除策略"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.logger.info(f"已移除策略: {strategy_id}")
    
    def set_market_data(self, symbol: str, data: pd.DataFrame):
        """
        设置市场数据
        
        Args:
            symbol: 股票代码
            data: 市场数据，包含 open, high, low, close, volume 列
        """
        # 验证数据格式
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"数据必须包含列: {required_columns}")
        
        self.market_data[symbol] = data
        
        # 如果启用预处理，立即预处理数据
        if self.enable_preprocessing:
            self.performance_monitor.start_timer('data_preprocessing')
            self.preprocessed_data[symbol] = self.data_preprocessor.preprocess_market_data(data, symbol)
            self.performance_monitor.end_timer('data_preprocessing')
            
        self.logger.info(f"已设置 {symbol} 的市场数据，共 {len(data)} 条记录")

    def run_backtest(self, start_date: datetime, end_date: datetime,
                    frequency: str = 'D') -> Dict[str, Any]:
        """
        运行回测（优化版本）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率 ('D', 'H', 'T')
            
        Returns:
            回测结果字典
        """
        backtest_start_time = time.time()
        self.performance_monitor.start_timer('total_backtest')
        
        self.logger.info(f"开始回测: {start_date} 到 {end_date}, 频率: {frequency}")
        
        # 初始化回测状态
        self.is_running = True
        self.start_date = start_date
        self.end_date = end_date
        self.current_time = start_date
        
        # 数据预处理（如果启用）
        if self.enable_preprocessing and self.data_preprocessor:
            self.performance_monitor.start_timer('data_processing')
            self.preprocessed_data = self.data_preprocessor.batch_preprocess(self.market_data)
            self.performance_monitor.end_timer('data_processing')
        
        # 生成时间序列
        if frequency == 'D':
            time_index = pd.date_range(start_date, end_date, freq='D')
        elif frequency == 'H':
            time_index = pd.date_range(start_date, end_date, freq='H')
        elif frequency == 'T':
            time_index = pd.date_range(start_date, end_date, freq='T')
        else:
            raise ValueError(f"不支持的频率: {frequency}")
        
        # 向量化数据准备
        self.performance_monitor.start_timer('data_vectorization')
        vectorized_data = self._prepare_vectorized_data(time_index)
        self.performance_monitor.end_timer('data_vectorization')
        
        # 批量信号生成（如果可能）
        if len(self.strategies) > 0:
            self.performance_monitor.start_timer('signal_generation')
            batch_signals = self.signal_generator.generate_batch_signals(
                self.strategies, 
                self.preprocessed_data if self.preprocessed_data else self.market_data,
                time_index
            )
            self.performance_monitor.end_timer('signal_generation')
        else:
            batch_signals = {}
        
        # 按时间顺序处理（优化版本）
        self.performance_monitor.start_timer('event_processing')
        for i, timestamp in enumerate(time_index):
            if not self.is_running:
                break
            
            self.current_time = timestamp
            
            # 获取当前时间点的市场数据（向量化）
            current_market_data = self._get_current_market_data_vectorized(vectorized_data, i, timestamp)
            
            if current_market_data:
                # 使用优化的事件队列
                market_event = Event(
                    event_type=EventType.MARKET_DATA,
                    timestamp=timestamp,
                    data=current_market_data,
                    priority=0
                )
                
                self.optimized_queue.add_event(market_event)
                
                # 添加预生成的信号事件
                for strategy_id, signals in batch_signals.items():
                    if i < len(signals) and signals[i]:
                        signal_event = Event(
                            event_type=EventType.SIGNAL,
                            timestamp=timestamp,
                            data=signals[i],
                            priority=1
                        )
                        self.optimized_queue.add_event(signal_event)
                
                # 处理当前时间点的所有事件
                self._process_events_optimized(timestamp)
                
                # 记录净值
                self.portfolio.record_equity(timestamp)
                
                # 增加处理计数
                self.performance_monitor.increment_counter('events_processed')
        
        self.performance_monitor.end_timer('event_processing')
        self.is_running = False
        
        # 计算性能统计
        self.performance_monitor.start_timer('performance_calculation')
        self.performance_stats = self._calculate_performance_stats()
        self.performance_monitor.end_timer('performance_calculation')
        
        # 结束总计时
        self.performance_monitor.end_timer('total_backtest')
        
        # 记录性能指标
        performance_summary = self.performance_monitor.get_summary()
        self.logger.info(f"回测完成，性能摘要: {performance_summary}")
        
        return self.get_backtest_results()
    
    def _process_events_optimized(self, timestamp: datetime):
        """优化的事件处理"""
        # 获取当前时间点的所有事件
        events = self.optimized_queue.get_events_by_timestamp(timestamp)
        
        # 按优先级排序处理
        events.sort(key=lambda x: x.priority)
        
        for event in events:
            handler = self.event_handlers.get(event.event_type)
            if handler:
                new_events = handler.handle_event(event)
                # 将新生成的事件添加到队列
                for new_event in new_events:
                    self.optimized_queue.add_event(new_event)
        
        # 清理已处理的事件
        self.optimized_queue.clear_processed_events(timestamp)
    
    def _prepare_vectorized_data(self, time_index: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """准备向量化数据"""
        vectorized_data = {}
        
        for symbol in self.market_data.keys():
            # 使用预处理数据或原始数据
            data = self.preprocessed_data.get(symbol, self.market_data[symbol])
            
            # 重新索引到回测时间范围
            reindexed_data = data.reindex(time_index, method='ffill')
            vectorized_data[symbol] = reindexed_data
        
        return vectorized_data
    
    def _get_current_market_data_vectorized(self, vectorized_data: Dict[str, pd.DataFrame], 
                                          index: int, timestamp: datetime) -> Dict[str, pd.DataFrame]:
        """向量化获取当前市场数据"""
        current_market_data = {}
        
        for symbol, data in vectorized_data.items():
            if index < len(data) and not pd.isna(data.iloc[index]['close']):
                # 返回当前时间点的数据
                current_market_data[symbol] = data.iloc[[index]]
        
        return current_market_data

    def generate_orders_from_signal(self, signal_data: Dict[str, Any], 
                                   timestamp: datetime) -> List[Order]:
        """从信号生成订单"""
        orders = []
        
        strategy_id = signal_data.get('strategy_id')
        symbol = signal_data.get('symbol')
        action = signal_data.get('action')
        quantity = signal_data.get('quantity', 0)
        price = signal_data.get('price')
        
        if not all([strategy_id, symbol, action, quantity]):
            self.logger.warning(f"信号数据不完整: {signal_data}")
            return orders
        
        # 生成订单ID
        self.order_id_counter += 1
        order_id = f"ORDER_{self.order_id_counter:06d}"
        
        # 确定订单方向
        side = OrderSide.BUY if action.lower() == 'buy' else OrderSide.SELL
        
        # 创建订单
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,  # 默认市价单
            quantity=abs(quantity),
            price=price,
            timestamp=timestamp,
            strategy_id=strategy_id
        )
        
        orders.append(order)
        self.portfolio.orders[order_id] = order
        
        return orders
    
    def execute_order(self, order: Order, timestamp: datetime) -> List[Fill]:
        """执行订单"""
        fills = []
        
        # 获取当前市场数据
        if order.symbol not in self.market_data:
            self.logger.warning(f"没有 {order.symbol} 的市场数据")
            order.status = OrderStatus.REJECTED
            return fills
        
        symbol_data = self.market_data[order.symbol]
        
        # 找到当前时间点的数据
        current_data = symbol_data[symbol_data.index <= timestamp]
        if current_data.empty:
            self.logger.warning(f"没有找到 {order.symbol} 在 {timestamp} 的数据")
            order.status = OrderStatus.REJECTED
            return fills
        
        latest_data = current_data.iloc[-1]
        
        # 确定成交价格
        if order.order_type == OrderType.MARKET:
            # 市价单，使用收盘价加滑点
            base_price = latest_data['close']
            if order.side == OrderSide.BUY:
                fill_price = base_price * (1 + self.slippage_rate)
            else:
                fill_price = base_price * (1 - self.slippage_rate)
        else:
            # 限价单等其他类型暂时简化处理
            fill_price = order.price or latest_data['close']
        
        # 计算手续费
        commission = order.quantity * fill_price * self.commission_rate
        
        # 检查资金是否充足（买入时）
        if order.side == OrderSide.BUY:
            required_cash = order.quantity * fill_price + commission
            if self.portfolio.cash < required_cash:
                self.logger.warning(f"资金不足，需要 {required_cash:.2f}，可用 {self.portfolio.cash:.2f}")
                order.status = OrderStatus.REJECTED
                return fills
        
        # 检查持仓是否充足（卖出时）
        if order.side == OrderSide.SELL:
            current_position = self.portfolio.get_position(order.symbol).quantity
            if current_position < order.quantity:
                self.logger.warning(f"持仓不足，需要 {order.quantity}，持有 {current_position}")
                order.status = OrderStatus.REJECTED
                return fills
        
        # 生成成交记录
        self.fill_id_counter += 1
        fill_id = f"FILL_{self.fill_id_counter:06d}"
        
        fill = Fill(
            fill_id=fill_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=timestamp,
            commission=commission,
            strategy_id=order.strategy_id
        )
        
        fills.append(fill)
        
        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        
        return fills
    
    def update_position(self, fill: Fill):
        """更新持仓"""
        self.portfolio.add_fill(fill)
        
        # 更新策略持仓
        if fill.strategy_id and fill.strategy_id in self.strategies:
            strategy = self.strategies[fill.strategy_id]
            quantity_change = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            strategy.update_position(fill.symbol, quantity_change)
    
    def update_account(self, fill: Fill):
        """更新账户"""
        # 持仓和现金更新已在update_position中处理
        pass
    
    def add_event(self, event: Event):
        """添加事件到队列"""
        self.event_queue.append(event)
        self.event_queue.sort()  # 按时间和优先级排序
    
    def process_events(self):
        """处理事件队列"""
        while self.event_queue:
            event = self.event_queue.pop(0)
            
            # 更新当前时间
            self.current_time = event.timestamp
            
            # 获取对应的事件处理器
            handler = self.event_handlers.get(event.event_type)
            if handler:
                # 处理事件，可能产生新事件
                new_events = handler.handle_event(event)
                
                # 将新事件添加到队列
                for new_event in new_events:
                    self.add_event(new_event)
            else:
                self.logger.warning(f"没有找到 {event.event_type} 的处理器")
    
    def run_backtest(self, start_date: datetime, end_date: datetime,
                    frequency: str = 'D') -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率 ('D', 'H', 'T')
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        self.logger.info(f"开始回测: {start_date} 到 {end_date}")
        
        self.start_date = start_date
        self.end_date = end_date
        self.is_running = True
        
        # 生成时间序列
        if frequency == 'D':
            time_index = pd.date_range(start_date, end_date, freq='D')
        elif frequency == 'H':
            time_index = pd.date_range(start_date, end_date, freq='H')
        elif frequency == 'T':
            time_index = pd.date_range(start_date, end_date, freq='T')
        else:
            raise ValueError(f"不支持的频率: {frequency}")
        
        # 按时间顺序处理
        for timestamp in time_index:
            if not self.is_running:
                break
            
            # 获取当前时间点的市场数据
            current_market_data = {}
            for symbol, data in self.market_data.items():
                # 获取到当前时间为止的数据
                available_data = data[data.index <= timestamp]
                if not available_data.empty:
                    # 只传递最新的一条数据
                    current_market_data[symbol] = available_data.tail(1)
            
            if current_market_data:
                # 创建市场数据事件
                market_event = Event(
                    event_type=EventType.MARKET_DATA,
                    timestamp=timestamp,
                    data=current_market_data,
                    priority=0
                )
                
                self.add_event(market_event)
                
                # 处理所有事件
                self.process_events()
                
                # 记录净值
                self.portfolio.record_equity(timestamp)
        
        self.is_running = False
        
        # 计算性能统计
        self.performance_stats = self._calculate_performance_stats()
        
        self.logger.info("回测完成")
        return self.get_backtest_results()
    
    def _calculate_performance_stats(self) -> Dict[str, Any]:
        """计算性能统计"""
        if not self.portfolio.equity_curve:
            return {}
        
        # 提取净值数据
        dates, values = zip(*self.portfolio.equity_curve)
        equity_series = pd.Series(values, index=dates)
        
        # 计算收益率
        returns = equity_series.pct_change().dropna()
        
        # 基本统计
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        annualized_return = (1 + total_return) ** (252 / len(equity_series)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_trades = len([r for r in returns if r > 0])
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calmar比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'total_trades': self.portfolio.total_trades,
            'total_commission': self.portfolio.total_commission,
            'final_value': equity_series.iloc[-1],
            'start_value': equity_series.iloc[0]
        }
    
    def get_backtest_results(self) -> Dict[str, Any]:
        """获取回测结果"""
        return {
            'performance_stats': self.performance_stats,
            'equity_curve': self.portfolio.equity_curve,
            'drawdown_curve': self.portfolio.drawdown_curve,
            'positions': {symbol: pos.__dict__ for symbol, pos in self.portfolio.positions.items()},
            'fills': [fill.__dict__ for fill in self.portfolio.fills],
            'orders': {order_id: order.__dict__ for order_id, order in self.portfolio.orders.items()},
            'final_portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash
        }
    
    def stop_backtest(self):
        """停止回测"""
        self.is_running = False
        self.logger.info("回测已停止")
    
    def reset(self):
        """重置回测引擎"""
        self.portfolio = Portfolio(self.initial_capital)
        self.event_queue.clear()
        self.order_id_counter = 0
        self.fill_id_counter = 0
        self.current_time = None
        self.performance_stats.clear()
        
        # 重置策略状态
        for strategy in self.strategies.values():
            strategy.positions.clear()
            strategy.signals_history.clear()
        
        self.logger.info("回测引擎已重置")


# 多策略回测管理器
class MultiStrategyBacktestManager:
    """多策略回测管理器"""
    
    def __init__(self):
        self.engines: Dict[str, 'EnhancedBacktestEngine'] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.logger = logger
    
    def add_engine(self, name: str, engine: 'EnhancedBacktestEngine'):
        """添加回测引擎"""
        self.engines[name] = engine
        self.logger.info(f"已添加回测引擎: {name}")
    
    def run_parallel_backtests(self, start_date: datetime, end_date: datetime,
                              frequency: str = 'D') -> Dict[str, Dict[str, Any]]:
        """并行运行多个回测"""
        self.logger.info(f"开始并行回测，共 {len(self.engines)} 个引擎")
        
        with ThreadPoolExecutor(max_workers=min(len(self.engines), 4)) as executor:
            # 提交所有回测任务
            futures = {}
            for name, engine in self.engines.items():
                future = executor.submit(engine.run_backtest, start_date, end_date, frequency)
                futures[name] = future
            
            # 收集结果
            for name, future in futures.items():
                try:
                    result = future.result()
                    self.results[name] = result
                    self.logger.info(f"回测 {name} 完成")
                except Exception as e:
                    self.logger.error(f"回测 {name} 失败: {e}")
                    self.results[name] = {'error': str(e)}
        
        return self.results
    
    def compare_strategies(self) -> pd.DataFrame:
        """比较策略性能"""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, result in self.results.items():
            if 'error' in result:
                continue
            
            stats = result.get('performance_stats', {})
            comparison_data.append({
                'Strategy': name,
                'Total Return': stats.get('total_return', 0),
                'Annualized Return': stats.get('annualized_return', 0),
                'Volatility': stats.get('volatility', 0),
                'Sharpe Ratio': stats.get('sharpe_ratio', 0),
                'Max Drawdown': stats.get('max_drawdown', 0),
                'Win Rate': stats.get('win_rate', 0),
                'Calmar Ratio': stats.get('calmar_ratio', 0),
                'Total Trades': stats.get('total_trades', 0),
                'Final Value': stats.get('final_value', 0)
            })
        
        return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    # 测试增强版回测引擎
    
    # 创建测试数据
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # 生成模拟股价数据
    dates = pd.date_range(start_date, end_date, freq='D')
    np.random.seed(42)
    
    # 股票A
    returns_a = np.random.normal(0.0005, 0.02, len(dates))
    prices_a = 100 * np.exp(np.cumsum(returns_a))
    
    stock_a_data = pd.DataFrame({
        'open': prices_a * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices_a * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices_a * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices_a,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # 创建回测引擎
    engine = EnhancedBacktestEngine(
        initial_capital=1000000,
        commission_rate=0.001,
        slippage_rate=0.001
    )
    
    # 设置市场数据
    engine.set_market_data('STOCK_A', stock_a_data)
    
    # 添加策略
    ma_strategy = SimpleMovingAverageStrategy('MA_STRATEGY', short_window=10, long_window=30)
    engine.add_strategy(ma_strategy)
    
    # 运行回测
    results = engine.run_backtest(start_date, end_date)
    
    # 打印结果
    print("回测结果:")
    stats = results['performance_stats']
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n最终组合价值: {results['final_portfolio_value']:.2f}")
    print(f"现金余额: {results['cash']:.2f}")
    print(f"总交易次数: {len(results['fills'])}")