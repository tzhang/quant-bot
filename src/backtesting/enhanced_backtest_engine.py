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

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
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
        self.backtest_engine.update_market_data(market_data)
        
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


class EnhancedBacktestEngine:
    """增强版回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # 核心组件
        self.portfolio = Portfolio(initial_capital)
        self.strategies: Dict[str, BaseStrategy] = {}
        self.event_handlers: Dict[EventType, EventHandler] = {}
        
        # 市场数据
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_time: Optional[datetime] = None
        
        # 事件队列
        self.event_queue: List[Event] = []
        
        # 订单管理
        self.order_id_counter = 0
        self.fill_id_counter = 0
        
        # 回测状态
        self.is_running = False
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        
        # 性能统计
        self.performance_stats: Dict[str, Any] = {}
        
        # 初始化事件处理器
        self._initialize_event_handlers()
        
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
        """设置市场数据"""
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"市场数据缺少必要列: {col}")
        
        # 确保索引是datetime类型
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        self.market_data[symbol] = data.sort_index()
        self.logger.info(f"已设置 {symbol} 的市场数据，共 {len(data)} 条记录")
    
    def update_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """更新市场数据"""
        self.portfolio.update_market_data(market_data)
    
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
        self.engines: Dict[str, EnhancedBacktestEngine] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.logger = logger
    
    def add_engine(self, name: str, engine: EnhancedBacktestEngine):
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