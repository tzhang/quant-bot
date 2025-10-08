#!/usr/bin/env python3
"""
Citadel高频交易策略与Interactive Brokers API集成
结合Citadel策略信号生成与IB实时交易执行
"""

import sys
import os
import time
import threading
import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

# 添加项目根目录到路径
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入Citadel策略
from competitions.citadel.citadel_hft_strategy import CitadelHFTStrategy
from ib_adapter import IBAdapter
from ib_market_data_stream import IBMarketDataStream
from nasdaq100_symbols import get_nasdaq100_symbols, get_top_symbols, get_symbols_by_category
from console_formatter import console_formatter, setup_enhanced_logging
from enhanced_data_module import EnhancedDataModule, MarketDataPoint, BarData
from factor_analysis_module import FactorAnalyzer, TechnicalFactors, FundamentalFactors
from strategy_discovery_module import StrategyDiscoveryEngine, MomentumStrategy, MeanReversionStrategy

# 尝试导入市场日历模块
try:
    from src.utils.market_calendar import market_calendar
    from src.utils.timezone_manager import timezone_manager
    HAS_MARKET_CALENDAR = True
except ImportError:
    HAS_MARKET_CALENDAR = False
    print("警告: 市场日历模块未找到，将使用简化版本")
    # 创建简化版本的市场日历类
    class SimpleMarketCalendar:
        def is_market_open_now(self):
            from datetime import datetime
            now = datetime.now()
            return (now.weekday() < 5 and 
                   9 <= now.hour < 16)
    
    market_calendar = SimpleMarketCalendar()

# 尝试导入GUI模块，如果失败则跳过
try:
    from hft_monitor_system import HFTMonitorSystem, TradingSignal, TradeExecution, RiskMetrics
    GUI_AVAILABLE = True
except ImportError:
    print("GUI模块不可用，将在无GUI模式下运行")
    GUI_AVAILABLE = False
    # 定义基本数据结构
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict
    
    @dataclass
    class TradingSignal:
        timestamp: datetime
        symbol: str
        signal_type: str
        strength: float
        price: float
        confidence: float
        components: Dict[str, float]
    
    @dataclass
    class TradeExecution:
        timestamp: datetime
        symbol: str
        action: str
        quantity: int
        price: float
        order_id: str
        status: str
        pnl: float = 0.0
    
    @dataclass
    class RiskMetrics:
        timestamp: datetime
        portfolio_value: float
        cash: float
        total_exposure: float
        max_drawdown: float
        var_1d: float
        sharpe_ratio: float
        positions: Dict[str, float]

@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: int
    order_type: str  # 'MKT', 'LMT', 'STP'
    price: Optional[float]
    status: str  # 'Submitted', 'Filled', 'Cancelled'
    filled_qty: int
    avg_fill_price: float
    timestamp: datetime

class CitadelIBIntegration:
    """Citadel策略与IB API集成系统"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化集成系统
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.strategy = CitadelHFTStrategy()
        
        # 创建IB配置
        from ib_adapter import IBConfig
        ib_config = IBConfig(
            host=config.get('ib_host', '127.0.0.1'),
            port=config.get('ib_port', 7497),
            client_id=config.get('client_id', 1)
        )
        
        self.ib_adapter = IBAdapter(ib_config)
        
        # 创建市场数据流（使用不同的客户端ID避免冲突）
        self.market_data_stream = IBMarketDataStream(
            host=config.get('ib_host', '127.0.0.1'),
            port=config.get('ib_port', 7497),
            client_id=config.get('client_id', 1) + 1  # 使用不同的客户端ID
        )
        
        # 监控系统（可选）
        self.monitor = None
        if config.get('enable_monitor', False) and GUI_AVAILABLE:
            self.monitor = HFTMonitorSystem()
        elif config.get('enable_monitor', False) and not GUI_AVAILABLE:
            print("警告: 监控系统已禁用，因为GUI模块不可用")
        
        # 交易配置
        self.symbols = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
        self.max_position_size = config.get('max_position_size', 1000)
        self.max_daily_trades = config.get('max_daily_trades', 100)
        self.risk_limit = config.get('risk_limit', 10000)  # 最大风险敞口
        
        # 状态管理
        self.running = False
        self.positions: Dict[str, PositionInfo] = {}
        self.orders: Dict[str, OrderInfo] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # 数据存储
        self.market_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.signals: deque = deque(maxlen=1000)
        self.trades: deque = deque(maxlen=1000)
        
        # 线程管理
        self.data_thread = None
        self.strategy_thread = None
        self.execution_thread = None
        
        # 队列
        self.signal_queue = queue.Queue()
        self.order_queue = queue.Queue()
        
        # 风险管理
        self.risk_manager = RiskManager(config.get('risk_config', {}))
        
        # 新增：高级数据模块
        self.data_module = EnhancedDataModule(
            data_dir="data"
        )
        
        # 新增：因子分析模块
        self.factor_analyzer = FactorAnalyzer()
        
        # 新增：策略发现模块
        self.strategy_discovery = StrategyDiscoveryEngine()
        
        # 性能统计
        self.performance_stats = {
            'total_signals': 0,
            'executed_trades': 0,
            'winning_trades': 0,
            'total_volume': 0,
            'avg_execution_time': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    def initialize(self) -> bool:
        """初始化系统"""
        try:
            # 使用格式化输出
            console_formatter.print_title("🚀 Citadel-IB 集成系统初始化")
            
            # 连接IB
            console_formatter.print_section("📡 连接到IB TWS/Gateway")
            console_formatter.print_info("正在建立与Interactive Brokers的连接...")
            if not self.ib_adapter.connect_to_ib():
                console_formatter.print_error("❌ 无法连接到IB TWS/Gateway")
                return False
            console_formatter.print_success("✅ IB Adapter连接成功")
            
            console_formatter.print_section("📊 连接市场数据流")
            console_formatter.print_info("正在建立市场数据连接...")
            if not self.market_data_stream.connect_to_ib():
                console_formatter.print_error("❌ 无法连接到IB市场数据流")
                return False
            console_formatter.print_success("✅ 市场数据流连接成功")
            
            # 等待连接稳定
            console_formatter.print_info("⏳ 等待连接稳定 (3秒)...")
            time.sleep(3)
            
            # 获取账户信息
            console_formatter.print_section("💰 获取账户信息")
            account_info = self.ib_adapter.get_account_info()
            if account_info:
                console_formatter.print_key_value("账户总价值", f"${account_info.total_cash}")
            console_formatter.print_key_value("净清算价值", f"${account_info.net_liquidation}")
            console_formatter.print_key_value("购买力", f"${account_info.buying_power}")
            
            # 获取当前持仓
            console_formatter.print_section("📈 获取持仓信息")
            self.update_positions()
            if self.positions:
                console_formatter.print_info(f"当前持有 {len(self.positions)} 个股票仓位")
                for symbol, pos in list(self.positions.items())[:5]:  # 显示前5个
                    console_formatter.print_key_value(f"{symbol} 持仓", f"{pos.quantity}股 @ ${pos.avg_price:.2f}")
            else:
                console_formatter.print_info("当前无持仓")
            
            # 订阅市场数据
            console_formatter.print_section("📡 订阅市场数据")
            console_formatter.print_info(f"正在订阅 {len(self.symbols)} 个NASDAQ100股票的市场数据...")
            for i, symbol in enumerate(self.symbols, 1):
                console_formatter.print_info(f"[{i}/{len(self.symbols)}] 订阅 {symbol}")
                self.market_data_stream.subscribe_market_data(symbol)
                self.market_data_stream.subscribe_realtime_bars(symbol)
            
            # 设置回调
            console_formatter.print_section("⚙️ 设置回调函数")
            self.market_data_stream.add_data_callback(self.on_tick_data)
            self.market_data_stream.add_bar_callback(self.on_bar_data)
            console_formatter.print_success("✅ 回调函数设置完成")
            
            console_formatter.print_title("🎯 系统初始化完成")
            console_formatter.print_success("系统已准备就绪，开始交易监控...")
            return True
            
        except Exception as e:
            console_formatter.print_error(f"❌ 初始化失败: {e}")
            return False
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def start(self):
        """启动系统"""
        if not self.initialize():
            return False
        
        self.running = True
        self.logger.info("启动Citadel-IB集成系统")
        
        # 启动数据处理线程
        self.data_thread = threading.Thread(target=self.data_processing_loop, daemon=True)
        self.data_thread.start()
        
        # 启动策略线程
        self.strategy_thread = threading.Thread(target=self.strategy_loop, daemon=True)
        self.strategy_thread.start()
        
        # 启动执行线程
        self.execution_thread = threading.Thread(target=self.execution_loop, daemon=True)
        self.execution_thread.start()
        
        # 启动监控系统
        if self.monitor:
            monitor_thread = threading.Thread(target=self.monitor.run, daemon=True)
            monitor_thread.start()
        
        return True
    
    def stop(self):
        """停止系统"""
        self.logger.info("停止Citadel-IB集成系统")
        self.running = False
        
        # 断开连接
        self.ib_adapter.disconnect()
        self.market_data_stream.disconnect()
        
        # 等待线程结束
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=5)
        if self.strategy_thread and self.strategy_thread.is_alive():
            self.strategy_thread.join(timeout=5)
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)
    
    def data_processing_loop(self):
        """数据处理循环"""
        while self.running:
            try:
                # 处理市场数据更新
                self.process_market_data()
                
                # 更新持仓信息
                self.update_positions()
                
                # 计算风险指标
                self.update_risk_metrics()
                
                time.sleep(0.1)  # 100ms更新间隔
                
            except Exception as e:
                self.logger.error(f"数据处理错误: {e}")
                time.sleep(1)
    
    def strategy_loop(self):
        """策略信号生成循环"""
        while self.running:
            try:
                # 检查市场是否开放
                if not market_calendar.is_market_open_now():
                    self.logger.info("市场未开放，暂停信号生成")
                    time.sleep(60)  # 市场关闭时每分钟检查一次
                    continue
                
                # 为每个交易品种生成信号
                for symbol in self.symbols:
                    signal = self.generate_signal(symbol)
                    if signal:
                        self.signal_queue.put(signal)
                        self.signals.append(signal)
                        
                        # 发送到监控系统
                        if self.monitor:
                            self.monitor.add_signal(signal)
                
                time.sleep(0.05)  # 50ms信号生成间隔
                
            except Exception as e:
                self.logger.error(f"策略循环错误: {e}")
                time.sleep(1)
    
    def execution_loop(self):
        """交易执行循环"""
        while self.running:
            try:
                # 处理信号队列
                if not self.signal_queue.empty():
                    signal = self.signal_queue.get()
                    self.process_signal(signal)
                
                # 处理订单状态更新
                self.process_order_updates()
                
                time.sleep(0.01)  # 10ms执行间隔
                
            except Exception as e:
                self.logger.error(f"执行循环错误: {e}")
                time.sleep(1)
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """生成交易信号 - 集成因子分析和策略发现"""
        try:
            # 获取最新市场数据
            if symbol not in self.market_data or len(self.market_data[symbol]) < 50:
                return None
            
            # 准备数据
            data_list = list(self.market_data[symbol])
            df = pd.DataFrame(data_list)
            
            if len(df) < 50:
                return None
            
            # 1. 使用原始Citadel策略生成信号
            citadel_signals = self.strategy.generate_signals(df)
            
            # 2. 使用因子分析增强信号
            technical_factors = self.factor_analyzer.calculate_technical_factors(df)
            
            # 3. 获取策略发现的最佳策略
            discovered_strategies = self.strategy_discovery.discover_strategies(df, symbol)
            
            # 4. 综合信号生成
            composite_signal = 0.0
            signal_components = {}
            
            # Citadel策略权重 40%
            if citadel_signals is not None and len(citadel_signals) > 0:
                latest_citadel = citadel_signals.iloc[-1]
                citadel_signal = latest_citadel.get('composite_signal', 0)
                composite_signal += 0.4 * citadel_signal
                signal_components['citadel'] = citadel_signal
            
            # 技术因子权重 35%
            factor_signal = 0.0
            if technical_factors:
                # 动量因子
                momentum_score = technical_factors.get('momentum_20', 0)
                # 均值回归因子
                mean_reversion_score = -technical_factors.get('rsi_14', 50) / 50 + 1  # RSI转换为信号
                # 波动率因子
                volatility_score = technical_factors.get('volatility_20', 0)
                
                factor_signal = (momentum_score * 0.5 + mean_reversion_score * 0.3 + 
                               volatility_score * 0.2)
                composite_signal += 0.35 * factor_signal
                signal_components.update({
                    'momentum_factor': momentum_score,
                    'mean_reversion_factor': mean_reversion_score,
                    'volatility_factor': volatility_score
                })
            
            # 策略发现权重 25%
            strategy_signal = 0.0
            if discovered_strategies:
                # 使用表现最好的策略
                best_strategy = max(discovered_strategies.items(), 
                                  key=lambda x: x[1].sharpe_ratio)
                strategy_name, performance = best_strategy
                
                # 根据策略类型生成信号
                if 'momentum' in strategy_name.lower():
                    strategy_signal = technical_factors.get('momentum_20', 0)
                elif 'mean_reversion' in strategy_name.lower():
                    strategy_signal = -technical_factors.get('rsi_14', 50) / 50 + 1
                
                composite_signal += 0.25 * strategy_signal
                signal_components['strategy_discovery'] = strategy_signal
                signal_components['best_strategy'] = strategy_name
            
            # 计算信号强度和置信度
            signal_strength = abs(composite_signal)
            confidence = self.calculate_enhanced_confidence(df, technical_factors, discovered_strategies)
            
            # 确定信号类型
            if composite_signal > 0.2:
                signal_type = 'BUY'
            elif composite_signal < -0.2:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            if signal_type == 'HOLD':
                return None
            
            # 创建增强的交易信号
            trading_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                price=df['close'].iloc[-1] if 'close' in df.columns else df['price'].iloc[-1],
                confidence=confidence,
                components=signal_components
            )
            
            self.performance_stats['total_signals'] += 1
            
            # 记录详细的信号信息
            self.logger.info(f"生成增强信号 {symbol}: {signal_type} "
                           f"(强度: {signal_strength:.3f}, 置信度: {confidence:.3f})")
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"信号生成错误 {symbol}: {e}")
            return None
    
    def calculate_signal_confidence(self, df: pd.DataFrame, signal: pd.Series) -> float:
        """计算信号置信度"""
        try:
            # 基于多个因素计算置信度
            confidence_factors = []
            
            # 1. 信号一致性
            if len(df) >= 10:
                recent_signals = [signal.get('composite_signal', 0) for _ in range(min(10, len(df)))]
                consistency = 1 - np.std(recent_signals) if len(recent_signals) > 1 else 0.5
                confidence_factors.append(consistency)
            
            # 2. 市场波动性
            if 'close' in df.columns and len(df) >= 20:
                returns = df['close'].pct_change().dropna()
                volatility = returns.std()
                vol_confidence = max(0, 1 - volatility * 10)  # 低波动性 = 高置信度
                confidence_factors.append(vol_confidence)
            
            # 3. 成交量确认
            volume_signal = signal.get('volume_signal', 0)
            volume_confidence = min(1, abs(volume_signal) * 2)
            confidence_factors.append(volume_confidence)
            
            # 4. 技术指标一致性
            technical_signals = [
                signal.get('momentum_signal', 0),
                signal.get('mean_reversion_signal', 0),
                signal.get('volatility_signal', 0)
            ]
            
            # 计算信号方向一致性
            positive_signals = sum(1 for s in technical_signals if s > 0.1)
            negative_signals = sum(1 for s in technical_signals if s < -0.1)
            
            if positive_signals > negative_signals:
                direction_confidence = positive_signals / len(technical_signals)
            elif negative_signals > positive_signals:
                direction_confidence = negative_signals / len(technical_signals)
            else:
                direction_confidence = 0.5
            
            confidence_factors.append(direction_confidence)
            
            # 综合置信度
            final_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            return max(0.1, min(0.95, final_confidence))
            
        except Exception as e:
            self.logger.error(f"置信度计算错误: {e}")
            return 0.5
    
    def calculate_enhanced_confidence(self, df: pd.DataFrame, technical_factors: Dict, 
                                   discovered_strategies: Dict) -> float:
        """计算增强的信号置信度 - 集成多种分析方法"""
        try:
            confidence_factors = []
            
            # 1. 技术因子置信度
            if technical_factors:
                # 动量因子置信度
                momentum_conf = min(1.0, abs(technical_factors.get('momentum_20', 0)) * 2)
                confidence_factors.append(momentum_conf)
                
                # RSI置信度 (极值区域置信度更高)
                rsi = technical_factors.get('rsi_14', 50)
                rsi_conf = max(abs(rsi - 50) / 50, 0.2)  # RSI偏离50越远置信度越高
                confidence_factors.append(rsi_conf)
                
                # 波动率置信度
                volatility = technical_factors.get('volatility_20', 0)
                vol_conf = max(0.2, 1 - volatility * 5)  # 适度波动性提供更高置信度
                confidence_factors.append(vol_conf)
            
            # 2. 策略发现置信度
            if discovered_strategies:
                # 使用最佳策略的夏普比率作为置信度指标
                best_sharpe = max(perf.sharpe_ratio for perf in discovered_strategies.values())
                strategy_conf = min(1.0, max(0.1, (best_sharpe + 1) / 3))  # 标准化夏普比率
                confidence_factors.append(strategy_conf)
                
                # 策略一致性 - 多个策略同向信号
                positive_strategies = sum(1 for perf in discovered_strategies.values() 
                                        if perf.total_return > 0)
                total_strategies = len(discovered_strategies)
                consistency_conf = abs(positive_strategies / total_strategies - 0.5) * 2
                confidence_factors.append(consistency_conf)
            
            # 3. 市场状态置信度
            if len(df) >= 20:
                # 趋势强度
                if 'close' in df.columns:
                    prices = df['close'].tail(20)
                    trend_strength = abs(np.corrcoef(range(len(prices)), prices)[0, 1])
                    confidence_factors.append(trend_strength)
                
                # 成交量确认
                if 'volume' in df.columns:
                    recent_volume = df['volume'].tail(5).mean()
                    avg_volume = df['volume'].tail(20).mean()
                    volume_conf = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5
                    confidence_factors.append(volume_conf)
            
            # 4. 数据质量置信度
            data_quality = 1.0
            if len(df) < 50:
                data_quality *= 0.8  # 数据不足降低置信度
            if df.isnull().sum().sum() > 0:
                data_quality *= 0.9  # 缺失数据降低置信度
            confidence_factors.append(data_quality)
            
            # 计算综合置信度
            final_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            
            # 应用置信度边界
            final_confidence = max(0.1, min(0.95, final_confidence))
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"计算增强置信度失败: {e}")
            return 0.5
    
    def process_signal(self, signal: TradingSignal):
        """处理交易信号"""
        try:
            # 风险检查
            if not self.risk_manager.check_signal_risk(signal, self.positions, self.daily_trades):
                self.logger.warning(f"信号被风险管理拒绝: {signal.symbol} {signal.signal_type}")
                return
            
            # 计算交易数量
            quantity = self.calculate_position_size(signal)
            if quantity <= 0:
                return
            
            # 检查日交易限制
            if self.daily_trades >= self.max_daily_trades:
                self.logger.warning("已达到日交易限制")
                return
            
            # 创建订单
            order = self.create_order(signal, quantity)
            if order:
                # 提交订单
                order_id = self.ib_adapter.place_order(
                    symbol=signal.symbol,
                    action=signal.signal_type,
                    quantity=quantity,
                    order_type='MKT'  # 市价单，确保快速执行
                )
                
                if order_id:
                    self.orders[order_id] = OrderInfo(
                        order_id=order_id,
                        symbol=signal.symbol,
                        action=signal.signal_type,
                        quantity=quantity,
                        order_type='MKT',
                        price=None,
                        status='Submitted',
                        filled_qty=0,
                        avg_fill_price=0.0,
                        timestamp=datetime.now()
                    )
                    
                    self.logger.info(f"订单已提交: {order_id} {signal.symbol} {signal.signal_type} {quantity}")
                    self.daily_trades += 1
                    
        except Exception as e:
            self.logger.error(f"信号处理错误: {e}")
    
    def calculate_position_size(self, signal: TradingSignal) -> int:
        """计算仓位大小"""
        try:
            # 基础仓位大小
            base_size = 100
            
            # 根据信号强度调整
            strength_multiplier = min(3.0, signal.strength * 2)
            
            # 根据置信度调整
            confidence_multiplier = signal.confidence
            
            # 根据当前持仓调整
            current_position = self.positions.get(signal.symbol)
            if current_position:
                # 如果已有持仓，减少新仓位
                if (signal.signal_type == 'BUY' and current_position.quantity > 0) or \
                   (signal.signal_type == 'SELL' and current_position.quantity < 0):
                    strength_multiplier *= 0.5
            
            # 计算最终数量
            quantity = int(base_size * strength_multiplier * confidence_multiplier)
            
            # 应用限制
            quantity = min(quantity, self.max_position_size)
            
            # 检查资金限制
            estimated_cost = quantity * signal.price
            account_info = self.ib_adapter.get_account_info()
            if account_info and 'AvailableFunds' in account_info:
                available_funds = float(account_info['AvailableFunds'])
                if estimated_cost > available_funds * 0.1:  # 不超过可用资金的10%
                    quantity = int(available_funds * 0.1 / signal.price)
            
            return max(0, quantity)
            
        except Exception as e:
            self.logger.error(f"仓位计算错误: {e}")
            return 0
    
    def create_order(self, signal: TradingSignal, quantity: int) -> Optional[Dict]:
        """创建订单"""
        try:
            order = {
                'symbol': signal.symbol,
                'action': signal.signal_type,
                'quantity': quantity,
                'order_type': 'MKT',
                'signal_strength': signal.strength,
                'signal_confidence': signal.confidence,
                'timestamp': signal.timestamp
            }
            return order
            
        except Exception as e:
            self.logger.error(f"订单创建错误: {e}")
            return None
    
    def on_tick_data(self, symbol: str, tick_data):
        """处理tick数据"""
        try:
            # 处理MarketData对象或字典格式的数据
            if hasattr(tick_data, 'bid'):  # MarketData对象
                price = tick_data.last or tick_data.bid or tick_data.ask or 0
                bid = tick_data.bid
                ask = tick_data.ask
                volume = tick_data.volume
                bid_size = tick_data.bid_size
                ask_size = tick_data.ask_size
            else:  # 字典格式
                price = tick_data.get('price', 0)
                bid = tick_data.get('bid', 0)
                ask = tick_data.get('ask', 0)
                volume = tick_data.get('volume', 0)
                bid_size = tick_data.get('bid_size', 0)
                ask_size = tick_data.get('ask_size', 0)
            
            # 创建标准化的市场数据字典用于增强数据模块
            tick_dict = {
                'price': price,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'bid_size': bid_size,
                'ask_size': ask_size
            }
            
            # 使用增强数据模块处理数据
            self.data_module.process_market_data(symbol, tick_dict)
            
            # 更新传统市场数据存储（保持兼容性）
            self.market_data[symbol].append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': price,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'type': 'tick'
            })
            
            # 实时因子计算
            if len(self.market_data[symbol]) >= 20:  # 确保有足够数据
                recent_data = list(self.market_data[symbol])[-20:]
                df = pd.DataFrame(recent_data)
                
                # 计算技术因子
                technical_factors = self.factor_analyzer.calculate_technical_factors(df)
                
                # 存储因子值
                for factor_name, factor_value in technical_factors.items():
                    self.data_module.store_factor_value(
                        symbol=symbol,
                        factor_name=factor_name,
                        factor_value=factor_value,
                        timestamp=datetime.now()
                    )
            
            # 发送到监控系统（转换为字典格式）
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.add_market_data(symbol, tick_dict)
                
        except Exception as e:
            self.logger.error(f"Tick数据处理错误: {e}")
    
    def on_bar_data(self, symbol: str, bar_data: Dict):
        """处理K线数据"""
        try:
            # 创建标准化的K线数据
            bar = BarData(
                timestamp=datetime.now(),
                symbol=symbol,
                open_price=bar_data.get('open', 0),
                high_price=bar_data.get('high', 0),
                low_price=bar_data.get('low', 0),
                close_price=bar_data.get('close', 0),
                volume=bar_data.get('volume', 0),
                vwap=bar_data.get('vwap', 0),
                count=bar_data.get('count', 0)
            )
            
            # 使用增强数据模块处理K线数据
            self.data_module.add_bar_data(bar)
            
            # 更新传统K线数据存储（保持兼容性）
            self.bar_data[symbol].append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'open': bar_data.get('open', 0),
                'high': bar_data.get('high', 0),
                'low': bar_data.get('low', 0),
                'close': bar_data.get('close', 0),
                'volume': bar_data.get('volume', 0),
                'type': 'bar'
            })
            
            # 策略发现和优化
            if len(self.bar_data[symbol]) >= 50:  # 确保有足够的历史数据
                recent_bars = list(self.bar_data[symbol])[-50:]
                df = pd.DataFrame(recent_bars)
                
                # 运行策略发现
                best_strategies = self.strategy_discovery.discover_strategies(df, symbol)
                
                # 如果发现了更好的策略，记录日志
                if best_strategies:
                    self.logger.info(f"为 {symbol} 发现了 {len(best_strategies)} 个潜在策略")
                    for strategy_name, performance in best_strategies.items():
                        self.logger.info(f"策略 {strategy_name}: 收益率 {performance.total_return:.2%}, "
                                       f"夏普比率 {performance.sharpe_ratio:.2f}")
            
            # 发送到监控系统
            if self.monitor_system:
                self.monitor_system.add_bar_data(symbol, bar_data)
                
        except Exception as e:
            self.logger.error(f"K线数据处理错误: {e}")
    
    def on_order_status(self, order_id: str, status: str, filled_qty: int, avg_fill_price: float):
        """处理订单状态更新"""
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                order.status = status
                order.filled_qty = filled_qty
                order.avg_fill_price = avg_fill_price
                
                self.logger.info(f"订单状态更新: {order_id} {status} {filled_qty}@{avg_fill_price}")
                
                # 如果订单完全成交，更新持仓
                if status == 'Filled' and filled_qty == order.quantity:
                    self.update_position_from_fill(order)
                    
        except Exception as e:
            self.logger.error(f"订单状态处理错误: {e}")
    
    def on_execution(self, execution_data: Dict):
        """处理成交回报"""
        try:
            trade = TradeExecution(
                timestamp=datetime.now(),
                symbol=execution_data.get('symbol', ''),
                action=execution_data.get('side', ''),
                quantity=execution_data.get('shares', 0),
                price=execution_data.get('price', 0),
                order_id=execution_data.get('orderId', ''),
                status='FILLED',
                pnl=0.0  # 将在后续计算
            )
            
            self.trades.append(trade)
            self.performance_stats['executed_trades'] += 1
            
            # 发送到监控系统
            if self.monitor:
                self.monitor.add_trade(trade)
            
            self.logger.info(f"交易执行: {trade.symbol} {trade.action} {trade.quantity}@{trade.price}")
            
        except Exception as e:
            self.logger.error(f"成交处理错误: {e}")
    
    def update_positions(self):
        """更新持仓信息"""
        try:
            positions = self.ib_adapter.get_positions()
            if positions:
                for pos_data in positions:
                    symbol = pos_data.get('symbol', '')
                    if symbol:
                        self.positions[symbol] = PositionInfo(
                            symbol=symbol,
                            quantity=pos_data.get('position', 0),
                            avg_price=pos_data.get('avgCost', 0),
                            market_value=pos_data.get('marketValue', 0),
                            unrealized_pnl=pos_data.get('unrealizedPNL', 0),
                            realized_pnl=pos_data.get('realizedPNL', 0)
                        )
                        
        except Exception as e:
            self.logger.error(f"持仓更新错误: {e}")
    
    def update_position_from_fill(self, order: OrderInfo):
        """从成交更新持仓"""
        try:
            symbol = order.symbol
            if symbol not in self.positions:
                self.positions[symbol] = PositionInfo(
                    symbol=symbol,
                    quantity=0,
                    avg_price=0,
                    market_value=0,
                    unrealized_pnl=0,
                    realized_pnl=0
                )
            
            position = self.positions[symbol]
            
            # 更新持仓数量和平均价格
            if order.action == 'BUY':
                new_qty = position.quantity + order.filled_qty
                if new_qty != 0:
                    position.avg_price = (position.avg_price * position.quantity + 
                                        order.avg_fill_price * order.filled_qty) / new_qty
                position.quantity = new_qty
            else:  # SELL
                position.quantity -= order.filled_qty
                
                # 计算已实现盈亏
                if position.quantity >= 0:
                    realized_pnl = (order.avg_fill_price - position.avg_price) * order.filled_qty
                    position.realized_pnl += realized_pnl
                    self.daily_pnl += realized_pnl
                    self.total_pnl += realized_pnl
            
        except Exception as e:
            self.logger.error(f"持仓更新错误: {e}")
    
    def process_market_data(self):
        """处理市场数据"""
        # 这里可以添加额外的市场数据处理逻辑
        pass
    
    def update_risk_metrics(self):
        """更新风险指标"""
        try:
            # 计算组合价值
            portfolio_value = sum(pos.market_value for pos in self.positions.values())
            
            # 计算总敞口
            total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
            
            # 计算最大回撤
            # 这里简化处理，实际应该基于历史净值计算
            max_drawdown = min(0, self.daily_pnl / max(1, portfolio_value) * 100)
            
            # 创建风险指标
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                cash=0,  # 需要从账户信息获取
                total_exposure=total_exposure,
                max_drawdown=max_drawdown,
                var_1d=portfolio_value * 0.02,  # 简化的VaR计算
                sharpe_ratio=self.calculate_sharpe_ratio(),
                positions={pos.symbol: pos.quantity for pos in self.positions.values()}
            )
            
            # 发送到监控系统
            if self.monitor:
                self.monitor.add_risk_metrics(risk_metrics)
                
        except Exception as e:
            self.logger.error(f"风险指标更新错误: {e}")
    
    def calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        try:
            if len(self.trades) < 10:
                return 0.0
            
            # 计算交易收益率
            returns = [trade.pnl for trade in list(self.trades)[-50:]]  # 最近50笔交易
            
            if len(returns) < 2:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # 简化的夏普比率计算
            sharpe = mean_return / std_return * np.sqrt(252)  # 年化
            return sharpe
            
        except Exception as e:
            self.logger.error(f"夏普比率计算错误: {e}")
            return 0.0
    
    def process_order_updates(self):
        """处理订单更新"""
        # 检查订单状态，处理超时订单等
        current_time = datetime.now()
        for order_id, order in list(self.orders.items()):
            if order.status == 'Submitted':
                # 检查订单是否超时（5分钟）
                if (current_time - order.timestamp).total_seconds() > 300:
                    self.logger.warning(f"订单超时，尝试取消: {order_id}")
                    self.ib_adapter.cancel_order(order_id)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'total_signals': self.performance_stats['total_signals'],
            'executed_trades': self.performance_stats['executed_trades'],
            'execution_rate': self.performance_stats['executed_trades'] / max(1, self.performance_stats['total_signals']),
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'positions': len(self.positions),
            'active_orders': len([o for o in self.orders.values() if o.status == 'Submitted']),
            'sharpe_ratio': self.calculate_sharpe_ratio()
        }

class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_position_value = config.get('max_position_value', 50000)
        self.max_daily_loss = config.get('max_daily_loss', 5000)
        self.max_symbol_exposure = config.get('max_symbol_exposure', 20000)
        self.logger = logging.getLogger(__name__)
    
    def check_signal_risk(self, signal: TradingSignal, positions: Dict[str, PositionInfo], 
                         daily_trades: int) -> bool:
        """检查信号风险"""
        try:
            # 检查信号置信度
            if signal.confidence < 0.6:
                return False
            
            # 检查单个品种敞口
            current_position = positions.get(signal.symbol)
            if current_position:
                current_exposure = abs(current_position.market_value)
                if current_exposure > self.max_symbol_exposure:
                    return False
            
            # 检查日交易次数
            if daily_trades > 100:  # 日交易限制
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"风险检查错误: {e}")
            return False

if __name__ == "__main__":
    # 配置参数
    config = {
        'ib_host': '127.0.0.1',
        'ib_port': 7497,
        'client_id': 1,
        # 使用NASDAQ100股票列表 - 可以选择不同的配置
        'symbols': get_top_symbols(20),  # 前20个最重要的股票
        # 'symbols': get_symbols_by_category('tech_giants'),  # 科技巨头
        # 'symbols': get_symbols_by_category('high_volatility'),  # 高波动性股票
        # 'symbols': get_nasdaq100_symbols(),  # 完整NASDAQ100列表
        'max_position_size': 1000,
        'max_daily_trades': 100,
        'risk_limit': 10000,
        'enable_monitor': True,
        'risk_config': {
            'max_position_value': 50000,
            'max_daily_loss': 5000,
            'max_symbol_exposure': 20000
        }
    }
    
    # 设置增强的日志格式
    setup_enhanced_logging()
    
    # 显示系统启动信息
    console_formatter.print_title("🏛️ Citadel-IB 量化交易系统")
    console_formatter.print_section("📋 系统配置")
    console_formatter.print_key_value("IB连接", f"{config['ib_host']}:{config['ib_port']}")
    console_formatter.print_key_value("客户端ID", str(config['client_id']))
    console_formatter.print_key_value("监控股票数量", f"{len(config['symbols'])} 个NASDAQ100股票")
    console_formatter.print_key_value("最大仓位", f"{config['max_position_size']} 股")
    console_formatter.print_key_value("日交易限制", f"{config['max_daily_trades']} 笔")
    console_formatter.print_key_value("风险限额", f"${config['risk_limit']:,}")
    
    # 创建集成系统
    integration = CitadelIBIntegration(config)
    
    try:
        # 启动系统
        if integration.start():
            console_formatter.print_title("✅ 系统启动成功")
            console_formatter.print_info("系统正在运行中，按 Ctrl+C 停止系统")
            console_formatter.print_separator()
            
            # 主循环
            while True:
                time.sleep(10)
                
                # 显示性能摘要
                summary = integration.get_performance_summary()
                console_formatter.print_performance_summary(summary)

    except KeyboardInterrupt:
        console_formatter.print_warning("\n⚠️ 收到停止信号，正在安全关闭系统...")
        integration.stop()
        console_formatter.print_success("✅ 系统已安全停止")
    except Exception as e:
        console_formatter.print_error(f"❌ 系统错误: {e}")
        integration.stop()