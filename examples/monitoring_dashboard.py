#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控面板和实时数据展示模块
提供Web界面监控交易系统状态和实时数据
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from collections import deque, defaultdict

# 导入配置和券商工厂
try:
    from config import config, load_config_from_file
    from broker_factory import broker_factory, TradingSystemInterface
    HAS_BROKER_FACTORY = True
    logger = logging.getLogger(__name__)
    logger.info("券商工厂模块加载成功")
except ImportError as e:
    print(f"警告: 券商工厂模块未找到 ({e})，将使用模拟数据")
    HAS_BROKER_FACTORY = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    error_count: int
    warning_count: int

@dataclass
class TradingMetrics:
    """交易指标"""
    timestamp: str
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_volume: float
    total_profit: float
    win_rate: float
    avg_profit_per_trade: float
    active_positions: int
    portfolio_value: float

@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    timestamp: str
    price: float
    volume: int
    change: float
    change_percent: float
    bid: float
    ask: float
    high_24h: float
    low_24h: float

@dataclass
class TradingInfo:
    """交易信息数据结构 - 替换原有的Alert类"""
    id: str
    timestamp: str
    type: str  # ORDER, RISK, MARKET, SYSTEM
    level: str  # INFO, SUCCESS, WARNING, ERROR, CRITICAL
    category: str  # 具体分类
    title: str  # 标题
    message: str  # 详细信息
    details: Dict[str, Any]  # 额外详情
    acknowledged: bool = False

@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    order_type: str  # MARKET/LIMIT/STOP
    status: str  # FILLED/PARTIAL/PENDING/CANCELLED
    timestamp: str
    profit_loss: Optional[float] = None

@dataclass
class RiskEvent:
    """风险事件"""
    event_id: str
    risk_type: str  # POSITION_LIMIT, DRAWDOWN, VOLATILITY, CORRELATION
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    current_value: float
    threshold_value: float
    timestamp: str
    action_required: bool = False

@dataclass
class MarketEvent:
    """市场事件"""
    event_id: str
    event_type: str  # PRICE_MOVEMENT, VOLUME_SPIKE, NEWS, EARNINGS
    symbol: str
    description: str
    impact: str  # POSITIVE, NEGATIVE, NEUTRAL
    magnitude: float  # 影响程度 0-1
    timestamp: str
    related_positions: List[str] = None

class DataCollector:
    """数据收集器，负责收集系统指标、交易数据和市场数据"""
    
    def __init__(self, trading_system: Optional[TradingSystemInterface] = None):
        self.system_metrics = deque(maxlen=1000)
        self.trading_metrics = deque(maxlen=1000)
        self.market_data = {}
        self.trading_info = deque(maxlen=500)  # 替换alerts为trading_info
        self.is_running = False
        self.collection_thread = None
        self.trading_system = trading_system
        
        # 监控股票列表
        self.watchlist = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    def start_collection(self):
        """开始数据收集"""
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collect_data)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("数据收集器已启动")
    
    def stop_collection(self):
        """停止数据收集"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("数据收集器已停止")
    
    def _collect_data(self):
        """数据收集主循环"""
        while self.is_running:
            try:
                logger.debug("开始数据收集循环")
                
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                logger.debug(f"系统指标收集完成: CPU {system_metrics.cpu_usage:.1f}%")
                self.system_metrics.append(system_metrics)
                
                # 收集交易指标
                trading_metrics = self._collect_trading_metrics()
                logger.debug(f"交易指标收集完成: 投资组合价值 ${trading_metrics.portfolio_value:,.0f}")
                self.trading_metrics.append(trading_metrics)
                
                # 收集市场数据
                self._collect_market_data()
                
                # 收集交易信息
                self._collect_trading_info(system_metrics, trading_metrics)
                
                logger.debug("数据收集循环完成")
                
                time.sleep(5)  # 每5秒收集一次数据
                
            except Exception as e:
                import traceback
                logger.error(f"数据收集错误: {e}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                time.sleep(10)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            import psutil
            
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # 网络IO
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # 活跃连接数
            try:
                active_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, PermissionError, psutil.NoSuchProcess):
                # 在 macOS 上可能需要特殊权限，使用模拟值
                active_connections = np.random.randint(10, 50)
            except Exception:
                # 其他异常情况，使用模拟值
                active_connections = np.random.randint(10, 50)
            
        except ImportError:
            # 如果psutil不可用，使用模拟数据
            cpu_usage = np.random.uniform(10, 80)
            memory_usage = np.random.uniform(30, 70)
            disk_usage = np.random.uniform(20, 60)
            network_io = {
                'bytes_sent': np.random.randint(1000000, 10000000),
                'bytes_recv': np.random.randint(1000000, 10000000),
                'packets_sent': np.random.randint(1000, 10000),
                'packets_recv': np.random.randint(1000, 10000)
            }
            active_connections = np.random.randint(10, 100)
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            active_connections=active_connections,
            error_count=np.random.randint(0, 5),
            warning_count=np.random.randint(0, 10)
        )
    
    def _collect_trading_metrics(self) -> TradingMetrics:
        """收集交易指标"""
        if self.trading_system and HAS_BROKER_FACTORY:
            try:
                # 尝试从真实的交易系统获取数据
                logger.debug("尝试获取投资组合状态...")
                portfolio_status = self.trading_system.get_portfolio_status()
                logger.debug(f"投资组合状态: {portfolio_status}")
                
                logger.debug("尝试获取详细持仓...")
                positions = self.trading_system.get_detailed_positions()
                logger.debug(f"持仓数量: {len(positions) if positions else 0}")
                
                logger.debug("尝试计算投资组合绩效...")
                performance = self.trading_system.calculate_portfolio_performance()
                logger.debug(f"绩效数据: {performance}")
                
                # 计算交易指标
                total_trades = len(self.trading_system.trade_history) if hasattr(self.trading_system, 'trade_history') else 0
                successful_trades = len([t for t in self.trading_system.trade_history if t.get('status') == 'filled']) if hasattr(self.trading_system, 'trade_history') else 0
                failed_trades = total_trades - successful_trades
                
                return TradingMetrics(
                    timestamp=datetime.now().isoformat(),
                    total_trades=total_trades,
                    successful_trades=successful_trades,
                    failed_trades=failed_trades,
                    total_volume=sum([p.get('market_value', 0) for p in positions]) if positions else 0,
                    total_profit=performance.get('total_return', 0) if performance else 0,
                    win_rate=successful_trades / total_trades if total_trades > 0 else 0,
                    avg_profit_per_trade=performance.get('avg_return_per_trade', 0) if performance else 0,
                    active_positions=len(positions) if positions else 0,
                    portfolio_value=portfolio_status.get('total_value', 0) if portfolio_status else 0
                )
            except Exception as e:
                logger.warning(f"获取真实交易数据失败，使用模拟数据: {str(e)}")
        
        # 模拟交易数据（作为备用）
        logger.debug("使用模拟交易数据")
        total_trades = np.random.randint(100, 1000)
        successful_trades = int(total_trades * np.random.uniform(0.6, 0.9))
        failed_trades = total_trades - successful_trades
        
        return TradingMetrics(
            timestamp=datetime.now().isoformat(),
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            total_volume=np.random.uniform(1000000, 10000000),
            total_profit=np.random.uniform(-50000, 100000),
            win_rate=successful_trades / total_trades if total_trades > 0 else 0,
            avg_profit_per_trade=np.random.uniform(-100, 500),
            active_positions=np.random.randint(5, 50),
            portfolio_value=np.random.uniform(900000, 1100000)
        )
    
    def _collect_market_data(self):
        """收集市场数据"""
        if self.trading_system and HAS_BROKER_FACTORY:
            try:
                # 尝试从真实的交易系统获取市场数据
                market_data = self.trading_system.connector.get_market_data(self.watchlist)
                
                for symbol, data in market_data.items():
                    self.market_data[symbol] = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now().isoformat(),
                        price=data.get('price', 0),
                        volume=data.get('volume', 0),
                        change=data.get('change', 0),
                        change_percent=data.get('change_percent', 0),
                        bid=data.get('bid', 0),
                        ask=data.get('ask', 0),
                        high_24h=data.get('high', 0),
                        low_24h=data.get('low', 0)
                    )
                return
            except Exception as e:
                logger.warning(f"获取真实市场数据失败，使用模拟数据: {str(e)}")
        
        # 模拟市场数据（作为备用）
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
        
        for symbol in symbols:
            # 模拟市场数据
            base_price = np.random.uniform(100, 300)
            change = np.random.uniform(-10, 10)
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                price=base_price,
                volume=np.random.randint(1000000, 50000000),
                change=change,
                change_percent=change / base_price * 100,
                bid=base_price - np.random.uniform(0.01, 0.1),
                ask=base_price + np.random.uniform(0.01, 0.1),
                high_24h=base_price + np.random.uniform(0, 20),
                low_24h=base_price - np.random.uniform(0, 20)
            )
            
            self.market_data[symbol] = market_data
            
        logger.debug(f"生成了 {len(self.market_data)} 个股票的市场数据")
    
    def _collect_trading_info(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """收集交易信息，包括订单、风险事件、市场事件"""
        trading_info_list = []
        
        # 1. 生成订单信息
        self._generate_order_info(trading_metrics, trading_info_list)
        
        # 2. 生成风险事件
        self._generate_risk_events(system_metrics, trading_metrics, trading_info_list)
        
        # 3. 生成市场事件
        self._generate_market_events(trading_info_list)
        
        # 4. 生成系统信息
        self._generate_system_info(system_metrics, trading_info_list)
        
        # 添加到队列
        for info in trading_info_list:
            self.trading_info.append(info)
    
    def _generate_order_info(self, trading_metrics: TradingMetrics, trading_info_list: List[TradingInfo]):
        """生成订单相关信息"""
        current_time = datetime.now().isoformat()
        
        # 模拟重要订单信息
        if hasattr(self, '_last_order_check'):
            time_diff = time.time() - self._last_order_check
        else:
            time_diff = 60  # 首次运行
            
        self._last_order_check = time.time()
        
        # 每分钟可能生成订单信息
        if time_diff >= 30 and np.random.random() < 0.3:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN']
            symbol = np.random.choice(symbols)
            side = np.random.choice(['BUY', 'SELL'])
            quantity = np.random.randint(10, 500)
            price = np.random.uniform(100, 300)
            
            # 大订单信息
            if quantity >= 200:
                trading_info_list.append(TradingInfo(
                    id=f"order_{int(time.time())}_{np.random.randint(1000, 9999)}",
                    timestamp=current_time,
                    type="ORDER",
                    level="INFO",
                    category="LARGE_ORDER",
                    title=f"大额订单执行",
                    message=f"{side} {quantity} 股 {symbol} @ ${price:.2f}",
                    details={
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "order_value": quantity * price
                    }
                ))
            
            # 盈利订单信息
            if trading_metrics.total_profit > 0 and np.random.random() < 0.4:
                profit = np.random.uniform(500, 2000)
                trading_info_list.append(TradingInfo(
                    id=f"profit_{int(time.time())}_{np.random.randint(1000, 9999)}",
                    timestamp=current_time,
                    type="ORDER",
                    level="SUCCESS",
                    category="PROFITABLE_TRADE",
                    title=f"盈利交易完成",
                    message=f"{symbol} 交易获利 ${profit:.2f}",
                    details={
                        "symbol": symbol,
                        "profit": profit,
                        "return_rate": profit / (quantity * price) * 100
                    }
                ))
    
    def _generate_risk_events(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics, trading_info_list: List[TradingInfo]):
        """生成风险事件信息"""
        current_time = datetime.now().isoformat()
        
        # 仓位风险监控
        if trading_metrics.active_positions > 10:
            trading_info_list.append(TradingInfo(
                id=f"risk_position_{int(time.time())}",
                timestamp=current_time,
                type="RISK",
                level="WARNING",
                category="POSITION_RISK",
                title="仓位数量较高",
                message=f"当前持有 {trading_metrics.active_positions} 个活跃仓位，建议关注风险分散",
                details={
                    "active_positions": trading_metrics.active_positions,
                    "recommended_max": 10
                }
            ))
        
        # 胜率监控（改为正面信息）
        if trading_metrics.win_rate > 0.6:
            trading_info_list.append(TradingInfo(
                id=f"risk_winrate_{int(time.time())}",
                timestamp=current_time,
                type="RISK",
                level="SUCCESS",
                category="PERFORMANCE",
                title="交易表现优秀",
                message=f"当前胜率 {trading_metrics.win_rate:.1%}，表现良好",
                details={
                    "win_rate": trading_metrics.win_rate,
                    "total_trades": trading_metrics.total_trades
                }
            ))
        elif trading_metrics.win_rate < 0.4 and trading_metrics.total_trades > 10:
            trading_info_list.append(TradingInfo(
                id=f"risk_winrate_low_{int(time.time())}",
                timestamp=current_time,
                type="RISK",
                level="WARNING",
                category="PERFORMANCE",
                title="胜率需要关注",
                message=f"当前胜率 {trading_metrics.win_rate:.1%}，建议优化策略",
                details={
                    "win_rate": trading_metrics.win_rate,
                    "total_trades": trading_metrics.total_trades
                }
            ))
        
        # 系统资源监控
        if system_metrics.cpu_usage > 80:
            trading_info_list.append(TradingInfo(
                id=f"risk_cpu_{int(time.time())}",
                timestamp=current_time,
                type="RISK",
                level="WARNING",
                category="SYSTEM_RESOURCE",
                title="CPU使用率较高",
                message=f"CPU使用率 {system_metrics.cpu_usage:.1f}%，可能影响交易执行速度",
                details={
                    "cpu_usage": system_metrics.cpu_usage,
                    "threshold": 80
                }
            ))
    
    def _generate_market_events(self, trading_info_list: List[TradingInfo]):
        """生成市场事件信息"""
        current_time = datetime.now().isoformat()
        
        # 模拟市场事件
        if hasattr(self, '_last_market_event'):
            time_diff = time.time() - self._last_market_event
        else:
            time_diff = 120  # 首次运行
            
        # 每2分钟可能生成市场事件
        if time_diff >= 120 and np.random.random() < 0.2:
            self._last_market_event = time.time()
            
            events = [
                {
                    "type": "VOLUME_SPIKE",
                    "title": "交易量异常",
                    "message": "AAPL 交易量较平均水平增长 150%",
                    "impact": "POSITIVE",
                    "level": "INFO"
                },
                {
                    "type": "PRICE_MOVEMENT", 
                    "title": "价格突破",
                    "message": "TSLA 突破关键阻力位，上涨 3.2%",
                    "impact": "POSITIVE",
                    "level": "SUCCESS"
                },
                {
                    "type": "NEWS",
                    "title": "市场消息",
                    "message": "科技股普遍上涨，市场情绪乐观",
                    "impact": "POSITIVE",
                    "level": "INFO"
                }
            ]
            
            event = np.random.choice(events)
            trading_info_list.append(TradingInfo(
                id=f"market_{int(time.time())}_{np.random.randint(1000, 9999)}",
                timestamp=current_time,
                type="MARKET",
                level=event["level"],
                category=event["type"],
                title=event["title"],
                message=event["message"],
                details={
                    "impact": event["impact"],
                    "event_type": event["type"]
                }
            ))
    
    def _generate_system_info(self, system_metrics: SystemMetrics, trading_info_list: List[TradingInfo]):
        """生成系统信息"""
        current_time = datetime.now().isoformat()
        
        # 系统状态良好信息
        if system_metrics.cpu_usage < 50 and system_metrics.memory_usage < 70:
            if hasattr(self, '_last_system_good'):
                time_diff = time.time() - self._last_system_good
            else:
                time_diff = 300  # 首次运行
                
            # 每5分钟报告一次系统状态良好
            if time_diff >= 300:
                self._last_system_good = time.time()
                trading_info_list.append(TradingInfo(
                    id=f"system_good_{int(time.time())}",
                    timestamp=current_time,
                    type="SYSTEM",
                    level="SUCCESS",
                    category="SYSTEM_STATUS",
                    title="系统运行正常",
                    message=f"CPU: {system_metrics.cpu_usage:.1f}%, 内存: {system_metrics.memory_usage:.1f}%",
                    details={
                        "cpu_usage": system_metrics.cpu_usage,
                        "memory_usage": system_metrics.memory_usage,
                        "status": "healthy"
                    }
                ))

class MonitoringDashboard:
    """监控面板"""
    
    def __init__(self, host='localhost', port=5000, trading_system: Optional[TradingSystemInterface] = None):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'trading_system_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.data_collector = DataCollector(trading_system)
        
        self._setup_routes()
        self._setup_socketio_events()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/system_metrics')
        def get_system_metrics():
            """获取系统指标"""
            metrics = list(self.data_collector.system_metrics)
            return jsonify([asdict(m) for m in metrics[-50:]])  # 返回最近50个数据点
        
        @self.app.route('/api/trading_metrics')
        def get_trading_metrics():
            """获取交易指标"""
            metrics = list(self.data_collector.trading_metrics)
            return jsonify([asdict(m) for m in metrics[-50:]])
        
        @self.app.route('/api/market_data')
        def get_market_data():
            """获取市场数据"""
            return jsonify({k: asdict(v) for k, v in self.data_collector.market_data.items()})
        
        @self.app.route('/api/system-info')
        def get_system_info():
            """获取系统信息，包括是否为模拟数据模式"""
            # 检查是否为模拟数据模式：要么没有broker_factory，要么没有实际的trading_system
            is_simulation = not HAS_BROKER_FACTORY or self.data_collector.trading_system is None
            return jsonify({
                "simulation_mode": is_simulation,
                "broker_factory_available": HAS_BROKER_FACTORY,
                "trading_system_available": self.data_collector.trading_system is not None,
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/important_info')
        def get_important_info():
            """获取重要信息"""
            trading_info = list(self.data_collector.trading_info)
            # 按时间倒序排列，最新的在前面
            sorted_info = sorted(trading_info, key=lambda x: x.timestamp, reverse=True)
            return jsonify([asdict(info) for info in sorted_info[:50]])  # 返回最近50条
        
        @self.app.route('/api/important_info/<info_id>/acknowledge', methods=['POST'])
        def acknowledge_important_info(info_id):
            """确认重要信息"""
            for info in self.data_collector.trading_info:
                if info.id == info_id:
                    info.acknowledged = True
                    return jsonify({"status": "success"})
            return jsonify({"status": "error", "message": "重要信息未找到"}), 404
        
        @self.app.route('/api/important_info_history')
        def get_important_info_history():
            """获取完整的重要信息历史数据"""
            trading_info = list(self.data_collector.trading_info)
            # 按时间倒序排列，最新的在前面
            sorted_info = sorted(trading_info, key=lambda x: x.timestamp, reverse=True)
            return jsonify([asdict(info) for info in sorted_info])  # 返回所有历史数据
        
        @self.app.route('/api/orders')
        def get_orders():
            """获取订单明细数据"""
            # 生成模拟订单数据
            orders = []
            current_time = datetime.now()
            
            # 生成最近的订单数据
            for i in range(20):
                order_time = current_time - timedelta(minutes=i*5)
                order = OrderInfo(
                    order_id=f"ORD{1000+i:04d}",
                    symbol=np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META']),
                    side=np.random.choice(['BUY', 'SELL']),
                    quantity=np.random.randint(10, 500),
                    price=round(np.random.uniform(100, 300), 2),
                    order_type=np.random.choice(['MARKET', 'LIMIT', 'STOP']),
                    status=np.random.choice(['FILLED', 'PARTIAL', 'PENDING', 'CANCELLED'], p=[0.6, 0.2, 0.1, 0.1]),
                    timestamp=order_time.strftime('%Y-%m-%d %H:%M:%S'),
                    profit_loss=round(np.random.uniform(-500, 1000), 2) if np.random.random() > 0.3 else None
                )
                orders.append(asdict(order))
            
            # 按时间倒序排列
            orders.sort(key=lambda x: x['timestamp'], reverse=True)
            return jsonify(orders)
    
    def _setup_socketio_events(self):
        """设置WebSocket事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """客户端连接"""
            logger.info(f"客户端已连接: {request.sid}")
            emit('status', {'message': '连接成功'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """客户端断开连接"""
            logger.info(f"客户端已断开: {request.sid}")
        
        @self.socketio.on('request_data')
        def handle_request_data(data):
            """处理数据请求"""
            data_type = data.get('type')
            
            if data_type == 'system_metrics':
                metrics = list(self.data_collector.system_metrics)
                emit('system_metrics', [asdict(m) for m in metrics[-10:]])
            
            elif data_type == 'trading_metrics':
                metrics = list(self.data_collector.trading_metrics)
                emit('trading_metrics', [asdict(m) for m in metrics[-10:]])
            
            elif data_type == 'market_data':
                emit('market_data', {k: asdict(v) for k, v in self.data_collector.market_data.items()})
            
            elif data_type == 'important_info':
                trading_info = list(self.data_collector.trading_info)
                emit('important_info', [asdict(info) for info in trading_info[-20:]])
    
    def start_real_time_updates(self):
        """启动实时更新"""
        def update_clients():
            while True:
                try:
                    # 发送最新的系统指标
                    if self.data_collector.system_metrics:
                        latest_system = self.data_collector.system_metrics[-1]
                        self.socketio.emit('system_metrics_update', asdict(latest_system))
                    
                    # 发送最新的交易指标
                    if self.data_collector.trading_metrics:
                        latest_trading = self.data_collector.trading_metrics[-1]
                        self.socketio.emit('trading_metrics_update', asdict(latest_trading))
                    
                    # 发送市场数据
                    if self.data_collector.market_data:
                        self.socketio.emit('market_data_update', 
                                         {k: asdict(v) for k, v in self.data_collector.market_data.items()})
                    
                    # 发送新重要信息
                    if self.data_collector.trading_info:
                        recent_info = [info for info in self.data_collector.trading_info 
                                     if not info.acknowledged and 
                                     datetime.fromisoformat(info.timestamp) > datetime.now() - timedelta(minutes=5)]
                        if recent_info:
                            self.socketio.emit('new_important_info', [asdict(info) for info in recent_info])
                    
                    time.sleep(5)  # 每5秒更新一次
                    
                except Exception as e:
                    logger.error(f"实时更新错误: {e}")
                    time.sleep(10)
        
        update_thread = threading.Thread(target=update_clients)
        update_thread.daemon = True
        update_thread.start()
    
    def run(self, debug=False):
        """运行监控面板"""
        # 启动数据收集
        self.data_collector.start_collection()
        
        # 启动实时更新
        self.start_real_time_updates()
        
        logger.info(f"监控面板启动在 http://{self.host}:{self.port}")
        
        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            logger.info("正在关闭监控面板...")
        finally:
            self.data_collector.stop_collection()

# HTML模板
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IB TWS 量化交易系统监控面板</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
            overflow-x: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 300;
            margin: 0;
        }

        /* 响应式标题 */
        @media (max-width: 768px) {
            .header {
                padding: 0.75rem 1rem;
            }
            
            .header h1 {
                font-size: 1.5rem;
            }
        }

        @media (max-width: 480px) {
            .header h1 {
                font-size: 1.25rem;
            }
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 10px;
            animation: pulse 2s infinite;
        }
        
        .status-online {
            background-color: #4CAF50;
        }
        
        .status-offline {
            background-color: #f44336;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* 响应式网格布局 */
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.25rem;
                padding: 1.5rem;
            }
        }

        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
                gap: 1rem;
                padding: 1rem;
            }
        }

        @media (max-width: 480px) {
            .dashboard {
                padding: 0.75rem;
                gap: 0.75rem;
            }
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border: 1px solid rgba(0,0,0,0.05);
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.15);
        }

        /* 响应式卡片 */
        @media (max-width: 768px) {
            .card {
                padding: 1.25rem;
                border-radius: 8px;
            }
        }

        @media (max-width: 480px) {
            .card {
                padding: 1rem;
                border-radius: 6px;
            }
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            border-bottom: 2px solid #eee;
            padding-bottom: 0.5rem;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0.5rem 0;
            padding: 0.75rem;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 3px solid #667eea;
            transition: background-color 0.2s ease;
        }

        .metric:hover {
            background: #e9ecef;
        }

        .metric-label {
            font-weight: 500;
            color: #495057;
            font-size: 0.9rem;
        }

        .metric-value {
            font-weight: bold;
            color: #667eea;
            font-size: 1rem;
            text-align: right;
        }

        /* 动画效果 */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* 加载状态样式 */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* 状态样式 */
        .metric-warning {
            color: #ff9800 !important;
        }

        .metric-error {
            color: #f44336 !important;
        }

        .empty-state {
            padding: 2rem;
            text-align: center;
            color: #666;
        }

        .empty-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .error-state {
            padding: 2rem;
            text-align: center;
            color: #dc3545;
        }

        .error-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .error-message {
            font-weight: 500;
            margin-bottom: 0.5rem;
        }

        .error-details {
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 1rem;
        }

        .retry-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }

        .retry-btn:hover {
            background: #5a6fd8;
        }

        /* 响应式动画优化 */
        @media (prefers-reduced-motion: reduce) {
            .metric,
            .order-row {
                animation: none !important;
            }
            
            .loading-spinner {
                animation: none !important;
            }
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .alert {
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        
        .alert-success {
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }

        .alert-info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }

        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }

        .alert-error {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }

        .alert-critical {
            background-color: #f5c6cb;
            border-color: #dc3545;
            color: #721c24;
            animation: blink 1s infinite;
        }

        .important-info-item {
            padding: 0.15rem;
            margin: 0.05rem 0;
            border-radius: 3px;
            border: 1px solid #e9ecef;
            border-left: 2px solid transparent;
            background: #fff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            transition: box-shadow 0.15s ease, transform 0.15s ease, opacity 0.3s ease;
            overflow: hidden;
        }

        .important-info-item.new-item {
            animation: slideInFromTop 0.5s ease-out;
        }

        @keyframes slideInFromTop {
                0% {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                100% {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .more-info-container {
                text-align: center;
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #e0e0e0;
            }
            
            .more-info-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .more-info-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            
            .more-info-btn:active {
                transform: translateY(0);
            }
            
            /* 模态框样式 */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
                backdrop-filter: blur(5px);
            }
            
            .modal-content {
                background-color: #fefefe;
                margin: 2% auto;
                padding: 0;
                border: none;
                border-radius: 12px;
                width: 90%;
                max-width: 1000px;
                height: 90%;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }
            
            .modal-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .modal-header h2 {
                margin: 0;
                font-size: 20px;
                font-weight: 600;
            }
            
            .close {
                color: white;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
                line-height: 1;
                transition: opacity 0.3s ease;
            }
            
            .close:hover {
                opacity: 0.7;
            }
            
            .modal-body {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background-color: #f8f9fa;
            }
            
            .modal-controls {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                gap: 15px;
                flex-wrap: wrap;
            }
            
            .search-container {
                flex: 1;
                min-width: 200px;
            }
            
            /* 订单明细表格样式 */
            .orders-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .orders-table th,
            .orders-table td {
                padding: 12px 8px;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
                font-size: 13px;
            }
            
            .orders-table th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 11px;
            }
            
            .orders-table tbody tr:hover {
                background-color: #f8f9fa;
                transition: background-color 0.2s ease;
            }
            
            .orders-table tbody tr:last-child td {
                border-bottom: none;
            }
            
            /* 订单状态样式 */
            .status-badge {
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .status-filled {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status-pending {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }
            
            .status-partial {
                background-color: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            
            .status-cancelled {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            /* 买卖方向样式 */
            .side-buy {
                color: #28a745;
                font-weight: 600;
            }
            
            .side-sell {
                color: #dc3545;
                font-weight: 600;
            }
            
            /* 盈亏样式 */
            .profit-positive {
                color: #28a745;
                font-weight: 600;
            }
            
            .profit-negative {
                color: #dc3545;
                font-weight: 600;
            }
            
            /* 订单明细卡片特殊样式 */
            .orders-card {
                grid-column: span 2;
                min-height: 400px;
            }
            
            .orders-card .loading {
                text-align: center;
                color: #666;
                font-style: italic;
                padding: 2rem;
            }
            
            /* 响应式设计 */
            @media (max-width: 768px) {
                .orders-card {
                    grid-column: span 1;
                }
                
                .orders-table {
                    font-size: 11px;
                }
                
                .orders-table th,
                .orders-table td {
                    padding: 8px 4px;
                }
            }
            }
            
            .search-input {
                width: 100%;
                padding: 10px 15px;
                border: 1px solid #ddd;
                border-radius: 25px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.3s ease;
            }
            
            .search-input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .pagination-info {
                color: #666;
                font-size: 14px;
                white-space: nowrap;
            }
            
            .pagination-controls {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            
            .pagination-btn {
                background: #fff;
                border: 1px solid #ddd;
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s ease;
            }
            
            .pagination-btn:hover:not(:disabled) {
                background: #f0f0f0;
                border-color: #bbb;
            }
            
            .pagination-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .pagination-btn.active {
                background: #667eea;
                color: white;
                border-color: #667eea;
            }
            
            .modal-important-info {
                display: grid;
                gap: 12px;
            }
            
            .modal-important-item {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border-left: 4px solid #ddd;
                transition: all 0.3s ease;
            }
            
            .modal-important-item:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transform: translateY(-1px);
            }
            
            .modal-important-item.type-order {
                border-left-color: #28a745;
            }
            
            .modal-important-item.type-risk {
                border-left-color: #dc3545;
            }
            
            .modal-important-item.type-market {
                border-left-color: #007bff;
            }
            
            .modal-important-item.type-system {
                border-left-color: #6c757d;
            }
            
            .modal-important-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 8px;
            }
            
            .modal-important-type {
                background: #f8f9fa;
                color: #495057;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 500;
                text-transform: uppercase;
            }
            
            .modal-important-time {
                color: #6c757d;
                font-size: 12px;
            }
            
            .modal-important-title {
                font-weight: 600;
                color: #333;
                margin-bottom: 5px;
                font-size: 14px;
            }
            
            .modal-important-details {
                color: #666;
                font-size: 13px;
                line-height: 1.4;
            }

        .important-info-item:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }

        .important-info-item::before {
            content: none;
        }

        /* shimmer removed for flatter design */

        .important-info-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.1rem;
            gap: 0.2rem;
        }

        .important-info-type {
            font-size: 0.6rem;
            padding: 1px 3px;
            border-radius: 6px;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06);
            white-space: nowrap;
        }

        .type-order {
            background-color: #3b82f6;
        }

        .type-risk {
            background-color: #f59e0b;
            color: #fff;
        }

        .type-market {
            background-color: #10b981;
        }

        .type-system {
            background-color: #6b7280;
        }

        /* Flatten alert backgrounds on important-info cards while keeping border color */
        .important-info-item.alert-success,
        .important-info-item.alert-info,
        .important-info-item.alert-warning,
        .important-info-item.alert-error,
        .important-info-item.alert-critical {
            background: #fff !important;
            color: #2c3e50;
        }
        .important-info-item.alert-success { border-left-color: #28a745; }
        .important-info-item.alert-info { border-left-color: #17a2b8; }
        .important-info-item.alert-warning { border-left-color: #ffc107; }
        .important-info-item.alert-error { border-left-color: #dc3545; }
        .important-info-item.alert-critical { border-left-color: #dc3545; }

        .important-info-title {
            font-weight: 600;
            font-size: 0.75rem;
            color: #2c3e50;
            margin-bottom: 0.05rem;
            line-height: 1.0;
        }

        .important-info-time {
            font-size: 0.65rem;
            color: #6c757d;
            font-weight: 500;
            background: rgba(108, 117, 125, 0.08);
            padding: 1px 2px;
            border-radius: 2px;
            white-space: nowrap;
        }

        .important-info-message {
            color: #495057;
            line-height: 1.2;
            margin-bottom: 0.05rem;
            font-size: 0.7rem;
        }

        .important-info-details {
            margin-top: 0.05rem;
            font-size: 0.65rem;
            background: rgba(102, 126, 234, 0.04);
            padding: 0.15rem;
            border-radius: 3px;
            border: 1px solid rgba(102, 126, 234, 0.08);
        }

        .detail-item {
            display: inline-block;
            margin-right: 1.5rem;
            margin-bottom: 0.5rem;
            padding: 0.25rem 0.5rem;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 4px;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }

        .detail-label {
            font-weight: 600;
            color: #667eea;
            margin-right: 0.25rem;
        }

        .detail-value {
            color: #2c3e50;
            font-weight: 500;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.7; }
        }
        
        .market-data-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        .market-item {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
        }
        
        .market-symbol {
            font-weight: bold;
            font-size: 1.1rem;
            color: #333;
        }
        
        .market-price {
            font-size: 1.2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .market-change {
            font-size: 0.9rem;
        }
        
        .positive {
            color: #4CAF50;
        }
        
        .negative {
            color: #f44336;
        }
        
        .loading {
            text-align: center;
            color: #666;
            font-style: italic;
        }
        
        /* 加载动画样式 */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* 错误状态样式 */
        .error-state {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        
        .error-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .error-message {
            font-weight: bold;
            color: #f44336;
            margin-bottom: 0.5rem;
        }
        
        .error-details {
            font-size: 0.9rem;
            color: #999;
            margin-bottom: 1rem;
        }
        
        .retry-btn {
            background-color: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .retry-btn:hover {
            background-color: #5a6fd8;
        }
        
        /* 空状态样式 */
        .empty-state {
            text-align: center;
            color: #999;
            font-style: italic;
            padding: 1rem;
        }
        
        /* 风险警告样式 */
        .high-risk {
            background-color: #ffebee !important;
            border-left: 4px solid #f44336 !important;
        }
        
        .medium-risk {
            background-color: #fff3e0 !important;
            border-left: 4px solid #ff9800 !important;
        }
        
        .low-risk {
            background-color: #f3e5f5 !important;
            border-left: 4px solid #9c27b0 !important;
        }
        
        /* 数据更新动画 */
        .data-updating {
            opacity: 0.7;
            transition: opacity 0.3s ease;
        }
        
        .data-updated {
            animation: highlight 0.5s ease;
        }
        
        @keyframes highlight {
            0% { background-color: #e8f5e8; }
            100% { background-color: transparent; }
        }
        
        /* 缓存状态指示器 */
        .cache-indicator {
            position: absolute;
            top: 5px;
            right: 5px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #4caf50;
            opacity: 0.7;
        }
        
        .cache-indicator.stale {
            background-color: #ff9800;
        }
        
        .cache-indicator.expired {
            background-color: #f44336;
        }
        
        /* 性能监控指示器 */
        .performance-indicator {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            z-index: 1000;
            display: none;
        }
        
        .performance-indicator.show {
            display: block;
        }
        
        /* 批量更新优化样式 */
        .batch-updating {
            pointer-events: none;
            opacity: 0.8;
        }
        
        .batch-updating::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            animation: shimmer 1.5s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        /* 响应式设计增强 */
        @media (max-width: 768px) {
            .error-state {
                padding: 1rem;
            }
            
            .error-icon {
                font-size: 1.5rem;
            }
            
            .performance-indicator {
                bottom: 10px;
                right: 10px;
                font-size: 0.7rem;
            }
        }
        
        @media (max-width: 480px) {
            .retry-btn {
                padding: 0.4rem 0.8rem;
                font-size: 0.9rem;
            }
            
            .cache-indicator {
                width: 6px;
                height: 6px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>IB TWS 量化交易系统监控面板 <span id="status-indicator" class="status-indicator status-offline"></span></h1>
        <div id="simulation-mode-indicator" style="display: none; background-color: #ff9800; color: white; padding: 8px 16px; border-radius: 5px; margin-top: 10px; font-weight: bold; text-align: center;">
            ⚠️ 模拟数据模式 - 当前显示的是模拟数据
        </div>
    </div>
    
    <div class="dashboard">
        <!-- 系统指标 -->
        <div class="card">
            <h3>系统指标</h3>
            <div id="system-metrics">
                <div class="loading">加载中...</div>
            </div>
            <div class="chart-container">
                <canvas id="system-chart"></canvas>
            </div>
        </div>
        
        <!-- 交易指标 -->
        <div class="card">
            <h3>交易指标</h3>
            <div id="trading-metrics">
                <div class="loading">加载中...</div>
            </div>
            <div class="chart-container">
                <canvas id="trading-chart"></canvas>
            </div>
        </div>
        
        <!-- 市场数据 -->
        <div class="card">
            <h3>市场数据</h3>
            <div id="market-data" class="market-data-grid">
                <div class="loading">加载中...</div>
            </div>
        </div>
        
        <!-- 重要信息 -->
        <div class="card">
            <h3>重要信息</h3>
            <div id="important-info">
                <div class="loading">正在加载重要信息...</div>
            </div>
            <div class="more-info-container">
                <button id="more-info-btn" class="more-info-btn" onclick="openImportantInfoModal()">
                    查看更多历史信息
                </button>
            </div>
        </div>
        
        <!-- 订单明细 -->
        <div class="card">
            <h3>订单明细</h3>
            <div id="orders-container">
                <div class="loading">正在加载订单数据...</div>
            </div>
            <div class="orders-table-container" style="display: none;">
                <table class="orders-table">
                    <thead>
                        <tr>
                            <th>订单ID</th>
                            <th>股票代码</th>
                            <th>方向</th>
                            <th>数量</th>
                            <th>价格</th>
                            <th>类型</th>
                            <th>状态</th>
                            <th>时间</th>
                            <th>盈亏</th>
                        </tr>
                    </thead>
                    <tbody id="orders-table-body">
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <!-- 重要信息历史模态框 -->
    <div id="important-info-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>重要信息历史记录</h2>
                <span class="close" onclick="closeImportantInfoModal()">&times;</span>
            </div>
            <div class="modal-body">
                <div class="modal-controls">
                    <div class="search-container">
                        <input type="text" id="modal-search-input" class="search-input" placeholder="搜索重要信息..." onkeyup="filterModalImportantInfo()">
                    </div>
                    <div class="pagination-info">
                        <span id="pagination-info-text">显示 1-20 条，共 0 条</span>
                    </div>
                </div>
                <div id="modal-important-info" class="modal-important-info">
                    <div class="loading">正在加载历史信息...</div>
                </div>
                <div class="pagination-controls">
                    <button id="prev-page-btn" class="pagination-btn" onclick="changePage(-1)" disabled>上一页</button>
                    <span id="page-info">第 1 页</span>
                    <button id="next-page-btn" class="pagination-btn" onclick="changePage(1)" disabled>下一页</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket连接
        const socket = io();
        
        // 图表实例
        let systemChart, tradingChart;
        
        // 模态框相关变量
        let allImportantInfo = [];
        let filteredImportantInfo = [];
        let currentPage = 1;
        const itemsPerPage = 20;
        
        // 数据缓存机制
        const dataCache = {
            system_metrics: { data: null, timestamp: 0, ttl: 8000 },
            trading_metrics: { data: null, timestamp: 0, ttl: 12000 },
            market_data: { data: null, timestamp: 0, ttl: 3000 },
            orders: { data: null, timestamp: 0, ttl: 6000 },
            important_info: { data: null, timestamp: 0, ttl: 25000 }
        };

        // 检查缓存是否有效
        function isCacheValid(cacheKey) {
            const cache = dataCache[cacheKey];
            if (!cache || !cache.data) return false;
            return (Date.now() - cache.timestamp) < cache.ttl;
        }

        // 更新缓存
        function updateCache(cacheKey, data) {
            dataCache[cacheKey] = {
                data: data,
                timestamp: Date.now(),
                ttl: dataCache[cacheKey].ttl
            };
        }

        // 获取缓存数据
        function getCachedData(cacheKey) {
            return isCacheValid(cacheKey) ? dataCache[cacheKey].data : null;
        }

        // 批量DOM更新优化
        const domUpdateQueue = [];
        let domUpdateScheduled = false;

        function scheduleDOMUpdate(updateFunction) {
            domUpdateQueue.push(updateFunction);
            if (!domUpdateScheduled) {
                domUpdateScheduled = true;
                requestAnimationFrame(() => {
                    // 批量执行DOM更新
                    domUpdateQueue.forEach(fn => fn());
                    domUpdateQueue.length = 0;
                    domUpdateScheduled = false;
                });
            }
        }

        // 防抖函数
        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }

        // 节流函数
        function throttle(func, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            }
        }
        const updateConfig = {
            intervals: {
                system_metrics: 10000,    // 系统指标：10秒
                trading_metrics: 15000,   // 交易指标：15秒
                market_data: 5000,        // 市场数据：5秒（更频繁）
                orders: 8000,             // 订单数据：8秒
                important_info: 30000     // 重要信息：30秒
            },
            lastUpdate: {},
            timers: {},
            isVisible: true,
            retryCount: {},
            maxRetries: 3
        };

        // 页面可见性检测
        document.addEventListener('visibilitychange', function() {
            updateConfig.isVisible = !document.hidden;
            if (updateConfig.isVisible) {
                console.log('页面变为可见，恢复数据更新');
                startPeriodicUpdates();
            } else {
                console.log('页面不可见，暂停数据更新');
                stopPeriodicUpdates();
            }
        });

        // 启动周期性更新
        function startPeriodicUpdates() {
            if (!updateConfig.isVisible) return;
            
            // 系统指标更新
            updateConfig.timers.system_metrics = setInterval(() => {
                if (updateConfig.isVisible) {
                    socket.emit('request_data', {type: 'system_metrics'});
                }
            }, updateConfig.intervals.system_metrics);

            // 交易指标更新
            updateConfig.timers.trading_metrics = setInterval(() => {
                if (updateConfig.isVisible) {
                    socket.emit('request_data', {type: 'trading_metrics'});
                }
            }, updateConfig.intervals.trading_metrics);

            // 市场数据更新
            updateConfig.timers.market_data = setInterval(() => {
                if (updateConfig.isVisible) {
                    socket.emit('request_data', {type: 'market_data'});
                }
            }, updateConfig.intervals.market_data);

            // 订单数据更新
            updateConfig.timers.orders = setInterval(() => {
                if (updateConfig.isVisible) {
                    updateOrderData();
                }
            }, updateConfig.intervals.orders);

            // 重要信息更新
            updateConfig.timers.important_info = setInterval(() => {
                if (updateConfig.isVisible) {
                    socket.emit('request_data', {type: 'important_info'});
                }
            }, updateConfig.intervals.important_info);
        }

        // 停止周期性更新
        function stopPeriodicUpdates() {
            Object.values(updateConfig.timers).forEach(timer => {
                if (timer) clearInterval(timer);
            });
            updateConfig.timers = {};
        }

        // 连接状态
        socket.on('connect', function() {
            console.log('已连接到服务器');
            document.getElementById('status-indicator').className = 'status-indicator status-online';
            
            // 重置重试计数
            updateConfig.retryCount = {};
            
            // 检查是否为模拟数据模式
            checkSimulationMode();
            
            // 请求初始数据
            socket.emit('request_data', {type: 'system_metrics'});
            socket.emit('request_data', {type: 'trading_metrics'});
            socket.emit('request_data', {type: 'market_data'});
            socket.emit('request_data', {type: 'important_info'});
            
            // 请求订单数据
            updateOrderData();
            
            // 启动周期性更新
            setTimeout(startPeriodicUpdates, 2000); // 延迟2秒启动，避免初始数据冲突
        });

        // 检查模拟数据模式
        function checkSimulationMode() {
            fetch('/api/system-info')
                .then(response => response.json())
                .then(data => {
                    if (data.simulation_mode) {
                        document.getElementById('simulation-mode-indicator').style.display = 'block';
                    }
                })
                .catch(error => {
                    console.log('无法获取系统信息，假设为模拟模式');
                    document.getElementById('simulation-mode-indicator').style.display = 'block';
                });
        }
        
        socket.on('disconnect', function() {
            console.log('与服务器断开连接');
            document.getElementById('status-indicator').className = 'status-indicator status-offline';
            
            // 停止所有定时器
            stopPeriodicUpdates();
        });
        
        // 系统指标更新
        socket.on('system_metrics', function(data) {
            updateSystemMetrics(data);
            updateSystemChart(data);
        });
        
        socket.on('system_metrics_update', function(data) {
            updateSystemMetrics([data]);
            updateSystemChart([data]);
        });
        
        // 交易指标更新
        socket.on('trading_metrics', function(data) {
            updateTradingMetrics(data);
            updateTradingChart(data);
        });
        
        socket.on('trading_metrics_update', function(data) {
            updateTradingMetrics([data]);
            updateTradingChart([data]);
        });
        
        // 市场数据更新
        socket.on('market_data', function(data) {
            updateMarketData(data);
        });
        
        socket.on('market_data_update', function(data) {
            updateMarketData(data);
        });
        
        // 重要信息更新
        socket.on('important_info', function(data) {
            updateImportantInfo(data);
        });

        socket.on('new_important_info', function(data) {
            updateImportantInfo(data, true);
        });
        
        // 更新系统指标
        // 优化后的系统指标更新函数
        function updateSystemMetrics(data) {
            try {
                // 检查缓存
                const cachedData = getCachedData('system_metrics');
                if (cachedData && JSON.stringify(cachedData) === JSON.stringify(data)) {
                    return; // 数据未变化，跳过更新
                }

                // 更新缓存
                updateCache('system_metrics', data);

                const container = document.getElementById('system-metrics');
                if (!container) return;

                // 使用批量DOM更新
                scheduleDOMUpdate(() => {
                    if (!data || data.length === 0) {
                        container.innerHTML = `
                            <div class="empty-state" style="text-align: center; padding: 2rem; color: #666;">
                                <div class="empty-icon">📊</div>
                                <div>暂无系统指标数据</div>
                            </div>
                        `;
                        return;
                    }
                    
                    try {
                        const latest = data[data.length - 1];
                        
                        // 添加数据更新动画效果
                        container.style.opacity = '0.7';
                        
                        container.innerHTML = `
                            <div class="metric" style="animation: slideIn 0.3s ease-out;">
                                <span class="metric-label">CPU使用率</span>
                                <span class="metric-value ${latest.cpu_usage > 80 ? 'metric-warning' : ''}">${latest.cpu_usage.toFixed(1)}%</span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.1s;">
                                <span class="metric-label">内存使用率</span>
                                <span class="metric-value ${latest.memory_usage > 80 ? 'metric-warning' : ''}">${latest.memory_usage.toFixed(1)}%</span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.2s;">
                                <span class="metric-label">磁盘使用率</span>
                                <span class="metric-value ${latest.disk_usage > 80 ? 'metric-warning' : ''}">${latest.disk_usage.toFixed(1)}%</span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.3s;">
                                <span class="metric-label">活跃连接</span>
                                <span class="metric-value">${latest.active_connections}</span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.4s;">
                                <span class="metric-label">错误数</span>
                                <span class="metric-value ${latest.error_count > 0 ? 'metric-error' : ''}">${latest.error_count}</span>
                            </div>
                        `;
                        
                        // 恢复透明度
                        setTimeout(() => {
                            container.style.opacity = '1';
                        }, 100);
                        
                    } catch (error) {
                        console.error('更新系统指标失败:', error);
                        container.innerHTML = `
                            <div class="error-state" style="text-align: center; padding: 2rem; color: #dc3545;">
                                <div class="error-icon">⚠️</div>
                                <div class="error-message">系统指标加载失败</div>
                                <button class="retry-btn" onclick="socket.emit('request_system_metrics')">刷新页面</button>
                            </div>
                        `;
                        container.style.opacity = '1';
                    }
                });
            } catch (error) {
                console.error('系统指标更新异常:', error);
            }
        }
                        <div>系统指标更新失败</div>
                        <div style="font-size: 0.8rem; margin-top: 0.5rem; color: #666;">${error.message}</div>
                    </div>
                `;
            }
        }
        
        // 更新交易指标
        // 优化后的交易指标更新函数
        function updateTradingMetrics(data) {
            try {
                // 检查缓存
                const cachedData = getCachedData('trading_metrics');
                if (cachedData && JSON.stringify(cachedData) === JSON.stringify(data)) {
                    return; // 数据未变化，跳过更新
                }

                // 更新缓存
                updateCache('trading_metrics', data);

                const container = document.getElementById('trading-metrics');
                if (!container) return;

                // 使用批量DOM更新
                scheduleDOMUpdate(() => {
                    if (!data || data.length === 0) {
                        container.innerHTML = `
                            <div class="empty-state" style="text-align: center; padding: 2rem; color: #666;">
                                <div class="empty-icon">📈</div>
                                <div>暂无交易指标数据</div>
                            </div>
                        `;
                        return;
                    }
                    
                    try {
                        const latest = data[data.length - 1];
                        
                        // 添加数据更新动画效果
                        container.style.opacity = '0.7';
                        
                        container.innerHTML = `
                            <div class="metric" style="animation: slideIn 0.3s ease-out;">
                                <span class="metric-label">总交易数</span>
                                <span class="metric-value">${latest.total_trades}</span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.1s;">
                                <span class="metric-label">成功交易</span>
                                <span class="metric-value">${latest.successful_trades}</span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.2s;">
                                <span class="metric-label">失败交易</span>
                                <span class="metric-value ${latest.failed_trades > 0 ? 'metric-warning' : ''}">${latest.failed_trades || 0}</span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.3s;">
                                <span class="metric-label">胜率</span>
                                <span class="metric-value ${latest.win_rate < 0.5 ? 'metric-warning' : ''}">${(latest.win_rate * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.4s;">
                                <span class="metric-label">总盈利</span>
                                <span class="metric-value ${latest.total_profit >= 0 ? 'profit-positive' : 'profit-negative'}">
                                    $${latest.total_profit.toLocaleString()}
                                </span>
                            </div>
                            <div class="metric" style="animation: slideIn 0.3s ease-out 0.5s;">
                                <span class="metric-label">投资组合价值</span>
                                <span class="metric-value">$${latest.portfolio_value.toLocaleString()}</span>
                            </div>
                        `;
                        
                        // 恢复透明度
                        setTimeout(() => {
                            container.style.opacity = '1';
                        }, 100);
                        
                    } catch (error) {
                        console.error('更新交易指标失败:', error);
                        container.innerHTML = `
                            <div class="error-state" style="text-align: center; padding: 2rem; color: #dc3545;">
                                <div class="error-icon">⚠️</div>
                                <div class="error-message">交易指标加载失败</div>
                                <div class="error-details" style="font-size: 0.8rem; margin-top: 0.5rem; color: #666;">${error.message}</div>
                                <button class="retry-btn" onclick="socket.emit('request_trading_metrics')" style="margin-top: 1rem;">重试</button>
                            </div>
                        `;
                        container.style.opacity = '1';
                    }
                });
            } catch (error) {
                console.error('交易指标更新异常:', error);
            }
        }
        
        // 更新市场数据
        function updateMarketData(data) {
            const container = document.getElementById('market-data');
            
            try {
                // 检查缓存，避免重复更新
                const cacheKey = 'market_data';
                const cachedData = getFromCache(cacheKey);
                if (cachedData && JSON.stringify(cachedData) === JSON.stringify(data)) {
                    return; // 数据未变化，跳过更新
                }
                
                // 更新缓存
                updateCache(cacheKey, data);
                
                // 显示加载状态
                container.style.opacity = '0.6';
                
                if (!data || Object.keys(data).length === 0) {
                    // 批量DOM更新
                    scheduleDOMUpdate(() => {
                        container.innerHTML = `
                            <div class="empty-state">
                                <div class="empty-icon">📊</div>
                                <div>暂无市场数据</div>
                                <div class="empty-hint">等待市场数据更新...</div>
                            </div>
                        `;
                        container.style.opacity = '1';
                    });
                    return;
                }
                
                // 批量DOM更新
                scheduleDOMUpdate(() => {
                    let html = '';
                    for (const [symbol, info] of Object.entries(data)) {
                        const changeClass = info.change >= 0 ? 'positive' : 'negative';
                        const changeSign = info.change >= 0 ? '+' : '';
                        
                        // 添加价格变化警告样式
                        let priceClass = '';
                        if (Math.abs(info.change_percent) > 5) {
                            priceClass = 'metric-warning';
                        }
                        if (Math.abs(info.change_percent) > 10) {
                            priceClass = 'metric-error';
                        }
                        
                        // 添加成交量警告
                        let volumeClass = '';
                        if (info.volume > 1000000) {
                            volumeClass = 'high-volume';
                        }
                        
                        html += `
                            <div class="market-item" style="animation: slideIn 0.3s ease-out;">
                                <div class="market-symbol">${symbol}</div>
                                <div class="market-price ${priceClass}">$${info.price.toFixed(2)}</div>
                                <div class="market-change ${changeClass}">
                                    ${changeSign}${info.change.toFixed(2)} (${changeSign}${info.change_percent.toFixed(2)}%)
                                </div>
                                <div class="market-volume ${volumeClass}">
                                    成交量: ${info.volume.toLocaleString()}
                                </div>
                            </div>
                        `;
                    }
                    
                    container.innerHTML = html;
                    container.style.opacity = '1';
                });
                
            } catch (error) {
                console.error('更新市场数据失败:', error);
                scheduleDOMUpdate(() => {
                    container.innerHTML = `
                        <div class="error-state">
                            <div class="error-icon">⚠️</div>
                            <div class="error-message">市场数据更新失败</div>
                            <div class="error-details">${error.message}</div>
                            <button class="retry-btn" onclick="socket.emit('request_market_data')" style="margin-top: 1rem;">重新获取</button>
                        </div>
                    `;
                    container.style.opacity = '1';
                });
            }
        }
                        <div class="error-message">更新市场数据失败</div>
                        <div class="error-details">${error.message}</div>
                        <button class="retry-btn" onclick="location.reload()">刷新页面</button>
                    </div>
                `;
                container.style.opacity = '1';
            }
        }
        
        // 重要信息最多显示条数
        const MAX_IMPORTANT_ITEMS = 20;
        // 更新重要信息
        function updateImportantInfo(data, isNew = false) {
            const container = document.getElementById('important-info');

            try {
                // 检查缓存，避免重复更新
                const cacheKey = 'important_info';
                if (!isNew) {
                    const cachedData = getFromCache(cacheKey);
                    if (cachedData && JSON.stringify(cachedData) === JSON.stringify(data)) {
                        return; // 数据未变化，无需更新
                    }
                }

                // 显示加载状态
                if (!container.querySelector('.loading-spinner')) {
                    container.style.opacity = '0.7';
                    container.style.transition = 'opacity 0.3s ease';
                }

                if (!data || data.length === 0) {
                    if (!isNew) {
                        scheduleDOMUpdate(() => {
                            container.innerHTML = '<div class="empty-state">📢 暂无重要信息</div>';
                            container.style.opacity = '1';
                        });
                    }
                    return;
                }

                // 更新缓存
                if (!isNew) {
                    updateCache(cacheKey, data);
                }

                // 按时间倒序排列数据
                const sortedData = data.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

                scheduleDOMUpdate(() => {
                    let html = '';
                    sortedData.forEach((info, index) => {
                        const levelClass = `alert-${info.level.toLowerCase()}`;
                        const typeClass = `type-${info.type.toLowerCase()}`;
                        const time = new Date(info.timestamp).toLocaleTimeString();
                        
                        // 根据级别添加警告样式
                        let additionalClass = '';
                        if (info.level === 'ERROR' || info.level === 'CRITICAL') {
                            additionalClass = 'metric-error';
                        } else if (info.level === 'WARNING') {
                            additionalClass = 'metric-warning';
                        }

                        // 构建详情信息
                        let detailsHtml = '';
                        if (info.details && Object.keys(info.details).length > 0) {
                            for (const [key, value] of Object.entries(info.details)) {
                                if (value !== null && value !== undefined) {
                                    detailsHtml += `
                                        <span class="detail-item">
                                            <span class="detail-label">${key}:</span>
                                            <span class="detail-value">${value}</span>
                                        </span>
                                    `;
                                }
                            }
                        }

                        // 为新信息添加动画类
                        const newItemClass = (isNew && index === 0) ? ' new-item' : '';

                        html += `
                            <div class="important-item ${levelClass} ${additionalClass}${newItemClass}" style="animation: fadeIn 0.3s ease-in;">
                                <div class="important-header">
                                    <div>
                                        <span class="important-type ${typeClass}">${info.type}</span>
                                        <span class="important-title">${info.title}</span>
                                    </div>
                                    <span class="important-time">${time}</span>
                                </div>
                                <div class="important-message">${info.message}</div>
                                ${detailsHtml ? `<div class="important-details">${detailsHtml}</div>` : ''}
                            </div>
                        `;
                    });

                    container.innerHTML = html;
                    container.style.opacity = '1';
                });

            } catch (error) {
                console.error('重要信息更新异常:', error);
                scheduleDOMUpdate(() => {
                    container.innerHTML = `
                        <div class="error-state">
                            <div class="error-icon">⚠️</div>
                            <div class="error-message">重要信息更新失败</div>
                            <div class="error-details">${error.message}</div>
                            <button class="retry-btn" onclick="socket.emit('request_important_info')" style="margin-top: 1rem;">重试</button>
                        </div>
                    `;
                    container.style.opacity = '1';
                });
            }
        }

                    html += `
                        <div class="important-info-item ${levelClass} ${additionalClass}${newItemClass}" style="animation: slideIn 0.3s ease-out;">
                            <div class="important-info-header">
                                <div>
                                    <span class="important-info-type ${typeClass}">${info.type}</span>
                                    <span class="important-info-title">${info.title}</span>
                                </div>
                                <span class="important-info-time">${time}</span>
                            </div>
                            <div class="important-info-message">${info.message}</div>
                            ${detailsHtml ? `<div class="important-info-details">${detailsHtml}</div>` : ''}
                        </div>
                    `;
                });

                if (isNew) {
                    // 创建新的DOM元素并插入到顶部
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = html;
                    const newItems = tempDiv.querySelectorAll('.important-info-item');
                    
                    // 将新项目插入到容器顶部
                    newItems.forEach(item => {
                        container.insertBefore(item, container.firstChild);
                    });
                } else {
                    container.innerHTML = html;
                }
                
                // 保持重要信息最多显示 MAX_IMPORTANT_ITEMS 条，移除更早的项
                const items = container.querySelectorAll('.important-info-item');
                if (items.length > MAX_IMPORTANT_ITEMS) {
                    for (let i = MAX_IMPORTANT_ITEMS; i < items.length; i++) {
                        items[i].remove();
                    }
                }

                // 移除动画类，避免重复触发
                setTimeout(() => {
                    const newItems = container.querySelectorAll('.important-info-item.new-item');
                    newItems.forEach(item => {
                        item.classList.remove('new-item');
                    });
                }, 500);

                // 恢复正常透明度
                container.style.opacity = '1';

            } catch (error) {
                console.error('更新重要信息失败:', error);
                container.innerHTML = `
                    <div class="error-state">
                        <div class="error-icon">⚠️</div>
                        <div class="error-message">更新重要信息失败</div>
                        <div class="error-details">${error.message}</div>
                        <button class="retry-btn" onclick="location.reload()" style="margin-top: 1rem;">刷新页面</button>
                    </div>
                `;
                container.style.opacity = '1';
            }
        }
        
        // 初始化图表
        function initCharts() {
            // 系统指标图表
            const systemCtx = document.getElementById('system-chart').getContext('2d');
            systemChart = new Chart(systemCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU使用率',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }, {
                        label: '内存使用率',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // 交易指标图表
            const tradingCtx = document.getElementById('trading-chart').getContext('2d');
            tradingChart = new Chart(tradingCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '投资组合价值',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
        
        // 更新系统图表
        function updateSystemChart(data) {
            if (!systemChart || !data || data.length === 0) return;
            
            data.forEach(item => {
                const time = new Date(item.timestamp).toLocaleTimeString();
                systemChart.data.labels.push(time);
                systemChart.data.datasets[0].data.push(item.cpu_usage);
                systemChart.data.datasets[1].data.push(item.memory_usage);
                
                // 保持最多20个数据点
                if (systemChart.data.labels.length > 20) {
                    systemChart.data.labels.shift();
                    systemChart.data.datasets[0].data.shift();
                    systemChart.data.datasets[1].data.shift();
                }
            });
            
            systemChart.update();
        }
        
        // 更新交易图表
        function updateTradingChart(data) {
            if (!tradingChart || !data || data.length === 0) return;
            
            data.forEach(item => {
                const time = new Date(item.timestamp).toLocaleTimeString();
                tradingChart.data.labels.push(time);
                tradingChart.data.datasets[0].data.push(item.portfolio_value);
                
                // 保持最多20个数据点
                if (tradingChart.data.labels.length > 20) {
                    tradingChart.data.labels.shift();
                    tradingChart.data.datasets[0].data.shift();
                }
            });
            
            tradingChart.update();
        }
        
        // 页面加载完成后初始化图表
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
        });
        
        // 更新订单数据
        function updateOrderData() {
            const tbody = document.getElementById('orders-table-body');
            
            try {
                // 检查缓存，避免重复请求
                const cacheKey = 'orders';
                const cachedData = getFromCache(cacheKey);
                if (cachedData) {
                    renderOrderData(cachedData, tbody);
                    return;
                }
                
                // 显示加载状态
                tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #667eea;"><div class="loading-spinner"></div>正在加载订单数据...</td></tr>';
                
                fetch('/api/orders')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        // 更新缓存
                        updateCache(cacheKey, data);
                        renderOrderData(data, tbody);
                    })
                    .catch(error => {
                        console.error('获取订单数据失败:', error);
                        scheduleDOMUpdate(() => {
                            tbody.innerHTML = `
                                <tr>
                                    <td colspan="9" style="text-align: center; padding: 2rem;">
                                        <div class="error-state">
                                            <div class="error-icon">⚠️</div>
                                            <div class="error-message">获取订单数据失败</div>
                                            <div class="error-details">${error.message}</div>
                                            <button class="retry-btn" onclick="updateOrderData()" style="margin-top: 1rem;">重试</button>
                                        </div>
                                    </td>
                                </tr>
                            `;
                        });
                    });
            } catch (error) {
                console.error('订单数据更新异常:', error);
                scheduleDOMUpdate(() => {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="9" style="text-align: center; padding: 2rem;">
                                <div class="error-state">
                                    <div class="error-icon">⚠️</div>
                                    <div class="error-message">订单数据更新异常</div>
                                    <div class="error-details">${error.message}</div>
                                    <button class="retry-btn" onclick="updateOrderData()" style="margin-top: 1rem;">重试</button>
                                </div>
                            </td>
                        </tr>
                    `;
                });
            }
        }
        
        // 渲染订单数据的辅助函数
        function renderOrderData(data, tbody) {
            scheduleDOMUpdate(() => {
                if (!data || data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #666;"><div class="empty-state">📋 暂无订单数据</div></td></tr>';
                    return;
                }

                let html = '';
                data.forEach(order => {
                    const time = new Date(order.timestamp).toLocaleString();
                    const statusClass = order.status === 'FILLED' ? 'status-filled' : 
                                      order.status === 'PENDING' ? 'status-pending' : 
                                      order.status === 'CANCELLED' ? 'status-cancelled' : 'status-partial';
                    
                    const profitLossClass = order.profit_loss > 0 ? 'profit-positive' : 
                                          order.profit_loss < 0 ? 'profit-negative' : '';
                    
                    // 添加风险警告样式
                    let riskClass = '';
                    if (order.profit_loss < -1000) {
                        riskClass = 'high-risk';
                    } else if (order.profit_loss < -500) {
                        riskClass = 'medium-risk';
                    }
                    
                    html += `
                        <tr class="order-row ${riskClass}" style="animation: fadeIn 0.3s ease-in;">
                            <td>${order.order_id}</td>
                            <td>${order.symbol}</td>
                            <td class="side-${order.side.toLowerCase()}">${order.side}</td>
                            <td>${order.quantity}</td>
                            <td>$${order.price.toFixed(2)}</td>
                            <td>${order.order_type}</td>
                            <td><span class="status-badge ${statusClass}">${order.status}</span></td>
                            <td>${time}</td>
                            <td class="${profitLossClass}">
                                ${order.profit_loss !== null ? '$' + order.profit_loss.toFixed(2) : '-'}
                            </td>
                        </tr>
                    `;
                });

                tbody.innerHTML = html;
            });
        }
        
        // 定期更新订单数据 - 移除原有的setInterval，改用智能更新机制
        // setInterval(updateOrderData, 5000); // 已移除，使用新的智能更新机制
        
        // 模态框相关函数
        function openImportantInfoModal() {
            const modal = document.getElementById('important-info-modal');
            modal.style.display = 'block';
            
            // 请求完整的重要信息历史数据
            fetch('/api/important_info_history')
                .then(response => response.json())
                .then(data => {
                    allImportantInfo = data.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                    filteredImportantInfo = [...allImportantInfo];
                    currentPage = 1;
                    renderModalImportantInfo();
                })
                .catch(error => {
                    console.error('获取重要信息历史失败:', error);
                    document.getElementById('modal-important-info').innerHTML = 
                        '<div class="loading">获取历史信息失败</div>';
                });
        }
        
        function closeImportantInfoModal() {
            const modal = document.getElementById('important-info-modal');
            modal.style.display = 'none';
        }
        
        function filterModalImportantInfo() {
            const searchTerm = document.getElementById('modal-search-input').value.toLowerCase();
            
            if (searchTerm === '') {
                filteredImportantInfo = [...allImportantInfo];
            } else {
                filteredImportantInfo = allImportantInfo.filter(info => 
                    info.title.toLowerCase().includes(searchTerm) ||
                    info.message.toLowerCase().includes(searchTerm) ||
                    info.type.toLowerCase().includes(searchTerm) ||
                    info.category.toLowerCase().includes(searchTerm)
                );
            }
            
            currentPage = 1;
            renderModalImportantInfo();
        }
        
        function changePage(direction) {
            const totalPages = Math.ceil(filteredImportantInfo.length / itemsPerPage);
            const newPage = currentPage + direction;
            
            if (newPage >= 1 && newPage <= totalPages) {
                currentPage = newPage;
                renderModalImportantInfo();
            }
        }
        
        function renderModalImportantInfo() {
            const container = document.getElementById('modal-important-info');
            const totalItems = filteredImportantInfo.length;
            const totalPages = Math.ceil(totalItems / itemsPerPage);
            const startIndex = (currentPage - 1) * itemsPerPage;
            const endIndex = Math.min(startIndex + itemsPerPage, totalItems);
            const pageData = filteredImportantInfo.slice(startIndex, endIndex);
            
            if (totalItems === 0) {
                container.innerHTML = '<div class="loading">没有找到匹配的信息</div>';
                document.getElementById('pagination-info-text').textContent = '显示 0-0 条，共 0 条';
                document.getElementById('page-info').textContent = '第 0 页';
                document.getElementById('prev-page-btn').disabled = true;
                document.getElementById('next-page-btn').disabled = true;
                return;
            }
            
            let html = '';
            pageData.forEach(info => {
                const levelClass = `alert-${info.level.toLowerCase()}`;
                const typeClass = `type-${info.type.toLowerCase()}`;
                const time = new Date(info.timestamp).toLocaleString();
                
                // 构建详情信息
                let detailsHtml = '';
                if (info.details && Object.keys(info.details).length > 0) {
                    for (const [key, value] of Object.entries(info.details)) {
                        if (value !== null && value !== undefined) {
                            detailsHtml += `
                                <span class="detail-item">
                                    <span class="detail-label">${key}:</span>
                                    <span class="detail-value">${value}</span>
                                </span>
                            `;
                        }
                    }
                }
                
                html += `
                    <div class="modal-important-item ${levelClass}">
                        <div class="modal-important-header">
                            <div>
                                <span class="modal-important-type ${typeClass}">${info.type}</span>
                                <span class="modal-important-title">${info.title}</span>
                            </div>
                            <span class="modal-important-time">${time}</span>
                        </div>
                        <div class="important-info-message">${info.message}</div>
                        ${detailsHtml ? `<div class="modal-important-details">${detailsHtml}</div>` : ''}
                    </div>
                `;
            });
            
            container.innerHTML = html;
            
            // 更新分页信息
            document.getElementById('pagination-info-text').textContent = 
                `显示 ${startIndex + 1}-${endIndex} 条，共 ${totalItems} 条`;
            document.getElementById('page-info').textContent = `第 ${currentPage} 页`;
            
            // 更新分页按钮状态
            document.getElementById('prev-page-btn').disabled = currentPage <= 1;
            document.getElementById('next-page-btn').disabled = currentPage >= totalPages;
        }
        
        // 点击模态框外部关闭
        window.onclick = function(event) {
            const modal = document.getElementById('important-info-modal');
            if (event.target === modal) {
                closeImportantInfoModal();
            }
        }
    </script>
</body>
</html>
"""

def create_dashboard_template():
    """创建仪表板HTML模板文件"""
    import os
    
    # 创建templates目录在当前工作目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # 写入HTML模板
    template_path = os.path.join(templates_dir, 'dashboard.html')
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(DASHBOARD_HTML_TEMPLATE)
    
    logger.info(f"仪表板模板已创建: {template_path}")

# 示例使用
if __name__ == "__main__":
    # 尝试初始化交易系统
    trading_system = None
    if HAS_BROKER_FACTORY:
        try:
            # 使用券商工厂创建交易系统实例
            # trading_system = broker_factory.create_trading_system('firstrade', config)
            logger.info("券商工厂可用，但需要配置凭据")
        except Exception as e:
            logger.warning(f"初始化交易系统失败: {str(e)}")
    
    # 如果没有真实的交易系统，记录信息并使用模拟数据
    if trading_system is None:
        logger.info("使用模拟数据模式运行监控面板")
    
    # 创建HTML模板
    create_dashboard_template()
    
    # 启动监控面板
    dashboard = MonitoringDashboard(host='0.0.0.0', port=8080, trading_system=trading_system)
    
    try:
        dashboard.run(debug=False)
    except KeyboardInterrupt:
        print("\n监控面板已关闭")