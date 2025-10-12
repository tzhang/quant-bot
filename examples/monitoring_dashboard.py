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
class Alert:
    """警报"""
    id: str
    timestamp: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    category: str  # SYSTEM, TRADING, MARKET, RISK
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False

class DataCollector:
    """数据收集器，负责收集系统指标、交易数据和市场数据"""
    
    def __init__(self, trading_system: Optional[TradingSystemInterface] = None):
        self.system_metrics = deque(maxlen=1000)
        self.trading_metrics = deque(maxlen=1000)
        self.market_data = {}
        self.alerts = deque(maxlen=500)
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
                
                # 检查警报条件
                self._check_alerts(system_metrics, trading_metrics)
                
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
    
    def _check_alerts(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """检查警报条件"""
        alerts = []
        
        # 系统警报
        if system_metrics.cpu_usage > 80:
            alerts.append(Alert(
                id=f"cpu_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                level="WARNING",
                category="SYSTEM",
                message=f"CPU使用率过高: {system_metrics.cpu_usage:.1f}%",
                details={"cpu_usage": system_metrics.cpu_usage}
            ))
        
        if system_metrics.memory_usage > 85:
            alerts.append(Alert(
                id=f"memory_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                level="ERROR",
                category="SYSTEM",
                message=f"内存使用率过高: {system_metrics.memory_usage:.1f}%",
                details={"memory_usage": system_metrics.memory_usage}
            ))
        
        # 交易警报
        if trading_metrics.win_rate < 0.4:
            alerts.append(Alert(
                id=f"winrate_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                level="WARNING",
                category="TRADING",
                message=f"胜率过低: {trading_metrics.win_rate:.1%}",
                details={"win_rate": trading_metrics.win_rate}
            ))
        
        if trading_metrics.total_profit < -10000:
            alerts.append(Alert(
                id=f"loss_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                category="RISK",
                message=f"总亏损过大: ${trading_metrics.total_profit:,.2f}",
                details={"total_profit": trading_metrics.total_profit}
            ))
        
        # 添加警报到队列
        for alert in alerts:
            self.alerts.append(alert)

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
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """获取警报"""
            alerts = list(self.data_collector.alerts)
            # 按时间倒序排列，最新的在前面
            sorted_alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
            return jsonify([asdict(a) for a in sorted_alerts[:100]])  # 返回最近100个警报，按时间倒序
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            """确认警报"""
            for alert in self.data_collector.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return jsonify({"status": "success"})
            return jsonify({"status": "error", "message": "Alert not found"}), 404
    
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
            
            elif data_type == 'alerts':
                alerts = list(self.data_collector.alerts)
                emit('alerts', [asdict(a) for a in alerts[-20:]])
    
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
                    
                    # 发送新警报
                    if self.data_collector.alerts:
                        recent_alerts = [a for a in self.data_collector.alerts 
                                       if not a.acknowledged and 
                                       datetime.fromisoformat(a.timestamp) > datetime.now() - timedelta(minutes=5)]
                        if recent_alerts:
                            self.socketio.emit('new_alerts', [asdict(a) for a in recent_alerts])
                    
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
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2rem;
            font-weight: 300;
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
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
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
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .metric-label {
            font-weight: 500;
        }
        
        .metric-value {
            font-weight: bold;
            color: #667eea;
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
        
        <!-- 警报 -->
        <div class="card">
            <h3>系统警报</h3>
            <div id="alerts">
                <div class="loading">加载中...</div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket连接
        const socket = io();
        
        // 图表实例
        let systemChart, tradingChart;
        
        // 连接状态
        socket.on('connect', function() {
            console.log('已连接到服务器');
            document.getElementById('status-indicator').className = 'status-indicator status-online';
            
            // 检查是否为模拟数据模式
            checkSimulationMode();
            
            // 请求初始数据
            socket.emit('request_data', {type: 'system_metrics'});
            socket.emit('request_data', {type: 'trading_metrics'});
            socket.emit('request_data', {type: 'market_data'});
            socket.emit('request_data', {type: 'alerts'});
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
        
        // 警报更新
        socket.on('alerts', function(data) {
            updateAlerts(data);
        });
        
        socket.on('new_alerts', function(data) {
            updateAlerts(data, true);
        });
        
        // 更新系统指标
        function updateSystemMetrics(data) {
            if (!data || data.length === 0) return;
            
            const latest = data[data.length - 1];
            const container = document.getElementById('system-metrics');
            
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">CPU使用率</span>
                    <span class="metric-value">${latest.cpu_usage.toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">内存使用率</span>
                    <span class="metric-value">${latest.memory_usage.toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">磁盘使用率</span>
                    <span class="metric-value">${latest.disk_usage.toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">活跃连接</span>
                    <span class="metric-value">${latest.active_connections}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">错误数</span>
                    <span class="metric-value">${latest.error_count}</span>
                </div>
            `;
        }
        
        // 更新交易指标
        function updateTradingMetrics(data) {
            if (!data || data.length === 0) return;
            
            const latest = data[data.length - 1];
            const container = document.getElementById('trading-metrics');
            
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">总交易数</span>
                    <span class="metric-value">${latest.total_trades}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">成功交易</span>
                    <span class="metric-value">${latest.successful_trades}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">胜率</span>
                    <span class="metric-value">${(latest.win_rate * 100).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">总盈利</span>
                    <span class="metric-value ${latest.total_profit >= 0 ? 'positive' : 'negative'}">
                        $${latest.total_profit.toLocaleString()}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">投资组合价值</span>
                    <span class="metric-value">$${latest.portfolio_value.toLocaleString()}</span>
                </div>
            `;
        }
        
        // 更新市场数据
        function updateMarketData(data) {
            const container = document.getElementById('market-data');
            
            if (!data || Object.keys(data).length === 0) {
                container.innerHTML = '<div class="loading">暂无市场数据</div>';
                return;
            }
            
            let html = '';
            for (const [symbol, info] of Object.entries(data)) {
                const changeClass = info.change >= 0 ? 'positive' : 'negative';
                const changeSign = info.change >= 0 ? '+' : '';
                
                html += `
                    <div class="market-item">
                        <div class="market-symbol">${symbol}</div>
                        <div class="market-price">$${info.price.toFixed(2)}</div>
                        <div class="market-change ${changeClass}">
                            ${changeSign}${info.change.toFixed(2)} (${changeSign}${info.change_percent.toFixed(2)}%)
                        </div>
                    </div>
                `;
            }
            
            container.innerHTML = html;
        }
        
        // 更新警报
        function updateAlerts(data, isNew = false) {
            const container = document.getElementById('alerts');
            
            if (!data || data.length === 0) {
                if (!isNew) {
                    container.innerHTML = '<div class="loading">暂无警报</div>';
                }
                return;
            }
            
            let html = '';
            data.forEach(alert => {
                const alertClass = `alert-${alert.level.toLowerCase()}`;
                const time = new Date(alert.timestamp).toLocaleTimeString();
                
                html += `
                    <div class="alert ${alertClass}">
                        <strong>[${alert.level}] ${alert.category}</strong><br>
                        ${alert.message}<br>
                        <small>${time}</small>
                    </div>
                `;
            });
            
            if (isNew) {
                container.innerHTML = html + container.innerHTML;
            } else {
                container.innerHTML = html;
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