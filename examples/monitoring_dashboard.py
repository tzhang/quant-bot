#!/usr/bin/env python3
"""
量化交易监控面板
纯Python实现，使用外部HTML模板文件
"""

import os
import sys
import time
import json
import logging
import threading
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from functools import wraps
import traceback

# Flask和SocketIO相关导入
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import psutil
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, backoff_factor=1.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败: {str(e)}")
                        raise
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败，{wait_time}秒后重试: {str(e)}")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self):
        self.error_count = 0
        self.warning_count = 0
        self.last_errors = []
        self.max_error_history = 100
        
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        self.error_count += 1
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        self.last_errors.append(error_info)
        
        if len(self.last_errors) > self.max_error_history:
            self.last_errors.pop(0)
            
        logger.error(f"错误记录 [{context}]: {str(error)}")
        
    def log_warning(self, message: str, context: str = ""):
        """记录警告"""
        self.warning_count += 1
        logger.warning(f"警告 [{context}]: {message}")
        
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'recent_errors': self.last_errors[-10:] if self.last_errors else []
        }

@dataclass
class SystemMetrics:
    """系统指标数据类"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    error_count: int
    warning_count: int
    timestamp: str

@dataclass
class TradingMetrics:
    """交易指标数据类"""
    total_trades: int
    successful_trades: int
    win_rate: float
    total_profit: float
    active_positions: int
    timestamp: str

@dataclass
class MarketData:
    """市场数据类"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str

@dataclass
class TradingInfo:
    """交易信息类"""
    title: str
    message: str
    level: str  # info, warning, error, success
    timestamp: str

@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    symbol: str
    action: str  # buy/sell (改名为action以匹配HTML模板)
    quantity: int
    price: float
    order_type: str  # 订单类型
    status: str
    timestamp: str
    pnl: float  # 盈亏

@dataclass
class RiskEvent:
    """风险事件类"""
    risk_type: str
    severity: str  # low, medium, high, critical
    description: str
    timestamp: str

@dataclass
class MarketEvent:
    """市场事件类"""
    event_type: str
    description: str
    impact: str  # low, medium, high
    timestamp: str

class DataCollector:
    """数据收集器"""
    
    def __init__(self):
        self.is_running = False
        self.data_cache = {}
        self.error_manager = ErrorRecoveryManager()
        self.performance_stats = {
            'data_points_collected': 0,
            'collection_errors': 0,
            'last_collection_time': None
        }
        
    def start_collection(self):
        """启动数据收集"""
        self.is_running = True
        logger.info("数据收集已启动")
        
    def stop_collection(self):
        """停止数据收集"""
        self.is_running = False
        logger.info("数据收集已停止")
        
    def reset_data(self):
        """重置数据"""
        self.data_cache.clear()
        self.error_manager = ErrorRecoveryManager()
        self.performance_stats = {
            'data_points_collected': 0,
            'collection_errors': 0,
            'last_collection_time': None
        }
        logger.info("数据已重置")
        
    @retry_with_backoff(max_retries=3)
    def collect_data(self) -> Dict[str, Any]:
        """收集所有数据"""
        try:
            current_time = datetime.now().isoformat()
            
            data = {
                'system_metrics': self.get_system_metrics(),
                'trading_metrics': self.get_trading_metrics(),
                'market_data': self.get_market_data(),
                'trading_info': self.get_trading_info(),
                'order_info': self.get_order_info(),
                'risk_events': self.get_risk_events(),
                'market_events': self.get_market_events(),
                'timestamp': current_time
            }
            
            self.data_cache = data
            self.performance_stats['data_points_collected'] += 1
            self.performance_stats['last_collection_time'] = current_time
            
            return data
            
        except Exception as e:
            self.error_manager.log_error(e, "数据收集")
            self.performance_stats['collection_errors'] += 1
            raise
            
    def get_system_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 模拟活跃连接数
            active_connections = random.randint(5, 50)
            
            error_stats = self.error_manager.get_error_stats()
            
            return SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_connections=active_connections,
                error_count=error_stats['error_count'],
                warning_count=error_stats['warning_count'],
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.error_manager.log_error(e, "系统指标收集")
            # 返回默认值
            return SystemMetrics(0, 0, 0, 0, 0, 0, datetime.now().isoformat())
            
    def get_trading_metrics(self) -> TradingMetrics:
        """获取交易指标"""
        try:
            # 模拟交易数据
            total_trades = random.randint(100, 1000)
            successful_trades = random.randint(int(total_trades * 0.6), int(total_trades * 0.9))
            win_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
            total_profit = random.uniform(-10000, 50000)
            active_positions = random.randint(0, 20)
            
            return TradingMetrics(
                total_trades=total_trades,
                successful_trades=successful_trades,
                win_rate=win_rate,
                total_profit=total_profit,
                active_positions=active_positions,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.error_manager.log_error(e, "交易指标收集")
            return TradingMetrics(0, 0, 0, 0, 0, datetime.now().isoformat())
            
    def get_market_data(self) -> List[MarketData]:
        """获取市场数据"""
        try:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
            market_data = []
            
            for symbol in symbols:
                price = random.uniform(100, 500)
                change = random.uniform(-10, 10)
                change_percent = (change / price) * 100
                volume = random.randint(1000000, 10000000)
                
                market_data.append(MarketData(
                    symbol=symbol,
                    price=price,
                    change=change,
                    change_percent=change_percent,
                    volume=volume,
                    timestamp=datetime.now().isoformat()
                ))
                
            return market_data
            
        except Exception as e:
            self.error_manager.log_error(e, "市场数据收集")
            return []
            
    def get_trading_info(self) -> List[TradingInfo]:
        """获取交易信息"""
        try:
            info_list = []
            
            # 生成一些示例交易信息
            messages = [
                ("订单执行", "AAPL买入订单已成功执行", "success"),
                ("策略信号", "检测到TSLA强势突破信号", "info"),
                ("风险提醒", "当前持仓集中度较高", "warning"),
                ("市场分析", "科技股板块表现强劲", "info")
            ]
            
            for title, message, level in messages:
                info_list.append(TradingInfo(
                    title=title,
                    message=message,
                    level=level,
                    timestamp=datetime.now().isoformat()
                ))
                
            return info_list
            
        except Exception as e:
            self.error_manager.log_error(e, "交易信息收集")
            return []
            
    def get_order_info(self) -> List[OrderInfo]:
        """获取订单信息"""
        try:
            orders = []
            symbols = ['AAPL', 'GOOGL', 'MSFT']
            
            for i, symbol in enumerate(symbols):
                orders.append(OrderInfo(
                    order_id=f"ORD{1000 + i}",
                    symbol=symbol,
                    action=random.choice(['BUY', 'SELL']),
                    quantity=random.randint(10, 1000),
                    price=random.uniform(100, 500),
                    order_type=random.choice(['MKT', 'LMT', 'STP']),
                    status=random.choice(['Filled', 'Pending', 'Cancelled']),
                    timestamp=datetime.now().isoformat(),
                    pnl=random.uniform(-1000, 1000)
                ))
                
            return orders
            
        except Exception as e:
            self.error_manager.log_error(e, "订单信息收集")
            return []
            
    def get_risk_events(self) -> List[RiskEvent]:
        """获取风险事件"""
        try:
            events = []
            
            # 随机生成风险事件
            if random.random() < 0.3:  # 30%概率生成风险事件
                risk_types = ["持仓集中", "波动率异常", "流动性不足", "市场异动"]
                severities = ["low", "medium", "high"]
                
                events.append(RiskEvent(
                    risk_type=random.choice(risk_types),
                    severity=random.choice(severities),
                    description=f"检测到{random.choice(risk_types)}风险，建议关注",
                    timestamp=datetime.now().isoformat()
                ))
                
            return events
            
        except Exception as e:
            self.error_manager.log_error(e, "风险事件收集")
            return []
            
    def get_market_events(self) -> List[MarketEvent]:
        """获取市场事件"""
        try:
            events = []
            
            # 随机生成市场事件
            if random.random() < 0.2:  # 20%概率生成市场事件
                event_types = ["财报发布", "政策变化", "技术突破", "行业动态"]
                impacts = ["low", "medium", "high"]
                
                events.append(MarketEvent(
                    event_type=random.choice(event_types),
                    description=f"{random.choice(event_types)}可能影响市场走势",
                    impact=random.choice(impacts),
                    timestamp=datetime.now().isoformat()
                ))
                
            return events
            
        except Exception as e:
            self.error_manager.log_error(e, "市场事件收集")
            return []

class MonitoringDashboard:
    """监控面板主类"""
    
    def __init__(self, host='0.0.0.0', port=8080, debug=False):
        self.host = host
        self.port = port
        self.debug = debug
        
        # 初始化Flask应用
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'your-secret-key-here'
        
        # 初始化SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 初始化数据收集器
        self.data_collector = DataCollector()
        
        # 设置路由
        self._setup_routes()
        self._setup_socketio_events()
        
        # 数据收集线程
        self.collection_thread = None
        self.is_running = False
        
    def _setup_routes(self):
        """设置Flask路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('monitoring_dashboard.html')
            
        @self.app.route('/health')
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'is_collecting': self.data_collector.is_running,
                'performance_stats': self.data_collector.performance_stats
            })
            
        @self.app.route('/api/system-metrics')
        def get_system_metrics():
            """获取系统指标API"""
            try:
                metrics = self.data_collector.get_system_metrics()
                return jsonify(asdict(metrics))
            except Exception as e:
                logger.error(f"获取系统指标失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/trading-metrics')
        def get_trading_metrics():
            """获取交易指标API"""
            try:
                metrics = self.data_collector.get_trading_metrics()
                return jsonify(asdict(metrics))
            except Exception as e:
                logger.error(f"获取交易指标失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/market-data')
        def get_market_data():
            """获取市场数据API"""
            try:
                data = self.data_collector.get_market_data()
                return jsonify([asdict(item) for item in data])
            except Exception as e:
                logger.error(f"获取市场数据失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/orders')
        def get_orders():
            """获取订单信息API"""
            try:
                orders = self.data_collector.get_order_info()
                return jsonify([asdict(item) for item in orders])
            except Exception as e:
                logger.error(f"获取订单信息失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/trading-info')
        def get_trading_info():
            """获取交易信息API"""
            try:
                info = self.data_collector.get_trading_info()
                return jsonify([asdict(item) for item in info])
            except Exception as e:
                logger.error(f"获取交易信息失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/risk-events')
        def get_risk_events():
            """获取风险事件API"""
            try:
                events = self.data_collector.get_risk_events()
                return jsonify([asdict(item) for item in events])
            except Exception as e:
                logger.error(f"获取风险事件失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/market-events')
        def get_market_events():
            """获取市场事件API"""
            try:
                events = self.data_collector.get_market_events()
                return jsonify([asdict(item) for item in events])
            except Exception as e:
                logger.error(f"获取市场事件失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/control/start', methods=['POST'])
        def start_monitoring():
            """启动监控API"""
            try:
                self.data_collector.start_collection()
                self._start_data_collection_thread()
                return jsonify({'status': 'success', 'message': '监控已启动'})
            except Exception as e:
                logger.error(f"启动监控失败: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/api/control/stop', methods=['POST'])
        def stop_monitoring():
            """停止监控API"""
            try:
                self.data_collector.stop_collection()
                self._stop_data_collection_thread()
                return jsonify({'status': 'success', 'message': '监控已停止'})
            except Exception as e:
                logger.error(f"停止监控失败: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/api/control/reset', methods=['POST'])
        def reset_data():
            """重置数据API"""
            try:
                self.data_collector.reset_data()
                return jsonify({'status': 'success', 'message': '数据已重置'})
            except Exception as e:
                logger.error(f"重置数据失败: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
    def _setup_socketio_events(self):
        """设置SocketIO事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """客户端连接事件"""
            logger.info(f"客户端已连接: {request.sid}")
            emit('status', {'message': '已连接到监控面板'})
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """客户端断开连接事件"""
            logger.info(f"客户端已断开连接: {request.sid}")
            
        @self.socketio.on('request_data')
        def handle_data_request(data=None):
            """处理数据请求"""
            try:
                if self.data_collector.is_running:
                    data = self.data_collector.collect_data()
                    # 转换数据类为字典
                    serializable_data = {}
                    for key, value in data.items():
                        if isinstance(value, list):
                            serializable_data[key] = [asdict(item) if hasattr(item, '__dict__') else item for item in value]
                        elif hasattr(value, '__dict__'):
                            serializable_data[key] = asdict(value)
                        else:
                            serializable_data[key] = value
                    
                    emit('data_update', serializable_data)
                else:
                    emit('status', {'message': '数据收集未启动'})
            except Exception as e:
                logger.error(f"处理数据请求失败: {str(e)}")
                emit('error', {'message': str(e)})
                
    def _start_data_collection_thread(self):
        """启动数据收集线程"""
        if self.collection_thread is None or not self.collection_thread.is_alive():
            self.is_running = True
            self.collection_thread = threading.Thread(target=self._data_collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info("数据收集线程已启动")
            
    def _stop_data_collection_thread(self):
        """停止数据收集线程"""
        self.is_running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
            logger.info("数据收集线程已停止")
            
    def _data_collection_loop(self):
        """数据收集循环"""
        while self.is_running and self.data_collector.is_running:
            try:
                data = self.data_collector.collect_data()
                
                # 转换数据类为字典以便序列化
                serializable_data = {}
                for key, value in data.items():
                    if isinstance(value, list):
                        serializable_data[key] = [asdict(item) if hasattr(item, '__dict__') else item for item in value]
                    elif hasattr(value, '__dict__'):
                        serializable_data[key] = asdict(value)
                    else:
                        serializable_data[key] = value
                
                # 广播数据更新
                self.socketio.emit('data_update', serializable_data)
                
                time.sleep(5)  # 每5秒收集一次数据
                
            except Exception as e:
                logger.error(f"数据收集循环出错: {str(e)}")
                time.sleep(10)  # 出错时等待更长时间
                
    def start(self):
        """启动监控面板"""
        try:
            logger.info(f"启动监控面板服务器 http://{self.host}:{self.port}")
            self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug)
        except Exception as e:
            logger.error(f"启动监控面板失败: {str(e)}")
            raise
            
    def stop(self):
        """停止监控面板"""
        try:
            self.data_collector.stop_collection()
            self._stop_data_collection_thread()
            logger.info("监控面板已停止")
        except Exception as e:
            logger.error(f"停止监控面板失败: {str(e)}")

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='量化交易监控面板')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    try:
        dashboard = MonitoringDashboard(host=args.host, port=args.port, debug=args.debug)
        dashboard.start()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭监控面板...")
        dashboard.stop()
    except Exception as e:
        logger.error(f"监控面板运行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()