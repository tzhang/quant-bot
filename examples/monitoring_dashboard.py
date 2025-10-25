#!/usr/bin/env python3
"""
量化交易监控面板 - 使用真实数据源
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import psutil

# 导入真实数据收集器
from real_data_collector import RealDataCollector
from src.trading.ib_trading_manager import IBTradingManager

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
    total_pnl: float  # 改名为total_pnl以匹配前端
    portfolio_value: float  # 新增portfolio_value字段
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

# DataCollector类已被RealDataCollector替代，删除模拟数据生成代码

class MonitoringDashboard:
    """监控面板主类 - 使用真实数据源"""
    
    def __init__(self, host='0.0.0.0', port=8080, debug=False, ib_manager=None):
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
        
        # 初始化真实数据收集器
        self.data_collector = RealDataCollector(ib_manager)
        
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
            
        # 添加通用事件监听器用于调试
        @self.socketio.on_error_default
        def default_error_handler(e):
            logger.error(f"SocketIO错误: {e}")
            
        # 添加ping测试事件
        @self.socketio.on('ping')
        def handle_ping(data=None):
            logger.info(f"收到ping事件: {data}")
            emit('pong', {'message': 'pong response', 'timestamp': datetime.now().isoformat()})
            
        @self.socketio.on('request_data')
        def handle_data_request(data=None):
            """处理数据请求"""
            try:
                logger.info(f"=== 收到WebSocket事件 request_data ===")
                logger.info(f"事件数据: {data}")
                logger.info(f"数据类型: {type(data)}")
                
                if not self.data_collector.is_running:
                    logger.warning("数据收集器未运行")
                    emit('status', {'message': '数据收集未启动'})
                    return
                
                # 获取请求的数据类型
                request_type = data.get('type') if data else None
                logger.info(f"解析的请求类型: {request_type}")
                
                if request_type == 'system_metrics':
                    metrics = self.data_collector.get_system_metrics()
                    logger.info(f"发送系统指标数据: {metrics}")
                    emit('system_metrics', [asdict(metrics)])
                    
                elif request_type == 'trading_metrics':
                    metrics = self.data_collector.get_trading_metrics()
                    logger.info(f"发送交易指标数据: {metrics}")
                    emit('trading_metrics', [asdict(metrics)])
                    
                elif request_type == 'market_data':
                    market_data = self.data_collector.get_market_data()
                    logger.info(f"发送市场数据: {len(market_data)} 条记录")
                    emit('market_data', [asdict(item) for item in market_data])
                    
                elif request_type == 'important_info':
                    # 合并交易信息和风险事件作为重要信息
                    trading_info = self.data_collector.get_trading_info()
                    risk_events = self.data_collector.get_risk_events()
                    market_events = self.data_collector.get_market_events()
                    
                    important_info = []
                    # 添加交易信息
                    for info in trading_info:
                        important_info.append({
                            'type': 'trade',
                            'title': info.title,
                            'message': info.message,
                            'level': info.level,
                            'timestamp': info.timestamp
                        })
                    
                    # 添加风险事件
                    for event in risk_events:
                        important_info.append({
                            'type': 'risk',
                            'title': f'{event.risk_type}风险',
                            'message': event.description,
                            'level': event.severity,
                            'timestamp': event.timestamp
                        })
                    
                    # 添加市场事件
                    for event in market_events:
                        important_info.append({
                            'type': 'market',
                            'title': f'{event.event_type}事件',
                            'message': event.description,
                            'level': event.impact,
                            'timestamp': event.timestamp
                        })
                    
                    logger.info(f"发送重要信息: {len(important_info)} 条记录")
                    emit('important_info', important_info)
                    
                elif request_type == 'orders':
                    orders = self.data_collector.get_order_info()
                    logger.info(f"发送订单数据: {len(orders)} 条记录")
                    emit('orders', [asdict(item) for item in orders])
                    
                else:
                    # 如果没有指定类型，返回所有数据
                    collected_data = self.data_collector.collect_data()
                    serializable_data = {}
                    for key, value in collected_data.items():
                        if isinstance(value, list):
                            serializable_data[key] = [asdict(item) if hasattr(item, '__dict__') else item for item in value]
                        elif hasattr(value, '__dict__'):
                            serializable_data[key] = asdict(value)
                        else:
                            serializable_data[key] = value
                    
                    logger.info(f"发送所有数据: {list(serializable_data.keys())}")
                    emit('data_update', serializable_data)
                    
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
                # 收集各类数据
                data = {
                    'system_metrics': self.data_collector.get_system_metrics(),
                    'trading_metrics': self.data_collector.get_trading_metrics(),
                    'market_data': self.data_collector.get_market_data(),
                    'positions': self.data_collector.get_positions(),
                    'trading_info': [],  # 暂时为空
                    'order_info': [],    # 暂时为空
                    'risk_events': [],   # 暂时为空
                    'market_events': []  # 暂时为空
                }
                
                # 广播数据更新
                self.socketio.emit('data_update', data)
                
                time.sleep(5)  # 每5秒收集一次数据
                
            except Exception as e:
                logger.error(f"数据收集循环出错: {str(e)}")
                time.sleep(10)  # 出错时等待更长时间
                
    def start(self):
        """启动监控面板"""
        try:
            # 启动数据收集器
            self.data_collector.start_collection()
            logger.info("数据收集器已启动")
            
            # 启动数据收集线程
            self._start_data_collection_thread()
            
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
    parser.add_argument('--client-id', type=int, default=2, help='IB客户端ID')
    
    args = parser.parse_args()
    
    try:
        # 初始化IB交易管理器
        ib_manager = None
        try:
            ib_manager = IBTradingManager(client_id=args.client_id)
            logger.info(f"IB交易管理器初始化成功，客户端ID: {args.client_id}")
        except Exception as e:
            logger.warning(f"IB交易管理器初始化失败: {e}，将使用模拟数据")
        
        dashboard = MonitoringDashboard(
            host=args.host, 
            port=args.port, 
            debug=args.debug,
            ib_manager=ib_manager
        )
        dashboard.start()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭监控面板...")
        dashboard.stop()
    except Exception as e:
        logger.error(f"监控面板运行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()