#!/usr/bin/env python3
"""
高频交易实时监控和日志记录系统
提供全面的交易活动监控、性能分析和日志管理功能
"""

import logging
import json
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
import queue
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradeEvent:
    """交易事件"""
    timestamp: datetime
    event_type: str  # ORDER, FILL, CANCEL, REJECT
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    order_id: str
    strategy: str
    latency_ms: float = 0.0
    commission: float = 0.0
    pnl: float = 0.0

@dataclass
class MarketEvent:
    """市场事件"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    spread: float
    volatility: float = 0.0

@dataclass
class StrategyEvent:
    """策略事件"""
    timestamp: datetime
    strategy: str
    signal_type: str  # BUY, SELL, HOLD
    signal_strength: float
    symbol: str
    confidence: float
    indicators: Dict[str, float]

@dataclass
class RiskEvent:
    """风险事件"""
    timestamp: datetime
    risk_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    symbol: str
    message: str
    current_value: float
    threshold: float
    action_taken: str

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_latency_ms: float
    orders_per_second: float

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建交易事件表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        order_id TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        latency_ms REAL DEFAULT 0.0,
                        commission REAL DEFAULT 0.0,
                        pnl REAL DEFAULT 0.0
                    )
                ''')
                
                # 创建市场事件表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        bid REAL NOT NULL,
                        ask REAL NOT NULL,
                        last REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        spread REAL NOT NULL,
                        volatility REAL DEFAULT 0.0
                    )
                ''')
                
                # 创建策略事件表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategy_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        signal_strength REAL NOT NULL,
                        symbol TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        indicators TEXT
                    )
                ''')
                
                # 创建风险事件表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        risk_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        message TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold REAL NOT NULL,
                        action_taken TEXT
                    )
                ''')
                
                # 创建性能指标表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_pnl REAL NOT NULL,
                        realized_pnl REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        avg_win REAL NOT NULL,
                        avg_loss REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        sortino_ratio REAL NOT NULL,
                        profit_factor REAL NOT NULL,
                        avg_latency_ms REAL NOT NULL,
                        orders_per_second REAL NOT NULL
                    )
                ''')
                
                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trade_events(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_events(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_timestamp ON strategy_events(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_timestamp ON risk_events(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"数据库初始化错误: {e}")
    
    def insert_trade_event(self, event: TradeEvent):
        """插入交易事件"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trade_events 
                    (timestamp, event_type, symbol, action, quantity, price, order_id, strategy, latency_ms, commission, pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.symbol,
                    event.action,
                    event.quantity,
                    event.price,
                    event.order_id,
                    event.strategy,
                    event.latency_ms,
                    event.commission,
                    event.pnl
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"插入交易事件错误: {e}")
    
    def insert_market_event(self, event: MarketEvent):
        """插入市场事件"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO market_events 
                    (timestamp, symbol, bid, ask, last, volume, spread, volatility)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.timestamp.isoformat(),
                    event.symbol,
                    event.bid,
                    event.ask,
                    event.last,
                    event.volume,
                    event.spread,
                    event.volatility
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"插入市场事件错误: {e}")
    
    def insert_strategy_event(self, event: StrategyEvent):
        """插入策略事件"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO strategy_events 
                    (timestamp, strategy, signal_type, signal_strength, symbol, confidence, indicators)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.timestamp.isoformat(),
                    event.strategy,
                    event.signal_type,
                    event.signal_strength,
                    event.symbol,
                    event.confidence,
                    json.dumps(event.indicators)
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"插入策略事件错误: {e}")
    
    def insert_risk_event(self, event: RiskEvent):
        """插入风险事件"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO risk_events 
                    (timestamp, risk_type, severity, symbol, message, current_value, threshold, action_taken)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.timestamp.isoformat(),
                    event.risk_type,
                    event.severity,
                    event.symbol,
                    event.message,
                    event.current_value,
                    event.threshold,
                    event.action_taken
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"插入风险事件错误: {e}")
    
    def insert_performance_metrics(self, metrics: PerformanceMetrics):
        """插入性能指标"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, total_pnl, realized_pnl, unrealized_pnl, total_trades, win_rate, 
                     avg_win, avg_loss, max_drawdown, sharpe_ratio, sortino_ratio, profit_factor, 
                     avg_latency_ms, orders_per_second)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp.isoformat(),
                    metrics.total_pnl,
                    metrics.realized_pnl,
                    metrics.unrealized_pnl,
                    metrics.total_trades,
                    metrics.win_rate,
                    metrics.avg_win,
                    metrics.avg_loss,
                    metrics.max_drawdown,
                    metrics.sharpe_ratio,
                    metrics.sortino_ratio,
                    metrics.profit_factor,
                    metrics.avg_latency_ms,
                    metrics.orders_per_second
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"插入性能指标错误: {e}")

class AlertManager:
    """警报管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_config = config.get('email', {})
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.logger = logging.getLogger(__name__)
    
    def send_email_alert(self, subject: str, message: str, severity: str = "MEDIUM"):
        """发送邮件警报"""
        try:
            if not self.email_config.get('enabled', False):
                return
            
            # 根据严重程度决定是否发送
            min_severity = self.email_config.get('min_severity', 'MEDIUM')
            severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
            
            if severity_levels.get(severity, 2) < severity_levels.get(min_severity, 2):
                return
            
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"[HFT Alert - {severity}] {subject}"
            
            body = f"""
            高频交易系统警报
            
            时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            严重程度: {severity}
            主题: {subject}
            
            详细信息:
            {message}
            
            请及时处理相关问题。
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"邮件警报已发送: {subject}")
            
        except Exception as e:
            self.logger.error(f"邮件警报发送失败: {e}")
    
    def check_performance_alerts(self, metrics: PerformanceMetrics):
        """检查性能警报"""
        alerts = []
        
        # 检查最大回撤
        max_drawdown_threshold = self.alert_thresholds.get('max_drawdown', 0.05)
        if metrics.max_drawdown > max_drawdown_threshold:
            alerts.append({
                'type': 'MAX_DRAWDOWN',
                'severity': 'HIGH',
                'message': f"最大回撤超过阈值: {metrics.max_drawdown:.2%} > {max_drawdown_threshold:.2%}"
            })
        
        # 检查胜率
        min_win_rate = self.alert_thresholds.get('min_win_rate', 0.4)
        if metrics.win_rate < min_win_rate:
            alerts.append({
                'type': 'LOW_WIN_RATE',
                'severity': 'MEDIUM',
                'message': f"胜率过低: {metrics.win_rate:.2%} < {min_win_rate:.2%}"
            })
        
        # 检查延迟
        max_latency = self.alert_thresholds.get('max_latency_ms', 100)
        if metrics.avg_latency_ms > max_latency:
            alerts.append({
                'type': 'HIGH_LATENCY',
                'severity': 'HIGH',
                'message': f"平均延迟过高: {metrics.avg_latency_ms:.2f}ms > {max_latency}ms"
            })
        
        # 发送警报
        for alert in alerts:
            self.send_email_alert(
                subject=f"性能警报: {alert['type']}",
                message=alert['message'],
                severity=alert['severity']
            )

class HFTMonitoringLogger:
    """高频交易监控日志系统"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化监控日志系统
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.log_dir.mkdir(exist_ok=True)
        
        # 数据库管理器
        db_path = self.log_dir / 'hft_data.db'
        self.db_manager = DatabaseManager(str(db_path))
        
        # 警报管理器
        self.alert_manager = AlertManager(config)
        
        # 事件队列
        self.event_queue = queue.Queue(maxsize=10000)
        
        # 数据缓存
        self.trade_events: deque = deque(maxlen=10000)
        self.market_events: deque = deque(maxlen=10000)
        self.strategy_events: deque = deque(maxlen=10000)
        self.risk_events: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=1000)
        
        # 实时统计
        self.real_time_stats = {
            'orders_per_second': 0,
            'fills_per_second': 0,
            'avg_latency_ms': 0,
            'total_pnl': 0,
            'active_orders': 0,
            'positions_count': 0
        }
        
        # 线程管理
        self.running = False
        self.worker_thread = None
        self.stats_thread = None
        
        # 日志配置
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 性能计算器
        self.performance_calculator = PerformanceCalculator()
        
        # 文件轮转配置
        self.max_log_size = config.get('max_log_size_mb', 100) * 1024 * 1024
        self.max_log_files = config.get('max_log_files', 10)
    
    def setup_logging(self):
        """设置日志配置"""
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 主日志文件
        main_log_file = self.log_dir / f"hft_main_{datetime.now().strftime('%Y%m%d')}.log"
        main_handler = logging.FileHandler(main_log_file, encoding='utf-8')
        main_handler.setFormatter(formatter)
        main_handler.setLevel(logging.INFO)
        
        # 交易日志文件
        trade_log_file = self.log_dir / f"hft_trades_{datetime.now().strftime('%Y%m%d')}.log"
        trade_handler = logging.FileHandler(trade_log_file, encoding='utf-8')
        trade_handler.setFormatter(formatter)
        trade_handler.setLevel(logging.INFO)
        
        # 风险日志文件
        risk_log_file = self.log_dir / f"hft_risk_{datetime.now().strftime('%Y%m%d')}.log"
        risk_handler = logging.FileHandler(risk_log_file, encoding='utf-8')
        risk_handler.setFormatter(formatter)
        risk_handler.setLevel(logging.WARNING)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(main_handler)
        
        # 配置专用日志器
        trade_logger = logging.getLogger('hft.trades')
        trade_logger.addHandler(trade_handler)
        
        risk_logger = logging.getLogger('hft.risk')
        risk_logger.addHandler(risk_handler)
    
    def start(self):
        """启动监控系统"""
        if self.running:
            return
        
        self.running = True
        
        # 启动事件处理线程
        self.worker_thread = threading.Thread(target=self._event_worker, daemon=True)
        self.worker_thread.start()
        
        # 启动统计计算线程
        self.stats_thread = threading.Thread(target=self._stats_worker, daemon=True)
        self.stats_thread.start()
        
        self.logger.info("HFT监控日志系统已启动")
    
    def stop(self):
        """停止监控系统"""
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        
        if self.stats_thread:
            self.stats_thread.join(timeout=5)
        
        self.logger.info("HFT监控日志系统已停止")
    
    def _event_worker(self):
        """事件处理工作线程"""
        while self.running:
            try:
                # 处理事件队列
                try:
                    event_data = self.event_queue.get(timeout=1)
                    self._process_event(event_data)
                    self.event_queue.task_done()
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"事件处理错误: {e}")
    
    def _stats_worker(self):
        """统计计算工作线程"""
        while self.running:
            try:
                # 每秒更新统计数据
                self._update_real_time_stats()
                
                # 每分钟计算性能指标
                if datetime.now().second == 0:
                    self._calculate_performance_metrics()
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"统计计算错误: {e}")
    
    def _process_event(self, event_data: Dict[str, Any]):
        """处理事件数据"""
        event_type = event_data.get('type')
        
        if event_type == 'trade':
            event = TradeEvent(**event_data['data'])
            self.trade_events.append(event)
            self.db_manager.insert_trade_event(event)
            self._log_trade_event(event)
            
        elif event_type == 'market':
            event = MarketEvent(**event_data['data'])
            self.market_events.append(event)
            self.db_manager.insert_market_event(event)
            
        elif event_type == 'strategy':
            event = StrategyEvent(**event_data['data'])
            self.strategy_events.append(event)
            self.db_manager.insert_strategy_event(event)
            self._log_strategy_event(event)
            
        elif event_type == 'risk':
            event = RiskEvent(**event_data['data'])
            self.risk_events.append(event)
            self.db_manager.insert_risk_event(event)
            self._log_risk_event(event)
    
    def _log_trade_event(self, event: TradeEvent):
        """记录交易事件日志"""
        trade_logger = logging.getLogger('hft.trades')
        
        if event.event_type == 'FILL':
            pnl_str = f", PnL: {event.pnl:.2f}" if event.pnl != 0 else ""
            trade_logger.info(
                f"成交 - {event.symbol} {event.action} {event.quantity}@{event.price:.2f}, "
                f"订单: {event.order_id}, 策略: {event.strategy}, "
                f"延迟: {event.latency_ms:.2f}ms{pnl_str}"
            )
        elif event.event_type == 'ORDER':
            trade_logger.info(
                f"下单 - {event.symbol} {event.action} {event.quantity}@{event.price:.2f}, "
                f"订单: {event.order_id}, 策略: {event.strategy}"
            )
        elif event.event_type == 'CANCEL':
            trade_logger.info(
                f"撤单 - {event.symbol} 订单: {event.order_id}, 策略: {event.strategy}"
            )
        elif event.event_type == 'REJECT':
            trade_logger.warning(
                f"拒单 - {event.symbol} {event.action} {event.quantity}@{event.price:.2f}, "
                f"订单: {event.order_id}, 策略: {event.strategy}"
            )
    
    def _log_strategy_event(self, event: StrategyEvent):
        """记录策略事件日志"""
        if event.signal_type != 'HOLD':  # 只记录有效信号
            self.logger.info(
                f"策略信号 - {event.strategy}: {event.symbol} {event.signal_type}, "
                f"强度: {event.signal_strength:.2f}, 置信度: {event.confidence:.2f}"
            )
    
    def _log_risk_event(self, event: RiskEvent):
        """记录风险事件日志"""
        risk_logger = logging.getLogger('hft.risk')
        
        if event.severity in ['HIGH', 'CRITICAL']:
            risk_logger.error(
                f"风险警报 [{event.severity}] - {event.risk_type}: {event.message}, "
                f"当前值: {event.current_value:.2f}, 阈值: {event.threshold:.2f}, "
                f"处理措施: {event.action_taken}"
            )
        else:
            risk_logger.warning(
                f"风险提醒 [{event.severity}] - {event.risk_type}: {event.message}"
            )
    
    def _update_real_time_stats(self):
        """更新实时统计数据"""
        try:
            current_time = datetime.now()
            one_second_ago = current_time - timedelta(seconds=1)
            
            # 计算每秒订单数
            recent_orders = [e for e in self.trade_events 
                           if e.timestamp > one_second_ago and e.event_type == 'ORDER']
            self.real_time_stats['orders_per_second'] = len(recent_orders)
            
            # 计算每秒成交数
            recent_fills = [e for e in self.trade_events 
                          if e.timestamp > one_second_ago and e.event_type == 'FILL']
            self.real_time_stats['fills_per_second'] = len(recent_fills)
            
            # 计算平均延迟
            if recent_fills:
                avg_latency = np.mean([e.latency_ms for e in recent_fills])
                self.real_time_stats['avg_latency_ms'] = avg_latency
            
            # 计算总PnL
            total_pnl = sum(e.pnl for e in self.trade_events if e.event_type == 'FILL')
            self.real_time_stats['total_pnl'] = total_pnl
            
        except Exception as e:
            self.logger.error(f"实时统计更新错误: {e}")
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        try:
            metrics = self.performance_calculator.calculate_metrics(
                self.trade_events, self.market_events
            )
            
            if metrics:
                self.performance_history.append(metrics)
                self.db_manager.insert_performance_metrics(metrics)
                
                # 检查性能警报
                self.alert_manager.check_performance_alerts(metrics)
                
                # 记录性能日志
                self.logger.info(
                    f"性能指标 - 总PnL: {metrics.total_pnl:.2f}, "
                    f"胜率: {metrics.win_rate:.2%}, "
                    f"夏普比率: {metrics.sharpe_ratio:.2f}, "
                    f"最大回撤: {metrics.max_drawdown:.2%}, "
                    f"平均延迟: {metrics.avg_latency_ms:.2f}ms"
                )
                
        except Exception as e:
            self.logger.error(f"性能指标计算错误: {e}")
    
    def log_trade_event(self, event_type: str, symbol: str, action: str, quantity: int, 
                       price: float, order_id: str, strategy: str, latency_ms: float = 0.0,
                       commission: float = 0.0, pnl: float = 0.0):
        """记录交易事件"""
        event_data = {
            'type': 'trade',
            'data': {
                'timestamp': datetime.now(),
                'event_type': event_type,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'order_id': order_id,
                'strategy': strategy,
                'latency_ms': latency_ms,
                'commission': commission,
                'pnl': pnl
            }
        }
        
        try:
            self.event_queue.put_nowait(event_data)
        except queue.Full:
            self.logger.warning("事件队列已满，丢弃事件")
    
    def log_market_event(self, symbol: str, bid: float, ask: float, last: float, 
                        volume: int, volatility: float = 0.0):
        """记录市场事件"""
        event_data = {
            'type': 'market',
            'data': {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': last,
                'volume': volume,
                'spread': ask - bid,
                'volatility': volatility
            }
        }
        
        try:
            self.event_queue.put_nowait(event_data)
        except queue.Full:
            pass  # 市场数据可以丢弃
    
    def log_strategy_event(self, strategy: str, signal_type: str, signal_strength: float,
                          symbol: str, confidence: float, indicators: Dict[str, float]):
        """记录策略事件"""
        event_data = {
            'type': 'strategy',
            'data': {
                'timestamp': datetime.now(),
                'strategy': strategy,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'symbol': symbol,
                'confidence': confidence,
                'indicators': indicators
            }
        }
        
        try:
            self.event_queue.put_nowait(event_data)
        except queue.Full:
            self.logger.warning("事件队列已满，丢弃策略事件")
    
    def log_risk_event(self, risk_type: str, severity: str, symbol: str, message: str,
                      current_value: float, threshold: float, action_taken: str = ""):
        """记录风险事件"""
        event_data = {
            'type': 'risk',
            'data': {
                'timestamp': datetime.now(),
                'risk_type': risk_type,
                'severity': severity,
                'symbol': symbol,
                'message': message,
                'current_value': current_value,
                'threshold': threshold,
                'action_taken': action_taken
            }
        }
        
        try:
            self.event_queue.put_nowait(event_data)
        except queue.Full:
            self.logger.warning("事件队列已满，丢弃风险事件")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """获取实时统计数据"""
        return self.real_time_stats.copy()
    
    def get_recent_events(self, event_type: str, limit: int = 100) -> List[Dict]:
        """获取最近事件"""
        if event_type == 'trade':
            events = list(self.trade_events)[-limit:]
            return [asdict(e) for e in events]
        elif event_type == 'market':
            events = list(self.market_events)[-limit:]
            return [asdict(e) for e in events]
        elif event_type == 'strategy':
            events = list(self.strategy_events)[-limit:]
            return [asdict(e) for e in events]
        elif event_type == 'risk':
            events = list(self.risk_events)[-limit:]
            return [asdict(e) for e in events]
        else:
            return []
    
    def export_daily_report(self, date: datetime = None) -> str:
        """导出日报"""
        if date is None:
            date = datetime.now()
        
        report_file = self.log_dir / f"daily_report_{date.strftime('%Y%m%d')}.json"
        
        try:
            # 获取当日数据
            start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=1)
            
            daily_trades = [e for e in self.trade_events 
                          if start_time <= e.timestamp < end_time]
            daily_performance = [p for p in self.performance_history 
                               if start_time <= p.timestamp < end_time]
            daily_risks = [r for r in self.risk_events 
                         if start_time <= r.timestamp < end_time]
            
            # 生成报告
            report = {
                'date': date.strftime('%Y-%m-%d'),
                'summary': {
                    'total_trades': len([e for e in daily_trades if e.event_type == 'FILL']),
                    'total_orders': len([e for e in daily_trades if e.event_type == 'ORDER']),
                    'total_pnl': sum(e.pnl for e in daily_trades if e.event_type == 'FILL'),
                    'risk_alerts': len(daily_risks),
                    'avg_latency': np.mean([e.latency_ms for e in daily_trades 
                                          if e.event_type == 'FILL' and e.latency_ms > 0]) if daily_trades else 0
                },
                'performance': [asdict(p) for p in daily_performance],
                'trades': [asdict(t) for t in daily_trades],
                'risks': [asdict(r) for r in daily_risks]
            }
            
            # 保存报告
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"日报已导出: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"日报导出错误: {e}")
            return ""

class PerformanceCalculator:
    """性能计算器"""
    
    def calculate_metrics(self, trade_events: deque, market_events: deque) -> Optional[PerformanceMetrics]:
        """计算性能指标"""
        try:
            current_time = datetime.now()
            
            # 获取成交记录
            fills = [e for e in trade_events if e.event_type == 'FILL']
            
            if not fills:
                return None
            
            # 基本统计
            total_trades = len(fills)
            total_pnl = sum(e.pnl for e in fills)
            realized_pnl = sum(e.pnl for e in fills if e.pnl != 0)
            unrealized_pnl = 0  # 需要从持仓数据计算
            
            # 胜负统计
            winning_trades = [e for e in fills if e.pnl > 0]
            losing_trades = [e for e in fills if e.pnl < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = np.mean([e.pnl for e in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([e.pnl for e in losing_trades]) if losing_trades else 0
            
            # 最大回撤
            pnl_series = np.cumsum([e.pnl for e in fills])
            peak = np.maximum.accumulate(pnl_series)
            drawdown = (peak - pnl_series) / np.maximum(peak, 1)
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # 风险调整收益
            returns = np.diff(pnl_series) if len(pnl_series) > 1 else np.array([0])
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            negative_returns = returns[returns < 0]
            sortino_ratio = (np.mean(returns) / np.std(negative_returns) 
                           if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0)
            
            # 盈利因子
            gross_profit = sum(e.pnl for e in winning_trades)
            gross_loss = abs(sum(e.pnl for e in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # 延迟统计
            latencies = [e.latency_ms for e in fills if e.latency_ms > 0]
            avg_latency_ms = np.mean(latencies) if latencies else 0
            
            # 交易频率
            if len(fills) > 1:
                time_span = (fills[-1].timestamp - fills[0].timestamp).total_seconds()
                orders_per_second = len(fills) / time_span if time_span > 0 else 0
            else:
                orders_per_second = 0
            
            return PerformanceMetrics(
                timestamp=current_time,
                total_pnl=total_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_trades=total_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                profit_factor=profit_factor,
                avg_latency_ms=avg_latency_ms,
                orders_per_second=orders_per_second
            )
            
        except Exception as e:
            logging.error(f"性能指标计算错误: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    # 监控配置
    config = {
        'log_dir': './logs',
        'max_log_size_mb': 100,
        'max_log_files': 10,
        'email': {
            'enabled': False,  # 设置为True启用邮件警报
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_password',
            'from_email': 'your_email@gmail.com',
            'to_emails': ['alert@company.com'],
            'min_severity': 'HIGH'
        },
        'alert_thresholds': {
            'max_drawdown': 0.05,
            'min_win_rate': 0.4,
            'max_latency_ms': 100
        }
    }
    
    # 创建监控系统
    monitor = HFTMonitoringLogger(config)
    monitor.start()
    
    try:
        # 模拟交易事件
        for i in range(100):
            # 模拟下单
            monitor.log_trade_event(
                event_type='ORDER',
                symbol='AAPL',
                action='BUY',
                quantity=100,
                price=150.0 + np.random.normal(0, 1),
                order_id=f'ORDER_{i}',
                strategy='CitadelHFT',
                latency_ms=np.random.uniform(10, 50)
            )
            
            # 模拟成交
            if np.random.random() > 0.1:  # 90%成交率
                monitor.log_trade_event(
                    event_type='FILL',
                    symbol='AAPL',
                    action='BUY',
                    quantity=100,
                    price=150.0 + np.random.normal(0, 1),
                    order_id=f'ORDER_{i}',
                    strategy='CitadelHFT',
                    latency_ms=np.random.uniform(10, 50),
                    pnl=np.random.normal(10, 50)
                )
            
            # 模拟市场数据
            monitor.log_market_event(
                symbol='AAPL',
                bid=149.9 + np.random.normal(0, 1),
                ask=150.1 + np.random.normal(0, 1),
                last=150.0 + np.random.normal(0, 1),
                volume=np.random.randint(1000, 5000),
                volatility=np.random.uniform(0.1, 0.3)
            )
            
            # 模拟策略信号
            if np.random.random() > 0.7:  # 30%概率产生信号
                monitor.log_strategy_event(
                    strategy='CitadelHFT',
                    signal_type=np.random.choice(['BUY', 'SELL']),
                    signal_strength=np.random.uniform(0.5, 1.0),
                    symbol='AAPL',
                    confidence=np.random.uniform(0.6, 0.9),
                    indicators={
                        'rsi': np.random.uniform(30, 70),
                        'macd': np.random.uniform(-1, 1),
                        'volume_ratio': np.random.uniform(0.8, 1.5)
                    }
                )
            
            time.sleep(0.1)
        
        # 获取实时统计
        stats = monitor.get_real_time_stats()
        print(f"实时统计: {stats}")
        
        # 导出日报
        report_file = monitor.export_daily_report()
        print(f"日报文件: {report_file}")
        
        # 等待一段时间让系统处理事件
        time.sleep(5)
        
    finally:
        monitor.stop()