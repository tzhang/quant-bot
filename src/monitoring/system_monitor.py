"""
系统监控模块
提供实时系统状态监控、性能指标收集和告警功能
"""

import psutil
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from enum import Enum

# 添加AlertLevel和AlertInfo的定义
class AlertLevel(Enum):
    """告警级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AlertInfo:
    """告警信息"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['level'] = self.level.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        return result

# 尝试导入项目日志模块，如果失败则使用标准日志
try:
    from ..utils.logger import get_logger, logger_manager
except ImportError:
    import logging
    
    def get_logger(name):
        return logging.getLogger(name)
    
    class LoggerManager:
        def log_performance(self, *args, **kwargs):
            pass
    
    logger_manager = LoggerManager()


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    disk_free: int
    network_sent: int
    network_recv: int
    load_average: List[float]
    process_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ProcessMetrics:
    """进程指标数据类"""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    status: str
    create_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class AlertRule:
    """告警规则数据类"""
    name: str
    metric: str
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    duration: int  # 持续时间（秒）
    enabled: bool = True
    
    def check(self, value: float) -> bool:
        """检查是否触发告警"""
        if not self.enabled:
            return False
        
        if self.operator == '>':
            return value > self.threshold
        elif self.operator == '<':
            return value < self.threshold
        elif self.operator == '>=':
            return value >= self.threshold
        elif self.operator == '<=':
            return value <= self.threshold
        elif self.operator == '==':
            return value == self.threshold
        elif self.operator == '!=':
            return value != self.threshold
        else:
            return False


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        """初始化告警管理器"""
        self.logger = get_logger('monitoring.alert')
        self.rules: List[AlertRule] = []
        self.alert_history: Dict[str, List[datetime]] = {}
        self.notification_handlers: List[Callable] = []
        self.alerts: List[AlertInfo] = []
        self.alert_handlers: List[Callable] = []
    
    def add_rule(self, rule: AlertRule):
        """
        添加告警规则
        
        Args:
            rule: 告警规则
        """
        self.rules.append(rule)
        self.logger.info(f"添加告警规则: {rule.name}")
    
    def add_notification_handler(self, handler: Callable):
        """
        添加通知处理器
        
        Args:
            handler: 通知处理函数
        """
        self.notification_handlers.append(handler)
    
    def add_alert_handler(self, handler: Callable):
        """
        添加告警处理器
        
        Args:
            handler: 告警处理函数
        """
        self.alert_handlers.append(handler)
    
    def create_alert(self, level: 'AlertLevel', title: str, message: str, source: str, data: Dict = None) -> 'AlertInfo':
        """
        创建告警
        
        Args:
            level: 告警级别
            title: 告警标题
            message: 告警消息
            source: 告警源
            data: 附加数据
            
        Returns:
            AlertInfo: 创建的告警信息
        """
        alert = AlertInfo(
            id=f"alert_{len(self.alerts)}_{int(datetime.now().timestamp())}",
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            data=data or {}
        )
        
        self.alerts.append(alert)
        
        # 调用告警处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"告警处理器执行失败: {e}")
        
        return alert
    
    def resolve_alert(self, alert_id: str):
        """
        解决告警
        
        Args:
            alert_id: 告警ID
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"告警已解决: {alert_id}")
                break
    
    def get_active_alerts(self) -> List['AlertInfo']:
        """
        获取活跃告警
        
        Returns:
            List[AlertInfo]: 活跃告警列表
        """
        return [alert for alert in self.alerts if not getattr(alert, 'resolved', False)]
    
    def check_alerts(self, metrics: SystemMetrics):
        """
        检查告警条件
        
        Args:
            metrics: 系统指标
        """
        current_time = datetime.now()
        
        for rule in self.rules:
            try:
                # 获取指标值
                value = getattr(metrics, rule.metric, None)
                if value is None:
                    continue
                
                # 检查是否触发告警
                if rule.check(value):
                    # 记录告警历史
                    if rule.name not in self.alert_history:
                        self.alert_history[rule.name] = []
                    
                    self.alert_history[rule.name].append(current_time)
                    
                    # 检查持续时间
                    recent_alerts = [
                        t for t in self.alert_history[rule.name]
                        if (current_time - t).total_seconds() <= rule.duration
                    ]
                    
                    if len(recent_alerts) >= rule.duration / 60:  # 假设每分钟检查一次
                        self._send_alert(rule, value, metrics)
                        
                        # 清理历史记录，避免重复告警
                        self.alert_history[rule.name] = []
                
            except Exception as e:
                self.logger.error(f"检查告警规则 {rule.name} 时出错: {e}")
    
    def _send_alert(self, rule: AlertRule, value: float, metrics: SystemMetrics):
        """
        发送告警通知
        
        Args:
            rule: 告警规则
            value: 触发值
            metrics: 系统指标
        """
        alert_data = {
            'rule_name': rule.name,
            'metric': rule.metric,
            'threshold': rule.threshold,
            'current_value': value,
            'timestamp': metrics.timestamp,
            'system_metrics': metrics.to_dict()
        }
        
        self.logger.warning(f"告警触发: {rule.name}, 当前值: {value}, 阈值: {rule.threshold}")
        
        # 调用所有通知处理器
        for handler in self.notification_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                self.logger.error(f"发送告警通知失败: {e}")


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        """初始化指标收集器"""
        self.logger = get_logger('monitoring.collector')
        self.network_counters = psutil.net_io_counters()
        self.metrics = []  # 添加指标存储列表
    
    def add_metric(self, name: str, value: float, unit: str = "", tags: Dict[str, str] = None):
        """添加自定义指标"""
        metric = {
            'name': name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now(),
            'tags': tags or {}
        }
        self.metrics.append(metric)
    
    def get_metrics(self, since: datetime = None) -> List[Dict[str, Any]]:
        """获取指标数据"""
        if since is None:
            return self.metrics
        
        return [m for m in self.metrics if m['timestamp'] >= since]
    
    def collect_system_metrics(self) -> SystemMetrics:
        """
        收集系统指标
        
        Returns:
            SystemMetrics: 系统指标数据
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存信息
            memory = psutil.virtual_memory()
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            
            # 网络信息
            network = psutil.net_io_counters()
            
            # 负载平均值
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # 进程数量
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available,
                disk_usage_percent=disk.percent,
                disk_free=disk.free,
                network_sent=network.bytes_sent,
                network_recv=network.bytes_recv,
                load_average=list(load_avg),
                process_count=process_count
            )
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
            raise
    
    def collect_process_metrics(self, process_names: List[str] = None) -> List[ProcessMetrics]:
        """
        收集进程指标
        
        Args:
            process_names: 要监控的进程名称列表
            
        Returns:
            List[ProcessMetrics]: 进程指标列表
        """
        process_metrics = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info', 'status', 'create_time']):
                try:
                    proc_info = proc.info
                    
                    # 如果指定了进程名称，只收集匹配的进程
                    if process_names and proc_info['name'] not in process_names:
                        continue
                    
                    metrics = ProcessMetrics(
                        pid=proc_info['pid'],
                        name=proc_info['name'],
                        cpu_percent=proc_info['cpu_percent'] or 0,
                        memory_percent=proc_info['memory_percent'] or 0,
                        memory_rss=proc_info['memory_info'].rss if proc_info['memory_info'] else 0,
                        status=proc_info['status'],
                        create_time=proc_info['create_time']
                    )
                    
                    process_metrics.append(metrics)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.logger.error(f"收集进程指标失败: {e}")
        
        return process_metrics


class MetricsStorage:
    """指标存储器"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        """
        初始化指标存储器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.logger = get_logger('monitoring.storage')
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建系统指标表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        cpu_percent REAL,
                        memory_percent REAL,
                        memory_available INTEGER,
                        disk_usage_percent REAL,
                        disk_free INTEGER,
                        network_sent INTEGER,
                        network_recv INTEGER,
                        load_average TEXT,
                        process_count INTEGER
                    )
                ''')
                
                # 创建进程指标表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS process_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        pid INTEGER,
                        name TEXT,
                        cpu_percent REAL,
                        memory_percent REAL,
                        memory_rss INTEGER,
                        status TEXT,
                        create_time REAL
                    )
                ''')
                
                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_process_timestamp ON process_metrics(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"初始化数据库失败: {e}")
            raise
    
    def store_system_metrics(self, metrics: SystemMetrics):
        """
        存储系统指标
        
        Args:
            metrics: 系统指标数据
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, memory_available,
                        disk_usage_percent, disk_free, network_sent, network_recv,
                        load_average, process_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp,
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.memory_available,
                    metrics.disk_usage_percent,
                    metrics.disk_free,
                    metrics.network_sent,
                    metrics.network_recv,
                    json.dumps(metrics.load_average),
                    metrics.process_count
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"存储系统指标失败: {e}")
    
    def store_process_metrics(self, metrics_list: List[ProcessMetrics]):
        """
        存储进程指标
        
        Args:
            metrics_list: 进程指标列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                
                for metrics in metrics_list:
                    cursor.execute('''
                        INSERT INTO process_metrics (
                            timestamp, pid, name, cpu_percent, memory_percent,
                            memory_rss, status, create_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp,
                        metrics.pid,
                        metrics.name,
                        metrics.cpu_percent,
                        metrics.memory_percent,
                        metrics.memory_rss,
                        metrics.status,
                        metrics.create_time
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"存储进程指标失败: {e}")
    
    def get_system_metrics(self, start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
        """
        获取系统指标
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict]: 系统指标列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM system_metrics"
                params = []
                
                if start_time or end_time:
                    query += " WHERE"
                    conditions = []
                    
                    if start_time:
                        conditions.append(" timestamp >= ?")
                        params.append(start_time.isoformat())
                    
                    if end_time:
                        conditions.append(" timestamp <= ?")
                        params.append(end_time.isoformat())
                    
                    query += " AND".join(conditions)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # 转换为字典列表
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"获取系统指标失败: {e}")
            return []
    
    def cleanup_old_data(self, days: int = 30):
        """
        清理旧数据
        
        Args:
            days: 保留天数
        """
        try:
            cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 删除旧的系统指标
                cursor.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_time,))
                system_deleted = cursor.rowcount
                
                # 删除旧的进程指标
                cursor.execute("DELETE FROM process_metrics WHERE timestamp < ?", (cutoff_time,))
                process_deleted = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"清理旧数据完成: 系统指标 {system_deleted} 条, 进程指标 {process_deleted} 条")
                
        except Exception as e:
            self.logger.error(f"清理旧数据失败: {e}")


class SystemMonitor:
    """系统监控器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化系统监控器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = get_logger('monitoring')
        
        # 初始化组件
        self.collector = MetricsCollector()
        self.storage = MetricsStorage(self.config.get('db_path', 'monitoring.db'))
        self.alert_manager = AlertManager()
        
        # 添加别名属性以兼容测试
        self.metric_collector = self.collector
        
        # 监控状态
        self.running = False
        self.monitor_thread = None
        
        # 添加 system_status 属性以兼容测试
        self.system_status = "stopped"
        
        # 配置参数
        self.collect_interval = self.config.get('collect_interval', 60)  # 收集间隔（秒）
        self.monitored_processes = self.config.get('monitored_processes', [])
        
        # 设置默认告警规则
        self._setup_default_alerts()
        
        # 设置通知处理器
        self._setup_notification_handlers()
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """获取指标数据"""
        return getattr(self.collector, 'metrics', [])
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警数据"""
        return [alert.to_dict() for alert in self.alert_manager.get_active_alerts()]
    
    def add_metric(self, metric: Dict[str, Any]):
        """添加指标"""
        if not hasattr(self.collector, 'metrics'):
            self.collector.metrics = []
        self.collector.metrics.append(metric)
    
    def create_alert(self, level: str, title: str, message: str, source: str = "system", data: Dict[str, Any] = None):
        """创建告警"""
        return self.alert_manager.create_alert(level, title, message, source, data)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": "running" if self.running else "stopped",
            "metrics_count": len(getattr(self.collector, 'metrics', [])),
            "alerts_count": len(self.alert_manager.get_active_alerts())
        }
    
    def _setup_default_alerts(self):
        """设置默认告警规则"""
        default_rules = [
            AlertRule('高CPU使用率', 'cpu_percent', '>', 80.0, 300),
            AlertRule('高内存使用率', 'memory_percent', '>', 85.0, 300),
            AlertRule('磁盘空间不足', 'disk_usage_percent', '>', 90.0, 600),
            AlertRule('系统负载过高', 'load_average', '>', 5.0, 300),
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
    
    def _setup_notification_handlers(self):
        """设置通知处理器"""
        # 邮件通知
        if self.config.get('email_notifications'):
            self.alert_manager.add_notification_handler(self._send_email_notification)
        
        # Webhook通知
        if self.config.get('webhook_url'):
            self.alert_manager.add_notification_handler(self._send_webhook_notification)
    
    def _send_email_notification(self, alert_data: Dict[str, Any]):
        """发送邮件通知"""
        try:
            email_config = self.config.get('email_config', {})
            
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_email')
            msg['To'] = email_config.get('to_email')
            msg['Subject'] = f"系统告警: {alert_data['rule_name']}"
            
            body = f"""
            告警规则: {alert_data['rule_name']}
            监控指标: {alert_data['metric']}
            告警阈值: {alert_data['threshold']}
            当前数值: {alert_data['current_value']}
            告警时间: {alert_data['timestamp']}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(email_config.get('smtp_server'), email_config.get('smtp_port'))
            server.starttls()
            server.login(email_config.get('username'), email_config.get('password'))
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"邮件告警发送成功: {alert_data['rule_name']}")
            
        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {e}")
    
    def _send_webhook_notification(self, alert_data: Dict[str, Any]):
        """发送Webhook通知"""
        try:
            webhook_url = self.config.get('webhook_url')
            
            payload = {
                'text': f"系统告警: {alert_data['rule_name']}",
                'attachments': [{
                    'color': 'danger',
                    'fields': [
                        {'title': '告警规则', 'value': alert_data['rule_name'], 'short': True},
                        {'title': '监控指标', 'value': alert_data['metric'], 'short': True},
                        {'title': '告警阈值', 'value': str(alert_data['threshold']), 'short': True},
                        {'title': '当前数值', 'value': str(alert_data['current_value']), 'short': True},
                        {'title': '告警时间', 'value': alert_data['timestamp'], 'short': False},
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook告警发送成功: {alert_data['rule_name']}")
            
        except Exception as e:
            self.logger.error(f"发送Webhook告警失败: {e}")
    
    def start(self):
        """启动监控"""
        if self.running:
            self.logger.warning("监控已在运行中")
            return
        
        self.running = True
        self.system_status = "running"  # 更新状态
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("系统监控已启动")
    
    def start_monitoring(self):
        """启动监控（别名方法）"""
        return self.start()
    
    def stop(self):
        """停止监控"""
        if not self.running:
            return
        
        self.running = False
        self.system_status = "stopped"  # 更新状态
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("系统监控已停止")
    
    def stop_monitoring(self):
        """停止监控（别名方法）"""
        return self.stop()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 收集系统指标
                system_metrics = self.collector.collect_system_metrics()
                
                # 存储指标
                self.storage.store_system_metrics(system_metrics)
                
                # 收集进程指标
                if self.monitored_processes:
                    process_metrics = self.collector.collect_process_metrics(self.monitored_processes)
                    self.storage.store_process_metrics(process_metrics)
                
                # 检查告警
                self.alert_manager.check_alerts(system_metrics)
                
                # 记录性能日志
                logger_manager.log_performance(
                    'system_monitoring',
                    self.collect_interval,
                    cpu_percent=system_metrics.cpu_percent,
                    memory_percent=system_metrics.memory_percent,
                    disk_usage_percent=system_metrics.disk_usage_percent
                )
                
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
            
            # 等待下次收集
            time.sleep(self.collect_interval)
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        获取当前系统状态
        
        Returns:
            Dict[str, Any]: 系统状态信息
        """
        try:
            metrics = self.collector.collect_system_metrics()
            return {
                'status': 'running' if self.running else 'stopped',
                'metrics': metrics.to_dict(),
                'alert_rules_count': len(self.alert_manager.rules),
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_manager.add_rule(rule)
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict]:
        """
        获取指标历史
        
        Args:
            hours: 历史小时数
            
        Returns:
            List[Dict]: 历史指标数据
        """
        start_time = datetime.now() - timedelta(hours=hours)
        return self.storage.get_system_metrics(start_time=start_time)


# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        'collect_interval': 30,
        'monitored_processes': ['python', 'nginx', 'redis-server'],
        'email_notifications': True,
        'email_config': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from_email': 'monitor@example.com',
            'to_email': 'admin@example.com',
            'username': 'monitor@example.com',
            'password': 'your_password'
        },
        'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    }
    
    # 创建监控器
    monitor = SystemMonitor(config)
    
    # 添加自定义告警规则
    custom_rule = AlertRule('进程数过多', 'process_count', '>', 500, 180)
    monitor.add_alert_rule(custom_rule)
    
    # 启动监控
    monitor.start()
    
    try:
        # 保持运行
        while True:
            status = monitor.get_current_status()
            print(f"系统状态: {status}")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("停止监控...")
        monitor.stop()