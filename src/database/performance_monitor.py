"""
数据库性能监控模块

实现查询性能监控、慢查询分析、资源使用监控等功能
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
import statistics
from contextlib import contextmanager

from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import Pool

from .connection import get_db_session, get_engine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """查询指标"""
    query_id: str
    sql: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    rows_affected: Optional[int] = None
    rows_returned: Optional[int] = None
    connection_id: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
    
    def finish(self, rows_affected: int = None, rows_returned: int = None, error: str = None):
        """完成查询记录"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.rows_affected = rows_affected
        self.rows_returned = rows_returned
        self.error = error


@dataclass
class ConnectionMetrics:
    """连接指标"""
    connection_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    query_count: int = 0
    total_duration: float = 0.0
    active: bool = True
    
    def update_usage(self, duration: float):
        """更新使用统计"""
        self.last_used = datetime.now()
        self.query_count += 1
        self.total_duration += duration


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used: int = 0
    disk_io_read: int = 0
    disk_io_write: int = 0
    network_io_sent: int = 0
    network_io_recv: int = 0
    active_connections: int = 0
    pool_size: int = 0
    pool_checked_out: int = 0


class PerformanceMonitor:
    """
    数据库性能监控器
    
    实现查询性能监控、慢查询分析、资源使用监控等功能
    """
    
    def __init__(self, engine: Engine = None, slow_query_threshold: float = 1.0):
        """
        初始化性能监控器
        
        Args:
            engine: 数据库引擎
            slow_query_threshold: 慢查询阈值（秒）
        """
        self.engine = engine or get_engine()
        self.slow_query_threshold = slow_query_threshold
        self.logger = logger
        
        # 监控数据存储
        self.query_metrics: deque = deque(maxlen=10000)  # 最近10000条查询
        self.slow_queries: deque = deque(maxlen=1000)    # 最近1000条慢查询
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.system_metrics: deque = deque(maxlen=1440)  # 24小时的分钟级数据
        
        # 统计数据
        self.query_stats = defaultdict(list)  # 按SQL模式分组的统计
        self.hourly_stats = defaultdict(lambda: defaultdict(int))  # 按小时统计
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        # 注册SQLAlchemy事件监听器
        self._register_event_listeners()
        
        self.logger.info(f"性能监控器已初始化，慢查询阈值: {slow_query_threshold}秒")
    
    def _register_event_listeners(self):
        """注册SQLAlchemy事件监听器"""
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """查询执行前的事件处理"""
            context._query_start_time = time.time()
            context._query_id = f"{int(time.time() * 1000000)}"
            
            # 创建查询指标对象
            query_metrics = QueryMetrics(
                query_id=context._query_id,
                sql=statement,
                parameters=parameters if isinstance(parameters, dict) else {},
                connection_id=str(id(conn))
            )
            
            with self.lock:
                self.query_metrics.append(query_metrics)
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """查询执行后的事件处理"""
            if hasattr(context, '_query_start_time'):
                duration = time.time() - context._query_start_time
                query_id = getattr(context, '_query_id', 'unknown')
                
                # 更新查询指标
                with self.lock:
                    for metrics in reversed(self.query_metrics):
                        if metrics.query_id == query_id:
                            metrics.finish(
                                rows_affected=cursor.rowcount if hasattr(cursor, 'rowcount') else None
                            )
                            
                            # 检查是否为慢查询
                            if duration >= self.slow_query_threshold:
                                self.slow_queries.append(metrics)
                            
                            # 更新连接指标
                            conn_id = str(id(conn))
                            if conn_id not in self.connection_metrics:
                                self.connection_metrics[conn_id] = ConnectionMetrics(connection_id=conn_id)
                            
                            self.connection_metrics[conn_id].update_usage(duration)
                            
                            # 更新统计数据
                            self._update_query_stats(statement, duration)
                            break
        
        @event.listens_for(self.engine, "handle_error")
        def handle_error(exception_context):
            """错误处理事件"""
            if hasattr(exception_context.execution_context, '_query_id'):
                query_id = exception_context.execution_context._query_id
                
                with self.lock:
                    for metrics in reversed(self.query_metrics):
                        if metrics.query_id == query_id:
                            metrics.finish(error=str(exception_context.original_exception))
                            break
    
    def _update_query_stats(self, sql: str, duration: float):
        """更新查询统计"""
        # 简化SQL模式（移除参数值）
        sql_pattern = self._normalize_sql(sql)
        
        with self.lock:
            self.query_stats[sql_pattern].append(duration)
            
            # 按小时统计
            hour_key = datetime.now().strftime("%Y-%m-%d %H")
            self.hourly_stats[hour_key]['total_queries'] += 1
            self.hourly_stats[hour_key]['total_duration'] += duration
            
            if duration >= self.slow_query_threshold:
                self.hourly_stats[hour_key]['slow_queries'] += 1
    
    def _normalize_sql(self, sql: str) -> str:
        """
        标准化SQL语句，移除参数值
        
        Args:
            sql: 原始SQL语句
            
        Returns:
            str: 标准化后的SQL模式
        """
        # 简单的SQL标准化，实际应用中可能需要更复杂的逻辑
        import re
        
        # 移除多余空白
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # 替换数字参数
        sql = re.sub(r'\b\d+\b', '?', sql)
        
        # 替换字符串参数
        sql = re.sub(r"'[^']*'", '?', sql)
        sql = re.sub(r'"[^"]*"', '?', sql)
        
        return sql
    
    def start_monitoring(self, interval: int = 60):
        """
        启动系统监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring_active:
            self.logger.warning("监控已经在运行中")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system_metrics,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"系统监控已启动，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止系统监控"""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("系统监控已停止")
    
    def _monitor_system_metrics(self, interval: int):
        """监控系统指标"""
        while self.monitoring_active:
            try:
                # 获取系统指标
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # 获取连接池信息
                pool_size = self.engine.pool.size()
                pool_checked_out = self.engine.pool.checkedout()
                
                # 创建系统指标对象
                metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used=memory.used,
                    disk_io_read=disk_io.read_bytes if disk_io else 0,
                    disk_io_write=disk_io.write_bytes if disk_io else 0,
                    network_io_sent=network_io.bytes_sent if network_io else 0,
                    network_io_recv=network_io.bytes_recv if network_io else 0,
                    active_connections=len([c for c in self.connection_metrics.values() if c.active]),
                    pool_size=pool_size,
                    pool_checked_out=pool_checked_out
                )
                
                with self.lock:
                    self.system_metrics.append(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"监控系统指标时发生错误: {e}")
                time.sleep(interval)
    
    @contextmanager
    def query_profiler(self, query_name: str = "custom_query"):
        """
        查询性能分析上下文管理器
        
        Args:
            query_name: 查询名称
        """
        start_time = time.time()
        query_id = f"{query_name}_{int(time.time() * 1000000)}"
        
        try:
            yield query_id
        finally:
            duration = time.time() - start_time
            
            # 记录自定义查询指标
            metrics = QueryMetrics(
                query_id=query_id,
                sql=query_name,
                start_time=datetime.fromtimestamp(start_time)
            )
            metrics.finish()
            
            with self.lock:
                self.query_metrics.append(metrics)
                
                if duration >= self.slow_query_threshold:
                    self.slow_queries.append(metrics)
    
    def get_slow_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取慢查询列表
        
        Args:
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 慢查询列表
        """
        with self.lock:
            slow_queries = list(self.slow_queries)[-limit:]
        
        return [
            {
                'query_id': q.query_id,
                'sql': q.sql[:500] + '...' if len(q.sql) > 500 else q.sql,
                'duration': q.duration,
                'start_time': q.start_time.isoformat(),
                'rows_affected': q.rows_affected,
                'error': q.error
            }
            for q in reversed(slow_queries)
        ]
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        获取查询统计信息
        
        Returns:
            Dict[str, Any]: 查询统计信息
        """
        with self.lock:
            total_queries = len(self.query_metrics)
            slow_queries_count = len(self.slow_queries)
            
            if total_queries == 0:
                return {
                    'total_queries': 0,
                    'slow_queries': 0,
                    'slow_query_rate': 0.0,
                    'average_duration': 0.0,
                    'top_slow_patterns': []
                }
            
            # 计算平均执行时间
            durations = [q.duration for q in self.query_metrics if q.duration is not None]
            avg_duration = statistics.mean(durations) if durations else 0.0
            
            # 统计最慢的SQL模式
            pattern_stats = {}
            for pattern, durations in self.query_stats.items():
                if durations:
                    pattern_stats[pattern] = {
                        'count': len(durations),
                        'avg_duration': statistics.mean(durations),
                        'max_duration': max(durations),
                        'total_duration': sum(durations)
                    }
            
            # 按平均执行时间排序
            top_slow_patterns = sorted(
                pattern_stats.items(),
                key=lambda x: x[1]['avg_duration'],
                reverse=True
            )[:10]
            
            return {
                'total_queries': total_queries,
                'slow_queries': slow_queries_count,
                'slow_query_rate': slow_queries_count / total_queries * 100,
                'average_duration': avg_duration,
                'top_slow_patterns': [
                    {
                        'sql_pattern': pattern,
                        'count': stats['count'],
                        'avg_duration': stats['avg_duration'],
                        'max_duration': stats['max_duration'],
                        'total_duration': stats['total_duration']
                    }
                    for pattern, stats in top_slow_patterns
                ]
            }
    
    def get_connection_statistics(self) -> Dict[str, Any]:
        """
        获取连接统计信息
        
        Returns:
            Dict[str, Any]: 连接统计信息
        """
        with self.lock:
            active_connections = [c for c in self.connection_metrics.values() if c.active]
            
            if not active_connections:
                return {
                    'total_connections': 0,
                    'active_connections': 0,
                    'pool_size': self.engine.pool.size(),
                    'pool_checked_out': self.engine.pool.checkedout(),
                    'connection_details': []
                }
            
            # 计算连接统计
            total_queries = sum(c.query_count for c in active_connections)
            total_duration = sum(c.total_duration for c in active_connections)
            
            connection_details = []
            for conn in sorted(active_connections, key=lambda x: x.total_duration, reverse=True)[:20]:
                connection_details.append({
                    'connection_id': conn.connection_id,
                    'created_at': conn.created_at.isoformat(),
                    'last_used': conn.last_used.isoformat(),
                    'query_count': conn.query_count,
                    'total_duration': conn.total_duration,
                    'avg_duration': conn.total_duration / conn.query_count if conn.query_count > 0 else 0
                })
            
            return {
                'total_connections': len(self.connection_metrics),
                'active_connections': len(active_connections),
                'pool_size': self.engine.pool.size(),
                'pool_checked_out': self.engine.pool.checkedout(),
                'total_queries': total_queries,
                'total_duration': total_duration,
                'avg_queries_per_connection': total_queries / len(active_connections) if active_connections else 0,
                'connection_details': connection_details
            }
    
    def get_system_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """
        获取系统指标
        
        Args:
            hours: 获取最近几小时的数据
            
        Returns:
            List[Dict[str, Any]]: 系统指标列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_metrics = [
                m for m in self.system_metrics
                if m.timestamp >= cutoff_time
            ]
        
        return [
            {
                'timestamp': m.timestamp.isoformat(),
                'cpu_percent': m.cpu_percent,
                'memory_percent': m.memory_percent,
                'memory_used': m.memory_used,
                'disk_io_read': m.disk_io_read,
                'disk_io_write': m.disk_io_write,
                'network_io_sent': m.network_io_sent,
                'network_io_recv': m.network_io_recv,
                'active_connections': m.active_connections,
                'pool_size': m.pool_size,
                'pool_checked_out': m.pool_checked_out
            }
            for m in recent_metrics
        ]
    
    def get_hourly_statistics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        获取按小时统计的数据
        
        Args:
            hours: 获取最近几小时的数据
            
        Returns:
            List[Dict[str, Any]]: 按小时统计的数据
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            hourly_data = []
            
            for hour_key, stats in self.hourly_stats.items():
                hour_time = datetime.strptime(hour_key, "%Y-%m-%d %H")
                
                if hour_time >= cutoff_time:
                    avg_duration = (
                        stats['total_duration'] / stats['total_queries']
                        if stats['total_queries'] > 0 else 0
                    )
                    
                    hourly_data.append({
                        'hour': hour_key,
                        'total_queries': stats['total_queries'],
                        'slow_queries': stats['slow_queries'],
                        'total_duration': stats['total_duration'],
                        'avg_duration': avg_duration,
                        'slow_query_rate': (
                            stats['slow_queries'] / stats['total_queries'] * 100
                            if stats['total_queries'] > 0 else 0
                        )
                    })
            
            # 按时间排序
            hourly_data.sort(key=lambda x: x['hour'])
        
        return hourly_data
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        生成性能报告
        
        Returns:
            Dict[str, Any]: 性能报告
        """
        return {
            'report_time': datetime.now().isoformat(),
            'query_statistics': self.get_query_statistics(),
            'connection_statistics': self.get_connection_statistics(),
            'slow_queries': self.get_slow_queries(20),
            'hourly_statistics': self.get_hourly_statistics(24),
            'system_metrics_summary': self._get_system_metrics_summary()
        }
    
    def _get_system_metrics_summary(self) -> Dict[str, Any]:
        """获取系统指标摘要"""
        with self.lock:
            if not self.system_metrics:
                return {}
            
            recent_metrics = list(self.system_metrics)[-60:]  # 最近60个数据点
            
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            
            return {
                'cpu_avg': statistics.mean(cpu_values) if cpu_values else 0,
                'cpu_max': max(cpu_values) if cpu_values else 0,
                'memory_avg': statistics.mean(memory_values) if memory_values else 0,
                'memory_max': max(memory_values) if memory_values else 0,
                'current_connections': recent_metrics[-1].active_connections if recent_metrics else 0,
                'pool_utilization': (
                    recent_metrics[-1].pool_checked_out / recent_metrics[-1].pool_size * 100
                    if recent_metrics and recent_metrics[-1].pool_size > 0 else 0
                )
            }
    
    def clear_metrics(self, older_than_hours: int = 24):
        """
        清理旧的监控数据
        
        Args:
            older_than_hours: 清理多少小时前的数据
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self.lock:
            # 清理查询指标
            self.query_metrics = deque(
                [m for m in self.query_metrics if m.start_time >= cutoff_time],
                maxlen=self.query_metrics.maxlen
            )
            
            # 清理慢查询
            self.slow_queries = deque(
                [m for m in self.slow_queries if m.start_time >= cutoff_time],
                maxlen=self.slow_queries.maxlen
            )
            
            # 清理系统指标
            self.system_metrics = deque(
                [m for m in self.system_metrics if m.timestamp >= cutoff_time],
                maxlen=self.system_metrics.maxlen
            )
            
            # 清理小时统计
            cutoff_hour = cutoff_time.strftime("%Y-%m-%d %H")
            self.hourly_stats = {
                k: v for k, v in self.hourly_stats.items()
                if k >= cutoff_hour
            }
            
            # 清理查询统计
            for pattern in list(self.query_stats.keys()):
                # 保留最近的统计数据
                if len(self.query_stats[pattern]) > 1000:
                    self.query_stats[pattern] = self.query_stats[pattern][-1000:]
        
        self.logger.info(f"已清理 {older_than_hours} 小时前的监控数据")


# 全局性能监控器实例
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


if __name__ == "__main__":
    # 测试性能监控器
    monitor = PerformanceMonitor()
    
    # 启动监控
    monitor.start_monitoring(interval=10)
    
    # 模拟一些查询
    with get_db_session() as session:
        session.execute(text("SELECT 1"))
        time.sleep(0.1)
        session.execute(text("SELECT COUNT(*) FROM sqlite_master"))
    
    # 获取统计信息
    stats = monitor.get_query_statistics()
    print(f"查询统计: {stats}")
    
    # 生成报告
    report = monitor.generate_performance_report()
    print(f"性能报告: {json.dumps(report, indent=2, ensure_ascii=False)}")
    
    # 停止监控
    monitor.stop_monitoring()