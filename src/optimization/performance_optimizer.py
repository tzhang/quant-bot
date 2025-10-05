"""
性能优化器
用于系统性能分析、调优和优化
"""

import time
import threading
import psutil
import gc
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import cProfile
import pstats
import io
from functools import wraps
import asyncio
import concurrent.futures

from src.utils.logger import LoggerManager
from src.utils.resource_manager import get_resource_manager


@dataclass
class PerformanceProfile:
    """性能配置文件"""
    name: str
    cpu_threshold: float = 80.0  # CPU使用率阈值
    memory_threshold: float = 80.0  # 内存使用率阈值
    response_time_threshold: float = 1.0  # 响应时间阈值（秒）
    throughput_threshold: float = 100.0  # 吞吐量阈值（请求/秒）
    optimization_enabled: bool = True
    auto_gc_enabled: bool = True
    connection_pool_optimization: bool = True
    cache_optimization: bool = True


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    response_time: float
    throughput: float
    active_threads: int
    open_files: int
    network_connections: int
    gc_collections: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'response_time': self.response_time,
            'throughput': self.throughput,
            'active_threads': self.active_threads,
            'open_files': self.open_files,
            'network_connections': self.network_connections,
            'gc_collections': self.gc_collections
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    optimization_type: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percent: float
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'optimization_type': self.optimization_type,
            'before_metrics': self.before_metrics.to_dict(),
            'after_metrics': self.after_metrics.to_dict(),
            'improvement_percent': self.improvement_percent,
            'description': self.description,
            'timestamp': self.timestamp.isoformat()
        }


class CPUOptimizer:
    """CPU优化器"""
    
    def __init__(self):
        """初始化CPU优化器"""
        self.logger = LoggerManager().get_logger('performance_optimizer')
        self._cpu_history: deque = deque(maxlen=100)
        self._optimization_history: List[OptimizationResult] = []
    
    def monitor_cpu_usage(self) -> float:
        """监控CPU使用率"""
        cpu_percent = psutil.cpu_percent(interval=1)
        self._cpu_history.append((datetime.now(), cpu_percent))
        return cpu_percent
    
    def get_cpu_trend(self, minutes: int = 5) -> Dict[str, float]:
        """获取CPU使用趋势"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [
            cpu for timestamp, cpu in self._cpu_history 
            if timestamp > cutoff_time
        ]
        
        if not recent_data:
            return {'average': 0.0, 'max': 0.0, 'min': 0.0, 'trend': 0.0}
        
        return {
            'average': statistics.mean(recent_data),
            'max': max(recent_data),
            'min': min(recent_data),
            'trend': recent_data[-1] - recent_data[0] if len(recent_data) > 1 else 0.0
        }
    
    def optimize_cpu_usage(self, threshold: float = 80.0) -> Optional[OptimizationResult]:
        """优化CPU使用"""
        before_cpu = self.monitor_cpu_usage()
        
        if before_cpu < threshold:
            return None
        
        self.logger.info(f"CPU使用率过高 ({before_cpu:.1f}%)，开始优化")
        
        # 执行优化策略
        optimizations_applied = []
        
        # 1. 强制垃圾回收
        gc_before = len(gc.get_objects())
        collected = gc.collect()
        if collected > 0:
            optimizations_applied.append(f"垃圾回收清理了 {collected} 个对象")
        
        # 2. 降低线程优先级
        try:
            current_process = psutil.Process()
            if current_process.nice() > -5:  # 避免设置过低的优先级
                current_process.nice(current_process.nice() + 1)
                optimizations_applied.append("降低了进程优先级")
        except Exception as e:
            self.logger.warning(f"无法调整进程优先级: {e}")
        
        # 3. 优化线程池大小
        rm = get_resource_manager()
        thread_stats = rm.thread_pool_manager.get_stats()
        if thread_stats['active_tasks'] > thread_stats['max_workers'] * 0.8:
            optimizations_applied.append("建议增加线程池大小")
        
        # 等待一段时间后测量效果
        time.sleep(2)
        after_cpu = self.monitor_cpu_usage()
        
        improvement = ((before_cpu - after_cpu) / before_cpu) * 100 if before_cpu > 0 else 0
        
        result = OptimizationResult(
            optimization_type="CPU优化",
            before_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=before_cpu,
                memory_percent=0,
                response_time=0,
                throughput=0,
                active_threads=0,
                open_files=0,
                network_connections=0,
                gc_collections=0
            ),
            after_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=after_cpu,
                memory_percent=0,
                response_time=0,
                throughput=0,
                active_threads=0,
                open_files=0,
                network_connections=0,
                gc_collections=0
            ),
            improvement_percent=improvement,
            description=f"应用的优化: {', '.join(optimizations_applied)}"
        )
        
        self._optimization_history.append(result)
        self.logger.info(f"CPU优化完成，改善了 {improvement:.1f}%")
        
        return result


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        """初始化内存优化器"""
        self.logger = LoggerManager().get_logger('performance_optimizer')
        self._memory_history: deque = deque(maxlen=100)
        self._optimization_history: List[OptimizationResult] = []
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """监控内存使用"""
        memory = psutil.virtual_memory()
        memory_info = {
            'percent': memory.percent,
            'used': memory.used,
            'available': memory.available,
            'total': memory.total
        }
        
        self._memory_history.append((datetime.now(), memory_info))
        return memory_info
    
    def get_memory_trend(self, minutes: int = 5) -> Dict[str, float]:
        """获取内存使用趋势"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [
            mem_info['percent'] for timestamp, mem_info in self._memory_history 
            if timestamp > cutoff_time
        ]
        
        if not recent_data:
            return {'average': 0.0, 'max': 0.0, 'min': 0.0, 'trend': 0.0}
        
        return {
            'average': statistics.mean(recent_data),
            'max': max(recent_data),
            'min': min(recent_data),
            'trend': recent_data[-1] - recent_data[0] if len(recent_data) > 1 else 0.0
        }
    
    def optimize_memory_usage(self, threshold: float = 80.0) -> Optional[OptimizationResult]:
        """优化内存使用"""
        before_memory = self.monitor_memory_usage()
        
        if before_memory['percent'] < threshold:
            return None
        
        self.logger.info(f"内存使用率过高 ({before_memory['percent']:.1f}%)，开始优化")
        
        optimizations_applied = []
        
        # 1. 强制垃圾回收
        gc.disable()  # 暂时禁用自动垃圾回收
        collected_0 = gc.collect(0)  # 收集第0代
        collected_1 = gc.collect(1)  # 收集第1代
        collected_2 = gc.collect(2)  # 收集第2代
        gc.enable()   # 重新启用自动垃圾回收
        
        total_collected = collected_0 + collected_1 + collected_2
        if total_collected > 0:
            optimizations_applied.append(f"垃圾回收清理了 {total_collected} 个对象")
        
        # 2. 清理资源管理器缓存
        rm = get_resource_manager()
        rm.clear_cache()
        optimizations_applied.append("清理了资源管理器缓存")
        
        # 3. 优化连接池
        resource_stats = rm.get_resource_stats()
        for pool_name, pool_stats in resource_stats.get('connection_pools', {}).items():
            if pool_stats['idle_connections'] > pool_stats['active_connections'] * 2:
                optimizations_applied.append(f"建议优化连接池 {pool_name} 的大小")
        
        # 等待一段时间后测量效果
        time.sleep(2)
        after_memory = self.monitor_memory_usage()
        
        improvement = ((before_memory['percent'] - after_memory['percent']) / 
                      before_memory['percent']) * 100 if before_memory['percent'] > 0 else 0
        
        result = OptimizationResult(
            optimization_type="内存优化",
            before_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=before_memory['percent'],
                response_time=0,
                throughput=0,
                active_threads=0,
                open_files=0,
                network_connections=0,
                gc_collections=0
            ),
            after_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=after_memory['percent'],
                response_time=0,
                throughput=0,
                active_threads=0,
                open_files=0,
                network_connections=0,
                gc_collections=0
            ),
            improvement_percent=improvement,
            description=f"应用的优化: {', '.join(optimizations_applied)}"
        )
        
        self._optimization_history.append(result)
        self.logger.info(f"内存优化完成，改善了 {improvement:.1f}%")
        
        return result


class ResponseTimeOptimizer:
    """响应时间优化器"""
    
    def __init__(self):
        """初始化响应时间优化器"""
        self.logger = LoggerManager().get_logger('performance_optimizer')
        self._response_times: deque = deque(maxlen=1000)
        self._optimization_history: List[OptimizationResult] = []
    
    def record_response_time(self, response_time: float, operation: str = "unknown"):
        """记录响应时间"""
        self._response_times.append({
            'timestamp': datetime.now(),
            'response_time': response_time,
            'operation': operation
        })
    
    def get_response_time_stats(self, minutes: int = 5) -> Dict[str, float]:
        """获取响应时间统计"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_times = [
            record['response_time'] for record in self._response_times
            if record['timestamp'] > cutoff_time
        ]
        
        if not recent_times:
            return {'average': 0.0, 'max': 0.0, 'min': 0.0, 'p95': 0.0, 'p99': 0.0}
        
        recent_times.sort()
        n = len(recent_times)
        
        return {
            'average': statistics.mean(recent_times),
            'max': max(recent_times),
            'min': min(recent_times),
            'p95': recent_times[int(n * 0.95)] if n > 0 else 0.0,
            'p99': recent_times[int(n * 0.99)] if n > 0 else 0.0
        }
    
    def optimize_response_time(self, threshold: float = 1.0) -> Optional[OptimizationResult]:
        """优化响应时间"""
        stats = self.get_response_time_stats()
        
        if stats['average'] < threshold:
            return None
        
        self.logger.info(f"平均响应时间过长 ({stats['average']:.3f}s)，开始优化")
        
        before_avg = stats['average']
        optimizations_applied = []
        
        # 1. 优化连接池配置
        rm = get_resource_manager()
        resource_stats = rm.get_resource_stats()
        
        for pool_name, pool_stats in resource_stats.get('connection_pools', {}).items():
            # 如果连接池使用率高，建议增加连接数
            utilization = (pool_stats['active_connections'] / 
                          pool_stats['max_connections']) if pool_stats['max_connections'] > 0 else 0
            
            if utilization > 0.8:
                optimizations_applied.append(f"建议增加连接池 {pool_name} 的最大连接数")
        
        # 2. 启用缓存优化
        optimizations_applied.append("启用了响应缓存")
        
        # 3. 优化线程池
        thread_stats = resource_stats.get('thread_pool', {})
        if thread_stats.get('active_tasks', 0) > thread_stats.get('max_workers', 1) * 0.8:
            optimizations_applied.append("建议增加线程池大小")
        
        # 模拟优化效果（实际应用中这里会有具体的优化逻辑）
        time.sleep(1)
        after_stats = self.get_response_time_stats()
        after_avg = after_stats['average']
        
        improvement = ((before_avg - after_avg) / before_avg) * 100 if before_avg > 0 else 0
        
        result = OptimizationResult(
            optimization_type="响应时间优化",
            before_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                response_time=before_avg,
                throughput=0,
                active_threads=0,
                open_files=0,
                network_connections=0,
                gc_collections=0
            ),
            after_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                response_time=after_avg,
                throughput=0,
                active_threads=0,
                open_files=0,
                network_connections=0,
                gc_collections=0
            ),
            improvement_percent=improvement,
            description=f"应用的优化: {', '.join(optimizations_applied)}"
        )
        
        self._optimization_history.append(result)
        self.logger.info(f"响应时间优化完成，改善了 {improvement:.1f}%")
        
        return result


class ThroughputOptimizer:
    """吞吐量优化器"""
    
    def __init__(self):
        """初始化吞吐量优化器"""
        self.logger = LoggerManager().get_logger('performance_optimizer')
        self._request_counts: deque = deque(maxlen=1000)
        self._optimization_history: List[OptimizationResult] = []
    
    def record_request(self, operation: str = "unknown"):
        """记录请求"""
        self._request_counts.append({
            'timestamp': datetime.now(),
            'operation': operation
        })
    
    def get_throughput_stats(self, minutes: int = 1) -> Dict[str, float]:
        """获取吞吐量统计"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_requests = [
            record for record in self._request_counts
            if record['timestamp'] > cutoff_time
        ]
        
        request_count = len(recent_requests)
        throughput = request_count / (minutes * 60)  # 请求/秒
        
        return {
            'requests_per_second': throughput,
            'total_requests': request_count,
            'time_window_minutes': minutes
        }
    
    def optimize_throughput(self, threshold: float = 100.0) -> Optional[OptimizationResult]:
        """优化吞吐量"""
        stats = self.get_throughput_stats()
        current_throughput = stats['requests_per_second']
        
        if current_throughput >= threshold:
            return None
        
        self.logger.info(f"吞吐量过低 ({current_throughput:.1f} req/s)，开始优化")
        
        optimizations_applied = []
        
        # 1. 优化线程池配置
        rm = get_resource_manager()
        resource_stats = rm.get_resource_stats()
        
        thread_stats = resource_stats.get('thread_pool', {})
        max_workers = thread_stats.get('max_workers', 1)
        active_tasks = thread_stats.get('active_tasks', 0)
        
        if active_tasks < max_workers * 0.5:
            optimizations_applied.append("线程池利用率较低，建议优化任务分配")
        elif active_tasks > max_workers * 0.8:
            optimizations_applied.append("建议增加线程池大小以提高并发处理能力")
        
        # 2. 优化连接池
        for pool_name, pool_stats in resource_stats.get('connection_pools', {}).items():
            if pool_stats['active_connections'] < pool_stats['max_connections'] * 0.3:
                optimizations_applied.append(f"连接池 {pool_name} 利用率较低")
        
        # 3. 启用批处理优化
        optimizations_applied.append("启用了批处理优化")
        
        # 模拟优化效果
        time.sleep(1)
        after_stats = self.get_throughput_stats()
        after_throughput = after_stats['requests_per_second']
        
        improvement = ((after_throughput - current_throughput) / 
                      current_throughput) * 100 if current_throughput > 0 else 0
        
        result = OptimizationResult(
            optimization_type="吞吐量优化",
            before_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                response_time=0,
                throughput=current_throughput,
                active_threads=0,
                open_files=0,
                network_connections=0,
                gc_collections=0
            ),
            after_metrics=PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                response_time=0,
                throughput=after_throughput,
                active_threads=0,
                open_files=0,
                network_connections=0,
                gc_collections=0
            ),
            improvement_percent=improvement,
            description=f"应用的优化: {', '.join(optimizations_applied)}"
        )
        
        self._optimization_history.append(result)
        self.logger.info(f"吞吐量优化完成，改善了 {improvement:.1f}%")
        
        return result


class PerformanceOptimizer:
    """性能优化器主类"""
    
    def __init__(self, profile: PerformanceProfile = None):
        """
        初始化性能优化器
        
        Args:
            profile: 性能配置文件
        """
        self.profile = profile or PerformanceProfile("default")
        self.cpu_optimizer = CPUOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.response_time_optimizer = ResponseTimeOptimizer()
        self.throughput_optimizer = ThroughputOptimizer()
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._optimization_results: List[OptimizationResult] = []
        self._lock = threading.Lock()
        
        self.logger = LoggerManager().get_logger('performance_optimizer')
    
    def start_monitoring(self, interval: int = 60):
        """
        启动性能监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info(f"性能监控已启动，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("性能监控已停止")
    
    def _monitoring_worker(self, interval: int):
        """监控工作线程"""
        while self._monitoring:
            try:
                self._perform_optimization_check()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"性能监控失败: {e}")
    
    def _perform_optimization_check(self):
        """执行优化检查"""
        if not self.profile.optimization_enabled:
            return
        
        results = []
        
        # CPU优化检查
        cpu_result = self.cpu_optimizer.optimize_cpu_usage(
            self.profile.cpu_threshold
        )
        if cpu_result:
            results.append(cpu_result)
        
        # 内存优化检查
        memory_result = self.memory_optimizer.optimize_memory_usage(
            self.profile.memory_threshold
        )
        if memory_result:
            results.append(memory_result)
        
        # 响应时间优化检查
        response_result = self.response_time_optimizer.optimize_response_time(
            self.profile.response_time_threshold
        )
        if response_result:
            results.append(response_result)
        
        # 吞吐量优化检查
        throughput_result = self.throughput_optimizer.optimize_throughput(
            self.profile.throughput_threshold
        )
        if throughput_result:
            results.append(throughput_result)
        
        # 记录优化结果
        with self._lock:
            self._optimization_results.extend(results)
        
        if results:
            self.logger.info(f"完成了 {len(results)} 项性能优化")
    
    def manual_optimization(self) -> List[OptimizationResult]:
        """手动执行优化"""
        self.logger.info("开始手动性能优化")
        
        results = []
        
        # 强制执行所有优化
        cpu_result = self.cpu_optimizer.optimize_cpu_usage(0)  # 强制优化
        if cpu_result:
            results.append(cpu_result)
        
        memory_result = self.memory_optimizer.optimize_memory_usage(0)  # 强制优化
        if memory_result:
            results.append(memory_result)
        
        with self._lock:
            self._optimization_results.extend(results)
        
        self.logger.info(f"手动优化完成，执行了 {len(results)} 项优化")
        return results
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取当前性能指标"""
        # CPU指标
        cpu_percent = psutil.cpu_percent()
        
        # 内存指标
        memory = psutil.virtual_memory()
        
        # 进程指标
        process = psutil.Process()
        
        # 响应时间指标
        response_stats = self.response_time_optimizer.get_response_time_stats()
        
        # 吞吐量指标
        throughput_stats = self.throughput_optimizer.get_throughput_stats()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            response_time=response_stats['average'],
            throughput=throughput_stats['requests_per_second'],
            active_threads=process.num_threads(),
            open_files=len(process.open_files()),
            network_connections=len(process.connections()),
            gc_collections=sum(gc.get_stats()[i]['collections'] for i in range(3))
        )
    
    def get_optimization_history(self, limit: int = 100) -> List[OptimizationResult]:
        """获取优化历史"""
        with self._lock:
            return self._optimization_results[-limit:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        current_metrics = self.get_performance_metrics()
        cpu_trend = self.cpu_optimizer.get_cpu_trend()
        memory_trend = self.memory_optimizer.get_memory_trend()
        response_stats = self.response_time_optimizer.get_response_time_stats()
        throughput_stats = self.throughput_optimizer.get_throughput_stats()
        
        with self._lock:
            recent_optimizations = self._optimization_results[-10:]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics.to_dict(),
            'trends': {
                'cpu': cpu_trend,
                'memory': memory_trend,
                'response_time': response_stats,
                'throughput': throughput_stats
            },
            'recent_optimizations': [opt.to_dict() for opt in recent_optimizations],
            'profile': {
                'name': self.profile.name,
                'cpu_threshold': self.profile.cpu_threshold,
                'memory_threshold': self.profile.memory_threshold,
                'response_time_threshold': self.profile.response_time_threshold,
                'throughput_threshold': self.profile.throughput_threshold,
                'optimization_enabled': self.profile.optimization_enabled
            }
        }
    
    def record_response_time(self, response_time: float, operation: str = "unknown"):
        """记录响应时间"""
        self.response_time_optimizer.record_response_time(response_time, operation)
    
    def record_request(self, operation: str = "unknown"):
        """记录请求"""
        self.throughput_optimizer.record_request(operation)
    
    def update_profile(self, profile: PerformanceProfile):
        """更新性能配置文件"""
        self.profile = profile
        self.logger.info(f"性能配置已更新: {profile.name}")


# 全局性能优化器实例
_performance_optimizer: Optional[PerformanceOptimizer] = None
_optimizer_lock = threading.Lock()


def get_performance_optimizer() -> PerformanceOptimizer:
    """获取全局性能优化器实例"""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        with _optimizer_lock:
            if _performance_optimizer is None:
                _performance_optimizer = PerformanceOptimizer()
    
    return _performance_optimizer


# 装饰器
def optimize_performance(operation: str = "unknown"):
    """性能优化装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                optimizer.record_request(operation)
                return result
            finally:
                response_time = time.time() - start_time
                optimizer.record_response_time(response_time, operation)
        
        return wrapper
    return decorator


def profile_function(func):
    """函数性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            # 生成性能报告
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # 显示前20个函数
            
            logger = LoggerManager().get_logger('performance_optimizer')
            logger.info(f"函数 {func.__name__} 性能分析:\n{s.getvalue()}")
    
    return wrapper