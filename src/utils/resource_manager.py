"""
资源管理器
用于管理系统资源、连接池和性能优化
"""

import threading
import time
import queue
import weakref
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
from functools import wraps

from src.utils.logger import LoggerManager


@dataclass
class ResourceMetrics:
    """资源使用指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: int  # bytes
    open_files: int
    network_connections: int
    thread_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used': self.memory_used,
            'open_files': self.open_files,
            'network_connections': self.network_connections,
            'thread_count': self.thread_count
        }


@dataclass
class ConnectionPoolConfig:
    """连接池配置"""
    min_connections: int = 5
    max_connections: int = 20
    max_idle_time: int = 300  # 秒
    connection_timeout: int = 30  # 秒
    retry_attempts: int = 3
    retry_delay: float = 1.0  # 秒
    health_check_interval: int = 60  # 秒


class Connection:
    """连接对象包装器"""
    
    def __init__(self, connection: Any, pool: 'ConnectionPool'):
        """
        初始化连接
        
        Args:
            connection: 实际的连接对象
            pool: 所属的连接池
        """
        self.connection = connection
        self.pool = pool
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.in_use = False
        self.is_healthy = True
    
    def __enter__(self):
        """进入上下文管理器"""
        self.in_use = True
        self.last_used = datetime.now()
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self.in_use = False
        self.pool.return_connection(self)
    
    def is_expired(self, max_idle_time: int) -> bool:
        """检查连接是否过期"""
        if self.in_use:
            return False
        
        idle_time = (datetime.now() - self.last_used).total_seconds()
        return idle_time > max_idle_time
    
    def close(self):
        """关闭连接"""
        try:
            if hasattr(self.connection, 'close'):
                self.connection.close()
        except Exception:
            pass  # 忽略关闭错误


class ConnectionPool:
    """通用连接池"""
    
    def __init__(self, 
                 connection_factory: Callable[[], Any],
                 config: ConnectionPoolConfig = None,
                 health_check: Callable[[Any], bool] = None):
        """
        初始化连接池
        
        Args:
            connection_factory: 创建连接的工厂函数
            config: 连接池配置
            health_check: 健康检查函数
        """
        self.connection_factory = connection_factory
        self.config = config or ConnectionPoolConfig()
        self.health_check = health_check
        
        self._pool: queue.Queue = queue.Queue(maxsize=self.config.max_connections)
        self._all_connections: List[Connection] = []
        self._lock = threading.RLock()
        self._closed = False
        
        # 统计信息
        self.created_connections = 0
        self.active_connections = 0
        self.failed_connections = 0
        
        # 启动健康检查线程
        if self.health_check:
            self._health_check_thread = threading.Thread(
                target=self._health_check_worker,
                daemon=True
            )
            self._health_check_thread.start()
        
        # 初始化最小连接数
        self._initialize_pool()
        
        self.logger = LoggerManager().get_logger('resource_manager')
    
    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(self.config.min_connections):
            try:
                conn = self._create_connection()
                if conn:
                    self._pool.put(conn, block=False)
            except queue.Full:
                break
            except Exception as e:
                self.logger.error(f"初始化连接池失败: {e}")
    
    def _create_connection(self) -> Optional[Connection]:
        """创建新连接"""
        if self._closed:
            return None
        
        if len(self._all_connections) >= self.config.max_connections:
            return None
        
        try:
            raw_conn = self.connection_factory()
            conn = Connection(raw_conn, self)
            
            with self._lock:
                self._all_connections.append(conn)
                self.created_connections += 1
            
            self.logger.debug(f"创建新连接，当前连接数: {len(self._all_connections)}")
            return conn
            
        except Exception as e:
            self.failed_connections += 1
            self.logger.error(f"创建连接失败: {e}")
            return None
    
    def get_connection(self, timeout: Optional[float] = None) -> Optional[Connection]:
        """
        获取连接
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            连接对象或None
        """
        if self._closed:
            return None
        
        timeout = timeout or self.config.connection_timeout
        
        # 尝试从池中获取连接
        try:
            conn = self._pool.get(timeout=timeout)
            if conn and conn.is_healthy:
                with self._lock:
                    self.active_connections += 1
                return conn
            elif conn:
                # 连接不健康，丢弃并创建新连接
                self._remove_connection(conn)
        except queue.Empty:
            pass
        
        # 池中没有可用连接，尝试创建新连接
        conn = self._create_connection()
        if conn:
            with self._lock:
                self.active_connections += 1
            return conn
        
        return None
    
    def return_connection(self, conn: Connection):
        """归还连接到池中"""
        if self._closed or not conn:
            return
        
        with self._lock:
            if self.active_connections > 0:
                self.active_connections -= 1
        
        # 检查连接是否过期或不健康
        if (conn.is_expired(self.config.max_idle_time) or 
            not conn.is_healthy or
            (self.health_check and not self.health_check(conn.connection))):
            self._remove_connection(conn)
            return
        
        # 归还到池中
        try:
            self._pool.put(conn, block=False)
        except queue.Full:
            # 池已满，关闭连接
            self._remove_connection(conn)
    
    def _remove_connection(self, conn: Connection):
        """从池中移除连接"""
        with self._lock:
            if conn in self._all_connections:
                self._all_connections.remove(conn)
        
        conn.close()
        self.logger.debug(f"移除连接，当前连接数: {len(self._all_connections)}")
    
    def _health_check_worker(self):
        """健康检查工作线程"""
        while not self._closed:
            try:
                time.sleep(self.config.health_check_interval)
                self._perform_health_check()
            except Exception as e:
                self.logger.error(f"健康检查失败: {e}")
    
    def _perform_health_check(self):
        """执行健康检查"""
        if not self.health_check:
            return
        
        unhealthy_connections = []
        
        with self._lock:
            for conn in self._all_connections:
                if not conn.in_use:
                    try:
                        if not self.health_check(conn.connection):
                            conn.is_healthy = False
                            unhealthy_connections.append(conn)
                    except Exception:
                        conn.is_healthy = False
                        unhealthy_connections.append(conn)
        
        # 移除不健康的连接
        for conn in unhealthy_connections:
            self._remove_connection(conn)
        
        if unhealthy_connections:
            self.logger.info(f"移除了 {len(unhealthy_connections)} 个不健康的连接")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self._lock:
            return {
                'total_connections': len(self._all_connections),
                'active_connections': self.active_connections,
                'idle_connections': len(self._all_connections) - self.active_connections,
                'pool_size': self._pool.qsize(),
                'created_connections': self.created_connections,
                'failed_connections': self.failed_connections,
                'max_connections': self.config.max_connections,
                'min_connections': self.config.min_connections
            }
    
    def close(self):
        """关闭连接池"""
        self._closed = True
        
        # 关闭所有连接
        with self._lock:
            for conn in self._all_connections:
                conn.close()
            self._all_connections.clear()
        
        # 清空队列
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("连接池已关闭")


class ThreadPoolManager:
    """线程池管理器"""
    
    def __init__(self, max_workers: int = None):
        """
        初始化线程池管理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        
        self.logger = LoggerManager().get_logger('resource_manager')
    
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> Future:
        """
        提交任务到线程池
        
        Args:
            task_id: 任务ID
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Future对象
        """
        future = self.executor.submit(func, *args, **kwargs)
        
        with self._lock:
            self.futures[task_id] = future
        
        # 添加完成回调
        future.add_done_callback(lambda f: self._cleanup_future(task_id))
        
        self.logger.debug(f"提交任务: {task_id}")
        return future
    
    def _cleanup_future(self, task_id: str):
        """清理完成的Future"""
        with self._lock:
            self.futures.pop(task_id, None)
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间
            
        Returns:
            任务结果
        """
        with self._lock:
            future = self.futures.get(task_id)
        
        if not future:
            raise ValueError(f"任务不存在: {task_id}")
        
        return future.result(timeout=timeout)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        with self._lock:
            future = self.futures.get(task_id)
        
        if future:
            return future.cancel()
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取线程池统计信息"""
        with self._lock:
            active_tasks = len(self.futures)
        
        return {
            'max_workers': self.max_workers,
            'active_tasks': active_tasks,
            'completed_tasks': getattr(self.executor, '_threads_queues', {})
        }
    
    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        self.executor.shutdown(wait=wait)
        self.logger.info("线程池已关闭")


class MemoryManager:
    """内存管理器"""
    
    def __init__(self, 
                 memory_threshold: float = 0.8,
                 gc_interval: int = 300):
        """
        初始化内存管理器
        
        Args:
            memory_threshold: 内存使用阈值（0-1）
            gc_interval: 垃圾回收间隔（秒）
        """
        self.memory_threshold = memory_threshold
        self.gc_interval = gc_interval
        self._weak_refs: weakref.WeakSet = weakref.WeakSet()
        self._cache: Dict[str, Any] = {}
        self._cache_access_times: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        # 启动内存监控线程
        self._monitor_thread = threading.Thread(
            target=self._memory_monitor,
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger = LoggerManager().get_logger('resource_manager')
    
    def register_object(self, obj: Any):
        """注册对象用于内存监控"""
        self._weak_refs.add(obj)
    
    def cache_object(self, key: str, obj: Any, ttl: Optional[int] = None):
        """
        缓存对象
        
        Args:
            key: 缓存键
            obj: 要缓存的对象
            ttl: 生存时间（秒）
        """
        with self._lock:
            self._cache[key] = obj
            self._cache_access_times[key] = datetime.now()
            
            if ttl:
                # 设置过期时间
                expire_time = datetime.now() + timedelta(seconds=ttl)
                threading.Timer(ttl, self._expire_cache_item, args=[key]).start()
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """获取缓存对象"""
        with self._lock:
            if key in self._cache:
                self._cache_access_times[key] = datetime.now()
                return self._cache[key]
        return None
    
    def _expire_cache_item(self, key: str):
        """过期缓存项"""
        with self._lock:
            self._cache.pop(key, None)
            self._cache_access_times.pop(key, None)
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._cache_access_times.clear()
        
        self.logger.info("缓存已清空")
    
    def _memory_monitor(self):
        """内存监控线程"""
        while True:
            try:
                time.sleep(self.gc_interval)
                self._check_memory_usage()
            except Exception as e:
                self.logger.error(f"内存监控失败: {e}")
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        if memory_percent > self.memory_threshold:
            self.logger.warning(f"内存使用率过高: {memory_percent:.2%}")
            self._perform_cleanup()
    
    def _perform_cleanup(self):
        """执行内存清理"""
        # 强制垃圾回收
        collected = gc.collect()
        self.logger.info(f"垃圾回收清理了 {collected} 个对象")
        
        # 清理旧的缓存项
        self._cleanup_old_cache_items()
    
    def _cleanup_old_cache_items(self, max_age: int = 3600):
        """清理旧的缓存项"""
        cutoff_time = datetime.now() - timedelta(seconds=max_age)
        expired_keys = []
        
        with self._lock:
            for key, access_time in self._cache_access_times.items():
                if access_time < cutoff_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._cache.pop(key, None)
                self._cache_access_times.pop(key, None)
        
        if expired_keys:
            self.logger.info(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        memory = psutil.virtual_memory()
        
        with self._lock:
            cache_size = len(self._cache)
        
        return {
            'total_memory': memory.total,
            'available_memory': memory.available,
            'used_memory': memory.used,
            'memory_percent': memory.percent,
            'cached_objects': cache_size,
            'registered_objects': len(self._weak_refs)
        }


class ResourceManager:
    """资源管理器主类"""
    
    def __init__(self):
        """初始化资源管理器"""
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.thread_pool_manager = ThreadPoolManager()
        self.memory_manager = MemoryManager()
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        self.logger = LoggerManager().get_logger('resource_manager')
    
    def create_connection_pool(self, 
                             name: str,
                             connection_factory: Callable[[], Any],
                             config: ConnectionPoolConfig = None,
                             health_check: Callable[[Any], bool] = None) -> ConnectionPool:
        """
        创建连接池
        
        Args:
            name: 连接池名称
            connection_factory: 连接工厂函数
            config: 连接池配置
            health_check: 健康检查函数
            
        Returns:
            连接池对象
        """
        with self._lock:
            if name in self.connection_pools:
                raise ValueError(f"连接池已存在: {name}")
            
            pool = ConnectionPool(connection_factory, config, health_check)
            self.connection_pools[name] = pool
            
            self.logger.info(f"创建连接池: {name}")
            return pool
    
    def get_connection_pool(self, name: str) -> Optional[ConnectionPool]:
        """获取连接池"""
        with self._lock:
            return self.connection_pools.get(name)
    
    @contextmanager
    def get_connection(self, pool_name: str, timeout: Optional[float] = None):
        """
        获取连接的上下文管理器
        
        Args:
            pool_name: 连接池名称
            timeout: 超时时间
        """
        pool = self.get_connection_pool(pool_name)
        if not pool:
            raise ValueError(f"连接池不存在: {pool_name}")
        
        conn = pool.get_connection(timeout)
        if not conn:
            raise RuntimeError(f"无法获取连接: {pool_name}")
        
        try:
            with conn as connection:
                yield connection
        finally:
            pass  # 连接会自动归还
    
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> Future:
        """提交任务到线程池"""
        return self.thread_pool_manager.submit_task(task_id, func, *args, **kwargs)
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        return self.thread_pool_manager.get_task_result(task_id, timeout)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.thread_pool_manager.cancel_task(task_id)
    
    def cache_object(self, key: str, obj: Any, ttl: Optional[int] = None):
        """缓存对象"""
        self.memory_manager.cache_object(key, obj, ttl)
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """获取缓存对象"""
        return self.memory_manager.get_cached_object(key)
    
    def clear_cache(self):
        """清空缓存"""
        self.memory_manager.clear_cache()
    
    def start_monitoring(self):
        """启动资源监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._resource_monitor,
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info("资源监控已启动")
    
    def stop_monitoring(self):
        """停止资源监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("资源监控已停止")
    
    def _resource_monitor(self):
        """资源监控线程"""
        while self._monitoring:
            try:
                time.sleep(60)  # 每分钟检查一次
                self._log_resource_stats()
            except Exception as e:
                self.logger.error(f"资源监控失败: {e}")
    
    def _log_resource_stats(self):
        """记录资源统计信息"""
        stats = self.get_resource_stats()
        self.logger.info(f"资源统计: {stats}")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        # 系统资源
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # 连接池统计
        pool_stats = {}
        with self._lock:
            for name, pool in self.connection_pools.items():
                pool_stats[name] = pool.get_stats()
        
        # 线程池统计
        thread_stats = self.thread_pool_manager.get_stats()
        
        # 内存管理统计
        memory_stats = self.memory_manager.get_memory_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used': memory.used,
                'memory_available': memory.available
            },
            'connection_pools': pool_stats,
            'thread_pool': thread_stats,
            'memory_manager': memory_stats
        }
    
    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        
        # 关闭所有连接池
        with self._lock:
            for pool in self.connection_pools.values():
                pool.close()
            self.connection_pools.clear()
        
        # 关闭线程池
        self.thread_pool_manager.shutdown()
        
        # 清空缓存
        self.memory_manager.clear_cache()
        
        self.logger.info("资源管理器已清理")


# 全局资源管理器实例
_resource_manager: Optional[ResourceManager] = None
_resource_manager_lock = threading.Lock()


def get_resource_manager() -> ResourceManager:
    """获取全局资源管理器实例"""
    global _resource_manager
    
    if _resource_manager is None:
        with _resource_manager_lock:
            if _resource_manager is None:
                _resource_manager = ResourceManager()
    
    return _resource_manager


def cleanup_resources():
    """清理全局资源"""
    global _resource_manager
    
    if _resource_manager:
        _resource_manager.cleanup()
        _resource_manager = None


# 装饰器
def with_connection(pool_name: str, timeout: Optional[float] = None):
    """连接装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rm = get_resource_manager()
            with rm.get_connection(pool_name, timeout) as conn:
                return func(conn, *args, **kwargs)
        return wrapper
    return decorator


def async_task(task_id: Optional[str] = None):
    """异步任务装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rm = get_resource_manager()
            tid = task_id or f"{func.__name__}_{int(time.time())}"
            return rm.submit_task(tid, func, *args, **kwargs)
        return wrapper
    return decorator


def cached(key: Optional[str] = None, ttl: Optional[int] = None):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rm = get_resource_manager()
            cache_key = key or f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            result = rm.get_cached_object(cache_key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            rm.cache_object(cache_key, result, ttl)
            return result
        return wrapper
    return decorator