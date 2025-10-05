#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化器
用于优化Firstrade交易系统的性能和内存使用
"""

import gc
import psutil
import threading
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from functools import lru_cache, wraps
import weakref

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标类"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    cache_hit_rate: float = 0.0

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_mb: int = 512):
        """
        初始化内存管理器
        
        Args:
            max_memory_mb: 最大内存使用量(MB)
        """
        self.max_memory_mb = max_memory_mb
        self.cache_registry = weakref.WeakSet()
        self.cleanup_threshold = 0.8  # 内存使用超过80%时清理
        
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # 物理内存
            "vms_mb": memory_info.vms / 1024 / 1024,  # 虚拟内存
            "percent": process.memory_percent(),       # 内存使用百分比
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def should_cleanup(self) -> bool:
        """检查是否需要清理内存"""
        memory_usage = self.get_memory_usage()
        return memory_usage["rss_mb"] > self.max_memory_mb * self.cleanup_threshold
    
    def cleanup_memory(self):
        """清理内存"""
        logger.info("开始内存清理...")
        
        # 强制垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收清理了 {collected} 个对象")
        
        # 清理缓存
        self._clear_caches()
        
        # 记录清理后的内存使用
        memory_usage = self.get_memory_usage()
        logger.info(f"内存清理完成，当前使用: {memory_usage['rss_mb']:.2f}MB")
    
    def _clear_caches(self):
        """清理所有注册的缓存"""
        for cache in self.cache_registry:
            if hasattr(cache, 'clear'):
                cache.clear()
    
    def register_cache(self, cache):
        """注册缓存对象"""
        self.cache_registry.add(cache)

class ConnectionPool:
    """连接池管理器"""
    
    def __init__(self, max_connections: int = 10, timeout: int = 30):
        """
        初始化连接池
        
        Args:
            max_connections: 最大连接数
            timeout: 连接超时时间
        """
        self.max_connections = max_connections
        self.timeout = timeout
        self.active_connections = 0
        self.connection_lock = threading.Lock()
        self.session = None
        
    async def get_session(self) -> aiohttp.ClientSession:
        """获取异步HTTP会话"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections // 2,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        
        return self.session
    
    async def close(self):
        """关闭连接池"""
        if self.session and not self.session.closed:
            await self.session.close()

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间(秒)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Optional[any]:
        """获取缓存值"""
        if key in self.cache:
            # 检查是否过期
            if time.time() - self.access_times[key] < self.ttl:
                self.hit_count += 1
                self.access_times[key] = time.time()
                return self.cache[key]
            else:
                # 过期删除
                del self.cache[key]
                del self.access_times[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: any):
        """设置缓存值"""
        # 如果缓存已满，删除最旧的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, sample_interval: int = 5):
        """
        初始化性能监控器
        
        Args:
            sample_interval: 采样间隔(秒)
        """
        self.sample_interval = sample_interval
        self.metrics_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """开始监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # 保持历史记录在合理范围内
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]
            
            time.sleep(self.sample_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        process = psutil.Process()
        
        return PerformanceMetrics(
            cpu_usage=process.cpu_percent(),
            memory_usage=process.memory_info().rss / 1024 / 1024,
            memory_available=psutil.virtual_memory().available / 1024 / 1024,
            response_time=0.0,  # 需要在具体操作中测量
            throughput=0.0,     # 需要在具体操作中测量
            error_rate=0.0,     # 需要在具体操作中测量
            active_connections=0,  # 需要从连接池获取
            cache_hit_rate=0.0     # 需要从缓存管理器获取
        )
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, last_n: int = 10) -> Optional[PerformanceMetrics]:
        """获取平均性能指标"""
        if not self.metrics_history:
            return None
        
        recent_metrics = self.metrics_history[-last_n:]
        
        return PerformanceMetrics(
            cpu_usage=sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            memory_usage=sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            memory_available=sum(m.memory_available for m in recent_metrics) / len(recent_metrics),
            response_time=sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            throughput=sum(m.throughput for m in recent_metrics) / len(recent_metrics),
            error_rate=sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            active_connections=int(sum(m.active_connections for m in recent_metrics) / len(recent_metrics)),
            cache_hit_rate=sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        )

def performance_timer(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            logger.debug(f"{func.__name__} 执行时间: {end_time - start_time:.4f}秒")
    return wrapper

def memory_limit(max_memory_mb: int):
    """内存限制装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            if memory_before > max_memory_mb:
                logger.warning(f"内存使用超限: {memory_before:.2f}MB > {max_memory_mb}MB")
                gc.collect()
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / 1024 / 1024
            if memory_after > max_memory_mb:
                logger.warning(f"函数执行后内存超限: {memory_after:.2f}MB")
            
            return result
        return wrapper
    return decorator

class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        """
        初始化批处理器
        
        Args:
            batch_size: 批处理大小
            max_workers: 最大工作线程数
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batch(self, items: List[any], processor: Callable) -> List[any]:
        """
        批量处理项目
        
        Args:
            items: 待处理项目列表
            processor: 处理函数
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(processor, item) for item in batch]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"批处理项目失败: {e}")
                        results.append(None)
        
        return results

class PerformanceOptimizer:
    """性能优化器主类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化性能优化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 初始化各个组件
        self.memory_manager = MemoryManager(
            max_memory_mb=self.config.get('max_memory_mb', 512)
        )
        
        self.connection_pool = ConnectionPool(
            max_connections=self.config.get('max_connections', 10),
            timeout=self.config.get('timeout', 30)
        )
        
        self.cache_manager = CacheManager(
            max_size=self.config.get('cache_max_size', 1000),
            ttl=self.config.get('cache_ttl', 300)
        )
        
        self.performance_monitor = PerformanceMonitor(
            sample_interval=self.config.get('monitor_interval', 5)
        )
        
        self.batch_processor = BatchProcessor(
            batch_size=self.config.get('batch_size', 10),
            max_workers=self.config.get('max_workers', 4)
        )
        
        # 注册缓存到内存管理器
        self.memory_manager.register_cache(self.cache_manager)
    
    def start(self):
        """启动性能优化器"""
        self.performance_monitor.start_monitoring()
        logger.info("性能优化器已启动")
    
    def stop(self):
        """停止性能优化器"""
        self.performance_monitor.stop_monitoring()
        asyncio.run(self.connection_pool.close())
        logger.info("性能优化器已停止")
    
    def optimize_memory(self):
        """优化内存使用"""
        if self.memory_manager.should_cleanup():
            self.memory_manager.cleanup_memory()
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        current_metrics = self.performance_monitor.get_current_metrics()
        average_metrics = self.performance_monitor.get_average_metrics()
        memory_usage = self.memory_manager.get_memory_usage()
        cache_hit_rate = self.cache_manager.get_hit_rate()
        
        return {
            "current_metrics": current_metrics.__dict__ if current_metrics else None,
            "average_metrics": average_metrics.__dict__ if average_metrics else None,
            "memory_usage": memory_usage,
            "cache_hit_rate": cache_hit_rate,
            "timestamp": datetime.now().isoformat()
        }

# 示例使用
if __name__ == "__main__":
    # 创建性能优化器
    optimizer = PerformanceOptimizer({
        'max_memory_mb': 256,
        'max_connections': 5,
        'cache_max_size': 500,
        'monitor_interval': 3
    })
    
    # 启动优化器
    optimizer.start()
    
    try:
        # 模拟一些操作
        time.sleep(10)
        
        # 获取性能报告
        report = optimizer.get_performance_report()
        print("性能报告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
    finally:
        # 停止优化器
        optimizer.stop()