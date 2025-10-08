"""
集成优化系统
整合智能缓存、内存池管理和性能分析功能
"""

import time
import threading
import os
import sys
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from contextlib import contextmanager

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入我们创建的优化组件
from smart_cache_system import SmartCacheSystem, get_cache_system
from memory_pool_manager import MemoryPoolManager, get_memory_pool, MemoryPoolContext
from performance_analyzer import PerformanceMonitor, get_performance_monitor, PerformanceContext
from adaptive_execution_strategy import AdaptiveExecutionStrategy, TaskMetrics, ExecutionStrategy


@dataclass
class OptimizationConfig:
    """优化配置"""
    enable_caching: bool = True
    enable_memory_pool: bool = True
    enable_performance_monitoring: bool = True
    cache_memory_limit_mb: int = 1024
    memory_pool_limit_mb: int = 2048
    auto_optimization: bool = True
    optimization_interval: float = 60.0  # 自动优化间隔(秒)


class IntegratedOptimizationSystem:
    """集成优化系统"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # 初始化各个组件
        self.cache_system = get_cache_system() if self.config.enable_caching else None
        self.memory_pool = get_memory_pool() if self.config.enable_memory_pool else None
        self.performance_monitor = get_performance_monitor() if self.config.enable_performance_monitoring else None
        self.adaptive_strategy = AdaptiveExecutionStrategy()
        
        # 优化状态
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_allocations': 0,
            'performance_optimizations': 0,
            'total_time_saved': 0.0
        }
        
        # 自动优化线程
        self._auto_optimization_thread = None
        self._stop_auto_optimization = False
        
        if self.config.auto_optimization:
            self.start_auto_optimization()
            
    def start_auto_optimization(self):
        """启动自动优化"""
        if self._auto_optimization_thread is None or not self._auto_optimization_thread.is_alive():
            self._stop_auto_optimization = False
            self._auto_optimization_thread = threading.Thread(
                target=self._auto_optimization_loop,
                daemon=True
            )
            self._auto_optimization_thread.start()
            
    def stop_auto_optimization(self):
        """停止自动优化"""
        self._stop_auto_optimization = True
        if self._auto_optimization_thread:
            self._auto_optimization_thread.join(timeout=1.0)
            
    def _auto_optimization_loop(self):
        """自动优化循环"""
        while not self._stop_auto_optimization:
            try:
                time.sleep(self.config.optimization_interval)
                self.optimize_system()
            except Exception as e:
                print(f"自动优化错误: {e}")
                
    def optimize_system(self):
        """系统优化"""
        optimizations_performed = 0
        
        # 缓存优化
        if self.cache_system:
            cache_stats = self.cache_system.get_comprehensive_stats()
            
            # 如果命中率低于50%，清理缓存
            if cache_stats.get('overall_hit_rate', 0) < 50:
                self.cache_system.clear_all_caches()
                optimizations_performed += 1
                
            # 如果内存使用过高，优化缓存
            if cache_stats.get('total_memory_mb', 0) > self.config.cache_memory_limit_mb:
                self.cache_system.optimize_performance()
                optimizations_performed += 1
                
        # 内存池优化
        if self.memory_pool:
            memory_usage = self.memory_pool.get_memory_usage()
            
            # 如果内存使用过高，进行优化
            if memory_usage.get('pool_memory_mb', 0) > self.config.memory_pool_limit_mb:
                self.memory_pool.optimize_memory()
                optimizations_performed += 1
                
        # 性能监控优化
        if self.performance_monitor:
            # 如果性能指标过多，清理旧数据
            if len(self.performance_monitor.metrics) > 10000:
                # 保留最近的5000个指标
                self.performance_monitor.metrics = self.performance_monitor.metrics[-5000:]
                optimizations_performed += 1
                
        self.optimization_stats['performance_optimizations'] += optimizations_performed
        
    @contextmanager
    def optimized_execution(self, operation_name: str, use_cache: bool = True, 
                          use_memory_pool: bool = True, monitor_performance: bool = True):
        """优化执行上下文管理器"""
        contexts = []
        
        try:
            # 性能监控上下文
            if monitor_performance and self.performance_monitor:
                perf_context = PerformanceContext(operation_name, self.performance_monitor)
                contexts.append(perf_context)
                perf_context.__enter__()
                
            # 内存池上下文
            if use_memory_pool and self.memory_pool:
                memory_context = MemoryPoolContext()
                contexts.append(memory_context)
                memory_context.__enter__()
                
            yield self
            
        finally:
            # 清理所有上下文
            for context in reversed(contexts):
                try:
                    context.__exit__(None, None, None)
                except Exception as e:
                    print(f"清理上下文时出错: {e}")
                    
    def cached_computation(self, key: str, computation_func: Callable, 
                          *args, **kwargs) -> Any:
        """缓存计算结果"""
        if not self.cache_system:
            return computation_func(*args, **kwargs)
            
        # 生成缓存键
        cache_key = self.cache_system._generate_key("computation", key, *args, **kwargs)
        
        # 尝试从缓存获取
        cached_result = self.cache_system.get_calculation("computation", {"key": key, "args": args, "kwargs": kwargs})
        if cached_result is not None:
            self.optimization_stats['cache_hits'] += 1
            return cached_result
            
        # 执行计算并缓存结果
        start_time = time.time()
        result = computation_func(*args, **kwargs)
        computation_time = time.time() - start_time
        
        self.cache_system.cache_calculation("computation", {"key": key, "args": args, "kwargs": kwargs}, result)
        self.optimization_stats['cache_misses'] += 1
        self.optimization_stats['total_time_saved'] += computation_time
        
        return result
        
    def allocate_optimized_array(self, shape: Tuple[int, ...], dtype=np.float64) -> Optional[np.ndarray]:
        """分配优化的数组"""
        if not self.memory_pool:
            return np.empty(shape, dtype=dtype)
            
        array = self.memory_pool.allocate_numpy_array(shape, dtype)
        if array is not None:
            self.optimization_stats['memory_allocations'] += 1
            return array
        else:
            # 回退到标准分配
            return np.empty(shape, dtype=dtype)
            
    def execute_with_adaptive_strategy(self, tasks: List[Callable]) -> List[Any]:
        """使用自适应策略执行任务"""
        # 计算任务指标
        task_metrics = TaskMetrics(
            num_symbols=len(tasks),
            data_length=1000,  # 假设数据长度
            num_strategies=1,
            complexity_score=len(tasks) * 100,
            estimated_memory_mb=len(tasks) * 10
        )
        
        # 选择执行策略
        strategy = self.adaptive_strategy.choose_execution_strategy(task_metrics)
        
        # 执行任务
        start_time = time.time()
        results = []
        
        if strategy == ExecutionStrategy.SEQUENTIAL:
            for task in tasks:
                results.append(task())
        elif strategy == ExecutionStrategy.THREADED:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(task) for task in tasks]
                results = [future.result() for future in futures]
        else:  # ASYNC or PARALLEL - 使用线程池而不是进程池避免序列化问题
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(task) for task in tasks]
                results = [future.result() for future in futures]
        
        execution_time = time.time() - start_time
        
        # 记录性能
        self.adaptive_strategy.record_performance(strategy, execution_time, task_metrics)
        
        return results
                
    def _execute_sequential(self, tasks: List[Any]) -> List[Any]:
        """顺序执行任务"""
        results = []
        for task in tasks:
            if callable(task):
                results.append(task())
            else:
                results.append(task)
        return results
        
    def _execute_threaded(self, tasks: List[Any]) -> List[Any]:
        """多线程执行任务"""
        import concurrent.futures
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for task in tasks:
                if callable(task):
                    futures.append(executor.submit(task))
                else:
                    futures.append(executor.submit(lambda x=task: x))
                    
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        return results
        
    def _execute_async(self, tasks: List[Any]) -> List[Any]:
        """异步执行任务"""
        # 简化的异步执行，实际应用中可以使用asyncio
        return self._execute_threaded(tasks)
        
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        report = {
            "优化统计": self.optimization_stats.copy(),
            "配置信息": {
                "缓存启用": self.config.enable_caching,
                "内存池启用": self.config.enable_memory_pool,
                "性能监控启用": self.config.enable_performance_monitoring,
                "自动优化启用": self.config.auto_optimization
            }
        }
        
        # 缓存统计
        if self.cache_system:
            cache_stats = self.cache_system.get_comprehensive_stats()
            report["缓存统计"] = cache_stats
            
        # 内存池统计
        if self.memory_pool:
            memory_stats = self.memory_pool.get_memory_usage()
            report["内存池统计"] = memory_stats
            
        # 性能统计
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_performance_summary()
            report["性能统计"] = perf_stats
            
        return report
        
    def benchmark_optimization(self, test_data_size: int = 10000) -> Dict[str, Any]:
        """优化效果基准测试"""
        results = {}
        
        # 测试1: 缓存效果
        if self.cache_system:
            # 第一次计算（无缓存）
            start_time = time.time()
            test_data = np.random.random((test_data_size, 100))
            result1 = np.sum(test_data ** 2)
            no_cache_time = time.time() - start_time
            
            # 缓存计算结果
            cache_key = f"benchmark_test_{test_data_size}"
            
            # 第二次计算（有缓存）
            start_time = time.time()
            result2 = self.cached_computation(
                cache_key, 
                lambda: np.sum(test_data ** 2)
            )
            cache_time = time.time() - start_time
            
            results["缓存测试"] = {
                "无缓存耗时": f"{no_cache_time:.4f}s",
                "缓存耗时": f"{cache_time:.4f}s",
                "加速比": f"{no_cache_time / max(cache_time, 0.0001):.2f}x"
            }
            
        # 测试2: 内存池效果
        if self.memory_pool:
            # 传统分配
            start_time = time.time()
            traditional_arrays = []
            for _ in range(100):
                arr = np.random.random((1000, 100))
                traditional_arrays.append(arr)
            traditional_time = time.time() - start_time
            
            # 内存池分配
            start_time = time.time()
            pool_arrays = []
            for _ in range(100):
                arr = self.allocate_optimized_array((1000, 100))
                if arr is not None:
                    arr[:] = np.random.random((1000, 100))
                    pool_arrays.append(arr)
            pool_time = time.time() - start_time
            
            results["内存池测试"] = {
                "传统分配耗时": f"{traditional_time:.4f}s",
                "内存池分配耗时": f"{pool_time:.4f}s",
                "加速比": f"{traditional_time / max(pool_time, 0.0001):.2f}x"
            }
            
        # 测试3: 自适应执行策略效果
        test_tasks = [lambda i=i: i ** 2 for i in range(1000)]
        
        # 顺序执行
        start_time = time.time()
        sequential_results = [task() for task in test_tasks]
        sequential_time = time.time() - start_time
        
        # 自适应执行
        start_time = time.time()
        adaptive_results = self.execute_with_adaptive_strategy(test_tasks)
        adaptive_time = time.time() - start_time
        
        results["自适应策略测试"] = {
            "顺序执行耗时": f"{sequential_time:.4f}s",
            "自适应执行耗时": f"{adaptive_time:.4f}s",
            "加速比": f"{sequential_time / max(adaptive_time, 0.0001):.2f}x"
        }
        
        return results
        
    def cleanup(self):
        """清理资源"""
        try:
            # 停止自动优化
            self.stop_auto_optimization()
            
            # 清理缓存
            if self.cache_system:
                self.cache_system.clear_all_caches()
                
            # 清理内存池
            if self.memory_pool:
                self.memory_pool.cleanup_all_pools()
                
            # 清理性能监控
            if self.performance_monitor:
                self.performance_monitor.stop_system_monitoring()
                
            print("优化系统资源清理完成")
        except Exception as e:
            print(f"清理资源时出错: {e}")


# 全局集成优化系统实例
_global_optimization_system = None
_optimization_lock = threading.Lock()


def get_optimization_system(config: Optional[OptimizationConfig] = None) -> IntegratedOptimizationSystem:
    """获取全局集成优化系统实例"""
    global _global_optimization_system
    if _global_optimization_system is None:
        with _optimization_lock:
            if _global_optimization_system is None:
                _global_optimization_system = IntegratedOptimizationSystem(config)
    return _global_optimization_system


def optimized_backtest_execution(backtest_func: Callable, *args, **kwargs):
    """优化的回测执行装饰器"""
    def wrapper(*args, **kwargs):
        optimization_system = get_optimization_system()
        
        with optimization_system.optimized_execution("backtest_execution"):
            return backtest_func(*args, **kwargs)
            
    return wrapper


if __name__ == "__main__":
    # 测试集成优化系统
    print("集成优化系统测试:")
    
    # 创建优化系统
    config = OptimizationConfig(
        enable_caching=True,
        enable_memory_pool=True,
        enable_performance_monitoring=True,
        auto_optimization=False  # 测试时关闭自动优化
    )
    
    optimization_system = IntegratedOptimizationSystem(config)
    
    # 测试1: 优化执行上下文
    print("\n1. 优化执行上下文测试:")
    with optimization_system.optimized_execution("test_operation"):
        # 模拟一些计算
        data = np.random.random((5000, 1000))
        result = np.sum(data ** 2)
        print(f"计算结果: {result:.2f}")
    
    # 测试2: 缓存计算
    print("\n2. 缓存计算测试:")
    
    def expensive_computation(n):
        """模拟昂贵的计算"""
        time.sleep(0.1)  # 模拟计算时间
        return sum(i ** 2 for i in range(n))
    
    # 第一次计算
    start_time = time.time()
    result1 = optimization_system.cached_computation("test_calc", expensive_computation, 10000)
    time1 = time.time() - start_time
    
    # 第二次计算（应该从缓存获取）
    start_time = time.time()
    result2 = optimization_system.cached_computation("test_calc", expensive_computation, 10000)
    time2 = time.time() - start_time
    
    print(f"第一次计算: {result1}, 耗时: {time1:.4f}s")
    print(f"第二次计算: {result2}, 耗时: {time2:.4f}s")
    print(f"加速比: {time1 / max(time2, 0.0001):.2f}x")
    
    # 测试3: 优化数组分配
    print("\n3. 优化数组分配测试:")
    arrays = []
    start_time = time.time()
    for i in range(10):
        arr = optimization_system.allocate_optimized_array((1000, 500))
        if arr is not None:
            arr.fill(i)
            arrays.append(arr)
    allocation_time = time.time() - start_time
    print(f"分配 {len(arrays)} 个数组，耗时: {allocation_time:.4f}s")
    
    # 测试4: 自适应策略执行
    print("\n4. 自适应策略执行测试:")
    test_tasks = [lambda i=i: i ** 2 + np.random.random() for i in range(100)]
    
    start_time = time.time()
    results = optimization_system.execute_with_adaptive_strategy(test_tasks)
    execution_time = time.time() - start_time
    
    print(f"执行 {len(test_tasks)} 个任务，耗时: {execution_time:.4f}s")
    print(f"结果样本: {results[:5] if results else '无结果'}")
    
    # 测试5: 基准测试
    print("\n5. 优化效果基准测试:")
    benchmark_results = optimization_system.benchmark_optimization(5000)
    
    for test_name, test_results in benchmark_results.items():
        print(f"\n{test_name}:")
        for key, value in test_results.items():
            print(f"  {key}: {value}")
    
    # 测试6: 优化报告
    print("\n6. 优化报告:")
    report = optimization_system.get_optimization_report()
    
    print("优化统计:")
    for key, value in report["优化统计"].items():
        print(f"  {key}: {value}")
    
    if "缓存统计" in report:
        print(f"\n缓存命中率: {report['缓存统计'].get('overall_hit_rate', 0):.2f}%")
    
    if "内存池统计" in report:
        print(f"内存池使用: {report['内存池统计'].get('pool_memory_mb', 0):.2f}MB")
    
    # 清理资源
    optimization_system.cleanup()
    print("\n集成优化系统测试完成!")