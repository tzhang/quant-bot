#!/usr/bin/env python3
"""
集成优化系统 - 整合智能缓存、内存池管理、性能分析和自适应执行策略
"""

import sys
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from smart_cache_system import SmartCacheSystem, get_cache_system
    from memory_pool_manager import MemoryPoolManager, get_memory_pool_manager, MemoryPoolContext
    from performance_analyzer import PerformanceMonitor, get_performance_monitor, PerformanceContext
    from adaptive_execution_strategy import AdaptiveExecutionStrategy, TaskMetrics, ExecutionStrategy
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖模块都在正确的位置")
    sys.exit(1)

@dataclass
class OptimizationConfig:
    """优化配置"""
    enable_cache: bool = True
    enable_memory_pool: bool = True
    enable_performance_monitoring: bool = True
    cache_memory_limit_mb: int = 1000
    memory_pool_limit_mb: int = 500
    performance_monitoring_interval: float = 1.0
    adaptive_strategy_enabled: bool = True

class IntegratedOptimizationSystem:
    """集成优化系统"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化各个优化组件"""
        # 智能缓存系统
        if self.config.enable_cache:
            self.cache_system = get_cache_system()
            print(f"✓ 智能缓存系统已启用 (内存限制: {self.config.cache_memory_limit_mb}MB)")
        else:
            self.cache_system = None
            
        # 内存池管理
        if self.config.enable_memory_pool:
            self.memory_pool = get_memory_pool_manager()
            print(f"✓ 内存池管理已启用 (内存限制: {self.config.memory_pool_limit_mb}MB)")
        else:
            self.memory_pool = None
            
        # 性能监控
        if self.config.enable_performance_monitoring:
            self.performance_monitor = get_performance_monitor()
            print(f"✓ 性能监控已启用 (监控间隔: {self.config.performance_monitoring_interval}s)")
        else:
            self.performance_monitor = None
            
        # 自适应执行策略
        if self.config.adaptive_strategy_enabled:
            self.adaptive_strategy = AdaptiveExecutionStrategy()
            print("✓ 自适应执行策略已启用")
        else:
            self.adaptive_strategy = None
    
    def optimized_execution(self, name: str):
        """优化执行上下文管理器"""
        class OptimizedContext:
            def __init__(self, system, context_name):
                self.system = system
                self.context_name = context_name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                if self.system.performance_monitor:
                    self.system.performance_monitor.start_timer(self.context_name)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.system.performance_monitor:
                    self.system.performance_monitor.end_timer(self.context_name)
                
                execution_time = time.time() - self.start_time
                print(f"✓ {self.context_name} 执行完成，耗时: {execution_time:.3f}s")
        
        return OptimizedContext(self, name)
    
    def cached_computation(self, cache_key: str, computation_func: Callable, *args, **kwargs) -> Any:
        """缓存计算结果"""
        if not self.cache_system:
            return computation_func(*args, **kwargs)
        
        # 生成缓存键
        full_key = f"{cache_key}_{hash(str(args) + str(kwargs))}"
        
        # 尝试从缓存获取结果
        cached_result = self.cache_system.get_calculation(cache_key, {"args": args, "kwargs": kwargs})
        if cached_result is not None:
            print(f"✓ 缓存命中: {cache_key}")
            return cached_result
        
        # 执行计算
        start_time = time.time()
        result = computation_func(*args, **kwargs)
        computation_time = time.time() - start_time
        
        # 缓存结果
        self.cache_system.cache_calculation(cache_key, {"args": args, "kwargs": kwargs}, result)
        print(f"✓ 计算完成并缓存: {cache_key} (耗时: {computation_time:.3f}s)")
        
        return result
    
    def optimized_array_allocation(self, shape: tuple, dtype=np.float32) -> np.ndarray:
        """优化的数组分配"""
        if not self.memory_pool:
            return np.zeros(shape, dtype=dtype)
        
        try:
            # 使用内存池分配
            with MemoryPoolContext() as pool_ctx:
                array = pool_ctx.allocate_numpy_array(shape, dtype=dtype)
                print(f"✓ 使用内存池分配数组: {shape}")
                return array
        except Exception as e:
            print(f"⚠ 内存池分配失败，使用标准分配: {e}")
            return np.zeros(shape, dtype=dtype)
    
    def execute_with_adaptive_strategy(self, tasks: List[Callable]) -> List[Any]:
        """使用自适应策略执行任务"""
        if not self.adaptive_strategy or not tasks:
            # 顺序执行
            return [task() for task in tasks]
        
        # 计算任务复杂度
        task_metrics = TaskMetrics(
            num_symbols=len(tasks),
            data_length=1000,  # 假设数据长度
            num_strategies=1,
            complexity_score=len(tasks) * 0.1,
            estimated_memory_mb=len(tasks) * 10
        )
        
        # 选择执行策略
        strategy = self.adaptive_strategy.choose_execution_strategy(task_metrics)
        print(f"✓ 选择执行策略: {strategy.value}")
        
        start_time = time.time()
        
        if strategy == ExecutionStrategy.SEQUENTIAL:
            # 顺序执行
            results = []
            for i, task in enumerate(tasks):
                result = task()
                results.append(result)
                print(f"  任务 {i+1}/{len(tasks)} 完成")
            
        elif strategy in [ExecutionStrategy.THREADED, ExecutionStrategy.ASYNC, ExecutionStrategy.PARALLEL]:
            # 使用线程池执行
            max_workers = min(len(tasks), 4)  # 限制最大线程数
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(task) for task in tasks]
                results = [future.result() for future in futures]
                print(f"  使用 {max_workers} 个线程并行执行")
        
        else:
            # 默认顺序执行
            results = [task() for task in tasks]
        
        execution_time = time.time() - start_time
        print(f"✓ 任务执行完成，总耗时: {execution_time:.3f}s")
        
        return results
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """生成优化报告"""
        report = {
            "timestamp": time.time(),
            "config": self.config.__dict__,
            "components": {}
        }
        
        # 缓存系统统计
        if self.cache_system:
            try:
                cache_stats = self.cache_system.get_comprehensive_stats()
                report["components"]["cache"] = cache_stats
            except Exception as e:
                report["components"]["cache"] = {"error": str(e)}
        
        # 内存池统计
        if self.memory_pool:
            try:
                pool_stats = self.memory_pool.get_comprehensive_stats()
                report["components"]["memory_pool"] = pool_stats
            except Exception as e:
                report["components"]["memory_pool"] = {"error": str(e)}
        
        # 性能监控统计
        if self.performance_monitor:
            try:
                perf_stats = self.performance_monitor.get_summary()
                report["components"]["performance"] = perf_stats
            except Exception as e:
                report["components"]["performance"] = {"error": str(e)}
        
        return report
    
    def benchmark_optimization_effects(self) -> Dict[str, float]:
        """基准测试优化效果"""
        print("\n=== 优化效果基准测试 ===")
        
        # 测试数据
        test_data_size = (1000, 100)
        test_iterations = 100
        
        results = {}
        
        # 1. 内存分配性能测试
        print("1. 内存分配性能测试...")
        
        # 标准分配
        start_time = time.time()
        for _ in range(test_iterations):
            arr = np.zeros(test_data_size, dtype=np.float32)
            del arr
        standard_time = time.time() - start_time
        
        # 优化分配
        start_time = time.time()
        for _ in range(test_iterations):
            try:
                arr = self.optimized_array_allocation(test_data_size)
                del arr
            except:
                arr = np.zeros(test_data_size, dtype=np.float32)
                del arr
        optimized_time = time.time() - start_time
        
        if optimized_time > 0:
            speedup = standard_time / optimized_time
            results["memory_allocation_speedup"] = speedup
            print(f"  内存分配加速比: {speedup:.2f}x")
        
        # 2. 缓存性能测试
        print("2. 缓存性能测试...")
        
        def expensive_computation(x):
            return np.sum(np.sin(np.arange(x * 1000)))
        
        # 无缓存
        start_time = time.time()
        for i in range(10):
            result = expensive_computation(100)
        no_cache_time = time.time() - start_time
        
        # 有缓存
        start_time = time.time()
        for i in range(10):
            result = self.cached_computation("expensive_comp", expensive_computation, 100)
        cached_time = time.time() - start_time
        
        if cached_time > 0:
            cache_speedup = no_cache_time / cached_time
            results["cache_speedup"] = cache_speedup
            print(f"  缓存加速比: {cache_speedup:.2f}x")
        
        return results
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.cache_system:
                self.cache_system.clear_all()
                print("✓ 缓存系统已清理")
        except Exception as e:
            print(f"⚠ 清理缓存时出错: {e}")
        
        try:
            if self.memory_pool:
                # 内存池会自动清理
                print("✓ 内存池已清理")
        except Exception as e:
            print(f"⚠ 清理内存池时出错: {e}")

# 全局实例
_global_optimization_system = None
_system_lock = threading.Lock()

def get_optimization_system(config: OptimizationConfig = None) -> IntegratedOptimizationSystem:
    """获取全局优化系统实例"""
    global _global_optimization_system
    
    with _system_lock:
        if _global_optimization_system is None:
            _global_optimization_system = IntegratedOptimizationSystem(config)
        return _global_optimization_system

def optimized_backtest_execution(func):
    """优化回测执行装饰器"""
    def wrapper(*args, **kwargs):
        system = get_optimization_system()
        with system.optimized_execution(f"backtest_{func.__name__}"):
            return func(*args, **kwargs)
    return wrapper

if __name__ == "__main__":
    print("=== 集成优化系统测试 ===\n")
    
    # 创建优化配置
    config = OptimizationConfig(
        enable_cache=True,
        enable_memory_pool=True,
        enable_performance_monitoring=True,
        cache_memory_limit_mb=500,
        memory_pool_limit_mb=300
    )
    
    # 获取优化系统
    optimization_system = get_optimization_system(config)
    
    try:
        # 1. 测试优化执行上下文
        print("1. 测试优化执行上下文...")
        with optimization_system.optimized_execution("test_context"):
            time.sleep(0.1)  # 模拟工作
            print("  执行上下文测试完成")
        
        # 2. 测试缓存计算
        print("\n2. 测试缓存计算...")
        def test_computation(n):
            return sum(i**2 for i in range(n))
        
        # 第一次计算（无缓存）
        result1 = optimization_system.cached_computation("test_calc", test_computation, 1000)
        # 第二次计算（有缓存）
        result2 = optimization_system.cached_computation("test_calc", test_computation, 1000)
        print(f"  计算结果一致: {result1 == result2}")
        
        # 3. 测试优化数组分配
        print("\n3. 测试优化数组分配...")
        test_array = optimization_system.optimized_array_allocation((500, 200), dtype=np.float32)
        print(f"  分配数组形状: {test_array.shape}")
        
        # 4. 测试自适应策略执行
        print("\n4. 测试自适应策略执行...")
        def create_test_task(task_id):
            def task():
                time.sleep(0.01)  # 模拟工作
                return f"Task {task_id} completed"
            return task
        
        tasks = [create_test_task(i) for i in range(5)]
        results = optimization_system.execute_with_adaptive_strategy(tasks)
        print(f"  执行了 {len(results)} 个任务")
        
        # 5. 基准测试
        print("\n5. 基准测试...")
        benchmark_results = optimization_system.benchmark_optimization_effects()
        for metric, value in benchmark_results.items():
            print(f"  {metric}: {value:.2f}")
        
        # 6. 生成优化报告
        print("\n6. 生成优化报告...")
        report = optimization_system.generate_optimization_report()
        print(f"  报告生成时间: {time.ctime(report['timestamp'])}")
        print(f"  启用的组件数: {len([k for k, v in report['components'].items() if 'error' not in v])}")
        
        print("\n=== 集成优化系统测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        print("\n清理资源...")
        optimization_system.cleanup()
        print("✓ 资源清理完成")