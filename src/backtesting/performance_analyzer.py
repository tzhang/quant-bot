"""
性能分析工具
用于识别性能瓶颈、分析执行时间和内存使用情况
"""

import time
import psutil
import threading
import functools
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    thread_id: int
    call_count: int = 1
    
    @property
    def memory_delta(self) -> float:
        """内存变化量"""
        return self.memory_after - self.memory_before


@dataclass
class FunctionProfile:
    """函数性能分析"""
    function_name: str
    total_calls: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    memory_usage: List[float] = field(default_factory=list)
    call_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_call(self, duration: float, memory_usage: float):
        """添加函数调用记录"""
        self.total_calls += 1
        self.total_time += duration
        self.avg_time = self.total_time / self.total_calls
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.memory_usage.append(memory_usage)
        self.call_times.append(duration)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.function_profiles: Dict[str, FunctionProfile] = defaultdict(FunctionProfile)
        self.active_timers: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.monitoring_enabled = True
        self.memory_samples: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        self.cpu_samples: List[Tuple[float, float]] = []     # (timestamp, cpu_percent)
        self._monitoring_thread = None
        self._stop_monitoring = False
        
    def start_system_monitoring(self, interval: float = 0.1):
        """启动系统监控"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring = False
            self._monitoring_thread = threading.Thread(
                target=self._monitor_system_resources,
                args=(interval,),
                daemon=True
            )
            self._monitoring_thread.start()
            
    def stop_system_monitoring(self):
        """停止系统监控"""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
            
    def _monitor_system_resources(self, interval: float):
        """监控系统资源使用情况"""
        process = psutil.Process()
        
        while not self._stop_monitoring:
            try:
                timestamp = time.time()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                with self.lock:
                    self.memory_samples.append((timestamp, memory_mb))
                    self.cpu_samples.append((timestamp, cpu_percent))
                    
                    # 保持最近1000个样本
                    if len(self.memory_samples) > 1000:
                        self.memory_samples = self.memory_samples[-1000:]
                    if len(self.cpu_samples) > 1000:
                        self.cpu_samples = self.cpu_samples[-1000:]
                        
                time.sleep(interval)
            except Exception as e:
                print(f"监控线程错误: {e}")
                break
                
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        return psutil.Process().memory_info().rss / 1024 / 1024
        
    def get_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        return psutil.Process().cpu_percent()
        
    def start_timer(self, name: str):
        """开始计时"""
        if not self.monitoring_enabled:
            return
            
        with self.lock:
            self.active_timers[name] = time.time()
            
    def end_timer(self, name: str) -> Optional[PerformanceMetric]:
        """结束计时并记录性能指标"""
        if not self.monitoring_enabled or name not in self.active_timers:
            return None
            
        end_time = time.time()
        
        with self.lock:
            start_time = self.active_timers.pop(name)
            duration = end_time - start_time
            
            # 获取内存和CPU信息
            memory_after = self.get_memory_usage()
            cpu_percent = self.get_cpu_usage()
            
            metric = PerformanceMetric(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before=0,  # 简化处理
                memory_after=memory_after,
                memory_peak=memory_after,
                cpu_percent=cpu_percent,
                thread_id=threading.get_ident()
            )
            
            self.metrics.append(metric)
            return metric
            
    def profile_function(self, func_name: str, duration: float, memory_usage: float):
        """记录函数性能分析"""
        with self.lock:
            if func_name not in self.function_profiles:
                self.function_profiles[func_name] = FunctionProfile(function_name=func_name)
            
            self.function_profiles[func_name].add_call(duration, memory_usage)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self.lock:
            if not self.metrics:
                return {"error": "没有性能数据"}
                
            total_time = sum(m.duration for m in self.metrics)
            avg_memory = np.mean([m.memory_after for m in self.metrics])
            peak_memory = max(m.memory_after for m in self.metrics)
            avg_cpu = np.mean([m.cpu_percent for m in self.metrics])
            
            # 最慢的操作
            slowest_operations = sorted(self.metrics, key=lambda x: x.duration, reverse=True)[:5]
            
            # 内存使用最多的操作
            memory_intensive = sorted(self.metrics, key=lambda x: x.memory_after, reverse=True)[:5]
            
            return {
                "总执行时间": f"{total_time:.4f}s",
                "平均内存使用": f"{avg_memory:.2f}MB",
                "峰值内存使用": f"{peak_memory:.2f}MB",
                "平均CPU使用率": f"{avg_cpu:.2f}%",
                "总操作数": len(self.metrics),
                "最慢操作": [
                    {"名称": op.name, "耗时": f"{op.duration:.4f}s", "内存": f"{op.memory_after:.2f}MB"}
                    for op in slowest_operations
                ],
                "内存密集操作": [
                    {"名称": op.name, "内存": f"{op.memory_after:.2f}MB", "耗时": f"{op.duration:.4f}s"}
                    for op in memory_intensive
                ]
            }
            
    def get_function_profiles(self) -> Dict[str, Dict[str, Any]]:
        """获取函数性能分析"""
        with self.lock:
            profiles = {}
            for name, profile in self.function_profiles.items():
                profiles[name] = {
                    "总调用次数": profile.total_calls,
                    "总耗时": f"{profile.total_time:.4f}s",
                    "平均耗时": f"{profile.avg_time:.4f}s",
                    "最小耗时": f"{profile.min_time:.4f}s",
                    "最大耗时": f"{profile.max_time:.4f}s",
                    "平均内存": f"{np.mean(profile.memory_usage) if profile.memory_usage else 0:.2f}MB",
                    "调用频率": f"{profile.total_calls / max(1, profile.total_time):.2f} calls/s"
                }
            return profiles
            
    def generate_performance_report(self, output_file: Optional[str] = None) -> str:
        """生成性能报告"""
        report = []
        report.append("=" * 60)
        report.append("性能分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 性能摘要
        summary = self.get_performance_summary()
        if "error" not in summary:
            report.append("性能摘要:")
            report.append("-" * 30)
            for key, value in summary.items():
                if key not in ["最慢操作", "内存密集操作"]:
                    report.append(f"{key}: {value}")
            report.append("")
            
            # 最慢操作
            report.append("最慢操作 (Top 5):")
            report.append("-" * 30)
            for i, op in enumerate(summary["最慢操作"], 1):
                report.append(f"{i}. {op['名称']}: {op['耗时']} (内存: {op['内存']})")
            report.append("")
            
            # 内存密集操作
            report.append("内存密集操作 (Top 5):")
            report.append("-" * 30)
            for i, op in enumerate(summary["内存密集操作"], 1):
                report.append(f"{i}. {op['名称']}: {op['内存']} (耗时: {op['耗时']})")
            report.append("")
        
        # 函数性能分析
        profiles = self.get_function_profiles()
        if profiles:
            report.append("函数性能分析:")
            report.append("-" * 30)
            for func_name, profile in profiles.items():
                report.append(f"\n函数: {func_name}")
                for key, value in profile.items():
                    report.append(f"  {key}: {value}")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
                
        return report_text
        
    def plot_performance_charts(self, output_dir: str = "performance_charts"):
        """生成性能图表"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 内存使用趋势图
        if self.memory_samples:
            plt.figure(figsize=(12, 6))
            timestamps, memory_values = zip(*self.memory_samples)
            start_time = timestamps[0]
            relative_times = [(t - start_time) for t in timestamps]
            
            plt.subplot(2, 1, 1)
            plt.plot(relative_times, memory_values, 'b-', linewidth=1)
            plt.title('内存使用趋势')
            plt.xlabel('时间 (秒)')
            plt.ylabel('内存使用 (MB)')
            plt.grid(True, alpha=0.3)
            
            # CPU使用趋势图
            if self.cpu_samples:
                cpu_timestamps, cpu_values = zip(*self.cpu_samples)
                cpu_relative_times = [(t - start_time) for t in cpu_timestamps]
                
                plt.subplot(2, 1, 2)
                plt.plot(cpu_relative_times, cpu_values, 'r-', linewidth=1)
                plt.title('CPU使用趋势')
                plt.xlabel('时间 (秒)')
                plt.ylabel('CPU使用率 (%)')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'system_resources.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # 操作耗时分布图
        if self.metrics:
            plt.figure(figsize=(10, 6))
            durations = [m.duration for m in self.metrics]
            plt.hist(durations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('操作耗时分布')
            plt.xlabel('耗时 (秒)')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'duration_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
    def clear_metrics(self):
        """清空性能指标"""
        with self.lock:
            self.metrics.clear()
            self.function_profiles.clear()
            self.active_timers.clear()
            self.memory_samples.clear()
            self.cpu_samples.clear()


# 全局性能监控器
_global_performance_monitor = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        with _monitor_lock:
            if _global_performance_monitor is None:
                _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


def performance_timer(name: Optional[str] = None):
    """性能计时装饰器"""
    def decorator(func: Callable) -> Callable:
        timer_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            
            start_time = time.time()
            memory_before = monitor.get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                memory_after = monitor.get_memory_usage()
                
                # 记录函数性能
                monitor.profile_function(timer_name, duration, memory_after)
                
        return wrapper
    return decorator


class PerformanceContext:
    """性能分析上下文管理器"""
    
    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None):
        self.name = name
        self.monitor = monitor or get_performance_monitor()
        self.start_time = None
        self.memory_before = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.memory_before = self.monitor.get_memory_usage()
        self.monitor.start_timer(self.name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        metric = self.monitor.end_timer(self.name)
        if metric and self.memory_before is not None:
            metric.memory_before = self.memory_before


class ProfilerContext:
    """代码分析器上下文管理器"""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.profiler = cProfile.Profile()
        
    def __enter__(self):
        self.profiler.enable()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        
        # 生成分析报告
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        profile_output = s.getvalue()
        
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(profile_output)
        else:
            print("代码分析结果:")
            print(profile_output)


if __name__ == "__main__":
    # 测试性能分析工具
    print("性能分析工具测试:")
    
    # 创建性能监控器
    monitor = get_performance_monitor()
    monitor.start_system_monitoring()
    
    # 测试1: 基本性能计时
    print("\n1. 基本性能计时测试:")
    
    @performance_timer("test_function")
    def test_function(n: int):
        """测试函数"""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    # 执行多次测试
    for i in range(5):
        with PerformanceContext(f"test_iteration_{i}"):
            result = test_function(100000)
            time.sleep(0.1)  # 模拟一些处理时间
    
    # 测试2: 内存密集操作
    print("\n2. 内存密集操作测试:")
    with PerformanceContext("memory_intensive_operation"):
        large_array = np.random.random((10000, 1000))
        processed_array = np.dot(large_array, large_array.T)
        del large_array, processed_array
    
    # 测试3: CPU密集操作
    print("\n3. CPU密集操作测试:")
    with PerformanceContext("cpu_intensive_operation"):
        # 计算斐波那契数列
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        result = fibonacci(30)
    
    # 等待一段时间收集系统监控数据
    time.sleep(1)
    
    # 停止系统监控
    monitor.stop_system_monitoring()
    
    # 测试4: 性能摘要
    print("\n4. 性能摘要:")
    summary = monitor.get_performance_summary()
    for key, value in summary.items():
        if key not in ["最慢操作", "内存密集操作"]:
            print(f"{key}: {value}")
    
    print("\n最慢操作:")
    for i, op in enumerate(summary["最慢操作"], 1):
        print(f"{i}. {op['名称']}: {op['耗时']}")
    
    # 测试5: 函数性能分析
    print("\n5. 函数性能分析:")
    profiles = monitor.get_function_profiles()
    for func_name, profile in profiles.items():
        print(f"\n函数: {func_name}")
        for key, value in profile.items():
            print(f"  {key}: {value}")
    
    # 测试6: 生成性能报告
    print("\n6. 生成性能报告:")
    report = monitor.generate_performance_report("performance_report.txt")
    print("性能报告已保存到 performance_report.txt")
    
    # 测试7: 生成性能图表
    print("\n7. 生成性能图表:")
    try:
        monitor.plot_performance_charts()
        print("性能图表已保存到 performance_charts/ 目录")
    except Exception as e:
        print(f"生成图表时出错: {e}")
    
    # 测试8: 代码分析器
    print("\n8. 代码分析器测试:")
    with ProfilerContext("detailed_profile.txt"):
        # 执行一些复杂操作
        data = np.random.random((1000, 1000))
        result = np.linalg.eigvals(data)
    
    print("详细代码分析已保存到 detailed_profile.txt")
    
    print("\n性能分析工具测试完成!")