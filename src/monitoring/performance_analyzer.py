"""
性能分析器
提供详细的性能分析、瓶颈识别和优化建议
"""

import time
import threading
import functools
import cProfile
import pstats
import io
import tracemalloc
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import gc
import sys
import inspect

from ..utils.logger import get_logger, PerformanceLogger


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    function_name: str
    module_name: str
    call_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    memory_usage: int
    cpu_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class MemorySnapshot:
    """内存快照数据类"""
    timestamp: str
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    top_traces: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class FunctionProfiler:
    """函数性能分析器"""
    
    def __init__(self):
        """初始化函数性能分析器"""
        self.logger = get_logger('performance.profiler')
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'times': deque(maxlen=1000),
            'memory_usage': 0,
            'cpu_usage': 0.0
        })
        self.lock = threading.Lock()
    
    def profile(self, func: Callable = None, *, 
                track_memory: bool = True, 
                track_cpu: bool = True):
        """
        性能分析装饰器
        
        Args:
            func: 被装饰的函数
            track_memory: 是否跟踪内存使用
            track_cpu: 是否跟踪CPU使用
            
        Returns:
            装饰后的函数
        """
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return self._profile_function(
                    f, args, kwargs, 
                    track_memory=track_memory,
                    track_cpu=track_cpu
                )
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _profile_function(self, func: Callable, args: tuple, kwargs: dict,
                         track_memory: bool = True, track_cpu: bool = True):
        """
        执行函数性能分析
        
        Args:
            func: 函数对象
            args: 位置参数
            kwargs: 关键字参数
            track_memory: 是否跟踪内存
            track_cpu: 是否跟踪CPU
            
        Returns:
            函数执行结果
        """
        func_name = f"{func.__module__}.{func.__name__}"
        
        # 记录开始状态
        start_time = time.perf_counter()
        start_memory = 0
        start_cpu = 0
        
        if track_memory:
            process = psutil.Process()
            start_memory = process.memory_info().rss
        
        if track_cpu:
            start_cpu = psutil.cpu_percent()
        
        try:
            # 执行函数
            result = func(*args, **kwargs)
            
            # 记录结束状态
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            end_memory = 0
            end_cpu = 0
            
            if track_memory:
                end_memory = process.memory_info().rss
            
            if track_cpu:
                end_cpu = psutil.cpu_percent()
            
            # 更新指标
            with self.lock:
                metrics = self.metrics[func_name]
                metrics['call_count'] += 1
                metrics['total_time'] += execution_time
                metrics['times'].append(execution_time)
                
                if track_memory:
                    metrics['memory_usage'] = max(metrics['memory_usage'], end_memory - start_memory)
                
                if track_cpu:
                    metrics['cpu_usage'] = max(metrics['cpu_usage'], end_cpu - start_cpu)
            
            # 记录性能日志
            if execution_time > 1.0:  # 只记录耗时超过1秒的函数
                self.logger.info(f"函数 {func_name} 执行时间: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"函数 {func_name} 执行失败: {e}")
            raise
    
    def get_metrics(self) -> List[PerformanceMetrics]:
        """
        获取性能指标
        
        Returns:
            List[PerformanceMetrics]: 性能指标列表
        """
        metrics_list = []
        
        with self.lock:
            for func_name, data in self.metrics.items():
                if data['call_count'] > 0:
                    times = list(data['times'])
                    
                    metrics = PerformanceMetrics(
                        function_name=func_name.split('.')[-1],
                        module_name='.'.join(func_name.split('.')[:-1]),
                        call_count=data['call_count'],
                        total_time=data['total_time'],
                        avg_time=data['total_time'] / data['call_count'],
                        min_time=min(times) if times else 0,
                        max_time=max(times) if times else 0,
                        memory_usage=data['memory_usage'],
                        cpu_usage=data['cpu_usage']
                    )
                    
                    metrics_list.append(metrics)
        
        # 按总执行时间排序
        return sorted(metrics_list, key=lambda x: x.total_time, reverse=True)
    
    def reset_metrics(self):
        """重置性能指标"""
        with self.lock:
            self.metrics.clear()
        self.logger.info("性能指标已重置")
    
    def get_top_functions(self, limit: int = 10) -> List[PerformanceMetrics]:
        """
        获取耗时最多的函数
        
        Args:
            limit: 返回数量限制
            
        Returns:
            List[PerformanceMetrics]: 耗时最多的函数列表
        """
        all_metrics = self.get_metrics()
        return all_metrics[:limit]


class MemoryProfiler:
    """内存性能分析器"""
    
    def __init__(self):
        """初始化内存性能分析器"""
        self.logger = get_logger('performance.memory')
        self.snapshots: List[MemorySnapshot] = []
        self.tracking = False
    
    def start_tracking(self):
        """开始内存跟踪"""
        if not self.tracking:
            tracemalloc.start()
            self.tracking = True
            self.logger.info("内存跟踪已启动")
    
    def stop_tracking(self):
        """停止内存跟踪"""
        if self.tracking:
            tracemalloc.stop()
            self.tracking = False
            self.logger.info("内存跟踪已停止")
    
    def take_snapshot(self) -> MemorySnapshot:
        """
        拍摄内存快照
        
        Returns:
            MemorySnapshot: 内存快照数据
        """
        if not self.tracking:
            self.start_tracking()
        
        # 获取系统内存信息
        memory = psutil.virtual_memory()
        
        # 获取内存跟踪信息
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # 转换为字典格式
        top_traces = []
        for stat in top_stats[:10]:  # 只保留前10个
            trace_info = {
                'filename': stat.traceback.format()[0] if stat.traceback.format() else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            }
            top_traces.append(trace_info)
        
        # 创建快照对象
        memory_snapshot = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            top_traces=top_traces
        )
        
        self.snapshots.append(memory_snapshot)
        
        # 限制快照数量
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]
        
        return memory_snapshot
    
    def compare_snapshots(self, snapshot1: MemorySnapshot, snapshot2: MemorySnapshot) -> Dict[str, Any]:
        """
        比较两个内存快照
        
        Args:
            snapshot1: 第一个快照
            snapshot2: 第二个快照
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        memory_diff = snapshot2.used_memory - snapshot1.used_memory
        percent_diff = snapshot2.memory_percent - snapshot1.memory_percent
        
        return {
            'time_diff': snapshot2.timestamp,
            'memory_diff_mb': memory_diff / 1024 / 1024,
            'percent_diff': percent_diff,
            'trend': 'increasing' if memory_diff > 0 else 'decreasing' if memory_diff < 0 else 'stable'
        }
    
    def detect_memory_leaks(self, threshold_mb: float = 100.0) -> List[Dict[str, Any]]:
        """
        检测内存泄漏
        
        Args:
            threshold_mb: 内存增长阈值（MB）
            
        Returns:
            List[Dict[str, Any]]: 可能的内存泄漏信息
        """
        if len(self.snapshots) < 2:
            return []
        
        leaks = []
        
        # 检查最近的快照
        for i in range(1, min(len(self.snapshots), 10)):
            current = self.snapshots[-1]
            previous = self.snapshots[-(i+1)]
            
            comparison = self.compare_snapshots(previous, current)
            
            if comparison['memory_diff_mb'] > threshold_mb:
                leaks.append({
                    'time_range': f"{previous.timestamp} -> {current.timestamp}",
                    'memory_increase_mb': comparison['memory_diff_mb'],
                    'severity': 'high' if comparison['memory_diff_mb'] > threshold_mb * 2 else 'medium'
                })
        
        return leaks
    
    def get_memory_trend(self, hours: int = 1) -> Dict[str, Any]:
        """
        获取内存使用趋势
        
        Args:
            hours: 分析时间范围（小时）
            
        Returns:
            Dict[str, Any]: 内存趋势信息
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_snapshots = [
            s for s in self.snapshots
            if datetime.fromisoformat(s.timestamp) > cutoff_time
        ]
        
        if len(recent_snapshots) < 2:
            return {'trend': 'insufficient_data'}
        
        # 计算趋势
        memory_values = [s.memory_percent for s in recent_snapshots]
        
        # 简单线性趋势计算
        n = len(memory_values)
        x_sum = sum(range(n))
        y_sum = sum(memory_values)
        xy_sum = sum(i * memory_values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        return {
            'trend': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable',
            'slope': slope,
            'current_usage': memory_values[-1],
            'min_usage': min(memory_values),
            'max_usage': max(memory_values),
            'avg_usage': sum(memory_values) / len(memory_values)
        }


class CodeProfiler:
    """代码性能分析器"""
    
    def __init__(self):
        """初始化代码性能分析器"""
        self.logger = get_logger('performance.code')
        self.profiler = None
        self.profiling = False
    
    def start_profiling(self):
        """开始代码分析"""
        if not self.profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.profiling = True
            self.logger.info("代码性能分析已启动")
    
    def stop_profiling(self) -> str:
        """
        停止代码分析并返回结果
        
        Returns:
            str: 分析结果报告
        """
        if self.profiling and self.profiler:
            self.profiler.disable()
            self.profiling = False
            
            # 生成报告
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(50)  # 显示前50个函数
            
            report = s.getvalue()
            self.logger.info("代码性能分析已停止")
            
            return report
        
        return "未进行性能分析"
    
    def profile_code_block(self, code_block: Callable) -> str:
        """
        分析代码块性能
        
        Args:
            code_block: 要分析的代码块
            
        Returns:
            str: 分析结果
        """
        profiler = cProfile.Profile()
        
        try:
            profiler.enable()
            result = code_block()
            profiler.disable()
            
            # 生成报告
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)
            
            return s.getvalue()
            
        except Exception as e:
            self.logger.error(f"代码块分析失败: {e}")
            return f"分析失败: {e}"


class PerformanceAnalyzer:
    """性能分析器主类"""
    
    def __init__(self):
        """初始化性能分析器"""
        self.logger = get_logger('performance')
        
        # 初始化子分析器
        self.function_profiler = FunctionProfiler()
        self.memory_profiler = MemoryProfiler()
        self.code_profiler = CodeProfiler()
        
        # 分析状态
        self.analysis_running = False
        self.analysis_thread = None
        
        # 性能基线
        self.baseline_metrics: Dict[str, Any] = {}
    
    def start_analysis(self):
        """启动性能分析"""
        if self.analysis_running:
            self.logger.warning("性能分析已在运行中")
            return
        
        self.analysis_running = True
        
        # 启动各种分析器
        self.memory_profiler.start_tracking()
        self.code_profiler.start_profiling()
        
        # 启动定期快照线程
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        self.logger.info("性能分析已启动")
    
    def stop_analysis(self) -> Dict[str, Any]:
        """
        停止性能分析并返回结果
        
        Returns:
            Dict[str, Any]: 分析结果
        """
        if not self.analysis_running:
            self.logger.warning("性能分析未在运行")
            return {}
        
        self.analysis_running = False
        
        # 等待分析线程结束
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)
        
        # 收集分析结果
        results = {
            'function_metrics': [m.to_dict() for m in self.function_profiler.get_metrics()],
            'memory_snapshots': [s.to_dict() for s in self.memory_profiler.snapshots],
            'memory_leaks': self.memory_profiler.detect_memory_leaks(),
            'memory_trend': self.memory_profiler.get_memory_trend(),
            'code_profile': self.code_profiler.stop_profiling(),
            'analysis_summary': self._generate_summary()
        }
        
        # 停止各种分析器
        self.memory_profiler.stop_tracking()
        
        self.logger.info("性能分析已停止")
        
        return results
    
    def _analysis_loop(self):
        """分析循环"""
        while self.analysis_running:
            try:
                # 定期拍摄内存快照
                self.memory_profiler.take_snapshot()
                
                # 检查内存泄漏
                leaks = self.memory_profiler.detect_memory_leaks()
                if leaks:
                    self.logger.warning(f"检测到 {len(leaks)} 个可能的内存泄漏")
                
                # 等待下次分析
                time.sleep(60)  # 每分钟分析一次
                
            except Exception as e:
                self.logger.error(f"性能分析循环出错: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        生成分析摘要
        
        Returns:
            Dict[str, Any]: 分析摘要
        """
        function_metrics = self.function_profiler.get_metrics()
        memory_trend = self.memory_profiler.get_memory_trend()
        
        # 找出性能瓶颈
        bottlenecks = []
        
        # 函数性能瓶颈
        if function_metrics:
            slowest_functions = function_metrics[:5]
            for func in slowest_functions:
                if func.avg_time > 0.1:  # 平均执行时间超过100ms
                    bottlenecks.append({
                        'type': 'slow_function',
                        'function': func.function_name,
                        'avg_time': func.avg_time,
                        'total_time': func.total_time
                    })
        
        # 内存使用瓶颈
        if memory_trend.get('current_usage', 0) > 80:
            bottlenecks.append({
                'type': 'high_memory_usage',
                'current_usage': memory_trend['current_usage'],
                'trend': memory_trend['trend']
            })
        
        # 生成优化建议
        recommendations = self._generate_recommendations(bottlenecks)
        
        return {
            'total_functions_analyzed': len(function_metrics),
            'total_execution_time': sum(m.total_time for m in function_metrics),
            'memory_trend': memory_trend,
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """
        生成优化建议
        
        Args:
            bottlenecks: 性能瓶颈列表
            
        Returns:
            List[str]: 优化建议列表
        """
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'slow_function':
                recommendations.append(
                    f"优化函数 {bottleneck['function']}，当前平均执行时间 {bottleneck['avg_time']:.3f}s"
                )
            elif bottleneck['type'] == 'high_memory_usage':
                recommendations.append(
                    f"内存使用率过高 ({bottleneck['current_usage']:.1f}%)，建议检查内存泄漏"
                )
        
        # 通用建议
        if not recommendations:
            recommendations.append("系统性能良好，无明显瓶颈")
        else:
            recommendations.extend([
                "考虑使用缓存减少重复计算",
                "优化数据库查询和索引",
                "使用异步处理提高并发性能",
                "定期进行垃圾回收和内存清理"
            ])
        
        return recommendations
    
    def set_baseline(self):
        """设置性能基线"""
        self.baseline_metrics = {
            'timestamp': datetime.now().isoformat(),
            'function_metrics': [m.to_dict() for m in self.function_profiler.get_metrics()],
            'memory_snapshot': self.memory_profiler.take_snapshot().to_dict()
        }
        
        self.logger.info("性能基线已设置")
    
    def compare_with_baseline(self) -> Dict[str, Any]:
        """
        与基线进行比较
        
        Returns:
            Dict[str, Any]: 比较结果
        """
        if not self.baseline_metrics:
            return {'error': '未设置性能基线'}
        
        current_metrics = self.function_profiler.get_metrics()
        current_memory = self.memory_profiler.take_snapshot()
        
        # 比较函数性能
        function_comparison = {}
        baseline_functions = {
            m['function_name']: m for m in self.baseline_metrics['function_metrics']
        }
        
        for current_func in current_metrics:
            func_name = current_func.function_name
            if func_name in baseline_functions:
                baseline_func = baseline_functions[func_name]
                
                time_diff = current_func.avg_time - baseline_func['avg_time']
                function_comparison[func_name] = {
                    'time_diff': time_diff,
                    'performance_change': 'improved' if time_diff < 0 else 'degraded' if time_diff > 0 else 'unchanged'
                }
        
        # 比较内存使用
        baseline_memory = self.baseline_metrics['memory_snapshot']
        memory_diff = current_memory.memory_percent - baseline_memory['memory_percent']
        
        return {
            'comparison_timestamp': datetime.now().isoformat(),
            'baseline_timestamp': self.baseline_metrics['timestamp'],
            'function_comparison': function_comparison,
            'memory_comparison': {
                'memory_diff_percent': memory_diff,
                'trend': 'increased' if memory_diff > 0 else 'decreased' if memory_diff < 0 else 'unchanged'
            }
        }
    
    def get_performance_report(self) -> str:
        """
        生成性能报告
        
        Returns:
            str: 性能报告
        """
        function_metrics = self.function_profiler.get_metrics()
        memory_trend = self.memory_profiler.get_memory_trend()
        
        report = []
        report.append("=== 性能分析报告 ===")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 函数性能统计
        report.append("函数性能统计:")
        if function_metrics:
            report.append(f"  总函数数: {len(function_metrics)}")
            report.append(f"  总执行时间: {sum(m.total_time for m in function_metrics):.3f}s")
            report.append("")
            
            report.append("耗时最多的函数:")
            for i, func in enumerate(function_metrics[:5], 1):
                report.append(f"  {i}. {func.function_name}")
                report.append(f"     调用次数: {func.call_count}")
                report.append(f"     总时间: {func.total_time:.3f}s")
                report.append(f"     平均时间: {func.avg_time:.3f}s")
                report.append("")
        else:
            report.append("  无函数性能数据")
            report.append("")
        
        # 内存使用统计
        report.append("内存使用统计:")
        if memory_trend.get('trend') != 'insufficient_data':
            report.append(f"  当前使用率: {memory_trend['current_usage']:.1f}%")
            report.append(f"  使用趋势: {memory_trend['trend']}")
            report.append(f"  最小使用率: {memory_trend['min_usage']:.1f}%")
            report.append(f"  最大使用率: {memory_trend['max_usage']:.1f}%")
            report.append(f"  平均使用率: {memory_trend['avg_usage']:.1f}%")
        else:
            report.append("  内存数据不足")
        report.append("")
        
        # 内存泄漏检测
        leaks = self.memory_profiler.detect_memory_leaks()
        if leaks:
            report.append("内存泄漏检测:")
            for leak in leaks:
                report.append(f"  时间范围: {leak['time_range']}")
                report.append(f"  内存增长: {leak['memory_increase_mb']:.1f}MB")
                report.append(f"  严重程度: {leak['severity']}")
                report.append("")
        
        return "\n".join(report)


# 全局性能分析器实例
performance_analyzer = PerformanceAnalyzer()


# 便捷装饰器
def profile_performance(track_memory: bool = True, track_cpu: bool = True):
    """
    性能分析装饰器
    
    Args:
        track_memory: 是否跟踪内存使用
        track_cpu: 是否跟踪CPU使用
        
    Returns:
        装饰器函数
    """
    return performance_analyzer.function_profiler.profile(
        track_memory=track_memory,
        track_cpu=track_cpu
    )


# 使用示例
if __name__ == "__main__":
    # 启动性能分析
    performance_analyzer.start_analysis()
    
    # 使用装饰器分析函数性能
    @profile_performance()
    def test_function():
        time.sleep(0.1)
        return "test"
    
    # 执行一些操作
    for i in range(10):
        test_function()
        time.sleep(0.5)
    
    # 停止分析并获取结果
    results = performance_analyzer.stop_analysis()
    
    # 生成报告
    report = performance_analyzer.get_performance_report()
    print(report)