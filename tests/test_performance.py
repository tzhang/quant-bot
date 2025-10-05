"""
性能分析器测试用例
测试性能分析、内存监控和代码分析功能
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.monitoring.performance_analyzer import (
    PerformanceAnalyzer, FunctionProfiler, MemoryProfiler, CodeProfiler,
    PerformanceMetrics, MemorySnapshot, profile_performance
)


class TestFunctionProfiler(unittest.TestCase):
    """函数性能分析器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.profiler = FunctionProfiler()
    
    def test_profile_decorator(self):
        """测试性能分析装饰器"""
        @self.profiler.profile
        def test_function():
            time.sleep(0.1)
            return "test"
        
        # 执行函数
        result = test_function()
        
        # 验证结果
        self.assertEqual(result, "test")
        
        # 验证指标收集
        metrics = self.profiler.get_metrics()
        self.assertEqual(len(metrics), 1)
        
        metric = metrics[0]
        self.assertEqual(metric.function_name, "test_function")
        self.assertEqual(metric.call_count, 1)
        self.assertGreater(metric.total_time, 0.09)  # 至少0.09秒
    
    def test_multiple_calls(self):
        """测试多次调用统计"""
        @self.profiler.profile
        def test_function():
            time.sleep(0.01)
            return "test"
        
        # 多次调用
        for _ in range(5):
            test_function()
        
        # 验证统计
        metrics = self.profiler.get_metrics()
        self.assertEqual(len(metrics), 1)
        
        metric = metrics[0]
        self.assertEqual(metric.call_count, 5)
        self.assertGreater(metric.total_time, 0.04)  # 至少0.04秒
        self.assertAlmostEqual(metric.avg_time, metric.total_time / 5, places=3)
    
    def test_exception_handling(self):
        """测试异常处理"""
        @self.profiler.profile
        def failing_function():
            raise ValueError("测试异常")
        
        # 验证异常被正确抛出
        with self.assertRaises(ValueError):
            failing_function()
        
        # 验证指标仍然被收集
        metrics = self.profiler.get_metrics()
        self.assertEqual(len(metrics), 1)
        
        metric = metrics[0]
        self.assertEqual(metric.function_name, "failing_function")
        self.assertEqual(metric.call_count, 1)
    
    def test_reset_metrics(self):
        """测试重置指标"""
        @self.profiler.profile
        def test_function():
            return "test"
        
        # 执行函数
        test_function()
        
        # 验证有指标
        metrics = self.profiler.get_metrics()
        self.assertEqual(len(metrics), 1)
        
        # 重置指标
        self.profiler.reset_metrics()
        
        # 验证指标被清空
        metrics = self.profiler.get_metrics()
        self.assertEqual(len(metrics), 0)
    
    def test_top_functions(self):
        """测试获取耗时最多的函数"""
        @self.profiler.profile
        def fast_function():
            time.sleep(0.01)
        
        @self.profiler.profile
        def slow_function():
            time.sleep(0.05)
        
        # 执行函数
        fast_function()
        slow_function()
        
        # 获取耗时最多的函数
        top_functions = self.profiler.get_top_functions(limit=2)
        
        # 验证排序正确
        self.assertEqual(len(top_functions), 2)
        self.assertEqual(top_functions[0].function_name, "slow_function")
        self.assertEqual(top_functions[1].function_name, "fast_function")


class TestMemoryProfiler(unittest.TestCase):
    """内存性能分析器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.profiler = MemoryProfiler()
    
    def test_start_stop_tracking(self):
        """测试启动和停止内存跟踪"""
        # 初始状态
        self.assertFalse(self.profiler.tracking)
        
        # 启动跟踪
        self.profiler.start_tracking()
        self.assertTrue(self.profiler.tracking)
        
        # 停止跟踪
        self.profiler.stop_tracking()
        self.assertFalse(self.profiler.tracking)
    
    def test_take_snapshot(self):
        """测试拍摄内存快照"""
        # 拍摄快照
        snapshot = self.profiler.take_snapshot()
        
        # 验证快照数据
        self.assertIsInstance(snapshot, MemorySnapshot)
        self.assertIsNotNone(snapshot.timestamp)
        self.assertGreater(snapshot.total_memory, 0)
        self.assertGreater(snapshot.used_memory, 0)
        self.assertGreaterEqual(snapshot.memory_percent, 0)
        self.assertLessEqual(snapshot.memory_percent, 100)
        self.assertIsInstance(snapshot.top_traces, list)
    
    def test_compare_snapshots(self):
        """测试比较内存快照"""
        # 拍摄两个快照
        snapshot1 = self.profiler.take_snapshot()
        time.sleep(0.1)
        
        # 分配一些内存
        data = [i for i in range(10000)]
        
        snapshot2 = self.profiler.take_snapshot()
        
        # 比较快照
        comparison = self.profiler.compare_snapshots(snapshot1, snapshot2)
        
        # 验证比较结果
        self.assertIn('memory_diff_mb', comparison)
        self.assertIn('percent_diff', comparison)
        self.assertIn('trend', comparison)
        self.assertIn(comparison['trend'], ['increasing', 'decreasing', 'stable'])
        
        # 清理内存
        del data
    
    @patch('src.monitoring.performance_analyzer.psutil.virtual_memory')
    def test_detect_memory_leaks(self, mock_memory):
        """测试内存泄漏检测"""
        # 模拟内存使用增长
        memory_values = [
            Mock(total=8000000000, available=4000000000, used=4000000000, percent=50.0),
            Mock(total=8000000000, available=3500000000, used=4500000000, percent=56.25),
            Mock(total=8000000000, available=3000000000, used=5000000000, percent=62.5)
        ]
        
        mock_memory.side_effect = memory_values
        
        # 拍摄多个快照
        for _ in range(3):
            self.profiler.take_snapshot()
            time.sleep(0.01)
        
        # 检测内存泄漏
        leaks = self.profiler.detect_memory_leaks(threshold_mb=100.0)
        
        # 验证检测结果
        self.assertIsInstance(leaks, list)
    
    def test_get_memory_trend(self):
        """测试获取内存趋势"""
        # 拍摄多个快照
        for _ in range(3):
            self.profiler.take_snapshot()
            time.sleep(0.01)
        
        # 获取内存趋势
        trend = self.profiler.get_memory_trend(hours=1)
        
        # 验证趋势数据
        self.assertIn('trend', trend)
        self.assertIn(trend['trend'], ['increasing', 'decreasing', 'stable', 'insufficient_data'])
        
        if trend['trend'] != 'insufficient_data':
            self.assertIn('current_usage', trend)
            self.assertIn('min_usage', trend)
            self.assertIn('max_usage', trend)
            self.assertIn('avg_usage', trend)


class TestCodeProfiler(unittest.TestCase):
    """代码性能分析器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.profiler = CodeProfiler()
    
    def test_start_stop_profiling(self):
        """测试启动和停止代码分析"""
        # 初始状态
        self.assertFalse(self.profiler.profiling)
        
        # 启动分析
        self.profiler.start_profiling()
        self.assertTrue(self.profiler.profiling)
        
        # 执行一些代码
        time.sleep(0.01)
        
        # 停止分析
        report = self.profiler.stop_profiling()
        
        # 验证结果
        self.assertFalse(self.profiler.profiling)
        self.assertIsInstance(report, str)
        self.assertIn('function calls', report)
    
    def test_profile_code_block(self):
        """测试分析代码块"""
        def test_code_block():
            # 执行一些计算
            result = sum(i * i for i in range(1000))
            return result
        
        # 分析代码块
        report = self.profiler.profile_code_block(test_code_block)
        
        # 验证报告
        self.assertIsInstance(report, str)
        self.assertIn('function calls', report)
    
    def test_profile_code_block_with_exception(self):
        """测试分析抛出异常的代码块"""
        def failing_code_block():
            raise ValueError("测试异常")
        
        # 分析代码块
        report = self.profiler.profile_code_block(failing_code_block)
        
        # 验证错误处理
        self.assertIsInstance(report, str)
        self.assertIn('分析失败', report)


class TestPerformanceAnalyzer(unittest.TestCase):
    """性能分析器主类测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.analyzer = PerformanceAnalyzer()
    
    def test_start_stop_analysis(self):
        """测试启动和停止性能分析"""
        # 初始状态
        self.assertFalse(self.analyzer.analysis_running)
        
        # 启动分析
        self.analyzer.start_analysis()
        self.assertTrue(self.analyzer.analysis_running)
        
        # 等待一段时间
        time.sleep(0.5)
        
        # 停止分析
        results = self.analyzer.stop_analysis()
        
        # 验证结果
        self.assertFalse(self.analyzer.analysis_running)
        self.assertIsInstance(results, dict)
        self.assertIn('function_metrics', results)
        self.assertIn('memory_snapshots', results)
        self.assertIn('analysis_summary', results)
    
    def test_set_baseline(self):
        """测试设置性能基线"""
        # 设置基线
        self.analyzer.set_baseline()
        
        # 验证基线数据
        self.assertIsNotNone(self.analyzer.baseline_metrics)
        self.assertIn('timestamp', self.analyzer.baseline_metrics)
        self.assertIn('function_metrics', self.analyzer.baseline_metrics)
        self.assertIn('memory_snapshot', self.analyzer.baseline_metrics)
    
    def test_compare_with_baseline(self):
        """测试与基线比较"""
        # 先设置基线
        self.analyzer.set_baseline()
        
        # 执行一些操作
        @profile_performance()
        def test_function():
            time.sleep(0.01)
        
        test_function()
        
        # 与基线比较
        comparison = self.analyzer.compare_with_baseline()
        
        # 验证比较结果
        self.assertIsInstance(comparison, dict)
        self.assertIn('comparison_timestamp', comparison)
        self.assertIn('baseline_timestamp', comparison)
        self.assertIn('function_comparison', comparison)
        self.assertIn('memory_comparison', comparison)
    
    def test_compare_with_baseline_no_baseline(self):
        """测试没有基线时的比较"""
        # 清空基线
        self.analyzer.baseline_metrics = {}
        
        # 尝试比较
        comparison = self.analyzer.compare_with_baseline()
        
        # 验证错误处理
        self.assertIn('error', comparison)
        self.assertEqual(comparison['error'], '未设置性能基线')
    
    def test_get_performance_report(self):
        """测试生成性能报告"""
        # 执行一些操作
        @profile_performance()
        def test_function():
            time.sleep(0.01)
        
        test_function()
        
        # 生成报告
        report = self.analyzer.get_performance_report()
        
        # 验证报告
        self.assertIsInstance(report, str)
        self.assertIn('性能分析报告', report)
        self.assertIn('函数性能统计', report)
        self.assertIn('内存使用统计', report)


class TestPerformanceDecorator(unittest.TestCase):
    """性能分析装饰器测试"""
    
    def test_profile_performance_decorator(self):
        """测试性能分析装饰器"""
        @profile_performance()
        def test_function():
            time.sleep(0.01)
            return "test"
        
        # 执行函数
        result = test_function()
        
        # 验证结果
        self.assertEqual(result, "test")
    
    def test_profile_performance_with_parameters(self):
        """测试带参数的性能分析装饰器"""
        @profile_performance(track_memory=False, track_cpu=False)
        def test_function():
            return "test"
        
        # 执行函数
        result = test_function()
        
        # 验证结果
        self.assertEqual(result, "test")


class TestPerformanceMetrics(unittest.TestCase):
    """性能指标测试"""
    
    def test_performance_metrics_creation(self):
        """测试性能指标创建"""
        metrics = PerformanceMetrics(
            function_name="test_function",
            module_name="test_module",
            call_count=5,
            total_time=1.0,
            avg_time=0.2,
            min_time=0.1,
            max_time=0.3,
            memory_usage=1024,
            cpu_usage=10.5
        )
        
        # 验证属性
        self.assertEqual(metrics.function_name, "test_function")
        self.assertEqual(metrics.call_count, 5)
        self.assertEqual(metrics.total_time, 1.0)
        
        # 验证转换为字典
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['function_name'], "test_function")
    
    def test_memory_snapshot_creation(self):
        """测试内存快照创建"""
        snapshot = MemorySnapshot(
            timestamp="2023-01-01T00:00:00",
            total_memory=8000000000,
            available_memory=4000000000,
            used_memory=4000000000,
            memory_percent=50.0,
            top_traces=[]
        )
        
        # 验证属性
        self.assertEqual(snapshot.timestamp, "2023-01-01T00:00:00")
        self.assertEqual(snapshot.memory_percent, 50.0)
        
        # 验证转换为字典
        snapshot_dict = snapshot.to_dict()
        self.assertIsInstance(snapshot_dict, dict)
        self.assertEqual(snapshot_dict['memory_percent'], 50.0)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_performance_analysis_workflow(self):
        """测试完整的性能分析工作流"""
        analyzer = PerformanceAnalyzer()
        
        # 启动分析
        analyzer.start_analysis()
        
        # 执行一些被监控的函数
        @profile_performance()
        def cpu_intensive_task():
            # CPU密集型任务
            result = sum(i * i for i in range(10000))
            return result
        
        @profile_performance()
        def memory_intensive_task():
            # 内存密集型任务
            data = [i for i in range(50000)]
            return len(data)
        
        # 执行任务
        for _ in range(3):
            cpu_intensive_task()
            memory_intensive_task()
            time.sleep(0.1)
        
        # 停止分析
        results = analyzer.stop_analysis()
        
        # 验证结果完整性
        self.assertIn('function_metrics', results)
        self.assertIn('memory_snapshots', results)
        self.assertIn('analysis_summary', results)
        
        # 验证函数指标
        function_metrics = results['function_metrics']
        self.assertGreater(len(function_metrics), 0)
        
        # 验证内存快照
        memory_snapshots = results['memory_snapshots']
        self.assertGreater(len(memory_snapshots), 0)
        
        # 验证分析摘要
        summary = results['analysis_summary']
        self.assertIn('total_functions_analyzed', summary)
        self.assertIn('recommendations', summary)
        
        # 生成报告
        report = analyzer.get_performance_report()
        self.assertIsInstance(report, str)
        self.assertIn('性能分析报告', report)


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestFunctionProfiler,
        TestMemoryProfiler,
        TestCodeProfiler,
        TestPerformanceAnalyzer,
        TestPerformanceDecorator,
        TestPerformanceMetrics,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    if result.wasSuccessful():
        print("\n所有性能分析器测试通过！")
    else:
        print(f"\n测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")