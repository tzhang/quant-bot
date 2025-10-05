"""
性能优化器测试用例
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.optimization.performance_optimizer import (
    PerformanceOptimizer,
    PerformanceProfile,
    PerformanceMetrics,
    OptimizationResult,
    CPUOptimizer,
    MemoryOptimizer,
    ResponseTimeOptimizer,
    ThroughputOptimizer,
    get_performance_optimizer,
    optimize_performance,
    profile_function
)


class TestPerformanceProfile(unittest.TestCase):
    """性能配置文件测试"""
    
    def test_profile_creation(self):
        """测试配置文件创建"""
        profile = PerformanceProfile(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            response_time_threshold=2.0,
            throughput_threshold=100.0,
            optimization_interval=60,
            enable_auto_optimization=True
        )
        
        self.assertEqual(profile.cpu_threshold, 80.0)
        self.assertEqual(profile.memory_threshold, 85.0)
        self.assertEqual(profile.response_time_threshold, 2.0)
        self.assertEqual(profile.throughput_threshold, 100.0)
        self.assertEqual(profile.optimization_interval, 60)
        self.assertTrue(profile.enable_auto_optimization)
    
    def test_profile_to_dict(self):
        """测试配置文件转换为字典"""
        profile = PerformanceProfile()
        profile_dict = profile.to_dict()
        
        expected_keys = [
            'cpu_threshold', 'memory_threshold', 'response_time_threshold',
            'throughput_threshold', 'optimization_interval', 'enable_auto_optimization'
        ]
        
        for key in expected_keys:
            self.assertIn(key, profile_dict)


class TestPerformanceMetrics(unittest.TestCase):
    """性能指标测试"""
    
    def test_metrics_creation(self):
        """测试指标创建"""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            response_time=1.5,
            throughput=150.0,
            error_rate=0.01,
            active_connections=25
        )
        
        self.assertEqual(metrics.cpu_usage, 50.0)
        self.assertEqual(metrics.memory_usage, 60.0)
        self.assertEqual(metrics.response_time, 1.5)
        self.assertEqual(metrics.throughput, 150.0)
        self.assertEqual(metrics.error_rate, 0.01)
        self.assertEqual(metrics.active_connections, 25)
    
    def test_metrics_to_dict(self):
        """测试指标转换为字典"""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            response_time=1.5,
            throughput=150.0,
            error_rate=0.01,
            active_connections=25
        )
        
        metrics_dict = metrics.to_dict()
        
        expected_keys = [
            'timestamp', 'cpu_usage', 'memory_usage', 'response_time',
            'throughput', 'error_rate', 'active_connections'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics_dict)


class TestOptimizationResult(unittest.TestCase):
    """优化结果测试"""
    
    def test_result_creation(self):
        """测试结果创建"""
        result = OptimizationResult(
            optimizer_name="TestOptimizer",
            success=True,
            message="Optimization successful",
            metrics_before={"cpu": 80.0},
            metrics_after={"cpu": 60.0},
            improvement=20.0,
            timestamp=datetime.now()
        )
        
        self.assertEqual(result.optimizer_name, "TestOptimizer")
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Optimization successful")
        self.assertEqual(result.improvement, 20.0)
    
    def test_result_to_dict(self):
        """测试结果转换为字典"""
        result = OptimizationResult(
            optimizer_name="TestOptimizer",
            success=True,
            message="Optimization successful",
            metrics_before={"cpu": 80.0},
            metrics_after={"cpu": 60.0},
            improvement=20.0,
            timestamp=datetime.now()
        )
        
        result_dict = result.to_dict()
        
        expected_keys = [
            'optimizer_name', 'success', 'message', 'metrics_before',
            'metrics_after', 'improvement', 'timestamp'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result_dict)


class TestCPUOptimizer(unittest.TestCase):
    """CPU优化器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.optimizer = CPUOptimizer()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.Process')
    def test_analyze_cpu_usage(self, mock_process, mock_cpu_percent):
        """测试CPU使用率分析"""
        mock_cpu_percent.return_value = 85.0
        mock_proc = Mock()
        mock_proc.cpu_percent.return_value = 20.0
        mock_proc.name.return_value = "test_process"
        mock_proc.pid = 1234
        mock_process.return_value = mock_proc
        
        with patch('psutil.process_iter', return_value=[mock_proc]):
            analysis = self.optimizer.analyze()
        
        self.assertIn('cpu_usage', analysis)
        self.assertIn('high_cpu_processes', analysis)
        self.assertEqual(analysis['cpu_usage'], 85.0)
    
    @patch('psutil.cpu_percent')
    def test_optimize_cpu_high_usage(self, mock_cpu_percent):
        """测试高CPU使用率优化"""
        mock_cpu_percent.return_value = 90.0
        
        result = self.optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.optimizer_name, "CPUOptimizer")
    
    @patch('psutil.cpu_percent')
    def test_optimize_cpu_normal_usage(self, mock_cpu_percent):
        """测试正常CPU使用率"""
        mock_cpu_percent.return_value = 50.0
        
        result = self.optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)


class TestMemoryOptimizer(unittest.TestCase):
    """内存优化器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.optimizer = MemoryOptimizer()
    
    @patch('psutil.virtual_memory')
    @patch('gc.collect')
    def test_analyze_memory_usage(self, mock_gc_collect, mock_virtual_memory):
        """测试内存使用率分析"""
        mock_memory = Mock()
        mock_memory.percent = 85.0
        mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.available = 2 * 1024 * 1024 * 1024  # 2GB
        mock_virtual_memory.return_value = mock_memory
        
        analysis = self.optimizer.analyze()
        
        self.assertIn('memory_usage', analysis)
        self.assertIn('memory_used', analysis)
        self.assertIn('memory_available', analysis)
        self.assertEqual(analysis['memory_usage'], 85.0)
    
    @patch('psutil.virtual_memory')
    @patch('gc.collect')
    def test_optimize_memory_high_usage(self, mock_gc_collect, mock_virtual_memory):
        """测试高内存使用率优化"""
        mock_memory = Mock()
        mock_memory.percent = 90.0
        mock_virtual_memory.return_value = mock_memory
        mock_gc_collect.return_value = 100  # 回收了100个对象
        
        result = self.optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.optimizer_name, "MemoryOptimizer")
        mock_gc_collect.assert_called()
    
    @patch('psutil.virtual_memory')
    def test_optimize_memory_normal_usage(self, mock_virtual_memory):
        """测试正常内存使用率"""
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        
        result = self.optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)


class TestResponseTimeOptimizer(unittest.TestCase):
    """响应时间优化器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.optimizer = ResponseTimeOptimizer()
    
    def test_record_response_time(self):
        """测试记录响应时间"""
        self.optimizer.record_response_time(1.5)
        self.optimizer.record_response_time(2.0)
        self.optimizer.record_response_time(1.2)
        
        self.assertEqual(len(self.optimizer.response_times), 3)
    
    def test_analyze_response_times(self):
        """测试响应时间分析"""
        # 记录一些响应时间
        response_times = [1.0, 1.5, 2.0, 2.5, 3.0]
        for rt in response_times:
            self.optimizer.record_response_time(rt)
        
        analysis = self.optimizer.analyze()
        
        self.assertIn('avg_response_time', analysis)
        self.assertIn('max_response_time', analysis)
        self.assertIn('min_response_time', analysis)
        self.assertIn('p95_response_time', analysis)
        
        self.assertEqual(analysis['avg_response_time'], 2.0)
        self.assertEqual(analysis['max_response_time'], 3.0)
        self.assertEqual(analysis['min_response_time'], 1.0)
    
    def test_optimize_high_response_time(self):
        """测试高响应时间优化"""
        # 记录一些高响应时间
        for _ in range(10):
            self.optimizer.record_response_time(5.0)  # 高于默认阈值
        
        result = self.optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.optimizer_name, "ResponseTimeOptimizer")
    
    def test_optimize_normal_response_time(self):
        """测试正常响应时间"""
        # 记录一些正常响应时间
        for _ in range(10):
            self.optimizer.record_response_time(1.0)  # 低于默认阈值
        
        result = self.optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)


class TestThroughputOptimizer(unittest.TestCase):
    """吞吐量优化器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.optimizer = ThroughputOptimizer()
    
    def test_record_request(self):
        """测试记录请求"""
        self.optimizer.record_request()
        self.optimizer.record_request()
        self.optimizer.record_request()
        
        self.assertEqual(len(self.optimizer.request_times), 3)
    
    def test_analyze_throughput(self):
        """测试吞吐量分析"""
        # 记录一些请求
        for _ in range(10):
            self.optimizer.record_request()
            time.sleep(0.01)  # 小延迟
        
        analysis = self.optimizer.analyze()
        
        self.assertIn('current_throughput', analysis)
        self.assertIn('total_requests', analysis)
        
        self.assertEqual(analysis['total_requests'], 10)
        self.assertGreater(analysis['current_throughput'], 0)
    
    def test_optimize_low_throughput(self):
        """测试低吞吐量优化"""
        # 模拟低吞吐量场景
        self.optimizer.request_times = [time.time() - 60]  # 1分钟前的一个请求
        
        result = self.optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.optimizer_name, "ThroughputOptimizer")
    
    def test_optimize_normal_throughput(self):
        """测试正常吞吐量"""
        # 模拟正常吞吐量场景
        current_time = time.time()
        for i in range(200):  # 200个请求在1分钟内
            self.optimizer.request_times.append(current_time - i * 0.3)
        
        result = self.optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)


class TestPerformanceOptimizer(unittest.TestCase):
    """性能优化器主类测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.profile = PerformanceProfile(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            response_time_threshold=2.0,
            throughput_threshold=100.0,
            optimization_interval=1,  # 短间隔用于测试
            enable_auto_optimization=False  # 禁用自动优化
        )
        self.optimizer = PerformanceOptimizer(self.profile)
    
    def tearDown(self):
        """清理测试环境"""
        self.optimizer.stop_monitoring()
    
    def test_optimizer_creation(self):
        """测试优化器创建"""
        self.assertEqual(self.optimizer.profile, self.profile)
        self.assertIsInstance(self.optimizer.cpu_optimizer, CPUOptimizer)
        self.assertIsInstance(self.optimizer.memory_optimizer, MemoryOptimizer)
        self.assertIsInstance(self.optimizer.response_time_optimizer, ResponseTimeOptimizer)
        self.assertIsInstance(self.optimizer.throughput_optimizer, ThroughputOptimizer)
    
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        self.assertFalse(self.optimizer.is_monitoring)
        
        self.optimizer.start_monitoring()
        self.assertTrue(self.optimizer.is_monitoring)
        
        self.optimizer.stop_monitoring()
        self.assertFalse(self.optimizer.is_monitoring)
    
    def test_collect_metrics(self):
        """测试收集指标"""
        metrics = self.optimizer.collect_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertIsInstance(metrics.timestamp, datetime)
        self.assertGreaterEqual(metrics.cpu_usage, 0)
        self.assertGreaterEqual(metrics.memory_usage, 0)
    
    def test_run_optimization_checks(self):
        """测试运行优化检查"""
        results = self.optimizer.run_optimization_checks()
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 4)  # 4个优化器
        
        for result in results:
            self.assertIsInstance(result, OptimizationResult)
    
    def test_optimize_manually(self):
        """测试手动优化"""
        results = self.optimizer.optimize()
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIsInstance(result, OptimizationResult)
    
    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        # 先收集一些指标
        self.optimizer.collect_metrics()
        time.sleep(0.1)
        self.optimizer.collect_metrics()
        
        metrics = self.optimizer.get_performance_metrics(limit=10)
        
        self.assertIsInstance(metrics, list)
        self.assertGreaterEqual(len(metrics), 1)
        
        for metric in metrics:
            self.assertIsInstance(metric, PerformanceMetrics)
    
    def test_get_optimization_history(self):
        """测试获取优化历史"""
        # 先运行一次优化
        self.optimizer.optimize()
        
        history = self.optimizer.get_optimization_history(limit=10)
        
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)
        
        for result in history:
            self.assertIsInstance(result, OptimizationResult)
    
    def test_generate_performance_report(self):
        """测试生成性能报告"""
        # 先收集一些数据
        self.optimizer.collect_metrics()
        self.optimizer.optimize()
        
        report = self.optimizer.generate_performance_report()
        
        self.assertIsInstance(report, dict)
        
        expected_keys = [
            'timestamp', 'profile', 'current_metrics',
            'recent_metrics', 'optimization_history', 'summary'
        ]
        
        for key in expected_keys:
            self.assertIn(key, report)
    
    def test_record_response_time(self):
        """测试记录响应时间"""
        self.optimizer.record_response_time(1.5)
        
        # 检查是否记录到响应时间优化器
        self.assertEqual(len(self.optimizer.response_time_optimizer.response_times), 1)
    
    def test_record_request(self):
        """测试记录请求"""
        self.optimizer.record_request()
        
        # 检查是否记录到吞吐量优化器
        self.assertEqual(len(self.optimizer.throughput_optimizer.request_times), 1)


class TestGlobalPerformanceOptimizer(unittest.TestCase):
    """全局性能优化器测试"""
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        optimizer1 = get_performance_optimizer()
        optimizer2 = get_performance_optimizer()
        
        self.assertIs(optimizer1, optimizer2)
    
    def test_custom_profile(self):
        """测试自定义配置"""
        custom_profile = PerformanceProfile(cpu_threshold=90.0)
        optimizer = get_performance_optimizer(custom_profile)
        
        self.assertEqual(optimizer.profile.cpu_threshold, 90.0)


class TestDecorators(unittest.TestCase):
    """装饰器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.optimizer = get_performance_optimizer()
    
    def test_optimize_performance_decorator(self):
        """测试性能优化装饰器"""
        @optimize_performance
        def test_function(x, y):
            time.sleep(0.1)  # 模拟一些工作
            return x + y
        
        result = test_function(1, 2)
        self.assertEqual(result, 3)
        
        # 检查是否记录了响应时间
        self.assertGreater(len(self.optimizer.response_time_optimizer.response_times), 0)
    
    def test_profile_function_decorator(self):
        """测试函数性能分析装饰器"""
        @profile_function
        def test_function(x):
            # 模拟一些计算
            total = 0
            for i in range(x):
                total += i
            return total
        
        result = test_function(1000)
        self.assertEqual(result, sum(range(1000)))
    
    def test_profile_function_with_stats(self):
        """测试带统计信息的函数性能分析"""
        stats_collected = []
        
        @profile_function
        def test_function(n):
            return sum(range(n))
        
        # 多次调用以收集统计信息
        for i in range(5):
            test_function(100)
        
        # 这里主要测试装饰器不会抛出异常
        # 实际的性能统计会在日志中


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.profile = PerformanceProfile(
            optimization_interval=1,
            enable_auto_optimization=False
        )
        self.optimizer = PerformanceOptimizer(self.profile)
    
    def tearDown(self):
        """清理测试环境"""
        self.optimizer.stop_monitoring()
    
    def test_full_optimization_workflow(self):
        """测试完整的优化工作流"""
        # 1. 启动监控
        self.optimizer.start_monitoring()
        
        # 2. 记录一些性能数据
        self.optimizer.record_response_time(2.5)  # 高响应时间
        self.optimizer.record_request()
        
        # 3. 收集指标
        metrics = self.optimizer.collect_metrics()
        self.assertIsInstance(metrics, PerformanceMetrics)
        
        # 4. 运行优化
        results = self.optimizer.optimize()
        self.assertIsInstance(results, list)
        
        # 5. 生成报告
        report = self.optimizer.generate_performance_report()
        self.assertIsInstance(report, dict)
        
        # 6. 停止监控
        self.optimizer.stop_monitoring()
        self.assertFalse(self.optimizer.is_monitoring)
    
    def test_concurrent_optimization(self):
        """测试并发优化"""
        def run_optimization():
            return self.optimizer.optimize()
        
        # 创建多个线程同时运行优化
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_optimization)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查优化历史
        history = self.optimizer.get_optimization_history()
        self.assertGreaterEqual(len(history), 3)


if __name__ == '__main__':
    unittest.main()