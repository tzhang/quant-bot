"""
系统集成测试用例
"""

import unittest
import time
import threading
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 导入所有需要测试的模块
from src.utils.logger import LoggerManager
from src.utils.error_handler import ErrorHandler
from src.utils.resource_manager import ResourceManager, get_resource_manager
from src.monitoring.system_monitor import SystemMonitor
from src.monitoring.performance_analyzer import PerformanceAnalyzer
from src.optimization.performance_optimizer import PerformanceOptimizer, get_performance_optimizer


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化各个组件
        self.logger_manager = LoggerManager()
        self.error_handler = ErrorHandler()
        self.resource_manager = get_resource_manager()
        self.system_monitor = SystemMonitor()
        self.performance_analyzer = PerformanceAnalyzer()
        self.performance_optimizer = get_performance_optimizer()
    
    def tearDown(self):
        """清理测试环境"""
        # 停止所有监控
        try:
            self.system_monitor.stop_monitoring()
            self.performance_analyzer.stop_analysis()
            self.performance_optimizer.stop_monitoring()
            self.resource_manager.cleanup()
        except:
            pass
        
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_logging_integration(self):
        """测试日志系统集成"""
        # 获取不同类型的日志器
        main_logger = self.logger_manager.get_logger('main')
        trading_logger = self.logger_manager.get_logger('trading')
        performance_logger = self.logger_manager.get_logger('performance')
        
        # 测试日志记录
        main_logger.info("系统启动")
        trading_logger.info("交易信号生成")
        
        # 测试性能日志记录
        with self.logger_manager.performance_logger('test_operation') as perf:
            time.sleep(0.1)
            perf.add_detail('processed_items', 100)
        
        # 测试异常记录
        try:
            raise ValueError("测试异常")
        except Exception as e:
            self.logger_manager.log_exception(e, context={'operation': 'test'})
        
        # 测试审计日志
        self.logger_manager.log_audit('user_login', {'user': 'test_user'})
        
        # 测试交易事件日志
        self.logger_manager.log_trading_event('order_placed', {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0
        })
    
    def test_error_handling_integration(self):
        """测试错误处理系统集成"""
        # 测试错误分类和处理
        def failing_function():
            raise ConnectionError("网络连接失败")
        
        # 测试重试机制
        @self.error_handler.retry(max_attempts=3, delay=0.1)
        def unstable_function(attempt_count=[0]):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("临时错误")
            return "成功"
        
        result = unstable_function()
        self.assertEqual(result, "成功")
        
        # 测试熔断器
        @self.error_handler.circuit_breaker(failure_threshold=2, timeout=1)
        def circuit_test_function(should_fail=True):
            if should_fail:
                raise RuntimeError("服务不可用")
            return "成功"
        
        # 触发熔断器
        for _ in range(3):
            try:
                circuit_test_function(True)
            except:
                pass
        
        # 测试安全执行
        result = self.error_handler.safe_execute(
            failing_function,
            default_value="默认值"
        )
        self.assertEqual(result, "默认值")
        
        # 获取错误统计
        stats = self.error_handler.get_error_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_errors', stats)
    
    def test_resource_management_integration(self):
        """测试资源管理系统集成"""
        # 创建连接池
        def mock_connection_factory():
            return Mock()
        
        pool = self.resource_manager.create_connection_pool(
            "test_pool",
            mock_connection_factory
        )
        
        # 测试连接使用
        with self.resource_manager.get_connection("test_pool") as conn:
            self.assertIsNotNone(conn)
        
        # 测试任务管理
        def test_task(x, y):
            return x + y
        
        future = self.resource_manager.submit_task("add_task", test_task, 1, 2)
        result = self.resource_manager.get_task_result("add_task", timeout=5)
        self.assertEqual(result, 3)
        
        # 测试缓存管理
        test_data = {"key": "value"}
        self.resource_manager.cache_object("test_key", test_data)
        cached_data = self.resource_manager.get_cached_object("test_key")
        self.assertEqual(cached_data, test_data)
        
        # 获取资源统计
        stats = self.resource_manager.get_resource_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('connection_pools', stats)
    
    def test_monitoring_integration(self):
        """测试监控系统集成"""
        # 启动系统监控
        self.system_monitor.start_monitoring()
        
        # 添加监控进程
        self.system_monitor.add_process("test_process", "python")
        
        # 等待收集一些数据
        time.sleep(1)
        
        # 获取当前状态
        status = self.system_monitor.get_current_status()
        self.assertIsInstance(status, dict)
        self.assertIn('system_metrics', status)
        self.assertIn('process_metrics', status)
        
        # 获取指标历史
        history = self.system_monitor.get_metrics_history(limit=10)
        self.assertIsInstance(history, list)
        
        # 停止监控
        self.system_monitor.stop_monitoring()
    
    def test_performance_analysis_integration(self):
        """测试性能分析系统集成"""
        # 启动性能分析
        self.performance_analyzer.start_analysis()
        
        # 测试函数性能分析
        @self.performance_analyzer.profile_performance
        def test_function(n):
            total = 0
            for i in range(n):
                total += i
            return total
        
        result = test_function(1000)
        self.assertEqual(result, sum(range(1000)))
        
        # 测试内存快照
        snapshot1 = self.performance_analyzer.memory_profiler.take_snapshot()
        
        # 创建一些对象
        test_objects = [list(range(100)) for _ in range(10)]
        
        snapshot2 = self.performance_analyzer.memory_profiler.take_snapshot()
        
        # 比较快照
        comparison = self.performance_analyzer.memory_profiler.compare_snapshots(
            snapshot1, snapshot2
        )
        self.assertIsInstance(comparison, dict)
        
        # 生成性能报告
        report = self.performance_analyzer.generate_report()
        self.assertIsInstance(report, dict)
        
        # 停止分析
        self.performance_analyzer.stop_analysis()
    
    def test_performance_optimization_integration(self):
        """测试性能优化系统集成"""
        # 启动性能监控
        self.performance_optimizer.start_monitoring()
        
        # 记录一些性能数据
        self.performance_optimizer.record_response_time(2.5)  # 高响应时间
        self.performance_optimizer.record_request()
        
        # 收集性能指标
        metrics = self.performance_optimizer.collect_metrics()
        self.assertIsNotNone(metrics)
        
        # 运行优化检查
        results = self.performance_optimizer.run_optimization_checks()
        self.assertIsInstance(results, list)
        
        # 手动优化
        optimization_results = self.performance_optimizer.optimize()
        self.assertIsInstance(optimization_results, list)
        
        # 生成性能报告
        report = self.performance_optimizer.generate_performance_report()
        self.assertIsInstance(report, dict)
        
        # 停止监控
        self.performance_optimizer.stop_monitoring()
    
    def test_cross_component_integration(self):
        """测试跨组件集成"""
        # 模拟一个完整的工作流程
        
        # 1. 启动所有监控系统
        self.system_monitor.start_monitoring()
        self.performance_analyzer.start_analysis()
        self.performance_optimizer.start_monitoring()
        
        # 2. 创建资源池
        def mock_db_connection():
            return Mock()
        
        self.resource_manager.create_connection_pool(
            "database",
            mock_db_connection
        )
        
        # 3. 模拟业务操作
        def business_operation():
            # 使用数据库连接
            with self.resource_manager.get_connection("database") as conn:
                # 模拟数据库操作
                time.sleep(0.1)
                
                # 记录性能指标
                self.performance_optimizer.record_response_time(0.1)
                self.performance_optimizer.record_request()
                
                # 记录业务日志
                logger = self.logger_manager.get_logger('business')
                logger.info("执行业务操作")
                
                return "操作成功"
        
        # 4. 执行业务操作（带错误处理）
        result = self.error_handler.safe_execute(
            business_operation,
            default_value="操作失败"
        )
        self.assertEqual(result, "操作成功")
        
        # 5. 异步执行任务
        future = self.resource_manager.submit_task(
            "async_business",
            business_operation
        )
        async_result = self.resource_manager.get_task_result(
            "async_business",
            timeout=5
        )
        self.assertEqual(async_result, "操作成功")
        
        # 6. 等待数据收集
        time.sleep(1)
        
        # 7. 获取各种统计信息
        system_status = self.system_monitor.get_current_status()
        resource_stats = self.resource_manager.get_resource_stats()
        performance_report = self.performance_optimizer.generate_performance_report()
        error_stats = self.error_handler.get_error_stats()
        
        # 验证数据完整性
        self.assertIsInstance(system_status, dict)
        self.assertIsInstance(resource_stats, dict)
        self.assertIsInstance(performance_report, dict)
        self.assertIsInstance(error_stats, dict)
        
        # 8. 清理资源
        self.system_monitor.stop_monitoring()
        self.performance_analyzer.stop_analysis()
        self.performance_optimizer.stop_monitoring()
        self.resource_manager.cleanup()
    
    def test_concurrent_operations(self):
        """测试并发操作"""
        # 创建多个线程同时执行不同操作
        results = []
        errors = []
        
        def worker_thread(worker_id):
            try:
                # 日志记录
                logger = self.logger_manager.get_logger(f'worker_{worker_id}')
                logger.info(f"工作线程 {worker_id} 启动")
                
                # 资源使用
                test_data = f"data_{worker_id}"
                self.resource_manager.cache_object(f"key_{worker_id}", test_data)
                cached = self.resource_manager.get_cached_object(f"key_{worker_id}")
                
                # 性能记录
                self.performance_optimizer.record_response_time(0.1 * worker_id)
                self.performance_optimizer.record_request()
                
                # 任务执行
                def task():
                    return worker_id * 2
                
                future = self.resource_manager.submit_task(
                    f"task_{worker_id}",
                    task
                )
                result = self.resource_manager.get_task_result(
                    f"task_{worker_id}",
                    timeout=5
                )
                
                results.append((worker_id, result, cached))
                
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # 创建并启动多个工作线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        
        for worker_id, result, cached in results:
            self.assertEqual(result, worker_id * 2)
            self.assertEqual(cached, f"data_{worker_id}")
    
    def test_error_propagation(self):
        """测试错误传播和处理"""
        # 模拟一个会产生错误的复杂操作
        def complex_operation():
            # 1. 资源获取可能失败
            with self.resource_manager.get_connection("nonexistent_pool") as conn:
                pass
        
        # 测试错误是否被正确捕获和处理
        result = self.error_handler.safe_execute(
            complex_operation,
            default_value="错误处理成功"
        )
        
        self.assertEqual(result, "错误处理成功")
        
        # 检查错误统计
        stats = self.error_handler.get_error_stats()
        self.assertGreater(stats['total_errors'], 0)
    
    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        # 启动内存监控
        self.performance_analyzer.start_analysis()
        
        # 创建初始快照
        initial_snapshot = self.performance_analyzer.memory_profiler.take_snapshot()
        
        # 模拟可能的内存泄漏
        leaked_objects = []
        for i in range(100):
            # 创建对象但不释放引用
            obj = [list(range(100)) for _ in range(10)]
            leaked_objects.append(obj)
            
            # 缓存对象（可能导致内存增长）
            self.resource_manager.cache_object(f"leak_test_{i}", obj)
        
        # 创建第二个快照
        second_snapshot = self.performance_analyzer.memory_profiler.take_snapshot()
        
        # 比较快照
        comparison = self.performance_analyzer.memory_profiler.compare_snapshots(
            initial_snapshot, second_snapshot
        )
        
        # 验证内存增长被检测到
        self.assertIsInstance(comparison, dict)
        self.assertIn('memory_growth', comparison)
        
        # 清理缓存
        self.resource_manager.clear_cache()
        
        # 创建清理后的快照
        final_snapshot = self.performance_analyzer.memory_profiler.take_snapshot()
        
        # 停止分析
        self.performance_analyzer.stop_analysis()
    
    def test_system_shutdown_cleanup(self):
        """测试系统关闭和清理"""
        # 启动所有组件
        self.system_monitor.start_monitoring()
        self.performance_analyzer.start_analysis()
        self.performance_optimizer.start_monitoring()
        
        # 创建一些资源
        def mock_factory():
            return Mock()
        
        self.resource_manager.create_connection_pool("test_pool", mock_factory)
        self.resource_manager.cache_object("test_key", "test_value")
        
        # 提交一些任务
        def test_task():
            time.sleep(0.5)
            return "完成"
        
        self.resource_manager.submit_task("test_task", test_task)
        
        # 模拟系统关闭
        self.system_monitor.stop_monitoring()
        self.performance_analyzer.stop_analysis()
        self.performance_optimizer.stop_monitoring()
        self.resource_manager.cleanup()
        
        # 验证所有组件都已正确停止
        self.assertFalse(self.system_monitor._monitoring)
        self.assertFalse(self.performance_analyzer._analyzing)
        self.assertFalse(self.performance_optimizer.is_monitoring)


class TestEndToEndScenarios(unittest.TestCase):
    """端到端场景测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_trading_system_simulation(self):
        """模拟交易系统场景"""
        # 初始化系统组件
        logger_manager = LoggerManager()
        error_handler = ErrorHandler()
        resource_manager = get_resource_manager()
        performance_optimizer = get_performance_optimizer()
        
        try:
            # 1. 系统启动
            trading_logger = logger_manager.get_logger('trading')
            trading_logger.info("交易系统启动")
            
            # 2. 创建数据库连接池
            def create_db_connection():
                # 模拟数据库连接
                return Mock()
            
            db_pool = resource_manager.create_connection_pool(
                "database",
                create_db_connection
            )
            
            # 3. 启动性能监控
            performance_optimizer.start_monitoring()
            
            # 4. 模拟交易操作
            def execute_trade(symbol, quantity, price):
                start_time = time.time()
                
                try:
                    # 使用数据库连接
                    with resource_manager.get_connection("database") as conn:
                        # 模拟数据库操作
                        time.sleep(0.05)  # 模拟延迟
                        
                        # 记录交易事件
                        logger_manager.log_trading_event('trade_executed', {
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': price,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # 记录性能指标
                        response_time = time.time() - start_time
                        performance_optimizer.record_response_time(response_time)
                        performance_optimizer.record_request()
                        
                        return f"交易成功: {symbol} {quantity}@{price}"
                        
                except Exception as e:
                    logger_manager.log_exception(e, context={
                        'operation': 'execute_trade',
                        'symbol': symbol
                    })
                    raise
            
            # 5. 执行多个交易（带错误处理）
            trades = [
                ('AAPL', 100, 150.0),
                ('GOOGL', 50, 2500.0),
                ('MSFT', 200, 300.0),
                ('TSLA', 75, 800.0),
                ('AMZN', 25, 3200.0)
            ]
            
            results = []
            for symbol, quantity, price in trades:
                result = error_handler.safe_execute(
                    execute_trade,
                    symbol, quantity, price,
                    default_value=f"交易失败: {symbol}"
                )
                results.append(result)
            
            # 6. 异步处理一些后台任务
            def background_task(task_name):
                time.sleep(0.1)
                return f"后台任务完成: {task_name}"
            
            background_futures = []
            for i in range(3):
                future = resource_manager.submit_task(
                    f"background_{i}",
                    background_task,
                    f"task_{i}"
                )
                background_futures.append(future)
            
            # 等待后台任务完成
            background_results = []
            for i in range(3):
                result = resource_manager.get_task_result(
                    f"background_{i}",
                    timeout=5
                )
                background_results.append(result)
            
            # 7. 生成性能报告
            time.sleep(1)  # 等待数据收集
            performance_report = performance_optimizer.generate_performance_report()
            
            # 8. 验证结果
            self.assertEqual(len(results), 5)
            self.assertEqual(len(background_results), 3)
            self.assertIsInstance(performance_report, dict)
            
            # 验证所有交易都成功或有适当的错误处理
            for result in results:
                self.assertIn("交易", result)
            
            for result in background_results:
                self.assertIn("后台任务完成", result)
            
        finally:
            # 9. 系统关闭
            performance_optimizer.stop_monitoring()
            resource_manager.cleanup()
            trading_logger.info("交易系统关闭")
    
    def test_high_load_scenario(self):
        """高负载场景测试"""
        resource_manager = get_resource_manager()
        performance_optimizer = get_performance_optimizer()
        
        try:
            # 创建连接池
            def create_connection():
                return Mock()
            
            resource_manager.create_connection_pool(
                "high_load_pool",
                create_connection,
                min_connections=5,
                max_connections=20
            )
            
            # 启动监控
            performance_optimizer.start_monitoring()
            
            # 模拟高并发请求
            def high_load_operation(request_id):
                with resource_manager.get_connection("high_load_pool") as conn:
                    # 模拟处理时间
                    processing_time = 0.01 + (request_id % 10) * 0.01
                    time.sleep(processing_time)
                    
                    # 记录性能指标
                    performance_optimizer.record_response_time(processing_time)
                    performance_optimizer.record_request()
                    
                    return f"请求 {request_id} 处理完成"
            
            # 提交大量并发任务
            futures = []
            for i in range(100):
                future = resource_manager.submit_task(
                    f"high_load_{i}",
                    high_load_operation,
                    i
                )
                futures.append((i, future))
            
            # 等待所有任务完成
            completed_count = 0
            for request_id, future in futures:
                try:
                    result = resource_manager.get_task_result(
                        f"high_load_{request_id}",
                        timeout=10
                    )
                    if result:
                        completed_count += 1
                except:
                    pass
            
            # 验证大部分任务都成功完成
            self.assertGreater(completed_count, 80)  # 至少80%成功
            
            # 检查性能指标
            time.sleep(1)
            performance_report = performance_optimizer.generate_performance_report()
            self.assertIsInstance(performance_report, dict)
            
        finally:
            performance_optimizer.stop_monitoring()
            resource_manager.cleanup()


if __name__ == '__main__':
    unittest.main()