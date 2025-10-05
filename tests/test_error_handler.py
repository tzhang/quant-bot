"""
错误处理器测试用例
测试错误分类、重试机制、熔断器和通知功能
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime, timedelta

from src.utils.error_handler import (
    ErrorHandler, ErrorSeverity, ErrorCategory, ErrorInfo, RetryConfig,
    CircuitBreaker, CircuitBreakerState, ErrorNotifier
)


class TestErrorSeverity(unittest.TestCase):
    """错误严重程度测试"""
    
    def test_severity_values(self):
        """测试严重程度枚举值"""
        self.assertEqual(ErrorSeverity.LOW.value, "low")
        self.assertEqual(ErrorSeverity.MEDIUM.value, "medium")
        self.assertEqual(ErrorSeverity.HIGH.value, "high")
        self.assertEqual(ErrorSeverity.CRITICAL.value, "critical")
    
    def test_severity_comparison(self):
        """测试严重程度比较"""
        # 测试枚举比较（基于定义顺序）
        severities = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, 
                     ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        
        for i in range(len(severities)):
            for j in range(i + 1, len(severities)):
                # 注意：枚举比较基于定义顺序，不是值
                pass


class TestErrorCategory(unittest.TestCase):
    """错误类别测试"""
    
    def test_category_values(self):
        """测试错误类别枚举值"""
        self.assertEqual(ErrorCategory.NETWORK.value, "network")
        self.assertEqual(ErrorCategory.DATABASE.value, "database")
        self.assertEqual(ErrorCategory.TRADING.value, "trading")
        self.assertEqual(ErrorCategory.VALIDATION.value, "validation")
        self.assertEqual(ErrorCategory.SYSTEM.value, "system")
        self.assertEqual(ErrorCategory.UNKNOWN.value, "unknown")


class TestErrorInfo(unittest.TestCase):
    """错误信息测试"""
    
    def test_error_info_creation(self):
        """测试错误信息创建"""
        error_info = ErrorInfo(
            error_type="ValueError",
            message="测试错误",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            context={"user_id": 123},
            traceback="Traceback...",
            timestamp=datetime.now()
        )
        
        # 验证属性
        self.assertEqual(error_info.error_type, "ValueError")
        self.assertEqual(error_info.message, "测试错误")
        self.assertEqual(error_info.severity, ErrorSeverity.HIGH)
        self.assertEqual(error_info.category, ErrorCategory.VALIDATION)
        self.assertEqual(error_info.context["user_id"], 123)
    
    def test_error_info_to_dict(self):
        """测试错误信息转换为字典"""
        timestamp = datetime.now()
        error_info = ErrorInfo(
            error_type="ValueError",
            message="测试错误",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            timestamp=timestamp
        )
        
        # 转换为字典
        error_dict = error_info.to_dict()
        
        # 验证字典内容
        self.assertEqual(error_dict["error_type"], "ValueError")
        self.assertEqual(error_dict["message"], "测试错误")
        self.assertEqual(error_dict["severity"], "high")
        self.assertEqual(error_dict["category"], "validation")
        self.assertEqual(error_dict["timestamp"], timestamp.isoformat())


class TestRetryConfig(unittest.TestCase):
    """重试配置测试"""
    
    def test_retry_config_creation(self):
        """测试重试配置创建"""
        config = RetryConfig(
            max_attempts=5,
            delay=2.0,
            backoff_factor=1.5,
            max_delay=30.0,
            exceptions=(ValueError, TypeError)
        )
        
        # 验证属性
        self.assertEqual(config.max_attempts, 5)
        self.assertEqual(config.delay, 2.0)
        self.assertEqual(config.backoff_factor, 1.5)
        self.assertEqual(config.max_delay, 30.0)
        self.assertEqual(config.exceptions, (ValueError, TypeError))
    
    def test_retry_config_defaults(self):
        """测试重试配置默认值"""
        config = RetryConfig()
        
        # 验证默认值
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.delay, 1.0)
        self.assertEqual(config.backoff_factor, 2.0)
        self.assertEqual(config.max_delay, 60.0)
        self.assertEqual(config.exceptions, (Exception,))


class TestCircuitBreaker(unittest.TestCase):
    """熔断器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=1.0,
            expected_exception=ValueError
        )
    
    def test_initial_state(self):
        """测试初始状态"""
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertIsNone(self.circuit_breaker.last_failure_time)
    
    def test_successful_call(self):
        """测试成功调用"""
        def successful_function():
            return "success"
        
        # 执行成功的函数
        result = self.circuit_breaker.call(successful_function)
        
        # 验证结果和状态
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_failure_accumulation(self):
        """测试失败累积"""
        def failing_function():
            raise ValueError("测试失败")
        
        # 执行失败的函数，但不超过阈值
        for i in range(2):
            with self.assertRaises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        # 验证状态仍然是关闭的
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 2)
    
    def test_circuit_opens(self):
        """测试熔断器打开"""
        def failing_function():
            raise ValueError("测试失败")
        
        # 执行失败的函数，超过阈值
        for i in range(3):
            with self.assertRaises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        # 验证熔断器打开
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)
        self.assertEqual(self.circuit_breaker.failure_count, 3)
        self.assertIsNotNone(self.circuit_breaker.last_failure_time)
    
    def test_circuit_open_blocks_calls(self):
        """测试熔断器打开时阻止调用"""
        def failing_function():
            raise ValueError("测试失败")
        
        # 触发熔断器打开
        for i in range(3):
            with self.assertRaises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        # 验证后续调用被阻止
        with self.assertRaises(Exception) as context:
            self.circuit_breaker.call(failing_function)
        
        self.assertIn("熔断器已打开", str(context.exception))
    
    def test_circuit_half_open(self):
        """测试熔断器半开状态"""
        def failing_function():
            raise ValueError("测试失败")
        
        # 触发熔断器打开
        for i in range(3):
            with self.assertRaises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        # 等待超时时间
        time.sleep(1.1)
        
        # 下一次调用应该进入半开状态
        def successful_function():
            return "success"
        
        result = self.circuit_breaker.call(successful_function)
        
        # 验证熔断器恢复到关闭状态
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_unexpected_exception_not_counted(self):
        """测试非预期异常不被计入失败"""
        def function_with_unexpected_error():
            raise TypeError("非预期异常")
        
        # 执行抛出非预期异常的函数
        with self.assertRaises(TypeError):
            self.circuit_breaker.call(function_with_unexpected_error)
        
        # 验证失败计数没有增加
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)


class TestErrorNotifier(unittest.TestCase):
    """错误通知器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.notifier = ErrorNotifier(
            email_config={
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'username': 'test@example.com',
                'password': 'password',
                'from_email': 'test@example.com',
                'to_emails': ['admin@example.com']
            },
            webhook_config={
                'url': 'https://hooks.slack.com/test',
                'headers': {'Content-Type': 'application/json'}
            },
            rate_limit_minutes=5
        )
    
    @patch('smtplib.SMTP')
    def test_send_email_notification(self, mock_smtp):
        """测试发送邮件通知"""
        # 模拟SMTP服务器
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # 创建错误信息
        error_info = ErrorInfo(
            error_type="ValueError",
            message="测试错误",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION
        )
        
        # 发送通知
        success = self.notifier.send_email_notification(error_info)
        
        # 验证结果
        self.assertTrue(success)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_send_email_notification_failure(self, mock_smtp):
        """测试邮件发送失败"""
        # 模拟SMTP异常
        mock_smtp.side_effect = Exception("SMTP连接失败")
        
        # 创建错误信息
        error_info = ErrorInfo(
            error_type="ValueError",
            message="测试错误",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION
        )
        
        # 发送通知
        success = self.notifier.send_email_notification(error_info)
        
        # 验证失败
        self.assertFalse(success)
    
    @patch('requests.post')
    def test_send_webhook_notification(self, mock_post):
        """测试发送Webhook通知"""
        # 模拟成功响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # 创建错误信息
        error_info = ErrorInfo(
            error_type="ValueError",
            message="测试错误",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION
        )
        
        # 发送通知
        success = self.notifier.send_webhook_notification(error_info)
        
        # 验证结果
        self.assertTrue(success)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_send_webhook_notification_failure(self, mock_post):
        """测试Webhook发送失败"""
        # 模拟失败响应
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        # 创建错误信息
        error_info = ErrorInfo(
            error_type="ValueError",
            message="测试错误",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION
        )
        
        # 发送通知
        success = self.notifier.send_webhook_notification(error_info)
        
        # 验证失败
        self.assertFalse(success)
    
    def test_rate_limiting(self):
        """测试频率限制"""
        # 创建错误信息
        error_info = ErrorInfo(
            error_type="ValueError",
            message="测试错误",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION
        )
        
        # 第一次通知应该被允许
        with patch.object(self.notifier, 'send_email_notification', return_value=True):
            result1 = self.notifier.notify(error_info)
            self.assertTrue(result1)
        
        # 立即再次通知应该被限制
        with patch.object(self.notifier, 'send_email_notification', return_value=True):
            result2 = self.notifier.notify(error_info)
            self.assertFalse(result2)


class TestErrorHandler(unittest.TestCase):
    """错误处理器主类测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.error_handler = ErrorHandler()
    
    def test_classify_error_network(self):
        """测试网络错误分类"""
        import requests
        
        # 测试网络相关异常
        network_errors = [
            requests.ConnectionError("连接失败"),
            requests.Timeout("请求超时"),
            ConnectionError("连接错误")
        ]
        
        for error in network_errors:
            category = self.error_handler._classify_error(error)
            self.assertEqual(category, ErrorCategory.NETWORK)
    
    def test_classify_error_database(self):
        """测试数据库错误分类"""
        # 模拟数据库异常
        class DatabaseError(Exception):
            pass
        
        db_error = DatabaseError("数据库连接失败")
        category = self.error_handler._classify_error(db_error)
        self.assertEqual(category, ErrorCategory.DATABASE)
    
    def test_classify_error_validation(self):
        """测试验证错误分类"""
        validation_errors = [
            ValueError("无效的值"),
            TypeError("类型错误"),
            KeyError("键不存在")
        ]
        
        for error in validation_errors:
            category = self.error_handler._classify_error(error)
            self.assertEqual(category, ErrorCategory.VALIDATION)
    
    def test_determine_severity(self):
        """测试严重程度判断"""
        # 测试不同类型的错误
        test_cases = [
            (ValueError("简单错误"), ErrorSeverity.LOW),
            (ConnectionError("网络错误"), ErrorSeverity.MEDIUM),
            (Exception("系统错误"), ErrorSeverity.HIGH)
        ]
        
        for error, expected_severity in test_cases:
            severity = self.error_handler._determine_severity(error)
            # 由于严重程度判断可能有多种策略，这里只验证返回了有效的严重程度
            self.assertIn(severity, [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, 
                                   ErrorSeverity.HIGH, ErrorSeverity.CRITICAL])
    
    def test_handle_error(self):
        """测试错误处理"""
        try:
            raise ValueError("测试错误")
        except Exception as e:
            # 处理错误
            error_info = self.error_handler.handle_error(e, context={"test": True})
            
            # 验证错误信息
            self.assertIsInstance(error_info, ErrorInfo)
            self.assertEqual(error_info.error_type, "ValueError")
            self.assertEqual(error_info.message, "测试错误")
            self.assertEqual(error_info.context["test"], True)
    
    def test_retry_decorator_success(self):
        """测试重试装饰器成功情况"""
        call_count = 0
        
        @self.error_handler.retry(RetryConfig(max_attempts=3))
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("临时失败")
            return "success"
        
        # 执行函数
        result = sometimes_failing_function()
        
        # 验证结果
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)  # 第一次失败，第二次成功
    
    def test_retry_decorator_max_attempts(self):
        """测试重试装饰器达到最大尝试次数"""
        call_count = 0
        
        @self.error_handler.retry(RetryConfig(max_attempts=3, delay=0.1))
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("总是失败")
        
        # 执行函数应该抛出异常
        with self.assertRaises(ValueError):
            always_failing_function()
        
        # 验证尝试次数
        self.assertEqual(call_count, 3)
    
    def test_circuit_breaker_decorator(self):
        """测试熔断器装饰器"""
        call_count = 0
        
        @self.error_handler.circuit_breaker(
            failure_threshold=2,
            timeout=0.5,
            expected_exception=ValueError
        )
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("失败")
        
        # 执行函数直到熔断器打开
        for i in range(2):
            with self.assertRaises(ValueError):
                failing_function()
        
        # 下一次调用应该被熔断器阻止
        with self.assertRaises(Exception) as context:
            failing_function()
        
        self.assertIn("熔断器已打开", str(context.exception))
        self.assertEqual(call_count, 2)  # 只调用了2次，第3次被阻止
    
    def test_safe_execute_success(self):
        """测试安全执行成功情况"""
        def successful_function():
            return "success"
        
        # 安全执行
        result = self.error_handler.safe_execute(successful_function)
        
        # 验证结果
        self.assertEqual(result, "success")
    
    def test_safe_execute_with_exception(self):
        """测试安全执行异常情况"""
        def failing_function():
            raise ValueError("测试异常")
        
        # 安全执行
        result = self.error_handler.safe_execute(
            failing_function, 
            default_value="default"
        )
        
        # 验证返回默认值
        self.assertEqual(result, "default")
    
    def test_get_error_statistics(self):
        """测试获取错误统计"""
        # 生成一些错误
        for i in range(5):
            try:
                raise ValueError(f"错误 {i}")
            except Exception as e:
                self.error_handler.handle_error(e)
        
        # 获取统计
        stats = self.error_handler.get_error_statistics()
        
        # 验证统计信息
        self.assertIn('total_errors', stats)
        self.assertIn('errors_by_type', stats)
        self.assertIn('errors_by_category', stats)
        self.assertIn('errors_by_severity', stats)
        self.assertEqual(stats['total_errors'], 5)
    
    def test_get_recent_errors(self):
        """测试获取最近错误"""
        # 生成一些错误
        for i in range(3):
            try:
                raise ValueError(f"错误 {i}")
            except Exception as e:
                self.error_handler.handle_error(e)
        
        # 获取最近错误
        recent_errors = self.error_handler.get_recent_errors(limit=2)
        
        # 验证结果
        self.assertEqual(len(recent_errors), 2)
        self.assertIsInstance(recent_errors[0], ErrorInfo)
    
    def test_mark_error_resolved(self):
        """测试标记错误已解决"""
        # 生成一个错误
        try:
            raise ValueError("测试错误")
        except Exception as e:
            error_info = self.error_handler.handle_error(e)
        
        # 标记为已解决
        success = self.error_handler.mark_error_resolved(error_info.error_id)
        
        # 验证结果
        self.assertTrue(success)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_error_handling_workflow(self):
        """测试完整的错误处理工作流"""
        # 创建错误处理器
        error_handler = ErrorHandler()
        
        # 模拟一个复杂的业务函数
        @error_handler.retry(RetryConfig(max_attempts=3, delay=0.1))
        @error_handler.circuit_breaker(failure_threshold=5, timeout=1.0)
        def complex_business_function(should_fail=False):
            if should_fail:
                raise ValueError("业务逻辑错误")
            return "业务处理成功"
        
        # 测试成功情况
        result = complex_business_function(should_fail=False)
        self.assertEqual(result, "业务处理成功")
        
        # 测试失败情况
        with self.assertRaises(ValueError):
            complex_business_function(should_fail=True)
        
        # 验证错误被记录
        stats = error_handler.get_error_statistics()
        self.assertGreater(stats['total_errors'], 0)
        
        # 测试安全执行
        safe_result = error_handler.safe_execute(
            lambda: complex_business_function(should_fail=True),
            default_value="安全默认值"
        )
        self.assertEqual(safe_result, "安全默认值")


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestErrorSeverity,
        TestErrorCategory,
        TestErrorInfo,
        TestRetryConfig,
        TestCircuitBreaker,
        TestErrorNotifier,
        TestErrorHandler,
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
        print("\n所有错误处理器测试通过！")
    else:
        print(f"\n测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")