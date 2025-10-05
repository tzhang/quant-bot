#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firstrade交易系统测试脚本

该脚本用于测试Firstrade交易系统的各项功能，包括：
- 连接和登录测试
- 账户信息获取测试
- 市场数据获取测试
- 订单执行测试（模拟模式）
- 投资组合分析测试
- 风险管理测试
- 错误处理测试

使用方法:
python test_firstrade_system.py
"""

import unittest
import sys
import os
import json
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试目标
try:
    from firstrade_trading_system import (
        FirstradeConnector, 
        FirstradeOrderExecutor, 
        FirstradeTradingSystem,
        RiskManager,
        SecurityConfig,
        SecurityManager,
        ErrorHandler,
        CircuitBreaker,
        RetryConfig
    )
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖模块都已正确安装")
    sys.exit(1)

class TestFirstradeConnector(unittest.TestCase):
    """测试FirstradeConnector类"""
    
    def setUp(self):
        """设置测试环境"""
        self.connector = FirstradeConnector("test_user", "test_pass", "1234")
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.connector.username, "test_user")
        self.assertEqual(self.connector.password, "test_pass")
        self.assertEqual(self.connector.pin, "1234")
        self.assertFalse(self.connector.is_logged_in)
        self.assertIsNotNone(self.connector.error_handler)
        self.assertIsNotNone(self.connector.circuit_breaker)
        
    @patch('firstrade_trading_system.Firstrade')
    def test_login_success(self, mock_firstrade):
        """测试成功登录"""
        # 模拟Firstrade实例
        mock_ft = Mock()
        mock_ft.get_account.return_value = {'account_number': '12345678'}
        mock_firstrade.return_value = mock_ft
        
        # 测试登录
        result = self.connector.login()
        
        self.assertTrue(result)
        self.assertTrue(self.connector.is_logged_in)
        
    @patch('firstrade_trading_system.Firstrade')
    def test_login_failure(self, mock_firstrade):
        """测试登录失败"""
        # 模拟登录异常
        mock_firstrade.side_effect = Exception("登录失败")
        
        # 测试登录
        result = self.connector.login()
        
        self.assertFalse(result)
        self.assertFalse(self.connector.is_logged_in)
        
    def test_get_quote_not_logged_in(self):
        """测试未登录时获取报价"""
        result = self.connector.get_quote("AAPL")
        self.assertEqual(result, {})


class TestFirstradeOrderExecutor(unittest.TestCase):
    """测试FirstradeOrderExecutor类"""
    
    def setUp(self):
        """设置测试环境"""
        self.connector = Mock()
        self.connector.is_logged_in = True
        self.connector.ft = Mock()
        self.executor = FirstradeOrderExecutor(self.connector)
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.executor.connector, self.connector)
        
    def test_place_market_order_dry_run(self):
        """测试模拟市价单"""
        result = self.executor.place_market_order("AAPL", 100, "buy", dry_run=True)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['order_type'], 'market')
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['quantity'], 100)
        self.assertEqual(result['side'], 'buy')
        
    def test_place_limit_order_dry_run(self):
        """测试模拟限价单"""
        result = self.executor.place_limit_order("AAPL", 100, "buy", 150.0, dry_run=True)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['order_type'], 'limit')
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['quantity'], 100)
        self.assertEqual(result['side'], 'buy')
        self.assertEqual(result['price'], 150.0)
        
    def test_calculate_position_size(self):
        """测试仓位大小计算"""
        # 模拟股票价格
        self.connector.get_quote.return_value = {'price': 150.0}
        
        result = self.executor.calculate_position_size("AAPL", 0.1, 100000)
        
        # 期望结果：100000 * 0.1 / 150 = 66.67，向下取整为66
        self.assertEqual(result, 66)


class TestFirstradeTradingSystem(unittest.TestCase):
    """测试FirstradeTradingSystem类"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = FirstradeTradingSystem("test_user", "test_pass", "1234", dry_run=True)
        
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.system.connector)
        self.assertIsNotNone(self.system.executor)
        self.assertTrue(self.system.dry_run)
        
    @patch.object(FirstradeConnector, 'login')
    def test_connect_success(self, mock_login):
        """测试连接成功"""
        mock_login.return_value = True
        
        result = self.system.connect()
        
        self.assertTrue(result)
        
    @patch.object(FirstradeConnector, 'login')
    def test_connect_failure(self, mock_login):
        """测试连接失败"""
        mock_login.return_value = False
        
        result = self.system.connect()
        
        self.assertFalse(result)


class TestRiskManager(unittest.TestCase):
    """测试RiskManager类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = SecurityConfig()
        self.risk_manager = RiskManager(self.config)
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.risk_manager.config, self.config)
        self.assertEqual(self.risk_manager.daily_trade_count, 0)
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
        
    def test_validate_symbol_allowed(self):
        """测试允许的股票代码验证"""
        self.config.allowed_symbols = ["AAPL", "GOOGL", "MSFT"]
        
        is_valid, message = self.risk_manager.validate_symbol("AAPL")
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "")
        
    def test_validate_symbol_forbidden(self):
        """测试禁止的股票代码验证"""
        self.config.forbidden_symbols = ["TSLA", "GME"]
        
        is_valid, message = self.risk_manager.validate_symbol("TSLA")
        
        self.assertFalse(is_valid)
        self.assertIn("禁止交易", message)
        
    def test_validate_order_size_too_large(self):
        """测试订单金额过大"""
        is_valid, message = self.risk_manager.validate_order_size(
            "AAPL", 1000, 150.0, 100000
        )
        
        # 订单金额：1000 * 150 = 150000，超过最大订单金额50000
        self.assertFalse(is_valid)
        self.assertIn("超过最大限制", message)
        
    def test_check_daily_limits_exceeded(self):
        """测试每日交易限制"""
        # 设置已达到每日交易限制
        self.risk_manager.daily_trade_count = self.config.max_daily_trades
        
        is_valid, message = self.risk_manager.check_daily_limits()
        
        self.assertFalse(is_valid)
        self.assertIn("已达到每日最大交易次数", message)
        
    def test_check_stop_loss(self):
        """测试止损检查"""
        # 当前价格135，平均成本150，跌幅10%，超过8%止损阈值
        should_stop, message = self.risk_manager.check_stop_loss("AAPL", 135.0, 150.0)
        
        self.assertTrue(should_stop)
        self.assertIn("触发止损", message)
        
    def test_check_take_profit(self):
        """测试止盈检查"""
        # 当前价格180，平均成本150，涨幅20%，达到20%止盈阈值
        should_take, message = self.risk_manager.check_take_profit("AAPL", 180.0, 150.0)
        
        self.assertTrue(should_take)
        self.assertIn("触发止盈", message)


class TestSecurityManager(unittest.TestCase):
    """测试SecurityManager类"""
    
    def setUp(self):
        """设置测试环境"""
        self.security_manager = SecurityManager()
        
    def test_generate_session_token(self):
        """测试会话令牌生成"""
        token = self.security_manager.generate_session_token()
        
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 0)
        
    def test_validate_session_token_valid(self):
        """测试有效会话令牌验证"""
        token = self.security_manager.generate_session_token()
        
        is_valid = self.security_manager.validate_session_token(token)
        
        self.assertTrue(is_valid)
        
    def test_validate_session_token_expired(self):
        """测试过期会话令牌验证"""
        # 生成一个过期的令牌（修改过期时间）
        token = self.security_manager.generate_session_token()
        
        # 模拟令牌过期（修改内部状态）
        if token in self.security_manager.active_tokens:
            self.security_manager.active_tokens[token] = datetime.now() - timedelta(hours=2)
        
        is_valid = self.security_manager.validate_session_token(token)
        
        self.assertFalse(is_valid)
        
    def test_hash_credentials(self):
        """测试凭据哈希"""
        hash1 = self.security_manager.hash_credentials("user1", "pass1")
        hash2 = self.security_manager.hash_credentials("user1", "pass1")
        hash3 = self.security_manager.hash_credentials("user2", "pass1")
        
        # 相同凭据应产生相同哈希
        self.assertEqual(hash1, hash2)
        # 不同凭据应产生不同哈希
        self.assertNotEqual(hash1, hash3)


class TestErrorHandler(unittest.TestCase):
    """测试ErrorHandler类"""
    
    def setUp(self):
        """设置测试环境"""
        self.error_handler = ErrorHandler()
        
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.error_handler.retry_config)
        self.assertEqual(self.error_handler.error_counts, {})
        
    def test_handle_api_error_recoverable(self):
        """测试可恢复的API错误"""
        error = Exception("网络超时")
        
        should_retry = self.error_handler.handle_api_error(error, "测试操作")
        
        self.assertTrue(should_retry)
        
    def test_execute_with_retry_success(self):
        """测试重试机制成功"""
        def success_func():
            return "成功"
        
        result = self.error_handler.execute_with_retry(success_func, "测试操作")
        
        self.assertEqual(result, "成功")
        
    def test_execute_with_retry_failure(self):
        """测试重试机制失败"""
        def failure_func():
            raise Exception("持续失败")
        
        with self.assertRaises(Exception):
            self.error_handler.execute_with_retry(failure_func, "测试操作")


class TestCircuitBreaker(unittest.TestCase):
    """测试CircuitBreaker类"""
    
    def setUp(self):
        """设置测试环境"""
        self.circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.circuit_breaker.failure_threshold, 2)
        self.assertEqual(self.circuit_breaker.recovery_timeout, 1)
        self.assertEqual(self.circuit_breaker.state, "CLOSED")
        
    def test_call_success(self):
        """测试成功调用"""
        def success_func():
            return "成功"
        
        result = self.circuit_breaker.call(success_func)
        
        self.assertEqual(result, "成功")
        self.assertEqual(self.circuit_breaker.state, "CLOSED")
        
    def test_call_failure_opens_circuit(self):
        """测试失败调用打开熔断器"""
        def failure_func():
            raise Exception("失败")
        
        # 连续失败直到熔断器打开
        for _ in range(3):
            try:
                self.circuit_breaker.call(failure_func)
            except:
                pass
        
        self.assertEqual(self.circuit_breaker.state, "OPEN")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = FirstradeTradingSystem("test_user", "test_pass", "1234", dry_run=True)
        
    @patch.object(FirstradeConnector, 'login')
    @patch.object(FirstradeConnector, 'get_account_info')
    def test_full_workflow_dry_run(self, mock_get_account, mock_login):
        """测试完整工作流程（模拟模式）"""
        # 模拟登录成功
        mock_login.return_value = True
        mock_get_account.return_value = {
            'account_number': '12345678',
            'buying_power': 50000,
            'total_value': 100000
        }
        
        # 测试连接
        connect_result = self.system.connect()
        self.assertTrue(connect_result)
        
        # 测试获取账户信息
        account_info = self.system.get_account_balance()
        self.assertIn('account_number', account_info)
        
        # 测试模拟交易
        order_result = self.system.executor.place_market_order("AAPL", 10, "buy", dry_run=True)
        self.assertTrue(order_result['success'])


def run_performance_test():
    """运行性能测试"""
    print("\n" + "="*50)
    print("性能测试")
    print("="*50)
    
    import time
    
    # 测试连接器初始化性能
    start_time = time.time()
    for _ in range(100):
        connector = FirstradeConnector("test", "test", "1234")
    init_time = time.time() - start_time
    print(f"连接器初始化 (100次): {init_time:.4f}秒")
    
    # 测试风险管理器性能
    config = SecurityConfig()
    risk_manager = RiskManager(config)
    
    start_time = time.time()
    for _ in range(1000):
        risk_manager.validate_symbol("AAPL")
    validation_time = time.time() - start_time
    print(f"股票代码验证 (1000次): {validation_time:.4f}秒")
    
    # 测试错误处理器性能
    error_handler = ErrorHandler()
    
    def dummy_func():
        return "success"
    
    start_time = time.time()
    for _ in range(100):
        error_handler.execute_with_retry(dummy_func, "测试")
    retry_time = time.time() - start_time
    print(f"重试机制执行 (100次): {retry_time:.4f}秒")


def run_stress_test():
    """运行压力测试"""
    print("\n" + "="*50)
    print("压力测试")
    print("="*50)
    
    # 测试大量并发订单处理
    system = FirstradeTradingSystem("test", "test", "1234", dry_run=True)
    
    orders = []
    for i in range(100):
        orders.append({
            'symbol': f'TEST{i:03d}',
            'quantity': 10,
            'side': 'buy',
            'order_type': 'market'
        })
    
    start_time = time.time()
    results = system.executor.execute_batch_orders(orders, dry_run=True)
    batch_time = time.time() - start_time
    
    print(f"批量处理100个订单: {batch_time:.4f}秒")
    print(f"成功订单数: {len([r for r in results if r.get('success')])}")
    
    # 测试风险管理器在高频交易下的性能
    config = SecurityConfig()
    risk_manager = RiskManager(config)
    
    start_time = time.time()
    for i in range(1000):
        risk_manager.get_risk_assessment(
            f"TEST{i%10:03d}", 10, 100.0, [], 100000
        )
    risk_time = time.time() - start_time
    
    print(f"风险评估 (1000次): {risk_time:.4f}秒")


def main():
    """主函数"""
    print("Firstrade交易系统测试")
    print("="*50)
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    # 运行单元测试
    print("运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行性能测试
    run_performance_test()
    
    # 运行压力测试
    run_stress_test()
    
    print("\n" + "="*50)
    print("测试完成！")
    print("="*50)


if __name__ == "__main__":
    main()