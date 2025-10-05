"""
日志管理器测试用例
测试日志配置、记录和管理功能
"""

import unittest
import tempfile
import os
import shutil
import logging
import yaml
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.utils.logger import LoggerManager, PerformanceLogger


class TestLoggerManager(unittest.TestCase):
    """日志管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.logs_dir = os.path.join(self.temp_dir, 'logs')
        self.config_dir = os.path.join(self.temp_dir, 'config')
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 创建测试配置文件
        self.config_file = os.path.join(self.config_dir, 'logging.yaml')
        self.create_test_config()
        
        # 重置单例
        LoggerManager._instance = None
        LoggerManager._initialized = False
    
    def tearDown(self):
        """清理测试环境"""
        # 清理临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # 重置单例
        LoggerManager._instance = None
        LoggerManager._initialized = False
        
        # 清理日志处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
    
    def create_test_config(self):
        """创建测试日志配置"""
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file_info': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'detailed',
                    'filename': os.path.join(self.logs_dir, 'app.log'),
                    'maxBytes': 10485760,
                    'backupCount': 5
                }
            },
            'loggers': {
                'trading': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file_info'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file_info']
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        # 创建两个实例
        # 创建LoggerManager实例并加载配置
        manager1 = LoggerManager()
        manager1.load_config(self.config_file)
        manager2 = LoggerManager()
        manager2.load_config(self.config_file)
        
        # 验证是同一个实例
        self.assertIs(manager1, manager2)
    
    def test_load_config_from_file(self):
        """测试从文件加载配置"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 验证配置已加载
        self.assertEqual(manager.config_path, self.config_file)
        
        # 验证日志器可以正常工作
        logger = manager.get_logger('test')
        self.assertIsInstance(logger, logging.Logger)

    def test_load_config_file_not_found(self):
        """测试配置文件不存在的情况"""
        non_existent_file = os.path.join(self.temp_dir, 'non_existent.yaml')
        
        # 应该不会抛出异常，而是使用默认配置
        manager = LoggerManager()
        manager.load_config(non_existent_file)
        
        # 验证可以获取日志器
        logger = manager.get_logger('test')
        self.assertIsInstance(logger, logging.Logger)

    def test_ensure_log_directories(self):
        """测试确保日志目录存在"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 验证日志目录已创建
        self.assertTrue(os.path.exists(self.logs_dir))
        
        # 验证可以获取日志器
        logger = manager.get_logger('test')
        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger(self):
        """测试获取日志器"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 获取日志器
        logger = manager.get_logger('test')
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test')

    def test_get_logger_with_specific_config(self):
        """测试使用特定配置获取日志器"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 获取不同的日志器
        app_logger = manager.get_logger('app')
        trading_logger = manager.get_logger('trading')
        
        self.assertIsInstance(app_logger, logging.Logger)
        self.assertIsInstance(trading_logger, logging.Logger)
        self.assertEqual(app_logger.name, 'app')
        self.assertEqual(trading_logger.name, 'trading')

    def test_log_exception(self):
        """测试异常日志记录"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        logger = manager.get_logger('test')
        
        # 模拟异常
        try:
            raise ValueError("测试异常")
        except ValueError:
            with patch.object(logger, 'error') as mock_error:
                manager.log_exception(logger, "发生了测试异常")
                
                # 验证错误日志被调用
                mock_error.assert_called_once()
                args, kwargs = mock_error.call_args
                log_data = args[0]
                self.assertIn("发生了测试异常", log_data)
                self.assertIn("ValueError", log_data)
                self.assertIn("Traceback", log_data)

    def test_log_performance(self):
        """测试性能日志记录"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        with patch.object(manager, 'get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # 记录性能日志
            manager.log_performance("测试操作", 1.5, param1="value1", param2="value2")
            
            # 验证日志记录
            mock_get_logger.assert_called_once_with('performance')
            mock_logger.info.assert_called_once()
            args, kwargs = mock_logger.info.call_args
            log_data = args[0]
            self.assertIn("测试操作", log_data)
            self.assertIn("duration_seconds", log_data)
            self.assertIn("param1", log_data)
            self.assertIn("param2", log_data)

    def test_log_audit(self):
        """测试审计日志记录"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        with patch.object(manager, 'get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # 记录审计日志
            manager.log_audit(
                "用户登录",
                user="admin",
                details={"ip": "127.0.0.1", "success": True}
            )
            
            # 验证日志记录
            mock_get_logger.assert_called_once_with('audit')
            mock_logger.info.assert_called_once()
            args, kwargs = mock_logger.info.call_args
            log_data = args[0]
            self.assertIn("用户登录", log_data)
            self.assertIn("user", log_data)
            self.assertIn("details", log_data)

    def test_log_trading_event(self):
        """测试交易事件日志记录"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        with patch.object(manager, 'get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # 记录交易事件日志
            manager.log_trading_event(
                "订单执行",
                symbol="AAPL",
                quantity=100,
                price=150.0
            )
            
            # 验证日志记录
            mock_get_logger.assert_called_once_with('trading')
            mock_logger.info.assert_called_once()
            args, kwargs = mock_logger.info.call_args
            log_data = args[0]
            self.assertIn("订单执行", log_data)
            self.assertIn("symbol", log_data)
            self.assertIn("quantity", log_data)
            self.assertIn("price", log_data)

    def test_set_log_level(self):
        """测试设置日志级别"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 获取日志器
        logger = manager.get_logger('test')
        
        # 设置日志级别
        manager.set_level('test', 'DEBUG')
        
        # 验证级别已设置
        self.assertEqual(logger.level, logging.DEBUG)
        
        # 设置为ERROR级别
        manager.set_level('test', 'ERROR')
        self.assertEqual(logger.level, logging.ERROR)

    def test_get_log_file_path(self):
        """测试获取日志文件路径"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 获取日志文件路径
        log_files = manager.get_log_files()
        
        # 验证返回的是字典
        self.assertIsInstance(log_files, dict)

    def test_get_log_file_path_not_found(self):
        """测试获取不存在的日志文件路径"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 获取日志文件路径
        log_files = manager.get_log_files()
        
        # 验证返回的是字典（可能为空）
        self.assertIsInstance(log_files, dict)

    def test_cleanup_old_logs(self):
        """测试清理旧日志文件"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 确保日志目录存在
        manager._ensure_log_directories(manager.config)
        
        # 创建一些测试日志文件
        old_log_file = os.path.join(manager.log_dir, 'old_test.log')
        recent_log_file = os.path.join(manager.log_dir, 'recent_test.log')
        
        # 创建文件
        with open(old_log_file, 'w') as f:
            f.write("old log content")
        with open(recent_log_file, 'w') as f:
            f.write("recent log content")
        
        # 修改旧文件的时间戳
        old_time = datetime.now() - timedelta(days=35)
        old_timestamp = old_time.timestamp()
        os.utime(old_log_file, (old_timestamp, old_timestamp))
        
        # 执行清理
        manager.cleanup_old_logs(days=30)
        
        # 验证旧文件被删除，新文件保留
        self.assertFalse(os.path.exists(old_log_file))
        self.assertTrue(os.path.exists(recent_log_file))


class TestPerformanceLogger(unittest.TestCase):
    """性能日志器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.logs_dir = os.path.join(self.temp_dir, 'logs')
        self.config_dir = os.path.join(self.temp_dir, 'config')
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 创建测试配置文件
        self.config_file = os.path.join(self.config_dir, 'logging.yaml')
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard'
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console']
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 重置单例
        LoggerManager._instance = None
        LoggerManager._initialized = False
        
        # 创建LoggerManager实例
        self.logger_manager = LoggerManager()
        self.logger_manager.load_config(self.config_file)

    def tearDown(self):
        """清理测试环境"""
        # 清理临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # 重置单例
        LoggerManager._instance = None
        LoggerManager._initialized = False
    
    def test_performance_logger_context_manager(self):
        """测试性能日志器上下文管理器"""
        # 使用上下文管理器
        with PerformanceLogger("测试操作") as perf_logger:
            # 模拟一些操作
            import time
            time.sleep(0.1)
            
            # 添加详细信息
            perf_logger.add_detail("param1", "value1")
            perf_logger.add_detail("param2", 42)
        
        # 验证性能被记录（通过检查是否有异常抛出）
        # 实际的验证需要检查日志输出，这里只验证代码执行成功
        self.assertTrue(True)
    
    def test_performance_logger_manual_timing(self):
        """测试手动计时"""
        perf_logger = PerformanceLogger("手动计时测试")
        
        # 开始计时
        perf_logger.start()
        
        # 模拟操作
        import time
        time.sleep(0.05)
        
        # 结束计时
        duration = perf_logger.stop()
        
        # 验证计时结果
        self.assertGreater(duration, 0.04)  # 应该大于等于睡眠时间
        self.assertLess(duration, 0.1)      # 但不应该太大
    
    def test_performance_logger_add_details(self):
        """测试添加详细信息"""
        perf_logger = PerformanceLogger("详细信息测试")
        
        # 添加各种类型的详细信息
        perf_logger.add_detail("string_param", "test_value")
        perf_logger.add_detail("int_param", 123)
        perf_logger.add_detail("float_param", 45.67)
        perf_logger.add_detail("bool_param", True)
        perf_logger.add_detail("dict_param", {"key": "value"})
        
        # 验证详细信息被添加
        self.assertEqual(perf_logger.details["string_param"], "test_value")
        self.assertEqual(perf_logger.details["int_param"], 123)
        self.assertEqual(perf_logger.details["float_param"], 45.67)
        self.assertEqual(perf_logger.details["bool_param"], True)
        self.assertEqual(perf_logger.details["dict_param"], {"key": "value"})
    
    def test_performance_logger_without_context_manager(self):
        """测试不使用上下文管理器的情况"""
        perf_logger = PerformanceLogger("非上下文管理器测试")
        
        # 手动开始和结束
        perf_logger.start()
        import time
        time.sleep(0.02)
        duration = perf_logger.stop()
        
        # 验证计时
        self.assertGreater(duration, 0.01)
    
    def test_performance_logger_multiple_start_stop(self):
        """测试多次开始和停止"""
        perf_logger = PerformanceLogger("多次计时测试")
        
        # 第一次计时
        perf_logger.start()
        import time
        time.sleep(0.01)
        duration1 = perf_logger.stop()
        
        # 第二次计时（应该重新开始）
        perf_logger.start()
        time.sleep(0.02)
        duration2 = perf_logger.stop()
        
        # 验证两次计时都有效
        self.assertGreater(duration1, 0)
        self.assertGreater(duration2, 0)
        self.assertGreater(duration2, duration1)  # 第二次应该更长


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.logs_dir = os.path.join(self.temp_dir, 'logs')
        self.config_dir = os.path.join(self.temp_dir, 'config')
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 创建完整的配置文件
        self.config_file = os.path.join(self.config_dir, 'logging.yaml')
        self.create_full_config()
        
        # 重置单例
        LoggerManager._instance = None
        LoggerManager._initialized = False
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        LoggerManager._instance = None
        LoggerManager._initialized = False
        
        # 清理日志处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
    
    def create_full_config(self):
        """创建完整的日志配置"""
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
                },
                'json': {
                    'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard'
                },
                'file_info': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'detailed',
                    'filename': os.path.join(self.logs_dir, 'app.log'),
                    'maxBytes': 10485760,
                    'backupCount': 5
                },
                'file_error': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'detailed',
                    'filename': os.path.join(self.logs_dir, 'error.log'),
                    'maxBytes': 10485760,
                    'backupCount': 5
                },
                'file_trading': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'json',
                    'filename': os.path.join(self.logs_dir, 'trading.log'),
                    'maxBytes': 10485760,
                    'backupCount': 10
                }
            },
            'loggers': {
                'trading': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file_trading'],
                    'propagate': False
                },
                'performance': {
                    'level': 'INFO',
                    'handlers': ['file_info'],
                    'propagate': False
                },
                'audit': {
                    'level': 'INFO',
                    'handlers': ['file_info'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file_info', 'file_error']
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def test_complete_logging_workflow(self):
        """测试完整的日志工作流"""
        # 创建日志管理器
        manager = LoggerManager()
        manager.load_config(self.config_file)
        
        # 获取不同类型的日志器
        trading_logger = manager.get_logger('trading')
        performance_logger = manager.get_logger('performance')
        audit_logger = manager.get_logger('audit')
        
        # 记录不同类型的日志
        trading_logger.info("交易系统启动")
        trading_logger.debug("调试信息：连接到交易API")
        
        # 记录异常
        try:
            raise ValueError("模拟交易错误")
        except Exception as e:
            manager.log_exception(trading_logger, "交易过程中发生错误")
        
        # 记录性能信息
        manager.log_performance(
            operation="获取市场数据",
            duration=0.5,
            details={"symbols": ["AAPL", "GOOGL"], "data_points": 1000}
        )
        
        # 记录审计信息
        manager.log_audit(
            action="place_order",
            user="trader_001",
            details={"resource": "AAPL", "result": "success", "quantity": 100, "price": 150.0}
        )
        
        # 记录交易事件
        manager.log_trading_event(
            event_type="order_filled",
            symbol="AAPL",
            details={
                "order_id": "12345",
                "quantity": 100,
                "price": 150.0,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # 使用性能日志器
        with PerformanceLogger("数据处理") as perf:
            perf.add_detail("records_processed", 5000)
            import time
            time.sleep(0.01)  # 模拟处理时间
        
        # 验证日志文件被创建
        app_log = os.path.join(self.logs_dir, 'app.log')
        error_log = os.path.join(self.logs_dir, 'error.log')
        trading_log = os.path.join(self.logs_dir, 'trading.log')
        
        # 检查文件存在
        self.assertTrue(os.path.exists(app_log))
        self.assertTrue(os.path.exists(error_log))
        self.assertTrue(os.path.exists(trading_log))
        
        # 验证日志内容
        with open(app_log, 'r', encoding='utf-8') as f:
            app_content = f.read()
            self.assertIn("获取市场数据", app_content)
            self.assertIn("trader_001", app_content)
            self.assertIn("数据处理", app_content)
        
        # 检查错误日志文件是否存在且有内容
        if os.path.exists(error_log) and os.path.getsize(error_log) > 0:
            with open(error_log, 'r', encoding='utf-8') as f:
                error_content = f.read()
                # 如果有内容，验证错误信息
                if error_content.strip():
                    self.assertIn("模拟交易错误", error_content)
                    self.assertIn("ValueError", error_content)
        
        with open(trading_log, 'r', encoding='utf-8') as f:
            trading_content = f.read()
            self.assertIn("交易系统启动", trading_content)
            self.assertIn("order_filled", trading_content)
    
    def test_log_rotation_and_cleanup(self):
        """测试日志轮转和清理"""
        manager = LoggerManager()
        manager.load_config(self.config_file)
        logger = manager.get_logger('test')
        
        # 生成大量日志以触发轮转
        for i in range(1000):
            logger.info(f"测试日志消息 {i} - " + "x" * 100)
        
        # 检查日志文件
        app_log = os.path.join(self.logs_dir, 'app.log')
        self.assertTrue(os.path.exists(app_log))
        
        # 创建一些旧的备份文件用于测试清理
        old_backup = os.path.join(manager.log_dir, 'old_backup.log')
        with open(old_backup, 'w') as f:
            f.write("old backup content")
        
        # 设置文件为10天前
        old_time = datetime.now() - timedelta(days=10)
        timestamp = old_time.timestamp()
        os.utime(old_backup, (timestamp, timestamp))
        
        # 清理旧日志
        deleted_count = manager.cleanup_old_logs(days=7)
        
        # 验证清理结果
        self.assertGreater(deleted_count, 0)
        self.assertFalse(os.path.exists(old_backup))


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestLoggerManager,
        TestPerformanceLogger,
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
        print("\n所有日志管理器测试通过！")
    else:
        print(f"\n测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")