"""
系统监控器测试用例
测试系统指标收集、告警管理和监控功能
"""

import unittest
import time
import threading
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.monitoring.system_monitor import (
    SystemMonitor, SystemMetrics, ProcessMetrics, AlertRule, 
    AlertManager, MetricsCollector, MetricsStorage
)


class TestSystemMetrics(unittest.TestCase):
    """系统指标测试"""
    
    def test_system_metrics_creation(self):
        """测试系统指标创建"""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            network_io_sent=1000,
            network_io_recv=2000,
            load_average=1.5
        )
        
        # 验证属性
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.disk_percent, 70.0)
        self.assertEqual(metrics.network_io_sent, 1000)
        self.assertEqual(metrics.network_io_recv, 2000)
        self.assertEqual(metrics.load_average, 1.5)
    
    def test_system_metrics_to_dict(self):
        """测试系统指标转换为字典"""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            network_io_sent=1000,
            network_io_recv=2000,
            load_average=1.5
        )
        
        # 转换为字典
        metrics_dict = metrics.to_dict()
        
        # 验证字典内容
        self.assertEqual(metrics_dict['cpu_percent'], 50.0)
        self.assertEqual(metrics_dict['memory_percent'], 60.0)
        self.assertEqual(metrics_dict['timestamp'], timestamp.isoformat())


class TestProcessMetrics(unittest.TestCase):
    """进程指标测试"""
    
    def test_process_metrics_creation(self):
        """测试进程指标创建"""
        metrics = ProcessMetrics(
            timestamp=datetime.now(),
            process_name="test_process",
            pid=1234,
            cpu_percent=25.0,
            memory_percent=30.0,
            memory_rss=1024*1024,
            status="running",
            num_threads=5
        )
        
        # 验证属性
        self.assertEqual(metrics.process_name, "test_process")
        self.assertEqual(metrics.pid, 1234)
        self.assertEqual(metrics.cpu_percent, 25.0)
        self.assertEqual(metrics.memory_percent, 30.0)
        self.assertEqual(metrics.memory_rss, 1024*1024)
        self.assertEqual(metrics.status, "running")
        self.assertEqual(metrics.num_threads, 5)
    
    def test_process_metrics_to_dict(self):
        """测试进程指标转换为字典"""
        timestamp = datetime.now()
        metrics = ProcessMetrics(
            timestamp=timestamp,
            process_name="test_process",
            pid=1234,
            cpu_percent=25.0,
            memory_percent=30.0,
            memory_rss=1024*1024,
            status="running",
            num_threads=5
        )
        
        # 转换为字典
        metrics_dict = metrics.to_dict()
        
        # 验证字典内容
        self.assertEqual(metrics_dict['process_name'], "test_process")
        self.assertEqual(metrics_dict['pid'], 1234)
        self.assertEqual(metrics_dict['cpu_percent'], 25.0)
        self.assertEqual(metrics_dict['timestamp'], timestamp.isoformat())


class TestAlertRule(unittest.TestCase):
    """告警规则测试"""
    
    def test_alert_rule_creation(self):
        """测试告警规则创建"""
        rule = AlertRule(
            name="高CPU使用率",
            metric="cpu_percent",
            threshold=80.0,
            operator=">",
            severity="high",
            message="CPU使用率过高: {value}%"
        )
        
        # 验证属性
        self.assertEqual(rule.name, "高CPU使用率")
        self.assertEqual(rule.metric, "cpu_percent")
        self.assertEqual(rule.threshold, 80.0)
        self.assertEqual(rule.operator, ">")
        self.assertEqual(rule.severity, "high")
        self.assertEqual(rule.message, "CPU使用率过高: {value}%")
    
    def test_alert_rule_check_greater_than(self):
        """测试大于操作符的告警检查"""
        rule = AlertRule(
            name="高CPU使用率",
            metric="cpu_percent",
            threshold=80.0,
            operator=">",
            severity="high"
        )
        
        # 测试触发告警
        self.assertTrue(rule.check(85.0))
        
        # 测试不触发告警
        self.assertFalse(rule.check(75.0))
        self.assertFalse(rule.check(80.0))
    
    def test_alert_rule_check_less_than(self):
        """测试小于操作符的告警检查"""
        rule = AlertRule(
            name="低磁盘空间",
            metric="disk_free_percent",
            threshold=10.0,
            operator="<",
            severity="critical"
        )
        
        # 测试触发告警
        self.assertTrue(rule.check(5.0))
        
        # 测试不触发告警
        self.assertFalse(rule.check(15.0))
        self.assertFalse(rule.check(10.0))
    
    def test_alert_rule_check_equals(self):
        """测试等于操作符的告警检查"""
        rule = AlertRule(
            name="进程状态检查",
            metric="process_status",
            threshold="stopped",
            operator="==",
            severity="medium"
        )
        
        # 测试触发告警
        self.assertTrue(rule.check("stopped"))
        
        # 测试不触发告警
        self.assertFalse(rule.check("running"))
    
    def test_alert_rule_invalid_operator(self):
        """测试无效操作符"""
        rule = AlertRule(
            name="测试规则",
            metric="test_metric",
            threshold=50.0,
            operator="invalid",
            severity="low"
        )
        
        # 无效操作符应该返回False
        self.assertFalse(rule.check(60.0))


class TestAlertManager(unittest.TestCase):
    """告警管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.alert_manager = AlertManager()
    
    def test_add_rule(self):
        """测试添加告警规则"""
        rule = AlertRule(
            name="测试规则",
            metric="cpu_percent",
            threshold=80.0,
            operator=">",
            severity="high"
        )
        
        # 添加规则
        self.alert_manager.add_rule(rule)
        
        # 验证规则被添加
        self.assertEqual(len(self.alert_manager.rules), 1)
        self.assertEqual(self.alert_manager.rules[0].name, "测试规则")
    
    def test_remove_rule(self):
        """测试移除告警规则"""
        rule = AlertRule(
            name="测试规则",
            metric="cpu_percent",
            threshold=80.0,
            operator=">",
            severity="high"
        )
        
        # 添加规则
        self.alert_manager.add_rule(rule)
        self.assertEqual(len(self.alert_manager.rules), 1)
        
        # 移除规则
        success = self.alert_manager.remove_rule("测试规则")
        
        # 验证规则被移除
        self.assertTrue(success)
        self.assertEqual(len(self.alert_manager.rules), 0)
    
    def test_remove_nonexistent_rule(self):
        """测试移除不存在的规则"""
        success = self.alert_manager.remove_rule("不存在的规则")
        self.assertFalse(success)
    
    def test_check_alerts_no_triggers(self):
        """测试检查告警但没有触发"""
        rule = AlertRule(
            name="高CPU使用率",
            metric="cpu_percent",
            threshold=80.0,
            operator=">",
            severity="high"
        )
        
        self.alert_manager.add_rule(rule)
        
        # 检查告警（CPU使用率正常）
        metrics = {"cpu_percent": 50.0}
        alerts = self.alert_manager.check_alerts(metrics)
        
        # 验证没有告警
        self.assertEqual(len(alerts), 0)
    
    def test_check_alerts_with_triggers(self):
        """测试检查告警并触发"""
        rule = AlertRule(
            name="高CPU使用率",
            metric="cpu_percent",
            threshold=80.0,
            operator=">",
            severity="high",
            message="CPU使用率过高: {value}%"
        )
        
        self.alert_manager.add_rule(rule)
        
        # 检查告警（CPU使用率过高）
        metrics = {"cpu_percent": 90.0}
        alerts = self.alert_manager.check_alerts(metrics)
        
        # 验证有告警
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert['rule_name'], "高CPU使用率")
        self.assertEqual(alert['severity'], "high")
        self.assertEqual(alert['value'], 90.0)
        self.assertIn("CPU使用率过高: 90.0%", alert['message'])
    
    def test_check_multiple_alerts(self):
        """测试检查多个告警"""
        cpu_rule = AlertRule(
            name="高CPU使用率",
            metric="cpu_percent",
            threshold=80.0,
            operator=">",
            severity="high"
        )
        
        memory_rule = AlertRule(
            name="高内存使用率",
            metric="memory_percent",
            threshold=85.0,
            operator=">",
            severity="medium"
        )
        
        self.alert_manager.add_rule(cpu_rule)
        self.alert_manager.add_rule(memory_rule)
        
        # 检查告警（两个指标都超标）
        metrics = {"cpu_percent": 90.0, "memory_percent": 95.0}
        alerts = self.alert_manager.check_alerts(metrics)
        
        # 验证两个告警都被触发
        self.assertEqual(len(alerts), 2)
        alert_names = [alert['rule_name'] for alert in alerts]
        self.assertIn("高CPU使用率", alert_names)
        self.assertIn("高内存使用率", alert_names)


class TestMetricsCollector(unittest.TestCase):
    """指标收集器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.collector = MetricsCollector()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('os.getloadavg')
    def test_collect_system_metrics(self, mock_loadavg, mock_net_io, 
                                   mock_disk, mock_memory, mock_cpu):
        """测试收集系统指标"""
        # 模拟系统指标
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=70.0)
        mock_net_io.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
        mock_loadavg.return_value = (1.5, 1.2, 1.0)
        
        # 收集指标
        metrics = self.collector.collect_system_metrics()
        
        # 验证指标
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.disk_percent, 70.0)
        self.assertEqual(metrics.network_io_sent, 1000)
        self.assertEqual(metrics.network_io_recv, 2000)
        self.assertEqual(metrics.load_average, 1.5)
    
    @patch('psutil.Process')
    def test_collect_process_metrics(self, mock_process_class):
        """测试收集进程指标"""
        # 模拟进程
        mock_process = Mock()
        mock_process.name.return_value = "test_process"
        mock_process.pid = 1234
        mock_process.cpu_percent.return_value = 25.0
        mock_process.memory_percent.return_value = 30.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024)
        mock_process.status.return_value = "running"
        mock_process.num_threads.return_value = 5
        mock_process_class.return_value = mock_process
        
        # 收集指标
        metrics = self.collector.collect_process_metrics(1234)
        
        # 验证指标
        self.assertIsInstance(metrics, ProcessMetrics)
        self.assertEqual(metrics.process_name, "test_process")
        self.assertEqual(metrics.pid, 1234)
        self.assertEqual(metrics.cpu_percent, 25.0)
        self.assertEqual(metrics.memory_percent, 30.0)
        self.assertEqual(metrics.memory_rss, 1024*1024)
        self.assertEqual(metrics.status, "running")
        self.assertEqual(metrics.num_threads, 5)
    
    @patch('psutil.Process')
    def test_collect_process_metrics_not_found(self, mock_process_class):
        """测试收集不存在进程的指标"""
        # 模拟进程不存在
        import psutil
        mock_process_class.side_effect = psutil.NoSuchProcess(1234)
        
        # 收集指标
        metrics = self.collector.collect_process_metrics(1234)
        
        # 验证返回None
        self.assertIsNone(metrics)


class TestMetricsStorage(unittest.TestCase):
    """指标存储器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 使用临时数据库
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.storage = MetricsStorage(self.temp_db.name)
    
    def tearDown(self):
        """清理测试环境"""
        self.storage.close()
        os.unlink(self.temp_db.name)
    
    def test_store_system_metrics(self):
        """测试存储系统指标"""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            network_io_sent=1000,
            network_io_recv=2000,
            load_average=1.5
        )
        
        # 存储指标
        self.storage.store_system_metrics(metrics)
        
        # 验证存储成功（通过查询数据库）
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM system_metrics")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1)
    
    def test_store_process_metrics(self):
        """测试存储进程指标"""
        metrics = ProcessMetrics(
            timestamp=datetime.now(),
            process_name="test_process",
            pid=1234,
            cpu_percent=25.0,
            memory_percent=30.0,
            memory_rss=1024*1024,
            status="running",
            num_threads=5
        )
        
        # 存储指标
        self.storage.store_process_metrics(metrics)
        
        # 验证存储成功
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM process_metrics")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1)
    
    def test_get_system_metrics(self):
        """测试获取系统指标"""
        # 存储一些指标
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_percent=50.0 + i,
                memory_percent=60.0,
                disk_percent=70.0,
                network_io_sent=1000,
                network_io_recv=2000,
                load_average=1.5
            )
            self.storage.store_system_metrics(metrics)
        
        # 获取指标
        retrieved_metrics = self.storage.get_system_metrics(limit=3)
        
        # 验证结果
        self.assertEqual(len(retrieved_metrics), 3)
        # 应该按时间倒序排列
        self.assertGreaterEqual(retrieved_metrics[0]['cpu_percent'], 
                               retrieved_metrics[1]['cpu_percent'])
    
    def test_get_process_metrics(self):
        """测试获取进程指标"""
        # 存储一些指标
        for i in range(3):
            metrics = ProcessMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                process_name="test_process",
                pid=1234,
                cpu_percent=25.0 + i,
                memory_percent=30.0,
                memory_rss=1024*1024,
                status="running",
                num_threads=5
            )
            self.storage.store_process_metrics(metrics)
        
        # 获取指标
        retrieved_metrics = self.storage.get_process_metrics(1234, limit=2)
        
        # 验证结果
        self.assertEqual(len(retrieved_metrics), 2)
        self.assertEqual(retrieved_metrics[0]['pid'], 1234)
    
    def test_cleanup_old_metrics(self):
        """测试清理旧指标"""
        # 存储一些旧指标
        old_time = datetime.now() - timedelta(days=10)
        for i in range(3):
            metrics = SystemMetrics(
                timestamp=old_time - timedelta(minutes=i),
                cpu_percent=50.0,
                memory_percent=60.0,
                disk_percent=70.0,
                network_io_sent=1000,
                network_io_recv=2000,
                load_average=1.5
            )
            self.storage.store_system_metrics(metrics)
        
        # 清理旧指标（保留7天）
        deleted_count = self.storage.cleanup_old_metrics(days=7)
        
        # 验证清理结果
        self.assertEqual(deleted_count, 3)


class TestSystemMonitor(unittest.TestCase):
    """系统监控器主类测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 使用临时数据库
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.monitor = SystemMonitor(
            collect_interval=0.1,  # 快速收集用于测试
            storage_path=self.temp_db.name
        )
    
    def tearDown(self):
        """清理测试环境"""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
        os.unlink(self.temp_db.name)
    
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        # 初始状态
        self.assertFalse(self.monitor.monitoring)
        
        # 启动监控
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring)
        
        # 等待一段时间让监控收集数据
        time.sleep(0.3)
        
        # 停止监控
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring)
    
    def test_add_process_to_monitor(self):
        """测试添加监控进程"""
        # 添加当前进程到监控
        import os
        current_pid = os.getpid()
        
        self.monitor.add_process_to_monitor(current_pid)
        
        # 验证进程被添加
        self.assertIn(current_pid, self.monitor.monitored_processes)
    
    def test_remove_process_from_monitor(self):
        """测试移除监控进程"""
        import os
        current_pid = os.getpid()
        
        # 先添加进程
        self.monitor.add_process_to_monitor(current_pid)
        self.assertIn(current_pid, self.monitor.monitored_processes)
        
        # 移除进程
        self.monitor.remove_process_from_monitor(current_pid)
        self.assertNotIn(current_pid, self.monitor.monitored_processes)
    
    def test_add_alert_rule(self):
        """测试添加告警规则"""
        rule = AlertRule(
            name="测试规则",
            metric="cpu_percent",
            threshold=80.0,
            operator=">",
            severity="high"
        )
        
        # 添加规则
        self.monitor.add_alert_rule(rule)
        
        # 验证规则被添加
        self.assertEqual(len(self.monitor.alert_manager.rules), 1)
    
    @patch('smtplib.SMTP')
    def test_configure_email_alerts(self, mock_smtp):
        """测试配置邮件告警"""
        email_config = {
            'smtp_server': 'smtp.example.com',
            'smtp_port': 587,
            'username': 'test@example.com',
            'password': 'password',
            'from_email': 'test@example.com',
            'to_emails': ['admin@example.com']
        }
        
        # 配置邮件告警
        self.monitor.configure_email_alerts(email_config)
        
        # 验证配置被设置
        self.assertIsNotNone(self.monitor.email_config)
        self.assertEqual(self.monitor.email_config['smtp_server'], 'smtp.example.com')
    
    def test_configure_webhook_alerts(self):
        """测试配置Webhook告警"""
        webhook_config = {
            'url': 'https://hooks.slack.com/test',
            'headers': {'Content-Type': 'application/json'}
        }
        
        # 配置Webhook告警
        self.monitor.configure_webhook_alerts(webhook_config)
        
        # 验证配置被设置
        self.assertIsNotNone(self.monitor.webhook_config)
        self.assertEqual(self.monitor.webhook_config['url'], 'https://hooks.slack.com/test')
    
    def test_get_current_status(self):
        """测试获取当前状态"""
        # 获取当前状态
        status = self.monitor.get_current_status()
        
        # 验证状态信息
        self.assertIn('system_metrics', status)
        self.assertIn('process_metrics', status)
        self.assertIn('monitoring_status', status)
        
        # 验证系统指标
        system_metrics = status['system_metrics']
        self.assertIn('cpu_percent', system_metrics)
        self.assertIn('memory_percent', system_metrics)
        self.assertIn('disk_percent', system_metrics)
    
    def test_get_metrics_history(self):
        """测试获取指标历史"""
        # 启动监控收集一些数据
        self.monitor.start_monitoring()
        time.sleep(0.3)
        self.monitor.stop_monitoring()
        
        # 获取指标历史
        history = self.monitor.get_metrics_history(hours=1)
        
        # 验证历史数据
        self.assertIn('system_metrics', history)
        self.assertIn('process_metrics', history)
        
        # 应该有一些系统指标数据
        system_metrics = history['system_metrics']
        self.assertGreater(len(system_metrics), 0)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_monitoring_workflow(self):
        """测试完整的监控工作流"""
        # 使用临时数据库
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        try:
            # 创建监控器
            monitor = SystemMonitor(
                collect_interval=0.1,
                storage_path=temp_db.name
            )
            
            # 添加告警规则
            cpu_rule = AlertRule(
                name="高CPU使用率",
                metric="cpu_percent",
                threshold=0.1,  # 设置很低的阈值确保触发
                operator=">",
                severity="high",
                message="CPU使用率过高: {value}%"
            )
            monitor.add_alert_rule(cpu_rule)
            
            # 添加当前进程到监控
            import os
            current_pid = os.getpid()
            monitor.add_process_to_monitor(current_pid)
            
            # 启动监控
            monitor.start_monitoring()
            
            # 等待收集数据
            time.sleep(0.5)
            
            # 获取当前状态
            status = monitor.get_current_status()
            
            # 验证状态完整性
            self.assertIn('system_metrics', status)
            self.assertIn('process_metrics', status)
            self.assertIn('monitoring_status', status)
            
            # 验证系统指标
            system_metrics = status['system_metrics']
            self.assertIsInstance(system_metrics['cpu_percent'], (int, float))
            self.assertIsInstance(system_metrics['memory_percent'], (int, float))
            
            # 验证进程指标
            process_metrics = status['process_metrics']
            self.assertGreater(len(process_metrics), 0)
            
            # 获取指标历史
            history = monitor.get_metrics_history(hours=1)
            self.assertIn('system_metrics', history)
            self.assertGreater(len(history['system_metrics']), 0)
            
            # 停止监控
            monitor.stop_monitoring()
            
        finally:
            # 清理
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestSystemMetrics,
        TestProcessMetrics,
        TestAlertRule,
        TestAlertManager,
        TestMetricsCollector,
        TestMetricsStorage,
        TestSystemMonitor,
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
        print("\n所有系统监控器测试通过！")
    else:
        print(f"\n测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")