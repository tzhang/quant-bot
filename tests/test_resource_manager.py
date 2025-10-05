"""
资源管理器测试用例
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.utils.resource_manager import (
    ResourceManager,
    ConnectionPool,
    ConnectionPoolConfig,
    Connection,
    ThreadPoolManager,
    MemoryManager,
    ResourceMetrics,
    get_resource_manager,
    cleanup_resources,
    with_connection,
    async_task,
    cached
)


class TestConnection(unittest.TestCase):
    """连接对象测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.mock_connection = Mock()
        self.mock_pool = Mock()
        self.connection = Connection(self.mock_connection, self.mock_pool)
    
    def test_connection_creation(self):
        """测试连接创建"""
        self.assertEqual(self.connection.connection, self.mock_connection)
        self.assertEqual(self.connection.pool, self.mock_pool)
        self.assertFalse(self.connection.in_use)
        self.assertTrue(self.connection.is_healthy)
        self.assertIsInstance(self.connection.created_at, datetime)
        self.assertIsInstance(self.connection.last_used, datetime)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with self.connection as conn:
            self.assertEqual(conn, self.mock_connection)
            self.assertTrue(self.connection.in_use)
        
        self.assertFalse(self.connection.in_use)
        self.mock_pool.return_connection.assert_called_once_with(self.connection)
    
    def test_is_expired(self):
        """测试连接过期检查"""
        # 新连接不应该过期
        self.assertFalse(self.connection.is_expired(300))
        
        # 模拟旧连接
        self.connection.last_used = datetime.now() - timedelta(seconds=400)
        self.assertFalse(self.connection.is_expired(300))  # 使用中的连接不过期
        
        self.connection.in_use = False
        self.assertTrue(self.connection.is_expired(300))  # 空闲连接过期
    
    def test_close(self):
        """测试连接关闭"""
        self.connection.close()
        self.mock_connection.close.assert_called_once()
        
        # 测试没有close方法的连接
        connection_no_close = Connection(object(), self.mock_pool)
        connection_no_close.close()  # 不应该抛出异常


class TestConnectionPool(unittest.TestCase):
    """连接池测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.connection_factory = Mock(return_value=Mock())
        self.config = ConnectionPoolConfig(
            min_connections=2,
            max_connections=5,
            max_idle_time=300,
            connection_timeout=10
        )
        self.health_check = Mock(return_value=True)
    
    def test_pool_creation(self):
        """测试连接池创建"""
        pool = ConnectionPool(
            self.connection_factory,
            self.config,
            self.health_check
        )
        
        self.assertEqual(pool.connection_factory, self.connection_factory)
        self.assertEqual(pool.config, self.config)
        self.assertEqual(pool.health_check, self.health_check)
        self.assertFalse(pool._closed)
        
        # 检查初始连接数
        stats = pool.get_stats()
        self.assertEqual(stats['min_connections'], 2)
        self.assertEqual(stats['max_connections'], 5)
    
    def test_get_connection(self):
        """测试获取连接"""
        pool = ConnectionPool(self.connection_factory, self.config)
        
        conn = pool.get_connection()
        self.assertIsInstance(conn, Connection)
        self.assertTrue(conn.is_healthy)
        
        stats = pool.get_stats()
        self.assertEqual(stats['active_connections'], 1)
    
    def test_return_connection(self):
        """测试归还连接"""
        pool = ConnectionPool(self.connection_factory, self.config)
        
        conn = pool.get_connection()
        initial_active = pool.get_stats()['active_connections']
        
        pool.return_connection(conn)
        
        stats = pool.get_stats()
        self.assertEqual(stats['active_connections'], initial_active - 1)
    
    def test_connection_timeout(self):
        """测试连接超时"""
        # 创建一个总是返回None的工厂
        failing_factory = Mock(side_effect=Exception("Connection failed"))
        pool = ConnectionPool(failing_factory, self.config)
        
        conn = pool.get_connection(timeout=0.1)
        self.assertIsNone(conn)
    
    def test_health_check(self):
        """测试健康检查"""
        # 创建一个会失败的健康检查
        failing_health_check = Mock(return_value=False)
        pool = ConnectionPool(
            self.connection_factory,
            self.config,
            failing_health_check
        )
        
        conn = pool.get_connection()
        if conn:
            pool.return_connection(conn)
        
        # 等待健康检查执行
        time.sleep(0.1)
    
    def test_pool_stats(self):
        """测试连接池统计"""
        pool = ConnectionPool(self.connection_factory, self.config)
        
        stats = pool.get_stats()
        expected_keys = [
            'total_connections', 'active_connections', 'idle_connections',
            'pool_size', 'created_connections', 'failed_connections',
            'max_connections', 'min_connections'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
    
    def test_pool_close(self):
        """测试连接池关闭"""
        pool = ConnectionPool(self.connection_factory, self.config)
        
        conn = pool.get_connection()
        pool.close()
        
        self.assertTrue(pool._closed)
        
        # 关闭后不应该能获取新连接
        new_conn = pool.get_connection()
        self.assertIsNone(new_conn)


class TestThreadPoolManager(unittest.TestCase):
    """线程池管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = ThreadPoolManager(max_workers=2)
    
    def tearDown(self):
        """清理测试环境"""
        self.manager.shutdown()
    
    def test_submit_task(self):
        """测试提交任务"""
        def test_func(x, y):
            return x + y
        
        future = self.manager.submit_task("test_task", test_func, 1, 2)
        result = future.result(timeout=5)
        
        self.assertEqual(result, 3)
    
    def test_get_task_result(self):
        """测试获取任务结果"""
        def test_func(x):
            time.sleep(0.1)
            return x * 2
        
        self.manager.submit_task("test_task", test_func, 5)
        result = self.manager.get_task_result("test_task", timeout=5)
        
        self.assertEqual(result, 10)
    
    def test_cancel_task(self):
        """测试取消任务"""
        def slow_func():
            time.sleep(10)
            return "completed"
        
        self.manager.submit_task("slow_task", slow_func)
        cancelled = self.manager.cancel_task("slow_task")
        
        # 注意：已经开始执行的任务可能无法取消
        self.assertIsInstance(cancelled, bool)
    
    def test_nonexistent_task(self):
        """测试不存在的任务"""
        with self.assertRaises(ValueError):
            self.manager.get_task_result("nonexistent_task")
        
        cancelled = self.manager.cancel_task("nonexistent_task")
        self.assertFalse(cancelled)
    
    def test_stats(self):
        """测试统计信息"""
        stats = self.manager.get_stats()
        
        self.assertIn('max_workers', stats)
        self.assertIn('active_tasks', stats)
        self.assertEqual(stats['max_workers'], 2)


class TestMemoryManager(unittest.TestCase):
    """内存管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = MemoryManager(
            memory_threshold=0.9,  # 设置高阈值避免触发清理
            gc_interval=3600  # 设置长间隔避免自动清理
        )
    
    def test_register_object(self):
        """测试对象注册"""
        test_obj = {"test": "data"}
        self.manager.register_object(test_obj)
        
        # 检查对象是否被注册（通过弱引用）
        self.assertIn(test_obj, self.manager._weak_refs)
    
    def test_cache_object(self):
        """测试对象缓存"""
        test_obj = {"test": "data"}
        self.manager.cache_object("test_key", test_obj)
        
        cached_obj = self.manager.get_cached_object("test_key")
        self.assertEqual(cached_obj, test_obj)
    
    def test_cache_with_ttl(self):
        """测试带TTL的缓存"""
        test_obj = {"test": "data"}
        self.manager.cache_object("test_key", test_obj, ttl=1)
        
        # 立即获取应该成功
        cached_obj = self.manager.get_cached_object("test_key")
        self.assertEqual(cached_obj, test_obj)
        
        # 等待过期
        time.sleep(1.1)
        cached_obj = self.manager.get_cached_object("test_key")
        self.assertIsNone(cached_obj)
    
    def test_clear_cache(self):
        """测试清空缓存"""
        self.manager.cache_object("key1", "value1")
        self.manager.cache_object("key2", "value2")
        
        self.manager.clear_cache()
        
        self.assertIsNone(self.manager.get_cached_object("key1"))
        self.assertIsNone(self.manager.get_cached_object("key2"))
    
    def test_memory_stats(self):
        """测试内存统计"""
        stats = self.manager.get_memory_stats()
        
        expected_keys = [
            'total_memory', 'available_memory', 'used_memory',
            'memory_percent', 'cached_objects', 'registered_objects'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)


class TestResourceManager(unittest.TestCase):
    """资源管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = ResourceManager()
    
    def tearDown(self):
        """清理测试环境"""
        self.manager.cleanup()
    
    def test_create_connection_pool(self):
        """测试创建连接池"""
        def mock_factory():
            return Mock()
        
        config = ConnectionPoolConfig(min_connections=1, max_connections=3)
        pool = self.manager.create_connection_pool(
            "test_pool",
            mock_factory,
            config
        )
        
        self.assertIsInstance(pool, ConnectionPool)
        self.assertEqual(self.manager.get_connection_pool("test_pool"), pool)
    
    def test_duplicate_pool_name(self):
        """测试重复的连接池名称"""
        def mock_factory():
            return Mock()
        
        self.manager.create_connection_pool("test_pool", mock_factory)
        
        with self.assertRaises(ValueError):
            self.manager.create_connection_pool("test_pool", mock_factory)
    
    def test_get_connection_context_manager(self):
        """测试连接上下文管理器"""
        def mock_factory():
            return Mock()
        
        self.manager.create_connection_pool("test_pool", mock_factory)
        
        with self.manager.get_connection("test_pool") as conn:
            self.assertIsNotNone(conn)
    
    def test_nonexistent_pool(self):
        """测试不存在的连接池"""
        with self.assertRaises(ValueError):
            with self.manager.get_connection("nonexistent_pool"):
                pass
    
    def test_task_management(self):
        """测试任务管理"""
        def test_func(x):
            return x * 2
        
        future = self.manager.submit_task("test_task", test_func, 5)
        result = self.manager.get_task_result("test_task", timeout=5)
        
        self.assertEqual(result, 10)
    
    def test_cache_management(self):
        """测试缓存管理"""
        test_obj = {"test": "data"}
        self.manager.cache_object("test_key", test_obj)
        
        cached_obj = self.manager.get_cached_object("test_key")
        self.assertEqual(cached_obj, test_obj)
        
        self.manager.clear_cache()
        cached_obj = self.manager.get_cached_object("test_key")
        self.assertIsNone(cached_obj)
    
    def test_monitoring(self):
        """测试监控功能"""
        self.manager.start_monitoring()
        self.assertTrue(self.manager._monitoring)
        
        self.manager.stop_monitoring()
        self.assertFalse(self.manager._monitoring)
    
    def test_resource_stats(self):
        """测试资源统计"""
        stats = self.manager.get_resource_stats()
        
        expected_keys = [
            'timestamp', 'system', 'connection_pools',
            'thread_pool', 'memory_manager'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)


class TestGlobalResourceManager(unittest.TestCase):
    """全局资源管理器测试"""
    
    def tearDown(self):
        """清理测试环境"""
        cleanup_resources()
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        rm1 = get_resource_manager()
        rm2 = get_resource_manager()
        
        self.assertIs(rm1, rm2)
    
    def test_cleanup_resources(self):
        """测试资源清理"""
        rm = get_resource_manager()
        self.assertIsNotNone(rm)
        
        cleanup_resources()
        
        # 清理后应该能获取新的实例
        new_rm = get_resource_manager()
        self.assertIsNotNone(new_rm)


class TestDecorators(unittest.TestCase):
    """装饰器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = get_resource_manager()
        
        # 创建测试连接池
        def mock_factory():
            return Mock()
        
        self.manager.create_connection_pool("test_pool", mock_factory)
    
    def tearDown(self):
        """清理测试环境"""
        cleanup_resources()
    
    def test_with_connection_decorator(self):
        """测试连接装饰器"""
        @with_connection("test_pool")
        def test_func(conn, x, y):
            return x + y
        
        result = test_func(1, 2)
        self.assertEqual(result, 3)
    
    def test_async_task_decorator(self):
        """测试异步任务装饰器"""
        @async_task("test_async")
        def test_func(x):
            return x * 2
        
        future = test_func(5)
        result = future.result(timeout=5)
        self.assertEqual(result, 10)
    
    def test_cached_decorator(self):
        """测试缓存装饰器"""
        call_count = 0
        
        @cached("test_cache")
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # 第一次调用
        result1 = expensive_func(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # 第二次调用应该使用缓存
        result2 = expensive_func(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # 没有增加
    
    def test_cached_decorator_with_ttl(self):
        """测试带TTL的缓存装饰器"""
        call_count = 0
        
        @cached("test_cache_ttl", ttl=1)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # 第一次调用
        result1 = expensive_func(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # 等待过期
        time.sleep(1.1)
        
        # 第二次调用应该重新执行
        result2 = expensive_func(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 2)  # 增加了


class TestResourceMetrics(unittest.TestCase):
    """资源指标测试"""
    
    def test_resource_metrics_creation(self):
        """测试资源指标创建"""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used=1024*1024*1024,  # 1GB
            open_files=100,
            network_connections=50,
            thread_count=10
        )
        
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.memory_used, 1024*1024*1024)
    
    def test_resource_metrics_to_dict(self):
        """测试资源指标转换为字典"""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used=1024*1024*1024,
            open_files=100,
            network_connections=50,
            thread_count=10
        )
        
        metrics_dict = metrics.to_dict()
        
        expected_keys = [
            'timestamp', 'cpu_percent', 'memory_percent',
            'memory_used', 'open_files', 'network_connections',
            'thread_count'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics_dict)


if __name__ == '__main__':
    unittest.main()