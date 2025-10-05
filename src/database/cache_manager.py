"""
缓存管理器

实现智能缓存策略、缓存失效、性能监控等功能
"""

import json
import hashlib
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import threading
import time
import psutil
import redis
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from .optimized_models import CachedMarketData
from .connection import get_db_session

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """缓存级别枚举"""
    MEMORY = "memory"      # 内存缓存
    REDIS = "redis"        # Redis缓存
    DATABASE = "database"  # 数据库缓存


class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"           # 最近最少使用
    LFU = "lfu"           # 最少使用频率
    TTL = "ttl"           # 生存时间
    FIFO = "fifo"         # 先进先出


@dataclass
class CacheConfig:
    """缓存配置"""
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    max_redis_size: int = 500 * 1024 * 1024   # 500MB
    default_ttl: int = 3600                    # 1小时
    compression_threshold: int = 1024          # 1KB以上压缩
    enable_compression: bool = True
    enable_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


class CacheStats:
    """缓存统计信息"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_usage = 0
        self.redis_usage = 0
        self.db_usage = 0
        self.start_time = datetime.now()
        self._lock = threading.Lock()
    
    def record_hit(self):
        """记录缓存命中"""
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        """记录缓存未命中"""
        with self._lock:
            self.misses += 1
    
    def record_eviction(self):
        """记录缓存驱逐"""
        with self._lock:
            self.evictions += 1
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            uptime = datetime.now() - self.start_time
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': self.get_hit_rate(),
                'memory_usage': self.memory_usage,
                'redis_usage': self.redis_usage,
                'db_usage': self.db_usage,
                'uptime_seconds': uptime.total_seconds()
            }


class SmartCacheManager:
    """
    智能缓存管理器
    
    实现多级缓存、智能失效、性能监控等功能
    """
    
    def __init__(self, config: CacheConfig = None):
        """
        初始化缓存管理器
        
        Args:
            config: 缓存配置
        """
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        self.logger = logger
        
        # 内存缓存
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_lock = threading.RLock()
        
        # Redis缓存
        self._redis_client = None
        if self.config.enable_redis:
            try:
                self._redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=False
                )
                # 测试连接
                self._redis_client.ping()
                self.logger.info("Redis缓存已启用")
            except Exception as e:
                self.logger.warning(f"Redis连接失败，禁用Redis缓存: {e}")
                self._redis_client = None
        
        # 启动后台清理任务
        self._start_cleanup_thread()
    
    def _generate_cache_key(self, key: str, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            key: 基础键
            **kwargs: 额外参数
            
        Returns:
            str: 生成的缓存键
        """
        if kwargs:
            # 将参数排序并序列化
            params_str = json.dumps(kwargs, sort_keys=True, default=str)
            key_with_params = f"{key}:{params_str}"
        else:
            key_with_params = key
        
        # 使用MD5生成固定长度的键
        return hashlib.md5(key_with_params.encode()).hexdigest()
    
    def _serialize_data(self, data: Any) -> bytes:
        """
        序列化数据
        
        Args:
            data: 要序列化的数据
            
        Returns:
            bytes: 序列化后的数据
        """
        try:
            # 使用pickle序列化
            serialized = pickle.dumps(data)
            
            # 如果数据大于阈值，进行压缩
            if self.config.enable_compression and len(serialized) > self.config.compression_threshold:
                serialized = gzip.compress(serialized)
                return b'compressed:' + serialized
            
            return serialized
        except Exception as e:
            self.logger.error(f"数据序列化失败: {e}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """
        反序列化数据
        
        Args:
            data: 序列化的数据
            
        Returns:
            Any: 反序列化后的数据
        """
        try:
            # 检查是否压缩
            if data.startswith(b'compressed:'):
                data = gzip.decompress(data[11:])  # 去掉'compressed:'前缀
            
            return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"数据反序列化失败: {e}")
            raise
    
    def get(self, key: str, **kwargs) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            **kwargs: 额外参数
            
        Returns:
            Optional[Any]: 缓存的数据，如果不存在返回None
        """
        cache_key = self._generate_cache_key(key, **kwargs)
        
        # 1. 尝试从内存缓存获取
        data = self._get_from_memory(cache_key)
        if data is not None:
            self.stats.record_hit()
            return data
        
        # 2. 尝试从Redis缓存获取
        if self._redis_client:
            data = self._get_from_redis(cache_key)
            if data is not None:
                # 将数据放入内存缓存
                self._set_to_memory(cache_key, data)
                self.stats.record_hit()
                return data
        
        # 3. 尝试从数据库缓存获取
        data = self._get_from_database(cache_key)
        if data is not None:
            # 将数据放入上级缓存
            if self._redis_client:
                self._set_to_redis(cache_key, data)
            self._set_to_memory(cache_key, data)
            self.stats.record_hit()
            return data
        
        self.stats.record_miss()
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            ttl: 生存时间（秒）
            **kwargs: 额外参数
            
        Returns:
            bool: 是否设置成功
        """
        cache_key = self._generate_cache_key(key, **kwargs)
        ttl = ttl or self.config.default_ttl
        
        try:
            # 设置到所有缓存级别
            self._set_to_memory(cache_key, data, ttl)
            
            if self._redis_client:
                self._set_to_redis(cache_key, data, ttl)
            
            self._set_to_database(cache_key, data, ttl, key)
            
            return True
        except Exception as e:
            self.logger.error(f"设置缓存失败: {e}")
            return False
    
    def delete(self, key: str, **kwargs) -> bool:
        """
        删除缓存数据
        
        Args:
            key: 缓存键
            **kwargs: 额外参数
            
        Returns:
            bool: 是否删除成功
        """
        cache_key = self._generate_cache_key(key, **kwargs)
        
        try:
            # 从所有缓存级别删除
            self._delete_from_memory(cache_key)
            
            if self._redis_client:
                self._delete_from_redis(cache_key)
            
            self._delete_from_database(cache_key)
            
            return True
        except Exception as e:
            self.logger.error(f"删除缓存失败: {e}")
            return False
    
    def clear(self, pattern: Optional[str] = None) -> bool:
        """
        清空缓存
        
        Args:
            pattern: 键模式（可选）
            
        Returns:
            bool: 是否清空成功
        """
        try:
            if pattern:
                # 清空匹配模式的缓存
                self._clear_pattern(pattern)
            else:
                # 清空所有缓存
                with self._memory_lock:
                    self._memory_cache.clear()
                
                if self._redis_client:
                    self._redis_client.flushdb()
                
                # 清空数据库缓存
                with get_db_session() as session:
                    session.query(CachedMarketData).delete()
                    session.commit()
            
            return True
        except Exception as e:
            self.logger.error(f"清空缓存失败: {e}")
            return False
    
    def _get_from_memory(self, cache_key: str) -> Optional[Any]:
        """从内存缓存获取数据"""
        with self._memory_lock:
            if cache_key in self._memory_cache:
                cache_item = self._memory_cache[cache_key]
                
                # 检查是否过期
                if cache_item['expires_at'] > datetime.now():
                    cache_item['last_accessed'] = datetime.now()
                    cache_item['access_count'] += 1
                    return cache_item['data']
                else:
                    # 过期，删除
                    del self._memory_cache[cache_key]
        
        return None
    
    def _set_to_memory(self, cache_key: str, data: Any, ttl: int = None) -> None:
        """设置数据到内存缓存"""
        ttl = ttl or self.config.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        with self._memory_lock:
            # 检查内存使用量
            self._check_memory_limit()
            
            self._memory_cache[cache_key] = {
                'data': data,
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'last_accessed': datetime.now(),
                'access_count': 1,
                'size': len(str(data))  # 简单估算大小
            }
    
    def _delete_from_memory(self, cache_key: str) -> None:
        """从内存缓存删除数据"""
        with self._memory_lock:
            self._memory_cache.pop(cache_key, None)
    
    def _get_from_redis(self, cache_key: str) -> Optional[Any]:
        """从Redis缓存获取数据"""
        if not self._redis_client:
            return None
        
        try:
            data = self._redis_client.get(cache_key)
            if data:
                return self._deserialize_data(data)
        except Exception as e:
            self.logger.error(f"从Redis获取数据失败: {e}")
        
        return None
    
    def _set_to_redis(self, cache_key: str, data: Any, ttl: int = None) -> None:
        """设置数据到Redis缓存"""
        if not self._redis_client:
            return
        
        try:
            serialized_data = self._serialize_data(data)
            ttl = ttl or self.config.default_ttl
            self._redis_client.setex(cache_key, ttl, serialized_data)
        except Exception as e:
            self.logger.error(f"设置Redis缓存失败: {e}")
    
    def _delete_from_redis(self, cache_key: str) -> None:
        """从Redis缓存删除数据"""
        if not self._redis_client:
            return
        
        try:
            self._redis_client.delete(cache_key)
        except Exception as e:
            self.logger.error(f"从Redis删除数据失败: {e}")
    
    def _get_from_database(self, cache_key: str) -> Optional[Any]:
        """从数据库缓存获取数据"""
        try:
            with get_db_session() as session:
                cache_item = session.query(CachedMarketData).filter(
                    and_(
                        CachedMarketData.cache_key == cache_key,
                        or_(
                            CachedMarketData.expires_at.is_(None),
                            CachedMarketData.expires_at > datetime.now()
                        )
                    )
                ).first()
                
                if cache_item:
                    # 更新访问统计
                    cache_item.access_count += 1
                    cache_item.last_accessed = datetime.now()
                    session.commit()
                    
                    return cache_item.cached_data
        except Exception as e:
            self.logger.error(f"从数据库获取缓存失败: {e}")
        
        return None
    
    def _set_to_database(self, cache_key: str, data: Any, ttl: int, original_key: str) -> None:
        """设置数据到数据库缓存"""
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl)
            data_size = len(json.dumps(data, default=str))
            
            with get_db_session() as session:
                # 检查是否已存在
                existing = session.query(CachedMarketData).filter(
                    CachedMarketData.cache_key == cache_key
                ).first()
                
                if existing:
                    # 更新现有记录
                    existing.cached_data = data
                    existing.expires_at = expires_at
                    existing.data_size = data_size
                    existing.last_accessed = datetime.now()
                else:
                    # 创建新记录
                    cache_item = CachedMarketData(
                        cache_key=cache_key,
                        data_type=original_key.split(':')[0],  # 从原始键提取数据类型
                        cached_data=data,
                        expires_at=expires_at,
                        data_size=data_size
                    )
                    session.add(cache_item)
                
                session.commit()
        except Exception as e:
            self.logger.error(f"设置数据库缓存失败: {e}")
    
    def _delete_from_database(self, cache_key: str) -> None:
        """从数据库缓存删除数据"""
        try:
            with get_db_session() as session:
                session.query(CachedMarketData).filter(
                    CachedMarketData.cache_key == cache_key
                ).delete()
                session.commit()
        except Exception as e:
            self.logger.error(f"从数据库删除缓存失败: {e}")
    
    def _check_memory_limit(self) -> None:
        """检查内存使用限制"""
        current_size = sum(item['size'] for item in self._memory_cache.values())
        
        if current_size > self.config.max_memory_size:
            # 使用LRU策略清理内存
            sorted_items = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            # 删除最旧的25%
            items_to_remove = len(sorted_items) // 4
            for i in range(items_to_remove):
                key_to_remove = sorted_items[i][0]
                del self._memory_cache[key_to_remove]
                self.stats.record_eviction()
    
    def _clear_pattern(self, pattern: str) -> None:
        """清空匹配模式的缓存"""
        # 内存缓存
        with self._memory_lock:
            keys_to_remove = [key for key in self._memory_cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self._memory_cache[key]
        
        # Redis缓存
        if self._redis_client:
            try:
                keys = self._redis_client.keys(f"*{pattern}*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception as e:
                self.logger.error(f"清空Redis模式缓存失败: {e}")
        
        # 数据库缓存
        try:
            with get_db_session() as session:
                session.query(CachedMarketData).filter(
                    CachedMarketData.cache_key.like(f"%{pattern}%")
                ).delete(synchronize_session=False)
                session.commit()
        except Exception as e:
            self.logger.error(f"清空数据库模式缓存失败: {e}")
    
    def _start_cleanup_thread(self) -> None:
        """启动后台清理线程"""
        def cleanup_worker():
            while True:
                try:
                    # 清理过期的内存缓存
                    with self._memory_lock:
                        now = datetime.now()
                        expired_keys = [
                            key for key, item in self._memory_cache.items()
                            if item['expires_at'] <= now
                        ]
                        for key in expired_keys:
                            del self._memory_cache[key]
                    
                    # 清理过期的数据库缓存
                    with get_db_session() as session:
                        expired_count = session.query(CachedMarketData).filter(
                            and_(
                                CachedMarketData.expires_at.isnot(None),
                                CachedMarketData.expires_at <= datetime.now()
                            )
                        ).delete()
                        
                        if expired_count > 0:
                            session.commit()
                            self.logger.info(f"清理了 {expired_count} 个过期缓存项")
                    
                    # 更新统计信息
                    self._update_usage_stats()
                    
                except Exception as e:
                    self.logger.error(f"缓存清理任务出错: {e}")
                
                # 每5分钟执行一次清理
                time.sleep(300)
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _update_usage_stats(self) -> None:
        """更新使用统计"""
        # 内存使用统计
        with self._memory_lock:
            self.stats.memory_usage = sum(item['size'] for item in self._memory_cache.values())
        
        # Redis使用统计
        if self._redis_client:
            try:
                info = self._redis_client.info('memory')
                self.stats.redis_usage = info.get('used_memory', 0)
            except:
                pass
        
        # 数据库使用统计
        try:
            with get_db_session() as session:
                result = session.query(func.sum(CachedMarketData.data_size)).scalar()
                self.stats.db_usage = result or 0
        except:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.stats.get_stats()
    
    def cache_decorator(self, key_prefix: str, ttl: Optional[int] = None):
        """
        缓存装饰器
        
        Args:
            key_prefix: 缓存键前缀
            ttl: 生存时间
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{key_prefix}:{func.__name__}"
                
                # 尝试从缓存获取
                cached_result = self.get(cache_key, args=args, kwargs=kwargs)
                if cached_result is not None:
                    return cached_result
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl, args=args, kwargs=kwargs)
                
                return result
            
            return wrapper
        return decorator


# 全局缓存管理器实例
_cache_manager = None


def get_cache_manager() -> SmartCacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = SmartCacheManager()
    return _cache_manager


def cache_result(key_prefix: str, ttl: Optional[int] = None):
    """缓存结果装饰器"""
    return get_cache_manager().cache_decorator(key_prefix, ttl)


if __name__ == "__main__":
    # 测试缓存管理器
    cache_manager = SmartCacheManager()
    
    # 测试基本功能
    cache_manager.set("test_key", {"data": "test_value"}, ttl=60)
    result = cache_manager.get("test_key")
    print(f"缓存结果: {result}")
    
    # 测试装饰器
    @cache_result("test_func", ttl=30)
    def expensive_function(x, y):
        time.sleep(1)  # 模拟耗时操作
        return x + y
    
    start_time = time.time()
    result1 = expensive_function(1, 2)
    time1 = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_function(1, 2)  # 应该从缓存获取
    time2 = time.time() - start_time
    
    print(f"第一次调用: {result1}, 耗时: {time1:.3f}s")
    print(f"第二次调用: {result2}, 耗时: {time2:.3f}s")
    
    # 打印统计信息
    stats = cache_manager.get_stats()
    print(f"缓存统计: {stats}")