#!/usr/bin/env python3
"""
智能缓存系统
用于缓存市场数据、计算结果和策略输出，提升回测性能
"""

import hashlib
import pickle
import time
import threading
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import OrderedDict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None  # Time to live in seconds


class LRUCache:
    """LRU (Least Recently Used) 缓存实现"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 500.0):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_bytes = 0
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_mb': 0.0
        }
    
    def _calculate_size(self, obj: Any) -> int:
        """计算对象大小"""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                return len(pickle.dumps(obj))
        except Exception:
            return 1024  # 默认大小
    
    def _evict_if_needed(self):
        """根据需要驱逐缓存条目"""
        while (len(self.cache) > self.max_size or 
               self.current_memory_bytes > self.max_memory_bytes):
            if not self.cache:
                break
                
            # 移除最少使用的条目
            key, entry = self.cache.popitem(last=False)
            self.current_memory_bytes -= entry.size_bytes
            self.stats['evictions'] += 1
            logger.debug(f"驱逐缓存条目: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # 检查TTL
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    self.remove(key)
                    self.stats['misses'] += 1
                    return None
                
                # 移动到末尾（最近使用）
                self.cache.move_to_end(key)
                entry.access_count += 1
                self.stats['hits'] += 1
                return entry.value
            else:
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """存储缓存值"""
        with self.lock:
            size_bytes = self._calculate_size(value)
            
            # 如果单个对象太大，不缓存
            if size_bytes > self.max_memory_bytes * 0.5:
                logger.warning(f"对象太大，不缓存: {key} ({size_bytes / 1024 / 1024:.1f} MB)")
                return
            
            # 如果键已存在，更新
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes
            
            # 创建新条目
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            self.cache[key] = entry
            self.current_memory_bytes += size_bytes
            
            # 驱逐旧条目
            self._evict_if_needed()
            
            # 更新统计
            self.stats['memory_usage_mb'] = self.current_memory_bytes / 1024 / 1024
    
    def remove(self, key: str) -> bool:
        """移除缓存条目"""
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_memory_bytes -= entry.size_bytes
                self.stats['memory_usage_mb'] = self.current_memory_bytes / 1024 / 1024
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.current_memory_bytes = 0
            self.stats['memory_usage_mb'] = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
            return {
                **self.stats,
                'size': len(self.cache),
                'hit_rate': hit_rate,
                'max_size': self.max_size,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024
            }


class SmartCacheSystem:
    """智能缓存系统"""
    
    def __init__(self, 
                 market_data_cache_size: int = 500,
                 calculation_cache_size: int = 1000,
                 result_cache_size: int = 200,
                 max_memory_mb: float = 1000.0):
        
        # 不同类型的缓存
        self.market_data_cache = LRUCache(
            max_size=market_data_cache_size,
            max_memory_mb=max_memory_mb * 0.6  # 60% 用于市场数据
        )
        
        self.calculation_cache = LRUCache(
            max_size=calculation_cache_size,
            max_memory_mb=max_memory_mb * 0.3  # 30% 用于计算结果
        )
        
        self.result_cache = LRUCache(
            max_size=result_cache_size,
            max_memory_mb=max_memory_mb * 0.1  # 10% 用于最终结果
        )
        
        self.enabled = True
        logger.info(f"智能缓存系统初始化完成，总内存限制: {max_memory_mb} MB")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = f"{prefix}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cache_market_data(self, symbol: str, data: pd.DataFrame, 
                         start_date: str = None, end_date: str = None) -> str:
        """缓存市场数据"""
        if not self.enabled:
            return ""
            
        key = self._generate_key("market_data", symbol, start_date, end_date)
        self.market_data_cache.put(key, data.copy(), ttl=3600)  # 1小时TTL
        logger.debug(f"缓存市场数据: {symbol}")
        return key
    
    def get_market_data(self, symbol: str, 
                       start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """获取缓存的市场数据"""
        if not self.enabled:
            return None
            
        key = self._generate_key("market_data", symbol, start_date, end_date)
        data = self.market_data_cache.get(key)
        if data is not None:
            logger.debug(f"命中市场数据缓存: {symbol}")
        return data
    
    def cache_calculation(self, calculation_type: str, params: Dict[str, Any], 
                         result: Any, ttl: float = 1800) -> str:
        """缓存计算结果"""
        if not self.enabled:
            return ""
            
        key = self._generate_key("calculation", calculation_type, **params)
        self.calculation_cache.put(key, result, ttl=ttl)
        logger.debug(f"缓存计算结果: {calculation_type}")
        return key
    
    def get_calculation(self, calculation_type: str, params: Dict[str, Any]) -> Optional[Any]:
        """获取缓存的计算结果"""
        if not self.enabled:
            return None
            
        key = self._generate_key("calculation", calculation_type, **params)
        result = self.calculation_cache.get(key)
        if result is not None:
            logger.debug(f"命中计算缓存: {calculation_type}")
        return result
    
    def cache_backtest_result(self, strategy_name: str, symbol: str, 
                            params: Dict[str, Any], result: Dict[str, Any]) -> str:
        """缓存回测结果"""
        if not self.enabled:
            return ""
            
        key = self._generate_key("backtest", strategy_name, symbol, **params)
        self.result_cache.put(key, result, ttl=7200)  # 2小时TTL
        logger.debug(f"缓存回测结果: {strategy_name}_{symbol}")
        return key
    
    def get_backtest_result(self, strategy_name: str, symbol: str, 
                          params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取缓存的回测结果"""
        if not self.enabled:
            return None
            
        key = self._generate_key("backtest", strategy_name, symbol, **params)
        result = self.result_cache.get(key)
        if result is not None:
            logger.debug(f"命中回测结果缓存: {strategy_name}_{symbol}")
        return result
    
    def invalidate_symbol_cache(self, symbol: str):
        """使特定股票的缓存失效"""
        # 这里简化实现，实际应该遍历所有相关键
        logger.info(f"使股票缓存失效: {symbol}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合缓存统计"""
        market_stats = self.market_data_cache.get_stats()
        calc_stats = self.calculation_cache.get_stats()
        result_stats = self.result_cache.get_stats()
        
        total_memory = (market_stats['memory_usage_mb'] + 
                       calc_stats['memory_usage_mb'] + 
                       result_stats['memory_usage_mb'])
        
        total_hits = market_stats['hits'] + calc_stats['hits'] + result_stats['hits']
        total_misses = market_stats['misses'] + calc_stats['misses'] + result_stats['misses']
        overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        
        return {
            'enabled': self.enabled,
            'total_memory_usage_mb': total_memory,
            'overall_hit_rate': overall_hit_rate,
            'market_data_cache': market_stats,
            'calculation_cache': calc_stats,
            'result_cache': result_stats
        }
    
    def optimize_cache(self):
        """优化缓存性能"""
        stats = self.get_comprehensive_stats()
        
        # 如果命中率太低，可能需要调整缓存策略
        if stats['overall_hit_rate'] < 0.3:
            logger.warning(f"缓存命中率较低: {stats['overall_hit_rate']:.2%}")
        
        # 如果内存使用率过高，可能需要清理
        if stats['total_memory_usage_mb'] > 800:  # 80% 阈值
            logger.info("内存使用率过高，执行缓存清理")
            self._cleanup_old_entries()
    
    def _cleanup_old_entries(self):
        """清理旧的缓存条目"""
        current_time = time.time()
        
        # 清理超过1小时的市场数据缓存
        for cache in [self.market_data_cache, self.calculation_cache, self.result_cache]:
            with cache.lock:
                keys_to_remove = []
                for key, entry in cache.cache.items():
                    if current_time - entry.timestamp > 3600:  # 1小时
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    cache.remove(key)
    
    def enable(self):
        """启用缓存"""
        self.enabled = True
        logger.info("缓存系统已启用")
    
    def disable(self):
        """禁用缓存"""
        self.enabled = False
        logger.info("缓存系统已禁用")
    
    def clear_all(self):
        """清空所有缓存"""
        self.market_data_cache.clear()
        self.calculation_cache.clear()
        self.result_cache.clear()
        logger.info("所有缓存已清空")


# 全局缓存实例
_global_cache_system: Optional[SmartCacheSystem] = None


def get_cache_system() -> SmartCacheSystem:
    """获取全局缓存系统实例"""
    global _global_cache_system
    if _global_cache_system is None:
        _global_cache_system = SmartCacheSystem()
    return _global_cache_system


def cache_decorator(cache_type: str = "calculation", ttl: float = 1800):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_system = get_cache_system()
            
            if not cache_system.enabled:
                return func(*args, **kwargs)
            
            # 生成缓存键
            func_name = func.__name__
            key_data = f"{func_name}_{args}_{sorted(kwargs.items())}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # 尝试从缓存获取
            if cache_type == "calculation":
                cached_result = cache_system.calculation_cache.get(cache_key)
            else:
                cached_result = None
            
            if cached_result is not None:
                logger.debug(f"命中函数缓存: {func_name}")
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            if cache_type == "calculation":
                cache_system.calculation_cache.put(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试缓存系统
    cache_system = SmartCacheSystem(max_memory_mb=100)
    
    # 测试市场数据缓存
    test_data = pd.DataFrame({
        'close': np.random.randn(1000),
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    cache_system.cache_market_data("AAPL", test_data)
    cached_data = cache_system.get_market_data("AAPL")
    
    print("缓存测试:")
    print(f"原始数据形状: {test_data.shape}")
    print(f"缓存数据形状: {cached_data.shape if cached_data is not None else None}")
    
    # 测试计算缓存
    @cache_decorator(cache_type="calculation", ttl=300)
    def expensive_calculation(n: int) -> float:
        time.sleep(0.1)  # 模拟耗时计算
        return sum(i**2 for i in range(n))
    
    start_time = time.time()
    result1 = expensive_calculation(1000)
    time1 = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_calculation(1000)  # 应该从缓存获取
    time2 = time.time() - start_time
    
    print(f"\n计算缓存测试:")
    print(f"第一次计算: {result1}, 耗时: {time1:.3f}s")
    print(f"第二次计算: {result2}, 耗时: {time2:.3f}s")
    print(f"加速比: {time1/time2:.1f}x")
    
    # 显示统计信息
    stats = cache_system.get_comprehensive_stats()
    print(f"\n缓存统计:")
    print(f"总内存使用: {stats['total_memory_usage_mb']:.1f} MB")
    print(f"总命中率: {stats['overall_hit_rate']:.2%}")