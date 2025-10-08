"""
内存池管理系统
用于预分配内存块，减少内存碎片和分配开销
"""

import numpy as np
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import gc
import psutil
import os
from dataclasses import dataclass
from enum import Enum


class PoolType(Enum):
    """内存池类型"""
    SMALL = "small"      # 小对象池 (< 1KB)
    MEDIUM = "medium"    # 中等对象池 (1KB - 1MB)
    LARGE = "large"      # 大对象池 (> 1MB)
    NUMPY = "numpy"      # NumPy数组专用池


@dataclass
class PoolStats:
    """内存池统计信息"""
    pool_type: PoolType
    total_allocated: int
    total_used: int
    total_free: int
    allocation_count: int
    deallocation_count: int
    hit_rate: float
    fragmentation_rate: float


class MemoryBlock:
    """内存块"""
    
    def __init__(self, size: int, pool_type: PoolType):
        self.size = size
        self.pool_type = pool_type
        self.data = None
        self.is_free = True
        self.ref_count = 0
        self.created_at = None
        
    def allocate(self, requested_size: int) -> Optional[Any]:
        """分配内存块"""
        if not self.is_free or requested_size > self.size:
            return None
            
        if self.pool_type == PoolType.NUMPY:
            # 为NumPy数组分配内存
            shape = self._calculate_shape(requested_size)
            self.data = np.empty(shape, dtype=np.float64)
        else:
            # 为普通对象分配内存
            self.data = bytearray(self.size)
            
        self.is_free = False
        self.ref_count = 1
        return self.data
        
    def deallocate(self):
        """释放内存块"""
        if self.ref_count > 0:
            self.ref_count -= 1
            
        if self.ref_count == 0:
            self.data = None
            self.is_free = True
            
    def _calculate_shape(self, size: int) -> Tuple[int, ...]:
        """计算NumPy数组形状"""
        # 假设每个float64占8字节
        elements = size // 8
        if elements < 1000:
            return (elements,)
        elif elements < 1000000:
            rows = int(np.sqrt(elements))
            cols = elements // rows
            return (rows, cols)
        else:
            # 三维数组用于大数据
            cube_root = int(np.cbrt(elements))
            return (cube_root, cube_root, elements // (cube_root * cube_root))


class MemoryPool:
    """单个内存池"""
    
    def __init__(self, pool_type: PoolType, block_size: int, initial_blocks: int = 10):
        self.pool_type = pool_type
        self.block_size = block_size
        self.blocks: List[MemoryBlock] = []
        self.free_blocks: List[MemoryBlock] = []
        self.used_blocks: List[MemoryBlock] = []
        self.lock = threading.Lock()
        
        # 统计信息
        self.allocation_count = 0
        self.deallocation_count = 0
        self.total_allocated_size = 0
        
        # 初始化内存块
        self._initialize_blocks(initial_blocks)
        
    def _initialize_blocks(self, count: int):
        """初始化内存块"""
        for _ in range(count):
            block = MemoryBlock(self.block_size, self.pool_type)
            self.blocks.append(block)
            self.free_blocks.append(block)
            
    def allocate(self, size: int) -> Optional[Any]:
        """从池中分配内存"""
        with self.lock:
            # 查找合适的空闲块
            for i, block in enumerate(self.free_blocks):
                if block.size >= size:
                    data = block.allocate(size)
                    if data is not None:
                        self.free_blocks.pop(i)
                        self.used_blocks.append(block)
                        self.allocation_count += 1
                        self.total_allocated_size += size
                        return data
                        
            # 如果没有合适的空闲块，创建新块
            if len(self.blocks) < 1000:  # 限制最大块数
                new_size = max(size, self.block_size)
                new_block = MemoryBlock(new_size, self.pool_type)
                data = new_block.allocate(size)
                if data is not None:
                    self.blocks.append(new_block)
                    self.used_blocks.append(new_block)
                    self.allocation_count += 1
                    self.total_allocated_size += size
                    return data
                    
            return None
            
    def deallocate(self, data: Any) -> bool:
        """释放内存到池中"""
        with self.lock:
            for i, block in enumerate(self.used_blocks):
                if block.data is data:
                    block.deallocate()
                    self.used_blocks.pop(i)
                    self.free_blocks.append(block)
                    self.deallocation_count += 1
                    return True
            return False
            
    def get_stats(self) -> PoolStats:
        """获取池统计信息"""
        with self.lock:
            total_allocated = len(self.blocks) * self.block_size
            total_used = len(self.used_blocks) * self.block_size
            total_free = len(self.free_blocks) * self.block_size
            
            hit_rate = (self.allocation_count / max(1, self.allocation_count + self.deallocation_count)) * 100
            fragmentation_rate = ((total_allocated - total_used) / max(1, total_allocated)) * 100
            
            return PoolStats(
                pool_type=self.pool_type,
                total_allocated=total_allocated,
                total_used=total_used,
                total_free=total_free,
                allocation_count=self.allocation_count,
                deallocation_count=self.deallocation_count,
                hit_rate=hit_rate,
                fragmentation_rate=fragmentation_rate
            )
            
    def cleanup(self):
        """清理未使用的内存块"""
        with self.lock:
            # 保留一些空闲块，清理多余的
            if len(self.free_blocks) > 20:
                excess_blocks = self.free_blocks[20:]
                self.free_blocks = self.free_blocks[:20]
                
                for block in excess_blocks:
                    if block in self.blocks:
                        self.blocks.remove(block)
                        
                # 强制垃圾回收
                gc.collect()


class MemoryPoolManager:
    """内存池管理器"""
    
    def __init__(self):
        self.pools: Dict[PoolType, MemoryPool] = {}
        self.lock = threading.Lock()
        self.total_memory_limit = self._get_available_memory() * 0.3  # 使用30%可用内存
        
        # 初始化不同类型的内存池
        self._initialize_pools()
        
    def _get_available_memory(self) -> int:
        """获取可用内存大小"""
        return psutil.virtual_memory().available
        
    def _initialize_pools(self):
        """初始化内存池"""
        pool_configs = {
            PoolType.SMALL: (1024, 50),        # 1KB块，50个
            PoolType.MEDIUM: (1024 * 1024, 20), # 1MB块，20个
            PoolType.LARGE: (10 * 1024 * 1024, 5), # 10MB块，5个
            PoolType.NUMPY: (1024 * 1024, 10)   # 1MB NumPy块，10个
        }
        
        for pool_type, (block_size, initial_blocks) in pool_configs.items():
            self.pools[pool_type] = MemoryPool(pool_type, block_size, initial_blocks)
            
    def _determine_pool_type(self, size: int, data_type: str = "general") -> PoolType:
        """确定应该使用的内存池类型"""
        if data_type == "numpy":
            return PoolType.NUMPY
        elif size < 1024:
            return PoolType.SMALL
        elif size < 1024 * 1024:
            return PoolType.MEDIUM
        else:
            return PoolType.LARGE
            
    def allocate(self, size: int, data_type: str = "general") -> Optional[Any]:
        """分配内存"""
        pool_type = self._determine_pool_type(size, data_type)
        
        with self.lock:
            if pool_type in self.pools:
                return self.pools[pool_type].allocate(size)
            return None
            
    def deallocate(self, data: Any, pool_type: Optional[PoolType] = None) -> bool:
        """释放内存"""
        with self.lock:
            if pool_type and pool_type in self.pools:
                return self.pools[pool_type].deallocate(data)
            else:
                # 尝试在所有池中查找
                for pool in self.pools.values():
                    if pool.deallocate(data):
                        return True
            return False
            
    def allocate_numpy_array(self, shape: Tuple[int, ...], dtype=np.float64) -> Optional[np.ndarray]:
        """分配NumPy数组"""
        size = np.prod(shape) * np.dtype(dtype).itemsize
        data = self.allocate(size, "numpy")
        
        if data is not None and isinstance(data, np.ndarray):
            # 重塑数组到所需形状
            if data.size >= np.prod(shape):
                return data.reshape(shape)[:np.prod(shape)].reshape(shape)
        return None
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        pool_stats = {}
        total_pool_memory = 0
        
        for pool_type, pool in self.pools.items():
            stats = pool.get_stats()
            pool_stats[pool_type.value] = {
                'allocated': stats.total_allocated,
                'used': stats.total_used,
                'free': stats.total_free,
                'allocation_count': stats.allocation_count,
                'deallocation_count': stats.deallocation_count,
                'hit_rate': stats.hit_rate,
                'fragmentation_rate': stats.fragmentation_rate
            }
            total_pool_memory += stats.total_allocated
            
        return {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'pool_memory_mb': total_pool_memory / 1024 / 1024,
            'memory_limit_mb': self.total_memory_limit / 1024 / 1024,
            'pool_stats': pool_stats
        }
        
    def cleanup_all_pools(self):
        """清理所有内存池"""
        with self.lock:
            for pool in self.pools.values():
                pool.cleanup()
                
    def optimize_memory(self):
        """优化内存使用"""
        # 获取当前内存使用情况
        memory_usage = self.get_memory_usage()
        
        # 如果内存使用过高，进行清理
        if memory_usage['process_memory_mb'] > memory_usage['memory_limit_mb']:
            self.cleanup_all_pools()
            gc.collect()
            
        # 检查碎片率，如果过高则重新整理
        for pool_type, pool in self.pools.items():
            stats = pool.get_stats()
            if stats.fragmentation_rate > 50:  # 碎片率超过50%
                pool.cleanup()


# 全局内存池管理器实例
_global_memory_pool = None
_pool_lock = threading.Lock()


def get_memory_pool() -> MemoryPoolManager:
    """获取全局内存池管理器实例"""
    global _global_memory_pool
    if _global_memory_pool is None:
        with _pool_lock:
            if _global_memory_pool is None:
                _global_memory_pool = MemoryPoolManager()
    return _global_memory_pool


def memory_pool_allocate(size: int, data_type: str = "general") -> Optional[Any]:
    """便捷函数：从内存池分配内存"""
    return get_memory_pool().allocate(size, data_type)


def memory_pool_deallocate(data: Any, pool_type: Optional[PoolType] = None) -> bool:
    """便捷函数：释放内存到内存池"""
    return get_memory_pool().deallocate(data, pool_type)


def memory_pool_numpy_array(shape: Tuple[int, ...], dtype=np.float64) -> Optional[np.ndarray]:
    """便捷函数：从内存池分配NumPy数组"""
    return get_memory_pool().allocate_numpy_array(shape, dtype)


class MemoryPoolContext:
    """内存池上下文管理器"""
    
    def __init__(self):
        self.allocated_objects = []
        self.pool_manager = get_memory_pool()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 自动释放所有分配的对象
        for obj, pool_type in self.allocated_objects:
            self.pool_manager.deallocate(obj, pool_type)
        self.allocated_objects.clear()
        
    def allocate(self, size: int, data_type: str = "general") -> Optional[Any]:
        """在上下文中分配内存"""
        data = self.pool_manager.allocate(size, data_type)
        if data is not None:
            pool_type = self.pool_manager._determine_pool_type(size, data_type)
            self.allocated_objects.append((data, pool_type))
        return data
        
    def allocate_numpy_array(self, shape: Tuple[int, ...], dtype=np.float64) -> Optional[np.ndarray]:
        """在上下文中分配NumPy数组"""
        data = self.pool_manager.allocate_numpy_array(shape, dtype)
        if data is not None:
            self.allocated_objects.append((data, PoolType.NUMPY))
        return data


if __name__ == "__main__":
    # 测试内存池管理系统
    print("内存池管理系统测试:")
    
    # 创建内存池管理器
    pool_manager = get_memory_pool()
    
    # 测试1: 基本内存分配
    print("\n1. 基本内存分配测试:")
    small_data = pool_manager.allocate(512, "general")
    medium_data = pool_manager.allocate(1024 * 500, "general")
    large_data = pool_manager.allocate(5 * 1024 * 1024, "general")
    
    print(f"小内存分配: {'成功' if small_data else '失败'}")
    print(f"中等内存分配: {'成功' if medium_data else '失败'}")
    print(f"大内存分配: {'成功' if large_data else '失败'}")
    
    # 测试2: NumPy数组分配
    print("\n2. NumPy数组分配测试:")
    numpy_array = pool_manager.allocate_numpy_array((1000, 100))
    print(f"NumPy数组分配: {'成功' if numpy_array is not None else '失败'}")
    if numpy_array is not None:
        print(f"数组形状: {numpy_array.shape}")
        numpy_array.fill(1.0)
        print(f"数组填充测试: {'成功' if numpy_array[0, 0] == 1.0 else '失败'}")
    
    # 测试3: 内存使用统计
    print("\n3. 内存使用统计:")
    memory_usage = pool_manager.get_memory_usage()
    print(f"进程内存使用: {memory_usage['process_memory_mb']:.2f} MB")
    print(f"内存池使用: {memory_usage['pool_memory_mb']:.2f} MB")
    print(f"内存限制: {memory_usage['memory_limit_mb']:.2f} MB")
    
    for pool_name, stats in memory_usage['pool_stats'].items():
        print(f"\n{pool_name}池统计:")
        print(f"  分配: {stats['allocated'] / 1024 / 1024:.2f} MB")
        print(f"  使用: {stats['used'] / 1024 / 1024:.2f} MB")
        print(f"  空闲: {stats['free'] / 1024 / 1024:.2f} MB")
        print(f"  命中率: {stats['hit_rate']:.2f}%")
        print(f"  碎片率: {stats['fragmentation_rate']:.2f}%")
    
    # 测试4: 内存释放
    print("\n4. 内存释放测试:")
    release_success = 0
    if small_data:
        release_success += pool_manager.deallocate(small_data, PoolType.SMALL)
    if medium_data:
        release_success += pool_manager.deallocate(medium_data, PoolType.MEDIUM)
    if large_data:
        release_success += pool_manager.deallocate(large_data, PoolType.LARGE)
    if numpy_array is not None:
        release_success += pool_manager.deallocate(numpy_array, PoolType.NUMPY)
    
    print(f"成功释放 {release_success} 个内存块")
    
    # 测试5: 上下文管理器
    print("\n5. 上下文管理器测试:")
    with MemoryPoolContext() as ctx:
        ctx_data1 = ctx.allocate(1024)
        ctx_data2 = ctx.allocate_numpy_array((100, 50))
        print(f"上下文内分配: {'成功' if ctx_data1 and ctx_data2 is not None else '失败'}")
    print("上下文自动清理完成")
    
    # 测试6: 性能测试
    print("\n6. 性能测试:")
    import time
    
    # 传统分配方式
    start_time = time.time()
    traditional_arrays = []
    for _ in range(100):
        arr = np.random.random((1000, 100))
        traditional_arrays.append(arr)
    traditional_time = time.time() - start_time
    
    # 内存池分配方式
    start_time = time.time()
    pool_arrays = []
    for _ in range(100):
        arr = pool_manager.allocate_numpy_array((1000, 100))
        if arr is not None:
            arr[:] = np.random.random((1000, 100))
            pool_arrays.append(arr)
    pool_time = time.time() - start_time
    
    print(f"传统分配耗时: {traditional_time:.4f}s")
    print(f"内存池分配耗时: {pool_time:.4f}s")
    if pool_time > 0:
        print(f"性能提升: {traditional_time / pool_time:.2f}x")
    
    # 清理测试数据
    for arr in pool_arrays:
        pool_manager.deallocate(arr, PoolType.NUMPY)
    
    # 最终内存优化
    pool_manager.optimize_memory()
    print("\n内存优化完成")