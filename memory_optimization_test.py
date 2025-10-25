#!/usr/bin/env python3
"""
内存优化测试
监控大数据集处理时的内存使用情况并提供优化建议
"""

import sys
import os
import time
import gc
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import tracemalloc

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtesting.enhanced_backtest_engine import (
    EnhancedBacktestEngine, 
    ParallelBacktestEngine,
    SimpleMovingAverageStrategy
)

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_snapshots = []
        self.tracemalloc_snapshots = []
        
    def start_monitoring(self):
        """开始内存监控"""
        tracemalloc.start()
        self.memory_snapshots = []
        self.tracemalloc_snapshots = []
        
    def take_snapshot(self, label: str):
        """拍摄内存快照"""
        # 系统内存
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # Python内存
        snapshot = tracemalloc.take_snapshot()
        
        self.memory_snapshots.append({
            'label': label,
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': memory_percent,
            'tracemalloc_size_mb': sum(stat.size for stat in snapshot.statistics('lineno')) / 1024 / 1024
        })
        
        self.tracemalloc_snapshots.append((label, snapshot))
        
    def get_memory_usage(self) -> Dict:
        """获取当前内存使用情况"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
        
    def analyze_memory_growth(self) -> Dict:
        """分析内存增长"""
        if len(self.memory_snapshots) < 2:
            return {}
            
        first = self.memory_snapshots[0]
        last = self.memory_snapshots[-1]
        
        return {
            'initial_rss_mb': first['rss_mb'],
            'final_rss_mb': last['rss_mb'],
            'growth_rss_mb': last['rss_mb'] - first['rss_mb'],
            'growth_percent': ((last['rss_mb'] - first['rss_mb']) / first['rss_mb']) * 100,
            'peak_rss_mb': max(s['rss_mb'] for s in self.memory_snapshots),
            'snapshots': self.memory_snapshots
        }
        
    def get_top_memory_consumers(self, limit: int = 10) -> List[Dict]:
        """获取内存消耗最大的代码位置"""
        if not self.tracemalloc_snapshots:
            return []
            
        latest_snapshot = self.tracemalloc_snapshots[-1][1]
        top_stats = latest_snapshot.statistics('lineno')[:limit]
        
        consumers = []
        for stat in top_stats:
            consumers.append({
                'filename': stat.traceback.format()[-1] if stat.traceback.format() else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
            
        return consumers

def create_memory_efficient_data(num_symbols: int, days: int, chunk_size: int = 10) -> Dict[str, pd.DataFrame]:
    """创建内存高效的测试数据 - 仅用于测试和演示"""
    print(f"创建 {num_symbols} 个股票 {days} 天的数据 (分块处理)...")
    
    market_data = {}
    start_date = datetime(2023, 1, 1)
    
    # 分块处理以减少内存峰值
    for chunk_start in range(0, num_symbols, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_symbols)
        
        for i in range(chunk_start, chunk_end):
            symbol = f"STOCK_{i:03d}"
            dates = pd.date_range(start=start_date, periods=days, freq='D')
            
            # 使用更节省内存的数据类型 - 模拟数据仅用于演示
            np.random.seed(i)
            returns = np.random.normal(0.001, 0.02, days).astype(np.float32)  # 模拟收益率仅用于测试
            prices = np.zeros(days, dtype=np.float32)
            prices[0] = 100.0
            
            for j in range(1, days):
                prices[j] = prices[j-1] * (1 + returns[j])
            
            # 使用适当的数据类型
            market_data[symbol] = pd.DataFrame({
                'open': prices,
                'high': (prices * (1 + np.random.uniform(0, 0.02, days))).astype(np.float32),  # 模拟高价仅用于测试
                'low': (prices * (1 - np.random.uniform(0, 0.02, days))).astype(np.float32),   # 模拟低价仅用于测试
                'close': prices,
                'volume': np.random.randint(10000, 100000, days, dtype=np.int32)  # 模拟成交量仅用于测试
            }, index=dates)
        
        # 强制垃圾回收
        gc.collect()
    
    return market_data

def test_memory_optimization(num_symbols: int, days: int, scenario_name: str) -> Dict:
    """测试内存优化"""
    print(f"\n{'='*60}")
    print(f"内存优化测试: {scenario_name}")
    print(f"股票数量: {num_symbols}, 数据天数: {days}")
    print(f"{'='*60}")
    
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    
    try:
        # 初始内存快照
        monitor.take_snapshot("开始")
        initial_memory = monitor.get_memory_usage()
        print(f"初始内存使用: {initial_memory['rss_mb']:.1f} MB")
        
        # 生成数据
        start_time = time.time()
        market_data = create_memory_efficient_data(num_symbols, days)
        data_time = time.time() - start_time
        
        monitor.take_snapshot("数据生成完成")
        data_memory = monitor.get_memory_usage()
        print(f"数据生成后内存: {data_memory['rss_mb']:.1f} MB (+{data_memory['rss_mb'] - initial_memory['rss_mb']:.1f} MB)")
        
        # 创建回测引擎
        parallel_engine = ParallelBacktestEngine()
        
        # 批量添加引擎以减少内存峰值
        batch_size = 5
        for batch_start in range(0, len(market_data), batch_size):
            batch_symbols = list(market_data.keys())[batch_start:batch_start + batch_size]
            
            for symbol in batch_symbols:
                data = market_data[symbol]
                engine = EnhancedBacktestEngine(
                    initial_capital=1000000,
                    commission_rate=0.001,
                    slippage_rate=0.001
                )
                engine.set_market_data(symbol, data)
                
                strategy = SimpleMovingAverageStrategy(f"SMA_{symbol}", short_window=10, long_window=30)
                engine.add_strategy(strategy)
                
                parallel_engine.add_engine(symbol, engine)
            
            # 每批次后进行垃圾回收
            gc.collect()
        
        monitor.take_snapshot("引擎创建完成")
        engine_memory = monitor.get_memory_usage()
        print(f"引擎创建后内存: {engine_memory['rss_mb']:.1f} MB (+{engine_memory['rss_mb'] - data_memory['rss_mb']:.1f} MB)")
        
        # 执行回测
        print("开始回测...")
        backtest_start = time.time()
        
        start_date = min(data.index[0] for data in market_data.values())
        end_date = max(data.index[-1] for data in market_data.values())
        
        results = parallel_engine.run_parallel_backtest(start_date, end_date)
        
        backtest_time = time.time() - backtest_start
        
        monitor.take_snapshot("回测完成")
        final_memory = monitor.get_memory_usage()
        print(f"回测完成后内存: {final_memory['rss_mb']:.1f} MB (+{final_memory['rss_mb'] - engine_memory['rss_mb']:.1f} MB)")
        
        # 清理内存
        del market_data
        del parallel_engine
        del results
        gc.collect()
        
        monitor.take_snapshot("清理完成")
        cleanup_memory = monitor.get_memory_usage()
        print(f"清理后内存: {cleanup_memory['rss_mb']:.1f} MB ({cleanup_memory['rss_mb'] - initial_memory['rss_mb']:.1f} MB 净增长)")
        
        # 分析内存使用
        memory_analysis = monitor.analyze_memory_growth()
        top_consumers = monitor.get_top_memory_consumers(5)
        
        print(f"\n内存分析:")
        print(f"- 峰值内存: {memory_analysis['peak_rss_mb']:.1f} MB")
        print(f"- 总增长: {memory_analysis['growth_rss_mb']:.1f} MB ({memory_analysis['growth_percent']:.1f}%)")
        print(f"- 数据生成耗时: {data_time:.2f} 秒")
        print(f"- 回测执行耗时: {backtest_time:.2f} 秒")
        
        return {
            'scenario': scenario_name,
            'num_symbols': num_symbols,
            'days': days,
            'initial_memory_mb': initial_memory['rss_mb'],
            'peak_memory_mb': memory_analysis['peak_rss_mb'],
            'final_memory_mb': cleanup_memory['rss_mb'],
            'memory_growth_mb': memory_analysis['growth_rss_mb'],
            'memory_growth_percent': memory_analysis['growth_percent'],
            'data_time': data_time,
            'backtest_time': backtest_time,
            'total_time': data_time + backtest_time,
            'top_consumers': top_consumers,
            'memory_snapshots': memory_analysis['snapshots']
        }
        
    except Exception as e:
        print(f"测试失败: {e}")
        return {
            'scenario': scenario_name,
            'error': str(e)
        }

def main():
    """主函数"""
    print("🧠 内存优化测试")
    print("=" * 60)
    
    # 测试场景
    test_scenarios = [
        (10, 252, "基准测试 (10股票, 1年)"),
        (25, 252, "中等规模 (25股票, 1年)"),
        (50, 252, "大规模 (50股票, 1年)"),
        (100, 252, "超大规模 (100股票, 1年)"),
    ]
    
    results = []
    
    for num_symbols, days, scenario_name in test_scenarios:
        result = test_memory_optimization(num_symbols, days, scenario_name)
        results.append(result)
        
        # 强制垃圾回收和短暂休息
        gc.collect()
        time.sleep(2)
    
    # 输出汇总结果
    print(f"\n{'='*80}")
    print("内存优化测试结果汇总")
    print(f"{'='*80}")
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['scenario']}: 失败 - {result['error']}")
        else:
            print(f"✅ {result['scenario']}:")
            print(f"   - 峰值内存: {result['peak_memory_mb']:.1f} MB")
            print(f"   - 内存增长: {result['memory_growth_mb']:.1f} MB ({result['memory_growth_percent']:.1f}%)")
            print(f"   - 总耗时: {result['total_time']:.2f}s")
            print(f"   - 内存效率: {result['num_symbols']/result['peak_memory_mb']:.2f} 股票/MB")
    
    # 内存优化建议
    print(f"\n💡 内存优化建议:")
    print("1. 使用 float32 而不是 float64 可节省约50%内存")
    print("2. 分批处理大数据集，避免内存峰值")
    print("3. 及时调用 gc.collect() 释放不需要的对象")
    print("4. 使用适当的数据类型 (int32 vs int64)")
    print("5. 考虑使用数据流处理而不是一次性加载所有数据")
    
    print(f"\n🎉 内存优化测试完成！")

if __name__ == "__main__":
    main()