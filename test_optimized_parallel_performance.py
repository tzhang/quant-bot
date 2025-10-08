#!/usr/bin/env python3
"""
测试优化后的并行回测引擎性能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 直接导入避免循环导入问题
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'backtesting'))

from enhanced_backtest_engine import EnhancedBacktestEngine, ParallelBacktestEngine, AsyncBacktestEngine, SimpleMovingAverageStrategy

def create_large_market_data(symbols, days=500):
    """创建大规模市场数据"""
    print(f"创建 {len(symbols)} 个股票 {days} 天的市场数据...")
    
    market_data = {}
    start_date = datetime.now() - timedelta(days=days)
    
    for symbol in symbols:
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        # 生成更复杂的价格数据
        np.random.seed(hash(symbol) % 2**32)  # 为每个股票设置不同的随机种子
        
        # 使用几何布朗运动生成价格
        returns = np.random.normal(0.001, 0.02, days)  # 日收益率
        prices = [100.0]  # 初始价格
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 生成成交量
        volumes = np.random.randint(10000, 100000, days)
        
        # 生成高低价
        highs = [p * (1 + np.random.uniform(0, 0.03)) for p in prices]
        lows = [p * (1 - np.random.uniform(0, 0.03)) for p in prices]
        
        market_data[symbol] = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=dates)  # 使用dates作为索引而不是列
    
    return market_data

async def test_performance_comparison():
    """测试不同回测方式的性能对比"""
    
    # 测试不同规模的数据
    test_cases = [
        {"symbols": ["AAPL", "GOOGL"], "days": 252, "name": "小规模 (2股票, 1年)"},
        {"symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"], "days": 252, "name": "中规模 (4股票, 1年)"},
        {"symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"], "days": 252, "name": "大规模 (8股票, 1年)"},
        {"symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META"], "days": 504, "name": "长期 (6股票, 2年)"}
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"测试案例: {case['name']}")
        print(f"{'='*60}")
        
        symbols = case["symbols"]
        market_data = create_large_market_data(symbols, case["days"])
        
        # 1. 顺序回测
        print("\n1. 顺序回测...")
        start_time = time.time()
        
        sequential_results = {}
        for symbol in symbols:
            engine = EnhancedBacktestEngine(initial_capital=100000)
            engine.set_market_data(symbol, market_data[symbol])
            engine.add_strategy(SimpleMovingAverageStrategy(f"MA_{symbol}", short_window=10, long_window=30))
            
            result = engine.run_backtest(
                start_date=market_data[symbol].index[0],
                end_date=market_data[symbol].index[-1]
            )
            sequential_results[symbol] = result
        
        sequential_time = time.time() - start_time
        print(f"顺序回测耗时: {sequential_time:.3f} 秒")
        
        # 2. 并行回测（进程池）
        print("\n2. 并行回测（进程池）...")
        start_time = time.time()
        
        parallel_engine = ParallelBacktestEngine(max_workers=min(4, len(symbols)))
        for symbol in symbols:
            engine = EnhancedBacktestEngine(initial_capital=100000)
            engine.set_market_data(symbol, market_data[symbol])
            engine.add_strategy(SimpleMovingAverageStrategy(f"MA_{symbol}", short_window=10, long_window=30))
            parallel_engine.add_engine(symbol, engine)
        
        parallel_results = parallel_engine.run_parallel_backtest(
            start_date=market_data[symbols[0]].index[0],
            end_date=market_data[symbols[0]].index[-1]
        )
        
        parallel_time = time.time() - start_time
        print(f"并行回测耗时: {parallel_time:.3f} 秒")
        
        # 3. 线程池回测（如果引擎数量较少）
        if len(symbols) <= 4:
            print("\n3. 线程池回测...")
            start_time = time.time()
            
            threaded_results = parallel_engine.run_threaded_backtest(
                start_date=market_data[symbols[0]].index[0],
                end_date=market_data[symbols[0]].index[-1]
            )
            
            threaded_time = time.time() - start_time
            print(f"线程池回测耗时: {threaded_time:.3f} 秒")
        else:
            threaded_time = None
        
        # 4. 异步回测
        print("\n4. 异步回测...")
        start_time = time.time()
        
        async_engine = AsyncBacktestEngine()
        for symbol in symbols:
            engine = EnhancedBacktestEngine(initial_capital=100000)
            engine.set_market_data(symbol, market_data[symbol])
            engine.add_strategy(SimpleMovingAverageStrategy(f"MA_{symbol}", short_window=10, long_window=30))
            async_engine.add_engine(symbol, engine)
        
        async_results = await async_engine.run_async_backtest(
            start_date=list(market_data.values())[0].index[0],
            end_date=list(market_data.values())[0].index[-1]
        )
        
        async_time = time.time() - start_time
        print(f"异步回测耗时: {async_time:.3f} 秒")
        
        # 计算性能提升
        parallel_speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        async_speedup = sequential_time / async_time if async_time > 0 else 0
        threaded_speedup = sequential_time / threaded_time if threaded_time and threaded_time > 0 else None
        
        print(f"\n性能对比:")
        print(f"  并行回测性能提升: {parallel_speedup:.2f}x")
        if threaded_speedup:
            print(f"  线程池回测性能提升: {threaded_speedup:.2f}x")
        print(f"  异步回测性能提升: {async_speedup:.2f}x")
        
        # 验证结果一致性
        print(f"\n结果验证:")
        for symbol in symbols:
            seq_value = sequential_results[symbol]['final_portfolio_value']
            par_value = parallel_results[symbol]['final_portfolio_value']
            async_value = async_results[symbol]['final_portfolio_value']
            
            print(f"  {symbol}: 顺序={seq_value:.2f}, 并行={par_value:.2f}, 异步={async_value:.2f}")
        
        # 保存结果
        case_result = {
            'name': case['name'],
            'symbols_count': len(symbols),
            'days': case['days'],
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'threaded_time': threaded_time,
            'async_time': async_time,
            'parallel_speedup': parallel_speedup,
            'threaded_speedup': threaded_speedup,
            'async_speedup': async_speedup
        }
        results.append(case_result)
    
    # 总结报告
    print(f"\n{'='*80}")
    print("性能测试总结报告")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  数据规模: {result['symbols_count']} 股票 × {result['days']} 天")
        print(f"  顺序回测: {result['sequential_time']:.3f}s")
        print(f"  并行回测: {result['parallel_time']:.3f}s (提升 {result['parallel_speedup']:.2f}x)")
        if result['threaded_speedup']:
            print(f"  线程回测: {result['threaded_time']:.3f}s (提升 {result['threaded_speedup']:.2f}x)")
        print(f"  异步回测: {result['async_time']:.3f}s (提升 {result['async_speedup']:.2f}x)")
    
    return results

if __name__ == "__main__":
    print("开始优化后的并行回测性能测试...")
    import asyncio
    results = asyncio.run(test_performance_comparison())
    print("\n测试完成！")