#!/usr/bin/env python3
"""
大规模数据集性能测试脚本

测试自适应执行策略在真实大数据场景下的表现，包括：
1. 超大规模数据集测试（100+股票，2年+数据）
2. 内存使用监控
3. 执行时间对比
4. 策略选择验证
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtesting.enhanced_backtest_engine import (
    EnhancedBacktestEngine, 
    ParallelBacktestEngine,
    SimpleMovingAverageStrategy
)
from backtesting.adaptive_execution_strategy import ExecutionStrategy

def get_memory_usage() -> float:
    """获取当前内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_large_market_data(num_symbols: int, days: int, start_price: float = 100.0) -> Dict[str, pd.DataFrame]:
    """
    创建大规模市场数据
    
    Args:
        num_symbols: 股票数量
        days: 数据天数
        start_price: 起始价格
        
    Returns:
        Dict[str, pd.DataFrame]: 市场数据字典
    """
    print(f"正在生成 {num_symbols} 个股票 {days} 天的市场数据...")
    
    market_data = {}
    base_date = datetime(2020, 1, 1)
    
    for i in range(num_symbols):
        symbol = f"STOCK_{i:03d}"
        
        # 生成日期序列
        dates = pd.date_range(start=base_date, periods=days, freq='D')
        
        # 生成价格数据（随机游走）
        np.random.seed(42 + i)  # 确保可重复性
        returns = np.random.normal(0.001, 0.02, days)  # 日收益率
        
        prices = [start_price * (1 + np.random.uniform(-0.1, 0.1))]  # 起始价格加随机波动
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 生成OHLCV数据
        opens = np.array(prices)
        closes = opens * (1 + np.random.normal(0, 0.005, days))
        highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.02, days))
        lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.02, days))
        volumes = np.random.randint(100000, 1000000, days)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
        
        market_data[symbol] = df
        
        # 每生成10个股票显示进度
        if (i + 1) % 10 == 0:
            print(f"  已生成 {i + 1}/{num_symbols} 个股票数据")
    
    print(f"数据生成完成！总计 {num_symbols} 个股票，{days} 天数据")
    return market_data

def test_large_scale_scenarios():
    """测试大规模场景"""
    print("=== 大规模数据集性能测试 ===")
    print("=" * 60)
    
    # 测试场景配置
    test_scenarios = [
        {"name": "中等规模", "symbols": 20, "days": 252, "description": "20个股票，1年数据"},
        {"name": "大规模", "symbols": 50, "days": 504, "description": "50个股票，2年数据"},
        {"name": "超大规模", "symbols": 100, "days": 756, "description": "100个股票，3年数据"},
        {"name": "极大规模", "symbols": 200, "days": 504, "description": "200个股票，2年数据"},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']}测试 ({scenario['description']}) ---")
        
        # 记录初始内存
        initial_memory = get_memory_usage()
        print(f"初始内存使用: {initial_memory:.1f} MB")
        
        try:
            # 生成测试数据
            data_start_time = time.time()
            market_data = create_large_market_data(scenario['symbols'], scenario['days'])
            data_generation_time = time.time() - data_start_time
            
            # 记录数据生成后内存
            after_data_memory = get_memory_usage()
            print(f"数据生成后内存: {after_data_memory:.1f} MB (+{after_data_memory - initial_memory:.1f} MB)")
            print(f"数据生成耗时: {data_generation_time:.2f} 秒")
            
            # 创建并行回测引擎
            parallel_engine = ParallelBacktestEngine()
            
            # 为每个股票创建独立的回测引擎
            for symbol, data in market_data.items():
                # 创建单独的回测引擎
                engine = EnhancedBacktestEngine(
                    initial_capital=1000000,
                    commission_rate=0.001,
                    slippage_rate=0.001
                )
                
                # 设置市场数据
                engine.set_market_data(symbol, data)
                
                # 添加策略
                strategy = SimpleMovingAverageStrategy(f"SMA_{symbol}", short_window=10, long_window=30)
                engine.add_strategy(strategy)
                
                # 将引擎添加到并行引擎中
                parallel_engine.add_engine(symbol, engine)
            
            # 执行回测
            print("开始执行回测...")
            backtest_start_time = time.time()
            
            # 获取数据的日期范围
            start_date = min(data.index[0] for data in market_data.values())
            end_date = max(data.index[-1] for data in market_data.values())
            
            results_data = parallel_engine.run_parallel_backtest(
                start_date=start_date,
                end_date=end_date
            )
            
            backtest_execution_time = time.time() - backtest_start_time
            
            # 记录回测后内存
            after_backtest_memory = get_memory_usage()
            print(f"回测完成后内存: {after_backtest_memory:.1f} MB (+{after_backtest_memory - after_data_memory:.1f} MB)")
            
            # 获取自适应策略信息
            adaptive_strategy = parallel_engine.adaptive_strategy
            performance_history = adaptive_strategy.performance_history
            
            # 统计策略使用情况
            strategy_usage = {}
            total_executions = 0
            for key, times in performance_history.items():
                strategy_name = key.split('_')[0]
                strategy_usage[strategy_name] = strategy_usage.get(strategy_name, 0) + len(times)
                total_executions += len(times)
            
            # 记录结果
            scenario_result = {
                'name': scenario['name'],
                'symbols': scenario['symbols'],
                'days': scenario['days'],
                'data_generation_time': data_generation_time,
                'backtest_execution_time': backtest_execution_time,
                'total_time': data_generation_time + backtest_execution_time,
                'initial_memory': initial_memory,
                'peak_memory': after_backtest_memory,
                'memory_usage': after_backtest_memory - initial_memory,
                'strategy_usage': strategy_usage,
                'total_executions': total_executions,
                'success': len(results_data) > 0
            }
            
            results.append(scenario_result)
            
            # 显示结果摘要
            print(f"回测执行时间: {backtest_execution_time:.2f} 秒")
            print(f"总耗时: {scenario_result['total_time']:.2f} 秒")
            print(f"内存峰值: {after_backtest_memory:.1f} MB")
            print(f"内存增长: {scenario_result['memory_usage']:.1f} MB")
            print(f"成功回测股票数: {len(results_data)}")
            
            if strategy_usage:
                print("策略使用统计:")
                for strategy, count in strategy_usage.items():
                    percentage = (count / total_executions) * 100 if total_executions > 0 else 0
                    print(f"  {strategy}: {count}次 ({percentage:.1f}%)")
            
            # 清理内存
            del market_data
            del results_data
            gc.collect()
            
        except Exception as e:
            print(f"测试失败: {str(e)}")
            scenario_result = {
                'name': scenario['name'],
                'symbols': scenario['symbols'],
                'days': scenario['days'],
                'error': str(e),
                'success': False
            }
            results.append(scenario_result)
    
    return results

def analyze_large_scale_results(results: List[Dict]):
    """分析大规模测试结果"""
    print("\n" + "=" * 60)
    print("=== 大规模测试结果分析 ===")
    print("=" * 60)
    
    # 成功的测试
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    if successful_tests:
        print(f"\n成功测试: {len(successful_tests)}/{len(results)}")
        print("-" * 40)
        
        # 性能对比表
        print(f"{'场景':<12} {'股票数':<8} {'天数':<8} {'总耗时':<10} {'内存使用':<12} {'主要策略':<15}")
        print("-" * 80)
        
        for result in successful_tests:
            main_strategy = "N/A"
            if result.get('strategy_usage'):
                main_strategy = max(result['strategy_usage'].items(), key=lambda x: x[1])[0]
            
            print(f"{result['name']:<12} {result['symbols']:<8} {result['days']:<8} "
                  f"{result['total_time']:<10.2f} {result['memory_usage']:<12.1f} {main_strategy:<15}")
        
        # 性能趋势分析
        print(f"\n性能趋势分析:")
        print("-" * 40)
        
        # 计算复杂度与性能关系
        complexities = []
        times = []
        memories = []
        
        for result in successful_tests:
            complexity = result['symbols'] * result['days']
            complexities.append(complexity)
            times.append(result['backtest_execution_time'])
            memories.append(result['memory_usage'])
        
        if len(complexities) > 1:
            # 时间复杂度分析
            time_efficiency = [t/c for t, c in zip(times, complexities)]
            memory_efficiency = [m/c for m, c in zip(memories, complexities)]
            
            print(f"时间效率 (秒/复杂度单位):")
            for i, result in enumerate(successful_tests):
                print(f"  {result['name']}: {time_efficiency[i]*1000000:.2f} 微秒/单位")
            
            print(f"内存效率 (MB/复杂度单位):")
            for i, result in enumerate(successful_tests):
                print(f"  {result['name']}: {memory_efficiency[i]*1000:.2f} KB/单位")
        
        # 策略选择分析
        print(f"\n策略选择分析:")
        print("-" * 40)
        
        all_strategies = set()
        for result in successful_tests:
            if result.get('strategy_usage'):
                all_strategies.update(result['strategy_usage'].keys())
        
        for strategy in all_strategies:
            usage_by_scale = []
            for result in successful_tests:
                usage = result.get('strategy_usage', {}).get(strategy, 0)
                total = result.get('total_executions', 1)
                percentage = (usage / total) * 100 if total > 0 else 0
                usage_by_scale.append((result['name'], percentage))
            
            print(f"{strategy} 策略使用率:")
            for scale, percentage in usage_by_scale:
                print(f"  {scale}: {percentage:.1f}%")
    
    if failed_tests:
        print(f"\n失败测试: {len(failed_tests)}")
        print("-" * 40)
        for result in failed_tests:
            print(f"{result['name']}: {result.get('error', '未知错误')}")

def main():
    """主函数"""
    print("大规模数据集性能测试")
    print("=" * 60)
    print("本测试将验证自适应执行策略在大数据场景下的表现")
    print("包括内存使用、执行时间和策略选择的分析")
    print()
    
    # 显示系统信息
    print("系统信息:")
    print(f"  CPU核心数: {psutil.cpu_count()}")
    print(f"  可用内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    print(f"  Python进程初始内存: {get_memory_usage():.1f} MB")
    print()
    
    # 执行测试
    start_time = time.time()
    results = test_large_scale_scenarios()
    total_time = time.time() - start_time
    
    # 分析结果
    analyze_large_scale_results(results)
    
    print(f"\n总测试时间: {total_time:.2f} 秒")
    print("=" * 60)
    print("大规模测试完成！")

if __name__ == "__main__":
    main()