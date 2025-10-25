#!/usr/bin/env python3
"""
自适应执行策略测试脚本
测试不同规模任务下的自动策略选择功能
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtesting.enhanced_backtest_engine import (
    EnhancedBacktestEngine, 
    ParallelBacktestEngine,
    SimpleMovingAverageStrategy
)
from backtesting.adaptive_execution_strategy import ExecutionStrategy

def create_test_data(symbol: str, days: int, start_price: float = 100.0) -> pd.DataFrame:
    """创建测试市场数据 - 仅用于测试和演示"""
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=days, freq='D')
    
    # 生成随机价格数据 - 模拟数据仅用于演示
    np.random.seed(hash(symbol) % 2**32)  # 为每个symbol使用不同的随机种子
    returns = np.random.normal(0.0005, 0.02, days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, days))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, days)
    }, index=dates)

def test_adaptive_strategy_selection():
    """测试自适应策略选择功能"""
    print("=== 自适应执行策略测试 ===\n")
    
    # 测试场景配置
    test_scenarios = [
        {
            'name': '小规模任务 (1个资产, 30天)',
            'symbols': ['STOCK_A'],
            'days': 30,
            'expected_strategy': ExecutionStrategy.SEQUENTIAL
        },
        {
            'name': '中等规模任务 (3个资产, 90天)',
            'symbols': ['STOCK_A', 'STOCK_B', 'STOCK_C'],
            'days': 90,
            'expected_strategy': ExecutionStrategy.THREADED
        },
        {
            'name': '大规模任务 (5个资产, 365天)',
            'symbols': ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E'],
            'days': 365,
            'expected_strategy': ExecutionStrategy.ASYNC
        },
        {
            'name': '超大规模任务 (8个资产, 730天)',
            'symbols': [f'STOCK_{chr(65+i)}' for i in range(8)],
            'days': 730,
            'expected_strategy': ExecutionStrategy.PARALLEL
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"测试场景: {scenario['name']}")
        print("-" * 50)
        
        # 创建并行回测引擎
        parallel_engine = ParallelBacktestEngine(max_workers=4)
        
        # 为每个资产创建引擎和数据
        for symbol in scenario['symbols']:
            engine = EnhancedBacktestEngine(
                initial_capital=1000000,
                commission_rate=0.001,
                slippage_rate=0.001
            )
            
            # 创建市场数据
            market_data = create_test_data(symbol, scenario['days'])
            engine.set_market_data(symbol, market_data)
            
            # 添加策略
            strategy = SimpleMovingAverageStrategy(
                f'MA_STRATEGY_{symbol}', 
                short_window=10, 
                long_window=30
            )
            engine.add_strategy(strategy)
            
            # 添加到并行引擎
            parallel_engine.add_engine(symbol, engine)
        
        # 运行回测并记录时间
        start_date = datetime(2023, 1, 1)
        end_date = start_date + timedelta(days=scenario['days']-1)
        
        start_time = time.time()
        backtest_results = parallel_engine.run_parallel_backtest(start_date, end_date)
        execution_time = time.time() - start_time
        
        # 获取实际使用的策略
        actual_strategy = parallel_engine.adaptive_strategy.choose_execution_strategy(
            parallel_engine.adaptive_strategy.calculate_task_complexity(
                num_symbols=len(scenario['symbols']),
                data_length=scenario['days'],
                num_strategies=1
            )
        )
        
        # 记录结果
        scenario_result = {
            'scenario': scenario['name'],
            'symbols_count': len(scenario['symbols']),
            'days': scenario['days'],
            'expected_strategy': scenario['expected_strategy'].value,
            'actual_strategy': actual_strategy.value,
            'execution_time': execution_time,
            'strategy_match': actual_strategy == scenario['expected_strategy'],
            'success_count': sum(1 for result in backtest_results.values() 
                               if 'error' not in result)
        }
        
        results.append(scenario_result)
        
        # 打印结果
        print(f"资产数量: {scenario_result['symbols_count']}")
        print(f"数据天数: {scenario_result['days']}")
        print(f"预期策略: {scenario_result['expected_strategy']}")
        print(f"实际策略: {scenario_result['actual_strategy']}")
        print(f"策略匹配: {'✓' if scenario_result['strategy_match'] else '✗'}")
        print(f"执行时间: {scenario_result['execution_time']:.3f}秒")
        print(f"成功回测: {scenario_result['success_count']}/{len(scenario['symbols'])}")
        print()
    
    # 汇总结果
    print("=== 测试结果汇总 ===")
    print("-" * 50)
    
    total_tests = len(results)
    successful_matches = sum(1 for r in results if r['strategy_match'])
    successful_backtests = sum(r['success_count'] for r in results)
    total_backtests = sum(r['symbols_count'] for r in results)
    
    print(f"总测试场景: {total_tests}")
    print(f"策略选择正确: {successful_matches}/{total_tests} ({successful_matches/total_tests*100:.1f}%)")
    print(f"回测成功率: {successful_backtests}/{total_backtests} ({successful_backtests/total_backtests*100:.1f}%)")
    
    # 性能分析
    print("\n=== 性能分析 ===")
    print("-" * 50)
    for result in results:
        complexity_score = result['symbols_count'] * result['days']
        efficiency = complexity_score / result['execution_time'] if result['execution_time'] > 0 else 0
        print(f"{result['scenario'][:30]:<30} | "
              f"复杂度: {complexity_score:>8} | "
              f"时间: {result['execution_time']:>6.3f}s | "
              f"效率: {efficiency:>8.0f}")
    
    return results

def test_adaptive_learning():
    """测试自适应学习功能"""
    print("\n=== 自适应学习测试 ===")
    print("-" * 50)
    
    # 创建并行引擎
    parallel_engine = ParallelBacktestEngine(max_workers=4)
    
    # 创建测试引擎
    engine = EnhancedBacktestEngine(initial_capital=1000000)
    market_data = create_test_data('TEST_STOCK', 100)
    engine.set_market_data('TEST_STOCK', market_data)
    engine.add_strategy(SimpleMovingAverageStrategy('TEST_STRATEGY'))
    parallel_engine.add_engine('TEST_STOCK', engine)
    
    # 多次运行相同任务，观察策略选择的变化
    print("运行多次相同任务，观察策略选择变化:")
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 4, 10)  # 100天
    
    for i in range(5):
        print(f"\n第 {i+1} 次运行:")
        
        # 获取当前推荐策略
        task_metrics = parallel_engine.adaptive_strategy.calculate_task_complexity(
            num_symbols=1, data_length=100, num_strategies=1
        )
        recommended_strategy = parallel_engine.adaptive_strategy.choose_execution_strategy(task_metrics)
        
        print(f"  推荐策略: {recommended_strategy.value}")
        
        # 运行回测
        start_time = time.time()
        results = parallel_engine.run_parallel_backtest(start_date, end_date)
        execution_time = time.time() - start_time
        
        print(f"  执行时间: {execution_time:.3f}秒")
        print(f"  回测成功: {'✓' if 'error' not in results.get('TEST_STOCK', {}) else '✗'}")
        
        # 显示性能历史
        performance_history = parallel_engine.adaptive_strategy.performance_history
        if performance_history:
            print(f"  历史记录数: {len(performance_history)}")
            avg_times = {}
            for key, times in performance_history.items():
                # key格式: "strategy_complexity"，例如 "sequential_100"
                strategy = key.split('_')[0]
                if strategy not in avg_times:
                    avg_times[strategy] = []
                avg_times[strategy].extend(times)
            
            for strategy, times in avg_times.items():
                avg_time = sum(times) / len(times)
                print(f"    {strategy}: 平均 {avg_time:.3f}秒 ({len(times)}次)")

if __name__ == "__main__":
    # 运行测试
    results = test_adaptive_strategy_selection()
    test_adaptive_learning()
    
    print("\n=== 测试完成 ===")
    print("自适应执行策略功能测试完成！")