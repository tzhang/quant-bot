#!/usr/bin/env python3
"""
简化的大规模数据集性能测试
用于验证自适应执行策略在大数据场景下的表现
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtesting.enhanced_backtest_engine import (
    EnhancedBacktestEngine, 
    ParallelBacktestEngine,
    SimpleMovingAverageStrategy
)

def create_test_data(num_symbols: int, days: int) -> Dict[str, pd.DataFrame]:
    """创建测试数据"""
    print(f"创建 {num_symbols} 个股票 {days} 天的数据...")
    
    market_data = {}
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_symbols):
        symbol = f"STOCK_{i:03d}"
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        # 生成价格数据
        np.random.seed(i)  # 每个股票使用不同的种子
        returns = np.random.normal(0.001, 0.02, days)
        prices = [100.0]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        market_data[symbol] = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'close': prices,
            'volume': np.random.randint(10000, 100000, days)
        }, index=dates)
    
    return market_data

def test_single_scenario(num_symbols: int, days: int, scenario_name: str):
    """测试单个场景"""
    print(f"\n{'='*60}")
    print(f"测试场景: {scenario_name}")
    print(f"股票数量: {num_symbols}, 数据天数: {days}")
    print(f"{'='*60}")
    
    try:
        # 生成数据
        start_time = time.time()
        market_data = create_test_data(num_symbols, days)
        data_time = time.time() - start_time
        print(f"数据生成耗时: {data_time:.2f} 秒")
        
        # 创建并行回测引擎
        parallel_engine = ParallelBacktestEngine()
        
        # 为每个股票创建回测引擎
        for symbol, data in market_data.items():
            engine = EnhancedBacktestEngine(
                initial_capital=1000000,
                commission_rate=0.001,
                slippage_rate=0.001
            )
            engine.set_market_data(symbol, data)
            
            strategy = SimpleMovingAverageStrategy(f"SMA_{symbol}", short_window=10, long_window=30)
            engine.add_strategy(strategy)
            
            parallel_engine.add_engine(symbol, engine)
        
        # 执行回测
        print("开始回测...")
        backtest_start = time.time()
        
        start_date = min(data.index[0] for data in market_data.values())
        end_date = max(data.index[-1] for data in market_data.values())
        
        results = parallel_engine.run_parallel_backtest(start_date, end_date)
        
        backtest_time = time.time() - backtest_start
        print(f"回测执行耗时: {backtest_time:.2f} 秒")
        
        # 获取自适应策略信息
        adaptive_strategy = parallel_engine.adaptive_strategy
        
        # 计算任务复杂度
        task_metrics = adaptive_strategy.calculate_task_complexity(
            num_symbols=num_symbols,
            data_length=days,
            num_strategies=1
        )
        
        # 选择的执行策略
        chosen_strategy = adaptive_strategy.choose_execution_strategy(task_metrics)
        
        print(f"任务复杂度: {task_metrics.complexity_score:.2f}")
        print(f"选择的执行策略: {chosen_strategy.value}")
        print(f"成功完成 {len(results)} 个股票的回测")
        
        # 统计结果
        successful_results = sum(1 for r in results.values() if 'error' not in r)
        print(f"成功率: {successful_results}/{len(results)} ({successful_results/len(results)*100:.1f}%)")
        
        return {
            'scenario': scenario_name,
            'num_symbols': num_symbols,
            'days': days,
            'data_time': data_time,
            'backtest_time': backtest_time,
            'total_time': data_time + backtest_time,
            'complexity': task_metrics.complexity_score,
            'strategy': chosen_strategy.value,
            'success_rate': successful_results/len(results),
            'successful_count': successful_results
        }
        
    except Exception as e:
        print(f"测试失败: {e}")
        return {
            'scenario': scenario_name,
            'error': str(e)
        }

def main():
    """主函数"""
    print("🚀 大规模数据集性能测试")
    print("=" * 60)
    
    # 测试场景
    test_scenarios = [
        (5, 100, "小规模测试 (5股票, 100天)"),
        (10, 252, "中规模测试 (10股票, 1年)"),
        (20, 252, "大规模测试 (20股票, 1年)"),
        (50, 252, "超大规模测试 (50股票, 1年)"),
    ]
    
    results = []
    
    for num_symbols, days, scenario_name in test_scenarios:
        result = test_single_scenario(num_symbols, days, scenario_name)
        results.append(result)
        
        # 短暂休息，释放内存
        time.sleep(1)
    
    # 输出汇总结果
    print(f"\n{'='*80}")
    print("测试结果汇总")
    print(f"{'='*80}")
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['scenario']}: 失败 - {result['error']}")
        else:
            print(f"✅ {result['scenario']}:")
            print(f"   - 总耗时: {result['total_time']:.2f}s")
            print(f"   - 回测耗时: {result['backtest_time']:.2f}s")
            print(f"   - 任务复杂度: {result['complexity']:.2f}")
            print(f"   - 执行策略: {result['strategy']}")
            print(f"   - 成功率: {result['success_rate']*100:.1f}%")
    
    print(f"\n🎉 测试完成！")

if __name__ == "__main__":
    main()