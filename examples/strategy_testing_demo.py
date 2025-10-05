#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略测试演示脚本
展示如何使用策略测试器进行批量策略测试和性能分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data.data_manager import DataManager
from src.strategies.strategy_tester import StrategyTester
from src.visualization.strategy_dashboard import StrategyDashboard


def generate_sample_data(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    symbol: str = "AAPL"
) -> pd.DataFrame:
    """
    生成示例数据（如果无法获取真实数据）
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        symbol: 股票代码
        
    Returns:
        示例数据
    """
    print(f"生成 {symbol} 的示例数据...")
    
    # 创建日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 生成随机价格数据（模拟股价走势）
    np.random.seed(42)  # 确保结果可重现
    
    # 初始价格
    initial_price = 100.0
    
    # 生成收益率（带有趋势和波动）
    n_days = len(date_range)
    trend = 0.0002  # 日均趋势
    volatility = 0.02  # 日波动率
    
    returns = np.random.normal(trend, volatility, n_days)
    
    # 添加一些周期性和趋势
    for i in range(n_days):
        # 添加周期性（模拟季节性效应）
        cycle_effect = 0.001 * np.sin(2 * np.pi * i / 252)  # 年度周期
        returns[i] += cycle_effect
        
        # 添加动量效应
        if i > 20:
            momentum = np.mean(returns[i-20:i]) * 0.1
            returns[i] += momentum
    
    # 计算价格
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据
    data = []
    for i, (date, price) in enumerate(zip(date_range, prices)):
        # 生成日内波动
        daily_vol = volatility * 0.5
        high = price * (1 + np.random.uniform(0, daily_vol))
        low = price * (1 - np.random.uniform(0, daily_vol))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        
        # 生成成交量（与价格变化相关）
        price_change = abs(returns[i]) if i < len(returns) else 0.01
        base_volume = 1000000
        volume = int(base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 2.0))
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    # 只保留工作日
    df = df[df.index.dayofweek < 5]
    
    print(f"生成了 {len(df)} 天的数据")
    return df


def run_strategy_testing_demo():
    """
    运行策略测试演示
    """
    print("=" * 80)
    print("量化策略测试演示")
    print("=" * 80)
    print()
    
    # 1. 获取数据
    print("1. 获取测试数据...")
    
    try:
        # 尝试获取真实数据
        data_manager = DataManager()
        data = data_manager.get_data("AAPL", start_date="2020-01-01", end_date="2023-12-31")
        
        # 检查数据是否为空
        if data is None or len(data) == 0:
            raise ValueError("获取的数据为空")
            
        print("成功获取真实数据")
    except Exception as e:
        print(f"获取真实数据失败: {e}")
        print("使用示例数据...")
        data = generate_sample_data()
    
    print(f"数据期间: {data.index[0]} 至 {data.index[-1]}")
    print(f"数据点数: {len(data)}")
    print()
    
    # 2. 创建策略测试器
    print("2. 初始化策略测试器...")
    tester = StrategyTester(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005,
        test_period_ratio=0.3
    )
    print("策略测试器初始化完成")
    print()
    
    # 3. 运行综合测试
    print("3. 开始策略测试...")
    
    # 创建结果保存目录
    results_dir = "strategy_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行测试
    comprehensive_results = tester.run_comprehensive_test(data, results_dir)
    
    if not comprehensive_results:
        print("测试失败，没有获得结果")
        return
    
    results = comprehensive_results['results']
    comparison_df = comprehensive_results['comparison_df']
    
    print()
    print("4. 测试结果摘要:")
    print("-" * 50)
    print(f"成功测试策略数量: {len(results)}")
    
    if len(results) > 0:
        best_strategy = comparison_df.iloc[0]
        print(f"最佳策略: {best_strategy['策略名称']}")
        print(f"  - 夏普比率: {best_strategy['夏普比率']:.3f}")
        print(f"  - 总收益率: {best_strategy['总收益率(%)']:.2f}%")
        print(f"  - 最大回撤: {best_strategy['最大回撤(%)']:.2f}%")
        
        worst_strategy = comparison_df.iloc[-1]
        print(f"最差策略: {worst_strategy['策略名称']}")
        print(f"  - 夏普比率: {worst_strategy['夏普比率']:.3f}")
        print(f"  - 总收益率: {worst_strategy['总收益率(%)']:.2f}%")
        print(f"  - 最大回撤: {worst_strategy['最大回撤(%)']:.2f}%")
    
    print()
    
    # 5. 创建可视化仪表板
    print("5. 生成可视化报告...")
    
    dashboard = StrategyDashboard()
    
    # 生成交互式仪表板
    dashboard_path = os.path.join(results_dir, "strategy_dashboard.html")
    dashboard.create_comprehensive_dashboard(results, dashboard_path)
    
    # 导出图表为图片
    charts_dir = os.path.join(results_dir, "charts")
    dashboard.export_charts_to_images(results, charts_dir)
    
    print()
    print("6. 单个策略详细分析演示...")
    
    if len(results) > 0:
        # 选择最佳策略进行详细分析
        best_strategy_name = comparison_df.iloc[0]['策略名称']
        best_result = results[best_strategy_name]
        
        print(f"分析策略: {best_strategy_name}")
        
        # 创建月度收益热力图
        heatmap_chart = dashboard.create_monthly_returns_heatmap(
            best_strategy_name, best_result
        )
        heatmap_path = os.path.join(results_dir, f"{best_strategy_name}_monthly_heatmap.html")
        heatmap_chart.write_html(heatmap_path)
        
        # 创建滚动指标图表
        rolling_chart = dashboard.create_rolling_metrics_chart(
            best_strategy_name, best_result, window=60
        )
        rolling_path = os.path.join(results_dir, f"{best_strategy_name}_rolling_metrics.html")
        rolling_chart.write_html(rolling_path)
        
        print(f"详细分析图表已保存")
    
    print()
    print("7. 策略类型性能分析...")
    
    # 按策略类型分组分析
    strategy_types = {}
    for strategy_name, result in results.items():
        strategy_type = strategy_name.split('_')[0]
        if strategy_type not in strategy_types:
            strategy_types[strategy_type] = []
        
        metrics = result['performance_metrics']
        strategy_types[strategy_type].append({
            'name': strategy_name,
            'sharpe': metrics.get('sharpe_ratio', 0),
            'return': metrics.get('total_return', 0),
            'drawdown': metrics.get('max_drawdown', 0)
        })
    
    print("策略类型表现:")
    for strategy_type, strategies in strategy_types.items():
        avg_sharpe = np.mean([s['sharpe'] for s in strategies])
        avg_return = np.mean([s['return'] for s in strategies])
        avg_drawdown = np.mean([abs(s['drawdown']) for s in strategies])
        
        print(f"  {strategy_type:15}: 平均夏普 {avg_sharpe:6.3f} | "
              f"平均收益 {avg_return:7.2f}% | 平均回撤 {avg_drawdown:6.2f}%")
    
    print()
    print("=" * 80)
    print("策略测试演示完成!")
    print("=" * 80)
    print(f"结果文件保存在: {os.path.abspath(results_dir)}")
    print("主要文件:")
    print(f"  - 策略对比表: strategy_comparison.csv")
    print(f"  - 测试报告: strategy_test_report.txt")
    print(f"  - 交互式仪表板: strategy_dashboard.html")
    print(f"  - 图表文件夹: charts/")
    print()
    
    return comprehensive_results


def run_custom_strategy_test():
    """
    运行自定义策略测试演示
    """
    print("\n" + "=" * 60)
    print("自定义策略测试演示")
    print("=" * 60)
    
    # 获取数据
    try:
        data_manager = DataManager()
        data = data_manager.get_data("AAPL", start_date="2022-01-01", end_date="2023-12-31")
        
        # 检查数据是否为空
        if data is None or len(data) == 0:
            raise ValueError("获取的数据为空")
    except:
        data = generate_sample_data("2022-01-01", "2023-12-31")
    
    # 创建测试器
    tester = StrategyTester(initial_capital=50000.0)
    
    # 测试特定策略组合
    custom_strategies = ['MeanReversion_Conservative', 'Momentum_Standard', 'RSI_Oversold']
    
    print(f"测试自定义策略组合: {custom_strategies}")
    
    results = {}
    for strategy_name in custom_strategies:
        try:
            result = tester.test_single_strategy(strategy_name, data)
            results[strategy_name] = result
            
            metrics = result['performance_metrics']
            print(f"\n{strategy_name}:")
            print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  总收益率: {metrics.get('total_return', 0):.2f}%")
            print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
            
        except Exception as e:
            print(f"策略 {strategy_name} 测试失败: {e}")
    
    # 生成对比
    if results:
        comparison_df = tester.compare_strategies(results)
        print(f"\n策略排名:")
        print(comparison_df[['策略名称', '夏普比率', '总收益率(%)', '最大回撤(%)']].to_string(index=False))
    
    return results


if __name__ == "__main__":
    # 运行完整演示
    print("选择演示模式:")
    print("1. 完整策略测试演示")
    print("2. 自定义策略测试演示")
    print("3. 两个都运行")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        run_strategy_testing_demo()
    elif choice == "2":
        run_custom_strategy_test()
    elif choice == "3":
        run_strategy_testing_demo()
        run_custom_strategy_test()
    else:
        print("运行默认完整演示...")
        run_strategy_testing_demo()