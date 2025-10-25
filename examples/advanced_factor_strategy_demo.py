#!/usr/bin/env python3
"""
高级因子策略演示

展示择时和择股因子的综合应用：
1. 多因子择时策略
2. 多因子择股策略  
3. 因子组合策略
4. 因子轮动策略
5. 多时间框架整合
6. 策略回测和评估
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from src.factors.timing_factors import TimingFactorCalculator
from src.factors.selection_factors import SelectionFactorCalculator
from src.factors.factor_combination import FactorCombinationStrategy
from src.backtest.engine import BacktestEngine
try:
    from src.visualization.charts import ChartGenerator
except ImportError:
    ChartGenerator = None  # 可选模块

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_data(n_stocks=50, n_days=1000):
    """
    生成示例数据 - 仅用于测试和演示
    
    Args:
        n_stocks: 股票数量
        n_days: 交易日数量
        
    Returns:
        tuple: (价格数据, 财务数据, 市场数据)
    """
    print("生成示例数据...")
    
    # 生成日期索引
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # 生成股票代码
    stock_codes = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    # 生成价格数据 - 模拟数据仅用于演示
    np.random.seed(42)
    
    # 基础价格走势
    base_trend = np.cumsum(np.random.normal(0.0005, 0.02, n_days))
    
    price_data = {}
    for stock in stock_codes:
        # 个股特异性
        stock_factor = np.random.normal(0, 0.01, n_days)
        # 行业因子
        industry_factor = np.random.normal(0, 0.005, n_days)
        
        # 合成价格
        returns = base_trend + stock_factor + industry_factor
        prices = 100 * np.exp(np.cumsum(returns))
        
        price_data[stock] = {
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_days)
        }
    
    # 转换为DataFrame格式
    price_df = {}
    for field in ['open', 'high', 'low', 'close', 'volume']:
        price_df[field] = pd.DataFrame(
            {stock: price_data[stock][field] for stock in stock_codes},
            index=dates
        )
    
    # 生成财务数据
    financial_data = {}
    for stock in stock_codes:
        financial_data[stock] = {
            'revenue': np.random.lognormal(15, 0.5, n_days),
            'net_income': np.random.lognormal(13, 0.8, n_days),
            'total_assets': np.random.lognormal(16, 0.3, n_days),
            'shareholders_equity': np.random.lognormal(15, 0.4, n_days),
            'operating_cashflow': np.random.lognormal(13.5, 0.6, n_days),
            'free_cashflow': np.random.lognormal(13, 0.7, n_days),
            'roe': np.random.normal(0.12, 0.05, n_days),
            'pe_ratio': np.random.lognormal(2.5, 0.5, n_days),
            'pb_ratio': np.random.lognormal(0.8, 0.3, n_days),
            'ps_ratio': np.random.lognormal(1.2, 0.4, n_days),
            'pcf_ratio': np.random.lognormal(1.5, 0.4, n_days),
            'debt_ratio': np.random.beta(2, 5, n_days),
            'current_ratio': np.random.gamma(2, 1, n_days),
            'gross_margin': np.random.beta(3, 2, n_days),
            'net_margin': np.random.beta(2, 3, n_days),
            'asset_turnover': np.random.gamma(1.5, 0.5, n_days),
            'inventory_turnover': np.random.gamma(2, 2, n_days),
            'receivables_turnover': np.random.gamma(3, 1, n_days),
            'eps': np.random.normal(2, 1, n_days),
            'interest_coverage': np.random.gamma(3, 2, n_days)
        }
    
    # 转换为DataFrame格式
    financial_df = {}
    for field in financial_data[stock_codes[0]].keys():
        financial_df[field] = pd.DataFrame(
            {stock: financial_data[stock][field] for stock in stock_codes},
            index=dates
        )
    
    # 市场数据（使用第一只股票作为市场指数）
    market_data = pd.DataFrame({
        'open': price_df['open'].iloc[:, 0],
        'high': price_df['high'].iloc[:, 0],
        'low': price_df['low'].iloc[:, 0],
        'close': price_df['close'].iloc[:, 0],
        'volume': price_df['volume'].iloc[:, 0]
    })
    
    print(f"生成了 {n_stocks} 只股票 {n_days} 天的数据")
    return price_df, financial_df, market_data

def demonstrate_timing_factors():
    """演示择时因子计算"""
    print("\n" + "="*50)
    print("1. 择时因子计算演示")
    print("="*50)
    
    # 生成数据
    _, _, market_data = generate_sample_data(n_stocks=10, n_days=500)
    
    # 初始化择时因子计算器
    timing_calculator = TimingFactorCalculator()
    
    # 计算各类择时因子
    print("计算市场情绪因子...")
    sentiment_factors = timing_calculator.calculate_market_sentiment_factors(market_data)
    
    print("计算技术择时因子...")
    technical_factors = timing_calculator.calculate_technical_timing_factors(market_data)
    
    print("计算波动率择时因子...")
    volatility_factors = timing_calculator.calculate_volatility_timing_factors(market_data)
    
    print("计算资金流向因子...")
    flow_factors = timing_calculator.calculate_flow_timing_factors(market_data)
    
    # 合成择时信号
    print("合成择时信号...")
    all_timing_factors = {
        **sentiment_factors,
        **technical_factors,
        **volatility_factors,
        **flow_factors
    }
    timing_signal = pd.Series(index=market_data.index, data=0.0)
    
    # 简单等权重合成
    for factor_name, factor_values in all_timing_factors.items():
        if len(factor_values) > 0:
            timing_signal += factor_values.fillna(0) / len(all_timing_factors)
    
    # 可视化结果
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('择时因子分析结果', fontsize=16, fontweight='bold')
    
    # 市场价格
    axes[0, 0].plot(market_data.index, market_data['close'])
    axes[0, 0].set_title('市场价格走势')
    axes[0, 0].set_ylabel('价格')
    
    # 择时信号
    axes[0, 1].plot(timing_signal.index, timing_signal.values)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('综合择时信号')
    axes[0, 1].set_ylabel('信号强度')
    
    # 情绪因子
    if sentiment_factors:
        factor_name = list(sentiment_factors.keys())[0]
        axes[1, 0].plot(sentiment_factors[factor_name].index, sentiment_factors[factor_name].values)
        axes[1, 0].set_title(f'市场情绪因子: {factor_name}')
        axes[1, 0].set_ylabel('因子值')
    
    # 技术因子
    if technical_factors:
        factor_name = list(technical_factors.keys())[0]
        axes[1, 1].plot(technical_factors[factor_name].index, technical_factors[factor_name].values)
        axes[1, 1].set_title(f'技术择时因子: {factor_name}')
        axes[1, 1].set_ylabel('因子值')
    
    # 波动率因子
    if volatility_factors:
        factor_name = list(volatility_factors.keys())[0]
        axes[2, 0].plot(volatility_factors[factor_name].index, volatility_factors[factor_name].values)
        axes[2, 0].set_title(f'波动率因子: {factor_name}')
        axes[2, 0].set_ylabel('因子值')
    
    # 资金流向因子
    if flow_factors:
        factor_name = list(flow_factors.keys())[0]
        axes[2, 1].plot(flow_factors[factor_name].index, flow_factors[factor_name].values)
        axes[2, 1].set_title(f'资金流向因子: {factor_name}')
        axes[2, 1].set_ylabel('因子值')
    
    plt.tight_layout()
    plt.savefig('timing_factors_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"择时因子计算完成，共生成 {len(sentiment_factors) + len(technical_factors) + len(volatility_factors) + len(flow_factors)} 个因子")
    
    return {
        'sentiment': sentiment_factors,
        'technical': technical_factors,
        'volatility': volatility_factors,
        'flow': flow_factors,
        'composite_signal': timing_signal
    }

def demonstrate_selection_factors():
    """演示择股因子计算"""
    print("\n" + "="*50)
    print("2. 择股因子计算演示")
    print("="*50)
    
    # 生成数据
    price_data, financial_data, _ = generate_sample_data(n_stocks=30, n_days=500)
    
    # 初始化择股因子计算器
    selection_calculator = SelectionFactorCalculator()
    
    # 计算各类择股因子
    print("计算质量因子...")
    quality_factors = selection_calculator.calculate_quality_factors(
        pd.DataFrame({k: v.iloc[:, 0] for k, v in financial_data.items()})
    )
    
    print("计算价值因子...")
    # 财务数据已经是DataFrame格式，直接使用
    price_sample = price_data['close'].iloc[:, :5]
    financial_sample = {k: v.iloc[:, :5] for k, v in financial_data.items()}
    
    value_factors = selection_calculator.calculate_value_factors(
        price_sample,
        financial_sample
    )
    
    print("计算成长因子...")
    growth_factors = selection_calculator.calculate_growth_factors(
        financial_sample
    )
    
    print("计算动量因子...")
    momentum_factors = selection_calculator.calculate_momentum_factors(
        price_sample
    )
    
    # 构建多因子模型
    print("构建多因子选股模型...")
    factor_data = {
        'quality': pd.DataFrame(quality_factors),
        'value': pd.DataFrame(value_factors),
        'growth': pd.DataFrame(growth_factors),
        'momentum': pd.DataFrame(momentum_factors)
    }
    
    # 生成收益数据
    returns_data = price_sample.pct_change().dropna()
    
    multifactor_model = selection_calculator.build_multifactor_model(
        factor_data, returns_data
    )
    
    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('择股因子分析结果', fontsize=16, fontweight='bold')
    
    # 因子分布
    factor_types = ['quality', 'value', 'growth', 'momentum']
    for i, factor_type in enumerate(factor_types):
        if factor_type in factor_data and not factor_data[factor_type].empty:
            row, col = i // 2, i % 2
            factor_df = factor_data[factor_type]
            
            # 绘制因子分布
            axes[row, col].hist(factor_df.values.flatten(), bins=30, alpha=0.7)
            axes[row, col].set_title(f'{factor_type.title()} 因子分布')
            axes[row, col].set_xlabel('因子值')
            axes[row, col].set_ylabel('频数')
    
    # 因子相关性
    if multifactor_model and 'processed_factors' in multifactor_model:
        processed_factors = multifactor_model['processed_factors']
        if not processed_factors.empty:
            corr_matrix = processed_factors.corr()
            
            axes[1, 2].imshow(corr_matrix.values, cmap='coolwarm', aspect='auto')
            axes[1, 2].set_title('因子相关性矩阵')
            axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
            axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
            axes[1, 2].set_xticklabels(corr_matrix.columns, rotation=45)
            axes[1, 2].set_yticklabels(corr_matrix.columns)
            
            # 添加数值标签
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha='center', va='center', fontsize=8)
    
    # 模型表现
    if multifactor_model and 'model_performance' in multifactor_model:
        performance = multifactor_model['model_performance']
        axes[0, 2].bar(performance.keys(), performance.values())
        axes[0, 2].set_title('模型表现指标')
        axes[0, 2].set_ylabel('数值')
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('selection_factors_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"择股因子计算完成")
    print(f"质量因子: {len(quality_factors)} 个")
    print(f"价值因子: {len(value_factors)} 个")
    print(f"成长因子: {len(growth_factors)} 个")
    print(f"动量因子: {len(momentum_factors)} 个")
    
    return {
        'quality': quality_factors,
        'value': value_factors,
        'growth': growth_factors,
        'momentum': momentum_factors,
        'multifactor_model': multifactor_model
    }

def demonstrate_factor_combination():
    """演示因子组合策略"""
    print("\n" + "="*50)
    print("3. 因子组合策略演示")
    print("="*50)
    
    # 生成数据
    price_data, financial_data, market_data = generate_sample_data(n_stocks=20, n_days=500)
    
    # 获取择时和择股因子
    timing_calculator = TimingFactorCalculator()
    selection_calculator = SelectionFactorCalculator()
    
    # 计算择时因子
    print("计算择时因子...")
    timing_factors = {}
    timing_factors.update(timing_calculator.calculate_market_sentiment_factors(market_data))
    timing_factors.update(timing_calculator.calculate_technical_timing_factors(market_data))
    
    # 计算择股因子
    print("计算择股因子...")
    selection_factors = {}
    
    # 使用前10只股票的数据
    stock_subset = list(price_data['close'].columns[:10])
    
    # 准备财务数据子集 - 直接使用已经是DataFrame格式的数据
    financial_subset = {}
    for key, df in financial_data.items():
        financial_subset[key] = df[stock_subset]
    
    # 合并财务数据为单个DataFrame
    financial_combined = pd.concat(financial_subset, axis=1)
    
    quality_factors = selection_calculator.calculate_quality_factors(
        financial_combined
    )
    
    value_factors = selection_calculator.calculate_value_factors(
        price_data['close'][stock_subset],
        financial_combined
    )
    
    # 处理因子数据格式
    # 确保因子数据有正确的索引和列
    stock_index = price_data['close'].index
    stock_columns = stock_subset
    
    # 将字典类型的因子转换为DataFrame，确保有正确的索引
    quality_data = {}
    for factor_name, factor_series in quality_factors.items():
        if isinstance(factor_series, pd.Series) and len(factor_series) > 0:
            # 如果Series有索引，使用它；否则创建一个
            if factor_series.index.equals(stock_index):
                quality_data[factor_name] = factor_series
            else:
                # 重新索引到股票索引
                quality_data[factor_name] = pd.Series(
                    factor_series.values[:len(stock_index)], 
                    index=stock_index[:len(factor_series)]
                )
    
    value_data = {}
    for factor_name, factor_series in value_factors.items():
        if isinstance(factor_series, pd.Series) and len(factor_series) > 0:
            if factor_series.index.equals(stock_index):
                value_data[factor_name] = factor_series
            else:
                value_data[factor_name] = pd.Series(
                    factor_series.values[:len(stock_index)], 
                    index=stock_index[:len(factor_series)]
                )
    
    # 创建DataFrame
    quality_df = pd.DataFrame(quality_data, index=stock_index) if quality_data else pd.DataFrame(index=stock_index)
    value_df = pd.DataFrame(value_data, index=stock_index) if value_data else pd.DataFrame(index=stock_index)
    
    selection_factors['quality'] = quality_df
    selection_factors['value'] = value_df
    
    # 初始化因子组合策略
    print("构建因子组合策略...")
    combination_strategy = FactorCombinationStrategy()
    
    # 构建整合策略
    integrated_strategy = combination_strategy.build_integrated_strategy(
        timing_factors=timing_factors,
        selection_factors=selection_factors,
        market_data=market_data,
        universe=stock_subset
    )
    
    # 因子轮动策略
    print("构建因子轮动策略...")
    factor_data_for_rotation = {
        'timing': pd.DataFrame(timing_factors),
        'selection_quality': quality_df,
        'selection_value': value_df
    }
    
    rotation_strategy = combination_strategy.factor_rotation_strategy(
        factor_data_for_rotation,
        lookback_window=60,
        rotation_threshold=0.1
    )
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('因子组合策略分析', fontsize=16, fontweight='bold')
    
    # 策略权重变化
    if integrated_strategy and 'dynamic_weights' in integrated_strategy:
        weights = integrated_strategy['dynamic_weights']
        if 'timing_weight' in weights and 'selection_weight' in weights:
            axes[0, 0].plot(weights['timing_weight'].index, weights['timing_weight'].values, 
                           label='择时权重', linewidth=2)
            axes[0, 0].plot(weights['selection_weight'].index, weights['selection_weight'].values, 
                           label='择股权重', linewidth=2)
            axes[0, 0].set_title('动态权重变化')
            axes[0, 0].set_ylabel('权重')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
    
    # 策略表现
    if integrated_strategy and 'portfolio' in integrated_strategy:
        portfolio = integrated_strategy['portfolio']
        if 'cumulative_returns' in portfolio:
            cumulative_returns = portfolio['cumulative_returns']
            axes[0, 1].plot(cumulative_returns.index, cumulative_returns.values, 
                           linewidth=2, color='green')
            axes[0, 1].set_title('策略累计收益')
            axes[0, 1].set_ylabel('累计收益')
            axes[0, 1].grid(True, alpha=0.3)
    
    # 轮动信号
    if rotation_strategy and 'rotation_signals' in rotation_strategy:
        signals = rotation_strategy['rotation_signals']
        signal_names = list(signals.keys())[:3]  # 显示前3个信号
        
        for i, signal_name in enumerate(signal_names):
            if signal_name in signals:
                signal_data = signals[signal_name]
                axes[1, 0].plot(signal_data.index, signal_data.values, 
                               label=signal_name, alpha=0.7)
        
        axes[1, 0].set_title('因子轮动信号')
        axes[1, 0].set_ylabel('信号强度')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 策略评估指标
    if integrated_strategy and 'strategy_evaluation' in integrated_strategy:
        evaluation = integrated_strategy['strategy_evaluation']
        metrics = list(evaluation.keys())
        values = list(evaluation.values())
        
        bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        axes[1, 1].set_title('策略评估指标')
        axes[1, 1].set_ylabel('数值')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('factor_combination_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("因子组合策略构建完成")
    
    # 打印策略评估结果
    if integrated_strategy and 'strategy_evaluation' in integrated_strategy:
        evaluation = integrated_strategy['strategy_evaluation']
        print("\n策略评估结果:")
        for metric, value in evaluation.items():
            print(f"  {metric}: {value:.4f}")
    
    return {
        'integrated_strategy': integrated_strategy,
        'rotation_strategy': rotation_strategy,
        'timing_factors': timing_factors,
        'selection_factors': selection_factors
    }

def demonstrate_factor_effectiveness():
    """演示因子有效性评估"""
    print("\n" + "="*50)
    print("4. 因子有效性评估演示")
    print("="*50)
    
    # 生成数据
    price_data, financial_data, market_data = generate_sample_data(n_stocks=15, n_days=400)
    
    # 初始化计算器
    selection_calculator = SelectionFactorCalculator()
    
    # 计算单个因子
    print("计算测试因子...")
    
    # 使用前5只股票
    stock_subset = list(price_data['close'].columns[:5])
    returns_data = price_data['close'][stock_subset].pct_change().dropna()
    
    # 计算价值因子
    # 准备财务数据子集
    financial_subset = {}
    for key, df in financial_data.items():
        financial_subset[key] = df[stock_subset]
    
    # 合并财务数据为单个DataFrame
    financial_combined = pd.concat(financial_subset, axis=1)
    
    value_factors = selection_calculator.calculate_value_factors(
        price_data['close'][stock_subset],
        financial_combined
    )
    
    # 评估因子有效性
    factor_evaluations = {}
    
    for factor_name, factor_values in value_factors.items():
        if isinstance(factor_values, pd.Series) and len(factor_values) > 50:
            print(f"评估因子: {factor_name}")
            
            # 使用第一只股票的收益作为示例
            stock_returns = returns_data.iloc[:, 0]
            
            # 对齐数据
            aligned_data = pd.concat([factor_values, stock_returns], axis=1).dropna()
            
            if len(aligned_data) > 30:
                factor_eval = selection_calculator.evaluate_factor_effectiveness(
                    aligned_data.iloc[:, 0],  # 因子值
                    aligned_data.iloc[:, 1],  # 收益率
                    periods=[1, 5, 10, 20]
                )
                factor_evaluations[factor_name] = factor_eval
    
    # 可视化因子有效性
    if factor_evaluations:
        n_factors = len(factor_evaluations)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('因子有效性评估结果', fontsize=16, fontweight='bold')
        
        # IC分析
        factor_names = list(factor_evaluations.keys())[:4]  # 最多显示4个因子
        
        for i, factor_name in enumerate(factor_names):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            evaluation = factor_evaluations[factor_name]
            
            # 提取IC值
            periods = []
            ic_values = []
            
            for period_key, period_data in evaluation.items():
                if period_key.startswith('period_') and isinstance(period_data, dict):
                    period_num = int(period_key.split('_')[1])
                    periods.append(period_num)
                    ic_values.append(period_data.get('ic_pearson', 0))
            
            if periods and ic_values:
                axes[row, col].bar(periods, ic_values, alpha=0.7)
                axes[row, col].set_title(f'{factor_name} IC分析')
                axes[row, col].set_xlabel('预测期数')
                axes[row, col].set_ylabel('IC值')
                axes[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('factor_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印评估结果摘要
        print("\n因子有效性评估摘要:")
        for factor_name, evaluation in factor_evaluations.items():
            overall_score = evaluation.get('overall_score', 0)
            print(f"  {factor_name}: 综合评分 {overall_score:.4f}")
    
    return factor_evaluations

def run_comprehensive_backtest():
    """运行综合回测"""
    print("\n" + "="*50)
    print("5. 综合策略回测演示")
    print("="*50)
    
    # 生成数据
    price_data, financial_data, market_data = generate_sample_data(n_stocks=25, n_days=600)
    
    # 构建综合策略
    print("构建综合因子策略...")
    
    # 初始化组件
    timing_calculator = TimingFactorCalculator()
    selection_calculator = SelectionFactorCalculator()
    combination_strategy = FactorCombinationStrategy()
    
    # 计算因子
    timing_factors = {}
    timing_factors.update(timing_calculator.calculate_market_sentiment_factors(market_data))
    timing_factors.update(timing_calculator.calculate_technical_timing_factors(market_data))
    
    # 择股因子（使用前15只股票）
    stock_subset = list(price_data['close'].columns[:15])
    
    selection_factors = {}
    quality_factors = selection_calculator.calculate_quality_factors(
        pd.DataFrame({k: v[stock_subset].mean(axis=1) for k, v in financial_data.items()})
    )
    selection_factors['quality'] = pd.DataFrame(quality_factors)
    
    # 构建整合策略
    integrated_strategy = combination_strategy.build_integrated_strategy(
        timing_factors=timing_factors,
        selection_factors=selection_factors,
        market_data=market_data,
        universe=stock_subset
    )
    
    # 回测分析
    print("进行回测分析...")
    
    if integrated_strategy and 'portfolio' in integrated_strategy:
        portfolio = integrated_strategy['portfolio']
        
        if 'returns' in portfolio and 'cumulative_returns' in portfolio:
            returns = portfolio['returns']
            cumulative_returns = portfolio['cumulative_returns']
            
            # 计算基准收益（等权重组合）
            benchmark_returns = price_data['close'][stock_subset].pct_change().mean(axis=1)
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            
            # 可视化回测结果
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('综合策略回测结果', fontsize=16, fontweight='bold')
            
            # 累计收益对比
            axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 
                           label='因子策略', linewidth=2, color='blue')
            axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                           label='基准策略', linewidth=2, color='red', alpha=0.7)
            axes[0, 0].set_title('累计收益对比')
            axes[0, 0].set_ylabel('累计收益')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 回撤分析
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            
            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                                   alpha=0.3, color='red')
            axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            axes[0, 1].set_title('策略回撤')
            axes[0, 1].set_ylabel('回撤幅度')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 滚动收益分析
            rolling_returns = returns.rolling(window=20).mean() * 252  # 年化收益
            axes[1, 0].plot(rolling_returns.index, rolling_returns.values, 
                           linewidth=2, color='green')
            axes[1, 0].set_title('滚动年化收益率')
            axes[1, 0].set_ylabel('年化收益率')
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 收益分布
            axes[1, 1].hist(returns.dropna(), bins=50, alpha=0.7, color='purple')
            axes[1, 1].axvline(x=returns.mean(), color='red', linestyle='--', 
                              label=f'均值: {returns.mean():.4f}')
            axes[1, 1].set_title('收益率分布')
            axes[1, 1].set_xlabel('日收益率')
            axes[1, 1].set_ylabel('频数')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('comprehensive_backtest_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 计算回测指标
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            max_drawdown = drawdown.min()
            win_rate = (returns > 0).mean()
            
            # 与基准对比
            excess_returns = returns - benchmark_returns
            information_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252) 
                               if excess_returns.std() > 0 else 0)
            
            print("\n回测结果摘要:")
            print(f"  年化收益率: {annual_return:.2%}")
            print(f"  年化波动率: {annual_vol:.2%}")
            print(f"  夏普比率: {sharpe_ratio:.4f}")
            print(f"  最大回撤: {max_drawdown:.2%}")
            print(f"  胜率: {win_rate:.2%}")
            print(f"  信息比率: {information_ratio:.4f}")
            
            return {
                'returns': returns,
                'cumulative_returns': cumulative_returns,
                'annual_return': annual_return,
                'annual_vol': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'information_ratio': information_ratio
            }
    
    return {}

def main():
    """主函数"""
    print("高级因子策略演示")
    print("="*60)
    
    try:
        # 1. 择时因子演示
        timing_results = demonstrate_timing_factors()
        
        # 2. 择股因子演示
        selection_results = demonstrate_selection_factors()
        
        # 3. 因子组合策略演示
        combination_results = demonstrate_factor_combination()
        
        # 4. 因子有效性评估
        effectiveness_results = demonstrate_factor_effectiveness()
        
        # 5. 综合回测
        backtest_results = run_comprehensive_backtest()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        
        print("\n生成的图表文件:")
        print("  - timing_factors_analysis.png: 择时因子分析")
        print("  - selection_factors_analysis.png: 择股因子分析")
        print("  - factor_combination_analysis.png: 因子组合分析")
        print("  - factor_effectiveness_analysis.png: 因子有效性分析")
        print("  - comprehensive_backtest_results.png: 综合回测结果")
        
        return {
            'timing_results': timing_results,
            'selection_results': selection_results,
            'combination_results': combination_results,
            'effectiveness_results': effectiveness_results,
            'backtest_results': backtest_results
        }
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    results = main()