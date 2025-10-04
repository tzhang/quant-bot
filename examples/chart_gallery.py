#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图表画廊演示脚本
================

这个脚本展示了量化交易系统中各种图表的生成和使用方法。
包括技术分析图表、因子分析图表、策略回测图表等。

作者: 量化交易系统
日期: 2025-01-27
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manager import DataManager
from src.factors.technical import TechnicalFactors
from src.factors.engine import FactorEngine
from src.performance.analyzer import PerformanceAnalyzer

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_info(message):
    """打印信息"""
    print(f"ℹ️  {message}")

def print_success(message):
    """打印成功信息"""
    print(f"✅ {message}")

def create_sample_stock_data(symbol="AAPL", days=126):
    """
    创建模拟股票数据用于演示
    
    Args:
        symbol: 股票代码
        days: 数据天数
        
    Returns:
        包含OHLCV数据的DataFrame
    """
    # 设置随机种子以获得可重复的结果
    np.random.seed(42)
    
    # 生成日期索引
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # 只保留工作日
    
    # 生成价格数据
    initial_price = 250.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # 日收益率
    
    # 计算价格序列
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # 生成OHLC数据
    data = []
    for i, price in enumerate(prices):
        # 生成日内波动
        high_factor = 1 + abs(np.random.normal(0, 0.01))
        low_factor = 1 - abs(np.random.normal(0, 0.01))
        
        high = price * high_factor
        low = price * low_factor
        
        # 开盘价基于前一日收盘价
        if i == 0:
            open_price = price
        else:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        
        # 确保价格逻辑正确
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        # 生成成交量
        volume = int(np.random.normal(50000000, 10000000))
        volume = max(volume, 1000000)  # 最小成交量
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates[:len(data)])
    return df

def create_technical_analysis_charts():
    """创建技术分析图表"""
    print_header("技术分析图表演示")
    
    # 创建模拟数据
    symbol = "AAPL"
    print_info(f"生成 {symbol} 的模拟历史数据...")
    data = create_sample_stock_data(symbol, days=180)
    
    print_info(f"数据列: {list(data.columns)}")
    print_info(f"数据形状: {data.shape}")
    print_info(f"数据范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
    
    # 计算技术指标
    tech_calculator = TechnicalFactors()
    tech_data = tech_calculator.calculate_all_factors(data)
    
    print_info(f"技术指标计算完成，新增列: {len(tech_data.columns) - len(data.columns)} 个")
    
    # 创建技术分析综合图表
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{symbol} 技术分析图表集合 (模拟数据)', fontsize=16, fontweight='bold')
    
    # 1. 价格和移动平均线
    ax1 = axes[0, 0]
    ax1.plot(tech_data.index, tech_data['Close'], label='收盘价', linewidth=2, color='#2E86AB')
    ax1.plot(tech_data.index, tech_data['SMA20'], label='SMA20', alpha=0.8, color='#A23B72')
    ax1.plot(tech_data.index, tech_data['EMA20'], label='EMA20', alpha=0.8, color='#F18F01')
    ax1.set_title('价格与移动平均线', fontweight='bold')
    ax1.set_ylabel('价格 ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 布林带
    ax2 = axes[0, 1]
    ax2.plot(tech_data.index, tech_data['Close'], label='收盘价', linewidth=2, color='#2E86AB')
    ax2.fill_between(tech_data.index, tech_data['BBL_20_2.0_2.0'], tech_data['BBU_20_2.0_2.0'], 
                     alpha=0.2, color='#A23B72', label='布林带')
    ax2.plot(tech_data.index, tech_data['BBM_20_2.0_2.0'], '--', alpha=0.8, color='#A23B72', label='中轨')
    ax2.set_title('布林带指标', fontweight='bold')
    ax2.set_ylabel('价格 ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. RSI指标
    ax3 = axes[1, 0]
    ax3.plot(tech_data.index, tech_data['RSI14'], linewidth=2, color='#F18F01')
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线(70)')
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线(30)')
    ax3.fill_between(tech_data.index, 30, 70, alpha=0.1, color='gray')
    ax3.set_title('RSI相对强弱指标', fontweight='bold')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. MACD指标
    ax4 = axes[1, 1]
    ax4.plot(tech_data.index, tech_data['MACD_12_26_9'], label='MACD', linewidth=2, color='#2E86AB')
    ax4.plot(tech_data.index, tech_data['MACDs_12_26_9'], label='信号线', linewidth=2, color='#A23B72')
    ax4.bar(tech_data.index, tech_data['MACDh_12_26_9'], label='MACD柱', alpha=0.6, color='#F18F01')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('MACD指标', fontweight='bold')
    ax4.set_ylabel('MACD')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 成交量
    ax5 = axes[2, 0]
    colors = ['red' if close >= open_ else 'green' for close, open_ in zip(tech_data['Close'], tech_data['Open'])]
    ax5.bar(tech_data.index, tech_data['Volume'], color=colors, alpha=0.6)
    ax5.set_title('成交量', fontweight='bold')
    ax5.set_ylabel('成交量')
    ax5.grid(True, alpha=0.3)
    
    # 6. 价格变化率
    ax6 = axes[2, 1]
    returns = tech_data['Close'].pct_change() * 100
    ax6.plot(tech_data.index, returns, linewidth=1, color='#2E86AB', alpha=0.7)
    ax6.fill_between(tech_data.index, returns, 0, where=(returns > 0), color='red', alpha=0.3, label='上涨')
    ax6.fill_between(tech_data.index, returns, 0, where=(returns <= 0), color='green', alpha=0.3, label='下跌')
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.set_title('日收益率', fontweight='bold')
    ax6.set_ylabel('收益率 (%)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = "examples/technical_analysis_gallery.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_success(f"技术分析图表已保存: {chart_path}")
    return tech_data

def create_factor_analysis_charts(tech_data):
    """创建因子分析图表"""
    print_header("因子分析图表演示")
    
    # 计算额外的因子
    print_info("计算动量和波动率因子...")
    
    # 动量因子
    tech_data['momentum_5d'] = tech_data['Close'].pct_change(5)
    tech_data['momentum_20d'] = tech_data['Close'].pct_change(20)
    
    # 波动率因子
    tech_data['volatility_5d'] = tech_data['Close'].pct_change().rolling(5).std()
    tech_data['volatility_20d'] = tech_data['Close'].pct_change().rolling(20).std()
    
    # 计算未来收益（用于IC分析）
    tech_data['future_1d'] = tech_data['Close'].shift(-1) / tech_data['Close'] - 1
    tech_data['future_5d'] = tech_data['Close'].shift(-5) / tech_data['Close'] - 1
    
    # 创建因子分析图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('因子分析图表集合 (模拟数据)', fontsize=16, fontweight='bold')
    
    # 1. 动量因子分布
    ax1 = axes[0, 0]
    ax1.hist(tech_data['momentum_20d'].dropna(), bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.set_title('20日动量因子分布', fontweight='bold')
    ax1.set_xlabel('动量值')
    ax1.set_ylabel('频次')
    ax1.grid(True, alpha=0.3)
    
    # 2. 波动率因子时序
    ax2 = axes[0, 1]
    ax2.plot(tech_data.index, tech_data['volatility_20d'], linewidth=2, color='#A23B72')
    ax2.set_title('20日波动率因子时序', fontweight='bold')
    ax2.set_ylabel('波动率')
    ax2.grid(True, alpha=0.3)
    
    # 3. RSI因子与未来收益散点图
    ax3 = axes[0, 2]
    valid_data = tech_data[['RSI14', 'future_5d']].dropna()
    ax3.scatter(valid_data['RSI14'], valid_data['future_5d'], alpha=0.6, color='#F18F01')
    ax3.set_title('RSI vs 未来5日收益', fontweight='bold')
    ax3.set_xlabel('RSI14')
    ax3.set_ylabel('未来5日收益')
    ax3.grid(True, alpha=0.3)
    
    # 计算相关性
    correlation = valid_data['RSI14'].corr(valid_data['future_5d'])
    ax3.text(0.05, 0.95, f'相关性: {correlation:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. 因子IC分析
    ax4 = axes[1, 0]
    factors = ['RSI14', 'momentum_20d', 'volatility_20d']
    ic_1d = []
    ic_5d = []
    
    for factor in factors:
        valid_data = tech_data[[factor, 'future_1d', 'future_5d']].dropna()
        if len(valid_data) > 10:
            ic_1d.append(valid_data[factor].corr(valid_data['future_1d']))
            ic_5d.append(valid_data[factor].corr(valid_data['future_5d']))
        else:
            ic_1d.append(0)
            ic_5d.append(0)
    
    x = np.arange(len(factors))
    width = 0.35
    
    ax4.bar(x - width/2, ic_1d, width, label='1日IC', alpha=0.8, color='#2E86AB')
    ax4.bar(x + width/2, ic_5d, width, label='5日IC', alpha=0.8, color='#A23B72')
    ax4.set_title('因子IC分析', fontweight='bold')
    ax4.set_ylabel('IC值')
    ax4.set_xticks(x)
    ax4.set_xticklabels(factors, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 5. 因子分层回测
    ax5 = axes[1, 1]
    factor_name = 'RSI14'
    valid_data = tech_data[[factor_name, 'future_5d']].dropna()
    
    if len(valid_data) > 50:
        # 将因子值分为5层
        valid_data['quintile'] = pd.qcut(valid_data[factor_name], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        quintile_returns = valid_data.groupby('quintile')['future_5d'].mean()
        
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        bars = ax5.bar(quintile_returns.index, quintile_returns.values, 
                      color=colors, alpha=0.8, edgecolor='black')
        ax5.set_title(f'{factor_name} 分层回测 (5日收益)', fontweight='bold')
        ax5.set_ylabel('平均收益')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, quintile_returns.values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 6. 因子热力图
    ax6 = axes[1, 2]
    factor_cols = ['RSI14', 'momentum_5d', 'momentum_20d', 'volatility_5d', 'volatility_20d']
    factor_data = tech_data[factor_cols].dropna()
    
    if len(factor_data) > 10:
        correlation_matrix = factor_data.corr()
        im = ax6.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # 添加文本标签
        for i in range(len(factor_cols)):
            for j in range(len(factor_cols)):
                text = ax6.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax6.set_xticks(range(len(factor_cols)))
        ax6.set_yticks(range(len(factor_cols)))
        ax6.set_xticklabels([col.replace('_', '\n') for col in factor_cols], rotation=45)
        ax6.set_yticklabels([col.replace('_', '\n') for col in factor_cols])
        ax6.set_title('因子相关性热力图', fontweight='bold')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax6, shrink=0.8)
    
    plt.tight_layout()
    chart_path = "examples/factor_analysis_gallery.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_success(f"因子分析图表已保存: {chart_path}")
    return tech_data

def create_strategy_performance_charts():
    """创建策略表现图表"""
    print_header("策略表现图表演示")
    
    # 创建模拟策略数据
    print_info("生成模拟策略表现数据...")
    
    # 生成基准和策略收益数据
    np.random.seed(42)
    days = 252  # 一年交易日
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    dates = dates[dates.weekday < 5][:days]  # 只保留工作日
    
    # 基准收益（市场指数）
    benchmark_returns = np.random.normal(0.0008, 0.015, len(dates))  # 年化8%，波动率15%
    
    # 策略收益（稍微优于基准）
    strategy_returns = np.random.normal(0.001, 0.018, len(dates))  # 年化10%，波动率18%
    
    # 计算累计收益
    benchmark_cumret = (1 + pd.Series(benchmark_returns, index=dates)).cumprod()
    strategy_cumret = (1 + pd.Series(strategy_returns, index=dates)).cumprod()
    
    # 计算回撤
    def calculate_drawdown(cumret):
        running_max = cumret.expanding().max()
        drawdown = (cumret - running_max) / running_max
        return drawdown
    
    benchmark_dd = calculate_drawdown(benchmark_cumret)
    strategy_dd = calculate_drawdown(strategy_cumret)
    
    # 创建策略表现图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('策略表现分析图表集合 (模拟数据)', fontsize=16, fontweight='bold')
    
    # 1. 累计收益对比
    ax1 = axes[0, 0]
    ax1.plot(dates, benchmark_cumret, label='基准指数', linewidth=2, color='#2E86AB')
    ax1.plot(dates, strategy_cumret, label='量化策略', linewidth=2, color='#A23B72')
    ax1.set_title('累计收益对比', fontweight='bold')
    ax1.set_ylabel('累计收益')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤对比
    ax2 = axes[0, 1]
    ax2.fill_between(dates, benchmark_dd, 0, alpha=0.3, color='#2E86AB', label='基准回撤')
    ax2.fill_between(dates, strategy_dd, 0, alpha=0.3, color='#A23B72', label='策略回撤')
    ax2.plot(dates, benchmark_dd, linewidth=1, color='#2E86AB')
    ax2.plot(dates, strategy_dd, linewidth=1, color='#A23B72')
    ax2.set_title('回撤对比', fontweight='bold')
    ax2.set_ylabel('回撤')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 收益分布
    ax3 = axes[0, 2]
    ax3.hist(benchmark_returns, bins=30, alpha=0.7, color='#2E86AB', label='基准', density=True)
    ax3.hist(strategy_returns, bins=30, alpha=0.7, color='#A23B72', label='策略', density=True)
    ax3.set_title('日收益分布', fontweight='bold')
    ax3.set_xlabel('日收益率')
    ax3.set_ylabel('密度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 滚动夏普比率
    ax4 = axes[1, 0]
    window = 60  # 60日滚动窗口
    
    benchmark_rolling_sharpe = pd.Series(benchmark_returns, index=dates).rolling(window).mean() / \
                              pd.Series(benchmark_returns, index=dates).rolling(window).std() * np.sqrt(252)
    strategy_rolling_sharpe = pd.Series(strategy_returns, index=dates).rolling(window).mean() / \
                             pd.Series(strategy_returns, index=dates).rolling(window).std() * np.sqrt(252)
    
    ax4.plot(dates, benchmark_rolling_sharpe, label='基准夏普', linewidth=2, color='#2E86AB')
    ax4.plot(dates, strategy_rolling_sharpe, label='策略夏普', linewidth=2, color='#A23B72')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title(f'{window}日滚动夏普比率', fontweight='bold')
    ax4.set_ylabel('夏普比率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 月度收益热力图
    ax5 = axes[1, 1]
    
    # 计算月度收益
    monthly_returns = pd.Series(strategy_returns, index=dates).resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns.index = monthly_returns.index.to_period('M')
    
    # 创建月度收益矩阵
    monthly_data = []
    for year in monthly_returns.index.year.unique():
        year_data = []
        for month in range(1, 13):
            try:
                period = pd.Period(f'{year}-{month:02d}', freq='M')
                if period in monthly_returns.index:
                    year_data.append(monthly_returns[period] * 100)
                else:
                    year_data.append(np.nan)
            except:
                year_data.append(np.nan)
        monthly_data.append(year_data)
    
    if monthly_data:
        monthly_matrix = np.array(monthly_data)
        im = ax5.imshow(monthly_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
        
        # 设置标签
        ax5.set_xticks(range(12))
        ax5.set_xticklabels(['1月', '2月', '3月', '4月', '5月', '6月',
                            '7月', '8月', '9月', '10月', '11月', '12月'])
        ax5.set_yticks(range(len(monthly_returns.index.year.unique())))
        ax5.set_yticklabels(monthly_returns.index.year.unique())
        ax5.set_title('月度收益热力图 (%)', fontweight='bold')
        
        # 添加数值标签
        for i in range(len(monthly_data)):
            for j in range(12):
                if not np.isnan(monthly_matrix[i, j]):
                    text = ax5.text(j, i, f'{monthly_matrix[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax5, shrink=0.8)
    
    # 6. 策略指标汇总
    ax6 = axes[1, 2]
    ax6.axis('off')  # 隐藏坐标轴
    
    # 计算策略指标
    total_return_benchmark = (benchmark_cumret.iloc[-1] - 1) * 100
    total_return_strategy = (strategy_cumret.iloc[-1] - 1) * 100
    
    annual_return_benchmark = np.mean(benchmark_returns) * 252 * 100
    annual_return_strategy = np.mean(strategy_returns) * 252 * 100
    
    annual_vol_benchmark = np.std(benchmark_returns) * np.sqrt(252) * 100
    annual_vol_strategy = np.std(strategy_returns) * np.sqrt(252) * 100
    
    sharpe_benchmark = annual_return_benchmark / annual_vol_benchmark
    sharpe_strategy = annual_return_strategy / annual_vol_strategy
    
    max_dd_benchmark = benchmark_dd.min() * 100
    max_dd_strategy = strategy_dd.min() * 100
    
    # 创建指标表格
    metrics_text = f"""
    策略表现指标对比
    
    指标                基准        策略
    ────────────────────────────────
    总收益率          {total_return_benchmark:6.1f}%    {total_return_strategy:6.1f}%
    年化收益率        {annual_return_benchmark:6.1f}%    {annual_return_strategy:6.1f}%
    年化波动率        {annual_vol_benchmark:6.1f}%    {annual_vol_strategy:6.1f}%
    夏普比率          {sharpe_benchmark:6.2f}     {sharpe_strategy:6.2f}
    最大回撤          {max_dd_benchmark:6.1f}%    {max_dd_strategy:6.1f}%
    """
    
    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    chart_path = "examples/strategy_performance_gallery.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_success(f"策略表现图表已保存: {chart_path}")
    
    return {
        'benchmark_returns': benchmark_returns,
        'strategy_returns': strategy_returns,
        'benchmark_cumret': benchmark_cumret,
        'strategy_cumret': strategy_cumret
    }

def create_market_analysis_charts():
    """创建市场分析图表"""
    print_header("市场分析图表演示")
    
    # 创建模拟市场数据
    print_info("生成模拟市场分析数据...")
    
    np.random.seed(42)
    
    # 生成多只股票的数据
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    dates = dates[dates.weekday < 5][:200]  # 只保留工作日
    
    # 生成相关的股票收益数据
    correlation_matrix = np.array([
        [1.0, 0.6, 0.7, 0.3, 0.5],
        [0.6, 1.0, 0.8, 0.4, 0.6],
        [0.7, 0.8, 1.0, 0.2, 0.7],
        [0.3, 0.4, 0.2, 1.0, 0.3],
        [0.5, 0.6, 0.7, 0.3, 1.0]
    ])
    
    # 使用Cholesky分解生成相关的随机数
    L = np.linalg.cholesky(correlation_matrix)
    random_data = np.random.normal(0, 0.02, (len(dates), len(symbols)))
    correlated_returns = random_data @ L.T
    
    # 创建股票价格数据
    stock_data = {}
    initial_prices = [150, 2800, 300, 200, 3200]
    
    for i, symbol in enumerate(symbols):
        prices = [initial_prices[i]]
        for ret in correlated_returns[:, i]:
            prices.append(prices[-1] * (1 + ret))
        
        stock_data[symbol] = {
            'prices': np.array(prices[1:]),  # 去掉初始价格
            'returns': correlated_returns[:, i]
        }
    
    # 创建市场分析图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('市场分析图表集合 (模拟数据)', fontsize=16, fontweight='bold')
    
    # 1. 股票价格走势对比
    ax1 = axes[0, 0]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    
    for i, symbol in enumerate(symbols):
        # 标准化价格以便比较
        normalized_prices = stock_data[symbol]['prices'] / stock_data[symbol]['prices'][0]
        ax1.plot(dates, normalized_prices, label=symbol, linewidth=2, color=colors[i])
    
    ax1.set_title('股票价格走势对比 (标准化)', fontweight='bold')
    ax1.set_ylabel('标准化价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 收益率相关性热力图
    ax2 = axes[0, 1]
    returns_df = pd.DataFrame({symbol: stock_data[symbol]['returns'] for symbol in symbols})
    actual_corr = returns_df.corr()
    
    im = ax2.imshow(actual_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # 添加文本标签
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            text = ax2.text(j, i, f'{actual_corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax2.set_xticks(range(len(symbols)))
    ax2.set_yticks(range(len(symbols)))
    ax2.set_xticklabels(symbols)
    ax2.set_yticklabels(symbols)
    ax2.set_title('股票收益率相关性矩阵', fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # 3. 波动率对比
    ax3 = axes[0, 2]
    volatilities = []
    for symbol in symbols:
        vol = np.std(stock_data[symbol]['returns']) * np.sqrt(252) * 100  # 年化波动率
        volatilities.append(vol)
    
    bars = ax3.bar(symbols, volatilities, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('年化波动率对比', fontweight='bold')
    ax3.set_ylabel('年化波动率 (%)')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, vol in zip(bars, volatilities):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{vol:.1f}%', ha='center', va='bottom')
    
    # 4. 滚动相关性分析
    ax4 = axes[1, 0]
    window = 30
    
    # 计算AAPL与其他股票的滚动相关性
    base_returns = pd.Series(stock_data['AAPL']['returns'], index=dates)
    
    for i, symbol in enumerate(symbols[1:], 1):  # 跳过AAPL自己
        other_returns = pd.Series(stock_data[symbol]['returns'], index=dates)
        rolling_corr = base_returns.rolling(window).corr(other_returns)
        ax4.plot(dates, rolling_corr, label=f'AAPL vs {symbol}', 
                linewidth=2, color=colors[i])
    
    ax4.set_title(f'AAPL与其他股票的{window}日滚动相关性', fontweight='bold')
    ax4.set_ylabel('相关系数')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 5. 收益率分布对比
    ax5 = axes[1, 1]
    
    for i, symbol in enumerate(symbols[:3]):  # 只显示前3只股票避免过于拥挤
        returns = stock_data[symbol]['returns'] * 100  # 转换为百分比
        ax5.hist(returns, bins=30, alpha=0.6, color=colors[i], 
                label=symbol, density=True)
    
    ax5.set_title('日收益率分布对比', fontweight='bold')
    ax5.set_xlabel('日收益率 (%)')
    ax5.set_ylabel('密度')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 风险收益散点图
    ax6 = axes[1, 2]
    
    annual_returns = []
    annual_vols = []
    
    for symbol in symbols:
        annual_ret = np.mean(stock_data[symbol]['returns']) * 252 * 100
        annual_vol = np.std(stock_data[symbol]['returns']) * np.sqrt(252) * 100
        annual_returns.append(annual_ret)
        annual_vols.append(annual_vol)
    
    scatter = ax6.scatter(annual_vols, annual_returns, c=colors, s=100, alpha=0.8, edgecolors='black')
    
    # 添加股票标签
    for i, symbol in enumerate(symbols):
        ax6.annotate(symbol, (annual_vols[i], annual_returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax6.set_title('风险收益散点图', fontweight='bold')
    ax6.set_xlabel('年化波动率 (%)')
    ax6.set_ylabel('年化收益率 (%)')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 添加有效前沿线（简化版）
    efficient_vols = np.linspace(min(annual_vols), max(annual_vols), 50)
    efficient_rets = []
    for vol in efficient_vols:
        # 简化的有效前沿计算（实际应该用优化算法）
        ret = np.interp(vol, sorted(annual_vols), sorted(annual_returns))
        efficient_rets.append(ret)
    
    ax6.plot(efficient_vols, efficient_rets, '--', color='gray', alpha=0.7, label='有效前沿')
    ax6.legend()
    
    plt.tight_layout()
    chart_path = "examples/market_analysis_gallery.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_success(f"市场分析图表已保存: {chart_path}")
    
    return {
        'symbols': symbols,
        'stock_data': stock_data,
        'correlation_matrix': actual_corr
    }

def main():
    """主函数：运行所有图表演示"""
    print_header("量化交易系统 - 图表画廊演示")
    
    print_info("这个演示将生成多种类型的量化分析图表")
    print_info("包括技术分析、因子分析、策略表现和市场分析图表")
    print_info("所有图表将保存在 examples/ 目录下")
    
    try:
        # 1. 技术分析图表
        tech_data = create_technical_analysis_charts()
        
        # 2. 因子分析图表
        factor_data = create_factor_analysis_charts(tech_data)
        
        # 3. 策略表现图表
        create_strategy_performance_charts()
        
        # 4. 市场分析图表
        market_data = create_market_analysis_charts()
        
        print_header("图表画廊演示完成")
        print_success("所有图表已成功生成！")
        
        print_info("\n📊 生成的图表文件:")
        print("   • technical_analysis_gallery.png - 技术分析图表集合")
        print("   • factor_analysis_gallery.png - 因子分析图表集合") 
        print("   • strategy_performance_gallery.png - 策略表现图表集合")
        print("   • market_analysis_gallery.png - 市场分析图表集合")
        
        print_info("\n🎯 图表用途:")
        print("   • 技术分析: 价格走势、技术指标分析")
        print("   • 因子分析: 因子效果评估、相关性分析")
        print("   • 策略表现: 收益回撤、风险指标分析")
        print("   • 市场分析: 多股票对比、市场相关性分析")
        
        print_info("\n📚 学习建议:")
        print("   1. 仔细观察每个图表的含义和用途")
        print("   2. 尝试修改参数，观察图表变化")
        print("   3. 结合实际交易策略理解图表信息")
        print("   4. 学习如何从图表中提取有价值的信息")
        
    except Exception as e:
        print(f"❌ 图表生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()