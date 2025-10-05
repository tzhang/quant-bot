#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资组合分析演示
分析Tesla、Okta、NIO等股票的投资组合
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

# 导入系统模块
from src.data.data_manager import DataManager
from src.factors.engine import FactorEngine
from src.factors.multi_factor_model import MultiFactorModel, FactorConfig, ModelConfig
from src.performance.analyzer import PerformanceAnalyzer
from src.utils.indicators import TechnicalIndicators

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主演示函数"""
    print("🚀 量化交易系统 - 投资组合分析演示")
    print("=" * 60)
    
    # 1. 初始化数据管理器
    print("\n📊 步骤1: 初始化数据管理器")
    data_manager = DataManager()
    
    # 您的投资组合股票
    portfolio_symbols = ['TSLA', 'OKTA', 'NIO', 'AAPL', 'GOOGL']  # 添加一些对比股票
    print(f"分析股票: {', '.join(portfolio_symbols)}")
    
    # 2. 获取股票数据
    print("\n📈 步骤2: 获取股票数据")
    try:
        # 获取过去1年的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        stock_data = {}
        for symbol in portfolio_symbols:
            print(f"  正在获取 {symbol} 数据...")
            data = data_manager.get_data(
                symbols=[symbol],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            if not data.empty:
                stock_data[symbol] = data
                print(f"  ✅ {symbol}: {len(data)} 条数据")
            else:
                print(f"  ❌ {symbol}: 数据获取失败")
        
        if not stock_data:
            print("❌ 无法获取任何股票数据，请检查网络连接")
            return
            
    except Exception as e:
        print(f"❌ 数据获取错误: {e}")
        return
    
    # 3. 计算技术因子
    print("\n🧮 步骤3: 计算技术因子")
    factor_engine = FactorEngine()
    
    all_factors = {}
    for symbol, data in stock_data.items():
        print(f"  计算 {symbol} 的技术因子...")
        try:
            factors = factor_engine.compute_technical(data)
            all_factors[symbol] = factors
            print(f"  ✅ {symbol}: {len(factors.columns)} 个因子")
        except Exception as e:
            print(f"  ❌ {symbol} 因子计算失败: {e}")
    
    # 4. 投资组合分析
    print("\n📊 步骤4: 投资组合分析")
    analyze_portfolio(stock_data, all_factors)
    
    # 5. 风险分析
    print("\n⚠️ 步骤5: 风险分析")
    risk_analysis(stock_data)
    
    # 6. 技术指标分析
    print("\n📈 步骤6: 技术指标分析")
    technical_analysis(stock_data)
    
    # 7. 相关性分析
    print("\n🔗 步骤7: 相关性分析")
    correlation_analysis(stock_data)
    
    print("\n🎉 演示完成！请查看生成的图表文件。")

def analyze_portfolio(stock_data, all_factors):
    """投资组合分析"""
    try:
        # 计算收益率
        returns_data = {}
        for symbol, data in stock_data.items():
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        # 创建收益率DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            print("  ❌ 无法计算收益率数据")
            return
        
        # 计算基本统计信息
        print("  📊 投资组合基本统计:")
        stats = returns_df.describe()
        print(stats.round(4))
        
        # 计算年化收益和波动率
        annual_returns = returns_df.mean() * 252
        annual_volatility = returns_df.std() * np.sqrt(252)
        sharpe_ratio = annual_returns / annual_volatility
        
        print("\n  📈 年化指标:")
        performance_df = pd.DataFrame({
            '年化收益率': annual_returns,
            '年化波动率': annual_volatility,
            '夏普比率': sharpe_ratio
        })
        print(performance_df.round(4))
        
        # 绘制累计收益图
        plt.figure(figsize=(12, 8))
        cumulative_returns = (1 + returns_df).cumprod()
        
        for symbol in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, cumulative_returns[symbol], 
                    label=symbol, linewidth=2)
        
        plt.title('投资组合累计收益对比', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累计收益 (基准=1)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('portfolio_cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  ✅ 累计收益图已保存: portfolio_cumulative_returns.png")
        
    except Exception as e:
        print(f"  ❌ 投资组合分析失败: {e}")

def risk_analysis(stock_data):
    """风险分析"""
    try:
        # 计算收益率
        returns_data = {}
        for symbol, data in stock_data.items():
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            print("  ❌ 无法进行风险分析")
            return
        
        # 计算VaR (Value at Risk)
        confidence_level = 0.05  # 95% 置信度
        var_95 = returns_df.quantile(confidence_level)
        
        print("  📊 风险指标 (95% VaR):")
        for symbol, var_value in var_95.items():
            print(f"    {symbol}: {var_value:.4f} ({var_value*100:.2f}%)")
        
        # 计算最大回撤
        print("\n  📉 最大回撤分析:")
        for symbol, data in stock_data.items():
            prices = data['Close']
            cumulative = prices / prices.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            print(f"    {symbol}: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        
        # 绘制风险收益散点图
        plt.figure(figsize=(10, 8))
        annual_returns = returns_df.mean() * 252
        annual_volatility = returns_df.std() * np.sqrt(252)
        
        plt.scatter(annual_volatility, annual_returns, s=100, alpha=0.7)
        
        for i, symbol in enumerate(returns_df.columns):
            plt.annotate(symbol, 
                        (annual_volatility[symbol], annual_returns[symbol]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.title('风险收益分析', fontsize=16, fontweight='bold')
        plt.xlabel('年化波动率', fontsize=12)
        plt.ylabel('年化收益率', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('risk_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  ✅ 风险收益图已保存: risk_return_analysis.png")
        
    except Exception as e:
        print(f"  ❌ 风险分析失败: {e}")

def technical_analysis(stock_data):
    """技术指标分析"""
    try:
        # 选择一只股票进行详细技术分析 (Tesla)
        if 'TSLA' not in stock_data:
            symbol = list(stock_data.keys())[0]
        else:
            symbol = 'TSLA'
        
        data = stock_data[symbol]
        print(f"  📈 {symbol} 技术指标分析:")
        
        # 计算技术指标
        indicators = TechnicalIndicators()
        
        # 移动平均线
        ma_20 = indicators.sma(data['Close'], 20)
        ma_50 = indicators.sma(data['Close'], 50)
        
        # RSI
        rsi = indicators.rsi(data['Close'], 14)
        
        # MACD
        macd_line, macd_signal, macd_histogram = indicators.macd(data['Close'])
        
        # 布林带
        bb_upper, bb_middle, bb_lower = indicators.bollinger_bands(data['Close'], 20)
        
        # 绘制技术分析图
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 价格和移动平均线
        axes[0].plot(data.index, data['Close'], label='收盘价', linewidth=2)
        axes[0].plot(data.index, ma_20, label='MA20', alpha=0.7)
        axes[0].plot(data.index, ma_50, label='MA50', alpha=0.7)
        axes[0].fill_between(data.index, bb_upper, bb_lower, alpha=0.2, label='布林带')
        axes[0].set_title(f'{symbol} 价格走势与技术指标', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        axes[1].plot(data.index, rsi, label='RSI', color='orange', linewidth=2)
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线(70)')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线(30)')
        axes[1].set_title('RSI 相对强弱指标', fontsize=12)
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # MACD
        axes[2].plot(data.index, macd_line, label='MACD', linewidth=2)
        axes[2].plot(data.index, macd_signal, label='信号线', linewidth=2)
        axes[2].bar(data.index, macd_histogram, label='MACD柱', alpha=0.6)
        axes[2].set_title('MACD 指标', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_technical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出当前技术指标值
        current_rsi = rsi.iloc[-1] if not rsi.empty else None
        current_macd = macd_line.iloc[-1] if not macd_line.empty else None
        
        print(f"    当前RSI: {current_rsi:.2f}" if current_rsi else "    RSI: 无数据")
        print(f"    当前MACD: {current_macd:.4f}" if current_macd else "    MACD: 无数据")
        
        # 简单的交易信号
        if current_rsi:
            if current_rsi > 70:
                print("    🔴 RSI信号: 超买，考虑卖出")
            elif current_rsi < 30:
                print("    🟢 RSI信号: 超卖，考虑买入")
            else:
                print("    🟡 RSI信号: 中性")
        
        print(f"  ✅ {symbol}技术分析图已保存: {symbol}_technical_analysis.png")
        
    except Exception as e:
        print(f"  ❌ 技术分析失败: {e}")

def correlation_analysis(stock_data):
    """相关性分析"""
    try:
        # 计算收益率相关性
        returns_data = {}
        for symbol, data in stock_data.items():
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            print("  ❌ 无法进行相关性分析")
            return
        
        # 计算相关性矩阵
        correlation_matrix = returns_df.corr()
        
        print("  🔗 股票收益率相关性矩阵:")
        print(correlation_matrix.round(3))
        
        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title('投资组合相关性分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('portfolio_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  ✅ 相关性热力图已保存: portfolio_correlation.png")
        
        # 分析高相关性股票
        print("\n  📊 相关性分析结果:")
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                stock1 = correlation_matrix.columns[i]
                stock2 = correlation_matrix.columns[j]
                
                if abs(corr_value) > 0.7:
                    high_corr_pairs.append((stock1, stock2, corr_value))
                    print(f"    {stock1} vs {stock2}: {corr_value:.3f} (高相关)")
        
        if not high_corr_pairs:
            print("    没有发现高相关性股票对 (|相关系数| > 0.7)")
        
    except Exception as e:
        print(f"  ❌ 相关性分析失败: {e}")

if __name__ == "__main__":
    main()