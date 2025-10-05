#!/usr/bin/env python3
"""
投资组合分析演示 - 使用缓存数据版本

演示如何使用量化交易系统分析投资组合
使用本地缓存的数据，避免API限制问题
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入系统模块
from src.data.data_manager import DataManager
from src.factors.technical import TechnicalFactors
from src.factors.engine import FactorEngine
from src.risk.risk_manager import RiskManager
from src.utils.indicators import TechnicalIndicators

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_sample_data(symbols, days=252):
    """
    创建示例股票数据
    
    Args:
        symbols: 股票代码列表
        days: 数据天数
        
    Returns:
        股票数据字典
    """
    print("📊 创建示例数据...")
    
    # 设置随机种子以获得可重复的结果
    np.random.seed(42)
    
    # 创建日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # 只保留工作日
    
    stock_data = {}
    
    # 为每个股票创建模拟数据
    base_prices = {
        'TSLA': 250.0,
        'OKTA': 85.0, 
        'NIO': 8.0,
        'AAPL': 175.0,
        'GOOGL': 140.0
    }
    
    volatilities = {
        'TSLA': 0.35,
        'OKTA': 0.30,
        'NIO': 0.45,
        'AAPL': 0.25,
        'GOOGL': 0.28
    }
    
    for symbol in symbols:
        base_price = base_prices.get(symbol, 100.0)
        volatility = volatilities.get(symbol, 0.30)
        
        # 生成价格序列（几何布朗运动）
        returns = np.random.normal(0.0005, volatility/np.sqrt(252), len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 创建OHLCV数据
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.01, len(df)))
        df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, 0.02, len(df))))
        df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, 0.02, len(df))))
        df['Volume'] = np.random.randint(1000000, 10000000, len(df))
        
        # 填充第一行的开盘价
        df.iloc[0, df.columns.get_loc('Open')] = base_price
        
        stock_data[symbol] = df.dropna()
        print(f"  ✅ {symbol}: {len(df)} 天数据")
    
    return stock_data

def analyze_portfolio():
    """分析投资组合"""
    
    print("🚀 量化交易系统 - 投资组合分析演示")
    print("=" * 50)
    
    # 用户投资组合
    portfolio_symbols = ['TSLA', 'OKTA', 'NIO', 'AAPL', 'GOOGL']
    portfolio_weights = [0.25, 0.20, 0.15, 0.25, 0.15]  # 投资权重
    
    print(f"📈 分析投资组合: {', '.join(portfolio_symbols)}")
    print(f"💰 投资权重: {dict(zip(portfolio_symbols, portfolio_weights))}")
    print()
    
    # 步骤1: 创建示例数据
    stock_data = create_sample_data(portfolio_symbols)
    
    # 步骤2: 计算技术因子
    print("📊 步骤2: 计算技术因子")
    technical_factors = TechnicalFactors()
    factor_results = {}
    
    for symbol in portfolio_symbols:
        data = stock_data[symbol]
        if len(data) > 50:  # 确保有足够的数据
            try:
                factors = technical_factors.calculate_all_factors(data)
                factor_results[symbol] = factors
                print(f"  ✅ {symbol}: 计算了 {len(factors.columns)} 个技术因子")
            except Exception as e:
                print(f"  ❌ {symbol}: 因子计算失败 - {e}")
    
    # 步骤3: 投资组合分析
    print("\n📈 步骤3: 投资组合分析")
    
    # 计算收益率
    returns_data = {}
    for symbol in portfolio_symbols:
        if symbol in stock_data:
            data = stock_data[symbol]
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        
        # 投资组合收益率
        portfolio_returns = (returns_df * portfolio_weights).sum(axis=1)
        
        # 计算关键指标
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min()
        
        print(f"  📊 年化收益率: {annual_return:.2%}")
        print(f"  📊 年化波动率: {annual_volatility:.2%}")
        print(f"  📊 夏普比率: {sharpe_ratio:.3f}")
        print(f"  📊 最大回撤: {max_drawdown:.2%}")
        
        # 相关性分析
        correlation_matrix = returns_df.corr()
        print(f"\n📊 股票相关性矩阵:")
        print(correlation_matrix.round(3))
    
    # 步骤4: 风险分析
    print("\n⚠️ 步骤4: 风险分析")
    risk_manager = RiskManager()
    
    if returns_data:
        # 计算VaR
        portfolio_returns_array = portfolio_returns.values
        
        # 创建投资组合权重Series
        portfolio_weights_series = pd.Series(portfolio_weights, index=portfolio_symbols)
        
        # 设置历史收益数据
        risk_manager.risk_models['historical'].set_returns_data(returns_df)
        
        # 计算VaR
        var_95 = risk_manager.risk_models['historical'].calculate_var(
            portfolio_weights_series, confidence_level=0.05
        )
        var_99 = risk_manager.risk_models['historical'].calculate_var(
            portfolio_weights_series, confidence_level=0.01
        )
        
        print(f"  ⚠️ VaR (95%): {var_95.value:.2%}")
        print(f"  ⚠️ VaR (99%): {var_99.value:.2%}")
    
    # 步骤5: 技术指标分析
    print("\n📈 步骤5: 技术指标分析")
    
    for symbol in portfolio_symbols[:2]:  # 只分析前两个股票以节省时间
        if symbol in stock_data:
            data = stock_data[symbol]
            close_prices = data['Close']
            
            # 计算技术指标
            sma_20 = TechnicalIndicators.sma(close_prices, 20)
            rsi = TechnicalIndicators.rsi(close_prices)
            
            current_price = close_prices.iloc[-1]
            current_sma = sma_20.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            print(f"  📊 {symbol}:")
            print(f"    当前价格: ${current_price:.2f}")
            print(f"    20日均线: ${current_sma:.2f}")
            print(f"    RSI: {current_rsi:.1f}")
            
            # 简单信号判断
            if current_price > current_sma:
                trend_signal = "看涨 📈"
            else:
                trend_signal = "看跌 📉"
                
            if current_rsi > 70:
                rsi_signal = "超买 ⚠️"
            elif current_rsi < 30:
                rsi_signal = "超卖 💡"
            else:
                rsi_signal = "中性 ➡️"
                
            print(f"    趋势信号: {trend_signal}")
            print(f"    RSI信号: {rsi_signal}")
    
    # 步骤6: 生成图表
    print("\n📊 步骤6: 生成分析图表")
    
    try:
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('投资组合分析报告', fontsize=16, fontweight='bold')
        
        # 1. 投资组合权重饼图
        axes[0, 0].pie(portfolio_weights, labels=portfolio_symbols, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('投资组合权重分布')
        
        # 2. 股票价格走势
        if stock_data:
            for symbol in portfolio_symbols:
                if symbol in stock_data:
                    data = stock_data[symbol]
                    normalized_prices = data['Close'] / data['Close'].iloc[0]
                    axes[0, 1].plot(data.index, normalized_prices, label=symbol, linewidth=2)
            axes[0, 1].set_title('股票价格走势 (标准化)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 相关性热力图
        if returns_data:
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[1, 0])
            axes[1, 0].set_title('股票相关性矩阵')
        
        # 4. 投资组合累计收益
        if returns_data:
            cumulative_returns = (1 + portfolio_returns).cumprod()
            axes[1, 1].plot(cumulative_returns.index, cumulative_returns.values, 
                           linewidth=2, color='green')
            axes[1, 1].set_title('投资组合累计收益')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_filename = 'portfolio_analysis_demo.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"  ✅ 图表已保存: {chart_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"  ❌ 图表生成失败: {e}")
    
    print("\n🎉 投资组合分析完成！")
    print("\n💡 系统功能展示:")
    print("  ✅ 数据管理 - 创建和处理股票数据")
    print("  ✅ 技术因子计算 - 计算多种技术指标")
    print("  ✅ 投资组合分析 - 收益、风险、相关性分析")
    print("  ✅ 风险管理 - VaR计算")
    print("  ✅ 技术指标 - RSI、移动平均线等")
    print("  ✅ 可视化分析 - 生成专业图表")
    
    print(f"\n📚 更多功能请参考:")
    print(f"  - 使用指南: USAGE_GUIDE.md")
    print(f"  - 示例代码: examples/")
    print(f"  - 高级技巧: docs/ADVANCED_TIPS_PRACTICES.md")

if __name__ == "__main__":
    analyze_portfolio()