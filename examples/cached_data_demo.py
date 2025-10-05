#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存数据演示脚本
直接使用已缓存的数据进行量化分析演示，避免网络请求限制
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

def load_cached_data(symbol='AAPL'):
    """
    加载缓存的股票数据
    
    Args:
        symbol: 股票代码
        
    Returns:
        DataFrame: 股票数据
    """
    cache_dir = Path(__file__).parent.parent / 'data_cache'
    
    # 查找对应的缓存文件
    cache_files = list(cache_dir.glob(f'ohlcv_{symbol}_*.csv'))
    
    if not cache_files:
        print(f"❌ 未找到 {symbol} 的缓存数据")
        return None
    
    # 使用最新的缓存文件
    cache_file = cache_files[0]
    print(f"📁 加载缓存文件: {cache_file.name}")
    
    try:
        # 读取CSV文件，跳过前两行（Price和Ticker行）
        df = pd.read_csv(cache_file, skiprows=2)
        
        # 重命名列，去掉第一列（Price列）
        df = df.iloc[:, 1:]  # 去掉第一列
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 设置Date列为索引
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        
        # 转换数据类型
        for col in df.columns:
            if col != 'Volume':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')
        
        print(f"✅ 成功加载 {symbol} 数据")
        print(f"   数据形状: {df.shape}")
        print(f"   时间范围: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
        
        return df
        
    except Exception as e:
        print(f"❌ 加载缓存数据失败: {str(e)}")
        return None

def analyze_stock_data(data, symbol):
    """
    分析股票数据
    
    Args:
        data: 股票数据DataFrame
        symbol: 股票代码
    """
    print(f"\n📊 {symbol} 数据分析")
    print("=" * 50)
    
    # 基础统计
    close_price = data['Close']
    print(f"数据点数量: {len(close_price)}")
    print(f"最新收盘价: ${close_price.iloc[-1]:.2f}")
    print(f"期间最高价: ${close_price.max():.2f}")
    print(f"期间最低价: ${close_price.min():.2f}")
    print(f"平均收盘价: ${close_price.mean():.2f}")
    
    # 收益率分析
    returns = close_price.pct_change().dropna()
    total_return = (close_price.iloc[-1] / close_price.iloc[0] - 1) * 100
    
    print(f"\n💰 收益率分析:")
    print(f"总收益率: {total_return:+.2f}%")
    print(f"平均日收益率: {returns.mean():.4f} ({returns.mean() * 100:.2f}%)")
    print(f"收益率标准差: {returns.std():.4f} ({returns.std() * 100:.2f}%)")
    print(f"年化收益率: {returns.mean() * 252:.2%}")
    print(f"年化波动率: {returns.std() * np.sqrt(252):.2%}")
    
    # 风险指标
    print(f"\n⚠️  风险指标:")
    print(f"最大单日涨幅: {returns.max():.2%}")
    print(f"最大单日跌幅: {returns.min():.2%}")
    
    # 计算最大回撤
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    print(f"最大回撤: {max_drawdown:.2%}")
    
    return {
        'returns': returns,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'volatility': returns.std() * np.sqrt(252)
    }

def calculate_technical_indicators(data):
    """
    计算技术指标
    
    Args:
        data: 股票数据DataFrame
        
    Returns:
        DataFrame: 包含技术指标的数据
    """
    print(f"\n📈 技术指标计算")
    print("=" * 50)
    
    df = data.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # 移动平均线
    df['SMA_5'] = close.rolling(window=5).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    
    # 指数移动平均线
    df['EMA_12'] = close.ewm(span=12).mean()
    df['EMA_26'] = close.ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # 布林带
    df['BB_Middle'] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 成交量指标
    df['Volume_SMA'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA']
    
    print("✅ 技术指标计算完成:")
    print("   - 移动平均线 (SMA 5, 20, 50)")
    print("   - 指数移动平均线 (EMA 12, 26)")
    print("   - MACD指标")
    print("   - 布林带")
    print("   - RSI相对强弱指数")
    print("   - 成交量指标")
    
    return df

def create_analysis_chart(data, symbol):
    """
    创建分析图表
    
    Args:
        data: 包含技术指标的数据
        symbol: 股票代码
    """
    print(f"\n📊 生成 {symbol} 分析图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'{symbol} 股票技术分析图表', fontsize=16, fontweight='bold')
    
    # 子图1: 价格和移动平均线
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='收盘价', linewidth=2)
    ax1.plot(data.index, data['SMA_5'], label='5日均线', alpha=0.7)
    ax1.plot(data.index, data['SMA_20'], label='20日均线', alpha=0.7)
    ax1.plot(data.index, data['SMA_50'], label='50日均线', alpha=0.7)
    ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1, label='布林带')
    ax1.set_title('价格走势与移动平均线')
    ax1.set_ylabel('价格 ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: MACD
    ax2 = axes[1]
    ax2.plot(data.index, data['MACD'], label='MACD', linewidth=2)
    ax2.plot(data.index, data['MACD_Signal'], label='信号线', alpha=0.7)
    ax2.bar(data.index, data['MACD_Histogram'], label='MACD柱状图', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('MACD指标')
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: RSI
    ax3 = axes[2]
    ax3.plot(data.index, data['RSI'], label='RSI', linewidth=2, color='purple')
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='超买线(70)')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖线(30)')
    ax3.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    ax3.set_title('RSI相对强弱指数')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 成交量
    ax4 = axes[3]
    ax4.bar(data.index, data['Volume'], alpha=0.6, label='成交量')
    ax4.plot(data.index, data['Volume_SMA'], color='red', label='20日均量', linewidth=2)
    ax4.set_title('成交量分析')
    ax4.set_ylabel('成交量')
    ax4.set_xlabel('日期')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = f'{symbol}_technical_analysis.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {chart_file}")
    
    plt.show()

def main():
    """
    主函数：运行缓存数据分析演示
    """
    print("🚀 缓存数据量化分析演示")
    print("=" * 60)
    
    # 可用的股票列表
    available_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
    
    print("📋 可分析的股票:")
    for i, stock in enumerate(available_stocks, 1):
        print(f"   {i}. {stock}")
    
    # 分析多只股票
    analysis_results = {}
    
    for symbol in available_stocks[:3]:  # 分析前3只股票
        print(f"\n{'='*60}")
        print(f"🔍 分析 {symbol}")
        print(f"{'='*60}")
        
        # 加载数据
        data = load_cached_data(symbol)
        if data is None:
            continue
        
        # 基础分析
        result = analyze_stock_data(data, symbol)
        analysis_results[symbol] = result
        
        # 计算技术指标
        data_with_indicators = calculate_technical_indicators(data)
        
        # 生成图表
        create_analysis_chart(data_with_indicators, symbol)
        
        print(f"✅ {symbol} 分析完成")
    
    # 对比分析
    if analysis_results:
        print(f"\n{'='*60}")
        print("📊 股票对比分析")
        print(f"{'='*60}")
        
        comparison_df = pd.DataFrame(analysis_results).T
        comparison_df = comparison_df.round(4)
        
        print("股票表现对比:")
        print(comparison_df[['total_return', 'max_drawdown', 'volatility']])
        
        # 找出最佳表现
        best_return = comparison_df['total_return'].idxmax()
        lowest_risk = comparison_df['volatility'].idxmin()
        
        print(f"\n🏆 表现最佳: {best_return} (总收益率: {comparison_df.loc[best_return, 'total_return']:.2f}%)")
        print(f"🛡️  风险最低: {lowest_risk} (波动率: {comparison_df.loc[lowest_risk, 'volatility']:.2%})")
    
    print(f"\n{'='*60}")
    print("🎉 缓存数据分析演示完成！")
    print(f"{'='*60}")
    print("📚 生成的文件:")
    print("   - 各股票技术分析图表 (PNG格式)")
    print("   - 详细的量化分析结果")
    print("\n💡 这个演示展示了如何:")
    print("   ✅ 使用缓存数据避免API限制")
    print("   ✅ 进行全面的技术分析")
    print("   ✅ 计算关键风险指标")
    print("   ✅ 生成专业的分析图表")
    print("   ✅ 进行多股票对比分析")

if __name__ == "__main__":
    main()