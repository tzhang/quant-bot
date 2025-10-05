#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存数据演示脚本
使用DataManager的缓存功能进行量化分析演示
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_manager import DataManager
from src.factors.technical import TechnicalFactors

def get_cached_data_with_manager(symbols, start_date=None, end_date=None):
    """
    使用DataManager获取缓存数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        dict: 股票数据字典
    """
    print("🔧 初始化数据管理器...")
    data_manager = DataManager()
    
    # 设置默认日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    print(f"📅 数据时间范围: {start_date} 到 {end_date}")
    
    # 获取多只股票数据
    print(f"📊 获取 {len(symbols)} 只股票数据...")
    stock_data = data_manager.get_multiple_stocks_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # 显示缓存信息
    cache_info = data_manager.get_cache_info()
    print(f"💾 缓存信息:")
    print(f"   缓存目录: {cache_info.get('cache_dir', '未知')}")
    print(f"   文件数量: {cache_info.get('file_count', 0)}")
    print(f"   缓存大小: {cache_info.get('size', '未知')}")
    
    return stock_data

def analyze_stock_data(data, symbol):
    """
    分析股票数据
    
    Args:
        data: 股票数据DataFrame
        symbol: 股票代码
    """
    print(f"\n📊 {symbol} 数据分析")
    print("=" * 50)
    
    # 确定收盘价列名
    close_col = 'close' if 'close' in data.columns else 'Close'
    
    # 基础统计
    close_price = data[close_col]
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
    使用TechnicalFactors计算技术指标
    
    Args:
        data: 股票数据DataFrame
        
    Returns:
        DataFrame: 包含技术指标的数据
    """
    print(f"\n📈 技术指标计算")
    print("=" * 50)
    
    # 初始化技术因子计算器
    tech_factors = TechnicalFactors()
    
    # 确保数据格式正确（大写列名）
    df = data.copy()
    if 'close' in df.columns:
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
    
    try:
        # 使用TechnicalFactors计算所有技术指标
        factors_df = tech_factors.calculate_all_factors(df)
        
        # 合并原始数据和技术指标
        result_df = pd.concat([df, factors_df], axis=1)
        
        print("✅ 技术指标计算完成:")
        print("   - 简单移动平均线 (SMA)")
        print("   - 指数移动平均线 (EMA)")
        print("   - RSI相对强弱指数")
        print("   - MACD指标")
        print("   - 布林带")
        
        # 显示可用的技术指标
        tech_columns = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"   📊 技术指标数量: {len(tech_columns)}")
        
        return result_df
        
    except Exception as e:
        print(f"⚠️  技术指标计算失败: {e}")
        print("ℹ️  使用简化的技术指标计算...")
        
        # 简化的技术指标计算作为备选方案
        close = df['Close']
        
        # 移动平均线
        df['SMA20'] = close.rolling(window=20).mean()
        df['EMA20'] = close.ewm(span=20).mean()
        
        # 简单RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI14'] = 100 - (100 / (1 + rs))
        
        print("✅ 简化技术指标计算完成")
        return df

def create_analysis_chart(data, symbol):
    """
    创建分析图表
    
    Args:
        data: 包含技术指标的数据
        symbol: 股票代码
    """
    print(f"\n📊 生成 {symbol} 分析图表...")
    
    # 确定列名
    close_col = 'close' if 'close' in data.columns else 'Close'
    volume_col = 'volume' if 'volume' in data.columns else 'Volume'
    
    # 检查数据结构
    print(f"数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    print(f"Volume列形状: {data[volume_col].shape if volume_col in data.columns else 'N/A'}")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'{symbol} 股票技术分析图表', fontsize=16, fontweight='bold')
    
    # 子图1: 价格和移动平均线
    ax1 = axes[0]
    ax1.plot(data.index, data[close_col], label='收盘价', linewidth=2)
    if 'SMA_5' in data.columns:
        ax1.plot(data.index, data['SMA_5'], label='5日均线', alpha=0.7)
    if 'SMA_20' in data.columns:
        ax1.plot(data.index, data['SMA_20'], label='20日均线', alpha=0.7)
    if 'SMA_50' in data.columns:
        ax1.plot(data.index, data['SMA_50'], label='50日均线', alpha=0.7)
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1, label='布林带')
    ax1.set_title('价格走势与移动平均线')
    ax1.set_ylabel('价格 ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: MACD
    ax2 = axes[1]
    if 'MACD' in data.columns:
        ax2.plot(data.index, data['MACD'], label='MACD', linewidth=2)
    if 'MACD_Signal' in data.columns:
        ax2.plot(data.index, data['MACD_Signal'], label='信号线', alpha=0.7)
    if 'MACD_Histogram' in data.columns:
        ax2.bar(data.index, data['MACD_Histogram'], label='MACD柱状图', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('MACD指标')
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: RSI
    ax3 = axes[2]
    if 'RSI' in data.columns:
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
    # 确保成交量数据是一维的
    volume_data = data[volume_col]
    if hasattr(volume_data, 'iloc') and len(volume_data.shape) > 1:
        # 如果是多维数据，取第一列
        volume_data = volume_data.iloc[:, 0] if volume_data.shape[1] > 0 else volume_data.iloc[:, -1]
    
    ax4.bar(data.index, volume_data, alpha=0.6, label='成交量')
    if 'Volume_SMA' in data.columns:
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
    
    # 可用的股票列表（用户持仓股票）
    available_stocks = ['HUBS', 'MDB', 'NIO', 'OKTA', 'TSLA']
    
    print("📋 分析股票列表（用户持仓）:")
    for i, stock in enumerate(available_stocks, 1):
        print(f"   {i}. {stock}")
    
    # 使用DataManager获取数据
    print(f"\n{'='*60}")
    print("📊 数据获取")
    print(f"{'='*60}")
    
    stock_data = get_cached_data_with_manager(available_stocks)
    
    if not stock_data:
        print("❌ 未能获取任何股票数据")
        return
    
    # 分析结果存储
    analysis_results = {}
    
    # 分析每只股票
    for symbol in available_stocks:
        if symbol not in stock_data:
            print(f"⚠️  跳过 {symbol}：数据不可用")
            continue
            
        print(f"\n{'='*60}")
        print(f"🔍 分析 {symbol}")
        print(f"{'='*60}")
        
        data = stock_data[symbol]
        
        # 显示数据基本信息
        print(f"✅ 成功加载 {symbol} 数据")
        print(f"   数据形状: {data.shape}")
        print(f"   时间范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   数据列: {list(data.columns)}")
        
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
    print("   ✅ 使用DataManager获取和缓存数据")
    print("   ✅ 使用TechnicalFactors进行技术分析")
    print("   ✅ 计算关键风险指标")
    print("   ✅ 生成专业的分析图表")
    print("   ✅ 进行多股票对比分析")

if __name__ == "__main__":
    main()