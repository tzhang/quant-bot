#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取和缓存使用教程

本教程演示如何使用量化交易系统的数据管理功能：
1. 获取股票数据（支持Qlib和yfinance数据源）
2. 理解缓存机制
3. 数据处理基础
4. 常见问题解决
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_manager import DataManager

def tutorial_1_basic_data_fetch():
    """
    教程1: 基础数据获取
    演示如何获取单只股票的历史数据
    """
    print("=" * 60)
    print("📊 教程1: 基础数据获取")
    print("=" * 60)
    
    # 初始化数据管理器
    print("🔧 初始化数据管理器...")
    data_manager = DataManager()
    
    # 显示数据源信息
    cache_info = data_manager.get_cache_info()
    data_source = cache_info.get('data_source', {})
    print(f"   主要数据源: {data_source.get('primary_source', 'unknown')}")
    print(f"   Qlib可用: {data_source.get('qlib_available', False)}")
    print(f"   yfinance可用: {data_source.get('yfinance_available', False)}")
    if 'total_instruments' in data_source:
        print(f"   可用股票数: {data_source['total_instruments']}")
    
    # 获取苹果公司股票数据
    print("\n📈 获取AAPL股票数据...")
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-03-31'
    
    try:
        data = data_manager.get_stock_data(symbol, start_date, end_date)
        
        print(f"✅ 成功获取 {symbol} 股票数据")
        print(f"   数据形状: {data.shape}")
        print(f"   时间范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   包含字段: {list(data.columns)}")
        
        # 显示前5行数据
        print("\n📋 数据预览 (前5行):")
        print(data.head())
        
        # 显示基本统计信息
        print("\n📊 基本统计信息:")
        print(data.describe())
        
        return data
        
    except Exception as e:
        print(f"❌ 数据获取失败: {str(e)}")
        return None

def tutorial_2_multiple_stocks():
    """
    教程2: 多只股票数据获取
    演示如何同时获取多只股票的数据
    """
    print("\n" + "=" * 60)
    print("📊 教程2: 多只股票数据获取")
    print("=" * 60)
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 定义股票池
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date = '2020-01-01'
    end_date = '2020-02-29'
    print(f"🎯 目标股票池: {symbols}")
    print(f"   时间范围: {start_date} 到 {end_date}")
    
    try:
        # 获取数据
        print("📥 正在获取数据...")
        data = data_manager.get_multiple_stocks_data(symbols, start_date, end_date)
        
        print(f"✅ 成功获取 {len(symbols)} 只股票数据")
        print(f"   总数据形状: {data.shape}")
        
        # 分析每只股票的数据
        print("\n📈 各股票数据概览:")
        for symbol in symbols:
            if symbol in data.columns.get_level_values(0):
                symbol_data = data[symbol]
                close_col = 'close' if 'close' in symbol_data.columns else 'Close'
                latest_price = symbol_data[close_col].iloc[-1]
                price_change = (symbol_data[close_col].iloc[-1] / symbol_data[close_col].iloc[0] - 1) * 100
                print(f"   {symbol}: {len(symbol_data)} 条记录, 最新价格 ${latest_price:.2f}, 期间涨跌 {price_change:+.2f}%")
        
        return data
        
    except Exception as e:
        print(f"❌ 多股票数据获取失败: {str(e)}")
        return None

def tutorial_3_cache_mechanism():
    """
    教程3: 缓存机制详解
    演示缓存如何工作以及如何管理缓存
    """
    print("\n" + "=" * 60)
    print("💾 教程3: 缓存机制详解")
    print("=" * 60)
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 获取缓存信息
    cache_info = data_manager.get_cache_info()
    print(f"📁 缓存目录: {cache_info.get('cache_directory', 'unknown')}")
    print(f"📄 缓存文件数: {cache_info.get('cache_files', 0)}")
    print(f"💽 缓存大小: {cache_info.get('cache_size_mb', 0):.2f} MB")
    
    # 演示缓存效果
    print("\n⏱️  缓存性能测试:")
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-01-31'
    
    # 第一次获取（可能需要下载）
    import time
    start_time = time.time()
    data1 = data_manager.get_stock_data(symbol, start_date, end_date)
    first_time = time.time() - start_time
    print(f"   首次获取耗时: {first_time:.2f} 秒")
    
    # 第二次获取（使用缓存）
    start_time = time.time()
    data2 = data_manager.get_stock_data(symbol, start_date, end_date)
    second_time = time.time() - start_time
    print(f"   缓存获取耗时: {second_time:.2f} 秒")
    
    if second_time < first_time and first_time > 0:
        speedup = first_time / second_time
        print(f"   🚀 缓存加速: {speedup:.1f}x 倍")
    
    # 显示缓存管理功能
    print("\n🔧 缓存管理功能:")
    print("   - 自动缓存所有获取的数据")
    print("   - 基于股票代码、时间范围生成缓存键")
    print("   - 支持缓存清理和信息查询")
    
    return data1

def tutorial_4_data_processing():
    """
    教程4: 数据处理基础
    演示常见的数据处理操作
    """
    print("\n" + "=" * 60)
    print("🔧 教程4: 数据处理基础")
    print("=" * 60)
    
    # 获取数据
    data_manager = DataManager()
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-03-31'
    
    data = data_manager.get_stock_data(symbol, start_date, end_date)
    
    if data is None or data.empty:
        print("❌ 无法获取数据，跳过数据处理教程")
        return None
    
    print("📊 原始数据处理:")
    print(f"   数据形状: {data.shape}")
    print(f"   时间范围: {data.index[0]} 到 {data.index[-1]}")
    print(f"   包含字段: {list(data.columns)}")
    
    # 获取收盘价数据
    close_data = data['close'] if 'close' in data.columns else data['Close']
    
    # 计算收益率
    print("\n💰 计算收益率:")
    returns = close_data.pct_change()
    log_returns = np.log(close_data / close_data.shift(1))
    
    print(f"   日收益率均值: {returns.mean():.4f}")
    print(f"   日收益率标准差: {returns.std():.4f}")
    print(f"   年化收益率: {returns.mean() * 252:.2%}")
    print(f"   年化波动率: {returns.std() * np.sqrt(252):.2%}")
    
    # 计算技术指标
    print("\n📈 计算技术指标:")
    
    # 移动平均线
    sma_5 = close_data.rolling(window=5).mean()
    sma_20 = close_data.rolling(window=20).mean()
    
    # 布林带
    bb_middle = close_data.rolling(window=20).mean()
    bb_std = close_data.rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # RSI
    delta = close_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    print("   ✅ 5日移动平均线")
    print("   ✅ 20日移动平均线")
    print("   ✅ 布林带 (上轨、中轨、下轨)")
    print("   ✅ RSI相对强弱指数")
    
    # 创建处理后的数据框
    processed_data = pd.DataFrame({
        'close': close_data,
        'returns': returns,
        'sma_5': sma_5,
        'sma_20': sma_20,
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'rsi': rsi
    })
    
    # 显示最新指标值
    latest = processed_data.iloc[-1]
    print(f"\n📊 最新指标值 ({latest.name.strftime('%Y-%m-%d')}):")
    print(f"   收盘价: ${latest['close']:.2f}")
    print(f"   5日均线: ${latest['sma_5']:.2f}")
    print(f"   20日均线: ${latest['sma_20']:.2f}")
    print(f"   RSI: {latest['rsi']:.1f}")
    
    return processed_data

def tutorial_5_data_visualization():
    """
    教程5: 数据可视化
    演示如何创建常见的金融图表
    """
    print("\n" + "=" * 60)
    print("📊 教程5: 数据可视化")
    print("=" * 60)
    
    # 获取处理后的数据
    data = tutorial_4_data_processing()
    if data is None:
        return
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AAPL 股票数据分析图表', fontsize=16, fontweight='bold')
    
    # 1. 价格和移动平均线
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['close'], label='收盘价', linewidth=1)
    ax1.plot(data.index, data['sma_5'], label='5日均线', alpha=0.7)
    ax1.plot(data.index, data['sma_20'], label='20日均线', alpha=0.7)
    ax1.set_title('价格走势与移动平均线')
    ax1.set_ylabel('价格 ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 布林带
    ax2 = axes[0, 1]
    ax2.plot(data.index, data['close'], label='收盘价', color='black', linewidth=1)
    ax2.fill_between(data.index, data['bb_upper'], data['bb_lower'], 
                     alpha=0.2, color='blue', label='布林带')
    ax2.plot(data.index, data['bb_upper'], color='red', alpha=0.7, label='上轨')
    ax2.plot(data.index, data['bb_lower'], color='green', alpha=0.7, label='下轨')
    ax2.set_title('布林带指标')
    ax2.set_ylabel('价格 ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. RSI指标
    ax3 = axes[1, 0]
    ax3.plot(data.index, data['rsi'], color='purple', linewidth=1)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='超买线(70)')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖线(30)')
    ax3.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    ax3.set_title('RSI 相对强弱指数')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 收益率分布
    ax4 = axes[1, 1]
    returns_clean = data['returns'].dropna()
    ax4.hist(returns_clean, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(returns_clean.mean(), color='red', linestyle='--', 
                label=f'均值: {returns_clean.mean():.4f}')
    ax4.set_title('日收益率分布')
    ax4.set_xlabel('日收益率')
    ax4.set_ylabel('频次')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path('examples/data_tutorial_charts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    return data

def tutorial_6_troubleshooting():
    """
    教程6: 常见问题排查
    演示如何处理数据获取中的常见问题
    """
    print("\n" + "=" * 60)
    print("🔧 教程6: 常见问题排查")
    print("=" * 60)
    
    data_manager = DataManager()
    
    # 问题1: 无效股票代码
    print("❌ 问题1: 无效股票代码")
    try:
        invalid_data = data_manager.get_stock_data('INVALID_SYMBOL', '2020-01-01', '2020-01-31')
        if invalid_data is None or invalid_data.empty:
            print("   预期结果: 无效代码返回空数据")
        else:
            print("   意外：无效代码竟然返回了数据")
    except Exception as e:
        print(f"   预期错误: {str(e)}")
    print("   💡 解决方案: 使用有效的股票代码，如 AAPL, GOOGL, MSFT")
    
    # 问题2: 数据可用性检查
    print("\n🔍 问题2: 数据可用性检查")
    test_symbols = ['AAPL', 'INVALID_SYMBOL', 'GOOGL']
    for symbol in test_symbols:
        try:
            is_available = data_manager.check_data_availability(symbol)
            print(f"   {symbol}: {'✅ 可用' if is_available else '❌ 不可用'}")
        except Exception as e:
            print(f"   {symbol}: ❌ 检查失败 - {str(e)}")
    
    # 问题3: 网络连接问题模拟
    print("\n⚠️  问题3: 网络连接问题")
    print("   如果遇到网络错误，可以尝试:")
    print("   1. 检查网络连接")
    print("   2. 稍后重试")
    print("   3. 使用较短的时间周期")
    print("   4. 检查数据源状态")
    
    # 问题4: 数据缺失处理
    print("\n📊 问题4: 数据缺失处理")
    try:
        data = data_manager.get_stock_data('AAPL', '2020-01-01', '2020-01-31')
        if data is not None and not data.empty:
            # 检查缺失值
            missing_count = data.isnull().sum().sum()
            print(f"   数据缺失值总数: {missing_count}")
            
            if missing_count > 0:
                print("   💡 处理缺失值的方法:")
                print("   1. 前向填充: data.fillna(method='ffill')")
                print("   2. 后向填充: data.fillna(method='bfill')")
                print("   3. 删除缺失行: data.dropna()")
                print("   4. 插值填充: data.interpolate()")
            else:
                print("   ✅ 数据完整，无缺失值")
        else:
            print("   ❌ 无法获取测试数据")
    except Exception as e:
        print(f"   数据获取失败: {str(e)}")
    
    # 问题5: 内存使用优化
    print("\n💾 问题5: 内存使用优化")
    print("   对于大量数据，建议:")
    print("   1. 分批处理股票")
    print("   2. 使用较短的时间周期")
    print("   3. 及时删除不需要的变量")
    print("   4. 使用适当的数据类型")
    print("   5. 利用缓存机制避免重复下载")

def main():
    """
    主函数：运行所有教程
    """
    print("🎓 量化交易系统数据管理教程")
    print("本教程将带您了解数据获取、缓存、处理和可视化的完整流程")
    
    try:
        # 运行所有教程
        tutorial_1_basic_data_fetch()
        tutorial_2_multiple_stocks()
        tutorial_3_cache_mechanism()
        tutorial_5_data_visualization()  # 包含了tutorial_4
        tutorial_6_troubleshooting()
        
        print("\n" + "=" * 60)
        print("🎉 恭喜！您已完成数据管理教程")
        print("=" * 60)
        print("📚 您已学会:")
        print("   ✅ 获取单只和多只股票数据")
        print("   ✅ 理解和使用缓存机制")
        print("   ✅ 基础数据处理和技术指标计算")
        print("   ✅ 创建金融数据可视化图表")
        print("   ✅ 排查和解决常见问题")
        
        print("\n📖 下一步学习:")
        print("   1. 因子计算教程: python examples/factor_tutorial.py")
        print("   2. 完整因子评估: python examples/factor_evaluation.py")
        print("   3. 阅读进阶文档: docs/BEGINNER_GUIDE.md")
        
    except KeyboardInterrupt:
        print("\n⏹️  教程被用户中断")
    except Exception as e:
        print(f"\n❌ 教程执行出错: {str(e)}")
        print("请检查环境配置或联系技术支持")

if __name__ == "__main__":
    main()