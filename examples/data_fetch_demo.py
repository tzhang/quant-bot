#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取演示脚本
演示如何正确使用DataManager获取股票数据，支持IB TWS API、Qlib和OpenBB数据源
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_manager import DataManager
import pandas as pd
import numpy as np

def demo_single_stock_data():
    """
    演示单只股票数据获取
    """
    print("=" * 60)
    print("📊 单只股票数据获取演示")
    print("=" * 60)
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 显示数据源信息
    cache_info = data_manager.get_cache_info()
    data_source = cache_info.get('data_source', {})
    print(f"🔧 数据源信息:")
    print(f"   主要数据源: {data_source.get('primary_source', 'unknown')}")
    print(f"   Qlib可用: {data_source.get('qlib_available', False)}")
    if 'total_instruments' in data_source:
        print(f"   可用股票数: {data_source['total_instruments']}")
    
    # 获取苹果公司股票数据
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-06-30'
    print(f"\n🎯 获取 {symbol} 股票数据 ({start_date} 到 {end_date})...")
    
    try:
        # 使用新的数据获取方法
        data = data_manager.get_stock_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            print(f"✅ 成功获取 {symbol} 数据")
            print(f"   数据形状: {data.shape}")
            print(f"   时间范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   包含字段: {list(data.columns)}")
            
            # 显示基本统计信息
            close_price = data['close'] if 'close' in data.columns else data['Close']
            
            print(f"   最新收盘价: ${close_price.iloc[-1]:.2f}")
            print(f"   期间最高价: ${close_price.max():.2f}")
            print(f"   期间最低价: ${close_price.min():.2f}")
            print(f"   期间涨跌幅: {((close_price.iloc[-1] / close_price.iloc[0]) - 1) * 100:+.2f}%")
            
            return data
        else:
            print("❌ 未获取到有效数据")
            return None
            
    except Exception as e:
        print(f"❌ 数据获取失败: {str(e)}")
        return None

def demo_cached_data():
    """
    演示缓存数据的使用
    """
    print("\n" + "=" * 60)
    print("💾 缓存数据使用演示")
    print("=" * 60)
    
    data_manager = DataManager()
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-03-31'
    
    print("🔍 检查缓存状态...")
    
    # 第一次获取（可能从缓存或网络）
    start_time = time.time()
    data = data_manager.get_stock_data(symbol, start_date, end_date)
    fetch_time = time.time() - start_time
    
    if data is not None:
        print(f"✅ 数据获取完成，耗时: {fetch_time:.2f}秒")
        print(f"   数据来源: {'缓存' if fetch_time < 1.0 else '网络'}")
        
        # 立即再次获取相同数据（应该从缓存）
        start_time = time.time()
        cached_data = data_manager.get_stock_data(symbol, start_date, end_date)
        cache_time = time.time() - start_time
        
        print(f"✅ 缓存数据获取完成，耗时: {cache_time:.2f}秒")
        
        if fetch_time > 0 and cache_time > 0:
            speedup = fetch_time / cache_time
            print(f"🚀 缓存性能提升: {speedup:.1f}x")
        
        return cached_data
    else:
        print("❌ 数据获取失败")
        return None

def demo_multiple_stocks():
    """
    演示多只股票数据获取
    """
    print("\n" + "=" * 60)
    print("📊 多只股票数据获取演示")
    print("=" * 60)
    
    data_manager = DataManager()
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2020-03-31'
    
    print(f"🎯 获取多只股票数据: {', '.join(symbols)}")
    print(f"   时间范围: {start_date} 到 {end_date}")
    
    try:
        start_time = time.time()
        data = data_manager.get_multiple_stocks_data(symbols, start_date, end_date)
        fetch_time = time.time() - start_time
        
        if data is not None and not data.empty:
            print(f"✅ 成功获取多只股票数据，耗时: {fetch_time:.2f}秒")
            print(f"   数据形状: {data.shape}")
            
            # 显示每只股票的数据统计
            for symbol in symbols:
                if symbol in data.columns.get_level_values(0):
                    symbol_data = data[symbol]
                    close_price = symbol_data['close'] if 'close' in symbol_data.columns else symbol_data['Close']
                    print(f"   {symbol}: {len(close_price)} 条记录, 最新价格: ${close_price.iloc[-1]:.2f}")
            
            return data
        else:
            print("❌ 未获取到有效数据")
            return None
            
    except Exception as e:
        print(f"❌ 多只股票数据获取失败: {str(e)}")
        return None

def demo_data_analysis():
    """
    演示基础数据分析
    """
    print("\n" + "=" * 60)
    print("📈 基础数据分析演示")
    print("=" * 60)
    
    data_manager = DataManager()
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-06-30'
    
    # 获取数据
    data = data_manager.get_stock_data(symbol, start_date, end_date)
    
    if data is None or data.empty:
        print("❌ 无法获取数据进行分析")
        return None
    
    # 提取收盘价数据
    close_price = data['close'] if 'close' in data.columns else data['Close']
    volume = data['volume'] if 'volume' in data.columns else data['Volume']
    
    print(f"📊 {symbol} 数据分析结果:")
    print(f"   数据点数量: {len(close_price)}")
    print(f"   平均收盘价: ${close_price.mean():.2f}")
    print(f"   价格标准差: ${close_price.std():.2f}")
    print(f"   平均成交量: {volume.mean():,.0f}")
    
    # 计算简单收益率
    returns = close_price.pct_change().dropna()
    print(f"   平均日收益率: {returns.mean():.4f} ({returns.mean() * 100:.2f}%)")
    print(f"   收益率波动率: {returns.std():.4f} ({returns.std() * 100:.2f}%)")
    print(f"   年化收益率: {returns.mean() * 252:.2%}")
    print(f"   年化波动率: {returns.std() * np.sqrt(252):.2%}")
    
    # 计算简单移动平均
    sma_20 = close_price.rolling(window=20).mean()
    current_price = close_price.iloc[-1]
    current_sma = sma_20.iloc[-1]
    
    print(f"   当前价格: ${current_price:.2f}")
    print(f"   20日均线: ${current_sma:.2f}")
    print(f"   价格相对均线: {((current_price / current_sma) - 1) * 100:+.2f}%")
    
    return data

def main():
    """
    主函数：运行所有演示
    """
    print("🚀 量化数据获取演示系统")
    print("=" * 60)
    
    # 演示1：单只股票数据获取
    data1 = demo_single_stock_data()
    
    # 等待一段时间，避免API限制
    print("\n⏳ 等待2秒，避免API限制...")
    time.sleep(2)
    
    # 演示2：缓存数据使用
    data2 = demo_cached_data()
    
    # 等待一段时间
    print("\n⏳ 等待2秒...")
    time.sleep(2)
    
    # 演示3：多只股票数据获取
    data3 = demo_multiple_stocks()
    
    # 等待一段时间
    print("\n⏳ 等待2秒...")
    time.sleep(2)
    
    # 演示4：基础数据分析
    data4 = demo_data_analysis()
    
    # 显示缓存信息
    print("\n" + "=" * 60)
    print("💾 缓存信息总结")
    print("=" * 60)
    
    data_manager = DataManager()
    cache_info = data_manager.get_cache_info()
    
    print(f"📁 缓存目录: {cache_info.get('cache_directory', 'unknown')}")
    print(f"📄 缓存文件数: {cache_info.get('cache_files', 0)}")
    print(f"💽 缓存大小: {cache_info.get('cache_size_mb', 0):.2f} MB")
    
    data_source = cache_info.get('data_source', {})
    print(f"🔧 主要数据源: {data_source.get('primary_source', 'unknown')}")
    print(f"✅ Qlib可用: {data_source.get('qlib_available', False)}")
    print(f"✅ OpenBB可用: {data_source.get('openbb_available', False)}")
    print(f"✅ IB TWS API可用: {data_source.get('ib_available', False)}")
    
    print("\n🎉 演示完成！")
    print("💡 提示：重复运行脚本可以体验缓存加速效果")

if __name__ == "__main__":
    main()