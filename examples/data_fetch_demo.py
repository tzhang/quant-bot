#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取演示脚本
演示如何正确使用DataManager获取股票数据，避免频繁请求限制
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
    
    # 获取苹果公司股票数据
    symbol = 'AAPL'
    print(f"🎯 获取 {symbol} 股票数据...")
    
    try:
        # 使用较长的时间周期，减少API调用频率
        data = data_manager.get_data([symbol], period='6m')
        
        if data is not None and not data.empty:
            print(f"✅ 成功获取 {symbol} 数据")
            print(f"   数据形状: {data.shape}")
            print(f"   时间范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")
            
            # 显示基本统计信息
            if len(data.columns.levels) > 1:
                # 多级列索引
                close_price = data[(symbol, 'Close')]
            else:
                # 单级列索引
                close_price = data['Close']
            
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
    
    print("🔍 检查缓存状态...")
    
    # 第一次获取（可能从缓存或网络）
    start_time = time.time()
    data = data_manager.get_data([symbol], period='3m')
    fetch_time = time.time() - start_time
    
    if data is not None:
        print(f"✅ 数据获取完成，耗时: {fetch_time:.2f}秒")
        print(f"   数据来源: {'缓存' if fetch_time < 1.0 else '网络'}")
        
        # 立即再次获取相同数据（应该从缓存）
        start_time = time.time()
        cached_data = data_manager.get_data([symbol], period='3m')
        cache_time = time.time() - start_time
        
        print(f"✅ 缓存数据获取完成，耗时: {cache_time:.2f}秒")
        
        if fetch_time > 0 and cache_time > 0:
            speedup = fetch_time / cache_time
            print(f"🚀 缓存性能提升: {speedup:.1f}x")
        
        return cached_data
    else:
        print("❌ 数据获取失败")
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
    
    # 获取数据
    data = data_manager.get_data([symbol], period='3m')
    
    if data is None or data.empty:
        print("❌ 无法获取数据进行分析")
        return None
    
    # 提取收盘价数据
    if len(data.columns.levels) > 1:
        close_price = data[(symbol, 'Close')]
        volume = data[(symbol, 'Volume')]
    else:
        close_price = data['Close']
        volume = data['Volume']
    
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
    
    # 演示3：基础数据分析
    data3 = demo_data_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("=" * 60)
    
    # 总结
    success_count = sum([1 for data in [data1, data2, data3] if data is not None])
    print(f"✅ 成功完成 {success_count}/3 个演示")
    
    if success_count == 3:
        print("🎊 所有演示都成功运行！数据获取系统工作正常。")
    elif success_count > 0:
        print("⚠️  部分演示成功，可能存在网络限制或其他问题。")
    else:
        print("❌ 所有演示都失败，请检查网络连接和API限制。")
    
    print("\n📚 下一步建议:")
    print("   1. 运行因子计算演示: python examples/factor_tutorial.py")
    print("   2. 查看高级策略演示: python demo_advanced_factor_strategies.py")
    print("   3. 阅读使用指南: docs/USAGE_GUIDE.md")

if __name__ == "__main__":
    main()