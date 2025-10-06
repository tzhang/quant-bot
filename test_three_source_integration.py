#!/usr/bin/env python3
"""
测试三数据源（Qlib -> OpenBB -> yfinance）集成和回退机制
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_adapter import create_data_adapter
import pandas as pd
from datetime import datetime, timedelta

def test_three_source_integration():
    """测试三数据源集成"""
    print("=" * 60)
    print("测试三数据源（Qlib -> OpenBB -> yfinance）集成")
    print("=" * 60)
    
    # 创建数据适配器
    adapter = create_data_adapter(
        prefer_qlib=True,
        enable_openbb=True,
        fallback_to_yfinance=True
    )
    
    # 获取数据源信息
    print("\n1. 数据源信息:")
    source_info = adapter.get_data_source_info()
    for key, value in source_info.items():
        print(f"   {key}: {value}")
    
    # 测试股票列表
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    start_date = "2024-01-01"
    end_date = "2024-01-10"
    
    print(f"\n2. 测试股票数据获取 ({start_date} 到 {end_date}):")
    print("-" * 50)
    
    for symbol in test_symbols:
        print(f"\n测试 {symbol}:")
        
        # 检查数据可用性
        availability = adapter.check_data_availability(symbol)
        print(f"  数据可用性:")
        print(f"    Qlib: {availability.get('qlib_available', False)}")
        print(f"    OpenBB: {availability.get('openbb_available', False)}")
        print(f"    yfinance: {availability.get('yfinance_available', False)}")
        print(f"    推荐数据源: {availability.get('recommended_source', 'none')}")
        
        # 获取数据（自动回退）
        try:
            data = adapter.get_stock_data(symbol, start_date, end_date)
            if not data.empty:
                print(f"  数据获取成功: {len(data)} 条记录")
                print(f"  数据列: {list(data.columns)}")
                print(f"  日期范围: {data.index.min()} 到 {data.index.max()}")
                print(f"  样本数据:")
                print(data.head(2).to_string())
            else:
                print(f"  数据获取失败: 返回空数据")
        except Exception as e:
            print(f"  数据获取异常: {e}")
    
    print(f"\n3. 测试强制使用特定数据源:")
    print("-" * 50)
    
    test_symbol = "AAPL"
    sources = ["qlib", "openbb", "yfinance"]
    
    for source in sources:
        print(f"\n强制使用 {source.upper()} 获取 {test_symbol} 数据:")
        try:
            data = adapter.get_stock_data(
                test_symbol, 
                start_date, 
                end_date, 
                force_source=source
            )
            if not data.empty:
                print(f"  成功: {len(data)} 条记录")
                print(f"  价格范围: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            else:
                print(f"  失败: 返回空数据")
        except Exception as e:
            print(f"  异常: {e}")
    
    print(f"\n4. 测试多股票数据获取:")
    print("-" * 50)
    
    multi_symbols = ["AAPL", "GOOGL", "MSFT"]
    try:
        multi_data = adapter.get_multiple_stocks_data(
            multi_symbols, 
            start_date, 
            end_date
        )
        
        print(f"多股票数据获取结果:")
        for symbol, data in multi_data.items():
            if not data.empty:
                print(f"  {symbol}: {len(data)} 条记录")
            else:
                print(f"  {symbol}: 无数据")
    except Exception as e:
        print(f"多股票数据获取异常: {e}")
    
    print(f"\n5. 测试数据源回退机制:")
    print("-" * 50)
    
    # 测试一个可能在某些数据源中不存在的股票
    test_fallback_symbol = "UNKNOWN_STOCK"
    print(f"测试不存在的股票 {test_fallback_symbol}:")
    
    availability = adapter.check_data_availability(test_fallback_symbol)
    print(f"  数据可用性:")
    print(f"    Qlib: {availability.get('qlib_available', False)}")
    print(f"    OpenBB: {availability.get('openbb_available', False)}")
    print(f"    yfinance: {availability.get('yfinance_available', False)}")
    print(f"    推荐数据源: {availability.get('recommended_source', 'none')}")
    
    try:
        data = adapter.get_stock_data(test_fallback_symbol, start_date, end_date)
        if not data.empty:
            print(f"  意外获取到数据: {len(data)} 条记录")
        else:
            print(f"  正确处理: 返回空数据")
    except Exception as e:
        print(f"  正确处理异常: {e}")

if __name__ == "__main__":
    test_three_source_integration()