#!/usr/bin/env python3
"""
测试修改后的DataManager功能
验证Qlib数据源集成是否正常工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_manager import DataManager
import pandas as pd
from datetime import datetime, timedelta

def test_data_manager():
    """测试DataManager的各项功能"""
    print("=" * 60)
    print("测试修改后的DataManager功能")
    print("=" * 60)
    
    # 创建DataManager实例
    print("\n1. 创建DataManager实例...")
    dm = DataManager()
    
    # 获取缓存信息和数据源信息
    print("\n2. 获取数据源信息...")
    cache_info = dm.get_cache_info()
    print(f"缓存目录: {cache_info['cache_dir']}")
    print(f"缓存文件数: {cache_info['file_count']}")
    print(f"缓存大小: {cache_info['total_size_mb']} MB")
    print(f"启用缓存: {cache_info['enable_cache']}")
    
    data_source = cache_info.get('data_source', {})
    print(f"\n数据源信息:")
    print(f"  Qlib可用: {data_source.get('qlib_available', False)}")
    print(f"  主要数据源: {data_source.get('primary_source', 'unknown')}")
    if 'total_instruments' in data_source:
        print(f"  总股票数: {data_source['total_instruments']}")
    if 'trading_days' in data_source:
        print(f"  交易日数: {data_source['trading_days']}")
    if 'date_range' in data_source:
        print(f"  日期范围: {data_source['date_range']}")
    
    # 获取可用股票代码
    print("\n3. 获取可用股票代码...")
    available_symbols = dm.get_available_symbols()
    print(f"可用股票数量: {len(available_symbols)}")
    print(f"前10个股票代码: {available_symbols[:10]}")
    
    # 测试单只股票数据获取
    print("\n4. 测试单只股票数据获取...")
    test_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-03-01'
    
    print(f"获取 {test_symbol} 从 {start_date} 到 {end_date} 的数据...")
    
    # 检查数据可用性
    is_available = dm.check_data_availability(test_symbol, start_date, end_date)
    print(f"数据可用性: {is_available}")
    
    if is_available:
        try:
            stock_data = dm.get_stock_data(test_symbol, start_date, end_date)
            if not stock_data.empty:
                print(f"成功获取数据: {len(stock_data)} 条记录")
                print(f"数据列: {list(stock_data.columns)}")
                print(f"日期范围: {stock_data.index.min()} 到 {stock_data.index.max()}")
                print("\n前5行数据:")
                print(stock_data.head())
            else:
                print("获取到的数据为空")
        except Exception as e:
            print(f"获取数据失败: {e}")
    
    # 测试多只股票数据获取
    print("\n5. 测试多只股票数据获取...")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    print(f"获取股票: {test_symbols}")
    
    try:
        multiple_data = dm.get_multiple_stocks_data(test_symbols, start_date, end_date)
        print(f"成功获取 {len(multiple_data)} 只股票数据:")
        for symbol, data in multiple_data.items():
            print(f"  {symbol}: {len(data)} 条记录")
    except Exception as e:
        print(f"批量获取数据失败: {e}")
    
    # 测试基本面数据获取（如果支持）
    print("\n6. 测试基本面数据获取...")
    try:
        fundamental_data = dm.get_fundamental_data(test_symbol)
        if fundamental_data:
            print(f"成功获取 {test_symbol} 基本面数据")
            print(f"数据项数量: {len(fundamental_data)}")
            # 显示部分基本面数据
            sample_keys = list(fundamental_data.keys())[:5]
            for key in sample_keys:
                print(f"  {key}: {fundamental_data[key]}")
        else:
            print("未获取到基本面数据")
    except Exception as e:
        print(f"获取基本面数据失败: {e}")
    
    print("\n" + "=" * 60)
    print("DataManager功能测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_data_manager()