#!/usr/bin/env python3
"""
调试时间戳格式问题
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(file_path):
    """加载数据"""
    print(f"📊 加载数据: {file_path}")
    
    # 读取CSV文件，跳过注释行
    df = pd.read_csv(file_path, comment='#')
    
    print("原始数据前5行:")
    print(df.head())
    print(f"原始Price列类型: {df['Price'].dtype}")
    print(f"原始Price列示例: {df['Price'].iloc[0]}")
    
    # 重命名列以匹配预期格式
    column_mapping = {
        'Price': 'timestamp',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    df = df.rename(columns=column_mapping)
    
    print(f"\n重命名后timestamp列类型: {df['timestamp'].dtype}")
    print(f"重命名后timestamp列示例: {df['timestamp'].iloc[0]}")
    
    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    print(f"\n转换后时间戳类型: {df.index.dtype}")
    print(f"转换后时间戳示例: {df.index[0]}")
    print(f"时间戳是否为datetime: {isinstance(df.index[0], pd.Timestamp)}")
    
    # 重置索引，保留timestamp作为列
    df = df.reset_index()
    
    print(f"\n重置索引后timestamp列类型: {df['timestamp'].dtype}")
    print(f"重置索引后timestamp列示例: {df['timestamp'].iloc[0]}")
    
    return df

def main():
    """主函数"""
    print("🎯 调试时间戳格式问题")
    print("=" * 60)
    
    # 加载数据
    data_file = "examples/data_cache/ohlcv_AAPL_20251005_200622.csv"
    df = load_data(data_file)
    
    # 测试时间戳操作
    print("\n🔍 测试时间戳操作:")
    timestamp = df['timestamp'].iloc[0]
    print(f"timestamp类型: {type(timestamp)}")
    print(f"timestamp值: {timestamp}")
    
    # 测试时间差计算
    if len(df) > 1:
        timestamp1 = df['timestamp'].iloc[0]
        timestamp2 = df['timestamp'].iloc[1]
        time_diff = (timestamp2 - timestamp1).total_seconds()
        print(f"时间差计算: {time_diff} 秒")
    
    # 测试日期字符串格式化
    date_str = timestamp.strftime('%Y-%m-%d')
    print(f"日期字符串: {date_str}")

if __name__ == "__main__":
    main()