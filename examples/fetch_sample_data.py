#!/usr/bin/env python3
"""
获取样本数据用于市场情绪分析测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manager import DataManager
from datetime import datetime, timedelta

def main():
    """获取样本数据"""
    print("🚀 获取样本数据用于市场情绪分析...")
    
    # 初始化数据管理器
    dm = DataManager()
    
    # 获取一些股票数据
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    success_count = 0
    
    for symbol in symbols:
        print(f'📊 获取 {symbol} 数据...')
        try:
            data = dm.get_stock_data(symbol, start_date, end_date)
            if not data.empty:
                print(f'  ✅ {symbol}: {len(data)} 条记录')
                success_count += 1
            else:
                print(f'  ❌ {symbol}: 数据为空')
        except Exception as e:
            print(f'  ❌ {symbol}: 获取失败 - {e}')
    
    print(f"\n🎯 成功获取 {success_count}/{len(symbols)} 个股票的数据")
    
    if success_count > 0:
        print("✅ 数据获取完成，现在可以运行市场情绪分析了！")
    else:
        print("❌ 没有成功获取任何数据")

if __name__ == "__main__":
    main()