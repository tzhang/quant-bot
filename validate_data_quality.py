#!/usr/bin/env python3
"""
数据质量验证脚本

验证股票数据的完整性、准确性和统计信息
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.database.dao import stock_data_dao
import pandas as pd
from datetime import datetime

def validate_data_quality():
    """验证数据质量"""
    print('📊 数据质量验证报告')
    print('='*60)

    # 获取所有股票代码
    symbols = stock_data_dao.get_symbols()
    print(f'📈 数据库中的股票数量: {len(symbols)}')
    print(f'📋 股票代码: {symbols}')

    print('\n📅 数据时间范围分析:')
    for symbol in symbols:
        if symbol == 'TEST':  # 跳过测试数据
            continue
        records = stock_data_dao.get_by_symbol_and_date_range(
            symbol, 
            datetime(2020, 1, 1), 
            datetime(2025, 1, 1)
        )
        if records:
            dates = [r.date for r in records]
            print(f'{symbol}: {min(dates).strftime("%Y-%m-%d")} 到 {max(dates).strftime("%Y-%m-%d")} ({len(records)}条记录)')

    print('\n💰 价格数据统计:')
    for symbol in symbols:
        if symbol == 'TEST':
            continue
        records = stock_data_dao.get_by_symbol_and_date_range(
            symbol, 
            datetime(2020, 1, 1), 
            datetime(2025, 1, 1)
        )
        if records:
            prices = [float(r.close) for r in records if r.close]
            volumes = [int(r.volume) for r in records if r.volume]
            if prices:
                print(f'{symbol}: 收盘价 ${min(prices):.2f}-${max(prices):.2f}, 平均成交量 {sum(volumes)/len(volumes):,.0f}')

    print('\n🔍 数据完整性检查:')
    for symbol in symbols:
        if symbol == 'TEST':
            continue
        records = stock_data_dao.get_by_symbol_and_date_range(
            symbol, 
            datetime(2020, 1, 1), 
            datetime(2025, 1, 1)
        )
        missing_data = 0
        for r in records:
            if not all([r.open, r.high, r.low, r.close, r.volume]):
                missing_data += 1
        completeness = (len(records) - missing_data) / len(records) * 100 if records else 0
        print(f'{symbol}: 数据完整性 {completeness:.1f}% ({len(records)-missing_data}/{len(records)})')

    print('\n📊 数据统计摘要:')
    total_records = 0
    valid_symbols = [s for s in symbols if s != 'TEST']
    for symbol in valid_symbols:
        records = stock_data_dao.get_by_symbol_and_date_range(
            symbol, 
            datetime(2020, 1, 1), 
            datetime(2025, 1, 1)
        )
        total_records += len(records)
    print(f'有效股票数量: {len(valid_symbols)}')
    print(f'总记录数: {total_records}')
    print(f'平均每股记录数: {total_records/len(valid_symbols):.1f}')

    print('\n✅ 数据质量验证完成！')

if __name__ == "__main__":
    validate_data_quality()