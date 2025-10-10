#!/usr/bin/env python3
"""
索引比较调试脚本

专门调试IB数据和yfinance数据的索引比较问题

作者: AI Assistant
日期: 2024
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.ib_data_provider import create_ib_provider

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def debug_index_comparison():
    """调试索引比较问题"""
    
    # 测试参数
    symbol = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"🔍 调试索引比较问题")
    print(f"📊 股票: {symbol}")
    print(f"📅 日期范围: {start_date} 到 {end_date}")
    print("=" * 80)
    
    # 获取数据
    ib_provider = create_ib_provider()
    ib_data = ib_provider.get_stock_data(symbol, start_date, end_date)
    
    ticker = yf.Ticker(symbol)
    yf_data = ticker.history(start=start_date, end=end_date)
    yf_data.columns = [col.lower() for col in yf_data.columns]
    
    print(f"\n📈 数据获取结果:")
    print(f"   IB数据: {len(ib_data)} 条记录")
    print(f"   YF数据: {len(yf_data)} 条记录")
    
    # 详细检查索引
    print(f"\n🔍 索引详细分析:")
    print(f"   IB索引类型: {type(ib_data.index)}")
    print(f"   YF索引类型: {type(yf_data.index)}")
    
    print(f"\n   IB索引前3个:")
    for i, idx in enumerate(ib_data.index[:3]):
        print(f"     [{i}] {idx} (类型: {type(idx)})")
    
    print(f"\n   YF索引前3个:")
    for i, idx in enumerate(yf_data.index[:3]):
        print(f"     [{i}] {idx} (类型: {type(idx)})")
    
    # 测试索引交集
    print(f"\n🔍 索引交集测试:")
    common_indices = ib_data.index.intersection(yf_data.index)
    print(f"   直接交集数量: {len(common_indices)}")
    
    if len(common_indices) > 0:
        print(f"   交集示例: {common_indices[:3].tolist()}")
    else:
        print("   ❌ 没有直接交集")
        
        # 尝试转换为日期进行比较
        print(f"\n🔍 尝试日期转换比较:")
        
        # 转换IB索引为日期
        if hasattr(ib_data.index, 'date'):
            ib_dates = ib_data.index.date
        elif hasattr(ib_data.index, 'normalize'):
            ib_dates = ib_data.index.normalize()
        else:
            ib_dates = pd.to_datetime(ib_data.index).date
            
        # 转换YF索引为日期
        if hasattr(yf_data.index, 'date'):
            yf_dates = yf_data.index.date
        elif hasattr(yf_data.index, 'normalize'):
            yf_dates = yf_data.index.normalize()
        else:
            yf_dates = pd.to_datetime(yf_data.index).date
        
        print(f"   IB日期示例: {ib_dates[:3] if hasattr(ib_dates, '__getitem__') else list(ib_dates)[:3]}")
        print(f"   YF日期示例: {yf_dates[:3] if hasattr(yf_dates, '__getitem__') else list(yf_dates)[:3]}")
        
        # 找到共同日期
        ib_date_set = set(ib_dates) if hasattr(ib_dates, '__iter__') else {ib_dates}
        yf_date_set = set(yf_dates) if hasattr(yf_dates, '__iter__') else {yf_dates}
        
        common_dates = ib_date_set.intersection(yf_date_set)
        print(f"   共同日期数量: {len(common_dates)}")
        
        if common_dates:
            print(f"   共同日期示例: {sorted(list(common_dates))[:3]}")
    
    # 测试时区问题
    print(f"\n🔍 时区信息:")
    print(f"   IB索引时区: {getattr(ib_data.index, 'tz', 'None')}")
    print(f"   YF索引时区: {getattr(yf_data.index, 'tz', 'None')}")
    
    # 尝试标准化索引
    print(f"\n🔍 尝试标准化索引:")
    try:
        # 移除时区信息并标准化为日期
        ib_normalized = ib_data.copy()
        yf_normalized = yf_data.copy()
        
        # 标准化IB索引
        if hasattr(ib_normalized.index, 'tz_localize'):
            ib_normalized.index = ib_normalized.index.tz_localize(None)
        ib_normalized.index = pd.to_datetime(ib_normalized.index.date)
        
        # 标准化YF索引
        if hasattr(yf_normalized.index, 'tz_localize'):
            yf_normalized.index = yf_normalized.index.tz_localize(None)
        yf_normalized.index = pd.to_datetime(yf_normalized.index.date)
        
        # 重新测试交集
        normalized_common = ib_normalized.index.intersection(yf_normalized.index)
        print(f"   标准化后交集数量: {len(normalized_common)}")
        
        if len(normalized_common) > 0:
            print(f"   标准化后交集示例: {normalized_common[:3].tolist()}")
            
            # 测试数据一致性
            print(f"\n📊 标准化后数据一致性测试:")
            for date in normalized_common[:3]:
                ib_close = ib_normalized.loc[date, 'close']
                yf_close = yf_normalized.loc[date, 'close']
                diff = abs(ib_close - yf_close) / yf_close * 100
                print(f"   {date.date()}: IB={ib_close:.2f}, YF={yf_close:.2f}, 差异={diff:.4f}%")
        
    except Exception as e:
        print(f"   标准化失败: {e}")
    
    print("\n" + "=" * 80)
    print("🔍 索引比较调试完成")


if __name__ == "__main__":
    debug_index_comparison()