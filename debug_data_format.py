#!/usr/bin/env python3
"""
数据格式调试脚本

用于检查IB数据和yfinance数据的格式差异，
分析为什么数据一致性验证显示100%差异。

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


def debug_data_formats():
    """调试数据格式差异"""
    
    # 测试参数
    symbol = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"🔍 调试数据格式差异")
    print(f"📊 股票: {symbol}")
    print(f"📅 日期范围: {start_date} 到 {end_date}")
    print("=" * 80)
    
    # 获取IB数据
    print("\n📈 获取IB数据...")
    try:
        ib_provider = create_ib_provider()
        if ib_provider.is_available:
            ib_data = ib_provider.get_stock_data(symbol, start_date, end_date)
            
            print(f"✅ IB数据获取成功: {len(ib_data)} 条记录")
            print(f"📋 IB数据结构:")
            print(f"   - 索引类型: {type(ib_data.index)}")
            print(f"   - 索引名称: {ib_data.index.name}")
            print(f"   - 列名: {list(ib_data.columns)}")
            print(f"   - 数据类型: {dict(ib_data.dtypes)}")
            print(f"   - 日期范围: {ib_data.index.min()} 到 {ib_data.index.max()}")
            
            print(f"\n📊 IB数据前5行:")
            print(ib_data.head())
            
            print(f"\n📊 IB数据后5行:")
            print(ib_data.tail())
            
        else:
            print("❌ IB数据源不可用")
            ib_data = pd.DataFrame()
    except Exception as e:
        print(f"❌ IB数据获取失败: {e}")
        ib_data = pd.DataFrame()
    
    # 获取yfinance数据
    print("\n📈 获取yfinance数据...")
    try:
        ticker = yf.Ticker(symbol)
        yf_data = ticker.history(start=start_date, end=end_date)
        
        if not yf_data.empty:
            # 标准化列名为小写
            yf_data.columns = [col.lower() for col in yf_data.columns]
            
            print(f"✅ yfinance数据获取成功: {len(yf_data)} 条记录")
            print(f"📋 yfinance数据结构:")
            print(f"   - 索引类型: {type(yf_data.index)}")
            print(f"   - 索引名称: {yf_data.index.name}")
            print(f"   - 列名: {list(yf_data.columns)}")
            print(f"   - 数据类型: {dict(yf_data.dtypes)}")
            print(f"   - 日期范围: {yf_data.index.min()} 到 {yf_data.index.max()}")
            
            print(f"\n📊 yfinance数据前5行:")
            print(yf_data.head())
            
            print(f"\n📊 yfinance数据后5行:")
            print(yf_data.tail())
        else:
            print("❌ yfinance数据为空")
    except Exception as e:
        print(f"❌ yfinance数据获取失败: {e}")
        yf_data = pd.DataFrame()
    
    # 比较数据
    if not ib_data.empty and not yf_data.empty:
        print("\n🔍 数据比较分析:")
        
        # 检查共同日期
        ib_dates = set(ib_data.index.date) if hasattr(ib_data.index, 'date') else set(ib_data.index)
        yf_dates = set(yf_data.index.date) if hasattr(yf_data.index, 'date') else set(yf_data.index)
        
        common_dates = ib_dates.intersection(yf_dates)
        print(f"   - IB数据日期数量: {len(ib_dates)}")
        print(f"   - yfinance数据日期数量: {len(yf_dates)}")
        print(f"   - 共同日期数量: {len(common_dates)}")
        
        if common_dates:
            print(f"   - 共同日期示例: {sorted(list(common_dates))[:5]}")
            
            # 检查共同日期的数据
            print(f"\n📊 共同日期数据比较:")
            for date in sorted(list(common_dates))[:3]:  # 只检查前3个日期
                try:
                    if hasattr(ib_data.index, 'date'):
                        ib_row = ib_data[ib_data.index.date == date]
                    else:
                        ib_row = ib_data[ib_data.index == date]
                    
                    if hasattr(yf_data.index, 'date'):
                        yf_row = yf_data[yf_data.index.date == date]
                    else:
                        yf_row = yf_data[yf_data.index == date]
                    
                    if not ib_row.empty and not yf_row.empty:
                        print(f"   📅 {date}:")
                        if 'close' in ib_row.columns and 'close' in yf_row.columns:
                            ib_close = ib_row['close'].iloc[0]
                            yf_close = yf_row['close'].iloc[0]
                            diff = abs(ib_close - yf_close) / yf_close * 100
                            print(f"      IB收盘价: {ib_close:.2f}")
                            print(f"      YF收盘价: {yf_close:.2f}")
                            print(f"      差异: {diff:.2f}%")
                        else:
                            print(f"      缺少收盘价数据")
                except Exception as e:
                    print(f"      比较失败: {e}")
        else:
            print("   ❌ 没有共同日期，可能存在日期格式问题")
            
            # 显示日期格式示例
            print(f"\n📅 日期格式示例:")
            if len(ib_dates) > 0:
                print(f"   IB日期示例: {list(ib_dates)[:3]}")
            if len(yf_dates) > 0:
                print(f"   YF日期示例: {list(yf_dates)[:3]}")
    
    print("\n" + "=" * 80)
    print("🔍 调试完成")


if __name__ == "__main__":
    debug_data_formats()