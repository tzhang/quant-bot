#!/usr/bin/env python3
"""
测试yfinance数据获取修复

测试不同的日期范围和参数来修复yfinance数据获取问题

作者: AI Assistant
日期: 2024
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def test_yfinance_data():
    """测试yfinance数据获取"""
    
    symbol = 'AAPL'
    print(f"🔍 测试yfinance数据获取: {symbol}")
    print("=" * 60)
    
    # 测试不同的日期范围
    test_cases = [
        {
            'name': '最近30天',
            'start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d')
        },
        {
            'name': '最近7天',
            'start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d')
        },
        {
            'name': '使用period参数 - 1个月',
            'period': '1mo'
        },
        {
            'name': '使用period参数 - 3个月',
            'period': '3mo'
        },
        {
            'name': '固定历史日期',
            'start': '2024-01-01',
            'end': '2024-12-31'
        }
    ]
    
    ticker = yf.Ticker(symbol)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📊 测试 {i}: {case['name']}")
        
        try:
            if 'period' in case:
                data = ticker.history(period=case['period'])
            else:
                data = ticker.history(start=case['start'], end=case['end'])
            
            if not data.empty:
                print(f"   ✅ 成功获取 {len(data)} 条记录")
                print(f"   📅 日期范围: {data.index.min()} 到 {data.index.max()}")
                print(f"   📋 列名: {list(data.columns)}")
                print(f"   💰 最新收盘价: {data['Close'].iloc[-1]:.2f}")
            else:
                print(f"   ❌ 数据为空")
                
        except Exception as e:
            print(f"   ❌ 获取失败: {e}")
    
    # 测试股票信息
    print(f"\n📊 股票信息测试:")
    try:
        info = ticker.info
        print(f"   公司名称: {info.get('longName', 'N/A')}")
        print(f"   股票代码: {info.get('symbol', 'N/A')}")
        print(f"   交易所: {info.get('exchange', 'N/A')}")
        print(f"   货币: {info.get('currency', 'N/A')}")
    except Exception as e:
        print(f"   ❌ 获取股票信息失败: {e}")
    
    print("\n" + "=" * 60)
    print("🔍 yfinance测试完成")

if __name__ == "__main__":
    test_yfinance_data()