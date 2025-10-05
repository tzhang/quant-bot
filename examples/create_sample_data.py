#!/usr/bin/env python3
"""
创建模拟数据用于市场情绪分析测试
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_sample_stock_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    生成模拟股票数据
    
    Args:
        symbol: 股票代码
        days: 天数
        
    Returns:
        模拟股票数据DataFrame
    """
    # 生成日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 过滤掉周末
    dates = dates[dates.weekday < 5]
    
    # 生成价格数据
    np.random.seed(hash(symbol) % 2**32)  # 使用symbol作为种子，确保每次生成相同数据
    
    # 基础价格
    base_prices = {
        'AAPL': 150,
        'MSFT': 300,
        'GOOGL': 2500,
        'SPY': 400,
        'QQQ': 350,
        'NVDA': 800,
        'TSLA': 200,
        'AMZN': 3000,
        'META': 250,
        'NFLX': 400,
        'JPM': 150,
        'JNJ': 160,
        'PG': 140,
        'KO': 60,
        'IWM': 180
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # 生成随机价格走势
    returns = np.random.normal(0.001, 0.02, len(dates))  # 日收益率
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1))  # 确保价格不为负
    
    # 生成OHLCV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # 生成开盘价（基于前一日收盘价）
        if i == 0:
            open_price = close * (1 + np.random.normal(0, 0.005))
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        # 生成高低价
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        
        # 生成成交量
        volume = int(np.random.lognormal(15, 0.5))
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def create_cache_files():
    """创建缓存文件"""
    print("🚀 创建模拟数据缓存文件...")
    
    # 创建缓存目录
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # 股票列表
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'NVDA', 'TSLA', 
               'AMZN', 'META', 'NFLX', 'JPM', 'JNJ', 'PG', 'KO', 'IWM']
    
    success_count = 0
    
    for symbol in symbols:
        try:
            # 生成数据
            data = generate_sample_stock_data(symbol)
            
            # 创建缓存文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ohlcv_{symbol}_{timestamp}.csv"
            filepath = cache_dir / filename
            
            # 写入文件（包含元数据）
            with open(filepath, 'w') as f:
                f.write(f"# Symbol: {symbol}\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                # 添加Price列（索引）然后是OHLCV数据
                data_with_price = data.copy()
                data_with_price.insert(0, 'Price', data_with_price.index.strftime('%Y-%m-%d'))
                data_with_price.to_csv(f, index=False)
            
            print(f"  ✅ {symbol}: {len(data)} 条记录 -> {filename}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ {symbol}: 生成失败 - {e}")
    
    print(f"\n🎯 成功创建 {success_count}/{len(symbols)} 个股票的缓存文件")
    print("✅ 模拟数据创建完成，现在可以运行市场情绪分析了！")

if __name__ == "__main__":
    create_cache_files()