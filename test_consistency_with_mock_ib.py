#!/usr/bin/env python3
"""
测试数据一致性验证（使用模拟IB数据）

使用模拟的IB数据来测试修复后的数据一致性验证逻辑

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_ib_data(symbol: str, days: int = 22) -> pd.DataFrame:
    """创建模拟的IB数据 - 仅用于测试和演示"""
    # 生成日期范围 - 模拟数据仅用于测试
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 生成交易日（排除周末）- 模拟数据仅用于演示
    date_range = pd.bdate_range(start=start_date, end=end_date)
    
    # 生成模拟价格数据 - 仅用于测试
    np.random.seed(42)  # 确保可重复性
    base_price = 150.0
    
    # 生成随机价格变化 - 模拟数据仅用于演示
    price_changes = np.random.normal(0, 0.02, len(date_range))  # 2%标准差
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 创建OHLCV数据 - 模拟数据仅用于测试
    data = []
    for i, (date, close_price) in enumerate(zip(date_range, prices)):
        # 生成开盘、最高、最低价 - 模拟数据仅用于演示
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.normal(1000000, 200000))  # 平均100万股交易量 - 仅用于测试
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': max(volume, 100000)  # 确保最小交易量
        })
    
    df = pd.DataFrame(data, index=date_range)
    df.index.name = 'date'
    
    return df


def compare_price_data_fixed(data1: pd.DataFrame, data2: pd.DataFrame, 
                           source1: str, source2: str) -> dict:
    """修复后的价格数据比较函数"""
    # 标准化索引为日期（移除时区信息）
    data1_normalized = data1.copy()
    data2_normalized = data2.copy()
    
    # 标准化索引
    if hasattr(data1_normalized.index, 'tz_localize'):
        data1_normalized.index = data1_normalized.index.tz_localize(None)
    if hasattr(data2_normalized.index, 'tz_localize'):
        data2_normalized.index = data2_normalized.index.tz_localize(None)
    
    # 转换为日期索引
    data1_normalized.index = pd.to_datetime(data1_normalized.index.date)
    data2_normalized.index = pd.to_datetime(data2_normalized.index.date)
    
    # 找到共同的日期
    common_dates = data1_normalized.index.intersection(data2_normalized.index)
    
    print(f"📊 {source1} vs {source2} 比较:")
    print(f"   {source1}数据: {len(data1_normalized)} 条记录")
    print(f"   {source2}数据: {len(data2_normalized)} 条记录")
    print(f"   共同日期: {len(common_dates)} 个")
    
    if len(common_dates) == 0:
        print("   ❌ 没有共同日期")
        return {
            'common_dates': 0,
            'close_correlation': 0.0,
            'close_mean_diff_pct': 100.0,
            'volume_correlation': 0.0,
            'volume_mean_diff_pct': 100.0
        }
    
    # 获取共同日期的数据
    df1_common = data1_normalized.loc[common_dates]
    df2_common = data2_normalized.loc[common_dates]
    
    # 价格比较
    close_corr = df1_common['close'].corr(df2_common['close'])
    close_diff_pct = abs((df1_common['close'] - df2_common['close']) / df1_common['close'] * 100).mean()
    
    # 成交量比较
    volume_corr = 0.0
    volume_diff_pct = 100.0
    
    if 'volume' in df1_common.columns and 'volume' in df2_common.columns:
        # 过滤掉零成交量的数据
        valid_volume = (df1_common['volume'] > 0) & (df2_common['volume'] > 0)
        if valid_volume.sum() > 0:
            volume_corr = df1_common.loc[valid_volume, 'volume'].corr(
                df2_common.loc[valid_volume, 'volume']
            )
            volume_diff_pct = abs(
                (df1_common.loc[valid_volume, 'volume'] - df2_common.loc[valid_volume, 'volume']) / 
                df1_common.loc[valid_volume, 'volume'] * 100
            ).mean()
    
    print(f"   📈 价格相关性: {close_corr:.4f}")
    print(f"   📈 价格平均差异: {close_diff_pct:.2f}%")
    print(f"   📊 成交量相关性: {volume_corr:.4f}")
    print(f"   📊 成交量平均差异: {volume_diff_pct:.2f}%")
    
    return {
        'common_dates': len(common_dates),
        'close_correlation': close_corr if not np.isnan(close_corr) else 0.0,
        'close_mean_diff_pct': close_diff_pct if not np.isnan(close_diff_pct) else 100.0,
        'volume_correlation': volume_corr if not np.isnan(volume_corr) else 0.0,
        'volume_mean_diff_pct': volume_diff_pct if not np.isnan(volume_diff_pct) else 100.0
    }


def test_consistency_validation():
    """测试数据一致性验证"""
    symbol = 'AAPL'
    print(f"🔍 测试数据一致性验证: {symbol}")
    print("=" * 80)
    
    # 创建模拟IB数据
    print("📊 创建模拟IB数据...")
    mock_ib_data = create_mock_ib_data(symbol)
    print(f"   ✅ 模拟IB数据创建成功: {len(mock_ib_data)} 条记录")
    print(f"   📅 日期范围: {mock_ib_data.index.min()} 到 {mock_ib_data.index.max()}")
    
    # 获取yfinance数据
    print("\n📊 获取yfinance数据...")
    try:
        ticker = yf.Ticker(symbol)
        yf_data = ticker.history(period='1mo')
        
        if not yf_data.empty:
            # 标准化列名为小写
            yf_data.columns = [col.lower() for col in yf_data.columns]
            print(f"   ✅ yfinance数据获取成功: {len(yf_data)} 条记录")
            print(f"   📅 日期范围: {yf_data.index.min()} 到 {yf_data.index.max()}")
        else:
            print("   ❌ yfinance数据为空")
            return
    except Exception as e:
        print(f"   ❌ yfinance数据获取失败: {e}")
        return
    
    # 测试修复后的比较逻辑
    print("\n🔍 测试修复后的比较逻辑:")
    print("-" * 60)
    
    # 测试1: 模拟IB数据 vs yfinance数据
    result1 = compare_price_data_fixed(mock_ib_data, yf_data, "Mock_IB", "yfinance")
    
    # 测试2: 创建相同的数据进行比较（应该完全一致）
    print(f"\n🔍 测试相同数据比较（应该完全一致）:")
    print("-" * 60)
    identical_data = mock_ib_data.copy()
    result2 = compare_price_data_fixed(mock_ib_data, identical_data, "Mock_IB", "Mock_IB_Copy")
    
    # 测试3: 创建轻微差异的数据
    print(f"\n🔍 测试轻微差异数据比较:")
    print("-" * 60)
    slightly_different_data = mock_ib_data.copy()
    # 添加1%的随机噪声
    np.random.seed(123)
    noise = np.random.normal(1, 0.01, len(slightly_different_data))
    slightly_different_data['close'] = slightly_different_data['close'] * noise
    
    result3 = compare_price_data_fixed(mock_ib_data, slightly_different_data, "Mock_IB", "Mock_IB_Noisy")
    
    print("\n" + "=" * 80)
    print("🔍 数据一致性验证测试完成")
    print(f"📊 测试结果总结:")
    print(f"   Mock_IB vs yfinance: 相关性={result1['close_correlation']:.4f}, 差异={result1['close_mean_diff_pct']:.2f}%")
    print(f"   相同数据比较: 相关性={result2['close_correlation']:.4f}, 差异={result2['close_mean_diff_pct']:.2f}%")
    print(f"   轻微差异数据: 相关性={result3['close_correlation']:.4f}, 差异={result3['close_mean_diff_pct']:.2f}%")


if __name__ == "__main__":
    test_consistency_validation()