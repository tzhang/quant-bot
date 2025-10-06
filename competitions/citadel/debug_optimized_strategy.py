#!/usr/bin/env python3
"""
调试优化版策略的信号生成问题
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """加载数据"""
    print(f"📊 加载数据: {file_path}")
    
    # 读取CSV文件，跳过注释行
    df = pd.read_csv(file_path, comment='#')
    
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
    
    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    print(f"   数据行数: {len(df)}")
    print(f"   时间范围: {df.index.min()} 到 {df.index.max()}")
    
    return df

def calculate_technical_indicators(df, lookback_period=30):
    """计算技术指标"""
    print("🔧 计算技术指标...")
    
    # 移动平均线
    df['sma_short'] = df['close'].rolling(window=min(5, len(df))).mean()
    df['sma_long'] = df['close'].rolling(window=min(lookback_period, len(df))).mean()
    df['ema_short'] = df['close'].ewm(span=min(5, len(df))).mean()
    df['ema_long'] = df['close'].ewm(span=min(lookback_period, len(df))).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=min(12, len(df))).mean()
    exp2 = df['close'].ewm(span=min(26, len(df))).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=min(9, len(df))).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=min(14, len(df))).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, len(df))).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['bb_middle'] = df['close'].rolling(window=min(20, len(df))).mean()
    bb_std = df['close'].rolling(window=min(20, len(df))).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 波动率
    df['volatility'] = df['close'].pct_change().rolling(window=min(20, len(df))).std() * np.sqrt(252)
    
    # 成交量指标
    df['volume_sma'] = df['volume'].rolling(window=min(20, len(df))).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # 价格动量
    df['momentum'] = df['close'].pct_change(periods=min(5, len(df)-1))
    
    print("   技术指标计算完成")
    return df

def generate_signals(df, signal_threshold=0.08):
    """生成交易信号"""
    print("🔄 生成交易信号...")
    
    # 信号权重（优化版）
    momentum_weight = 0.35
    mean_reversion_weight = 0.25
    volatility_weight = 0.20
    microstructure_weight = 0.20
    
    # 动量信号
    momentum_signal = np.where(df['macd'] > df['macd_signal'], 1, -1) * 0.3
    momentum_signal += np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0)) * 0.4
    momentum_signal += np.where(df['close'] > df['sma_short'], 1, -1) * 0.3
    
    # 均值回归信号
    mean_reversion_signal = np.where(df['bb_position'] > 0.8, -1, 
                                   np.where(df['bb_position'] < 0.2, 1, 0)) * 0.6
    mean_reversion_signal += np.where(df['close'] > df['ema_long'], -0.2, 0.2) * 0.4
    
    # 波动率信号
    volatility_signal = np.where(df['volatility'] > df['volatility'].quantile(0.8), -0.5, 0.5)
    
    # 微观结构信号
    microstructure_signal = np.where(df['volume_ratio'] > 1.5, 0.3, -0.3)
    microstructure_signal += np.where(df['momentum'] > 0, 0.2, -0.2)
    
    # 综合信号
    df['signal'] = (momentum_signal * momentum_weight + 
                   mean_reversion_signal * mean_reversion_weight +
                   volatility_signal * volatility_weight +
                   microstructure_signal * microstructure_weight)
    
    # 过滤条件
    min_volume_ratio = 0.5
    max_volatility = 0.5
    
    # 应用过滤条件
    volume_filter = df['volume_ratio'] >= min_volume_ratio
    volatility_filter = df['volatility'] <= max_volatility
    
    # 趋势确认
    trend_filter = True  # 简化处理
    
    # 应用所有过滤条件
    valid_signals = volume_filter & volatility_filter & trend_filter
    df.loc[~valid_signals, 'signal'] = 0
    
    # 生成买入卖出信号
    df['buy_signal'] = (df['signal'] > signal_threshold).astype(int)
    df['sell_signal'] = (df['signal'] < -signal_threshold).astype(int)
    
    print(f"   信号阈值: ±{signal_threshold}")
    print(f"   信号范围: {df['signal'].min():.4f} 到 {df['signal'].max():.4f}")
    print(f"   信号均值: {df['signal'].mean():.4f}")
    print(f"   信号标准差: {df['signal'].std():.4f}")
    print(f"   买入信号数量: {df['buy_signal'].sum()}")
    print(f"   卖出信号数量: {df['sell_signal'].sum()}")
    print(f"   总交易信号: {df['buy_signal'].sum() + df['sell_signal'].sum()}")
    
    return df

def analyze_filters(df):
    """分析过滤条件的影响"""
    print("\n🔍 分析过滤条件:")
    
    min_volume_ratio = 0.5
    max_volatility = 0.5
    
    volume_filter = df['volume_ratio'] >= min_volume_ratio
    volatility_filter = df['volatility'] <= max_volatility
    
    print(f"   成交量过滤 (>= {min_volume_ratio}): {volume_filter.sum()}/{len(df)} 行通过")
    print(f"   波动率过滤 (<= {max_volatility}): {volatility_filter.sum()}/{len(df)} 行通过")
    print(f"   同时通过两个过滤条件: {(volume_filter & volatility_filter).sum()}/{len(df)} 行")
    
    # 分析各个信号组件
    print("\n📊 信号组件分析:")
    print(f"   成交量比率范围: {df['volume_ratio'].min():.4f} 到 {df['volume_ratio'].max():.4f}")
    print(f"   波动率范围: {df['volatility'].min():.4f} 到 {df['volatility'].max():.4f}")
    print(f"   RSI范围: {df['rsi'].min():.4f} 到 {df['rsi'].max():.4f}")
    print(f"   布林带位置范围: {df['bb_position'].min():.4f} 到 {df['bb_position'].max():.4f}")

def main():
    """主函数"""
    print("🎯 调试优化版策略信号生成")
    print("=" * 60)
    
    # 加载数据
    data_file = "examples/data_cache/ohlcv_AAPL_20251005_200622.csv"
    df = load_data(data_file)
    
    # 计算技术指标
    df = calculate_technical_indicators(df, lookback_period=30)
    
    # 生成信号
    df = generate_signals(df, signal_threshold=0.08)
    
    # 分析过滤条件
    analyze_filters(df)
    
    # 保存调试数据
    debug_file = "competitions/citadel/debug_optimized_signals.csv"
    df.to_csv(debug_file)
    print(f"\n💾 调试数据保存到: {debug_file}")
    
    # 建议
    print("\n💡 改进建议:")
    signal_max = abs(df['signal']).max()
    if signal_max < 0.08:
        print(f"   1. 降低信号阈值到 {signal_max * 0.8:.4f} 或更低")
    
    print("   2. 调整信号权重，增强信号强度")
    print("   3. 放宽过滤条件")
    print("   4. 检查技术指标计算是否正确")
    print("   5. 考虑使用更多历史数据")

if __name__ == "__main__":
    main()