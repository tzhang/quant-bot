#!/usr/bin/env python3
"""
详细调试交易执行过程
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(file_path):
    """加载数据"""
    df = pd.read_csv(file_path, comment='#')
    column_mapping = {
        'Price': 'timestamp',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.reset_index(drop=True)
    return df

def calculate_technical_indicators(data):
    """计算技术指标"""
    # 移动平均线
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    
    # MACD
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # 波动率
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
    
    # 成交量指标
    data['volume_sma'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma']
    
    return data

def generate_signal(data, i):
    """生成交易信号"""
    if i < 30:  # 需要足够的历史数据
        return 0
    
    current_data = data.iloc[i]
    
    # 权重
    momentum_weight = 0.35
    mean_reversion_weight = 0.25
    volatility_weight = 0.20
    microstructure_weight = 0.20
    
    # 1. 动量信号
    momentum_signal = 0
    
    # MACD信号
    if not pd.isna(current_data['macd']) and not pd.isna(current_data['macd_signal']):
        if current_data['macd'] > current_data['macd_signal']:
            momentum_signal += 0.4
        else:
            momentum_signal -= 0.4
    
    # RSI信号
    if not pd.isna(current_data['rsi']):
        if current_data['rsi'] > 70:
            momentum_signal -= 0.3
        elif current_data['rsi'] < 30:
            momentum_signal += 0.3
    
    # 移动平均线信号
    if not pd.isna(current_data['sma_5']) and not pd.isna(current_data['sma_20']):
        if current_data['sma_5'] > current_data['sma_20']:
            momentum_signal += 0.3
        else:
            momentum_signal -= 0.3
    
    # 2. 均值回归信号
    mean_reversion_signal = 0
    
    if not pd.isna(current_data['bb_position']):
        if current_data['bb_position'] > 0.8:
            mean_reversion_signal -= 0.6
        elif current_data['bb_position'] < 0.2:
            mean_reversion_signal += 0.6
    
    # 3. 波动率信号
    volatility_signal = 0
    if not pd.isna(current_data['volatility']):
        vol_percentile = np.percentile(data['volatility'].dropna(), 70)
        if current_data['volatility'] > vol_percentile * 1.5:
            volatility_signal -= 0.4
        elif current_data['volatility'] < vol_percentile * 0.3:
            volatility_signal += 0.2
    
    # 4. 微观结构信号
    microstructure_signal = 0
    if not pd.isna(current_data['volume_ratio']):
        if current_data['volume_ratio'] > 1.5:
            if current_data['returns'] > 0:
                microstructure_signal += 0.3
            else:
                microstructure_signal -= 0.3
    
    # 综合信号
    total_signal = (momentum_signal * momentum_weight +
                   mean_reversion_signal * mean_reversion_weight +
                   volatility_signal * volatility_weight +
                   microstructure_signal * microstructure_weight)
    
    return total_signal

def apply_filters(data, i, signal):
    """应用过滤条件"""
    if signal == 0:
        return 0
    
    current_data = data.iloc[i]
    
    # 成交量过滤
    min_volume_ratio = 0.5
    if not pd.isna(current_data['volume_ratio']):
        if current_data['volume_ratio'] < min_volume_ratio:
            return 0
    
    # 波动率过滤
    max_volatility = 0.5
    if not pd.isna(current_data['volatility']):
        if current_data['volatility'] > max_volatility:
            return 0
    
    return signal

def debug_backtest():
    """调试回测过程"""
    print("🎯 调试交易执行过程")
    print("=" * 60)
    
    # 加载数据
    data_file = "examples/data_cache/ohlcv_AAPL_20251005_200622.csv"
    data = load_data(data_file)
    print(f"📊 数据行数: {len(data)}")
    
    # 计算技术指标
    data = calculate_technical_indicators(data)
    
    # 策略参数
    signal_threshold = 0.08
    position = 0
    cash = 1000000
    trades = []
    
    print(f"\n🔍 信号阈值: ±{signal_threshold}")
    print("\n逐行分析:")
    
    strong_signals = []
    
    for i in range(len(data)):
        current_data = data.iloc[i]
        timestamp = current_data['timestamp']
        price = current_data['close']
        
        # 生成信号
        signal = generate_signal(data, i)
        filtered_signal = apply_filters(data, i, signal)
        
        # 记录强信号
        if abs(signal) > 0.05:  # 记录较强的信号
            strong_signals.append({
                'index': i,
                'timestamp': timestamp,
                'price': price,
                'raw_signal': signal,
                'filtered_signal': filtered_signal,
                'above_threshold': abs(filtered_signal) > signal_threshold,
                'position': position
            })
        
        # 检查交易条件
        if abs(filtered_signal) > signal_threshold:
            if filtered_signal > 0 and position == 0:  # 买入信号
                shares = int((cash * 0.25) / price)  # 25%仓位
                if shares > 0:
                    cost = shares * price
                    cash -= cost
                    position = shares
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'signal': filtered_signal
                    })
                    print(f"  [{i:2d}] {timestamp.strftime('%Y-%m-%d')} BUY  信号:{filtered_signal:6.3f} 价格:{price:7.2f} 股数:{shares}")
            
            elif filtered_signal < 0 and position > 0:  # 卖出信号
                proceeds = position * price
                cash += proceeds
                trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': price,
                    'shares': position,
                    'signal': filtered_signal
                })
                print(f"  [{i:2d}] {timestamp.strftime('%Y-%m-%d')} SELL 信号:{filtered_signal:6.3f} 价格:{price:7.2f} 股数:{position}")
                position = 0
    
    print(f"\n📊 结果统计:")
    print(f"   总交易次数: {len(trades)}")
    print(f"   强信号数量: {len(strong_signals)}")
    
    if strong_signals:
        print(f"\n🔍 强信号详情:")
        for sig in strong_signals[:10]:  # 显示前10个强信号
            print(f"   [{sig['index']:2d}] {sig['timestamp'].strftime('%Y-%m-%d')} "
                  f"原始信号:{sig['raw_signal']:6.3f} 过滤后:{sig['filtered_signal']:6.3f} "
                  f"超阈值:{sig['above_threshold']} 仓位:{sig['position']}")
    
    # 分析信号分布
    all_signals = []
    for i in range(30, len(data)):  # 从第30行开始
        signal = generate_signal(data, i)
        filtered_signal = apply_filters(data, i, signal)
        all_signals.append(filtered_signal)
    
    all_signals = np.array(all_signals)
    print(f"\n📈 信号分布分析:")
    print(f"   信号数量: {len(all_signals)}")
    print(f"   信号范围: {all_signals.min():.4f} 到 {all_signals.max():.4f}")
    print(f"   信号均值: {all_signals.mean():.4f}")
    print(f"   信号标准差: {all_signals.std():.4f}")
    print(f"   超过阈值的信号: {np.sum(np.abs(all_signals) > signal_threshold)}")

if __name__ == "__main__":
    debug_backtest()