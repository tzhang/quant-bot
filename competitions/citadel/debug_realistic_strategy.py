#!/usr/bin/env python3
"""
调试现实版策略 - 分析信号生成问题
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def debug_strategy():
    """调试策略信号生成"""
    print("🔍 调试现实版策略信号生成")
    print("=" * 50)
    
    # 加载数据
    data_path = 'examples/data_cache/ohlcv_AAPL_20251005_200622.csv'
    print(f"📊 加载数据: {data_path}")
    
    # 读取数据，跳过注释行
    data = pd.read_csv(data_path, comment='#')
    
    # 重命名列
    column_mapping = {
        'Price': 'timestamp',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    data = data.rename(columns=column_mapping)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   数据行数: {len(data)}")
    print(f"   数据列: {list(data.columns)}")
    print(f"   时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    
    # 计算技术指标
    print("\n🔧 计算技术指标...")
    
    # 移动平均线
    data['sma_5'] = data['close'].rolling(5).mean()
    data['sma_10'] = data['close'].rolling(10).mean()
    data['sma_20'] = data['close'].rolling(20).mean()
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
    data['bb_middle'] = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # 波动率
    data['volatility'] = data['close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # 成交量指标
    data['volume_sma'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma']
    
    # 价格变化
    data['returns'] = data['close'].pct_change()
    data['returns_5d'] = data['close'].pct_change(5)
    
    print("   技术指标计算完成")
    
    # 分析信号生成
    print("\n📊 分析信号生成...")
    
    lookback_period = 20
    signal_threshold = 0.5
    
    signals = []
    signal_components = []
    
    for i in range(len(data)):
        if i < lookback_period:
            signals.append(0)
            signal_components.append({
                'momentum': 0,
                'mean_reversion': 0,
                'volatility': 0,
                'microstructure': 0,
                'total': 0
            })
            continue
        
        current_data = data.iloc[i]
        
        # 动量信号
        momentum_signal = 0
        if not pd.isna(current_data['macd']) and not pd.isna(current_data['macd_signal']):
            if current_data['macd'] > current_data['macd_signal']:
                momentum_signal += 0.3
            else:
                momentum_signal -= 0.3
        
        if not pd.isna(current_data['rsi']):
            if current_data['rsi'] > 70:
                momentum_signal -= 0.2
            elif current_data['rsi'] < 30:
                momentum_signal += 0.2
        
        # 均值回归信号
        mean_reversion_signal = 0
        if not pd.isna(current_data['bb_upper']) and not pd.isna(current_data['bb_lower']):
            bb_position = (current_data['close'] - current_data['bb_lower']) / (current_data['bb_upper'] - current_data['bb_lower'])
            if bb_position > 0.8:
                mean_reversion_signal -= 0.4
            elif bb_position < 0.2:
                mean_reversion_signal += 0.4
        
        # 波动率信号
        volatility_signal = 0
        if not pd.isna(current_data['volatility']):
            vol_percentile = np.percentile(data['volatility'].dropna(), 50)
            if current_data['volatility'] > vol_percentile * 1.5:
                volatility_signal -= 0.2
            elif current_data['volatility'] < vol_percentile * 0.5:
                volatility_signal += 0.1
        
        # 微观结构信号
        microstructure_signal = 0
        if not pd.isna(current_data['volume_ratio']):
            if current_data['volume_ratio'] > 1.5:
                if current_data['returns'] > 0:
                    microstructure_signal += 0.2
                else:
                    microstructure_signal -= 0.2
        
        # 综合信号
        total_signal = (momentum_signal * 0.25 +
                       mean_reversion_signal * 0.25 +
                       volatility_signal * 0.25 +
                       microstructure_signal * 0.25)
        
        signals.append(total_signal)
        signal_components.append({
            'momentum': momentum_signal,
            'mean_reversion': mean_reversion_signal,
            'volatility': volatility_signal,
            'microstructure': microstructure_signal,
            'total': total_signal
        })
    
    data['signal'] = signals
    
    # 统计分析
    print(f"\n📈 信号统计分析:")
    print(f"   信号范围: {min(signals):.4f} 到 {max(signals):.4f}")
    print(f"   信号均值: {np.mean(signals):.4f}")
    print(f"   信号标准差: {np.std(signals):.4f}")
    print(f"   信号阈值: ±{signal_threshold}")
    
    # 统计超过阈值的信号
    buy_signals = [s for s in signals if s > signal_threshold]
    sell_signals = [s for s in signals if s < -signal_threshold]
    
    print(f"   买入信号数量: {len(buy_signals)}")
    print(f"   卖出信号数量: {len(sell_signals)}")
    print(f"   总交易信号: {len(buy_signals) + len(sell_signals)}")
    
    # 显示最强的几个信号
    if signals:
        max_signal_idx = np.argmax(np.abs(signals))
        max_signal = signals[max_signal_idx]
        max_components = signal_components[max_signal_idx]
        
        print(f"\n🎯 最强信号 (第{max_signal_idx}行):")
        print(f"   总信号: {max_signal:.4f}")
        print(f"   动量: {max_components['momentum']:.4f}")
        print(f"   均值回归: {max_components['mean_reversion']:.4f}")
        print(f"   波动率: {max_components['volatility']:.4f}")
        print(f"   微观结构: {max_components['microstructure']:.4f}")
        print(f"   价格: ${data.iloc[max_signal_idx]['close']:.2f}")
        print(f"   RSI: {data.iloc[max_signal_idx]['rsi']:.2f}")
    
    # 保存调试数据
    debug_data = data[['timestamp', 'close', 'signal', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'volatility']].copy()
    debug_file = 'competitions/citadel/debug_signals.csv'
    debug_data.to_csv(debug_file, index=False)
    print(f"\n💾 调试数据保存到: {debug_file}")
    
    # 建议
    print(f"\n💡 建议:")
    if len(buy_signals) + len(sell_signals) == 0:
        print("   ⚠️  没有信号超过阈值，建议:")
        print("   1. 降低信号阈值 (当前: 0.5)")
        print("   2. 调整信号权重")
        print("   3. 增加更多数据")
        print("   4. 检查技术指标计算")
    else:
        print("   ✅ 信号生成正常")

if __name__ == "__main__":
    debug_strategy()