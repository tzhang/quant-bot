#!/usr/bin/env python3
"""
调试信号过滤过程
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from citadel_optimized_strategy import CitadelOptimizedStrategy

def debug_signal_filtering():
    """调试信号过滤过程"""
    print("🔍 调试信号过滤过程")
    print("=" * 60)
    
    # 初始化策略
    config = {
        'initial_capital': 1000000,
        'lookback_period': 30,
        'signal_threshold': 0.05,
        'position_limit': 0.25,
        'max_trade_size': 0.08,
        'stop_loss': 0.015,
        'take_profit': 0.045,
        'trailing_stop': 0.01,
        'max_daily_trades': 3,
        'min_trade_interval': 0,
        'momentum_weight': 0.35,
        'mean_reversion_weight': 0.25,
        'volatility_weight': 0.20,
        'microstructure_weight': 0.20,
        'min_volume_ratio': 1.2,
        'max_volatility': 0.5,
        'trend_confirmation': True
    }
    
    strategy = CitadelOptimizedStrategy(config)
    
    # 加载数据
    data_file = "examples/data_cache/ohlcv_AAPL_20251005_200622.csv"
    data = strategy.load_data(data_file)
    print(f"📊 加载数据: {len(data)} 行")
    
    # 计算技术指标
    data = strategy.calculate_technical_indicators(data)
    print("🔧 技术指标计算完成")
    
    # 分析信号生成和过滤过程
    signals_before_filter = []
    signals_after_filter = []
    filter_details = []
    
    for i in range(strategy.lookback_period, len(data)):
        # 生成原始信号
        raw_signal = strategy.generate_signal(data, i)
        signals_before_filter.append(raw_signal)
        
        # 应用过滤器
        filtered_signal = strategy.apply_filters(data, i, raw_signal)
        signals_after_filter.append(filtered_signal)
        
        # 记录过滤详情
        timestamp = data.iloc[i]['timestamp']
        volume_ratio = data.iloc[i]['volume'] / data.iloc[i-20:i]['volume'].mean() if i >= 20 else 1.0
        volatility = data.iloc[i]['atr'] / data.iloc[i]['close']
        
        # 趋势确认
        sma_20 = data.iloc[i]['sma_20']
        sma_50 = data.iloc[i]['sma_50']
        trend_up = sma_20 > sma_50
        
        filter_info = {
            'timestamp': timestamp,
            'raw_signal': raw_signal,
            'filtered_signal': filtered_signal,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'trend_up': trend_up,
            'volume_pass': volume_ratio >= strategy.min_volume_ratio,
            'volatility_pass': volatility <= strategy.max_volatility,
            'trend_pass': (raw_signal > 0 and trend_up) or (raw_signal < 0 and not trend_up) or not strategy.trend_confirmation
        }
        filter_details.append(filter_info)
    
    # 分析结果
    signals_before = np.array(signals_before_filter)
    signals_after = np.array(signals_after_filter)
    
    print("\n📈 信号分析结果:")
    print(f"   原始信号数量: {len(signals_before)}")
    print(f"   原始信号范围: {signals_before.min():.4f} 到 {signals_before.max():.4f}")
    print(f"   原始信号均值: {signals_before.mean():.4f}")
    print(f"   原始信号标准差: {signals_before.std():.4f}")
    
    print(f"\n   过滤后信号数量: {len(signals_after)}")
    print(f"   过滤后信号范围: {signals_after.min():.4f} 到 {signals_after.max():.4f}")
    print(f"   过滤后信号均值: {signals_after.mean():.4f}")
    print(f"   过滤后信号标准差: {signals_after.std():.4f}")
    
    # 统计强信号
    strong_signals_before = np.abs(signals_before) > strategy.signal_threshold
    strong_signals_after = np.abs(signals_after) > strategy.signal_threshold
    
    print(f"\n🎯 强信号统计 (阈值: ±{strategy.signal_threshold}):")
    print(f"   过滤前强信号: {strong_signals_before.sum()}")
    print(f"   过滤后强信号: {strong_signals_after.sum()}")
    print(f"   信号保留率: {strong_signals_after.sum() / max(strong_signals_before.sum(), 1) * 100:.1f}%")
    
    # 分析过滤条件
    filter_df = pd.DataFrame(filter_details)
    
    print(f"\n🔍 过滤条件分析:")
    print(f"   成交量过滤通过率: {filter_df['volume_pass'].mean() * 100:.1f}%")
    print(f"   波动率过滤通过率: {filter_df['volatility_pass'].mean() * 100:.1f}%")
    print(f"   趋势确认通过率: {filter_df['trend_pass'].mean() * 100:.1f}%")
    
    all_filters_pass = filter_df['volume_pass'] & filter_df['volatility_pass'] & filter_df['trend_pass']
    print(f"   所有过滤条件通过率: {all_filters_pass.mean() * 100:.1f}%")
    
    # 显示被过滤掉的强信号
    strong_filtered_out = filter_df[
        (np.abs(filter_df['raw_signal']) > strategy.signal_threshold) & 
        (np.abs(filter_df['filtered_signal']) <= strategy.signal_threshold)
    ]
    
    if len(strong_filtered_out) > 0:
        print(f"\n❌ 被过滤掉的强信号 ({len(strong_filtered_out)} 个):")
        for _, row in strong_filtered_out.head(10).iterrows():
            print(f"   {row['timestamp'].strftime('%Y-%m-%d')}: 原始信号={row['raw_signal']:.4f}, "
                  f"成交量={row['volume_pass']}, 波动率={row['volatility_pass']}, 趋势={row['trend_pass']}")
    
    # 保存调试数据
    debug_file = "competitions/citadel/debug_signal_filtering.csv"
    filter_df.to_csv(debug_file, index=False)
    print(f"\n💾 调试数据保存到: {debug_file}")
    
    # 建议
    print(f"\n💡 优化建议:")
    if filter_df['volume_pass'].mean() < 0.5:
        print(f"   - 成交量过滤过于严格，建议降低 min_volume_ratio (当前: {strategy.min_volume_ratio})")
    if filter_df['volatility_pass'].mean() < 0.5:
        print(f"   - 波动率过滤过于严格，建议提高 max_volatility (当前: {strategy.max_volatility})")
    if filter_df['trend_pass'].mean() < 0.5:
        print(f"   - 趋势确认过于严格，建议关闭 trend_confirmation")
    if strong_signals_after.sum() == 0:
        print(f"   - 考虑进一步降低信号阈值 (当前: {strategy.signal_threshold})")

if __name__ == "__main__":
    debug_signal_filtering()