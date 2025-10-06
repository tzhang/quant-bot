#!/usr/bin/env python3
"""
Citadel Terminal AI Competition 数据分析脚本
专门用于分析和准备 Citadel 比赛的数据结构和特征
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ml.terminal_ai_tools import (
    RealTimeDataProcessor, 
    HighFrequencyStrategy, 
    AlgorithmOptimizer,
    PerformanceMonitor,
    create_terminal_ai_system,
    run_terminal_ai_simulation
)

class CitadelDataAnalyzer:
    """Citadel 比赛数据分析器"""
    
    def __init__(self, config_path=None):
        """初始化分析器"""
        if config_path is None:
            config_path = project_root / "configs" / "competitions" / "citadel_config.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.data_dir = Path(__file__).parent / "data"
        self.features_dir = Path(__file__).parent / "features"
        self.models_dir = Path(__file__).parent / "models"
        
        # 创建目录
        for dir_path in [self.data_dir, self.features_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def generate_sample_data(self, n_samples=10000, n_assets=50):
        """生成模拟的高频交易数据"""
        print(f"📊 生成 {n_samples} 条样本数据，{n_assets} 个资产...")
        
        # 时间序列
        start_time = datetime.now() - timedelta(days=30)
        timestamps = pd.date_range(start_time, periods=n_samples, freq='1min')
        
        data_list = []
        
        for asset_id in range(n_assets):
            # 生成价格数据
            np.random.seed(42 + asset_id)
            
            # 基础价格走势
            base_price = 100 + np.random.normal(0, 20)
            price_changes = np.random.normal(0, 0.001, n_samples)
            prices = base_price * np.exp(np.cumsum(price_changes))
            
            # OHLC 数据
            opens = prices * (1 + np.random.normal(0, 0.0005, n_samples))
            highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.001, n_samples)))
            lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.001, n_samples)))
            closes = prices
            
            # 成交量数据
            base_volume = 1000 + np.random.exponential(500, n_samples)
            volumes = base_volume * (1 + np.random.normal(0, 0.5, n_samples))
            volumes = np.maximum(volumes, 100)  # 最小成交量
            
            # 买卖价差
            spreads = np.random.exponential(0.01, n_samples)
            bid_prices = closes - spreads / 2
            ask_prices = closes + spreads / 2
            
            # 市场深度
            bid_sizes = np.random.exponential(1000, n_samples)
            ask_sizes = np.random.exponential(1000, n_samples)
            
            for i in range(n_samples):
                data_list.append({
                    'timestamp': timestamps[i],
                    'asset_id': f'ASSET_{asset_id:03d}',
                    'open': opens[i],
                    'high': highs[i],
                    'low': lows[i],
                    'close': closes[i],
                    'volume': volumes[i],
                    'bid_price': bid_prices[i],
                    'ask_price': ask_prices[i],
                    'bid_size': bid_sizes[i],
                    'ask_size': ask_sizes[i],
                    'spread': spreads[i]
                })
        
        df = pd.DataFrame(data_list)
        df = df.sort_values(['timestamp', 'asset_id']).reset_index(drop=True)
        
        # 保存数据
        data_file = self.data_dir / "sample_market_data.csv"
        df.to_csv(data_file, index=False)
        print(f"✅ 数据已保存到: {data_file}")
        
        return df
    
    def analyze_data_structure(self, df):
        """分析数据结构"""
        print("\n📈 数据结构分析:")
        print(f"数据形状: {df.shape}")
        print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        print(f"资产数量: {df['asset_id'].nunique()}")
        print(f"数据频率: 每分钟")
        
        print("\n📊 数据统计:")
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'spread']
        print(df[numeric_cols].describe())
        
        print("\n🔍 数据质量检查:")
        print(f"缺失值: {df.isnull().sum().sum()}")
        print(f"重复行: {df.duplicated().sum()}")
        
        # 价格一致性检查
        price_issues = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                      (df['low'] > df['open']) | (df['low'] > df['close'])
        print(f"价格不一致: {price_issues.sum()}")
        
        return {
            'shape': df.shape,
            'time_range': (df['timestamp'].min(), df['timestamp'].max()),
            'n_assets': df['asset_id'].nunique(),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'price_issues': price_issues.sum()
        }
    
    def extract_features(self, df):
        """提取 Citadel 比赛相关特征"""
        print("\n🔧 提取高频交易特征...")
        
        # 初始化实时数据处理器
        processor = RealTimeDataProcessor()
        
        feature_data = []
        
        for asset_id in df['asset_id'].unique():
            asset_data = df[df['asset_id'] == asset_id].copy()
            asset_data = asset_data.sort_values('timestamp').reset_index(drop=True)
            
            # 基础价格特征
            asset_data['returns'] = asset_data['close'].pct_change()
            asset_data['log_returns'] = np.log(asset_data['close'] / asset_data['close'].shift(1))
            
            # 波动率特征
            asset_data['volatility_5min'] = asset_data['returns'].rolling(5).std()
            asset_data['volatility_15min'] = asset_data['returns'].rolling(15).std()
            asset_data['volatility_60min'] = asset_data['returns'].rolling(60).std()
            
            # 动量特征
            asset_data['momentum_5min'] = asset_data['close'] / asset_data['close'].shift(5) - 1
            asset_data['momentum_15min'] = asset_data['close'] / asset_data['close'].shift(15) - 1
            asset_data['momentum_60min'] = asset_data['close'] / asset_data['close'].shift(60) - 1
            
            # 技术指标
            asset_data['rsi_14'] = processor._calculate_rsi(pd.Series(asset_data['close'].values), 14)
            asset_data['ma_5'] = asset_data['close'].rolling(5).mean()
            asset_data['ma_20'] = asset_data['close'].rolling(20).mean()
            asset_data['ma_ratio'] = asset_data['ma_5'] / asset_data['ma_20']
            
            # 成交量特征
            asset_data['volume_ma_5'] = asset_data['volume'].rolling(5).mean()
            asset_data['volume_ratio'] = asset_data['volume'] / asset_data['volume_ma_5']
            asset_data['vwap'] = (asset_data['close'] * asset_data['volume']).rolling(20).sum() / asset_data['volume'].rolling(20).sum()
            asset_data['price_vwap_ratio'] = asset_data['close'] / asset_data['vwap']
            
            # 买卖价差特征
            asset_data['spread_pct'] = asset_data['spread'] / asset_data['close']
            asset_data['spread_ma'] = asset_data['spread_pct'].rolling(10).mean()
            asset_data['spread_volatility'] = asset_data['spread_pct'].rolling(10).std()
            
            # 市场微观结构特征
            asset_data['bid_ask_imbalance'] = (asset_data['bid_size'] - asset_data['ask_size']) / (asset_data['bid_size'] + asset_data['ask_size'])
            asset_data['mid_price'] = (asset_data['bid_price'] + asset_data['ask_price']) / 2
            asset_data['price_impact'] = (asset_data['close'] - asset_data['mid_price']) / asset_data['spread']
            
            # 高频特征
            asset_data['price_acceleration'] = asset_data['returns'].diff()
            asset_data['volume_acceleration'] = asset_data['volume'].pct_change().diff()
            
            # 跨时间框架特征
            asset_data['intraday_return'] = asset_data.groupby(asset_data['timestamp'].dt.date)['returns'].cumsum()
            asset_data['time_of_day'] = asset_data['timestamp'].dt.hour * 60 + asset_data['timestamp'].dt.minute
            asset_data['day_of_week'] = asset_data['timestamp'].dt.dayofweek
            
            feature_data.append(asset_data)
        
        # 合并所有资产的特征数据
        features_df = pd.concat(feature_data, ignore_index=True)
        
        # 保存特征数据
        features_file = self.features_dir / "citadel_features.csv"
        features_df.to_csv(features_file, index=False)
        print(f"✅ 特征数据已保存到: {features_file}")
        
        # 特征统计
        feature_cols = [col for col in features_df.columns if col not in ['timestamp', 'asset_id']]
        print(f"\n📊 提取了 {len(feature_cols)} 个特征:")
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:2d}. {col}")
        
        return features_df
    
    def analyze_trading_patterns(self, df):
        """分析交易模式"""
        print("\n📈 交易模式分析:")
        
        # 时间模式
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # 成交量模式
        hourly_volume = df.groupby('hour')['volume'].mean()
        print(f"成交量最高时段: {hourly_volume.idxmax()}:00 (平均成交量: {hourly_volume.max():.0f})")
        print(f"成交量最低时段: {hourly_volume.idxmin()}:00 (平均成交量: {hourly_volume.min():.0f})")
        
        # 波动率模式
        df['returns'] = df.groupby('asset_id')['close'].pct_change()
        hourly_volatility = df.groupby('hour')['returns'].std()
        print(f"波动率最高时段: {hourly_volatility.idxmax()}:00 (标准差: {hourly_volatility.max():.4f})")
        print(f"波动率最低时段: {hourly_volatility.idxmin()}:00 (标准差: {hourly_volatility.min():.4f})")
        
        # 价差模式
        hourly_spread = df.groupby('hour')['spread'].mean()
        print(f"价差最大时段: {hourly_spread.idxmax()}:00 (平均价差: {hourly_spread.max():.4f})")
        print(f"价差最小时段: {hourly_spread.idxmin()}:00 (平均价差: {hourly_spread.min():.4f})")
        
        return {
            'volume_patterns': hourly_volume.to_dict(),
            'volatility_patterns': hourly_volatility.to_dict(),
            'spread_patterns': hourly_spread.to_dict()
        }
    
    def run_strategy_simulation(self, features_df):
        """运行策略模拟"""
        print("\n🚀 运行 Citadel 策略模拟...")
        
        # 选择一个资产进行模拟
        sample_asset = features_df[features_df['asset_id'] == 'ASSET_000'].copy()
        sample_asset = sample_asset.dropna().reset_index(drop=True)
        
        if len(sample_asset) < 100:
            print("⚠️ 数据不足，跳过策略模拟")
            return None
        
        # 准备数据
        price_data = sample_asset[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        price_data.set_index('timestamp', inplace=True)
        
        # 运行 Terminal AI 系统模拟
        try:
            results = run_terminal_ai_simulation(
                price_data=price_data,
                initial_capital=100000,
                lookback_period=60,
                rebalance_freq='5min'
            )
            
            print("✅ 策略模拟完成!")
            print(f"总收益率: {results['total_return']:.2%}")
            print(f"夏普比率: {results['sharpe_ratio']:.3f}")
            print(f"最大回撤: {results['max_drawdown']:.2%}")
            print(f"胜率: {results['win_rate']:.2%}")
            
            # 保存结果
            results_file = self.models_dir / "simulation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            return results
            
        except Exception as e:
            print(f"⚠️ 策略模拟失败: {e}")
            return None
    
    def generate_analysis_report(self, data_stats, trading_patterns, simulation_results=None):
        """生成分析报告"""
        print("\n📋 生成分析报告...")
        
        report = {
            "competition": "Citadel Terminal AI Competition",
            "analysis_date": datetime.now().isoformat(),
            "data_analysis": {
                "data_shape": data_stats['shape'],
                "time_range": [str(data_stats['time_range'][0]), str(data_stats['time_range'][1])],
                "n_assets": data_stats['n_assets'],
                "data_quality": {
                    "missing_values": data_stats['missing_values'],
                    "duplicates": data_stats['duplicates'],
                    "price_issues": data_stats['price_issues']
                }
            },
            "trading_patterns": trading_patterns,
            "key_insights": [
                "高频数据需要重点关注市场微观结构特征",
                "买卖价差和订单簿不平衡是重要的预测因子",
                "时间特征（小时、分钟）对策略表现有显著影响",
                "成交量模式可以帮助识别最佳交易时机",
                "实时风险管理对高频策略至关重要"
            ],
            "recommended_features": [
                "价格动量（多时间框架）",
                "波动率特征（实现波动率、GARCH）",
                "成交量特征（VWAP、成交量比率）",
                "市场微观结构（买卖价差、订单不平衡）",
                "技术指标（RSI、移动平均、布林带）",
                "时间特征（日内时间、星期几）",
                "跨资产特征（相关性、协整）"
            ],
            "strategy_recommendations": [
                "多策略组合：动量 + 均值回归 + 套利",
                "实时参数优化和策略切换",
                "严格的风险控制和仓位管理",
                "低延迟执行和订单优化",
                "机器学习模型的在线学习"
            ]
        }
        
        if simulation_results:
            report["simulation_results"] = simulation_results
        
        # 保存报告
        report_file = Path(__file__).parent / "citadel_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 分析报告已保存到: {report_file}")
        return report

def main():
    """主函数"""
    print("🎯 Citadel Terminal AI Competition 数据分析")
    print("=" * 50)
    
    # 初始化分析器
    analyzer = CitadelDataAnalyzer()
    
    # 生成样本数据
    df = analyzer.generate_sample_data(n_samples=5000, n_assets=10)
    
    # 分析数据结构
    data_stats = analyzer.analyze_data_structure(df)
    
    # 提取特征
    features_df = analyzer.extract_features(df)
    
    # 分析交易模式
    trading_patterns = analyzer.analyze_trading_patterns(df)
    
    # 运行策略模拟
    simulation_results = analyzer.run_strategy_simulation(features_df)
    
    # 生成分析报告
    report = analyzer.generate_analysis_report(data_stats, trading_patterns, simulation_results)
    
    print("\n🎉 Citadel 数据分析完成!")
    print(f"📁 结果保存在: {Path(__file__).parent}")
    print("\n📋 关键发现:")
    for insight in report['key_insights']:
        print(f"  • {insight}")
    
    print("\n🚀 下一步建议:")
    print("  1. 获取真实的高频市场数据")
    print("  2. 实施推荐的特征工程策略")
    print("  3. 开发和测试多策略组合")
    print("  4. 优化实时执行系统")
    print("  5. 进行充分的回测和风险评估")

if __name__ == "__main__":
    main()