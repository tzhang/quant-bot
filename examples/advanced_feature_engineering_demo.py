#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征工程演示

展示量化交易系统中的高级特征工程能力：
1. 高阶统计特征
2. 频域分析特征
3. 图论网络特征
4. 宏观经济特征
5. 非线性特征
6. 特征选择
7. 特征验证

作者: Quant Team
日期: 2024-01-20
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入我们的模块
from src.ml.advanced_feature_engineering import (
    HighOrderStatisticalFeatures,
    FrequencyDomainFeatures,
    NetworkGraphFeatures,
    MacroEconomicFeatures,
    NonLinearFeatures,
    AdvancedFeatureEngineeringPipeline
)
from src.ml.feature_selection import FeatureSelectionPipeline
from src.ml.feature_validation import FeatureValidationPipeline


def create_sample_data():
    """创建示例数据 - 仅用于测试和演示"""
    print("创建示例股票数据...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 时间序列
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    n_samples = len(dates)
    
    # 模拟股票价格数据 - 仅用于演示
    price_base = 100
    returns = np.random.randn(n_samples) * 0.02  # 2%日波动率
    returns[0] = 0
    
    # 添加一些趋势和周期性
    trend = np.linspace(0, 0.5, n_samples)  # 长期上涨趋势
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n_samples) / 252)  # 年度季节性
    
    returns += trend / n_samples + seasonal / n_samples
    
    # 计算价格
    prices = price_base * np.exp(np.cumsum(returns))
    
    # 计算其他基础指标
    volume = np.random.lognormal(10, 0.5, n_samples)  # 成交量
    high = prices * (1 + np.abs(np.random.randn(n_samples)) * 0.01)
    low = prices * (1 - np.abs(np.random.randn(n_samples)) * 0.01)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
        'returns': returns
    })
    
    # 添加一些基础技术指标
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # 创建目标变量（未来5日收益率）
    df['target'] = df['returns'].rolling(5).sum().shift(-5)
    
    return df.dropna()


def demo_high_order_statistical_features():
    """演示高阶统计特征"""
    print("\n" + "="*60)
    print("高阶统计特征演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    # 初始化特征工程器
    stat_features = HighOrderStatisticalFeatures()
    
    # 添加高阶统计特征
    print("添加高阶统计特征...")
    
    # 矩特征
    df = stat_features.add_moment_features(df, ['returns', 'volume'], windows=[5, 10, 20])
    
    # 分布检验特征
    df = stat_features.add_distribution_tests(df, ['returns'], windows=[20, 60])
    
    # 熵特征
    df = stat_features.add_entropy_features(df, ['returns', 'volume'], windows=[10, 20])
    
    # 显示新增特征
    new_features = [col for col in df.columns if any(x in col for x in ['moment', 'normality', 'entropy'])]
    print(f"新增 {len(new_features)} 个高阶统计特征:")
    for feature in new_features[:10]:  # 显示前10个
        print(f"  - {feature}")
    
    if len(new_features) > 10:
        print(f"  ... 还有 {len(new_features) - 10} 个特征")
    
    # 特征统计
    print(f"\n特征统计:")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
    
    return df


def demo_frequency_domain_features():
    """演示频域分析特征"""
    print("\n" + "="*60)
    print("频域分析特征演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    # 初始化特征工程器
    freq_features = FrequencyDomainFeatures()
    
    print("添加频域分析特征...")
    
    # FFT特征
    df = freq_features.add_fft_features(df, ['returns', 'volume'], windows=[60])
    
    # 小波特征
    df = freq_features.add_wavelet_features(df, ['returns', 'volume'], windows=[60])
    
    # 显示新增特征
    new_features = [col for col in df.columns if any(x in col for x in ['fft', 'wavelet'])]
    print(f"新增 {len(new_features)} 个频域特征:")
    for feature in new_features[:10]:
        print(f"  - {feature}")
    
    if len(new_features) > 10:
        print(f"  ... 还有 {len(new_features) - 10} 个特征")
    
    return df


def demo_graph_network_features():
    """演示图论网络特征"""
    print("\n" + "="*60)
    print("图论网络特征演示")
    print("="*60)
    
    # 创建多资产数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_assets = 5
    
    # 创建相关的收益率数据
    correlation_matrix = np.array([
        [1.0, 0.7, 0.5, 0.3, 0.2],
        [0.7, 1.0, 0.6, 0.4, 0.3],
        [0.5, 0.6, 1.0, 0.5, 0.4],
        [0.3, 0.4, 0.5, 1.0, 0.6],
        [0.2, 0.3, 0.4, 0.6, 1.0]
    ])
    
    returns_data = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=correlation_matrix * 0.02**2,
        size=len(dates)
    )
    
    # 创建DataFrame
    asset_names = [f'asset_{i}' for i in range(n_assets)]
    df = pd.DataFrame(returns_data, columns=asset_names)
    df['date'] = dates
    
    # 初始化特征工程器
    network_features = NetworkGraphFeatures()
    
    print("添加图论网络特征...")
    
    # 相关性网络特征
    df = network_features.create_correlation_network_features(df, asset_names, windows=[60])
    
    # 显示新增特征
    new_features = [col for col in df.columns if 'network' in col]
    print(f"新增 {len(new_features)} 个网络特征:")
    for feature in new_features:
        print(f"  - {feature}")
    
    return df


def demo_macro_economic_features():
    """演示宏观经济特征"""
    print("\n" + "="*60)
    print("宏观经济特征演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    # 初始化特征工程器
    macro_features = MacroEconomicFeatures()
    
    print("添加宏观经济特征...")
    print("注意: 这需要网络连接来获取实时数据")
    
    try:
        # 添加宏观经济特征
        df = macro_features.add_macro_features(df, lookback_days=252)
        
        # 显示新增特征
        new_features = [col for col in df.columns if any(x in col for x in ['VIX', 'DXY', 'TNX', 'GLD', 'USO'])]
        print(f"新增 {len(new_features)} 个宏观经济特征:")
        for feature in new_features[:15]:
            print(f"  - {feature}")
        
        if len(new_features) > 15:
            print(f"  ... 还有 {len(new_features) - 15} 个特征")
            
    except Exception as e:
        print(f"获取宏观经济数据失败: {e}")
        print("这通常是由于网络连接问题或API限制")
    
    return df


def demo_nonlinear_features():
    """演示非线性特征"""
    print("\n" + "="*60)
    print("非线性特征演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    # 初始化特征工程器
    nonlinear_features = NonLinearFeatures()
    
    print("添加非线性特征...")
    
    # 流形学习特征
    df = nonlinear_features.add_manifold_features(
        df, 
        ['returns', 'volume', 'volatility'], 
        windows=[50]
    )
    
    # 聚类特征
    df = nonlinear_features.add_clustering_features(
        df, 
        ['returns', 'volume', 'volatility'], 
        windows=[50],
        n_clusters=3
    )
    
    # 显示新增特征
    new_features = [col for col in df.columns if any(x in col for x in ['pca_', 'ica_', 'cluster_'])]
    print(f"新增 {len(new_features)} 个非线性特征:")
    for feature in new_features[:10]:  # 只显示前10个
        print(f"  - {feature}")
    if len(new_features) > 10:
        print(f"  ... 还有 {len(new_features) - 10} 个特征")
    
    print(f"\n特征统计:")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
    
    return df


def demo_feature_selection():
    """演示特征选择"""
    print("\n" + "="*60)
    print("特征选择演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    # 添加一些噪声特征
    np.random.seed(42)
    for i in range(10):
        df[f'noise_{i}'] = np.random.randn(len(df))
    
    # 准备特征和目标
    feature_columns = [col for col in df.columns if col not in ['date', 'target']]
    X = df[feature_columns].fillna(df[feature_columns].mean())
    y = df['target'].fillna(df['target'].mean())
    
    print(f"原始特征数量: {len(feature_columns)}")
    
    # 初始化特征选择器
    selector = FeatureSelectionPipeline()
    
    # 执行特征选择
    print("执行特征选择...")
    selected_features = selector.select_features(X, y, n_features=10)
    
    # 显示结果
    print(f"选择的特征数量: {len(selected_features)}")
    print("选择的特征:")
    for feature in selected_features:
        print(f"  - {feature}")
    
    # 获取选择摘要
    summary = selector.get_selection_summary()
    print(f"\n特征选择摘要:")
    print(summary.head(10))
    
    return selected_features


def demo_feature_validation():
    """演示特征验证"""
    print("\n" + "="*60)
    print("特征验证演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    # 添加一些问题特征
    np.random.seed(42)
    
    # 泄露特征（使用未来信息）
    df['leaky_feature'] = df['target'].shift(-1)
    
    # 不稳定特征
    unstable_feature = np.random.randn(len(df))
    unstable_feature[len(df)//2:] += 5  # 后半段均值跳跃
    df['unstable_feature'] = unstable_feature
    
    # 准备验证特征
    validation_features = ['returns', 'volume', 'volatility', 'leaky_feature', 'unstable_feature']
    
    print(f"验证 {len(validation_features)} 个特征...")
    
    # 初始化验证器
    validator = FeatureValidationPipeline()
    
    # 执行验证
    results = validator.validate_features(df, validation_features)
    
    # 显示综合报告
    print("\n综合验证报告:")
    comprehensive_report = results['comprehensive']
    print(comprehensive_report[['feature', 'quality_score', 'recommendation']].to_string(index=False))
    
    # 显示推荐分类
    print("\n特征推荐分类:")
    recommendations = validator.get_feature_recommendations()
    for category, features in recommendations.items():
        if features:
            print(f"{category.upper()}: {features}")
    
    return results


def demo_complete_pipeline():
    """演示完整的高级特征工程管道"""
    print("\n" + "="*60)
    print("完整高级特征工程管道演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    print(f"原始数据形状: {df.shape}")
    
    # 初始化完整管道
    pipeline = AdvancedFeatureEngineeringPipeline()
    
    # 配置管道
    config = {
        'high_order_stats': {
            'features': ['returns', 'volume'],
            'windows': [5, 10, 20]
        },
        'frequency_domain': {
            'features': ['returns'],
            'window': 60
        },
        'nonlinear': {
            'features': ['returns', 'volume', 'volatility'],
            'manifold_components': 3,
            'n_clusters': 5
        }
    }
    
    print("执行完整特征工程管道...")
    
    try:
        # 执行管道
        df_enhanced = pipeline.transform(df, config)
        
        print(f"增强后数据形状: {df_enhanced.shape}")
        print(f"新增特征数量: {df_enhanced.shape[1] - df.shape[1]}")
        
        # 显示一些新特征
        new_features = [col for col in df_enhanced.columns if col not in df.columns]
        print(f"\n新增特征示例 (前20个):")
        for feature in new_features[:20]:
            print(f"  - {feature}")
        
        if len(new_features) > 20:
            print(f"  ... 还有 {len(new_features) - 20} 个特征")
        
        return df_enhanced
        
    except Exception as e:
        print(f"管道执行失败: {e}")
        return df


def create_feature_importance_plot(results):
    """创建特征重要性图表"""
    if 'importance' not in results or results['importance'].empty:
        print("没有重要性数据可供绘图")
        return
    
    plt.figure(figsize=(12, 8))
    
    importance_df = results['importance'].head(15)  # 取前15个特征
    
    plt.barh(range(len(importance_df)), importance_df['mean_importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('特征重要性')
    plt.title('特征重要性排名 (Top 15)')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def create_feature_quality_plot(results):
    """创建特征质量图表"""
    if 'comprehensive' not in results or results['comprehensive'].empty:
        print("没有综合质量数据可供绘图")
        return
    
    plt.figure(figsize=(14, 10))
    
    # 子图1: 质量评分分布
    plt.subplot(2, 2, 1)
    plt.hist(results['comprehensive']['quality_score'], bins=20, alpha=0.7)
    plt.xlabel('质量评分')
    plt.ylabel('特征数量')
    plt.title('特征质量评分分布')
    
    # 子图2: 稳定性 vs 重要性
    plt.subplot(2, 2, 2)
    comp_df = results['comprehensive']
    plt.scatter(comp_df['stability_score'], comp_df['mean_importance'], alpha=0.6)
    plt.xlabel('稳定性评分')
    plt.ylabel('重要性')
    plt.title('稳定性 vs 重要性')
    
    # 子图3: 推荐分布
    plt.subplot(2, 2, 3)
    recommendations = results['comprehensive']['recommendation'].value_counts()
    plt.pie(recommendations.values, labels=recommendations.index, autopct='%1.1f%%')
    plt.title('特征推荐分布')
    
    # 子图4: 质量评分 vs 泄露风险
    plt.subplot(2, 2, 4)
    if 'risk_score' in comp_df.columns:
        plt.scatter(comp_df['quality_score'], comp_df['risk_score'], alpha=0.6)
        plt.xlabel('质量评分')
        plt.ylabel('泄露风险评分')
        plt.title('质量评分 vs 泄露风险')
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("高级特征工程演示")
    print("="*60)
    
    # 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        # 1. 高阶统计特征演示
        df_stats = demo_high_order_statistical_features()
        
        # 2. 频域分析特征演示
        df_freq = demo_frequency_domain_features()
        
        # 3. 图论网络特征演示
        df_graph = demo_graph_network_features()
        
        # 4. 宏观经济特征演示
        df_macro = demo_macro_economic_features()
        
        # 5. 非线性特征演示
        df_nonlinear = demo_nonlinear_features()
        
        # 6. 特征选择演示
        selected_features = demo_feature_selection()
        
        # 7. 特征验证演示
        validation_results = demo_feature_validation()
        
        # 8. 完整管道演示
        df_complete = demo_complete_pipeline()
        
        # 创建可视化图表
        print("\n" + "="*60)
        print("创建可视化图表")
        print("="*60)
        
        try:
            create_feature_importance_plot(validation_results)
            create_feature_quality_plot(validation_results)
        except Exception as e:
            print(f"图表创建失败: {e}")
        
        print("\n" + "="*60)
        print("演示完成!")
        print("="*60)
        
        print(f"\n总结:")
        print(f"- 高阶统计特征: 提供更深层的数据分布信息")
        print(f"- 频域分析特征: 捕捉时间序列的周期性和频率特性")
        print(f"- 图论网络特征: 分析资产间的关联性和网络结构")
        print(f"- 宏观经济特征: 整合宏观经济环境信息")
        print(f"- 非线性特征: 通过降维和聚类发现隐藏模式")
        print(f"- 特征选择: 自动筛选最有价值的特征")
        print(f"- 特征验证: 确保特征质量和防止信息泄露")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()