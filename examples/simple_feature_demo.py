#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的高级特征工程演示

展示项目中新增的特征工程能力：
1. 高阶统计特征
2. 频域分析特征
3. 非线性特征
4. 特征选择
5. 特征验证
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.ml.advanced_feature_engineering import (
    HighOrderStatisticalFeatures,
    FrequencyDomainFeatures,
    NonLinearFeatures,
    AdvancedFeatureEngineeringPipeline
)
from src.ml.feature_selection import FeatureSelectionPipeline
from src.ml.feature_validation import FeatureValidationPipeline


def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    
    # 生成4年的日度数据
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    n = len(dates)
    
    # 生成价格序列（随机游走）
    returns = np.random.normal(0.0005, 0.02, n)  # 日收益率
    price = 100 * np.exp(np.cumsum(returns))
    
    # 生成成交量（对数正态分布）
    volume = np.random.lognormal(15, 0.5, n)
    
    # 生成波动率（GARCH类似）
    volatility = np.abs(returns) * 100
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': price,
        'returns': returns,
        'volume': volume,
        'volatility': volatility
    })
    
    return df


def demo_statistical_features():
    """演示高阶统计特征"""
    print("\n" + "="*60)
    print("高阶统计特征演示")
    print("="*60)
    
    df = create_sample_data()
    stat_features = HighOrderStatisticalFeatures()
    
    print("原始数据形状:", df.shape)
    
    # 添加高阶矩特征
    df_enhanced = stat_features.add_moment_features(
        df, 
        ['returns', 'volume'], 
        windows=[10, 20]
    )
    
    # 添加熵特征
    df_enhanced = stat_features.add_entropy_features(
        df_enhanced, 
        ['returns', 'volume'], 
        windows=[20]
    )
    
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"新增 {len(new_features)} 个统计特征")
    print("示例特征:", new_features[:5])
    
    return df_enhanced


def demo_frequency_features():
    """演示频域特征"""
    print("\n" + "="*60)
    print("频域分析特征演示")
    print("="*60)
    
    df = create_sample_data()
    freq_features = FrequencyDomainFeatures()
    
    print("原始数据形状:", df.shape)
    
    # 添加FFT特征
    df_enhanced = freq_features.add_fft_features(
        df, 
        ['returns', 'volume'], 
        windows=[50]
    )
    
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"新增 {len(new_features)} 个频域特征")
    print("示例特征:", new_features[:5])
    
    return df_enhanced


def demo_nonlinear_features():
    """演示非线性特征"""
    print("\n" + "="*60)
    print("非线性特征演示")
    print("="*60)
    
    df = create_sample_data()
    nonlinear_features = NonLinearFeatures()
    
    print("原始数据形状:", df.shape)
    
    # 添加流形学习特征
    df_enhanced = nonlinear_features.add_manifold_features(
        df, 
        ['returns', 'volume', 'volatility'], 
        windows=[50]
    )
    
    # 添加聚类特征
    df_enhanced = nonlinear_features.add_clustering_features(
        df_enhanced, 
        ['returns', 'volume', 'volatility'], 
        windows=[50]
    )
    
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"新增 {len(new_features)} 个非线性特征")
    print("示例特征:", new_features[:5])
    
    return df_enhanced


def demo_feature_selection():
    """演示特征选择"""
    print("\n" + "="*60)
    print("特征选择演示")
    print("="*60)
    
    # 创建带有目标变量的数据
    df = create_sample_data()
    
    # 添加一些噪声特征
    for i in range(10):
        df[f'noise_{i}'] = np.random.randn(len(df))
    
    # 创建目标变量（未来收益）
    df['target'] = df['returns'].shift(-1)
    df = df.dropna()
    
    print("原始特征数:", df.shape[1] - 2)  # 减去date和target
    
    # 特征选择
    from src.ml.feature_selection import (
        ImportanceBasedSelection, 
        CorrelationBasedSelection,
        VarianceBasedSelection
    )
    
    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
    X = df[feature_cols]
    y = df['target']
    
    # 使用重要性选择
    importance_selector = ImportanceBasedSelection(method='random_forest', threshold='median')
    X_importance = importance_selector.fit_transform(X, y)
    
    # 使用相关性选择
    correlation_selector = CorrelationBasedSelection(threshold=0.8)
    X_correlation = correlation_selector.fit_transform(X_importance)
    
    # 使用方差选择
    variance_selector = VarianceBasedSelection(threshold=0.01)
    X_final = variance_selector.fit_transform(X_correlation)
    
    print(f"重要性选择后: {X_importance.shape[1]} 个特征")
    print(f"相关性选择后: {X_correlation.shape[1]} 个特征")
    print(f"方差选择后: {X_final.shape[1]} 个特征")
    print(f"最终选择的特征:")
    for feature in X_final.columns[:5]:  # 显示前5个
        print(f"  - {feature}")
    
    return X_final.columns.tolist()


def demo_feature_validation():
    """演示特征验证"""
    print("\n" + "="*60)
    print("特征验证演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    df['target'] = df['returns'].shift(-1)
    df = df.dropna()
    
    # 特征验证
    from src.ml.feature_validation import (
        FeatureStabilityTester,
        InformationLeakageDetector,
        FeatureImportanceAnalyzer
    )
    
    feature_cols = ['returns', 'volume', 'volatility']
    
    # 稳定性测试
    stability_tester = FeatureStabilityTester(time_column='date')
    stability_results = stability_tester.test_stability(df, feature_cols)
    
    # 信息泄露检测
    leakage_detector = InformationLeakageDetector(time_column='date', target_column='target')
    leakage_results = leakage_detector.detect_leakage(df, feature_cols)
    
    # 特征重要性分析
    importance_analyzer = FeatureImportanceAnalyzer()
    X = df[feature_cols]
    y = df['target']
    importance_results = importance_analyzer.analyze_importance(X, y)
    
    print("特征验证结果:")
    print(f"  - 稳定性测试: 完成 {len(feature_cols)} 个特征的稳定性分析")
    print(f"  - 信息泄露检测: 完成 {len(feature_cols)} 个特征的泄露检测")
    print(f"  - 特征重要性: 完成 {len(feature_cols)} 个特征的重要性分析")
    
    # 显示部分结果
    print("\n稳定性得分:")
    for feature in feature_cols:
        if feature in stability_results:
            score = stability_results[feature].get('stability_score', 0)
            print(f"  - {feature}: {score:.3f}")
    
    return {
        'stability': stability_results,
        'leakage': leakage_results,
        'importance': importance_results
    }


def demo_complete_pipeline():
    """演示完整的特征工程管道"""
    print("\n" + "="*60)
    print("完整特征工程管道演示")
    print("="*60)
    
    df = create_sample_data()
    
    print("原始数据形状:", df.shape)
    
    # 使用完整管道
    pipeline = AdvancedFeatureEngineeringPipeline()
    
    df_enhanced = pipeline.transform(
        df,
        feature_types=['stats', 'frequency', 'nonlinear'],
        columns=['returns', 'volume', 'volatility']
    )
    
    print("增强后数据形状:", df_enhanced.shape)
    print(f"新增特征数: {df_enhanced.shape[1] - df.shape[1]}")
    
    # 显示新增特征类型
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    
    stats_features = [f for f in new_features if any(x in f for x in ['moment', 'entropy', 'skewness', 'kurtosis'])]
    freq_features = [f for f in new_features if any(x in f for x in ['fft', 'spectral', 'wavelet'])]
    nonlinear_features = [f for f in new_features if any(x in f for x in ['pca', 'ica', 'cluster'])]
    
    print(f"\n特征类型分布:")
    print(f"  - 统计特征: {len(stats_features)}")
    print(f"  - 频域特征: {len(freq_features)}")
    print(f"  - 非线性特征: {len(nonlinear_features)}")
    
    return df_enhanced


def main():
    """主函数"""
    print("高级特征工程功能演示")
    print("="*60)
    
    try:
        # 1. 高阶统计特征
        demo_statistical_features()
        
        # 2. 频域分析特征
        demo_frequency_features()
        
        # 3. 非线性特征
        demo_nonlinear_features()
        
        # 4. 特征选择
        demo_feature_selection()
        
        # 5. 特征验证
        demo_feature_validation()
        
        # 6. 完整管道
        demo_complete_pipeline()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        
        print("\n总结:")
        print("✅ 高阶统计特征 - 矩、偏度、峰度、熵等")
        print("✅ 频域分析特征 - FFT、频谱能量、主频等")
        print("✅ 非线性特征 - PCA、ICA、聚类等")
        print("✅ 特征选择 - 重要性、相关性、稳定性选择")
        print("✅ 特征验证 - 稳定性、泄露检测、重要性分析")
        print("✅ 完整管道 - 一键式特征工程")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()