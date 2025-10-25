#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征工程可视化演示

展示特征工程前后的效果对比和特征质量分析
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from src.ml.advanced_feature_engineering import AdvancedFeatureEngineeringPipeline
from src.ml.feature_selection import ImportanceBasedSelection

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_sample_data():
    """创建示例数据 - 仅用于测试和演示"""
    np.random.seed(42)
    
    # 生成2年的日度数据 - 模拟数据仅用于演示
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    n = len(dates)
    
    # 生成价格序列（随机游走 + 趋势）
    trend = np.linspace(0, 0.5, n)
    returns = np.random.normal(0.001, 0.02, n) + trend * 0.0001
    price = 100 * np.exp(np.cumsum(returns))
    
    # 生成成交量（对数正态分布 + 周期性）
    volume_base = np.random.lognormal(15, 0.3, n)
    volume_cycle = 0.2 * np.sin(2 * np.pi * np.arange(n) / 20)  # 20天周期
    volume = volume_base * (1 + volume_cycle)
    
    # 生成波动率（GARCH类似）
    volatility = np.abs(returns) * 100
    for i in range(1, n):
        volatility[i] = 0.1 * volatility[i] + 0.9 * volatility[i-1]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': price,
        'returns': returns,
        'volume': volume,
        'volatility': volatility
    })
    
    return df


def visualize_original_vs_engineered():
    """可视化原始特征 vs 工程特征"""
    print("生成特征工程对比图...")
    
    # 创建数据
    df = create_sample_data()
    
    # 应用特征工程
    pipeline = AdvancedFeatureEngineeringPipeline()
    df_enhanced = pipeline.transform(
        df,
        feature_types=['stats', 'frequency'],
        columns=['returns', 'volume', 'volatility']
    )
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('特征工程前后对比', fontsize=16, fontweight='bold')
    
    # 原始特征分布
    axes[0, 0].hist(df['returns'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('原始收益率分布')
    axes[0, 0].set_xlabel('收益率')
    axes[0, 0].set_ylabel('频次')
    
    # 工程特征分布（选择一个统计特征）
    stat_features = [col for col in df_enhanced.columns if 'skewness' in col]
    if stat_features:
        axes[0, 1].hist(df_enhanced[stat_features[0]].dropna(), bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title(f'工程特征分布\n({stat_features[0]})')
        axes[0, 1].set_xlabel('特征值')
        axes[0, 1].set_ylabel('频次')
    
    # 原始特征时间序列
    axes[1, 0].plot(df['date'], df['returns'], alpha=0.7, color='blue')
    axes[1, 0].set_title('原始收益率时间序列')
    axes[1, 0].set_xlabel('日期')
    axes[1, 0].set_ylabel('收益率')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 工程特征时间序列
    if stat_features:
        valid_data = df_enhanced[stat_features[0]].dropna()
        valid_dates = df_enhanced.loc[valid_data.index, 'date']
        axes[1, 1].plot(valid_dates, valid_data, alpha=0.7, color='red')
        axes[1, 1].set_title(f'工程特征时间序列\n({stat_features[0]})')
        axes[1, 1].set_xlabel('日期')
        axes[1, 1].set_ylabel('特征值')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_engineering_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, df_enhanced


def visualize_feature_importance():
    """可视化特征重要性"""
    print("生成特征重要性图...")
    
    # 创建数据
    df = create_sample_data()
    
    # 应用特征工程
    pipeline = AdvancedFeatureEngineeringPipeline()
    df_enhanced = pipeline.transform(
        df,
        feature_types=['stats', 'frequency'],
        columns=['returns', 'volume', 'volatility']
    )
    
    # 创建目标变量
    df_enhanced['target'] = df_enhanced['returns'].shift(-1)
    df_enhanced = df_enhanced.dropna()
    
    # 特征选择
    feature_cols = [col for col in df_enhanced.columns if col not in ['date', 'target']]
    X = df_enhanced[feature_cols]
    y = df_enhanced['target']
    
    # 清理数据：处理无穷大值和NaN值
    print("清理特征工程后的数据...")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # 标准化特征以避免数值过大
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # 计算特征重要性
    selector = ImportanceBasedSelection(
        method='random_forest',
        threshold='median',
        n_estimators=100
    )
    selector.fit(X_scaled, y)
    
    # 获取重要性分数（使用feature_importance_属性）
    importance_df = pd.DataFrame({
        'feature': selector.feature_importance_.index,
        'importance': selector.feature_importance_.values
    }).sort_values('importance', ascending=False)
    
    # 绘制前20个最重要的特征
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    
    bars = plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title('Top 20 特征重要性排名', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # 为不同类型的特征设置不同颜色
    colors = []
    for feature in top_features['feature']:
        if any(x in feature for x in ['moment', 'skewness', 'kurtosis', 'entropy']):
            colors.append('skyblue')  # 统计特征
        elif any(x in feature for x in ['fft', 'spectral', 'dominant']):
            colors.append('lightcoral')  # 频域特征
        else:
            colors.append('lightgreen')  # 原始特征
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='统计特征'),
        Patch(facecolor='lightcoral', label='频域特征'),
        Patch(facecolor='lightgreen', label='原始特征')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('feature_importance_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df


def visualize_feature_correlation():
    """可视化特征相关性"""
    print("生成特征相关性热力图...")
    
    # 创建数据
    df = create_sample_data()
    
    # 应用特征工程
    pipeline = AdvancedFeatureEngineeringPipeline()
    df_enhanced = pipeline.transform(
        df,
        feature_types=['stats'],
        columns=['returns', 'volume', 'volatility']
    )
    
    # 选择一些代表性特征
    original_features = ['returns', 'volume', 'volatility']
    stat_features = [col for col in df_enhanced.columns if any(x in col for x in ['skewness', 'kurtosis', 'moment'])][:6]
    selected_features = original_features + stat_features
    
    # 清理数据：处理无穷大值和NaN值
    df_enhanced[selected_features] = df_enhanced[selected_features].replace([np.inf, -np.inf], np.nan)
    df_enhanced[selected_features] = df_enhanced[selected_features].fillna(df_enhanced[selected_features].median())
    
    # 计算相关性矩阵
    correlation_matrix = df_enhanced[selected_features].corr()
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8})
    
    plt.title('特征相关性热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix


def visualize_pca_analysis():
    """可视化PCA分析"""
    print("生成PCA分析图...")
    
    # 创建数据
    df = create_sample_data()
    
    # 应用特征工程
    pipeline = AdvancedFeatureEngineeringPipeline()
    df_enhanced = pipeline.transform(
        df,
        feature_types=['stats', 'frequency'],
        columns=['returns', 'volume', 'volatility']
    )
    
    # 准备数据
    feature_cols = [col for col in df_enhanced.columns if col not in ['date']]
    X = df_enhanced[feature_cols].dropna()
    
    # 清理数据：处理无穷大值和NaN值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA分析
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 解释方差比例
    cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
    axes[0].plot(range(1, min(21, len(cumsum_ratio)+1)), cumsum_ratio[:20], 'bo-')
    axes[0].axhline(y=0.95, color='r', linestyle='--', label='95%解释方差')
    axes[0].set_xlabel('主成分数量')
    axes[0].set_ylabel('累积解释方差比例')
    axes[0].set_title('PCA累积解释方差')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 前两个主成分的散点图
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c=range(len(X_pca)), cmap='viridis')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 解释方差)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 解释方差)')
    axes[1].set_title('前两个主成分散点图')
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca


def visualize_model_performance():
    """可视化模型性能对比"""
    print("生成模型性能对比图...")
    
    # 创建数据
    df = create_sample_data()
    
    # 原始特征
    original_features = ['returns', 'volume', 'volatility']
    df['target'] = df['returns'].shift(-1)
    df_clean = df.dropna()
    
    X_original = df_clean[original_features]
    y = df_clean['target']
    
    # 工程特征
    pipeline = AdvancedFeatureEngineeringPipeline()
    df_enhanced = pipeline.transform(
        df,
        feature_types=['stats', 'frequency'],
        columns=['returns', 'volume', 'volatility']
    )
    df_enhanced['target'] = df_enhanced['returns'].shift(-1)
    df_enhanced_clean = df_enhanced.dropna()
    
    feature_cols = [col for col in df_enhanced_clean.columns if col not in ['date', 'target']]
    X_enhanced = df_enhanced_clean[feature_cols]
    y_enhanced = df_enhanced_clean['target']
    
    # 清理数据：处理无穷大值和NaN值
    X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)
    X_enhanced = X_enhanced.fillna(X_enhanced.median())
    
    # 模型性能对比
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 原始特征性能
    scores_original = cross_val_score(model, X_original, y, cv=5, scoring='neg_mean_squared_error')
    mse_original = -scores_original.mean()
    
    # 工程特征性能
    scores_enhanced = cross_val_score(model, X_enhanced, y_enhanced, cv=5, scoring='neg_mean_squared_error')
    mse_enhanced = -scores_enhanced.mean()
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    
    categories = ['原始特征', '工程特征']
    mse_values = [mse_original, mse_enhanced]
    improvement = (mse_original - mse_enhanced) / mse_original * 100
    
    bars = plt.bar(categories, mse_values, color=['lightblue', 'lightcoral'], alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, mse in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + mse*0.01, 
                f'{mse:.6f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('均方误差 (MSE)')
    plt.title(f'模型性能对比\n(特征工程改善: {improvement:.1f}%)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加改善百分比注释
    plt.annotate(f'改善 {improvement:.1f}%', 
                xy=(1, mse_enhanced), xytext=(0.5, mse_enhanced * 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                ha='center')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'original_mse': mse_original,
        'enhanced_mse': mse_enhanced,
        'improvement': improvement
    }


def main():
    """主函数"""
    print("高级特征工程可视化演示")
    print("="*60)
    
    try:
        # 1. 特征工程前后对比
        print("\n1. 生成特征工程前后对比图...")
        df_original, df_enhanced = visualize_original_vs_engineered()
        print(f"   原始特征数: {df_original.shape[1]}")
        print(f"   工程后特征数: {df_enhanced.shape[1]}")
        
        # 2. 特征重要性分析
        print("\n2. 生成特征重要性排名图...")
        importance_df = visualize_feature_importance()
        print(f"   分析了 {len(importance_df)} 个特征的重要性")
        
        # 3. 特征相关性分析
        print("\n3. 生成特征相关性热力图...")
        correlation_matrix = visualize_feature_correlation()
        print(f"   分析了 {correlation_matrix.shape[0]} 个特征的相关性")
        
        # 4. PCA分析
        print("\n4. 生成PCA分析图...")
        pca = visualize_pca_analysis()
        print(f"   前5个主成分解释方差: {pca.explained_variance_ratio_[:5].sum():.1%}")
        
        # 5. 模型性能对比
        print("\n5. 生成模型性能对比图...")
        performance = visualize_model_performance()
        print(f"   模型性能改善: {performance['improvement']:.1f}%")
        
        print("\n" + "="*60)
        print("可视化演示完成！")
        print("="*60)
        
        print("\n生成的图表文件:")
        print("📊 feature_engineering_comparison.png - 特征工程前后对比")
        print("📊 feature_importance_ranking.png - 特征重要性排名")
        print("📊 feature_correlation_heatmap.png - 特征相关性热力图")
        print("📊 pca_analysis.png - PCA分析")
        print("📊 model_performance_comparison.png - 模型性能对比")
        
        print(f"\n总结:")
        print(f"✅ 特征数量从 {df_original.shape[1]} 增加到 {df_enhanced.shape[1]}")
        print(f"✅ 模型性能改善 {performance['improvement']:.1f}%")
        print(f"✅ 前5个主成分解释 {pca.explained_variance_ratio_[:5].sum():.1%} 的方差")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()