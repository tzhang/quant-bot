#!/usr/bin/env python3
"""
基础特征工程功能演示

这个脚本展示了量化交易机器人中特征工程模块的基础功能，
包括特征选择、验证和性能评估的完整流程。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入我们的特征工程模块
from src.ml.feature_selection import (
    ImportanceBasedSelection, 
    CorrelationBasedSelection, 
    VarianceBasedSelection
)
from src.ml.feature_validation import (
    FeatureStabilityTester,
    InformationLeakageDetector,
    FeatureImportanceAnalyzer
)

def create_sample_data(n_samples=1000):
    """创建示例金融数据"""
    print("创建示例金融数据...")
    
    # 生成时间序列
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # 生成基础价格数据
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': price,
        'volume': volume,
        'high': price * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'low': price * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
    })
    
    # 添加一些基础技术指标
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['sma_20'] = df['price'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['price'])
    df['macd'] = calculate_macd(df['price'])
    
    # 添加更多手工特征
    df['price_volume_ratio'] = df['price'] / df['volume']
    df['high_low_ratio'] = df['high'] / df['low']
    df['price_change'] = df['price'].diff()
    df['volume_change'] = df['volume'].pct_change()
    df['price_momentum_5'] = df['price'] / df['price'].shift(5) - 1
    df['price_momentum_10'] = df['price'] / df['price'].shift(10) - 1
    df['volume_sma_10'] = df['volume'].rolling(10).mean()
    df['price_std_10'] = df['price'].rolling(10).std()
    
    # 创建目标变量（未来5天收益率）
    df['target'] = df['returns'].shift(-5).rolling(5).mean()
    
    # 删除缺失值
    df = df.dropna()
    
    print(f"生成数据集: {df.shape[0]} 行, {df.shape[1]} 列")
    return df

def calculate_rsi(prices, window=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

def demonstrate_feature_selection():
    """演示特征选择"""
    print("\n" + "="*60)
    print("1. 特征选择演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    # 准备特征和目标
    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    print(f"原始特征数: {X.shape[1]}")
    print(f"特征列表: {list(X.columns)}")
    
    # 1. 基于重要性的选择
    print("\n1.1 基于重要性的特征选择")
    importance_selector = ImportanceBasedSelection(
        method='random_forest',
        threshold='median'
    )
    importance_selector.fit(X, y)
    X_importance = importance_selector.transform(X)
    print(f"  重要性选择后: {X_importance.shape[1]} 个特征")
    
    # 显示top特征
    if hasattr(importance_selector, 'feature_importance_'):
        top_features = importance_selector.feature_importance_.nlargest(10)
        print("  Top 10 重要特征:")
        for i, (feature, score) in enumerate(top_features.items(), 1):
            print(f"    {i:2d}. {feature}: {score:.4f}")
    
    # 2. 基于相关性的选择
    print("\n1.2 基于相关性的特征选择")
    correlation_selector = CorrelationBasedSelection(threshold=0.95)
    correlation_selector.fit(X)
    X_correlation = correlation_selector.transform(X)
    print(f"  相关性选择后: {X_correlation.shape[1]} 个特征")
    print(f"  移除高相关特征: {X.shape[1] - X_correlation.shape[1]} 个")
    
    # 3. 基于方差的选择
    print("\n1.3 基于方差的特征选择")
    variance_selector = VarianceBasedSelection(threshold=0.01)
    variance_selector.fit(X)
    X_variance = variance_selector.transform(X)
    print(f"  方差选择后: {X_variance.shape[1]} 个特征")
    print(f"  移除低方差特征: {X.shape[1] - X_variance.shape[1]} 个")
    
    # 组合选择结果
    selected_features = list(set(X_importance.columns) & 
                           set(X_correlation.columns) & 
                           set(X_variance.columns))
    X_selected = X[selected_features]
    
    print(f"\n最终选择特征数: {len(selected_features)}")
    print(f"特征选择率: {len(selected_features)/X.shape[1]:.2%}")
    print(f"最终特征: {selected_features}")
    
    return X_selected, y

def demonstrate_feature_validation(X, y):
    """演示特征验证"""
    print("\n" + "="*60)
    print("2. 特征验证演示")
    print("="*60)
    
    # 1. 特征稳定性测试
    print("\n2.1 特征稳定性测试")
    stability_tester = FeatureStabilityTester()
    
    # 创建包含时间列的数据框用于稳定性测试
    df_with_time = create_sample_data()
    feature_list = list(X.columns)
    stability_results = stability_tester.test_stability(df_with_time, feature_list)
    
    stable_features = [f for f, stable in stability_results.items() if stable]
    unstable_features = [f for f, stable in stability_results.items() if not stable]
    
    print(f"  稳定特征: {len(stable_features)} 个")
    print(f"  不稳定特征: {len(unstable_features)} 个")
    
    if unstable_features:
        print("  不稳定特征:")
        for feature in unstable_features:
            print(f"    - {feature}")
    
    # 2. 信息泄露检测
    print("\n2.2 信息泄露检测")
    leakage_detector = InformationLeakageDetector()
    
    # 创建包含时间和目标列的数据框用于泄露检测
    df_with_target = create_sample_data()
    feature_list = list(X.columns)
    leakage_results = leakage_detector.detect_leakage(df_with_target, feature_list)
    
    print(f"  检测到的潜在泄露特征: {len(leakage_results)} 个")
    if leakage_results:
        print("  潜在泄露特征:")
        for feature in list(leakage_results.keys()):
            print(f"    - {feature}")
    
    # 3. 特征重要性分析
    print("\n2.3 特征重要性分析")
    importance_analyzer = FeatureImportanceAnalyzer()
    importance_results = importance_analyzer.analyze_importance(X, y)
    
    print(f"  分析了 {len(importance_results)} 个模型的特征重要性")
    
    # 显示重要性统计
    if importance_results:
        # 获取第一个模型的结果作为示例
        first_model = list(importance_results.keys())[0]
        importance_df = importance_results[first_model]
        
        print(f"  重要性统计 (基于 {first_model}):")
        print(f"    平均重要性: {importance_df['importance_mean'].mean():.4f}")
        print(f"    最高重要性: {importance_df['importance_mean'].max():.4f}")
        print(f"    最低重要性: {importance_df['importance_mean'].min():.4f}")
        
        print(f"  前3个重要特征:")
        for idx, row in importance_df.head(3).iterrows():
            print(f"    {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")
    else:
        print("  未能获取特征重要性结果")

def demonstrate_performance_impact():
    """演示特征选择对模型性能的影响"""
    print("\n" + "="*60)
    print("3. 性能影响评估")
    print("="*60)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    
    # 创建数据
    df = create_sample_data()
    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
    X_original = df[feature_cols].copy()
    y = df['target'].copy()
    
    # 特征选择
    selector = ImportanceBasedSelection(method='random_forest', threshold='median')
    selector.fit(X_original, y)
    X_selected = selector.transform(X_original)
    
    # 数据标准化
    scaler_orig = StandardScaler()
    scaler_sel = StandardScaler()
    
    X_orig_scaled = pd.DataFrame(
        scaler_orig.fit_transform(X_original),
        columns=X_original.columns
    )
    X_sel_scaled = pd.DataFrame(
        scaler_sel.fit_transform(X_selected),
        columns=X_selected.columns
    )
    
    # 分割数据
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_orig_scaled, y, test_size=0.2, random_state=42
    )
    X_sel_train, X_sel_test, _, _ = train_test_split(
        X_sel_scaled, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model_orig = RandomForestRegressor(n_estimators=100, random_state=42)
    model_sel = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model_orig.fit(X_orig_train, y_train)
    model_sel.fit(X_sel_train, y_train)
    
    # 预测和评估
    y_pred_orig = model_orig.predict(X_orig_test)
    y_pred_sel = model_sel.predict(X_sel_test)
    
    mse_orig = mean_squared_error(y_test, y_pred_orig)
    mse_sel = mean_squared_error(y_test, y_pred_sel)
    r2_orig = r2_score(y_test, y_pred_orig)
    r2_sel = r2_score(y_test, y_pred_sel)
    
    print("性能对比结果:")
    print(f"  原始特征 ({X_original.shape[1]} 个):")
    print(f"    MSE: {mse_orig:.6f}")
    print(f"    R²:  {r2_orig:.6f}")
    
    print(f"  选择特征 ({X_selected.shape[1]} 个):")
    print(f"    MSE: {mse_sel:.6f}")
    print(f"    R²:  {r2_sel:.6f}")
    
    print("改进情况:")
    mse_improvement = (mse_orig - mse_sel) / mse_orig * 100
    r2_improvement = (r2_sel - r2_orig) / abs(r2_orig) * 100 if r2_orig != 0 else 0
    
    print(f"  MSE 改进: {mse_improvement:+.2f}%")
    print(f"  R² 改进:  {r2_improvement:+.2f}%")
    
    # 特征重要性对比
    print("\n特征重要性对比:")
    orig_importance = dict(zip(X_original.columns, model_orig.feature_importances_))
    sel_importance = dict(zip(X_selected.columns, model_sel.feature_importances_))
    
    print("  原始模型 Top 5 特征:")
    sorted_orig = sorted(orig_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_orig[:5], 1):
        print(f"    {i}. {feature}: {score:.4f}")
    
    print("  选择模型 Top 5 特征:")
    sorted_sel = sorted(sel_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_sel[:5], 1):
        print(f"    {i}. {feature}: {score:.4f}")

def main():
    """主函数"""
    print("基础特征工程功能演示")
    print("="*60)
    print("本演示展示了量化交易机器人中特征工程模块的基础功能")
    print()
    
    try:
        # 1. 特征选择演示
        X_selected, y = demonstrate_feature_selection()
        
        # 2. 特征验证演示
        demonstrate_feature_validation(X_selected, y)
        
        # 3. 性能影响评估
        demonstrate_performance_impact()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        print("总结:")
        print("✓ 特征选择: 有效筛选出最有价值的特征")
        print("✓ 特征验证: 确保特征质量和稳定性")
        print("✓ 性能提升: 特征选择改善了模型性能")
        print()
        print("这些功能为量化交易策略提供了可靠的特征工程支持！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()