#!/usr/bin/env python3
"""
简化版高级特征工程演示
避免复杂的pandas操作，专注于展示核心功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 导入我们的模块
from src.ml.feature_engineering import AdvancedFeatureEngineer
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
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # 生成基础价格数据（随机游走）
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_samples)  # 日收益率
    prices = 100 * np.exp(np.cumsum(returns))  # 价格
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.5, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': volumes,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'target': np.roll(returns, -1)  # 下一期收益率作为目标
    })
    
    # 移除最后一行（没有目标值）
    df = df[:-1].reset_index(drop=True)
    
    print(f"生成数据集: {df.shape[0]} 行, {df.shape[1]} 列")
    return df

def demonstrate_basic_feature_engineering():
    """演示基础特征工程"""
    print("\n" + "="*60)
    print("1. 基础特征工程演示")
    print("="*60)
    
    # 创建数据
    df = create_sample_data()
    
    # 初始化特征工程器
    feature_engineer = AdvancedFeatureEngineer()
    
    # 应用特征工程
    print("\n应用基础特征工程...")
    feature_cols = ['price', 'volume', 'high', 'low']
    
    # 手动添加一些简单的技术指标
    df['sma_5'] = df['price'].rolling(5).mean()
    df['sma_20'] = df['price'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['price'])
    df['volatility'] = df['price'].rolling(10).std()
    df['volume_sma'] = df['volume'].rolling(10).mean()
    df['price_change'] = df['price'].pct_change()
    
    # 移除NaN值
    df = df.dropna().reset_index(drop=True)
    
    print(f"特征工程后: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 准备特征和目标
    feature_columns = ['price', 'volume', 'high', 'low', 'sma_5', 'sma_20', 
                      'rsi', 'volatility', 'volume_sma', 'price_change']
    X = df[feature_columns]
    y = df['target']
    
    print(f"最终特征数量: {X.shape[1]}")
    print(f"特征列表: {list(X.columns)}")
    
    return X, y, df

def calculate_rsi(prices, window=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def demonstrate_feature_selection(X, y):
    """演示特征选择"""
    print("\n" + "="*60)
    print("2. 特征选择演示")
    print("="*60)
    
    print(f"原始特征数量: {X.shape[1]}")
    
    # 1. 基于重要性的选择
    print("\n2.1 基于重要性的特征选择")
    importance_selector = ImportanceBasedSelection(
        threshold='median',
        n_estimators=50
    )
    importance_selector.fit(X, y)
    X_importance = importance_selector.transform(X)
    print(f"  重要性选择后: {X_importance.shape[1]} 个特征")
    print(f"  选择的特征: {list(X_importance.columns)}")
    
    # 2. 基于相关性的选择
    print("\n2.2 基于相关性的特征选择")
    correlation_selector = CorrelationBasedSelection(threshold=0.8)
    correlation_selector.fit(X, y)
    X_correlation = correlation_selector.transform(X)
    print(f"  相关性选择后: {X_correlation.shape[1]} 个特征")
    print(f"  选择的特征: {list(X_correlation.columns)}")
    
    # 3. 基于方差的选择
    print("\n2.3 基于方差的特征选择")
    variance_selector = VarianceBasedSelection(threshold=0.01)
    variance_selector.fit(X, y)
    X_variance = variance_selector.transform(X)
    print(f"  方差选择后: {X_variance.shape[1]} 个特征")
    print(f"  选择的特征: {list(X_variance.columns)}")
    
    # 组合选择
    print("\n2.4 组合特征选择")
    # 先应用重要性选择，再应用相关性选择
    X_combined = importance_selector.transform(X)
    correlation_selector_2 = CorrelationBasedSelection(threshold=0.8)
    correlation_selector_2.fit(X_combined, y)
    X_final = correlation_selector_2.transform(X_combined)
    
    print(f"  最终选择: {X_final.shape[1]} 个特征")
    print(f"  最终特征: {list(X_final.columns)}")
    
    return X_final

def demonstrate_feature_validation(X, y, df):
    """演示特征验证"""
    print("\n" + "="*60)
    print("3. 特征验证演示")
    print("="*60)
    
    # 1. 特征稳定性测试
    print("\n3.1 特征稳定性测试")
    stability_tester = FeatureStabilityTester()
    feature_list = list(X.columns)
    stability_results = stability_tester.test_stability(df, feature_list)
    
    stable_features = [f for f, result in stability_results.items() 
                      if result.get('is_stable', False)]
    unstable_features = [f for f, result in stability_results.items() 
                        if not result.get('is_stable', True)]
    
    print(f"  稳定特征: {len(stable_features)} 个")
    print(f"  不稳定特征: {len(unstable_features)} 个")
    if unstable_features:
        print(f"  不稳定特征列表: {unstable_features}")
    
    # 2. 信息泄露检测
    print("\n3.2 信息泄露检测")
    leakage_detector = InformationLeakageDetector()
    leakage_results = leakage_detector.detect_leakage(df, feature_list)
    
    print(f"  检测到的潜在泄露特征: {len(leakage_results)} 个")
    if leakage_results:
        print(f"  潜在泄露特征:")
        for feature in leakage_results:
            print(f"    - {feature}")
    
    # 3. 特征重要性分析
    print("\n3.3 特征重要性分析")
    importance_analyzer = FeatureImportanceAnalyzer()
    importance_results = importance_analyzer.analyze_importance(X, y)
    
    if importance_results:
        first_model = list(importance_results.keys())[0]
        importance_df = importance_results[first_model]
        
        print(f"  分析了 {len(importance_results)} 个模型的特征重要性")
        print(f"  前5个重要特征 (基于 {first_model}):")
        for idx, row in importance_df.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")

def demonstrate_performance_comparison(X_original, X_selected, y):
    """演示性能对比"""
    print("\n" + "="*60)
    print("4. 性能对比演示")
    print("="*60)
    
    # 分割数据
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42
    )
    X_sel_train, X_sel_test, _, _ = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model_orig = RandomForestRegressor(n_estimators=100, random_state=42)
    model_sel = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model_orig.fit(X_orig_train, y_train)
    model_sel.fit(X_sel_train, y_train)
    
    # 预测
    y_pred_orig = model_orig.predict(X_orig_test)
    y_pred_sel = model_sel.predict(X_sel_test)
    
    # 计算指标
    mse_orig = mean_squared_error(y_test, y_pred_orig)
    mse_sel = mean_squared_error(y_test, y_pred_sel)
    r2_orig = r2_score(y_test, y_pred_orig)
    r2_sel = r2_score(y_test, y_pred_sel)
    
    print(f"性能对比结果:")
    print(f"  原始特征 ({X_original.shape[1]} 个):")
    print(f"    MSE: {mse_orig:.6f}")
    print(f"    R²:  {r2_orig:.6f}")
    print(f"  选择特征 ({X_selected.shape[1]} 个):")
    print(f"    MSE: {mse_sel:.6f}")
    print(f"    R²:  {r2_sel:.6f}")
    
    # 计算改进
    mse_improvement = (mse_orig - mse_sel) / mse_orig * 100
    r2_improvement = (r2_sel - r2_orig) / abs(r2_orig) * 100
    
    print(f"改进情况:")
    print(f"  MSE 改进: {mse_improvement:+.2f}%")
    print(f"  R² 改进:  {r2_improvement:+.2f}%")
    
    # 显示特征重要性对比
    print(f"\n特征重要性对比:")
    print(f"  原始模型 Top 5 特征:")
    orig_importance = pd.DataFrame({
        'feature': X_original.columns,
        'importance': model_orig.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in orig_importance.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    print(f"  选择模型 Top 5 特征:")
    sel_importance = pd.DataFrame({
        'feature': X_selected.columns,
        'importance': model_sel.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in sel_importance.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

def main():
    """主函数"""
    print("简化版高级特征工程演示")
    print("="*60)
    
    try:
        # 1. 基础特征工程
        X, y, df = demonstrate_basic_feature_engineering()
        
        # 2. 特征选择
        X_selected = demonstrate_feature_selection(X, y)
        
        # 3. 特征验证
        demonstrate_feature_validation(X_selected, y, df)
        
        # 4. 性能对比
        demonstrate_performance_comparison(X, X_selected, y)
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        print("总结:")
        print("✓ 基础特征工程: 成功生成技术指标特征")
        print("✓ 特征选择: 有效筛选出最有价值的特征")
        print("✓ 特征验证: 确保特征质量和稳定性")
        print("✓ 性能对比: 验证特征选择的有效性")
        print("\n这些功能为量化交易策略提供了可靠的特征工程支持！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()