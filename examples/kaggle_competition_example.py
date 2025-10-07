#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle量化比赛使用示例

演示如何使用项目的机器学习模块参加Kaggle量化比赛
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.ml import (
    KaggleDataProcessor,
    KaggleModelTrainer,
    KaggleSubmissionGenerator,
    AdvancedFeatureEngineer,
    StackingEnsemble,
    TimeSeriesCrossValidator,
    ModelValidator,
    quick_kaggle_pipeline
)


def example_jane_street_style_competition():
    """
    模拟Jane Street Market Prediction风格的比赛
    """
    print("=== Jane Street风格量化比赛示例 ===")
    
    # 1. 生成模拟数据（实际使用时替换为真实数据路径）
    print("生成模拟数据...")
    
    # 模拟特征数据
    n_samples = 10000
    n_features = 130  # Jane Street通常有130个特征
    
    # 生成时间序列特征
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    
    # 添加一些趋势和周期性
    time_trend = np.linspace(0, 10, n_samples)
    for i in range(10):  # 前10个特征添加趋势
        features[:, i] += 0.1 * time_trend + 0.05 * np.sin(time_trend * (i + 1))
    
    # 生成目标变量（收益率）
    # 使用特征的线性组合加上噪声
    weights = np.random.randn(n_features) * 0.01
    target = np.dot(features, weights) + np.random.randn(n_samples) * 0.1
    
    # 创建DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    train_df = pd.DataFrame(features, columns=feature_cols)
    train_df['target'] = target
    train_df['date'] = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    train_df['ts_id'] = range(n_samples)
    
    # 分割训练和测试集
    split_idx = int(0.8 * n_samples)
    train_data = train_df.iloc[:split_idx].copy()
    test_data = train_df.iloc[split_idx:].copy()
    test_data = test_data.drop('target', axis=1)  # 测试集不包含目标
    
    # 保存到临时文件
    train_path = '/tmp/train.csv'
    test_path = '/tmp/test.csv'
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    # 2. 使用快速流水线
    print("\n使用快速Kaggle流水线...")
    
    results = quick_kaggle_pipeline(
        train_path=train_path,
        test_path=test_path,
        target_col='target',
        competition_type='regression'
    )
    
    print("模型训练结果:")
    for model_name, result in results['models'].items():
        train_score = result['train_score']
        val_score = result.get('val_score', 'N/A')
        print(f"  {model_name}: 训练RMSE={train_score:.4f}, 验证RMSE={val_score}")
    
    print(f"\n生成了 {len(results['submission_files'])} 个提交文件")
    
    return results


def example_advanced_feature_engineering():
    """
    高级特征工程示例
    """
    print("\n=== 高级特征工程示例 ===")
    
    # 生成时间序列数据
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # 模拟股票价格数据
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    df = pd.DataFrame({
        'date': dates,
        'price': price,
        'volume': volume,
        'high': price * (1 + np.abs(np.random.randn(n_samples)) * 0.01),
        'low': price * (1 - np.abs(np.random.randn(n_samples)) * 0.01),
        'close': price
    })
    
    # 使用高级特征工程器
    feature_engineer = AdvancedFeatureEngineer()
    
    # 添加技术指标特征
    df_with_features = feature_engineer.add_technical_indicators(df)
    
    # 添加统计特征
    df_with_features = feature_engineer.add_statistical_features(df_with_features)
    
    # 添加时间序列特征
    df_with_features = feature_engineer.add_time_series_features(df_with_features)
    
    print(f"原始特征数: {len(df.columns)}")
    print(f"工程后特征数: {len(df_with_features.columns)}")
    print("新增特征示例:")
    new_features = [col for col in df_with_features.columns if col not in df.columns]
    for feature in new_features[:10]:  # 显示前10个新特征
        print(f"  {feature}")
    
    return df_with_features


def example_time_series_validation():
    """
    时间序列交叉验证示例
    """
    print("\n=== 时间序列交叉验证示例 ===")
    
    # 生成时间序列数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # 添加时间趋势
    time_trend = np.linspace(0, 5, n_samples)
    X[:, 0] += time_trend
    
    # 目标变量有时间依赖性
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1
    
    # 创建时间序列交叉验证器
    cv = TimeSeriesCrossValidator(n_splits=5, gap=10, expanding_window=True)
    
    # 创建模型
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # 使用模型验证器
    validator = ModelValidator(cv, scoring='neg_mean_squared_error', return_train_score=True)
    
    # 比较模型
    comparison_results = validator.compare_models(models, X, y)
    
    print("时间序列交叉验证结果:")
    print(comparison_results)
    
    return comparison_results


def example_ensemble_methods():
    """
    集成方法示例
    """
    print("\n=== 集成方法示例 ===")
    
    # 生成数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + 
         np.random.randn(n_samples) * 0.1)
    
    # 分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建基础模型
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import xgboost as xgb
    import lightgbm as lgb
    
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('ridge', Ridge(alpha=1.0)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    # 创建Stacking集成
    from sklearn.linear_model import LinearRegression
    stacking_ensemble = StackingEnsemble(
        base_models=base_models,
        meta_model=LinearRegression(),
        cv_folds=5
    )
    
    # 训练集成模型
    print("训练Stacking集成模型...")
    stacking_ensemble.fit(X_train, y_train)
    
    # 预测
    y_pred = stacking_ensemble.predict(X_test)
    
    # 评估
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Stacking集成结果:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # 比较单个模型性能
    print("\n单个模型性能:")
    for name, model in base_models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse_single = np.sqrt(mean_squared_error(y_test, pred))
        r2_single = r2_score(y_test, pred)
        print(f"  {name}: RMSE={rmse_single:.4f}, R²={r2_single:.4f}")
    
    return stacking_ensemble


def main():
    """主函数"""
    print("Kaggle量化比赛工具包使用示例")
    print("=" * 50)
    
    try:
        # 1. Jane Street风格比赛示例
        jane_street_results = example_jane_street_style_competition()
        
        # 2. 高级特征工程示例
        engineered_features = example_advanced_feature_engineering()
        
        # 3. 时间序列验证示例
        validation_results = example_time_series_validation()
        
        # 4. 集成方法示例
        ensemble_model = example_ensemble_methods()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("\n项目优势总结:")
        print("✓ 完整的数据预处理流水线")
        print("✓ 高级特征工程工具")
        print("✓ 时间序列交叉验证")
        print("✓ 多种集成学习方法")
        print("✓ 自动化提交文件生成")
        print("✓ 模型性能分析和比较")
        
        print("\n适合的Kaggle比赛类型:")
        print("• 股票价格预测 (如 Jane Street Market Prediction)")
        print("• 量化交易策略 (如 Optiver - Trading at the Close)")
        print("• 金融时间序列预测 (如 Ubiquant Market Prediction)")
        print("• 算法交易竞赛 (如 Two Sigma Financial Modeling)")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()