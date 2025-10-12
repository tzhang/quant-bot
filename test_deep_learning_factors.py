#!/usr/bin/env python3
"""
深度学习因子计算器测试脚本
测试所有深度学习模型和因子计算功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from factors.deep_learning_factors import DeepLearningFactorCalculator

warnings.filterwarnings('ignore')

def create_test_data(n_days=500):
    """创建测试数据"""
    print("创建测试数据...")
    
    # 生成日期序列
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 生成价格数据（模拟股票价格走势）
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # 日收益率
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 创建OHLC数据
    price_data = pd.DataFrame({
        'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'close': prices,
    }, index=dates)
    
    # 生成成交量数据
    volume_data = pd.DataFrame({
        'volume': np.random.lognormal(10, 1, len(dates)),
        'turnover': np.random.uniform(0.01, 0.1, len(dates))
    }, index=dates)
    
    # 生成基本面数据
    fundamental_data = pd.DataFrame({
        'pe_ratio': np.random.uniform(10, 30, len(dates)),
        'pb_ratio': np.random.uniform(1, 5, len(dates)),
        'roe': np.random.uniform(0.05, 0.25, len(dates)),
        'debt_ratio': np.random.uniform(0.2, 0.8, len(dates))
    }, index=dates)
    
    print(f"测试数据创建完成: {len(dates)} 天的数据")
    return price_data, volume_data, fundamental_data

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    try:
        # 创建计算器实例
        calculator = DeepLearningFactorCalculator(sequence_length=30)
        print("✓ 深度学习因子计算器创建成功")
        
        # 创建测试数据
        price_data, volume_data, fundamental_data = create_test_data(200)
        
        # 测试数据准备
        features = calculator._prepare_features(price_data, volume_data, fundamental_data)
        print(f"✓ 特征数据准备成功: {features.shape}")
        
        # 测试序列准备
        target = price_data['close'].pct_change().shift(-1)
        features_with_target = features.copy()
        features_with_target['target'] = target
        
        X, y = calculator.prepare_sequences(features_with_target, 'target')
        print(f"✓ 序列数据准备成功: X.shape={X.shape}, y.shape={y.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def test_individual_models():
    """测试各个模型"""
    print("\n=== 测试各个深度学习模型 ===")
    
    try:
        calculator = DeepLearningFactorCalculator(sequence_length=20)
        price_data, volume_data, fundamental_data = create_test_data(150)
        
        # 准备数据
        features = calculator._prepare_features(price_data, volume_data, fundamental_data)
        target = price_data['close'].pct_change().shift(-1)
        features_with_target = features.copy()
        features_with_target['target'] = target
        X, y = calculator.prepare_sequences(features_with_target, 'target')
        
        if len(X) == 0:
            print("✗ 序列数据为空，跳过模型测试")
            return False
        
        models_to_test = [
            ('LSTM', calculator.train_lstm_model),
            ('GRU', calculator.train_gru_model),
            ('Transformer', calculator.train_transformer_model),
            ('CNN', calculator.train_cnn_model),
            ('VAE', lambda x, y, name: calculator.train_vae_model(x, name)),
            ('Attention', calculator.train_attention_model),
            ('ResNet', calculator.train_resnet_model)
        ]
        
        successful_models = 0
        
        for model_name, train_func in models_to_test:
            try:
                if model_name == 'VAE':
                    result = train_func(X, y, f'{model_name.lower()}_test')
                else:
                    result = train_func(X, y, f'{model_name.lower()}_test')
                
                if result:
                    print(f"✓ {model_name} 模型训练成功")
                    successful_models += 1
                else:
                    print(f"✗ {model_name} 模型训练失败（返回空结果）")
                    
            except Exception as e:
                print(f"✗ {model_name} 模型训练失败: {e}")
        
        # 测试自编码器（无监督）
        try:
            autoencoder_result = calculator.train_autoencoder_model(X, 'autoencoder_test')
            if autoencoder_result:
                print("✓ Autoencoder 模型训练成功")
                successful_models += 1
            else:
                print("✗ Autoencoder 模型训练失败（返回空结果）")
        except Exception as e:
            print(f"✗ Autoencoder 模型训练失败: {e}")
        
        print(f"\n模型测试总结: {successful_models}/{len(models_to_test)+1} 个模型成功")
        return successful_models > 0
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def test_factor_calculation():
    """测试因子计算"""
    print("\n=== 测试因子计算 ===")
    
    try:
        calculator = DeepLearningFactorCalculator(sequence_length=30)
        price_data, volume_data, fundamental_data = create_test_data(300)
        
        # 测试基础深度学习因子
        print("测试基础深度学习因子...")
        dl_factors = calculator.calculate_deep_learning_factors(
            price_data, volume_data, fundamental_data
        )
        print(f"✓ 基础深度学习因子计算完成: {len(dl_factors)} 个因子")
        
        # 测试强化学习因子
        print("测试强化学习因子...")
        rl_factors = calculator.calculate_reinforcement_learning_factors(
            price_data, volume_data
        )
        print(f"✓ 强化学习因子计算完成: {len(rl_factors)} 个因子")
        
        # 测试集成学习因子
        print("测试集成学习因子...")
        ensemble_factors = calculator.calculate_ensemble_factors(
            price_data, volume_data
        )
        print(f"✓ 集成学习因子计算完成: {len(ensemble_factors)} 个因子")
        
        # 测试元学习因子
        print("测试元学习因子...")
        meta_factors = calculator.calculate_meta_learning_factors(price_data)
        print(f"✓ 元学习因子计算完成: {len(meta_factors)} 个因子")
        
        # 测试所有因子计算
        print("测试所有因子计算...")
        all_factors = calculator.calculate_all_deep_learning_factors(
            price_data, volume_data, fundamental_data
        )
        print(f"✓ 所有深度学习因子计算完成: {len(all_factors)} 个因子")
        
        # 显示因子统计信息
        if all_factors:
            print("\n因子统计信息:")
            for name, factor in list(all_factors.items())[:10]:  # 显示前10个因子
                valid_count = factor.notna().sum()
                latest_value = factor.dropna().iloc[-1] if not factor.dropna().empty else "N/A"
                print(f"  {name}: 有效值 {valid_count}/{len(factor)}, 最新值 {latest_value}")
        
        return len(all_factors) > 0
        
    except Exception as e:
        print(f"✗ 因子计算测试失败: {e}")
        return False

def test_report_generation():
    """测试报告生成"""
    print("\n=== 测试报告生成 ===")
    
    try:
        calculator = DeepLearningFactorCalculator(sequence_length=20)
        price_data, volume_data, fundamental_data = create_test_data(200)
        
        # 计算因子
        factors = calculator.calculate_all_deep_learning_factors(
            price_data, volume_data, fundamental_data
        )
        
        if not factors:
            print("✗ 无因子数据，跳过报告生成测试")
            return False
        
        # 生成报告
        report = calculator.generate_factor_report(factors)
        print("✓ 因子报告生成成功")
        print("\n报告内容预览:")
        print("=" * 50)
        print(report[:500] + "..." if len(report) > 500 else report)
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"✗ 报告生成测试失败: {e}")
        return False

def run_performance_test():
    """运行性能测试"""
    print("\n=== 性能测试 ===")
    
    try:
        import time
        
        calculator = DeepLearningFactorCalculator(sequence_length=20)
        price_data, volume_data, fundamental_data = create_test_data(100)  # 较小数据集
        
        start_time = time.time()
        factors = calculator.calculate_all_deep_learning_factors(
            price_data, volume_data, fundamental_data
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"✓ 性能测试完成")
        print(f"  执行时间: {execution_time:.2f} 秒")
        print(f"  生成因子数: {len(factors)}")
        print(f"  数据点数: {len(price_data)}")
        print(f"  平均每个因子耗时: {execution_time/max(len(factors), 1):.3f} 秒")
        
        return True
        
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("深度学习因子计算器测试开始")
    print("=" * 60)
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("基本功能测试", test_basic_functionality),
        ("模型训练测试", test_individual_models),
        ("因子计算测试", test_factor_calculation),
        ("报告生成测试", test_report_generation),
        ("性能测试", run_performance_test)
    ]
    
    for test_name, test_func in tests:
        print(f"\n开始 {test_name}...")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 执行异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！深度学习因子计算器功能正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)