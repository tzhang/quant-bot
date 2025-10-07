#!/usr/bin/env python3
"""
顶级量化比赛使用示例
展示如何使用本项目参加四个顶级量化比赛:
1. Jane Street Market Prediction
2. Optiver - Trading at the Close
3. Citadel Terminal AI Competition
4. CME Group Crypto Classic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from src.data.data_loader import DataLoader
from src.features.factor_calculator import FactorCalculator
from src.ml.feature_engineering import AdvancedFeatureEngineer
from src.ml.model_ensemble import ModelEnsemble
from src.ml.model_validation import TimeSeriesValidator
from src.ml.kaggle_tools import KaggleCompetitionTools
from src.ml.crypto_tools import create_crypto_pipeline, simulate_crypto_trading
from src.ml.terminal_ai_tools import create_terminal_ai_system, run_terminal_ai_simulation

def jane_street_example():
    """Jane Street Market Prediction 比赛示例"""
    print("=" * 60)
    print("Jane Street Market Prediction 比赛示例")
    print("=" * 60)
    
    # 1. 数据准备
    print("1. 准备Jane Street风格的数据...")
    
    # 模拟Jane Street数据格式
    np.random.seed(42)
    n_samples = 10000
    n_features = 130  # Jane Street有130个匿名特征
    
    # 生成模拟数据
    data = pd.DataFrame()
    data['date'] = pd.date_range(start='2020-01-01', periods=n_samples, freq='1min')
    
    # 生成130个匿名特征
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_samples) * np.random.uniform(0.5, 2.0)
    
    # 生成权重和响应变量
    data['weight'] = np.random.uniform(0.1, 2.0, n_samples)
    data['resp'] = np.random.randn(n_samples) * 0.01  # 收益率
    
    # 添加一些真实的市场特征
    data['ts_id'] = np.random.randint(0, 500, n_samples)  # 时间序列ID
    
    print(f"数据形状: {data.shape}")
    print(f"特征列数: {n_features}")
    
    # 2. 特征工程
    print("\n2. 进行高级特征工程...")
    
    feature_engineer = AdvancedFeatureEngineer()
    
    # 添加时间特征
    data = feature_engineer.add_time_features(data, 'date')
    
    # 添加滞后特征
    feature_cols = [f'feature_{i}' for i in range(10)]  # 使用前10个特征作为示例
    data = feature_engineer.add_lag_features(data, feature_cols, [1, 2, 3])
    
    # 添加滚动统计特征
    data = feature_engineer.add_rolling_features(data, feature_cols, [5, 10, 20])
    
    print(f"特征工程后数据形状: {data.shape}")
    
    # 3. 模型训练
    print("\n3. 训练集成模型...")
    
    # 准备训练数据
    feature_columns = [col for col in data.columns if col not in ['date', 'resp', 'weight', 'ts_id']]
    X = data[feature_columns].fillna(0)
    y = data['resp']
    sample_weight = data['weight']
    
    # 创建模型集成
    ensemble = ModelEnsemble()
    
    # 时间序列交叉验证
    validator = TimeSeriesValidator(n_splits=5)
    
    # 训练模型
    cv_scores = validator.cross_validate(ensemble, X, y, sample_weight=sample_weight)
    
    print(f"交叉验证分数: {cv_scores}")
    print(f"平均分数: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 4. 预测和评估
    print("\n4. 生成预测...")
    
    # 训练最终模型
    ensemble.fit(X, y, sample_weight=sample_weight)
    
    # 生成预测
    predictions = ensemble.predict(X)
    
    # 计算utility score (Jane Street的评估指标)
    utility_score = np.sum(predictions * y * sample_weight) / np.sqrt(np.sum(predictions**2 * sample_weight))
    print(f"Utility Score: {utility_score:.4f}")
    
    # 5. Kaggle提交准备
    print("\n5. 准备Kaggle提交...")
    
    kaggle_tools = KaggleCompetitionTools()
    
    # 创建提交文件
    submission = pd.DataFrame({
        'row_id': range(len(predictions)),
        'action': predictions
    })
    
    submission_path = 'jane_street_submission.csv'
    kaggle_tools.create_submission_file(submission, submission_path)
    print(f"提交文件已保存: {submission_path}")
    
    return {
        'model': ensemble,
        'predictions': predictions,
        'utility_score': utility_score,
        'submission_path': submission_path
    }

def optiver_example():
    """Optiver - Trading at the Close 比赛示例"""
    print("\n" + "=" * 60)
    print("Optiver - Trading at the Close 比赛示例")
    print("=" * 60)
    
    # 1. 数据准备
    print("1. 准备Optiver风格的订单簿数据...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # 模拟订单簿数据
    data = pd.DataFrame()
    data['date_id'] = np.repeat(range(100), 50)  # 100个交易日，每日50个时间点
    data['seconds_in_bucket'] = np.tile(range(0, 600, 12), 100)  # 10分钟内的秒数
    data['stock_id'] = np.random.randint(0, 200, n_samples)  # 200只股票
    
    # 订单簿特征
    for side in ['bid', 'ask']:
        for level in range(1, 3):  # 2档订单簿
            data[f'{side}_price_{level}'] = 100 + np.random.randn(n_samples) * 5
            data[f'{side}_size_{level}'] = np.random.exponential(1000, n_samples)
    
    # 成交数据
    data['wap'] = (data['bid_price_1'] + data['ask_price_1']) / 2  # 加权平均价格
    data['target'] = np.random.randn(n_samples) * 0.001  # 目标：未来价格变化
    
    print(f"数据形状: {data.shape}")
    
    # 2. 特征工程
    print("\n2. 构建订单簿特征...")
    
    feature_engineer = AdvancedFeatureEngineer()
    
    # 价差特征
    data['spread_1'] = data['ask_price_1'] - data['bid_price_1']
    data['spread_2'] = data['ask_price_2'] - data['bid_price_2']
    data['mid_price'] = (data['bid_price_1'] + data['ask_price_1']) / 2
    
    # 订单不平衡特征
    data['order_imbalance_1'] = (data['bid_size_1'] - data['ask_size_1']) / (data['bid_size_1'] + data['ask_size_1'])
    data['order_imbalance_2'] = (data['bid_size_2'] - data['ask_size_2']) / (data['bid_size_2'] + data['ask_size_2'])
    
    # 价格压力特征
    data['price_pressure'] = (data['bid_size_1'] * data['bid_price_1'] - data['ask_size_1'] * data['ask_price_1']) / \
                            (data['bid_size_1'] * data['bid_price_1'] + data['ask_size_1'] * data['ask_price_1'])
    
    # 添加滞后特征
    price_features = ['wap', 'mid_price', 'spread_1']
    data_sorted = data.sort_values(['stock_id', 'date_id', 'seconds_in_bucket'])
    data_sorted = feature_engineer.add_lag_features(data_sorted, price_features, [1, 2, 3], group_col='stock_id')
    
    print(f"特征工程后数据形状: {data_sorted.shape}")
    
    # 3. 模型训练
    print("\n3. 训练时间序列模型...")
    
    # 准备训练数据
    feature_columns = [col for col in data_sorted.columns 
                      if col not in ['date_id', 'seconds_in_bucket', 'stock_id', 'target'] 
                      and not col.startswith('bid_') and not col.startswith('ask_')]
    
    X = data_sorted[feature_columns].fillna(method='ffill').fillna(0)
    y = data_sorted['target']
    
    # 按时间分割数据
    train_mask = data_sorted['date_id'] < 80
    val_mask = data_sorted['date_id'] >= 80
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    # 训练模型
    ensemble = ModelEnsemble()
    ensemble.fit(X_train, y_train)
    
    # 验证
    val_predictions = ensemble.predict(X_val)
    mae = np.mean(np.abs(val_predictions - y_val))
    print(f"验证集MAE: {mae:.6f}")
    
    # 4. 生成提交
    print("\n4. 生成Optiver提交文件...")
    
    test_predictions = ensemble.predict(X_val)  # 使用验证集作为测试集示例
    
    submission = pd.DataFrame({
        'row_id': range(len(test_predictions)),
        'target': test_predictions
    })
    
    submission_path = 'optiver_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"提交文件已保存: {submission_path}")
    
    return {
        'model': ensemble,
        'mae': mae,
        'submission_path': submission_path
    }

def citadel_terminal_ai_example():
    """Citadel Terminal AI Competition 比赛示例"""
    print("\n" + "=" * 60)
    print("Citadel Terminal AI Competition 比赛示例")
    print("=" * 60)
    
    # 1. 创建Terminal AI系统
    print("1. 创建Terminal AI交易系统...")
    
    system = create_terminal_ai_system(buffer_size=5000)
    
    print("系统组件:")
    for component_name in system.keys():
        print(f"  - {component_name}")
    
    # 2. 准备高频数据
    print("\n2. 准备高频交易数据...")
    
    np.random.seed(42)
    n_ticks = 1000
    
    # 模拟高频tick数据
    historical_data = []
    base_price = 100.0
    
    for i in range(n_ticks):
        # 模拟价格随机游走
        price_change = np.random.randn() * 0.01
        base_price += price_change
        
        tick_data = {
            'timestamp': i,
            'price': base_price,
            'volume': np.random.exponential(1000),
            'price_change_pct': price_change,
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'momentum': np.random.randn() * 0.001,
            'volatility': np.random.uniform(0.005, 0.02),
            'sma_20': base_price + np.random.randn() * 0.5
        }
        
        historical_data.append(tick_data)
    
    print(f"生成了 {len(historical_data)} 个tick数据点")
    
    # 3. 运行AI交易模拟
    print("\n3. 运行Terminal AI交易模拟...")
    
    system_config = {
        'buffer_size': 5000,
        'optimization_target': 'sharpe_ratio',
        'simulate_latency': False
    }
    
    simulation_result = run_terminal_ai_simulation(historical_data, system_config)
    
    print("模拟结果:")
    print(f"  - 优化得分: {simulation_result['optimization_result']['best_score']:.4f}")
    print(f"  - 最优参数: {simulation_result['optimization_result']['best_params']}")
    
    # 4. 性能分析
    print("\n4. 分析交易性能...")
    
    performance = simulation_result['performance_summary']
    
    print("性能摘要:")
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")
    
    # 5. 策略优化建议
    print("\n5. 策略优化建议:")
    
    if performance.get('current_pnl', 0) > 0:
        print("  ✓ 策略表现良好，建议继续优化参数")
    else:
        print("  ⚠ 策略需要改进，建议调整信号生成逻辑")
    
    if performance.get('data_quality') == 'good':
        print("  ✓ 数据质量良好")
    else:
        print("  ⚠ 数据质量需要改善")
    
    return {
        'system': system,
        'simulation_result': simulation_result,
        'performance': performance
    }

def cme_crypto_classic_example():
    """CME Group Crypto Classic 比赛示例"""
    print("\n" + "=" * 60)
    print("CME Group Crypto Classic 比赛示例")
    print("=" * 60)
    
    # 1. 准备加密货币数据
    print("1. 准备加密货币期货数据...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # 模拟BTC和ETH期货数据
    data = pd.DataFrame()
    data['timestamp'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    # BTC数据
    btc_price = 40000
    btc_prices = []
    for i in range(n_samples):
        btc_price += np.random.randn() * 100  # 价格随机游走
        btc_prices.append(btc_price)
    
    data['close'] = btc_prices
    data['volume'] = np.random.exponential(1000, n_samples)
    data['futures_price'] = data['close'] + np.random.randn(n_samples) * 50  # 期货价格
    data['spot_price'] = data['close']
    data['open_interest'] = np.random.uniform(10000, 50000, n_samples)
    data['funding_rate'] = np.random.randn(n_samples) * 0.0001
    
    print(f"数据形状: {data.shape}")
    print(f"价格范围: ${data['close'].min():.0f} - ${data['close'].max():.0f}")
    
    # 2. 创建加密货币交易管道
    print("\n2. 创建加密货币交易管道...")
    
    pipeline_result = create_crypto_pipeline(
        data, 
        strategy_type='momentum',
        initial_capital=100000
    )
    
    print("管道组件:")
    for component_name in pipeline_result.keys():
        print(f"  - {component_name}")
    
    # 3. 策略优化
    print("\n3. 优化加密货币交易策略...")
    
    optimized_params = pipeline_result['optimized_params']
    
    print("优化结果:")
    for strategy_type, params in optimized_params.items():
        print(f"  {strategy_type}:")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"    - {key}: {value:.4f}")
            else:
                print(f"    - {key}: {value}")
    
    # 4. 交易模拟
    print("\n4. 运行加密货币交易模拟...")
    
    # 使用优化后的参数进行模拟
    strategy_params = {
        'momentum_strategy': {
            'short_window': 10,
            'long_window': 30
        }
    }
    
    simulation_result = simulate_crypto_trading(
        data, 
        strategy_params,
        initial_capital=100000
    )
    
    performance = simulation_result['performance']
    
    print("交易模拟结果:")
    print(f"  - 初始资金: ${performance['initial_capital']:,.0f}")
    print(f"  - 最终价值: ${performance['current_value']:,.0f}")
    print(f"  - 总收益率: {performance['total_return_pct']:.2f}%")
    print(f"  - 现金余额: ${performance['cash_balance']:,.0f}")
    print(f"  - 总交易次数: {performance['total_trades']}")
    print(f"  - 盈利交易: {performance['winning_trades']}")
    print(f"  - 亏损交易: {performance['losing_trades']}")
    
    # 5. 风险分析
    print("\n5. 风险分析...")
    
    risk_manager = pipeline_result['risk_manager']
    
    # 检查风险限制
    current_pnl = performance['current_value'] - performance['initial_capital']
    daily_pnl = current_pnl * 0.1  # 假设日盈亏为总盈亏的10%
    
    risk_status = risk_manager.check_risk_limits(current_pnl, daily_pnl)
    
    print("风险状态:")
    for key, status in risk_status.items():
        status_symbol = "✓" if status else "✗"
        print(f"  {status_symbol} {key}: {status}")
    
    # 6. 投资组合分析
    print("\n6. 投资组合分析...")
    
    portfolio_manager = simulation_result['portfolio_manager']
    
    print("当前持仓:")
    for symbol, quantity in performance['positions'].items():
        if quantity > 0:
            print(f"  - {symbol}: {quantity:.4f}")
    
    return {
        'pipeline_result': pipeline_result,
        'simulation_result': simulation_result,
        'performance': performance,
        'risk_status': risk_status
    }

def main():
    """主函数：运行所有比赛示例"""
    print("顶级量化比赛完整示例")
    print("本示例展示如何使用项目参加四个顶级量化比赛")
    
    results = {}
    
    try:
        # 1. Jane Street Market Prediction
        results['jane_street'] = jane_street_example()
        
        # 2. Optiver - Trading at the Close
        results['optiver'] = optiver_example()
        
        # 3. Citadel Terminal AI Competition
        results['citadel'] = citadel_terminal_ai_example()
        
        # 4. CME Group Crypto Classic
        results['cme_crypto'] = cme_crypto_classic_example()
        
        # 总结
        print("\n" + "=" * 60)
        print("所有比赛示例运行完成！")
        print("=" * 60)
        
        print("\n比赛结果摘要:")
        
        if 'jane_street' in results:
            print(f"1. Jane Street - Utility Score: {results['jane_street']['utility_score']:.4f}")
        
        if 'optiver' in results:
            print(f"2. Optiver - MAE: {results['optiver']['mae']:.6f}")
        
        if 'citadel' in results:
            pnl = results['citadel']['performance'].get('current_pnl', 0)
            print(f"3. Citadel Terminal AI - PnL: {pnl:.2f}")
        
        if 'cme_crypto' in results:
            return_pct = results['cme_crypto']['performance']['total_return_pct']
            print(f"4. CME Crypto Classic - 收益率: {return_pct:.2f}%")
        
        print("\n项目优势总结:")
        print("✓ 完整的数据处理管道")
        print("✓ 高级特征工程能力")
        print("✓ 多种机器学习模型集成")
        print("✓ 专业的时间序列验证")
        print("✓ 实时交易系统支持")
        print("✓ 全面的风险管理")
        print("✓ 针对性的比赛工具")
        
        print("\n下一步建议:")
        print("1. 根据具体比赛调整参数")
        print("2. 使用真实数据进行测试")
        print("3. 优化模型性能")
        print("4. 准备比赛提交")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return results

if __name__ == "__main__":
    results = main()