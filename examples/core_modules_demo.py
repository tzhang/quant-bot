#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心量化交易模块演示

展示从Citadel高频交易竞赛中提炼的通用量化交易能力
包括信号生成、风险管理、参数优化、机器学习、监控诊断等功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入核心模块
try:
    from src.core import (
        SignalGenerator, SignalFusion, SignalOptimizer,
        AdaptiveRiskManager, MarketRegimeDetector, VolatilityPredictor,
        BayesianOptimizer, GeneticOptimizer, MultiObjectiveOptimizer,
        MLFeatureAnalyzer, ModelEnsemble, TimeSeriesValidator,
        PerformanceMonitor, RiskMonitor, SystemHealthMonitor,
        StrategyDiagnostics, PerformanceProfiler, ErrorAnalyzer,
        DataValidator, TimeSeriesUtils, PerformanceUtils,
        global_config
    )
    print("✅ 成功导入所有核心模块")
except ImportError as e:
    print(f"❌ 导入核心模块失败: {e}")
    print("请确保已正确安装所有依赖包")
    sys.exit(1)


def generate_sample_data(n_days=252, n_assets=5):
    """生成示例数据"""
    print("📊 生成示例数据...")
    
    # 生成日期索引
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # 生成价格数据
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (n_days, n_assets))
    
    # 添加一些趋势和波动性聚集
    for i in range(n_assets):
        # 添加趋势
        trend = np.sin(np.linspace(0, 4*np.pi, n_days)) * 0.001
        returns[:, i] += trend
        
        # 添加波动性聚集
        volatility = 0.01 + 0.01 * np.abs(np.sin(np.linspace(0, 2*np.pi, n_days)))
        returns[:, i] *= volatility / 0.02
    
    # 计算价格
    prices = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), 
                         index=dates, 
                         columns=[f'Asset_{i+1}' for i in range(n_assets)])
    
    # 生成收益率
    returns_df = pd.DataFrame(returns, 
                             index=dates, 
                             columns=[f'Asset_{i+1}' for i in range(n_assets)])
    
    print(f"✅ 生成了 {n_days} 天 {n_assets} 个资产的数据")
    return prices, returns_df


def demo_signal_generation(prices, returns):
    """演示信号生成功能"""
    print("\n🎯 信号生成与处理演示")
    print("=" * 50)
    
    # 选择第一个资产进行演示
    asset = prices.columns[0]
    print(f"📈 为 {asset} 生成交易信号...")
    
    # 构造OHLC数据
    ohlc_data = pd.DataFrame({
        'close': prices[asset],
        'high': prices[asset] * (1 + np.random.uniform(0, 0.02, len(prices))),
        'low': prices[asset] * (1 - np.random.uniform(0, 0.02, len(prices))),
        'volume': np.random.uniform(1000, 10000, len(prices))
    })
    
    # 初始化信号生成器
    signal_gen = SignalGenerator()
    
    # 生成各类信号
    momentum_signals = signal_gen.add_momentum_signals(ohlc_data, periods=[10, 20])
    mean_reversion_signals = signal_gen.add_mean_reversion_signals(ohlc_data, periods=[20, 50])
    volatility_signals = signal_gen.add_volatility_signals(ohlc_data, periods=[10, 20])
    microstructure_signals = signal_gen.add_microstructure_signals(ohlc_data)
    
    print(f"✅ 生成了 {len(signal_gen.get_all_signals())} 个信号")
    
    # 信号融合
    print("🔄 融合多个信号...")
    fusion = SignalFusion(method='weighted_average')
    
    # 设置权重
    all_signals = signal_gen.get_all_signals()
    weights = {name: 1.0 / len(all_signals) for name in all_signals.keys()}
    fusion.set_weights(weights)
    
    # 融合信号
    fused_signal = fusion.fuse_signals(all_signals)
    print(f"✅ 融合信号生成完成，长度: {len(fused_signal)}")
    
    return all_signals, fused_signal


def demo_risk_management(returns):
    """演示风险管理功能"""
    print("\n🛡️ 自适应风险管理演示")
    print("=" * 50)
    
    # 初始化风险管理器
    risk_manager = AdaptiveRiskManager(
        max_position_size=0.1,
        max_portfolio_risk=0.02
    )
    
    print("📊 风险指标计算...")
    
    # 计算仓位大小
    signal_strength = 0.05  # 5%的信号强度
    volatility = returns.iloc[:, 0].std() * np.sqrt(252)  # 年化波动率
    confidence = 0.8
    
    position_size = risk_manager.calculate_position_size(
        signal_strength=signal_strength,
        volatility=volatility,
        confidence=confidence
    )
    print(f"   建议仓位大小: {position_size:.4f}")
    
    # 计算投资组合风险
    positions = {asset: 0.2 for asset in returns.columns}  # 等权重
    risk_metrics = risk_manager.calculate_portfolio_risk(positions, returns)
    
    print(f"   投资组合波动率: {risk_metrics['portfolio_volatility']:.4f}")
    print(f"   VaR (95%): {risk_metrics['var_95']:.4f}")
    print(f"   CVaR (95%): {risk_metrics['cvar_95']:.4f}")
    print(f"   最大回撤: {risk_metrics['max_drawdown']:.4f}")
    
    # 风险调整后的仓位
    adjusted_positions = risk_manager.adjust_positions_for_risk(positions, returns)
    print(f"   风险调整后仓位数量: {len(adjusted_positions)}")
    
    # 市场状态检测
    print("🔍 市场状态检测...")
    regime_detector = MarketRegimeDetector(lookback_period=60)
    
    # 构造价格数据
    prices = (1 + returns).cumprod()
    market_data = pd.DataFrame({
        'close': prices.iloc[:, 0],
        'volume': np.random.uniform(1000, 10000, len(prices))
    })
    
    regime = regime_detector.detect_regime(market_data)
    print(f"   当前市场状态: {regime}")
    
    regime_probs = regime_detector.get_regime_probabilities(market_data)
    for regime_name, prob in regime_probs.items():
        print(f"   {regime_name}: {prob:.2%}")
    
    # 波动率预测
    print("📈 波动率预测...")
    vol_predictor = VolatilityPredictor(model_type='ewma')
    vol_predictor.fit(returns.iloc[:, 0])
    
    vol_forecast = vol_predictor.predict(horizon=5)
    print(f"   未来5期波动率预测: {vol_forecast}")
    
    current_vol = returns.iloc[:, 0].rolling(20).std().iloc[-1] * np.sqrt(252)
    vol_regime = vol_predictor.get_volatility_regime(current_vol)
    print(f"   当前波动率状态: {vol_regime}")
    
    return risk_manager, regime_detector, vol_predictor


def demo_optimization(returns, signals):
    """演示参数优化功能"""
    print("\n🎯 多目标参数优化演示")
    print("=" * 50)
    
    # 转换信号字典为DataFrame
    if isinstance(signals, dict):
        signals_df = pd.DataFrame(signals).dropna()
        if signals_df.empty:
            print("⚠️ 信号数据为空，跳过优化演示")
            return {}, 0.0
    else:
        signals_df = signals
    
    # 定义优化目标函数
    def objective_function(params):
        """示例目标函数：最大化夏普比率"""
        try:
            window = int(params[0])
            threshold = params[1]
            
            # 简单的信号处理
            signal = signals_df.iloc[:, 0].rolling(window).mean()
            positions = np.where(signal > threshold, 1, -1)
            
            # 计算收益
            portfolio_returns = returns.mean(axis=1) * pd.Series(positions, index=returns.index)
            
            # 计算夏普比率
            if portfolio_returns.std() == 0:
                return 0
            
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            return sharpe_ratio
        except Exception as e:
            return 0
    
    print("🔍 贝叶斯优化...")
    
    # 贝叶斯优化
    bayesian_opt = BayesianOptimizer()
    
    # 定义参数空间
    param_bounds = [(5, 50), (-0.1, 0.1)]  # window, threshold
    
    try:
        best_params, best_score = bayesian_opt.optimize(
            objective_function, 
            param_bounds, 
            n_calls=20
        )
        print(f"   最优参数: {best_params}")
        print(f"   最优得分: {best_score:.4f}")
    except Exception as e:
        print(f"   贝叶斯优化失败，使用随机搜索: {str(e)}")
        # 使用随机搜索作为后备
        best_score = -np.inf
        best_params = None
        for _ in range(20):
            params = [
                np.random.randint(5, 51),
                np.random.uniform(-0.1, 0.1)
            ]
            score = objective_function(params)
            if score > best_score:
                best_score = score
                best_params = params
        print(f"   随机搜索最优参数: {best_params}")
        print(f"   随机搜索最优得分: {best_score:.4f}")
    
    return best_params, best_score


def demo_ml_engine(returns, signals):
    """演示机器学习引擎功能"""
    print("\n🤖 ML增强交易系统演示")
    print("=" * 50)
    
    # 特征分析
    ml_analyzer = MLFeatureAnalyzer()
    
    print("📊 特征重要性分析...")
    
    # 转换信号字典为DataFrame
    if isinstance(signals, dict):
        signals_df = pd.DataFrame(signals).dropna()
        if signals_df.empty:
            print("⚠️ 信号数据为空，跳过ML演示")
            return None, None
    else:
        signals_df = signals
    
    # 准备目标变量（未来收益）
    target = returns.mean(axis=1).shift(-1).dropna()
    features = signals_df.iloc[:-1]  # 对齐数据
    
    # 确保数据长度一致
    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]
    
    # 特征重要性分析
    try:
        importance_scores = ml_analyzer.analyze_feature_importance(features, target)
        print("   特征重要性排序:")
        for feature, score in importance_scores.items():
            print(f"     {feature}: {score:.4f}")
    except Exception as e:
        print(f"   特征重要性分析失败: {str(e)}")
    
    # 相关性分析
    try:
        correlation_matrix = ml_analyzer.analyze_feature_correlations(features)
        print(f"   特征间平均相关性: {correlation_matrix.mean().mean():.4f}")
    except Exception as e:
        print(f"   相关性分析失败: {str(e)}")
    
    # 模型集成
    print("\n🎯 模型集成...")
    model_ensemble = ModelEnsemble()
    
    try:
        # 训练集成模型
        model_ensemble.fit(features, target)
        
        # 预测
        predictions = model_ensemble.predict(features.iloc[-10:])
        print(f"   最近10期预测均值: {predictions.mean():.4f}")
        
        # 模型评估
        performance = model_ensemble.get_model_performance(features, target)
        print("   模型性能:")
        for model_name, score in performance.items():
            print(f"     {model_name}: {score:.4f}")
            
    except Exception as e:
        print(f"   模型集成失败: {str(e)}")
    
    return ml_analyzer, model_ensemble


def demo_monitoring_system(returns, signals):
    """演示监控系统功能"""
    print("\n📊 实时监控与预警系统演示")
    print("=" * 50)
    
    # 性能监控
    perf_monitor = PerformanceMonitor()
    
    # 模拟交易收益
    portfolio_returns = returns.mean(axis=1)
    
    print("📈 性能监控...")
    
    # 更新性能指标
    for i, ret in enumerate(portfolio_returns.iloc[-30:]):  # 最近30天
        perf_monitor.update_performance(ret, 0.5)  # 假设固定仓位0.5
    
    # 获取当前状态
    current_status = perf_monitor.get_current_status()
    print(f"   累计收益: {current_status['metrics'].get('cumulative_return', 0):.4f}")
    print(f"   夏普比率: {current_status['metrics'].get('sharpe_ratio', 0):.4f}")
    print(f"   最大回撤: {current_status['metrics'].get('max_drawdown', 0):.4f}")
    print(f"   胜率: {current_status['metrics'].get('win_rate', 0):.4f}")
    print(f"   预警数量: {len(current_status.get('alerts', []))}")
    
    # 风险监控
    print("\n🛡️ 风险监控...")
    risk_monitor = RiskMonitor()
    
    # 模拟持仓
    positions = {f'Asset_{i}': np.random.uniform(-0.3, 0.3) for i in range(5)}
    risk_monitor.update_positions(positions)
    
    # 计算VaR
    var_95 = risk_monitor.calculate_var(returns, confidence_level=0.05)
    print(f"   VaR (95%): {var_95:.4f}")
    
    # 获取风险报告
    risk_report = risk_monitor.get_risk_report()
    print(f"   当前持仓数量: {len(risk_report['positions'])}")
    print(f"   风险预警数量: {len(risk_report['recent_alerts'])}")
    
    # 系统健康监控
    print("\n💻 系统健康监控...")
    health_monitor = SystemHealthMonitor()
    
    # 模拟系统指标
    health_monitor.update_system_metrics(
        latency=np.random.uniform(10, 50),
        memory_usage=np.random.uniform(30, 80),
        cpu_usage=np.random.uniform(20, 70)
    )
    
    # 获取系统状态
    system_status = health_monitor.get_system_status()
    print(f"   系统状态: {system_status['overall_status']}")
    print(f"   健康指标数量: {len(system_status['health_metrics'])}")
    print(f"   组件状态数量: {len(system_status['component_statuses'])}")
    print(f"   系统预警数量: {len(system_status['recent_alerts'])}")
    
    return perf_monitor, risk_monitor, health_monitor


def demo_diagnostics_system(returns, signals):
    """演示诊断系统功能"""
    print("\n🔍 系统化调试与诊断演示")
    print("=" * 50)
    
    # 策略诊断
    strategy_diagnostics = StrategyDiagnostics()
    
    print("📊 策略性能诊断...")
    
    # 准备策略结果
    portfolio_returns = returns.mean(axis=1)
    
    # 将信号字典转换为DataFrame并确保数据类型正确
    if isinstance(signals, dict):
        signals_df = pd.DataFrame(signals)
        # 确保所有列都是数值类型
        for col in signals_df.columns:
            if signals_df[col].dtype == bool:
                signals_df[col] = signals_df[col].astype(float)
            elif not pd.api.types.is_numeric_dtype(signals_df[col]):
                try:
                    signals_df[col] = pd.to_numeric(signals_df[col], errors='coerce')
                except:
                    signals_df[col] = signals_df[col].astype(float)
    else:
        signals_df = signals
    
    strategy_results = {
        'returns': portfolio_returns,
        'signals': signals_df,
        'trades': [
            {'size': 1000, 'execution_time': 100, 'slippage': 0.001},
            {'size': 1500, 'execution_time': 150, 'slippage': 0.002},
            {'size': 800, 'execution_time': 80, 'slippage': 0.0015}
        ]
    }
    
    # 执行诊断
    diagnosis = strategy_diagnostics.diagnose_strategy_performance(strategy_results)
    
    print("   性能摘要:")
    perf_summary = diagnosis['performance_summary']
    print(f"     年化收益: {perf_summary.get('annual_return', 0):.4f}")
    print(f"     夏普比率: {perf_summary.get('sharpe_ratio', 0):.4f}")
    print(f"     最大回撤: {perf_summary.get('max_drawdown', 0):.4f}")
    print(f"     胜率: {perf_summary.get('win_rate', 0):.4f}")
    
    # 问题识别
    if diagnosis['issues_identified']:
        print("   ⚠️ 识别的问题:")
        for issue in diagnosis['issues_identified']:
            print(f"     {issue['type']}: {issue['description']}")
    
    # 改进建议
    if diagnosis['recommendations']:
        print("   💡 改进建议:")
        for rec in diagnosis['recommendations'][:3]:  # 显示前3个建议
            print(f"     • {rec}")
    
    # 性能分析器
    print("\n⚡ 性能分析...")
    profiler = PerformanceProfiler()
    
    # 装饰一个示例函数
    @profiler.profile_function
    def sample_calculation():
        """示例计算函数"""
        import time
        time.sleep(0.01)  # 模拟计算时间
        return np.random.random(1000).sum()
    
    # 执行几次以收集性能数据
    for _ in range(5):
        sample_calculation()
    
    # 获取性能报告
    perf_report = profiler.get_performance_report()
    if perf_report['function_performance']:
        func_name = 'sample_calculation'
        if func_name in perf_report['function_performance']:
            func_stats = perf_report['function_performance'][func_name]
            print(f"   函数 {func_name}:")
            print(f"     调用次数: {func_stats['total_calls']}")
            print(f"     平均执行时间: {func_stats['avg_execution_time']:.4f}s")
            print(f"     最大执行时间: {func_stats['max_execution_time']:.4f}s")
    
    # 错误分析器
    print("\n🚨 错误分析...")
    error_analyzer = ErrorAnalyzer()
    
    # 模拟一些错误
    try:
        raise ValueError("示例错误：参数值无效")
    except Exception as e:
        error_analyzer.log_error(e, context={'function': 'demo_function'})
    
    try:
        raise KeyError("示例错误：缺少必要的键")
    except Exception as e:
        error_analyzer.log_error(e, context={'function': 'data_processing'})
    
    # 获取错误报告
    error_report = error_analyzer.get_error_report()
    print(f"   总错误数: {error_report['error_statistics']['total_errors']}")
    
    if error_report['recommendations']:
        print("   错误处理建议:")
        for rec in error_report['recommendations']:
            print(f"     • {rec}")
    
    return strategy_diagnostics, profiler, error_analyzer


def demo_utils_functions(returns, signals):
    """演示工具函数"""
    print("\n🛠️ 工具函数演示")
    print("=" * 50)
    
    # 数据验证
    print("✅ 数据验证...")
    portfolio_returns = returns.mean(axis=1)
    
    # 将信号字典转换为DataFrame
    if isinstance(signals, dict):
        signals_df = pd.DataFrame(signals)
    else:
        signals_df = signals
    
    is_valid_returns = DataValidator.validate_returns(portfolio_returns)
    is_valid_prices = DataValidator.validate_prices(returns.cumsum() + 100)
    
    # 确保信号数据不为空
    if not signals_df.empty:
        is_valid_signals = DataValidator.validate_signals(signals_df.iloc[:, 0])
    else:
        is_valid_signals = False
    
    print(f"   收益率数据有效: {is_valid_returns}")
    print(f"   价格数据有效: {is_valid_prices}")
    print(f"   信号数据有效: {is_valid_signals}")
    
    # 时间序列工具
    print("\n📈 时间序列分析...")
    rolling_stats = TimeSeriesUtils.calculate_rolling_statistics(
        portfolio_returns, window=20, statistics=['mean', 'std', 'skew']
    )
    print(f"   滚动统计量形状: {rolling_stats.shape}")
    
    # 异常值检测
    outliers = TimeSeriesUtils.detect_outliers(portfolio_returns, method='iqr')
    print(f"   检测到异常值: {outliers.sum()} 个")
    
    # 性能计算
    print("\n📊 性能指标计算...")
    sharpe_ratio = PerformanceUtils.calculate_sharpe_ratio(portfolio_returns)
    sortino_ratio = PerformanceUtils.calculate_sortino_ratio(portfolio_returns)
    max_drawdown = PerformanceUtils.calculate_max_drawdown(portfolio_returns)
    var_95 = PerformanceUtils.calculate_var(portfolio_returns)
    
    print(f"   夏普比率: {sharpe_ratio:.4f}")
    print(f"   索提诺比率: {sortino_ratio:.4f}")
    print(f"   最大回撤: {max_drawdown:.4f}")
    print(f"   95% VaR: {var_95:.4f}")
    
    # 风险计算
    print("\n🛡️ 风险指标计算...")
    try:
        from src.core.utils import RiskUtils
        correlation_matrix = RiskUtils.calculate_correlation_matrix(returns)
        print(f"   相关性矩阵形状: {correlation_matrix.shape}")
        print(f"   平均相关性: {correlation_matrix.mean().mean():.4f}")
    except ImportError:
        print("   RiskUtils模块未找到，跳过风险计算")
        correlation_matrix = returns.corr()
    
    # 信号处理
    print("\n🎯 信号处理...")
    if not signals_df.empty:
        try:
            from src.core.utils import SignalUtils
            normalized_signal = SignalUtils.normalize_signal(signals_df.iloc[:, 0], method='zscore')
            print(f"   标准化信号范围: [{normalized_signal.min():.4f}, {normalized_signal.max():.4f}]")
        except ImportError:
            print("   SignalUtils模块未找到，使用简单标准化")
            signal_data = signals_df.iloc[:, 0]
            normalized_signal = (signal_data - signal_data.mean()) / signal_data.std()
            print(f"   标准化信号范围: [{normalized_signal.min():.4f}, {normalized_signal.max():.4f}]")
    else:
        print("   信号数据为空，跳过信号处理")
    
    return rolling_stats, correlation_matrix


def main():
    """主演示函数"""
    print("🚀 核心量化交易模块演示")
    print("=" * 60)
    print("展示从Citadel高频交易竞赛中提炼的通用量化交易能力")
    print("=" * 60)
    
    # 生成示例数据
    prices, returns = generate_sample_data(n_days=252, n_assets=5)
    
    # 1. 信号生成演示
    signals, fused_signal = demo_signal_generation(prices, returns)
    
    # 2. 风险管理演示
    risk_manager, regime_detector, vol_predictor = demo_risk_management(returns)
    
    # 3. 参数优化演示
    best_params, best_score = demo_optimization(returns, signals)
    
    # 4. 机器学习演示
    ml_analyzer, model_ensemble = demo_ml_engine(returns, signals)
    
    # 5. 监控系统演示
    perf_monitor, risk_monitor, health_monitor = demo_monitoring_system(returns, signals)
    
    # 6. 诊断系统演示
    strategy_diagnostics, profiler, error_analyzer = demo_diagnostics_system(returns, signals)
    
    # 7. 工具函数演示
    rolling_stats, correlation_matrix = demo_utils_functions(returns, signals)
    
    # 总结
    print("\n🎉 演示完成总结")
    print("=" * 50)
    print("✅ 信号生成与处理系统 - 多种技术信号生成和融合")
    print("✅ 自适应风险管理系统 - 动态风险控制和市场状态检测")
    print("✅ 多目标参数优化框架 - 贝叶斯优化和遗传算法")
    print("✅ ML增强交易系统 - 特征分析和模型集成")
    print("✅ 实时监控与预警系统 - 性能、风险和系统健康监控")
    print("✅ 系统化调试与诊断框架 - 策略诊断和性能分析")
    print("✅ 通用工具函数库 - 数据验证、时间序列分析等")
    
    print(f"\n📈 全局配置示例:")
    print(f"   风险免费利率: {global_config.get_config('risk_free_rate')}")
    print(f"   交易成本: {global_config.get_config('trading_cost')}")
    print(f"   最大仓位: {global_config.get_config('max_position_size')}")
    
    print("\n🎯 这些模块提供了构建高质量量化交易系统的核心能力！")


if __name__ == "__main__":
    main()