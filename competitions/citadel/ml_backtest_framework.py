#!/usr/bin/env python3
"""
ML增强的回测验证框架
包含过拟合检测、交叉验证、稳健性测试等功能
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML相关库
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# 统计测试
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 导入自定义模块
import sys
sys.path.append('.')

class MLBacktestFramework:
    """ML增强的回测验证框架"""
    
    def __init__(self, strategy_func=None):
        self.strategy_func = strategy_func
        self.backtest_results = {}
        self.validation_results = {}
        self.overfitting_metrics = {}
        self.robustness_tests = {}
        
        # 验证配置
        self.validation_config = {
            'n_splits': 5,              # 时间序列交叉验证折数
            'test_size_ratio': 0.2,     # 测试集比例
            'walk_forward_steps': 10,   # 前向验证步数
            'bootstrap_samples': 100,   # 自助法样本数
            'monte_carlo_runs': 1000    # 蒙特卡洛模拟次数
        }
        
        # 过拟合检测阈值
        self.overfitting_thresholds = {
            'performance_degradation': 0.3,  # 样本外表现下降阈值
            'stability_threshold': 0.5,      # 稳定性阈值
            'complexity_penalty': 0.1        # 复杂度惩罚
        }
    
    def time_series_cross_validation(self, data, strategy_params):
        """时间序列交叉验证"""
        print("🔄 执行时间序列交叉验证...")
        
        # 创建时间序列分割器
        tscv = TimeSeriesSplit(n_splits=self.validation_config['n_splits'])
        
        cv_results = []
        fold_performances = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            print(f"   处理第 {fold + 1}/{self.validation_config['n_splits']} 折...")
            
            # 分割数据
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # 在训练集上运行策略
            if self.strategy_func:
                train_result = self.strategy_func(train_data, strategy_params)
                test_result = self.strategy_func(test_data, strategy_params)
            else:
                # 模拟策略结果
                train_result = self._simulate_strategy_result(train_data, strategy_params)
                test_result = self._simulate_strategy_result(test_data, strategy_params)
            
            # 记录结果
            fold_result = {
                'fold': fold + 1,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'train_performance': train_result,
                'test_performance': test_result,
                'performance_ratio': test_result['total_return'] / max(train_result['total_return'], 0.001)
            }
            
            cv_results.append(fold_result)
            fold_performances.append({
                'fold': fold + 1,
                'train_return': train_result['total_return'],
                'test_return': test_result['total_return'],
                'train_sharpe': train_result['sharpe_ratio'],
                'test_sharpe': test_result['sharpe_ratio'],
                'train_drawdown': train_result['max_drawdown'],
                'test_drawdown': test_result['max_drawdown']
            })
        
        # 汇总交叉验证结果
        cv_summary = self._summarize_cv_results(fold_performances)
        
        self.validation_results['time_series_cv'] = {
            'fold_results': cv_results,
            'summary': cv_summary
        }
        
        return cv_summary
    
    def walk_forward_analysis(self, data, strategy_params):
        """前向分析验证"""
        print("🚶 执行前向分析验证...")
        
        n_steps = self.validation_config['walk_forward_steps']
        step_size = len(data) // (n_steps + 1)
        
        wf_results = []
        
        for step in range(n_steps):
            # 计算训练和测试窗口
            train_start = 0
            train_end = (step + 1) * step_size
            test_start = train_end
            test_end = min(test_start + step_size, len(data))
            
            if test_end <= test_start:
                break
            
            print(f"   步骤 {step + 1}/{n_steps}: 训练 {train_start}-{train_end}, 测试 {test_start}-{test_end}")
            
            # 分割数据
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # 运行策略
            if self.strategy_func:
                train_result = self.strategy_func(train_data, strategy_params)
                test_result = self.strategy_func(test_data, strategy_params)
            else:
                train_result = self._simulate_strategy_result(train_data, strategy_params)
                test_result = self._simulate_strategy_result(test_data, strategy_params)
            
            wf_results.append({
                'step': step + 1,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'train_return': train_result['total_return'],
                'test_return': test_result['total_return'],
                'train_sharpe': train_result['sharpe_ratio'],
                'test_sharpe': test_result['sharpe_ratio'],
                'performance_consistency': abs(test_result['sharpe_ratio'] - train_result['sharpe_ratio'])
            })
        
        # 分析前向验证结果
        wf_summary = self._analyze_walk_forward_results(wf_results)
        
        self.validation_results['walk_forward'] = {
            'step_results': wf_results,
            'summary': wf_summary
        }
        
        return wf_summary
    
    def bootstrap_validation(self, data, strategy_params):
        """自助法验证"""
        print("🎲 执行自助法验证...")
        
        n_samples = self.validation_config['bootstrap_samples']
        bootstrap_results = []
        
        for i in range(n_samples):
            if (i + 1) % 20 == 0:
                print(f"   完成 {i + 1}/{n_samples} 个自助样本...")
            
            # 生成自助样本
            bootstrap_data = data.sample(n=len(data), replace=True).sort_index()
            
            # 运行策略
            if self.strategy_func:
                result = self.strategy_func(bootstrap_data, strategy_params)
            else:
                result = self._simulate_strategy_result(bootstrap_data, strategy_params)
            
            bootstrap_results.append({
                'sample': i + 1,
                'total_return': result['total_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'win_rate': result.get('win_rate', 0.5)
            })
        
        # 分析自助法结果
        bootstrap_summary = self._analyze_bootstrap_results(bootstrap_results)
        
        self.validation_results['bootstrap'] = {
            'sample_results': bootstrap_results,
            'summary': bootstrap_summary
        }
        
        return bootstrap_summary
    
    def detect_overfitting(self):
        """检测过拟合"""
        print("🔍 检测策略过拟合...")
        
        overfitting_signals = {}
        
        # 1. 样本内外表现差异检测
        if 'time_series_cv' in self.validation_results:
            cv_results = self.validation_results['time_series_cv']['summary']
            
            # 计算表现下降程度
            train_test_ratio = cv_results['avg_test_return'] / max(cv_results['avg_train_return'], 0.001)
            performance_degradation = 1 - train_test_ratio
            
            overfitting_signals['performance_degradation'] = {
                'value': performance_degradation,
                'threshold': self.overfitting_thresholds['performance_degradation'],
                'is_overfitted': performance_degradation > self.overfitting_thresholds['performance_degradation']
            }
        
        # 2. 稳定性检测
        if 'walk_forward' in self.validation_results:
            wf_results = self.validation_results['walk_forward']['step_results']
            test_returns = [r['test_return'] for r in wf_results]
            
            # 计算收益率稳定性
            return_stability = np.std(test_returns) / max(np.mean(test_returns), 0.001)
            
            overfitting_signals['stability'] = {
                'value': return_stability,
                'threshold': self.overfitting_thresholds['stability_threshold'],
                'is_overfitted': return_stability > self.overfitting_thresholds['stability_threshold']
            }
        
        # 3. 自助法一致性检测
        if 'bootstrap' in self.validation_results:
            bootstrap_summary = self.validation_results['bootstrap']['summary']
            
            # 检查置信区间是否包含零
            return_ci = bootstrap_summary['return_confidence_interval']
            sharpe_ci = bootstrap_summary['sharpe_confidence_interval']
            
            overfitting_signals['bootstrap_consistency'] = {
                'return_includes_zero': return_ci[0] <= 0 <= return_ci[1],
                'sharpe_includes_zero': sharpe_ci[0] <= 0 <= sharpe_ci[1],
                'is_overfitted': return_ci[0] <= 0 or sharpe_ci[0] <= 0
            }
        
        # 4. 综合过拟合评分
        overfitting_score = 0
        total_signals = 0
        
        for signal_name, signal_data in overfitting_signals.items():
            if isinstance(signal_data, dict) and 'is_overfitted' in signal_data:
                if signal_data['is_overfitted']:
                    overfitting_score += 1
                total_signals += 1
        
        overall_overfitting = {
            'overfitting_score': overfitting_score / max(total_signals, 1),
            'is_likely_overfitted': overfitting_score / max(total_signals, 1) > 0.5,
            'signals': overfitting_signals
        }
        
        self.overfitting_metrics = overall_overfitting
        
        print(f"   过拟合评分: {overall_overfitting['overfitting_score']:.2f}")
        print(f"   可能过拟合: {'是' if overall_overfitting['is_likely_overfitted'] else '否'}")
        
        return overall_overfitting
    
    def robustness_testing(self, data, base_params):
        """稳健性测试"""
        print("🛡️  执行稳健性测试...")
        
        robustness_results = {}
        
        # 1. 参数敏感性测试
        param_sensitivity = self._test_parameter_sensitivity(data, base_params)
        robustness_results['parameter_sensitivity'] = param_sensitivity
        
        # 2. 数据扰动测试
        noise_robustness = self._test_noise_robustness(data, base_params)
        robustness_results['noise_robustness'] = noise_robustness
        
        # 3. 子期间稳定性测试
        period_stability = self._test_period_stability(data, base_params)
        robustness_results['period_stability'] = period_stability
        
        # 4. 市场制度稳定性测试
        market_regime_stability = self._test_market_regime_stability(data, base_params)
        robustness_results['market_regime_stability'] = market_regime_stability
        
        # 5. 市场状态适应性测试
        market_adaptability = self._test_market_adaptability(data, base_params)
        robustness_results['market_adaptability'] = market_adaptability
        
        self.robustness_tests = robustness_results
        
        return robustness_results
    
    def _simulate_strategy_result(self, data, params):
        """模拟策略结果（用于演示）"""
        # 简单的均值回归策略模拟
        returns = data['Close'].pct_change().dropna()
        
        # 模拟交易信号
        signals = np.random.choice([-1, 0, 1], size=len(returns), p=[0.3, 0.4, 0.3])
        strategy_returns = signals[:-1] * returns.values[1:] * 0.5  # 50%仓位
        
        # 计算性能指标
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / max(np.std(strategy_returns), 0.001) * np.sqrt(252)
        
        # 计算最大回撤
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'win_rate': (strategy_returns > 0).mean()
        }
    
    def _summarize_cv_results(self, fold_performances):
        """汇总交叉验证结果"""
        df = pd.DataFrame(fold_performances)
        
        return {
            'n_folds': len(df),
            'avg_train_return': df['train_return'].mean(),
            'avg_test_return': df['test_return'].mean(),
            'avg_train_sharpe': df['train_sharpe'].mean(),
            'avg_test_sharpe': df['test_sharpe'].mean(),
            'return_consistency': df['test_return'].std(),
            'sharpe_consistency': df['test_sharpe'].std(),
            'performance_degradation': 1 - (df['test_return'].mean() / max(df['train_return'].mean(), 0.001))
        }
    
    def _analyze_walk_forward_results(self, wf_results):
        """分析前向验证结果"""
        df = pd.DataFrame(wf_results)
        
        return {
            'n_steps': len(df),
            'avg_test_return': df['test_return'].mean(),
            'avg_test_sharpe': df['test_sharpe'].mean(),
            'return_trend': stats.linregress(range(len(df)), df['test_return']).slope,
            'sharpe_trend': stats.linregress(range(len(df)), df['test_sharpe']).slope,
            'consistency_score': 1 / (1 + df['performance_consistency'].mean())
        }
    
    def _analyze_bootstrap_results(self, bootstrap_results):
        """分析自助法结果"""
        df = pd.DataFrame(bootstrap_results)
        
        # 计算置信区间
        return_ci = np.percentile(df['total_return'], [2.5, 97.5])
        sharpe_ci = np.percentile(df['sharpe_ratio'], [2.5, 97.5])
        drawdown_ci = np.percentile(df['max_drawdown'], [2.5, 97.5])
        
        return {
            'mean_return': df['total_return'].mean(),
            'mean_sharpe': df['sharpe_ratio'].mean(),
            'mean_drawdown': df['max_drawdown'].mean(),
            'return_confidence_interval': return_ci,
            'sharpe_confidence_interval': sharpe_ci,
            'drawdown_confidence_interval': drawdown_ci,
            'return_stability': df['total_return'].std(),
            'positive_return_probability': (df['total_return'] > 0).mean()
        }
    
    def _test_parameter_sensitivity(self, data, base_params):
        """测试参数敏感性"""
        sensitivity_results = {}
        
        # 对每个参数进行敏感性测试
        for param_name, base_value in base_params.items():
            if isinstance(base_value, (int, float)):
                # 测试参数变化对结果的影响
                param_variations = [0.8, 0.9, 1.0, 1.1, 1.2]  # ±20%变化
                variation_results = []
                
                for variation in param_variations:
                    test_params = base_params.copy()
                    test_params[param_name] = base_value * variation
                    
                    result = self._simulate_strategy_result(data, test_params)
                    variation_results.append({
                        'variation': variation,
                        'return': result['total_return'],
                        'sharpe': result['sharpe_ratio']
                    })
                
                # 计算敏感性指标
                returns = [r['return'] for r in variation_results]
                sensitivity_score = np.std(returns) / max(np.mean(returns), 0.001)
                
                sensitivity_results[param_name] = {
                    'variations': variation_results,
                    'sensitivity_score': sensitivity_score,
                    'is_sensitive': sensitivity_score > 0.5
                }
        
        return sensitivity_results
    
    def _test_noise_robustness(self, data, params):
        """测试噪音稳健性"""
        noise_levels = [0.01, 0.02, 0.05, 0.1]  # 不同噪音水平
        noise_results = []
        
        for noise_level in noise_levels:
            # 添加噪音到价格数据
            noisy_data = data.copy()
            noise = np.random.normal(0, noise_level, len(data))
            noisy_data['Close'] = data['Close'] * (1 + noise)
            
            # 测试策略表现
            result = self._simulate_strategy_result(noisy_data, params)
            noise_results.append({
                'noise_level': noise_level,
                'return': result['total_return'],
                'sharpe': result['sharpe_ratio']
            })
        
        # 计算稳健性分数
        returns = [r['return'] for r in noise_results]
        robustness_score = 1 - (np.std(returns) / max(np.mean(returns), 0.001))
        
        return {
            'noise_tests': noise_results,
            'robustness_score': max(0, robustness_score),
            'is_robust': robustness_score > 0.7
        }
    
    def _test_period_stability(self, data, params):
        """测试子期间稳定性"""
        # 将数据分成4个子期间
        n_periods = 4
        period_size = len(data) // n_periods
        period_results = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(data)
            
            period_data = data.iloc[start_idx:end_idx]
            result = self._simulate_strategy_result(period_data, params)
            
            period_results.append({
                'period': i + 1,
                'start_date': period_data.index[0],
                'end_date': period_data.index[-1],
                'return': result['total_return'],
                'sharpe': result['sharpe_ratio']
            })
        
        # 计算期间稳定性
        returns = [r['return'] for r in period_results]
        stability_score = 1 - (np.std(returns) / max(np.mean(returns), 0.001))
        
        return {
            'period_results': period_results,
            'stability_score': max(0, stability_score),
            'is_stable': stability_score > 0.6
        }
    
    def _test_market_regime_stability(self, data, params):
        """测试策略在不同市场制度下的稳定性"""
        # 基于波动率分类市场状态
        returns = data['Close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().dropna()
        
        # 确保索引对齐
        valid_idx = volatility.index
        data_aligned = data.loc[valid_idx]
        
        # 定义市场状态
        vol_low = volatility.quantile(0.33)
        vol_high = volatility.quantile(0.67)
        
        market_states = []
        for vol in volatility:
            if vol <= vol_low:
                market_states.append('low_vol')
            elif vol >= vol_high:
                market_states.append('high_vol')
            else:
                market_states.append('normal')
        
        # 测试每个市场状态下的表现
        state_results = {}
        for state in ['low_vol', 'normal', 'high_vol']:
            state_mask = pd.Series(market_states, index=valid_idx) == state
            if state_mask.sum() > 20:  # 确保有足够数据
                state_data = data_aligned[state_mask]
                result = self._simulate_strategy_result(state_data, params)
                state_results[state] = result
        
        return state_results
    
    def _test_market_adaptability(self, data, params):
        """测试策略在不同市场状态下的适应性"""
        results = {}
        
        # 计算市场状态指标
        volatility = data['Close'].rolling(20).std()
        trend = data['Close'].rolling(20).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # 删除NaN值以确保索引对齐
        valid_idx = ~(volatility.isna() | trend.isna())
        data_clean = data[valid_idx].copy()
        volatility_clean = volatility[valid_idx]
        trend_clean = trend[valid_idx]
        
        # 定义市场状态
        vol_threshold = volatility_clean.quantile(0.5)
        trend_threshold = 0.02
        
        states = {
            'high_vol_bull': (volatility_clean > vol_threshold) & (trend_clean > trend_threshold),
            'high_vol_bear': (volatility_clean > vol_threshold) & (trend_clean < -trend_threshold),
            'low_vol_bull': (volatility_clean <= vol_threshold) & (trend_clean > trend_threshold),
            'low_vol_bear': (volatility_clean <= vol_threshold) & (trend_clean < -trend_threshold)
        }
        
        state_results = {}
        for state_name, state_mask in states.items():
            if state_mask.sum() > 50:  # 确保有足够的数据点
                state_data = data_clean[state_mask]
                performance = self._simulate_strategy_result(state_data, params)
                state_results[state_name] = performance
        
        # 计算适应性分数
        if len(state_results) >= 2:
            returns = [r['total_return'] for r in state_results.values()]
            adaptability_score = 1 - (np.std(returns) / max(np.mean(returns), 0.001))
        else:
            adaptability_score = 0.5
        
        return {
            'state_results': state_results,
            'adaptability_score': max(0, adaptability_score),
            'is_adaptable': adaptability_score > 0.5
        }
    
    def generate_validation_report(self):
        """生成验证报告"""
        print("\n📊 生成ML增强回测验证报告...")
        
        report = {
            'timestamp': datetime.now(),
            'validation_summary': {},
            'overfitting_analysis': self.overfitting_metrics,
            'robustness_analysis': self.robustness_tests,
            'recommendations': []
        }
        
        # 汇总验证结果
        if 'time_series_cv' in self.validation_results:
            cv_summary = self.validation_results['time_series_cv']['summary']
            report['validation_summary']['cross_validation'] = {
                'avg_test_return': cv_summary['avg_test_return'],
                'avg_test_sharpe': cv_summary['avg_test_sharpe'],
                'performance_degradation': cv_summary['performance_degradation']
            }
        
        if 'walk_forward' in self.validation_results:
            wf_summary = self.validation_results['walk_forward']['summary']
            report['validation_summary']['walk_forward'] = {
                'avg_test_return': wf_summary['avg_test_return'],
                'return_trend': wf_summary['return_trend'],
                'consistency_score': wf_summary['consistency_score']
            }
        
        if 'bootstrap' in self.validation_results:
            bs_summary = self.validation_results['bootstrap']['summary']
            report['validation_summary']['bootstrap'] = {
                'mean_return': bs_summary['mean_return'],
                'return_confidence_interval': bs_summary['return_confidence_interval'],
                'positive_return_probability': bs_summary['positive_return_probability']
            }
        
        # 生成建议
        recommendations = []
        
        if self.overfitting_metrics.get('is_likely_overfitted', False):
            recommendations.append("⚠️  检测到可能的过拟合，建议简化模型或增加正则化")
        
        if 'parameter_sensitivity' in self.robustness_tests:
            sensitive_params = [
                param for param, data in self.robustness_tests['parameter_sensitivity'].items()
                if data.get('is_sensitive', False)
            ]
            if sensitive_params:
                recommendations.append(f"🔧 参数 {', '.join(sensitive_params)} 较为敏感，需要谨慎调整")
        
        if 'noise_robustness' in self.robustness_tests:
            if not self.robustness_tests['noise_robustness'].get('is_robust', True):
                recommendations.append("🛡️  策略对噪音敏感，建议增加滤波或平滑机制")
        
        report['recommendations'] = recommendations
        
        return report
    
    def visualize_validation_results(self):
        """可视化验证结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML增强回测验证结果', fontsize=16, fontweight='bold')
        
        # 1. 交叉验证结果
        if 'time_series_cv' in self.validation_results:
            cv_data = self.validation_results['time_series_cv']['fold_results']
            folds = [r['fold'] for r in cv_data]
            train_returns = [r['train_performance']['total_return'] for r in cv_data]
            test_returns = [r['test_performance']['total_return'] for r in cv_data]
            
            axes[0, 0].bar([f-0.2 for f in folds], train_returns, width=0.4, label='训练集', alpha=0.7)
            axes[0, 0].bar([f+0.2 for f in folds], test_returns, width=0.4, label='测试集', alpha=0.7)
            axes[0, 0].set_title('时间序列交叉验证')
            axes[0, 0].set_xlabel('折数')
            axes[0, 0].set_ylabel('收益率')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 前向分析结果
        if 'walk_forward' in self.validation_results:
            wf_data = self.validation_results['walk_forward']['step_results']
            steps = [r['step'] for r in wf_data]
            test_returns = [r['test_return'] for r in wf_data]
            
            axes[0, 1].plot(steps, test_returns, 'o-', linewidth=2, markersize=6)
            axes[0, 1].set_title('前向分析验证')
            axes[0, 1].set_xlabel('步骤')
            axes[0, 1].set_ylabel('测试收益率')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 自助法结果分布
        if 'bootstrap' in self.validation_results:
            bs_data = self.validation_results['bootstrap']['sample_results']
            returns = [r['total_return'] for r in bs_data]
            
            axes[0, 2].hist(returns, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 2].axvline(np.mean(returns), color='red', linestyle='--', label=f'均值: {np.mean(returns):.3f}')
            axes[0, 2].set_title('自助法收益率分布')
            axes[0, 2].set_xlabel('收益率')
            axes[0, 2].set_ylabel('频次')
            axes[0, 2].legend()
        
        # 4. 过拟合检测
        if self.overfitting_metrics:
            signals = self.overfitting_metrics.get('signals', {})
            signal_names = []
            signal_values = []
            
            for name, data in signals.items():
                if isinstance(data, dict) and 'value' in data:
                    signal_names.append(name.replace('_', '\n'))
                    signal_values.append(data['value'])
            
            if signal_names:
                colors = ['red' if v > 0.5 else 'green' for v in signal_values]
                axes[1, 0].bar(signal_names, signal_values, color=colors, alpha=0.7)
                axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', label='风险阈值')
                axes[1, 0].set_title('过拟合检测信号')
                axes[1, 0].set_ylabel('风险分数')
                axes[1, 0].legend()
        
        # 5. 稳健性测试
        if 'parameter_sensitivity' in self.robustness_tests:
            param_data = self.robustness_tests['parameter_sensitivity']
            param_names = list(param_data.keys())[:5]  # 显示前5个参数
            sensitivity_scores = [param_data[name]['sensitivity_score'] for name in param_names]
            
            colors = ['red' if s > 0.5 else 'green' for s in sensitivity_scores]
            axes[1, 1].bar(param_names, sensitivity_scores, color=colors, alpha=0.7)
            axes[1, 1].axhline(y=0.5, color='orange', linestyle='--', label='敏感阈值')
            axes[1, 1].set_title('参数敏感性分析')
            axes[1, 1].set_ylabel('敏感性分数')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
        
        # 6. 综合评估雷达图
        if self.validation_results and self.robustness_tests:
            # 计算各维度得分
            scores = {}
            
            if 'time_series_cv' in self.validation_results:
                cv_score = 1 - self.validation_results['time_series_cv']['summary']['performance_degradation']
                scores['交叉验证'] = max(0, min(1, cv_score))
            
            if 'walk_forward' in self.validation_results:
                wf_score = self.validation_results['walk_forward']['summary']['consistency_score']
                scores['前向分析'] = max(0, min(1, wf_score))
            
            if 'bootstrap' in self.validation_results:
                bs_score = self.validation_results['bootstrap']['summary']['positive_return_probability']
                scores['自助法'] = max(0, min(1, bs_score))
            
            if 'noise_robustness' in self.robustness_tests:
                scores['噪音稳健性'] = self.robustness_tests['noise_robustness']['robustness_score']
            
            if 'period_stability' in self.robustness_tests:
                scores['期间稳定性'] = self.robustness_tests['period_stability']['stability_score']
            
            if scores:
                categories = list(scores.keys())
                values = list(scores.values())
                
                # 创建雷达图
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
                values += values[:1]  # 闭合图形
                angles = np.concatenate((angles, [angles[0]]))
                
                axes[1, 2].plot(angles, values, 'o-', linewidth=2, label='策略得分')
                axes[1, 2].fill(angles, values, alpha=0.25)
                axes[1, 2].set_xticks(angles[:-1])
                axes[1, 2].set_xticklabels(categories)
                axes[1, 2].set_ylim(0, 1)
                axes[1, 2].set_title('综合评估雷达图')
                axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('/tmp/ml_backtest_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 验证结果可视化已保存至: /tmp/ml_backtest_validation.png")

def main():
    """主函数 - 演示ML增强回测验证框架"""
    print("🧪 ML增强回测验证框架演示")
    print("=" * 60)
    
    # 创建验证框架
    framework = MLBacktestFramework()
    
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    
    # 生成更真实的价格数据
    returns = np.random.normal(0.0005, 0.015, len(dates))  # 日收益率
    # 添加一些趋势和周期性
    trend = np.linspace(0, 0.1, len(dates))
    cycle = 0.02 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # 年度周期
    returns += trend / len(dates) + cycle / len(dates)
    
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.lognormal(15, 0.3, len(dates))
    
    market_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'Volume': volumes
    }).set_index('Date')
    
    # 模拟策略参数
    strategy_params = {
        'signal_threshold': 0.03,
        'stop_loss': 0.02,
        'take_profit': 0.06,
        'max_position_size': 0.3
    }
    
    print(f"📊 数据期间: {market_data.index[0]} 至 {market_data.index[-1]}")
    print(f"📊 数据点数: {len(market_data)}")
    
    # 执行各种验证
    print("\n🔄 开始执行ML增强验证...")
    
    # 1. 时间序列交叉验证
    cv_results = framework.time_series_cross_validation(market_data, strategy_params)
    print(f"   交叉验证平均测试收益率: {cv_results['avg_test_return']:.3f}")
    print(f"   表现下降程度: {cv_results['performance_degradation']:.3f}")
    
    # 2. 前向分析验证
    wf_results = framework.walk_forward_analysis(market_data, strategy_params)
    print(f"   前向分析平均测试收益率: {wf_results['avg_test_return']:.3f}")
    print(f"   一致性得分: {wf_results['consistency_score']:.3f}")
    
    # 3. 自助法验证
    bs_results = framework.bootstrap_validation(market_data, strategy_params)
    print(f"   自助法平均收益率: {bs_results['mean_return']:.3f}")
    print(f"   正收益概率: {bs_results['positive_return_probability']:.3f}")
    
    # 4. 过拟合检测
    overfitting_results = framework.detect_overfitting()
    
    # 5. 稳健性测试
    robustness_results = framework.robustness_testing(market_data, strategy_params)
    
    # 6. 生成验证报告
    report = framework.generate_validation_report()
    
    print("\n📋 验证报告摘要:")
    print(f"   过拟合风险: {'高' if report['overfitting_analysis'].get('is_likely_overfitted', False) else '低'}")
    
    if report['recommendations']:
        print("   建议:")
        for rec in report['recommendations']:
            print(f"     {rec}")
    
    # 7. 可视化结果
    framework.visualize_validation_results()
    
    print("\n🚀 ML增强回测验证演示完成!")

if __name__ == "__main__":
    main()