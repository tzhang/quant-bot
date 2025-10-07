"""
系统化调试与诊断框架

提供策略诊断、性能分析和问题定位功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import traceback
import inspect
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class StrategyDiagnostics:
    """策略诊断器"""
    
    def __init__(self):
        self.diagnostic_results = {}
        self.performance_issues = []
        self.signal_analysis = {}
        self.execution_analysis = {}
        
    def diagnose_strategy_performance(self, 
                                   strategy_results: Dict[str, Any],
                                   benchmark_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """诊断策略性能"""
        diagnosis = {
            'timestamp': datetime.now(),
            'performance_summary': {},
            'issues_identified': [],
            'recommendations': []
        }
        
        # 基本性能分析
        if 'returns' in strategy_results:
            returns = pd.Series(strategy_results['returns'])
            diagnosis['performance_summary'] = self._analyze_returns(returns)
            
            # 识别性能问题
            issues = self._identify_performance_issues(returns, diagnosis['performance_summary'])
            diagnosis['issues_identified'].extend(issues)
            
            # 生成建议
            recommendations = self._generate_recommendations(issues, diagnosis['performance_summary'])
            diagnosis['recommendations'].extend(recommendations)
        
        # 与基准比较
        if benchmark_results and 'returns' in benchmark_results:
            benchmark_returns = pd.Series(benchmark_results['returns'])
            comparison = self._compare_with_benchmark(returns, benchmark_returns)
            diagnosis['benchmark_comparison'] = comparison
        
        # 信号质量分析
        if 'signals' in strategy_results:
            signal_diagnosis = self._diagnose_signals(strategy_results['signals'])
            diagnosis['signal_analysis'] = signal_diagnosis
        
        # 执行质量分析
        if 'trades' in strategy_results:
            execution_diagnosis = self._diagnose_execution(strategy_results['trades'])
            diagnosis['execution_analysis'] = execution_diagnosis
        
        self.diagnostic_results[datetime.now()] = diagnosis
        return diagnosis
    
    def _analyze_returns(self, returns: pd.Series) -> Dict[str, float]:
        """分析收益率"""
        if len(returns) == 0:
            return {}
        
        # 基本统计
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 风险指标
        max_drawdown = self._calculate_max_drawdown(returns)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # 分布特征
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # 胜率和盈亏比
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf,
            'sortino_ratio': self._calculate_sortino_ratio(returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """计算索提诺比率"""
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        return returns.mean() * np.sqrt(252) / (downside_std * np.sqrt(252)) if downside_std > 0 else np.inf
    
    def _identify_performance_issues(self, 
                                   returns: pd.Series, 
                                   performance_summary: Dict[str, float]) -> List[Dict[str, Any]]:
        """识别性能问题"""
        issues = []
        
        # 低夏普比率
        if performance_summary.get('sharpe_ratio', 0) < 0.5:
            issues.append({
                'type': 'low_sharpe_ratio',
                'severity': 'high' if performance_summary.get('sharpe_ratio', 0) < 0 else 'medium',
                'description': f"夏普比率过低: {performance_summary.get('sharpe_ratio', 0):.3f}",
                'metric_value': performance_summary.get('sharpe_ratio', 0)
            })
        
        # 高回撤
        if performance_summary.get('max_drawdown', 0) > 0.2:
            issues.append({
                'type': 'high_drawdown',
                'severity': 'high' if performance_summary.get('max_drawdown', 0) > 0.3 else 'medium',
                'description': f"最大回撤过高: {performance_summary.get('max_drawdown', 0):.3f}",
                'metric_value': performance_summary.get('max_drawdown', 0)
            })
        
        # 低胜率
        if performance_summary.get('win_rate', 0) < 0.4:
            issues.append({
                'type': 'low_win_rate',
                'severity': 'medium',
                'description': f"胜率过低: {performance_summary.get('win_rate', 0):.3f}",
                'metric_value': performance_summary.get('win_rate', 0)
            })
        
        # 高波动率
        if performance_summary.get('volatility', 0) > 0.3:
            issues.append({
                'type': 'high_volatility',
                'severity': 'medium',
                'description': f"波动率过高: {performance_summary.get('volatility', 0):.3f}",
                'metric_value': performance_summary.get('volatility', 0)
            })
        
        # 负偏度（左尾风险）
        if performance_summary.get('skewness', 0) < -1:
            issues.append({
                'type': 'negative_skewness',
                'severity': 'medium',
                'description': f"收益分布负偏: {performance_summary.get('skewness', 0):.3f}",
                'metric_value': performance_summary.get('skewness', 0)
            })
        
        # 高峰度（尾部风险）
        if performance_summary.get('kurtosis', 0) > 3:
            issues.append({
                'type': 'high_kurtosis',
                'severity': 'low',
                'description': f"收益分布高峰度: {performance_summary.get('kurtosis', 0):.3f}",
                'metric_value': performance_summary.get('kurtosis', 0)
            })
        
        return issues
    
    def _generate_recommendations(self, 
                                issues: List[Dict[str, Any]], 
                                performance_summary: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        issue_types = [issue['type'] for issue in issues]
        
        if 'low_sharpe_ratio' in issue_types:
            recommendations.append("考虑优化信号生成逻辑，提高信号质量")
            recommendations.append("检查交易成本是否过高，优化执行策略")
            recommendations.append("考虑增加风险调整机制")
        
        if 'high_drawdown' in issue_types:
            recommendations.append("加强风险管理，设置更严格的止损机制")
            recommendations.append("考虑降低仓位规模或增加分散化")
            recommendations.append("实施动态仓位调整策略")
        
        if 'low_win_rate' in issue_types:
            recommendations.append("优化入场时机，提高信号准确性")
            recommendations.append("考虑调整止盈止损比例")
            recommendations.append("检查市场环境适应性")
        
        if 'high_volatility' in issue_types:
            recommendations.append("增加投资组合分散化")
            recommendations.append("考虑使用波动率调整的仓位管理")
            recommendations.append("检查是否过度集中在高波动资产")
        
        if 'negative_skewness' in issue_types:
            recommendations.append("关注尾部风险管理")
            recommendations.append("考虑使用期权等工具对冲下行风险")
            recommendations.append("优化止损策略以减少极端损失")
        
        return recommendations
    
    def _compare_with_benchmark(self, 
                              strategy_returns: pd.Series, 
                              benchmark_returns: pd.Series) -> Dict[str, Any]:
        """与基准比较"""
        # 对齐时间序列
        min_length = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns.iloc[-min_length:]
        benchmark_returns = benchmark_returns.iloc[-min_length:]
        
        # 计算超额收益
        excess_returns = strategy_returns - benchmark_returns
        
        # 计算指标
        comparison = {
            'excess_return': excess_returns.mean() * 252,
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'information_ratio': excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0,
            'beta': self._calculate_beta(strategy_returns, benchmark_returns),
            'alpha': self._calculate_alpha(strategy_returns, benchmark_returns),
            'correlation': strategy_returns.corr(benchmark_returns),
            'up_capture': self._calculate_capture_ratio(strategy_returns, benchmark_returns, 'up'),
            'down_capture': self._calculate_capture_ratio(strategy_returns, benchmark_returns, 'down')
        }
        
        return comparison
    
    def _calculate_beta(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算贝塔系数"""
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    def _calculate_alpha(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算阿尔法系数"""
        beta = self._calculate_beta(strategy_returns, benchmark_returns)
        alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
        return alpha * 252  # 年化
    
    def _calculate_capture_ratio(self, 
                               strategy_returns: pd.Series, 
                               benchmark_returns: pd.Series, 
                               direction: str) -> float:
        """计算捕获比率"""
        if direction == 'up':
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0
        
        if mask.sum() == 0:
            return 0
        
        strategy_avg = strategy_returns[mask].mean()
        benchmark_avg = benchmark_returns[mask].mean()
        
        return strategy_avg / benchmark_avg if benchmark_avg != 0 else 0
    
    def _diagnose_signals(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """诊断信号质量"""
        signal_diagnosis = {
            'signal_statistics': {},
            'signal_quality': {},
            'signal_issues': []
        }
        
        for column in signals.columns:
            signal = signals[column].dropna()
            
            if len(signal) == 0:
                continue
            
            # 信号统计
            signal_stats = {
                'mean': signal.mean(),
                'std': signal.std(),
                'min': signal.min(),
                'max': signal.max(),
                'skewness': signal.skew(),
                'kurtosis': signal.kurtosis(),
                'autocorr_lag1': signal.autocorr(lag=1) if len(signal) > 1 else 0
            }
            signal_diagnosis['signal_statistics'][column] = signal_stats
            
            # 信号质量评估
            try:
                signal_range = float(signal_stats['max']) - float(signal_stats['min'])
            except (TypeError, ValueError):
                signal_range = 0.0
                
            quality_metrics = {
                'signal_strength': abs(signal_stats['mean']) / signal_stats['std'] if signal_stats['std'] > 0 else 0,
                'signal_stability': 1 / (1 + abs(signal_stats['autocorr_lag1'])),  # 自相关越低越稳定
                'signal_range': signal_range,
                'outlier_ratio': ((signal < signal.quantile(0.01)) | (signal > signal.quantile(0.99))).mean()
            }
            signal_diagnosis['signal_quality'][column] = quality_metrics
            
            # 识别信号问题
            issues = []
            if quality_metrics['signal_strength'] < 0.5:
                issues.append(f"{column}: 信号强度不足")
            if quality_metrics['outlier_ratio'] > 0.1:
                issues.append(f"{column}: 异常值比例过高")
            if abs(signal_stats['autocorr_lag1']) > 0.8:
                issues.append(f"{column}: 信号自相关性过高")
            
            signal_diagnosis['signal_issues'].extend(issues)
        
        return signal_diagnosis
    
    def _diagnose_execution(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """诊断执行质量"""
        execution_diagnosis = {
            'execution_statistics': {},
            'execution_quality': {},
            'execution_issues': []
        }
        
        if not trades:
            return execution_diagnosis
        
        # 提取执行数据
        trade_sizes = [t.get('size', 0) for t in trades if 'size' in t]
        execution_times = [t.get('execution_time', 0) for t in trades if 'execution_time' in t]
        slippage = [t.get('slippage', 0) for t in trades if 'slippage' in t]
        
        # 执行统计
        if trade_sizes:
            execution_diagnosis['execution_statistics']['trade_size'] = {
                'mean': np.mean(trade_sizes),
                'std': np.std(trade_sizes),
                'min': np.min(trade_sizes),
                'max': np.max(trade_sizes)
            }
        
        if execution_times:
            execution_diagnosis['execution_statistics']['execution_time'] = {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'p95': np.percentile(execution_times, 95),
                'p99': np.percentile(execution_times, 99)
            }
        
        if slippage:
            execution_diagnosis['execution_statistics']['slippage'] = {
                'mean': np.mean(slippage),
                'std': np.std(slippage),
                'max': np.max(slippage)
            }
        
        # 执行质量评估
        quality_issues = []
        
        if execution_times and np.mean(execution_times) > 1000:  # 1秒
            quality_issues.append("平均执行时间过长")
        
        if slippage and np.mean(slippage) > 0.01:  # 1%
            quality_issues.append("平均滑点过高")
        
        if execution_times and np.percentile(execution_times, 95) > 5000:  # 5秒
            quality_issues.append("95%执行时间过长")
        
        execution_diagnosis['execution_issues'] = quality_issues
        
        return execution_diagnosis


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiling_results = {}
        self.function_timings = defaultdict(list)
        self.memory_usage = []
        
    def profile_function(self, func: Callable) -> Callable:
        """函数性能分析装饰器"""
        def wrapper(*args, **kwargs):
            import time
            import psutil
            import os
            
            # 记录开始时间和内存
            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                
                # 记录结束时间和内存
                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # 保存性能数据
                timing_data = {
                    'function': func.__name__,
                    'execution_time': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'timestamp': datetime.now(),
                    'success': True
                }
                
                self.function_timings[func.__name__].append(timing_data)
                
                return result
                
            except Exception as e:
                # 记录异常
                end_time = time.time()
                timing_data = {
                    'function': func.__name__,
                    'execution_time': end_time - start_time,
                    'timestamp': datetime.now(),
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                
                self.function_timings[func.__name__].append(timing_data)
                raise
        
        return wrapper
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'timestamp': datetime.now(),
            'function_performance': {},
            'summary': {}
        }
        
        total_functions = 0
        total_calls = 0
        total_time = 0
        
        for func_name, timings in self.function_timings.items():
            if not timings:
                continue
            
            successful_calls = [t for t in timings if t.get('success', True)]
            failed_calls = [t for t in timings if not t.get('success', True)]
            
            if successful_calls:
                execution_times = [t['execution_time'] for t in successful_calls]
                memory_deltas = [t.get('memory_delta', 0) for t in successful_calls]
                
                func_stats = {
                    'total_calls': len(timings),
                    'successful_calls': len(successful_calls),
                    'failed_calls': len(failed_calls),
                    'avg_execution_time': np.mean(execution_times),
                    'max_execution_time': np.max(execution_times),
                    'min_execution_time': np.min(execution_times),
                    'total_execution_time': np.sum(execution_times),
                    'avg_memory_delta': np.mean(memory_deltas),
                    'max_memory_delta': np.max(memory_deltas)
                }
                
                report['function_performance'][func_name] = func_stats
                
                total_functions += 1
                total_calls += len(timings)
                total_time += func_stats['total_execution_time']
        
        # 汇总统计
        report['summary'] = {
            'total_functions_profiled': total_functions,
            'total_function_calls': total_calls,
            'total_execution_time': total_time,
            'avg_time_per_call': total_time / total_calls if total_calls > 0 else 0
        }
        
        return report
    
    def identify_bottlenecks(self, threshold_time: float = 1.0) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        for func_name, timings in self.function_timings.items():
            successful_calls = [t for t in timings if t.get('success', True)]
            
            if not successful_calls:
                continue
            
            execution_times = [t['execution_time'] for t in successful_calls]
            avg_time = np.mean(execution_times)
            max_time = np.max(execution_times)
            
            # 识别瓶颈
            if avg_time > threshold_time:
                bottlenecks.append({
                    'function': func_name,
                    'type': 'slow_average',
                    'avg_time': avg_time,
                    'max_time': max_time,
                    'call_count': len(successful_calls),
                    'severity': 'high' if avg_time > threshold_time * 2 else 'medium'
                })
            
            elif max_time > threshold_time * 5:
                bottlenecks.append({
                    'function': func_name,
                    'type': 'slow_peak',
                    'avg_time': avg_time,
                    'max_time': max_time,
                    'call_count': len(successful_calls),
                    'severity': 'medium'
                })
        
        # 按严重程度排序
        bottlenecks.sort(key=lambda x: (x['severity'] == 'high', x['avg_time']), reverse=True)
        
        return bottlenecks


class ErrorAnalyzer:
    """错误分析器"""
    
    def __init__(self):
        self.error_log = []
        self.error_patterns = defaultdict(int)
        self.error_statistics = {}
        
    def log_error(self, 
                  error: Exception, 
                  context: Optional[Dict[str, Any]] = None,
                  function_name: Optional[str] = None):
        """记录错误"""
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'function_name': function_name or self._get_calling_function(),
            'context': context or {}
        }
        
        self.error_log.append(error_info)
        
        # 更新错误模式统计
        error_pattern = f"{error_info['error_type']}:{error_info['function_name']}"
        self.error_patterns[error_pattern] += 1
        
        # 更新错误统计
        self._update_error_statistics()
    
    def _get_calling_function(self) -> str:
        """获取调用函数名"""
        frame = inspect.currentframe()
        try:
            # 向上查找调用栈
            caller_frame = frame.f_back.f_back
            if caller_frame:
                return caller_frame.f_code.co_name
        finally:
            del frame
        return 'unknown'
    
    def _update_error_statistics(self):
        """更新错误统计"""
        if not self.error_log:
            return
        
        # 按错误类型统计
        error_types = defaultdict(int)
        function_errors = defaultdict(int)
        recent_errors = []
        
        # 最近24小时的错误
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for error in self.error_log:
            error_types[error['error_type']] += 1
            function_errors[error['function_name']] += 1
            
            if error['timestamp'] > cutoff_time:
                recent_errors.append(error)
        
        self.error_statistics = {
            'total_errors': len(self.error_log),
            'recent_errors_24h': len(recent_errors),
            'error_types': dict(error_types),
            'function_errors': dict(function_errors),
            'most_common_patterns': dict(sorted(self.error_patterns.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)[:10])
        }
    
    def analyze_error_trends(self, hours: int = 24) -> Dict[str, Any]:
        """分析错误趋势"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_log if e['timestamp'] > cutoff_time]
        
        if not recent_errors:
            return {'message': 'No recent errors to analyze'}
        
        # 按小时分组
        hourly_counts = defaultdict(int)
        error_type_trends = defaultdict(lambda: defaultdict(int))
        
        for error in recent_errors:
            hour_key = error['timestamp'].strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] += 1
            error_type_trends[error['error_type']][hour_key] += 1
        
        # 计算趋势
        error_counts = list(hourly_counts.values())
        if len(error_counts) > 1:
            trend_slope = np.polyfit(range(len(error_counts)), error_counts, 1)[0]
            trend = 'increasing' if trend_slope > 0.1 else 'decreasing' if trend_slope < -0.1 else 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'analysis_period_hours': hours,
            'total_recent_errors': len(recent_errors),
            'hourly_distribution': dict(hourly_counts),
            'error_type_trends': {k: dict(v) for k, v in error_type_trends.items()},
            'overall_trend': trend,
            'avg_errors_per_hour': len(recent_errors) / hours
        }
    
    def get_error_report(self) -> Dict[str, Any]:
        """获取错误报告"""
        return {
            'timestamp': datetime.now(),
            'error_statistics': self.error_statistics.copy(),
            'recent_errors': self.error_log[-10:],  # 最近10个错误
            'error_trends': self.analyze_error_trends(),
            'recommendations': self._generate_error_recommendations()
        }
    
    def _generate_error_recommendations(self) -> List[str]:
        """生成错误处理建议"""
        recommendations = []
        
        if not self.error_statistics:
            return recommendations
        
        # 基于错误频率的建议
        if self.error_statistics.get('recent_errors_24h', 0) > 10:
            recommendations.append("错误频率较高，建议检查系统稳定性")
        
        # 基于错误类型的建议
        common_types = self.error_statistics.get('error_types', {})
        
        if 'KeyError' in common_types and common_types['KeyError'] > 3:
            recommendations.append("KeyError频发，建议检查数据完整性和字段访问逻辑")
        
        if 'ValueError' in common_types and common_types['ValueError'] > 3:
            recommendations.append("ValueError频发，建议加强输入验证")
        
        if 'ConnectionError' in common_types:
            recommendations.append("网络连接错误，建议检查网络稳定性和重试机制")
        
        if 'TimeoutError' in common_types:
            recommendations.append("超时错误，建议优化性能或调整超时设置")
        
        # 基于错误模式的建议
        common_patterns = self.error_statistics.get('most_common_patterns', {})
        if len(common_patterns) > 0:
            most_common = list(common_patterns.keys())[0]
            recommendations.append(f"最常见错误模式: {most_common}，建议重点关注")
        
        return recommendations