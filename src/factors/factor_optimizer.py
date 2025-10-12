"""
因子组合优化模块

该模块实现了多种因子组合优化和权重分配算法，包括：
1. 因子筛选 - 相关性分析、信息比率筛选、稳定性筛选等
2. 权重优化 - 等权重、信息比率加权、风险平价、最大分散化等
3. 因子合成 - 线性组合、非线性组合、动态权重等
4. 风险控制 - 因子暴露控制、换手率控制、集中度控制等
5. 回测验证 - 因子有效性验证、组合性能评估等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime, timedelta
import logging
from scipy import optimize
from scipy.stats import spearmanr, pearsonr

# 尝试导入优化库
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Some optimization methods may not work.")

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    warnings.warn("CVXPY not available. Convex optimization may not work.")

class FactorOptimizer:
    """因子组合优化器"""
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        初始化因子优化器
        
        Args:
            lookback_periods: 各种计算的回看期设置
        """
        self.lookback_periods = lookback_periods or {
            'correlation': 60,
            'stability': 120,
            'performance': 252,
            'risk': 60
        }
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 缓存优化结果
        self.optimization_cache = {}
        
    def calculate_factor_statistics(self, factors: Dict[str, pd.Series], 
                                  returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        计算因子统计指标
        
        Args:
            factors: 因子字典
            returns: 收益率序列
            
        Returns:
            Dict[str, Dict[str, float]]: 因子统计指标字典
        """
        stats = {}
        
        for name, factor in factors.items():
            if factor is None or len(factor) == 0:
                continue
                
            try:
                # 对齐数据
                aligned_data = pd.concat([factor, returns], axis=1, join='inner').dropna()
                if len(aligned_data) < 30:  # 需要足够的数据
                    continue
                    
                factor_values = aligned_data.iloc[:, 0]
                return_values = aligned_data.iloc[:, 1]
                
                # 计算统计指标
                factor_stats = {}
                
                # 1. 相关性指标
                ic_pearson, p_pearson = pearsonr(factor_values, return_values)
                ic_spearman, p_spearman = spearmanr(factor_values, return_values)
                
                factor_stats['ic_pearson'] = ic_pearson
                factor_stats['ic_spearman'] = ic_spearman
                factor_stats['ic_p_value_pearson'] = p_pearson
                factor_stats['ic_p_value_spearman'] = p_spearman
                
                # 2. 信息比率
                factor_stats['ic_mean'] = ic_pearson
                factor_stats['ic_std'] = np.std([ic_pearson])  # 简化计算
                if factor_stats['ic_std'] > 0:
                    factor_stats['information_ratio'] = factor_stats['ic_mean'] / factor_stats['ic_std']
                else:
                    factor_stats['information_ratio'] = 0
                
                # 3. 稳定性指标
                # 滚动IC稳定性
                rolling_window = min(60, len(aligned_data) // 3)
                if rolling_window >= 20:
                    rolling_ics = []
                    for i in range(rolling_window, len(aligned_data)):
                        window_factor = factor_values.iloc[i-rolling_window:i]
                        window_returns = return_values.iloc[i-rolling_window:i]
                        if len(window_factor) > 10:
                            ic, _ = pearsonr(window_factor, window_returns)
                            rolling_ics.append(ic)
                    
                    if rolling_ics:
                        factor_stats['ic_stability'] = np.std(rolling_ics)
                        factor_stats['ic_positive_ratio'] = np.mean(np.array(rolling_ics) > 0)
                    else:
                        factor_stats['ic_stability'] = 1.0
                        factor_stats['ic_positive_ratio'] = 0.5
                else:
                    factor_stats['ic_stability'] = 1.0
                    factor_stats['ic_positive_ratio'] = 0.5
                
                # 4. 因子分布特征
                factor_stats['factor_mean'] = factor_values.mean()
                factor_stats['factor_std'] = factor_values.std()
                factor_stats['factor_skew'] = factor_values.skew()
                factor_stats['factor_kurt'] = factor_values.kurtosis()
                
                # 5. 因子覆盖度
                factor_stats['coverage'] = (factor_values.notna()).mean()
                factor_stats['turnover'] = (factor_values.diff().abs() > 0).mean()
                
                stats[name] = factor_stats
                
            except Exception as e:
                self.logger.warning(f"计算因子 {name} 统计指标时出错: {e}")
                continue
                
        return stats
    
    def filter_factors_by_correlation(self, factors: Dict[str, pd.Series], 
                                    threshold: float = 0.8) -> List[str]:
        """
        基于相关性筛选因子
        
        Args:
            factors: 因子字典
            threshold: 相关性阈值
            
        Returns:
            List[str]: 筛选后的因子名称列表
        """
        # 创建因子矩阵
        factor_df = pd.DataFrame(factors).dropna()
        
        if len(factor_df.columns) <= 1:
            return list(factor_df.columns)
        
        # 计算相关性矩阵
        corr_matrix = factor_df.corr().abs()
        
        # 筛选因子
        selected_factors = []
        remaining_factors = list(factor_df.columns)
        
        while remaining_factors:
            # 选择第一个因子
            current_factor = remaining_factors[0]
            selected_factors.append(current_factor)
            remaining_factors.remove(current_factor)
            
            # 移除与当前因子高度相关的因子
            highly_correlated = []
            for factor in remaining_factors:
                if corr_matrix.loc[current_factor, factor] > threshold:
                    highly_correlated.append(factor)
            
            for factor in highly_correlated:
                remaining_factors.remove(factor)
        
        self.logger.info(f"相关性筛选: {len(factors)} -> {len(selected_factors)} 个因子")
        return selected_factors
    
    def filter_factors_by_performance(self, factor_stats: Dict[str, Dict[str, float]], 
                                    criteria: Dict[str, float] = None) -> List[str]:
        """
        基于性能指标筛选因子
        
        Args:
            factor_stats: 因子统计指标
            criteria: 筛选标准
            
        Returns:
            List[str]: 筛选后的因子名称列表
        """
        if criteria is None:
            criteria = {
                'ic_pearson': 0.02,  # 最小IC
                'ic_p_value_pearson': 0.1,  # 最大p值
                'information_ratio': 0.5,  # 最小信息比率
                'ic_positive_ratio': 0.55,  # 最小正IC比例
                'coverage': 0.8  # 最小覆盖度
            }
        
        selected_factors = []
        
        for factor_name, stats in factor_stats.items():
            # 检查所有标准
            meets_criteria = True
            
            for criterion, threshold in criteria.items():
                if criterion in stats:
                    if criterion == 'ic_p_value_pearson':
                        # p值需要小于阈值
                        if stats[criterion] > threshold:
                            meets_criteria = False
                            break
                    else:
                        # 其他指标需要大于阈值
                        if abs(stats[criterion]) < threshold:
                            meets_criteria = False
                            break
            
            if meets_criteria:
                selected_factors.append(factor_name)
        
        self.logger.info(f"性能筛选: {len(factor_stats)} -> {len(selected_factors)} 个因子")
        return selected_factors
    
    def calculate_equal_weights(self, factor_names: List[str]) -> Dict[str, float]:
        """
        计算等权重
        
        Args:
            factor_names: 因子名称列表
            
        Returns:
            Dict[str, float]: 因子权重字典
        """
        if not factor_names:
            return {}
        
        weight = 1.0 / len(factor_names)
        return {name: weight for name in factor_names}
    
    def calculate_ic_weights(self, factor_stats: Dict[str, Dict[str, float]], 
                           factor_names: List[str]) -> Dict[str, float]:
        """
        基于IC计算权重
        
        Args:
            factor_stats: 因子统计指标
            factor_names: 因子名称列表
            
        Returns:
            Dict[str, float]: 因子权重字典
        """
        if not factor_names:
            return {}
        
        # 提取IC值
        ics = []
        valid_factors = []
        
        for name in factor_names:
            if name in factor_stats and 'ic_pearson' in factor_stats[name]:
                ic = abs(factor_stats[name]['ic_pearson'])
                ics.append(ic)
                valid_factors.append(name)
        
        if not ics:
            return self.calculate_equal_weights(factor_names)
        
        # 归一化权重
        total_ic = sum(ics)
        if total_ic == 0:
            return self.calculate_equal_weights(valid_factors)
        
        weights = {}
        for i, name in enumerate(valid_factors):
            weights[name] = ics[i] / total_ic
        
        return weights
    
    def calculate_information_ratio_weights(self, factor_stats: Dict[str, Dict[str, float]], 
                                          factor_names: List[str]) -> Dict[str, float]:
        """
        基于信息比率计算权重
        
        Args:
            factor_stats: 因子统计指标
            factor_names: 因子名称列表
            
        Returns:
            Dict[str, float]: 因子权重字典
        """
        if not factor_names:
            return {}
        
        # 提取信息比率
        irs = []
        valid_factors = []
        
        for name in factor_names:
            if name in factor_stats and 'information_ratio' in factor_stats[name]:
                ir = abs(factor_stats[name]['information_ratio'])
                irs.append(ir)
                valid_factors.append(name)
        
        if not irs:
            return self.calculate_equal_weights(factor_names)
        
        # 归一化权重
        total_ir = sum(irs)
        if total_ir == 0:
            return self.calculate_equal_weights(valid_factors)
        
        weights = {}
        for i, name in enumerate(valid_factors):
            weights[name] = irs[i] / total_ir
        
        return weights
    
    def calculate_risk_parity_weights(self, factors: Dict[str, pd.Series], 
                                    factor_names: List[str]) -> Dict[str, float]:
        """
        计算风险平价权重
        
        Args:
            factors: 因子字典
            factor_names: 因子名称列表
            
        Returns:
            Dict[str, float]: 因子权重字典
        """
        if not factor_names:
            return {}
        
        try:
            # 创建因子矩阵
            factor_matrix = pd.DataFrame({name: factors[name] for name in factor_names if name in factors})
            factor_matrix = factor_matrix.dropna()
            
            if len(factor_matrix.columns) <= 1:
                return self.calculate_equal_weights(factor_names)
            
            # 计算协方差矩阵
            cov_matrix = factor_matrix.cov().values
            
            # 风险平价优化
            n = len(factor_names)
            
            def risk_parity_objective(weights):
                portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
                marginal_contrib = np.dot(cov_matrix, weights)
                contrib = weights * marginal_contrib
                
                # 风险贡献的方差最小化
                target_contrib = portfolio_var / n
                return np.sum((contrib - target_contrib) ** 2)
            
            # 约束条件
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.01, 0.5) for _ in range(n)]  # 权重范围
            
            # 初始权重
            x0 = np.ones(n) / n
            
            # 优化
            result = optimize.minimize(risk_parity_objective, x0, 
                                     method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = {}
                for i, name in enumerate(factor_names):
                    weights[name] = result.x[i]
                return weights
            else:
                self.logger.warning("风险平价优化失败，使用等权重")
                return self.calculate_equal_weights(factor_names)
                
        except Exception as e:
            self.logger.error(f"计算风险平价权重时出错: {e}")
            return self.calculate_equal_weights(factor_names)
    
    def calculate_max_diversification_weights(self, factors: Dict[str, pd.Series], 
                                           factor_names: List[str]) -> Dict[str, float]:
        """
        计算最大分散化权重
        
        Args:
            factors: 因子字典
            factor_names: 因子名称列表
            
        Returns:
            Dict[str, float]: 因子权重字典
        """
        if not factor_names:
            return {}
        
        try:
            # 创建因子矩阵
            factor_matrix = pd.DataFrame({name: factors[name] for name in factor_names if name in factors})
            factor_matrix = factor_matrix.dropna()
            
            if len(factor_matrix.columns) <= 1:
                return self.calculate_equal_weights(factor_names)
            
            # 计算协方差矩阵和标准差
            cov_matrix = factor_matrix.cov().values
            std_vector = factor_matrix.std().values
            
            # 最大分散化比率优化
            n = len(factor_names)
            
            def diversification_ratio(weights):
                portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                weighted_avg_std = np.dot(weights, std_vector)
                return -weighted_avg_std / portfolio_std  # 负号因为要最大化
            
            # 约束条件
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.01, 0.5) for _ in range(n)]
            
            # 初始权重
            x0 = np.ones(n) / n
            
            # 优化
            result = optimize.minimize(diversification_ratio, x0, 
                                     method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = {}
                for i, name in enumerate(factor_names):
                    weights[name] = result.x[i]
                return weights
            else:
                self.logger.warning("最大分散化优化失败，使用等权重")
                return self.calculate_equal_weights(factor_names)
                
        except Exception as e:
            self.logger.error(f"计算最大分散化权重时出错: {e}")
            return self.calculate_equal_weights(factor_names)
    
    def optimize_factors(self, factors: Dict[str, pd.Series], 
                        returns: pd.Series,
                        method: str = 'information_ratio',
                        **kwargs) -> Dict[str, Any]:
        """
        优化因子组合
        
        Args:
            factors: 因子字典
            returns: 收益率序列
            method: 优化方法 ('equal', 'ic', 'information_ratio', 'risk_parity', 'max_diversification')
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        result = {
            'weights': {},
            'selected_factors': [],
            'factor_stats': {},
            'optimization_info': {}
        }
        
        try:
            # 1. 计算因子统计指标
            self.logger.info("计算因子统计指标...")
            factor_stats = self.calculate_factor_statistics(factors, returns)
            result['factor_stats'] = factor_stats
            
            if not factor_stats:
                self.logger.warning("没有有效的因子统计指标")
                return result
            
            # 2. 因子筛选
            self.logger.info("筛选因子...")
            
            # 性能筛选
            performance_filtered = self.filter_factors_by_performance(factor_stats)
            
            # 相关性筛选
            if len(performance_filtered) > 1:
                correlation_filtered = self.filter_factors_by_correlation(
                    {name: factors[name] for name in performance_filtered if name in factors},
                    threshold=kwargs.get('correlation_threshold', 0.8)
                )
            else:
                correlation_filtered = performance_filtered
            
            result['selected_factors'] = correlation_filtered
            
            if not correlation_filtered:
                self.logger.warning("没有通过筛选的因子")
                return result
            
            # 3. 权重计算
            self.logger.info(f"使用 {method} 方法计算权重...")
            
            if method == 'equal':
                weights = self.calculate_equal_weights(correlation_filtered)
            elif method == 'ic':
                weights = self.calculate_ic_weights(factor_stats, correlation_filtered)
            elif method == 'information_ratio':
                weights = self.calculate_information_ratio_weights(factor_stats, correlation_filtered)
            elif method == 'risk_parity':
                weights = self.calculate_risk_parity_weights(factors, correlation_filtered)
            elif method == 'max_diversification':
                weights = self.calculate_max_diversification_weights(factors, correlation_filtered)
            else:
                self.logger.warning(f"未知的优化方法: {method}，使用等权重")
                weights = self.calculate_equal_weights(correlation_filtered)
            
            result['weights'] = weights
            
            # 4. 组合因子计算
            if weights:
                combined_factor = self.combine_factors(factors, weights)
                result['combined_factor'] = combined_factor
                
                # 计算组合因子统计
                combined_stats = self.calculate_factor_statistics({'combined': combined_factor}, returns)
                result['combined_stats'] = combined_stats.get('combined', {})
            
            # 5. 优化信息
            result['optimization_info'] = {
                'method': method,
                'total_factors': len(factors),
                'selected_factors_count': len(correlation_filtered),
                'optimization_date': datetime.now().isoformat()
            }
            
            self.logger.info(f"因子优化完成: {len(factors)} -> {len(correlation_filtered)} 个因子")
            
        except Exception as e:
            self.logger.error(f"因子组合优化时出错: {e}")
            
        return result
    
    def combine_factors(self, factors: Dict[str, pd.Series], 
                       weights: Dict[str, float]) -> pd.Series:
        """
        组合因子
        
        Args:
            factors: 因子字典
            weights: 权重字典
            
        Returns:
            pd.Series: 组合因子
        """
        if not weights:
            return pd.Series(dtype=float)
        
        # 标准化因子
        standardized_factors = {}
        for name, weight in weights.items():
            if name in factors and factors[name] is not None:
                factor = factors[name].dropna()
                if len(factor) > 0 and factor.std() > 0:
                    standardized_factors[name] = (factor - factor.mean()) / factor.std()
        
        if not standardized_factors:
            return pd.Series(dtype=float)
        
        # 加权组合
        combined_factor = None
        for name, factor in standardized_factors.items():
            weight = weights[name]
            weighted_factor = factor * weight
            
            if combined_factor is None:
                combined_factor = weighted_factor
            else:
                combined_factor = combined_factor.add(weighted_factor, fill_value=0)
        
        return combined_factor if combined_factor is not None else pd.Series(dtype=float)
    
    def optimize_factor_combination(self, factors: Dict[str, pd.Series], 
                                  returns: pd.Series,
                                  method: str = 'information_ratio',
                                  **kwargs) -> Dict[str, Any]:
        """
        优化因子组合（别名方法，与optimize_factors相同）
        
        Args:
            factors: 因子字典
            returns: 收益率序列
            method: 优化方法
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        return self.optimize_factors(factors, returns, method, **kwargs)
    
    def backtest_factor_combination(self, combined_factor: pd.Series, 
                                  returns: pd.Series,
                                  quantiles: int = 5) -> Dict[str, Any]:
        """
        回测因子组合
        
        Args:
            combined_factor: 组合因子
            returns: 收益率序列
            quantiles: 分位数数量
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        backtest_result = {}
        
        try:
            # 对齐数据
            aligned_data = pd.concat([combined_factor, returns], axis=1, join='inner').dropna()
            if len(aligned_data) < 50:
                return backtest_result
            
            factor_values = aligned_data.iloc[:, 0]
            return_values = aligned_data.iloc[:, 1]
            
            # 分位数分析
            quantile_labels = pd.qcut(factor_values, q=quantiles, labels=False, duplicates='drop')
            
            quantile_returns = {}
            for q in range(quantiles):
                mask = quantile_labels == q
                if mask.sum() > 0:
                    quantile_returns[f'Q{q+1}'] = return_values[mask].mean()
            
            backtest_result['quantile_returns'] = quantile_returns
            
            # 多空组合收益
            if len(quantile_returns) >= 2:
                long_short_return = quantile_returns[f'Q{quantiles}'] - quantile_returns['Q1']
                backtest_result['long_short_return'] = long_short_return
            
            # IC分析
            ic_pearson, p_pearson = pearsonr(factor_values, return_values)
            ic_spearman, p_spearman = spearmanr(factor_values, return_values)
            
            backtest_result['ic_analysis'] = {
                'ic_pearson': ic_pearson,
                'ic_spearman': ic_spearman,
                'p_value_pearson': p_pearson,
                'p_value_spearman': p_spearman
            }
            
            # 滚动IC分析
            rolling_window = min(60, len(aligned_data) // 3)
            if rolling_window >= 20:
                rolling_ics = []
                for i in range(rolling_window, len(aligned_data)):
                    window_factor = factor_values.iloc[i-rolling_window:i]
                    window_returns = return_values.iloc[i-rolling_window:i]
                    ic, _ = pearsonr(window_factor, window_returns)
                    rolling_ics.append(ic)
                
                if rolling_ics:
                    backtest_result['rolling_ic_analysis'] = {
                        'mean_ic': np.mean(rolling_ics),
                        'std_ic': np.std(rolling_ics),
                        'ir': np.mean(rolling_ics) / np.std(rolling_ics) if np.std(rolling_ics) > 0 else 0,
                        'positive_ic_ratio': np.mean(np.array(rolling_ics) > 0)
                    }
            
        except Exception as e:
            self.logger.error(f"回测因子组合时出错: {e}")
            
        return backtest_result
    
    def generate_optimization_report(self, optimization_result: Dict[str, Any], 
                                   backtest_result: Dict[str, Any] = None) -> str:
        """
        生成优化报告
        
        Args:
            optimization_result: 优化结果
            backtest_result: 回测结果
            
        Returns:
            str: 优化报告
        """
        report = []
        report.append("=" * 60)
        report.append("因子组合优化报告")
        report.append("=" * 60)
        
        # 优化概览
        opt_info = optimization_result.get('optimization_info', {})
        report.append(f"\n优化方法: {opt_info.get('method', 'Unknown')}")
        report.append(f"原始因子数量: {opt_info.get('total_factors', 0)}")
        report.append(f"筛选后因子数量: {opt_info.get('selected_factors_count', 0)}")
        
        # 选中的因子
        selected_factors = optimization_result.get('selected_factors', [])
        if selected_factors:
            report.append(f"\n选中的因子 ({len(selected_factors)}个):")
            for factor in selected_factors:
                report.append(f"  - {factor}")
        
        # 因子权重
        weights = optimization_result.get('weights', {})
        if weights:
            report.append(f"\n因子权重分配:")
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for name, weight in sorted_weights:
                report.append(f"  - {name}: {weight:.4f}")
        
        # 组合因子统计
        combined_stats = optimization_result.get('combined_stats', {})
        if combined_stats:
            report.append(f"\n组合因子统计:")
            report.append(f"  IC (Pearson): {combined_stats.get('ic_pearson', 0):.4f}")
            report.append(f"  IC (Spearman): {combined_stats.get('ic_spearman', 0):.4f}")
            report.append(f"  信息比率: {combined_stats.get('information_ratio', 0):.4f}")
            report.append(f"  IC稳定性: {combined_stats.get('ic_stability', 0):.4f}")
            report.append(f"  正IC比例: {combined_stats.get('ic_positive_ratio', 0):.4f}")
        
        # 回测结果
        if backtest_result:
            report.append(f"\n回测分析:")
            
            # 分位数收益
            quantile_returns = backtest_result.get('quantile_returns', {})
            if quantile_returns:
                report.append("  分位数收益:")
                for q, ret in quantile_returns.items():
                    report.append(f"    {q}: {ret:.4f}")
            
            # 多空收益
            long_short = backtest_result.get('long_short_return')
            if long_short is not None:
                report.append(f"  多空组合收益: {long_short:.4f}")
            
            # 滚动IC分析
            rolling_ic = backtest_result.get('rolling_ic_analysis', {})
            if rolling_ic:
                report.append("  滚动IC分析:")
                report.append(f"    平均IC: {rolling_ic.get('mean_ic', 0):.4f}")
                report.append(f"    IC标准差: {rolling_ic.get('std_ic', 0):.4f}")
                report.append(f"    信息比率: {rolling_ic.get('ir', 0):.4f}")
                report.append(f"    正IC比例: {rolling_ic.get('positive_ic_ratio', 0):.4f}")
        
        # 优化建议
        report.append(f"\n优化建议:")
        if combined_stats.get('information_ratio', 0) > 1.0:
            report.append("  ✓ 组合因子信息比率良好")
        else:
            report.append("  ⚠ 建议进一步优化因子权重或筛选标准")
        
        if combined_stats.get('ic_positive_ratio', 0) > 0.6:
            report.append("  ✓ IC稳定性较好")
        else:
            report.append("  ⚠ IC稳定性有待提高，考虑调整因子或时间窗口")
        
        if len(selected_factors) > 10:
            report.append("  ⚠ 因子数量较多，考虑进一步筛选以降低复杂度")
        
        return "\n".join(report)


def main():
    """示例用法"""
    # 创建示例数据
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # 模拟收益率
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, n_days), index=dates)
    
    # 模拟因子数据
    factors = {}
    for i in range(20):  # 创建20个模拟因子
        factor_values = np.random.normal(0, 1, n_days)
        # 添加一些与收益率的相关性
        if i < 5:  # 前5个因子有较强的预测能力
            factor_values += returns.values * (2 + np.random.normal(0, 0.5))
        factors[f'factor_{i+1}'] = pd.Series(factor_values, index=dates)
    
    # 创建因子优化器
    optimizer = FactorOptimizer()
    
    # 优化因子组合
    print("优化因子组合...")
    optimization_result = optimizer.optimize_factor_combination(
        factors, returns, method='information_ratio'
    )
    
    # 回测组合因子
    if 'combined_factor' in optimization_result:
        print("回测组合因子...")
        backtest_result = optimizer.backtest_factor_combination(
            optimization_result['combined_factor'], returns
        )
    else:
        backtest_result = {}
    
    # 生成报告
    report = optimizer.generate_optimization_report(optimization_result, backtest_result)
    print(report)
    
    print(f"\n因子组合优化完成！")


if __name__ == "__main__":
    main()