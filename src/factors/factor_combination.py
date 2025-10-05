"""
因子组合策略模块

实现择时和择股因子的有机结合：
1. 多层次因子组合框架
2. 动态权重分配机制
3. 风险预算管理
4. 因子轮动策略
5. 自适应调整机制
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorCombinationStrategy:
    """
    因子组合策略类
    
    整合择时和择股因子，构建综合投资策略
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化因子组合策略
        
        Args:
            config: 配置参数字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self.timing_weight = 0.3  # 择时因子权重
        self.selection_weight = 0.7  # 择股因子权重
        self.rebalance_frequency = 'monthly'  # 调仓频率
        self.risk_budget = 0.15  # 风险预算
        
        # 历史数据存储
        self.factor_history = {}
        self.performance_history = {}
        self.weight_history = {}
    
    def build_integrated_strategy(self, timing_factors: Dict[str, pd.Series],
                                selection_factors: Dict[str, pd.DataFrame],
                                market_data: pd.DataFrame,
                                universe: List[str]) -> Dict[str, Any]:
        """
        构建整合策略
        
        Args:
            timing_factors: 择时因子字典
            selection_factors: 择股因子字典
            market_data: 市场数据
            universe: 股票池
            
        Returns:
            Dict[str, Any]: 整合策略结果
        """
        try:
            # 1. 因子预处理和对齐
            aligned_factors = self._align_factors(timing_factors, selection_factors, market_data)
            
            # 2. 构建多层次因子框架
            factor_framework = self._build_factor_framework(aligned_factors)
            
            # 3. 动态权重分配
            dynamic_weights = self._calculate_dynamic_weights(factor_framework, market_data)
            
            # 4. 风险预算管理
            risk_adjusted_weights = self._apply_risk_budget(dynamic_weights, market_data)
            
            # 5. 生成投资组合
            portfolio = self._generate_portfolio(risk_adjusted_weights, universe, market_data)
            
            # 6. 策略评估
            strategy_evaluation = self._evaluate_strategy(portfolio, market_data)
            
            # 7. 因子轮动检测
            rotation_signals = self._detect_factor_rotation(factor_framework, market_data)
            
            result = {
                'aligned_factors': aligned_factors,
                'factor_framework': factor_framework,
                'dynamic_weights': dynamic_weights,
                'risk_adjusted_weights': risk_adjusted_weights,
                'portfolio': portfolio,
                'strategy_evaluation': strategy_evaluation,
                'rotation_signals': rotation_signals,
                'rebalance_dates': self._get_rebalance_dates(market_data.index)
            }
            
            # 更新历史记录
            self._update_history(result)
            
            self.logger.info("整合策略构建完成")
            return result
            
        except Exception as e:
            self.logger.error(f"构建整合策略时出错: {str(e)}")
            return {}
    
    def adaptive_factor_weighting(self, factor_performance: Dict[str, pd.Series],
                                market_regime: pd.Series) -> Dict[str, pd.Series]:
        """
        自适应因子权重调整
        
        Args:
            factor_performance: 因子历史表现
            market_regime: 市场状态序列
            
        Returns:
            Dict[str, pd.Series]: 动态因子权重
        """
        try:
            adaptive_weights = {}
            
            # 1. 基于市场状态的权重调整
            regime_weights = self._regime_based_weighting(factor_performance, market_regime)
            
            # 2. 基于因子动量的权重调整
            momentum_weights = self._momentum_based_weighting(factor_performance)
            
            # 3. 基于风险调整的权重
            risk_weights = self._risk_adjusted_weighting(factor_performance)
            
            # 4. 组合权重
            for factor_name in factor_performance.keys():
                regime_w = regime_weights.get(factor_name, 0.33)
                momentum_w = momentum_weights.get(factor_name, 0.33)
                risk_w = risk_weights.get(factor_name, 0.34)
                
                adaptive_weights[factor_name] = (
                    0.4 * regime_w + 0.3 * momentum_w + 0.3 * risk_w
                )
            
            self.logger.info("自适应权重调整完成")
            return adaptive_weights
            
        except Exception as e:
            self.logger.error(f"自适应权重调整时出错: {str(e)}")
            return {}
    
    def factor_rotation_strategy(self, factor_data: Dict[str, pd.DataFrame],
                               lookback_window: int = 60,
                               rotation_threshold: float = 0.1) -> Dict[str, Any]:
        """
        因子轮动策略
        
        Args:
            factor_data: 因子数据字典
            lookback_window: 回望窗口
            rotation_threshold: 轮动阈值
            
        Returns:
            Dict[str, Any]: 轮动策略结果
        """
        try:
            rotation_results = {}
            
            # 1. 计算因子动量
            factor_momentum = self._calculate_factor_momentum(factor_data, lookback_window)
            
            # 2. 识别轮动信号
            rotation_signals = self._identify_rotation_signals(factor_momentum, rotation_threshold)
            
            # 3. 构建轮动组合
            rotation_portfolio = self._build_rotation_portfolio(rotation_signals, factor_data)
            
            # 4. 风险控制
            risk_controlled_portfolio = self._apply_rotation_risk_control(rotation_portfolio)
            
            # 5. 轮动策略评估
            rotation_evaluation = self._evaluate_rotation_strategy(risk_controlled_portfolio)
            
            rotation_results = {
                'factor_momentum': factor_momentum,
                'rotation_signals': rotation_signals,
                'rotation_portfolio': rotation_portfolio,
                'risk_controlled_portfolio': risk_controlled_portfolio,
                'rotation_evaluation': rotation_evaluation
            }
            
            self.logger.info("因子轮动策略构建完成")
            return rotation_results
            
        except Exception as e:
            self.logger.error(f"构建因子轮动策略时出错: {str(e)}")
            return {}
    
    def multi_timeframe_integration(self, short_term_factors: Dict[str, pd.DataFrame],
                                  medium_term_factors: Dict[str, pd.DataFrame],
                                  long_term_factors: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        多时间框架因子整合
        
        Args:
            short_term_factors: 短期因子
            medium_term_factors: 中期因子
            long_term_factors: 长期因子
            
        Returns:
            Dict[str, Any]: 多时间框架整合结果
        """
        try:
            # 1. 时间框架权重
            timeframe_weights = {
                'short_term': 0.2,
                'medium_term': 0.5,
                'long_term': 0.3
            }
            
            # 2. 因子对齐和标准化
            aligned_short = self._standardize_factors(short_term_factors)
            aligned_medium = self._standardize_factors(medium_term_factors)
            aligned_long = self._standardize_factors(long_term_factors)
            
            # 3. 多时间框架合成
            integrated_factors = {}
            
            # 获取所有因子名称
            all_factors = set()
            all_factors.update(aligned_short.keys())
            all_factors.update(aligned_medium.keys())
            all_factors.update(aligned_long.keys())
            
            for factor_name in all_factors:
                factor_components = []
                weights = []
                
                if factor_name in aligned_short:
                    factor_components.append(aligned_short[factor_name])
                    weights.append(timeframe_weights['short_term'])
                
                if factor_name in aligned_medium:
                    factor_components.append(aligned_medium[factor_name])
                    weights.append(timeframe_weights['medium_term'])
                
                if factor_name in aligned_long:
                    factor_components.append(aligned_long[factor_name])
                    weights.append(timeframe_weights['long_term'])
                
                # 加权合成
                if factor_components:
                    # 对齐时间序列
                    aligned_components = pd.concat(factor_components, axis=1).fillna(0)
                    
                    # 加权平均
                    weight_sum = sum(weights)
                    integrated_factor = sum(w/weight_sum * aligned_components.iloc[:, i] 
                                          for i, w in enumerate(weights))
                    
                    integrated_factors[factor_name] = integrated_factor
            
            # 4. 时间框架一致性检验
            consistency_check = self._check_timeframe_consistency(
                aligned_short, aligned_medium, aligned_long
            )
            
            # 5. 动态时间框架权重
            dynamic_timeframe_weights = self._calculate_dynamic_timeframe_weights(
                aligned_short, aligned_medium, aligned_long
            )
            
            result = {
                'integrated_factors': integrated_factors,
                'timeframe_weights': timeframe_weights,
                'dynamic_timeframe_weights': dynamic_timeframe_weights,
                'consistency_check': consistency_check,
                'aligned_factors': {
                    'short_term': aligned_short,
                    'medium_term': aligned_medium,
                    'long_term': aligned_long
                }
            }
            
            self.logger.info("多时间框架整合完成")
            return result
            
        except Exception as e:
            self.logger.error(f"多时间框架整合时出错: {str(e)}")
            return {}
    
    # ==================== 私有方法：核心实现 ====================
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'timing_weight': 0.3,
            'selection_weight': 0.7,
            'rebalance_frequency': 'monthly',
            'risk_budget': 0.15,
            'max_position_size': 0.05,
            'min_position_size': 0.001,
            'transaction_cost': 0.002,
            'lookback_window': 252,
            'optimization_method': 'mean_variance'
        }
    
    def _align_factors(self, timing_factors: Dict[str, pd.Series],
                      selection_factors: Dict[str, pd.DataFrame],
                      market_data: pd.DataFrame) -> Dict[str, Any]:
        """对齐因子数据"""
        # 获取公共时间索引
        common_index = market_data.index
        
        # 对齐择时因子
        aligned_timing = {}
        for name, factor in timing_factors.items():
            aligned_timing[name] = factor.reindex(common_index).fillna(method='ffill')
        
        # 对齐择股因子
        aligned_selection = {}
        for name, factor in selection_factors.items():
            aligned_selection[name] = factor.reindex(common_index).fillna(method='ffill')
        
        return {
            'timing_factors': aligned_timing,
            'selection_factors': aligned_selection
        }
    
    def _build_factor_framework(self, aligned_factors: Dict[str, Any]) -> Dict[str, Any]:
        """构建多层次因子框架"""
        framework = {}
        
        # 择时层
        timing_composite = self._composite_timing_factors(aligned_factors['timing_factors'])
        framework['timing_layer'] = timing_composite
        
        # 择股层
        selection_composite = self._composite_selection_factors(aligned_factors['selection_factors'])
        framework['selection_layer'] = selection_composite
        
        # 整合层
        integrated_signal = self._integrate_layers(timing_composite, selection_composite)
        framework['integrated_layer'] = integrated_signal
        
        return framework
    
    def _composite_timing_factors(self, timing_factors: Dict[str, pd.Series]) -> pd.Series:
        """合成择时因子"""
        if not timing_factors:
            return pd.Series()
        
        # 等权重合成（可以改为动态权重）
        factor_df = pd.DataFrame(timing_factors)
        
        # 标准化
        standardized = (factor_df - factor_df.mean()) / factor_df.std()
        
        # 合成
        composite = standardized.mean(axis=1)
        
        return composite
    
    def _composite_selection_factors(self, selection_factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合成择股因子"""
        if not selection_factors:
            return pd.DataFrame()
        
        # 合并所有择股因子
        all_selection = pd.concat(selection_factors.values(), axis=1)
        
        # 标准化
        standardized = (all_selection - all_selection.mean()) / all_selection.std()
        
        return standardized.fillna(0)
    
    def _integrate_layers(self, timing_signal: pd.Series, 
                         selection_signal: pd.DataFrame) -> pd.DataFrame:
        """整合择时和择股层"""
        if timing_signal.empty or selection_signal.empty:
            return pd.DataFrame()
        
        # 将择时信号应用到择股信号
        integrated = selection_signal.copy()
        
        for col in integrated.columns:
            integrated[col] = integrated[col] * timing_signal
        
        return integrated
    
    def _calculate_dynamic_weights(self, factor_framework: Dict[str, Any],
                                 market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算动态权重"""
        weights = {}
        
        # 基于市场状态调整权重
        market_volatility = market_data['close'].pct_change().rolling(20).std()
        
        # 高波动时增加择时权重
        timing_weight = 0.2 + 0.3 * (market_volatility / market_volatility.rolling(252).mean())
        selection_weight = 1 - timing_weight
        
        weights['timing_weight'] = timing_weight.fillna(0.3)
        weights['selection_weight'] = selection_weight.fillna(0.7)
        
        return weights
    
    def _apply_risk_budget(self, weights: Dict[str, pd.Series],
                          market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """应用风险预算"""
        risk_adjusted = weights.copy()
        
        # 计算组合风险
        returns = market_data['close'].pct_change()
        portfolio_vol = returns.rolling(20).std()
        
        # 风险预算调整
        risk_multiplier = self.risk_budget / (portfolio_vol + 1e-8)
        risk_multiplier = risk_multiplier.clip(0.5, 2.0)  # 限制调整幅度
        
        for key in risk_adjusted:
            risk_adjusted[key] = risk_adjusted[key] * risk_multiplier
        
        return risk_adjusted
    
    def _generate_portfolio(self, weights: Dict[str, pd.Series],
                          universe: List[str],
                          market_data: pd.DataFrame) -> Dict[str, Any]:
        """生成投资组合"""
        portfolio = {}
        
        # 获取整合信号
        if 'integrated_layer' in weights:
            integrated_signal = weights['integrated_layer']
        else:
            # 使用简单的等权重组合
            integrated_signal = pd.DataFrame(
                np.ones((len(market_data), len(universe))) / len(universe),
                index=market_data.index,
                columns=universe
            )
        
        # 应用权重约束
        constrained_weights = self._apply_weight_constraints(integrated_signal)
        
        # 计算组合收益
        returns = market_data['close'].pct_change()
        portfolio_returns = (constrained_weights.shift(1) * returns).sum(axis=1)
        
        portfolio = {
            'weights': constrained_weights,
            'returns': portfolio_returns,
            'cumulative_returns': (1 + portfolio_returns).cumprod(),
            'universe': universe
        }
        
        return portfolio
    
    def _apply_weight_constraints(self, weights: pd.DataFrame) -> pd.DataFrame:
        """应用权重约束"""
        constrained = weights.copy()
        
        # 单个股票权重限制
        max_weight = self.config.get('max_position_size', 0.05)
        min_weight = self.config.get('min_position_size', 0.001)
        
        constrained = constrained.clip(lower=min_weight, upper=max_weight)
        
        # 权重归一化
        row_sums = constrained.sum(axis=1)
        constrained = constrained.div(row_sums, axis=0).fillna(0)
        
        return constrained
    
    def _evaluate_strategy(self, portfolio: Dict[str, Any],
                          market_data: pd.DataFrame) -> Dict[str, float]:
        """评估策略表现"""
        returns = portfolio['returns']
        
        # 基本指标
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / (annual_vol + 1e-8)
        
        # 最大回撤
        cumulative = portfolio['cumulative_returns']
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 信息比率（相对基准）
        benchmark_returns = market_data['close'].pct_change()
        excess_returns = returns - benchmark_returns
        information_ratio = excess_returns.mean() / (excess_returns.std() + 1e-8) * np.sqrt(252)
        
        evaluation = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'information_ratio': information_ratio
        }
        
        return evaluation
    
    def _detect_factor_rotation(self, factor_framework: Dict[str, Any],
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """检测因子轮动"""
        rotation_signals = {}
        
        # 计算因子表现
        if 'timing_layer' in factor_framework:
            timing_performance = self._calculate_factor_performance(
                factor_framework['timing_layer'], market_data
            )
            rotation_signals['timing_rotation'] = self._detect_rotation_signal(timing_performance)
        
        if 'selection_layer' in factor_framework:
            selection_performance = self._calculate_factor_performance(
                factor_framework['selection_layer'], market_data
            )
            rotation_signals['selection_rotation'] = self._detect_rotation_signal(selection_performance)
        
        return rotation_signals
    
    def _calculate_factor_performance(self, factor_data: Union[pd.Series, pd.DataFrame],
                                    market_data: pd.DataFrame) -> pd.Series:
        """计算因子表现"""
        returns = market_data['close'].pct_change()
        
        if isinstance(factor_data, pd.Series):
            # 择时因子表现
            factor_returns = factor_data.shift(1) * returns
        else:
            # 择股因子表现
            factor_returns = (factor_data.shift(1) * returns).sum(axis=1)
        
        return factor_returns.fillna(0)
    
    def _detect_rotation_signal(self, performance: pd.Series, 
                              window: int = 60) -> pd.Series:
        """检测轮动信号"""
        # 计算滚动表现
        rolling_perf = performance.rolling(window).mean()
        
        # 计算Z分数
        z_score = (rolling_perf - rolling_perf.rolling(252).mean()) / rolling_perf.rolling(252).std()
        
        # 生成轮动信号
        rotation_signal = pd.Series(0, index=performance.index)
        rotation_signal[z_score > 1] = 1  # 强势
        rotation_signal[z_score < -1] = -1  # 弱势
        
        return rotation_signal
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List[datetime]:
        """获取调仓日期"""
        if self.rebalance_frequency == 'monthly':
            # 每月最后一个交易日
            rebalance_dates = []
            for year in range(date_index.year.min(), date_index.year.max() + 1):
                for month in range(1, 13):
                    month_dates = date_index[(date_index.year == year) & (date_index.month == month)]
                    if len(month_dates) > 0:
                        rebalance_dates.append(month_dates[-1])
            return rebalance_dates
        
        elif self.rebalance_frequency == 'weekly':
            # 每周最后一个交易日
            return date_index[date_index.dayofweek == 4].tolist()  # 周五
        
        else:
            # 每日调仓
            return date_index.tolist()
    
    def _update_history(self, result: Dict[str, Any]):
        """更新历史记录"""
        timestamp = datetime.now()
        
        # 更新因子历史
        if 'factor_framework' in result:
            self.factor_history[timestamp] = result['factor_framework']
        
        # 更新权重历史
        if 'dynamic_weights' in result:
            self.weight_history[timestamp] = result['dynamic_weights']
        
        # 更新表现历史
        if 'strategy_evaluation' in result:
            self.performance_history[timestamp] = result['strategy_evaluation']
    
    def _regime_based_weighting(self, factor_performance: Dict[str, pd.Series],
                              market_regime: pd.Series) -> Dict[str, float]:
        """基于市场状态的权重分配"""
        regime_weights = {}
        
        # 获取当前市场状态
        current_regime = market_regime.iloc[-1] if len(market_regime) > 0 else 0
        
        for factor_name, performance in factor_performance.items():
            # 计算在不同市场状态下的表现
            regime_performance = {}
            for regime in market_regime.unique():
                regime_mask = market_regime == regime
                if regime_mask.sum() > 0:
                    regime_perf = performance[regime_mask].mean()
                    regime_performance[regime] = regime_perf
            
            # 基于当前状态分配权重
            if current_regime in regime_performance:
                relative_perf = regime_performance[current_regime]
                regime_weights[factor_name] = max(0, relative_perf)
            else:
                regime_weights[factor_name] = 1.0 / len(factor_performance)
        
        # 归一化权重
        total_weight = sum(regime_weights.values())
        if total_weight > 0:
            regime_weights = {k: v / total_weight for k, v in regime_weights.items()}
        
        return regime_weights
    
    def _momentum_based_weighting(self, factor_performance: Dict[str, pd.Series]) -> Dict[str, float]:
        """基于因子动量的权重分配"""
        momentum_weights = {}
        
        for factor_name, performance in factor_performance.items():
            # 计算短期动量
            short_momentum = performance.tail(20).mean()
            # 计算长期动量
            long_momentum = performance.tail(60).mean()
            
            # 动量得分
            momentum_score = 0.6 * short_momentum + 0.4 * long_momentum
            momentum_weights[factor_name] = max(0, momentum_score)
        
        # 归一化权重
        total_weight = sum(momentum_weights.values())
        if total_weight > 0:
            momentum_weights = {k: v / total_weight for k, v in momentum_weights.items()}
        
        return momentum_weights
    
    def _risk_adjusted_weighting(self, factor_performance: Dict[str, pd.Series]) -> Dict[str, float]:
        """基于风险调整的权重分配"""
        risk_weights = {}
        
        for factor_name, performance in factor_performance.items():
            # 计算风险调整收益
            mean_return = performance.mean()
            volatility = performance.std()
            
            # 夏普比率作为权重
            sharpe = mean_return / (volatility + 1e-8)
            risk_weights[factor_name] = max(0, sharpe)
        
        # 归一化权重
        total_weight = sum(risk_weights.values())
        if total_weight > 0:
            risk_weights = {k: v / total_weight for k, v in risk_weights.items()}
        
        return risk_weights
    
    def _calculate_factor_momentum(self, factor_data: Dict[str, pd.DataFrame],
                                 lookback_window: int) -> Dict[str, pd.Series]:
        """计算因子动量"""
        factor_momentum = {}
        
        for factor_name, data in factor_data.items():
            if isinstance(data, pd.DataFrame):
                # 计算因子收益
                factor_returns = data.pct_change()
                
                # 计算动量
                momentum = factor_returns.rolling(lookback_window).mean()
                
                # 取平均动量
                avg_momentum = momentum.mean(axis=1)
                factor_momentum[factor_name] = avg_momentum
            else:
                # 单一序列
                factor_returns = data.pct_change()
                momentum = factor_returns.rolling(lookback_window).mean()
                factor_momentum[factor_name] = momentum
        
        return factor_momentum
    
    def _identify_rotation_signals(self, factor_momentum: Dict[str, pd.Series],
                                 threshold: float) -> Dict[str, pd.Series]:
        """识别轮动信号"""
        rotation_signals = {}
        
        for factor_name, momentum in factor_momentum.items():
            # 标准化动量
            z_score = (momentum - momentum.rolling(252).mean()) / momentum.rolling(252).std()
            
            # 生成信号
            signals = pd.Series(0, index=momentum.index)
            signals[z_score > threshold] = 1  # 买入信号
            signals[z_score < -threshold] = -1  # 卖出信号
            
            rotation_signals[factor_name] = signals
        
        return rotation_signals
    
    def _build_rotation_portfolio(self, rotation_signals: Dict[str, pd.Series],
                                factor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """构建轮动组合"""
        # 获取公共时间索引
        common_index = None
        for signals in rotation_signals.values():
            if common_index is None:
                common_index = signals.index
            else:
                common_index = common_index.intersection(signals.index)
        
        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()
        
        # 构建轮动权重
        rotation_weights = pd.DataFrame(0, index=common_index, columns=list(rotation_signals.keys()))
        
        for factor_name, signals in rotation_signals.items():
            aligned_signals = signals.reindex(common_index).fillna(0)
            rotation_weights[factor_name] = aligned_signals
        
        # 归一化权重
        row_sums = rotation_weights.abs().sum(axis=1)
        rotation_weights = rotation_weights.div(row_sums, axis=0).fillna(0)
        
        return rotation_weights
    
    def _apply_rotation_risk_control(self, rotation_portfolio: pd.DataFrame) -> pd.DataFrame:
        """应用轮动风险控制"""
        risk_controlled = rotation_portfolio.copy()
        
        # 计算组合波动率
        portfolio_vol = rotation_portfolio.std(axis=1)
        
        # 风险调整
        vol_target = 0.15  # 目标波动率
        vol_multiplier = vol_target / (portfolio_vol + 1e-8)
        vol_multiplier = vol_multiplier.clip(0.5, 2.0)
        
        # 应用风险调整
        for col in risk_controlled.columns:
            risk_controlled[col] = risk_controlled[col] * vol_multiplier
        
        return risk_controlled
    
    def _evaluate_rotation_strategy(self, rotation_portfolio: pd.DataFrame) -> Dict[str, float]:
        """评估轮动策略"""
        if rotation_portfolio.empty:
            return {}
        
        # 计算组合收益
        portfolio_returns = rotation_portfolio.sum(axis=1)
        
        # 基本指标
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / (annual_vol + 1e-8)
        
        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        evaluation = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (portfolio_returns > 0).mean()
        }
        
        return evaluation
    
    def _standardize_factors(self, factors: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """标准化因子"""
        standardized = {}
        
        for name, factor in factors.items():
            # Z-score标准化
            standardized_factor = (factor - factor.mean()) / factor.std()
            standardized[name] = standardized_factor.fillna(0)
        
        return standardized
    
    def _check_timeframe_consistency(self, short_term: Dict[str, pd.DataFrame],
                                   medium_term: Dict[str, pd.DataFrame],
                                   long_term: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """检查时间框架一致性"""
        consistency_scores = {}
        
        # 获取公共因子
        common_factors = set(short_term.keys()) & set(medium_term.keys()) & set(long_term.keys())
        
        for factor_name in common_factors:
            # 计算相关性
            short_avg = short_term[factor_name].mean(axis=1)
            medium_avg = medium_term[factor_name].mean(axis=1)
            long_avg = long_term[factor_name].mean(axis=1)
            
            # 对齐时间序列
            aligned_data = pd.concat([short_avg, medium_avg, long_avg], axis=1).dropna()
            
            if len(aligned_data) > 10:
                corr_matrix = aligned_data.corr()
                avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                consistency_scores[factor_name] = avg_correlation
            else:
                consistency_scores[factor_name] = 0
        
        return consistency_scores
    
    def _calculate_dynamic_timeframe_weights(self, short_term: Dict[str, pd.DataFrame],
                                           medium_term: Dict[str, pd.DataFrame],
                                           long_term: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """计算动态时间框架权重"""
        dynamic_weights = {}
        
        # 基于市场波动率调整权重
        # 这里简化处理，实际应用中可以更复杂
        base_weights = {
            'short_term': 0.2,
            'medium_term': 0.5,
            'long_term': 0.3
        }
        
        # 可以基于市场状态、因子表现等动态调整
        # 这里返回基础权重
        for factor_name in set(short_term.keys()) | set(medium_term.keys()) | set(long_term.keys()):
            dynamic_weights[factor_name] = base_weights.copy()
        
        return dynamic_weights