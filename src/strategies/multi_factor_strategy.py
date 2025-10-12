"""
多因子组合策略模块

整合技术、基本面、宏观、情绪、机器学习和高级因子，
实现动态权重分配和Alpha生成优化策略
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# 导入因子计算模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from factors.technical import TechnicalFactors
    from factors.fundamental_factors import FundamentalFactorCalculator
    from factors.macro_factors import MacroFactorCalculator
    from factors.sentiment_factors import SentimentFactorCalculator
    from factors.ml_factors import MLFactorCalculator
    from factors.advanced_factors import AdvancedFactorCalculator
    from factors.factor_optimizer import FactorOptimizer
except ImportError as e:
    print(f"导入因子模块失败: {e}")

# 机器学习库
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 优化库
try:
    from scipy.optimize import minimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class MultiFactorStrategy:
    """多因子组合策略"""
    
    def __init__(self, lookback_window: int = 252, rebalance_freq: int = 20,
                 factor_decay: float = 0.95, min_factor_score: float = 0.1):
        """
        初始化多因子策略
        
        Args:
            lookback_window: 回望窗口期
            rebalance_freq: 再平衡频率（天）
            factor_decay: 因子衰减系数
            min_factor_score: 最小因子评分阈值
        """
        self.lookback_window = lookback_window
        self.rebalance_freq = rebalance_freq
        self.factor_decay = factor_decay
        self.min_factor_score = min_factor_score
        
        # 初始化因子计算器
        self.technical_calc = TechnicalFactors()
        self.fundamental_calc = FundamentalFactorCalculator()
        self.macro_calc = MacroFactorCalculator()
        self.sentiment_calc = SentimentFactorCalculator()
        self.ml_calc = MLFactorCalculator()
        self.advanced_calc = AdvancedFactorCalculator()
        self.optimizer = FactorOptimizer()
        
        # 策略状态
        self.factor_weights = {}
        self.factor_scores = {}
        self.strategy_performance = {}
        self.factor_history = {}
        
        print("多因子组合策略初始化完成")
    
    def calculate_all_factors(self, price_data: pd.DataFrame, 
                            volume_data: pd.DataFrame = None,
                            fundamental_data: pd.DataFrame = None,
                            macro_data: pd.DataFrame = None,
                            additional_data: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        计算所有类型的因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            macro_data: 宏观数据
            additional_data: 额外数据
        
        Returns:
            所有因子字典
        """
        all_factors = {}
        
        print("开始计算所有因子...")
        
        try:
            # 1. 技术因子
            tech_factors = self.technical_calc.calculate_all_factors(price_data, volume_data)
            all_factors.update({f"tech_{k}": v for k, v in tech_factors.items()})
            
            # 2. 基本面因子
            if fundamental_data is not None:
                fund_factors = self.fundamental_calc.calculate_all_factors(fundamental_data)
                all_factors.update({f"fund_{k}": v for k, v in fund_factors.items()})
            
            # 3. 宏观因子
            if macro_data is not None:
                macro_factors = self.macro_calc.calculate_all_factors(macro_data)
                all_factors.update({f"macro_{k}": v for k, v in macro_factors.items()})
            
            # 4. 情绪因子
            sentiment_factors = self.sentiment_calc.calculate_all_factors(price_data, volume_data)
            all_factors.update({f"sentiment_{k}": v for k, v in sentiment_factors.items()})
            
            # 5. 机器学习因子
            ml_factors = self.ml_calc.calculate_all_factors(price_data, volume_data, fundamental_data)
            all_factors.update({f"ml_{k}": v for k, v in ml_factors.items()})
            
            # 6. 高级因子
            advanced_factors = self.advanced_calc.calculate_all_advanced_factors(
                price_data, volume_data, additional_data
            )
            all_factors.update({f"advanced_{k}": v for k, v in advanced_factors.items()})
            
        except Exception as e:
            print(f"因子计算过程中出现错误: {e}")
        
        # 清理因子
        cleaned_factors = self._clean_factors(all_factors)
        
        print(f"因子计算完成，共生成 {len(cleaned_factors)} 个有效因子")
        return cleaned_factors
    
    def _clean_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """清理和预处理因子"""
        cleaned_factors = {}
        
        for name, factor in factors.items():
            if not isinstance(factor, pd.Series):
                continue
            
            # 检查数据质量
            valid_ratio = factor.notna().sum() / len(factor)
            if valid_ratio < 0.1:  # 至少10%的数据有效
                continue
            
            # 去除极值
            factor_clean = self._winsorize_factor(factor)
            
            # 标准化
            factor_std = self._standardize_factor(factor_clean)
            
            # 检查是否为常数
            if factor_std.std() > 1e-6:
                cleaned_factors[name] = factor_std
        
        return cleaned_factors
    
    def _winsorize_factor(self, factor: pd.Series, quantile: float = 0.01) -> pd.Series:
        """因子去极值处理"""
        factor_clean = factor.copy()
        
        # 计算分位数
        lower_bound = factor.quantile(quantile)
        upper_bound = factor.quantile(1 - quantile)
        
        # 去极值
        factor_clean = factor_clean.clip(lower_bound, upper_bound)
        
        return factor_clean
    
    def _standardize_factor(self, factor: pd.Series) -> pd.Series:
        """因子标准化"""
        # 使用滚动标准化
        rolling_mean = factor.rolling(window=60, min_periods=20).mean()
        rolling_std = factor.rolling(window=60, min_periods=20).std()
        
        standardized = (factor - rolling_mean) / rolling_std
        
        # 填充初始值
        standardized = standardized.fillna(method='bfill')
        
        return standardized
    
    def calculate_factor_scores(self, factors: Dict[str, pd.Series], 
                              returns: pd.Series, window: int = 60) -> Dict[str, float]:
        """
        计算因子评分
        
        Args:
            factors: 因子字典
            returns: 收益率序列
            window: 评分窗口
        
        Returns:
            因子评分字典
        """
        factor_scores = {}
        
        for name, factor in factors.items():
            try:
                # 对齐数据
                aligned_factor = factor.reindex(returns.index, method='ffill')
                
                # 计算IC（信息系数）
                ic_scores = []
                for i in range(window, len(returns)):
                    factor_window = aligned_factor.iloc[i-window:i]
                    return_window = returns.iloc[i-window+1:i+1]  # 前瞻1期
                    
                    if len(factor_window.dropna()) > 10 and len(return_window.dropna()) > 10:
                        ic = factor_window.corr(return_window)
                        if not np.isnan(ic):
                            ic_scores.append(ic)
                
                if ic_scores:
                    # 综合评分：IC均值 + IC稳定性
                    ic_mean = np.mean(ic_scores)
                    ic_std = np.std(ic_scores)
                    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
                    
                    # 综合评分
                    score = abs(ic_mean) * 0.6 + abs(ic_ir) * 0.4
                    factor_scores[name] = score
                else:
                    factor_scores[name] = 0.0
                    
            except Exception as e:
                print(f"计算因子 {name} 评分失败: {e}")
                factor_scores[name] = 0.0
        
        return factor_scores
    
    def optimize_factor_weights(self, factors: Dict[str, pd.Series], 
                              returns: pd.Series, method: str = 'ic_weighted') -> Dict[str, float]:
        """
        优化因子权重
        
        Args:
            factors: 因子字典
            returns: 收益率序列
            method: 权重优化方法
        
        Returns:
            因子权重字典
        """
        # 计算因子评分
        factor_scores = self.calculate_factor_scores(factors, returns)
        
        # 过滤低评分因子
        valid_factors = {k: v for k, v in factor_scores.items() if v >= self.min_factor_score}
        
        if not valid_factors:
            print("没有有效因子，使用等权重")
            return {k: 1.0/len(factors) for k in factors.keys()}
        
        if method == 'equal_weight':
            # 等权重
            weight = 1.0 / len(valid_factors)
            weights = {k: weight if k in valid_factors else 0.0 for k in factors.keys()}
            
        elif method == 'ic_weighted':
            # IC加权
            total_score = sum(valid_factors.values())
            weights = {k: (valid_factors.get(k, 0) / total_score) if total_score > 0 else 0 
                      for k in factors.keys()}
            
        elif method == 'risk_parity':
            # 风险平价
            weights = self._calculate_risk_parity_weights(factors, valid_factors)
            
        elif method == 'max_diversification':
            # 最大分散化
            weights = self._calculate_max_diversification_weights(factors, valid_factors)
            
        else:
            # 默认IC加权
            total_score = sum(valid_factors.values())
            weights = {k: (valid_factors.get(k, 0) / total_score) if total_score > 0 else 0 
                      for k in factors.keys()}
        
        return weights
    
    def _calculate_risk_parity_weights(self, factors: Dict[str, pd.Series], 
                                     valid_factors: Dict[str, float]) -> Dict[str, float]:
        """计算风险平价权重"""
        if not SCIPY_AVAILABLE:
            # 简化版本：使用波动率倒数
            factor_vols = {}
            for name in valid_factors.keys():
                if name in factors:
                    vol = factors[name].std()
                    factor_vols[name] = 1.0 / vol if vol > 0 else 0
            
            total_inv_vol = sum(factor_vols.values())
            weights = {k: (factor_vols.get(k, 0) / total_inv_vol) if total_inv_vol > 0 else 0 
                      for k in factors.keys()}
            return weights
        
        # 构建协方差矩阵
        factor_df = pd.DataFrame({k: v for k, v in factors.items() if k in valid_factors})
        factor_df = factor_df.dropna()
        
        if factor_df.empty or len(factor_df.columns) < 2:
            weight = 1.0 / len(valid_factors)
            return {k: weight if k in valid_factors else 0.0 for k in factors.keys()}
        
        cov_matrix = factor_df.cov().values
        n_factors = len(cov_matrix)
        
        # 风险平价优化
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # 最小化风险贡献的差异
            target_contrib = portfolio_vol / n_factors
            return np.sum((contrib - target_contrib) ** 2)
        
        # 约束条件
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(n_factors)]
        
        # 初始权重
        x0 = np.ones(n_factors) / n_factors
        
        try:
            result = minimize(risk_parity_objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = dict(zip(factor_df.columns, result.x))
                weights = {k: optimized_weights.get(k, 0.0) for k in factors.keys()}
            else:
                # 回退到等权重
                weight = 1.0 / len(valid_factors)
                weights = {k: weight if k in valid_factors else 0.0 for k in factors.keys()}
        except:
            # 回退到等权重
            weight = 1.0 / len(valid_factors)
            weights = {k: weight if k in valid_factors else 0.0 for k in factors.keys()}
        
        return weights
    
    def _calculate_max_diversification_weights(self, factors: Dict[str, pd.Series], 
                                             valid_factors: Dict[str, float]) -> Dict[str, float]:
        """计算最大分散化权重"""
        if not SCIPY_AVAILABLE:
            # 简化版本：使用相关性倒数
            factor_df = pd.DataFrame({k: v for k, v in factors.items() if k in valid_factors})
            factor_df = factor_df.dropna()
            
            if factor_df.empty:
                weight = 1.0 / len(valid_factors)
                return {k: weight if k in valid_factors else 0.0 for k in factors.keys()}
            
            # 计算平均相关性
            corr_matrix = factor_df.corr()
            avg_corr = {}
            
            for col in corr_matrix.columns:
                other_corrs = corr_matrix[col].drop(col)
                avg_corr[col] = 1.0 / (1 + abs(other_corrs.mean())) if len(other_corrs) > 0 else 1.0
            
            total_score = sum(avg_corr.values())
            weights = {k: (avg_corr.get(k, 0) / total_score) if total_score > 0 else 0 
                      for k in factors.keys()}
            return weights
        
        # 完整版本的最大分散化优化
        factor_df = pd.DataFrame({k: v for k, v in factors.items() if k in valid_factors})
        factor_df = factor_df.dropna()
        
        if factor_df.empty or len(factor_df.columns) < 2:
            weight = 1.0 / len(valid_factors)
            return {k: weight if k in valid_factors else 0.0 for k in factors.keys()}
        
        cov_matrix = factor_df.cov().values
        vol_vector = np.sqrt(np.diag(cov_matrix))
        
        # 最大分散化比率优化
        def diversification_ratio(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            weighted_avg_vol = np.dot(weights, vol_vector)
            return -weighted_avg_vol / portfolio_vol  # 负号因为要最大化
        
        # 约束条件
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(vol_vector))]
        
        # 初始权重
        x0 = np.ones(len(vol_vector)) / len(vol_vector)
        
        try:
            result = minimize(diversification_ratio, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = dict(zip(factor_df.columns, result.x))
                weights = {k: optimized_weights.get(k, 0.0) for k in factors.keys()}
            else:
                # 回退到等权重
                weight = 1.0 / len(valid_factors)
                weights = {k: weight if k in valid_factors else 0.0 for k in factors.keys()}
        except:
            # 回退到等权重
            weight = 1.0 / len(valid_factors)
            weights = {k: weight if k in valid_factors else 0.0 for k in factors.keys()}
        
        return weights
    
    def generate_composite_signal(self, factors: Dict[str, pd.Series], 
                                weights: Dict[str, float]) -> pd.Series:
        """
        生成综合信号
        
        Args:
            factors: 因子字典
            weights: 因子权重
        
        Returns:
            综合信号序列
        """
        if not factors or not weights:
            return pd.Series()
        
        # 获取共同的时间索引
        common_index = None
        for factor in factors.values():
            if isinstance(factor, pd.Series):
                if common_index is None:
                    common_index = factor.index
                else:
                    common_index = common_index.intersection(factor.index)
        
        if common_index is None or len(common_index) == 0:
            return pd.Series()
        
        # 计算加权综合信号
        composite_signal = pd.Series(0.0, index=common_index)
        
        for name, weight in weights.items():
            if name in factors and weight > 0:
                factor = factors[name].reindex(common_index, method='ffill')
                composite_signal += weight * factor.fillna(0)
        
        return composite_signal
    
    def backtest_strategy(self, price_data: pd.DataFrame, 
                         factors: Dict[str, pd.Series],
                         weights: Dict[str, float],
                         transaction_cost: float = 0.001) -> Dict[str, Union[pd.Series, float]]:
        """
        回测多因子策略
        
        Args:
            price_data: 价格数据
            factors: 因子字典
            weights: 因子权重
            transaction_cost: 交易成本
        
        Returns:
            回测结果字典
        """
        if 'close' not in price_data.columns:
            return {}
        
        # 生成综合信号
        signal = self.generate_composite_signal(factors, weights)
        
        if signal.empty:
            return {}
        
        # 对齐价格数据
        prices = price_data['close'].reindex(signal.index, method='ffill')
        returns = prices.pct_change()
        
        # 生成交易信号（标准化后取符号）
        signal_std = (signal - signal.rolling(60).mean()) / signal.rolling(60).std()
        positions = np.sign(signal_std.fillna(0))
        
        # 计算策略收益
        strategy_returns = positions.shift(1) * returns
        
        # 扣除交易成本
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * transaction_cost
        strategy_returns_net = strategy_returns - transaction_costs
        
        # 计算累积收益
        cumulative_returns = (1 + strategy_returns_net).cumprod()
        
        # 计算性能指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns_net)) - 1
        volatility = strategy_returns_net.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (strategy_returns_net > 0).mean()
        
        return {
            'cumulative_returns': cumulative_returns,
            'strategy_returns': strategy_returns_net,
            'positions': positions,
            'signal': signal,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def run_strategy(self, price_data: pd.DataFrame,
                    volume_data: pd.DataFrame = None,
                    fundamental_data: pd.DataFrame = None,
                    macro_data: pd.DataFrame = None,
                    additional_data: Dict[str, pd.Series] = None,
                    weight_method: str = 'ic_weighted') -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        运行完整的多因子策略
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            macro_data: 宏观数据
            additional_data: 额外数据
            weight_method: 权重优化方法
        
        Returns:
            策略运行结果
        """
        print("开始运行多因子策略...")
        
        # 1. 计算所有因子
        factors = self.calculate_all_factors(
            price_data, volume_data, fundamental_data, macro_data, additional_data
        )
        
        if not factors:
            print("没有有效因子，策略无法运行")
            return {}
        
        # 2. 计算收益率
        returns = price_data['close'].pct_change()
        
        # 3. 优化因子权重
        weights = self.optimize_factor_weights(factors, returns, weight_method)
        
        # 4. 回测策略
        backtest_results = self.backtest_strategy(price_data, factors, weights)
        
        # 5. 保存结果
        self.factor_weights = weights
        self.factor_scores = self.calculate_factor_scores(factors, returns)
        self.strategy_performance = backtest_results
        
        # 6. 生成报告
        report = self.generate_strategy_report(factors, weights, backtest_results)
        
        print("多因子策略运行完成")
        
        return {
            'factors': factors,
            'weights': weights,
            'backtest_results': backtest_results,
            'report': report
        }
    
    def generate_strategy_report(self, factors: Dict[str, pd.Series],
                               weights: Dict[str, float],
                               backtest_results: Dict) -> str:
        """生成策略报告"""
        report = "=== 多因子组合策略报告 ===\n\n"
        
        # 因子统计
        report += f"总因子数量: {len(factors)}\n"
        active_factors = sum(1 for w in weights.values() if w > 0.01)
        report += f"有效因子数量: {active_factors}\n\n"
        
        # 因子权重分布
        report += "主要因子权重:\n"
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for name, weight in sorted_weights[:10]:
            if weight > 0.01:
                report += f"  {name}: {weight:.3f}\n"
        
        # 因子类别分布
        categories = {
            '技术因子': 'tech_',
            '基本面因子': 'fund_',
            '宏观因子': 'macro_',
            '情绪因子': 'sentiment_',
            '机器学习因子': 'ml_',
            '高级因子': 'advanced_'
        }
        
        report += "\n因子类别权重分布:\n"
        for category, prefix in categories.items():
            category_weight = sum(w for k, w in weights.items() if k.startswith(prefix))
            if category_weight > 0:
                report += f"  {category}: {category_weight:.3f}\n"
        
        # 策略性能
        if backtest_results:
            report += "\n策略性能指标:\n"
            report += f"  总收益率: {backtest_results.get('total_return', 0):.2%}\n"
            report += f"  年化收益率: {backtest_results.get('annual_return', 0):.2%}\n"
            report += f"  年化波动率: {backtest_results.get('volatility', 0):.2%}\n"
            report += f"  夏普比率: {backtest_results.get('sharpe_ratio', 0):.3f}\n"
            report += f"  最大回撤: {backtest_results.get('max_drawdown', 0):.2%}\n"
            report += f"  胜率: {backtest_results.get('win_rate', 0):.2%}\n"
        
        # 策略评级
        if backtest_results:
            sharpe = backtest_results.get('sharpe_ratio', 0)
            if sharpe > 1.5:
                rating = "优秀 (A)"
            elif sharpe > 1.0:
                rating = "良好 (B)"
            elif sharpe > 0.5:
                rating = "一般 (C)"
            else:
                rating = "较差 (D)"
            
            report += f"\n策略评级: {rating}\n"
        
        return report

def main():
    """示例用法"""
    print("多因子组合策略示例")
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_periods = len(dates)
    
    np.random.seed(42)
    
    # 模拟价格数据
    price_data = pd.DataFrame({
        'open': np.random.uniform(95, 105, n_periods),
        'high': np.random.uniform(100, 110, n_periods),
        'low': np.random.uniform(90, 100, n_periods),
        'close': np.random.uniform(95, 105, n_periods),
    }, index=dates)
    
    # 添加趋势
    trend = np.cumsum(np.random.normal(0.001, 0.02, n_periods))
    price_data['close'] = 100 * np.exp(trend)
    price_data['open'] = price_data['close'] * np.random.uniform(0.99, 1.01, n_periods)
    price_data['high'] = price_data[['open', 'close']].max(axis=1) * np.random.uniform(1.0, 1.02, n_periods)
    price_data['low'] = price_data[['open', 'close']].min(axis=1) * np.random.uniform(0.98, 1.0, n_periods)
    
    # 模拟成交量数据
    volume_data = pd.DataFrame({
        'volume': np.random.uniform(1e6, 5e6, n_periods)
    }, index=dates)
    
    # 模拟基本面数据
    fundamental_data = pd.DataFrame({
        'revenue': np.random.uniform(1e9, 5e9, n_periods),
        'net_income': np.random.uniform(1e8, 5e8, n_periods),
        'total_assets': np.random.uniform(1e10, 5e10, n_periods),
        'total_equity': np.random.uniform(5e9, 2e10, n_periods),
        'market_cap': price_data['close'] * 1e6,
    }, index=dates)
    
    # 运行策略
    strategy = MultiFactorStrategy()
    results = strategy.run_strategy(
        price_data=price_data,
        volume_data=volume_data,
        fundamental_data=fundamental_data,
        weight_method='ic_weighted'
    )
    
    # 打印报告
    if 'report' in results:
        print(results['report'])

if __name__ == "__main__":
    main()