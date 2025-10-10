"""
风险指标计算模块

提供各种风险度量方法和指标计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RiskMetricResult:
    """风险指标结果"""
    metric_name: str
    value: float
    confidence_level: Optional[float] = None
    time_horizon: Optional[int] = None
    calculation_method: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class VaRCalculator:
    """VaR计算器"""
    
    @staticmethod
    def historical_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """历史模拟法计算VaR"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def parametric_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """参数法计算VaR（假设正态分布）"""
        if len(returns) == 0:
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(confidence_level)
        
        return mean + z_score * std
    
    @staticmethod
    def modified_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """修正VaR（考虑偏度和峰度）"""
        if len(returns) == 0:
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        z_score = stats.norm.ppf(confidence_level)
        
        # Cornish-Fisher展开
        modified_z = (z_score + 
                     (z_score**2 - 1) * skewness / 6 +
                     (z_score**3 - 3*z_score) * kurtosis / 24 -
                     (2*z_score**3 - 5*z_score) * skewness**2 / 36)
        
        return mean + modified_z * std
    
    @staticmethod
    def monte_carlo_var(returns: pd.Series, confidence_level: float = 0.05, 
                       n_simulations: int = 10000) -> float:
        """蒙特卡洛模拟VaR"""
        if len(returns) == 0:
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        
        # 生成随机收益
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        return np.percentile(simulated_returns, confidence_level * 100)


class CVaRCalculator:
    """条件VaR（期望损失）计算器"""
    
    @staticmethod
    def historical_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """历史模拟法计算CVaR"""
        if len(returns) == 0:
            return 0.0
        
        var = VaRCalculator.historical_var(returns, confidence_level)
        tail_losses = returns[returns <= var]
        
        return tail_losses.mean() if len(tail_losses) > 0 else var
    
    @staticmethod
    def parametric_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """参数法计算CVaR"""
        if len(returns) == 0:
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(confidence_level)
        
        # 正态分布的条件期望
        phi_z = stats.norm.pdf(z_score)
        cvar = mean - std * phi_z / confidence_level
        
        return cvar


class VolatilityCalculator:
    """波动率计算器"""
    
    @staticmethod
    def simple_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """简单波动率"""
        if len(returns) == 0:
            return 0.0
        
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # 假设252个交易日
        
        return vol
    
    @staticmethod
    def ewma_volatility(returns: pd.Series, lambda_param: float = 0.94, 
                       annualize: bool = True) -> float:
        """指数加权移动平均波动率"""
        if len(returns) == 0:
            return 0.0
        
        # 计算EWMA方差
        squared_returns = returns**2
        weights = np.array([(1 - lambda_param) * lambda_param**i 
                           for i in range(len(squared_returns))])
        weights = weights[::-1]  # 反转权重
        weights /= weights.sum()  # 标准化
        
        ewma_variance = np.sum(weights * squared_returns)
        vol = np.sqrt(ewma_variance)
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol
    
    @staticmethod
    def garch_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """GARCH波动率（简化版）"""
        if len(returns) < 10:
            return VolatilityCalculator.simple_volatility(returns, annualize)
        
        # 简化的GARCH(1,1)实现
        # 实际应用中建议使用arch库
        
        # 初始参数
        omega = 0.01
        alpha = 0.1
        beta = 0.8
        
        # 计算条件方差
        returns_array = returns.values
        n = len(returns_array)
        
        # 初始方差
        variance = np.var(returns_array)
        variances = [variance]
        
        for i in range(1, n):
            variance = (omega + 
                       alpha * returns_array[i-1]**2 + 
                       beta * variance)
            variances.append(variance)
        
        # 当前波动率
        vol = np.sqrt(variances[-1])
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol


class DrawdownCalculator:
    """回撤计算器"""
    
    @staticmethod
    def maximum_drawdown(prices: pd.Series) -> Tuple[float, datetime, datetime]:
        """最大回撤"""
        if len(prices) == 0:
            return 0.0, None, None
        
        # 计算累计最高点
        peak = prices.expanding().max()
        
        # 计算回撤
        drawdown = (prices - peak) / peak
        
        # 找到最大回撤
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # 找到最大回撤开始的峰值日期
        peak_date = peak.loc[:max_dd_date].idxmax()
        
        return abs(max_dd), peak_date, max_dd_date
    
    @staticmethod
    def current_drawdown(prices: pd.Series) -> float:
        """当前回撤"""
        if len(prices) == 0:
            return 0.0
        
        peak = prices.max()
        current_price = prices.iloc[-1]
        
        return (peak - current_price) / peak
    
    @staticmethod
    def drawdown_duration(prices: pd.Series) -> pd.Series:
        """回撤持续时间"""
        if len(prices) == 0:
            return pd.Series()
        
        # 计算累计最高点
        peak = prices.expanding().max()
        
        # 判断是否在回撤中
        in_drawdown = prices < peak
        
        # 计算回撤持续时间
        duration = pd.Series(0, index=prices.index)
        current_duration = 0
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd:
                current_duration += 1
            else:
                current_duration = 0
            duration.iloc[i] = current_duration
        
        return duration


class CorrelationCalculator:
    """相关性计算器"""
    
    @staticmethod
    def pearson_correlation(returns1: pd.Series, returns2: pd.Series) -> float:
        """皮尔逊相关系数"""
        return returns1.corr(returns2)
    
    @staticmethod
    def spearman_correlation(returns1: pd.Series, returns2: pd.Series) -> float:
        """斯皮尔曼相关系数"""
        return returns1.corr(returns2, method='spearman')
    
    @staticmethod
    def kendall_correlation(returns1: pd.Series, returns2: pd.Series) -> float:
        """肯德尔相关系数"""
        return returns1.corr(returns2, method='kendall')
    
    @staticmethod
    def rolling_correlation(returns1: pd.Series, returns2: pd.Series, 
                          window: int = 30) -> pd.Series:
        """滚动相关系数"""
        return returns1.rolling(window).corr(returns2)


class BetaCalculator:
    """Beta计算器"""
    
    @staticmethod
    def market_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """市场Beta"""
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 1.0
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 1.0
        
        covariance = aligned_data['asset'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    @staticmethod
    def rolling_beta(asset_returns: pd.Series, market_returns: pd.Series, 
                    window: int = 60) -> pd.Series:
        """滚动Beta"""
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < window:
            return pd.Series([1.0] * len(aligned_data), index=aligned_data.index)
        
        def calc_beta(data):
            if len(data) < 2:
                return 1.0
            cov = data['asset'].cov(data['market'])
            var = data['market'].var()
            return cov / var if var != 0 else 1.0
        
        return aligned_data.rolling(window).apply(
            lambda x: calc_beta(pd.DataFrame(x.values, columns=['asset', 'market'])),
            raw=False
        )['asset']


class SharpeRatioCalculator:
    """夏普比率计算器"""
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """夏普比率"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate  # 年化超额收益
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        return excess_returns / volatility if volatility != 0 else 0.0
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """索提诺比率"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        
        return excess_returns / downside_volatility if downside_volatility != 0 else 0.0
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
        """卡尔马比率"""
        if len(returns) == 0 or len(prices) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd, _, _ = DrawdownCalculator.maximum_drawdown(prices)
        
        return annual_return / max_dd if max_dd != 0 else 0.0


class RiskMetricsEngine:
    """风险指标计算引擎"""
    
    def __init__(self):
        self.var_calculator = VaRCalculator()
        self.cvar_calculator = CVaRCalculator()
        self.volatility_calculator = VolatilityCalculator()
        self.drawdown_calculator = DrawdownCalculator()
        self.correlation_calculator = CorrelationCalculator()
        self.beta_calculator = BetaCalculator()
        self.sharpe_calculator = SharpeRatioCalculator()
    
    def calculate_comprehensive_risk_metrics(self, 
                                           returns: pd.Series,
                                           prices: Optional[pd.Series] = None,
                                           market_returns: Optional[pd.Series] = None,
                                           confidence_levels: List[float] = [0.01, 0.05, 0.1],
                                           risk_free_rate: float = 0.02) -> Dict[str, RiskMetricResult]:
        """计算综合风险指标"""
        
        results = {}
        
        # VaR指标
        for cl in confidence_levels:
            # 历史VaR
            hist_var = self.var_calculator.historical_var(returns, cl)
            results[f'Historical_VaR_{int(cl*100)}%'] = RiskMetricResult(
                metric_name=f'Historical VaR {int(cl*100)}%',
                value=hist_var,
                confidence_level=cl,
                calculation_method='Historical Simulation'
            )
            
            # 参数VaR
            param_var = self.var_calculator.parametric_var(returns, cl)
            results[f'Parametric_VaR_{int(cl*100)}%'] = RiskMetricResult(
                metric_name=f'Parametric VaR {int(cl*100)}%',
                value=param_var,
                confidence_level=cl,
                calculation_method='Parametric (Normal)'
            )
            
            # 修正VaR
            mod_var = self.var_calculator.modified_var(returns, cl)
            results[f'Modified_VaR_{int(cl*100)}%'] = RiskMetricResult(
                metric_name=f'Modified VaR {int(cl*100)}%',
                value=mod_var,
                confidence_level=cl,
                calculation_method='Cornish-Fisher'
            )
            
            # CVaR
            hist_cvar = self.cvar_calculator.historical_cvar(returns, cl)
            results[f'CVaR_{int(cl*100)}%'] = RiskMetricResult(
                metric_name=f'CVaR {int(cl*100)}%',
                value=hist_cvar,
                confidence_level=cl,
                calculation_method='Historical'
            )
        
        # 波动率指标
        simple_vol = self.volatility_calculator.simple_volatility(returns)
        results['Volatility'] = RiskMetricResult(
            metric_name='Annualized Volatility',
            value=simple_vol,
            calculation_method='Simple'
        )
        
        ewma_vol = self.volatility_calculator.ewma_volatility(returns)
        results['EWMA_Volatility'] = RiskMetricResult(
            metric_name='EWMA Volatility',
            value=ewma_vol,
            calculation_method='EWMA'
        )
        
        # 回撤指标
        if prices is not None:
            max_dd, peak_date, trough_date = self.drawdown_calculator.maximum_drawdown(prices)
            results['Maximum_Drawdown'] = RiskMetricResult(
                metric_name='Maximum Drawdown',
                value=max_dd,
                calculation_method='Historical'
            )
            
            current_dd = self.drawdown_calculator.current_drawdown(prices)
            results['Current_Drawdown'] = RiskMetricResult(
                metric_name='Current Drawdown',
                value=current_dd,
                calculation_method='Current'
            )
        
        # 夏普比率
        sharpe = self.sharpe_calculator.sharpe_ratio(returns, risk_free_rate)
        results['Sharpe_Ratio'] = RiskMetricResult(
            metric_name='Sharpe Ratio',
            value=sharpe,
            calculation_method='Standard'
        )
        
        sortino = self.sharpe_calculator.sortino_ratio(returns, risk_free_rate)
        results['Sortino_Ratio'] = RiskMetricResult(
            metric_name='Sortino Ratio',
            value=sortino,
            calculation_method='Downside Deviation'
        )
        
        # Beta（如果提供市场收益）
        if market_returns is not None:
            beta = self.beta_calculator.market_beta(returns, market_returns)
            results['Beta'] = RiskMetricResult(
                metric_name='Market Beta',
                value=beta,
                calculation_method='Regression'
            )
        
        # Calmar比率（如果提供价格数据）
        if prices is not None:
            calmar = self.sharpe_calculator.calmar_ratio(returns, prices)
            results['Calmar_Ratio'] = RiskMetricResult(
                metric_name='Calmar Ratio',
                value=calmar,
                calculation_method='Return/Max Drawdown'
            )
        
        return results
    
    def calculate_portfolio_risk_metrics(self, 
                                       returns_matrix: pd.DataFrame,
                                       weights: pd.Series,
                                       confidence_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, RiskMetricResult]:
        """计算投资组合风险指标"""
        
        # 计算投资组合收益
        portfolio_returns = (returns_matrix * weights).sum(axis=1)
        
        # 计算投资组合价格（假设初始价格为100）
        portfolio_prices = (1 + portfolio_returns).cumprod() * 100
        
        # 使用综合风险指标计算
        return self.calculate_comprehensive_risk_metrics(
            returns=portfolio_returns,
            prices=portfolio_prices,
            confidence_levels=confidence_levels
        )


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    prices = (1 + returns).cumprod() * 100
    
    # 创建风险指标引擎
    engine = RiskMetricsEngine()
    
    # 计算综合风险指标
    risk_metrics = engine.calculate_comprehensive_risk_metrics(
        returns=returns,
        prices=prices
    )
    
    # 打印结果
    print("风险指标计算结果:")
    print("=" * 50)
    
    for metric_name, result in risk_metrics.items():
        print(f"{result.metric_name}: {result.value:.6f}")
        if result.confidence_level:
            print(f"  置信水平: {result.confidence_level:.1%}")
        print(f"  计算方法: {result.calculation_method}")
        print("-" * 30)