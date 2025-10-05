"""
风险管理模块

实现VaR计算、压力测试、风险归因分析等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
from enum import Enum

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """风险指标类型"""
    VAR = "var"
    CVAR = "cvar"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"


class VaRMethod(Enum):
    """VaR计算方法"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


@dataclass
class RiskMeasure:
    """风险度量结果"""
    metric: RiskMetric
    value: float
    confidence_level: Optional[float] = None
    time_horizon: Optional[int] = None
    method: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self):
        return f"{self.metric.value}: {self.value:.4f}"


@dataclass
class StressTestScenario:
    """压力测试场景"""
    name: str
    description: str
    factor_shocks: Dict[str, float]  # 因子冲击
    probability: Optional[float] = None
    
    def apply_shock(self, factor_exposures: pd.Series) -> float:
        """应用冲击到因子暴露"""
        shock_impact = 0.0
        for factor, shock in self.factor_shocks.items():
            if factor in factor_exposures.index:
                shock_impact += factor_exposures[factor] * shock
        return shock_impact


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario: StressTestScenario
    portfolio_impact: float
    asset_impacts: Dict[str, float]
    factor_contributions: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class BaseRiskModel(ABC):
    """风险模型基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger
        
        # 模型参数
        self.parameters: Dict[str, Any] = {}
        
        # 历史数据
        self.returns_history: Optional[pd.DataFrame] = None
        self.factor_returns: Optional[pd.DataFrame] = None
        
    @abstractmethod
    def calculate_var(self, portfolio_weights: pd.Series,
                     confidence_level: float = 0.05,
                     time_horizon: int = 1) -> RiskMeasure:
        """计算VaR"""
        pass
    
    @abstractmethod
    def calculate_portfolio_risk(self, portfolio_weights: pd.Series) -> Dict[str, RiskMeasure]:
        """计算组合风险指标"""
        pass
    
    def set_returns_data(self, returns: pd.DataFrame):
        """设置收益率数据"""
        self.returns_history = returns
        self.logger.info(f"已设置收益率数据，形状: {returns.shape}")
    
    def set_factor_data(self, factor_returns: pd.DataFrame):
        """设置因子收益率数据"""
        self.factor_returns = factor_returns
        self.logger.info(f"已设置因子数据，形状: {factor_returns.shape}")


class HistoricalVaRModel(BaseRiskModel):
    """历史模拟VaR模型"""
    
    def __init__(self, lookback_window: int = 252):
        super().__init__("HistoricalVaR")
        self.lookback_window = lookback_window
    
    def calculate_var(self, portfolio_weights: pd.Series,
                     confidence_level: float = 0.05,
                     time_horizon: int = 1) -> RiskMeasure:
        """
        计算历史模拟VaR
        
        Args:
            portfolio_weights: 组合权重
            confidence_level: 置信水平
            time_horizon: 时间窗口
            
        Returns:
            RiskMeasure: VaR结果
        """
        if self.returns_history is None:
            raise ValueError("未设置收益率数据")
        
        # 对齐权重和收益率数据
        common_assets = portfolio_weights.index.intersection(self.returns_history.columns)
        if len(common_assets) == 0:
            raise ValueError("组合权重与收益率数据没有共同资产")
        
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        aligned_returns = self.returns_history[common_assets].tail(self.lookback_window)
        
        # 计算组合收益率
        portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
        
        # 调整时间窗口
        if time_horizon > 1:
            portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # 计算VaR
        var_value = np.percentile(portfolio_returns, confidence_level * 100)
        
        return RiskMeasure(
            metric=RiskMetric.VAR,
            value=-var_value,  # VaR通常表示为正值
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method="historical"
        )
    
    def calculate_cvar(self, portfolio_weights: pd.Series,
                      confidence_level: float = 0.05,
                      time_horizon: int = 1) -> RiskMeasure:
        """
        计算条件VaR (CVaR/Expected Shortfall)
        
        Args:
            portfolio_weights: 组合权重
            confidence_level: 置信水平
            time_horizon: 时间窗口
            
        Returns:
            RiskMeasure: CVaR结果
        """
        if self.returns_history is None:
            raise ValueError("未设置收益率数据")
        
        # 对齐权重和收益率数据
        common_assets = portfolio_weights.index.intersection(self.returns_history.columns)
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        aligned_returns = self.returns_history[common_assets].tail(self.lookback_window)
        
        # 计算组合收益率
        portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
        
        # 调整时间窗口
        if time_horizon > 1:
            portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # 计算VaR阈值
        var_threshold = np.percentile(portfolio_returns, confidence_level * 100)
        
        # 计算CVaR（超过VaR的损失的期望值）
        tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
        cvar_value = tail_losses.mean() if len(tail_losses) > 0 else var_threshold
        
        return RiskMeasure(
            metric=RiskMetric.CVAR,
            value=-cvar_value,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method="historical"
        )
    
    def calculate_portfolio_risk(self, portfolio_weights: pd.Series) -> Dict[str, RiskMeasure]:
        """计算组合风险指标"""
        if self.returns_history is None:
            raise ValueError("未设置收益率数据")
        
        risk_measures = {}
        
        # 对齐数据
        common_assets = portfolio_weights.index.intersection(self.returns_history.columns)
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        aligned_returns = self.returns_history[common_assets].tail(self.lookback_window)
        
        # 计算组合收益率
        portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
        
        # VaR (5%)
        var_5 = self.calculate_var(portfolio_weights, 0.05, 1)
        risk_measures['VaR_5%'] = var_5
        
        # VaR (1%)
        var_1 = self.calculate_var(portfolio_weights, 0.01, 1)
        risk_measures['VaR_1%'] = var_1
        
        # CVaR (5%)
        cvar_5 = self.calculate_cvar(portfolio_weights, 0.05, 1)
        risk_measures['CVaR_5%'] = cvar_5
        
        # 波动率
        volatility = RiskMeasure(
            metric=RiskMetric.VOLATILITY,
            value=portfolio_returns.std() * np.sqrt(252),  # 年化波动率
            method="historical"
        )
        risk_measures['Volatility'] = volatility
        
        # 最大回撤
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = RiskMeasure(
            metric=RiskMetric.MAXIMUM_DRAWDOWN,
            value=-drawdown.min(),
            method="historical"
        )
        risk_measures['Max_Drawdown'] = max_drawdown
        
        # 夏普比率
        excess_returns = portfolio_returns - 0.02/252  # 假设无风险利率2%
        sharpe_ratio = RiskMeasure(
            metric=RiskMetric.SHARPE_RATIO,
            value=excess_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            method="historical"
        )
        risk_measures['Sharpe_Ratio'] = sharpe_ratio
        
        # 索提诺比率
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = RiskMeasure(
                metric=RiskMetric.SORTINO_RATIO,
                value=excess_returns.mean() * np.sqrt(252) / downside_deviation,
                method="historical"
            )
        else:
            sortino_ratio = RiskMeasure(
                metric=RiskMetric.SORTINO_RATIO,
                value=float('inf'),
                method="historical"
            )
        risk_measures['Sortino_Ratio'] = sortino_ratio
        
        return risk_measures


class ParametricVaRModel(BaseRiskModel):
    """参数化VaR模型"""
    
    def __init__(self, distribution: str = 'normal'):
        super().__init__("ParametricVaR")
        self.distribution = distribution
        
        # 协方差矩阵
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.expected_returns: Optional[pd.Series] = None
    
    def fit_covariance_matrix(self, returns: pd.DataFrame, 
                             method: str = 'sample') -> pd.DataFrame:
        """
        估计协方差矩阵
        
        Args:
            returns: 收益率数据
            method: 估计方法 ('sample', 'shrinkage', 'ewma')
            
        Returns:
            pd.DataFrame: 协方差矩阵
        """
        if method == 'sample':
            # 样本协方差矩阵
            cov_matrix = returns.cov() * 252  # 年化
            
        elif method == 'shrinkage':
            # 收缩估计
            sample_cov = returns.cov() * 252
            n_assets = len(returns.columns)
            
            # 目标矩阵（单位矩阵乘以平均方差）
            avg_var = np.trace(sample_cov) / n_assets
            target = np.eye(n_assets) * avg_var
            target = pd.DataFrame(target, index=sample_cov.index, columns=sample_cov.columns)
            
            # 收缩参数（简化版）
            shrinkage = 0.2
            cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * target
            
        elif method == 'ewma':
            # 指数加权移动平均
            lambda_param = 0.94
            cov_matrix = returns.ewm(alpha=1-lambda_param).cov().iloc[-len(returns.columns):] * 252
            
        else:
            raise ValueError(f"不支持的协方差估计方法: {method}")
        
        self.covariance_matrix = cov_matrix
        return cov_matrix
    
    def calculate_var(self, portfolio_weights: pd.Series,
                     confidence_level: float = 0.05,
                     time_horizon: int = 1) -> RiskMeasure:
        """
        计算参数化VaR
        
        Args:
            portfolio_weights: 组合权重
            confidence_level: 置信水平
            time_horizon: 时间窗口
            
        Returns:
            RiskMeasure: VaR结果
        """
        if self.covariance_matrix is None:
            if self.returns_history is None:
                raise ValueError("未设置协方差矩阵或收益率数据")
            self.fit_covariance_matrix(self.returns_history)
        
        # 对齐权重和协方差矩阵
        common_assets = portfolio_weights.index.intersection(self.covariance_matrix.index)
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        aligned_cov = self.covariance_matrix.loc[common_assets, common_assets]
        
        # 计算组合方差
        portfolio_variance = aligned_weights.T @ aligned_cov @ aligned_weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # 调整时间窗口
        if time_horizon > 1:
            portfolio_std = portfolio_std * np.sqrt(time_horizon)
        
        # 计算VaR
        if self.distribution == 'normal':
            z_score = stats.norm.ppf(confidence_level)
        elif self.distribution == 't':
            # 使用t分布（自由度需要估计）
            df = 6  # 简化假设
            z_score = stats.t.ppf(confidence_level, df)
        else:
            raise ValueError(f"不支持的分布: {self.distribution}")
        
        # 组合期望收益（如果有的话）
        expected_return = 0.0
        if self.expected_returns is not None:
            aligned_expected = self.expected_returns.reindex(common_assets, fill_value=0)
            expected_return = aligned_weights @ aligned_expected
            if time_horizon > 1:
                expected_return = expected_return * time_horizon
        
        var_value = -(expected_return + z_score * portfolio_std)
        
        return RiskMeasure(
            metric=RiskMetric.VAR,
            value=var_value,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=f"parametric_{self.distribution}"
        )
    
    def calculate_portfolio_risk(self, portfolio_weights: pd.Series) -> Dict[str, RiskMeasure]:
        """计算组合风险指标"""
        if self.covariance_matrix is None:
            if self.returns_history is None:
                raise ValueError("未设置协方差矩阵或收益率数据")
            self.fit_covariance_matrix(self.returns_history)
        
        risk_measures = {}
        
        # VaR
        var_5 = self.calculate_var(portfolio_weights, 0.05, 1)
        risk_measures['VaR_5%'] = var_5
        
        var_1 = self.calculate_var(portfolio_weights, 0.01, 1)
        risk_measures['VaR_1%'] = var_1
        
        # 组合波动率
        common_assets = portfolio_weights.index.intersection(self.covariance_matrix.index)
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        aligned_cov = self.covariance_matrix.loc[common_assets, common_assets]
        
        portfolio_variance = aligned_weights.T @ aligned_cov @ aligned_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        volatility = RiskMeasure(
            metric=RiskMetric.VOLATILITY,
            value=portfolio_volatility,
            method="parametric"
        )
        risk_measures['Volatility'] = volatility
        
        return risk_measures


class MonteCarloVaRModel(BaseRiskModel):
    """蒙特卡洛VaR模型"""
    
    def __init__(self, n_simulations: int = 10000, random_seed: Optional[int] = None):
        super().__init__("MonteCarloVaR")
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def calculate_var(self, portfolio_weights: pd.Series,
                     confidence_level: float = 0.05,
                     time_horizon: int = 1) -> RiskMeasure:
        """
        计算蒙特卡洛VaR
        
        Args:
            portfolio_weights: 组合权重
            confidence_level: 置信水平
            time_horizon: 时间窗口
            
        Returns:
            RiskMeasure: VaR结果
        """
        if self.returns_history is None:
            raise ValueError("未设置收益率数据")
        
        # 对齐数据
        common_assets = portfolio_weights.index.intersection(self.returns_history.columns)
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        aligned_returns = self.returns_history[common_assets]
        
        # 估计参数
        mean_returns = aligned_returns.mean()
        cov_matrix = aligned_returns.cov()
        
        # 蒙特卡洛模拟
        simulated_returns = []
        
        for _ in range(self.n_simulations):
            # 生成随机收益率
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
            random_returns_series = pd.Series(random_returns, index=common_assets)
            
            # 计算组合收益率
            portfolio_return = aligned_weights @ random_returns_series
            
            # 调整时间窗口
            if time_horizon > 1:
                portfolio_return = portfolio_return * np.sqrt(time_horizon)
            
            simulated_returns.append(portfolio_return)
        
        # 计算VaR
        simulated_returns = np.array(simulated_returns)
        var_value = np.percentile(simulated_returns, confidence_level * 100)
        
        return RiskMeasure(
            metric=RiskMetric.VAR,
            value=-var_value,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method="monte_carlo"
        )
    
    def calculate_portfolio_risk(self, portfolio_weights: pd.Series) -> Dict[str, RiskMeasure]:
        """计算组合风险指标"""
        risk_measures = {}
        
        # VaR
        var_5 = self.calculate_var(portfolio_weights, 0.05, 1)
        risk_measures['VaR_5%'] = var_5
        
        var_1 = self.calculate_var(portfolio_weights, 0.01, 1)
        risk_measures['VaR_1%'] = var_1
        
        return risk_measures


class StressTestEngine:
    """压力测试引擎"""
    
    def __init__(self):
        self.scenarios: List[StressTestScenario] = []
        self.factor_model: Optional[Any] = None
        self.logger = logger
        
        # 预定义场景
        self._create_default_scenarios()
    
    def _create_default_scenarios(self):
        """创建默认压力测试场景"""
        # 2008年金融危机场景
        crisis_2008 = StressTestScenario(
            name="2008_Financial_Crisis",
            description="2008年金融危机场景",
            factor_shocks={
                'market': -0.40,      # 市场下跌40%
                'size': -0.20,        # 小盘股相对表现差
                'value': 0.10,        # 价值股相对表现好
                'momentum': -0.15,    # 动量因子反转
                'quality': 0.05,      # 质量因子表现好
                'volatility': 0.30    # 波动率上升
            },
            probability=0.01
        )
        
        # 新冠疫情场景
        covid_2020 = StressTestScenario(
            name="COVID_2020",
            description="2020年新冠疫情场景",
            factor_shocks={
                'market': -0.35,
                'growth': 0.15,       # 成长股表现好
                'tech': 0.20,         # 科技股表现好
                'energy': -0.50,      # 能源股大跌
                'travel': -0.60,      # 旅游相关大跌
                'healthcare': 0.10    # 医疗保健表现好
            },
            probability=0.005
        )
        
        # 利率上升场景
        rate_rise = StressTestScenario(
            name="Interest_Rate_Rise",
            description="利率快速上升场景",
            factor_shocks={
                'market': -0.15,
                'duration': -0.25,    # 久期风险
                'growth': -0.20,      # 成长股受冲击
                'value': 0.10,        # 价值股相对表现好
                'financials': 0.15,   # 金融股受益
                'utilities': -0.15,   # 公用事业受冲击
                'reits': -0.20        # REITs受冲击
            },
            probability=0.10
        )
        
        # 通胀上升场景
        inflation_rise = StressTestScenario(
            name="Inflation_Rise",
            description="通胀快速上升场景",
            factor_shocks={
                'market': -0.10,
                'commodities': 0.30,  # 大宗商品上涨
                'energy': 0.25,       # 能源股受益
                'materials': 0.20,    # 材料股受益
                'tech': -0.15,        # 科技股受冲击
                'bonds': -0.20,       # 债券受冲击
                'reits': 0.10         # REITs部分受益
            },
            probability=0.15
        )
        
        self.scenarios.extend([crisis_2008, covid_2020, rate_rise, inflation_rise])
    
    def add_scenario(self, scenario: StressTestScenario):
        """添加压力测试场景"""
        self.scenarios.append(scenario)
        self.logger.info(f"已添加压力测试场景: {scenario.name}")
    
    def run_stress_test(self, portfolio_weights: pd.Series,
                       factor_exposures: pd.DataFrame,
                       scenarios: Optional[List[str]] = None) -> List[StressTestResult]:
        """
        运行压力测试
        
        Args:
            portfolio_weights: 组合权重
            factor_exposures: 因子暴露度矩阵 (资产 x 因子)
            scenarios: 要测试的场景名称列表，None表示测试所有场景
            
        Returns:
            List[StressTestResult]: 压力测试结果
        """
        if scenarios is None:
            test_scenarios = self.scenarios
        else:
            test_scenarios = [s for s in self.scenarios if s.name in scenarios]
        
        results = []
        
        for scenario in test_scenarios:
            self.logger.info(f"运行压力测试场景: {scenario.name}")
            
            # 计算组合层面的冲击
            portfolio_impact = 0.0
            asset_impacts = {}
            factor_contributions = {}
            
            # 对每个资产计算冲击
            for asset in portfolio_weights.index:
                if asset in factor_exposures.index:
                    asset_exposure = factor_exposures.loc[asset]
                    asset_impact = scenario.apply_shock(asset_exposure)
                    asset_impacts[asset] = asset_impact
                    
                    # 加权到组合层面
                    weight = portfolio_weights[asset]
                    portfolio_impact += weight * asset_impact
                else:
                    asset_impacts[asset] = 0.0
            
            # 计算因子贡献
            for factor, shock in scenario.factor_shocks.items():
                if factor in factor_exposures.columns:
                    factor_exposure = (portfolio_weights * factor_exposures[factor]).sum()
                    factor_contribution = factor_exposure * shock
                    factor_contributions[factor] = factor_contribution
            
            result = StressTestResult(
                scenario=scenario,
                portfolio_impact=portfolio_impact,
                asset_impacts=asset_impacts,
                factor_contributions=factor_contributions
            )
            
            results.append(result)
        
        return results
    
    def create_custom_scenario(self, name: str, description: str,
                              factor_shocks: Dict[str, float],
                              probability: Optional[float] = None) -> StressTestScenario:
        """创建自定义压力测试场景"""
        scenario = StressTestScenario(
            name=name,
            description=description,
            factor_shocks=factor_shocks,
            probability=probability
        )
        
        self.add_scenario(scenario)
        return scenario


class RiskAttributionAnalyzer:
    """风险归因分析器"""
    
    def __init__(self):
        self.logger = logger
    
    def factor_risk_attribution(self, portfolio_weights: pd.Series,
                               factor_exposures: pd.DataFrame,
                               factor_covariance: pd.DataFrame,
                               specific_risk: pd.Series) -> Dict[str, Any]:
        """
        因子风险归因分析
        
        Args:
            portfolio_weights: 组合权重
            factor_exposures: 因子暴露度矩阵
            factor_covariance: 因子协方差矩阵
            specific_risk: 特异性风险
            
        Returns:
            Dict[str, Any]: 风险归因结果
        """
        # 对齐数据
        common_assets = portfolio_weights.index.intersection(factor_exposures.index)
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        aligned_exposures = factor_exposures.loc[common_assets]
        aligned_specific = specific_risk.reindex(common_assets, fill_value=0)
        
        # 计算组合因子暴露
        portfolio_exposures = aligned_weights @ aligned_exposures
        
        # 计算因子风险贡献
        factor_risk_contrib = {}
        total_factor_risk = 0.0
        
        for i, factor_i in enumerate(factor_covariance.index):
            for j, factor_j in enumerate(factor_covariance.columns):
                if i <= j:  # 避免重复计算
                    cov_ij = factor_covariance.loc[factor_i, factor_j]
                    exposure_i = portfolio_exposures[factor_i] if factor_i in portfolio_exposures.index else 0
                    exposure_j = portfolio_exposures[factor_j] if factor_j in portfolio_exposures.index else 0
                    
                    if i == j:
                        # 对角线元素
                        contrib = exposure_i ** 2 * cov_ij
                        factor_risk_contrib[factor_i] = contrib
                        total_factor_risk += contrib
                    else:
                        # 非对角线元素
                        contrib = 2 * exposure_i * exposure_j * cov_ij
                        interaction_key = f"{factor_i}_{factor_j}"
                        factor_risk_contrib[interaction_key] = contrib
                        total_factor_risk += contrib
        
        # 计算特异性风险贡献
        specific_risk_contrib = (aligned_weights ** 2 * aligned_specific ** 2).sum()
        
        # 总风险
        total_risk = total_factor_risk + specific_risk_contrib
        portfolio_volatility = np.sqrt(total_risk)
        
        # 风险贡献百分比
        factor_risk_pct = {k: v / total_risk * 100 for k, v in factor_risk_contrib.items()}
        specific_risk_pct = specific_risk_contrib / total_risk * 100
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'total_risk': total_risk,
            'factor_risk': total_factor_risk,
            'specific_risk': specific_risk_contrib,
            'factor_risk_contributions': factor_risk_contrib,
            'factor_risk_percentages': factor_risk_pct,
            'specific_risk_percentage': specific_risk_pct,
            'portfolio_factor_exposures': portfolio_exposures.to_dict()
        }
    
    def marginal_risk_contribution(self, portfolio_weights: pd.Series,
                                  covariance_matrix: pd.DataFrame) -> pd.Series:
        """
        计算边际风险贡献 (Marginal Contribution to Risk)
        
        Args:
            portfolio_weights: 组合权重
            covariance_matrix: 协方差矩阵
            
        Returns:
            pd.Series: 边际风险贡献
        """
        # 对齐数据
        common_assets = portfolio_weights.index.intersection(covariance_matrix.index)
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        aligned_cov = covariance_matrix.loc[common_assets, common_assets]
        
        # 计算组合方差
        portfolio_variance = aligned_weights.T @ aligned_cov @ aligned_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 边际风险贡献 = (Σw) / σ_p
        marginal_contrib = (aligned_cov @ aligned_weights) / portfolio_volatility
        
        return marginal_contrib
    
    def component_risk_contribution(self, portfolio_weights: pd.Series,
                                   covariance_matrix: pd.DataFrame) -> pd.Series:
        """
        计算成分风险贡献 (Component Contribution to Risk)
        
        Args:
            portfolio_weights: 组合权重
            covariance_matrix: 协方差矩阵
            
        Returns:
            pd.Series: 成分风险贡献
        """
        marginal_contrib = self.marginal_risk_contribution(portfolio_weights, covariance_matrix)
        
        # 成分风险贡献 = w_i * MCR_i
        common_assets = portfolio_weights.index.intersection(covariance_matrix.index)
        aligned_weights = portfolio_weights.reindex(common_assets, fill_value=0)
        
        component_contrib = aligned_weights * marginal_contrib
        
        return component_contrib


class RiskManager:
    """风险管理器"""
    
    def __init__(self):
        self.risk_models: Dict[str, BaseRiskModel] = {}
        self.stress_test_engine = StressTestEngine()
        self.attribution_analyzer = RiskAttributionAnalyzer()
        self.logger = logger
        
        # 注册默认风险模型
        self.register_risk_model("historical", HistoricalVaRModel())
        self.register_risk_model("parametric", ParametricVaRModel())
        self.register_risk_model("monte_carlo", MonteCarloVaRModel())
        
        # 风险限制
        self.risk_limits: Dict[str, float] = {}
        
        # 监控历史
        self.monitoring_history: List[Dict[str, Any]] = []
    
    def register_risk_model(self, name: str, model: BaseRiskModel):
        """注册风险模型"""
        self.risk_models[name] = model
        self.logger.info(f"已注册风险模型: {name}")
    
    def set_risk_limit(self, metric: str, limit: float):
        """设置风险限制"""
        self.risk_limits[metric] = limit
        self.logger.info(f"已设置风险限制: {metric} = {limit}")
    
    def calculate_portfolio_risk(self, portfolio_weights: pd.Series,
                                model_name: str = "historical") -> Dict[str, RiskMeasure]:
        """计算组合风险"""
        if model_name not in self.risk_models:
            raise ValueError(f"未找到风险模型: {model_name}")
        
        model = self.risk_models[model_name]
        return model.calculate_portfolio_risk(portfolio_weights)
    
    def run_comprehensive_risk_analysis(self, portfolio_weights: pd.Series,
                                       returns_data: pd.DataFrame,
                                       factor_exposures: Optional[pd.DataFrame] = None,
                                       factor_covariance: Optional[pd.DataFrame] = None,
                                       specific_risk: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        运行综合风险分析
        
        Args:
            portfolio_weights: 组合权重
            returns_data: 收益率数据
            factor_exposures: 因子暴露度
            factor_covariance: 因子协方差矩阵
            specific_risk: 特异性风险
            
        Returns:
            Dict[str, Any]: 综合风险分析结果
        """
        results = {}
        
        # 设置数据
        for model in self.risk_models.values():
            model.set_returns_data(returns_data)
        
        # 1. 多模型VaR计算
        var_results = {}
        for model_name, model in self.risk_models.items():
            try:
                risk_measures = model.calculate_portfolio_risk(portfolio_weights)
                var_results[model_name] = risk_measures
            except Exception as e:
                self.logger.error(f"模型 {model_name} 计算失败: {e}")
        
        results['var_analysis'] = var_results
        
        # 2. 压力测试
        if factor_exposures is not None:
            stress_results = self.stress_test_engine.run_stress_test(
                portfolio_weights, factor_exposures
            )
            results['stress_test'] = [
                {
                    'scenario': result.scenario.name,
                    'description': result.scenario.description,
                    'portfolio_impact': result.portfolio_impact,
                    'factor_contributions': result.factor_contributions
                }
                for result in stress_results
            ]
        
        # 3. 风险归因分析
        if all(x is not None for x in [factor_exposures, factor_covariance, specific_risk]):
            attribution_results = self.attribution_analyzer.factor_risk_attribution(
                portfolio_weights, factor_exposures, factor_covariance, specific_risk
            )
            results['risk_attribution'] = attribution_results
        
        # 4. 风险限制检查
        risk_limit_violations = self.check_risk_limits(var_results)
        results['risk_limit_violations'] = risk_limit_violations
        
        # 5. 记录监控历史
        monitoring_record = {
            'timestamp': datetime.now(),
            'portfolio_weights': portfolio_weights.to_dict(),
            'risk_measures': {
                model_name: {measure_name: measure.value 
                           for measure_name, measure in measures.items()}
                for model_name, measures in var_results.items()
            },
            'violations': risk_limit_violations
        }
        self.monitoring_history.append(monitoring_record)
        
        return results
    
    def check_risk_limits(self, risk_measures: Dict[str, Dict[str, RiskMeasure]]) -> List[Dict[str, Any]]:
        """检查风险限制"""
        violations = []
        
        for model_name, measures in risk_measures.items():
            for measure_name, measure in measures.items():
                limit_key = f"{model_name}_{measure_name}"
                if limit_key in self.risk_limits:
                    limit = self.risk_limits[limit_key]
                    if measure.value > limit:
                        violations.append({
                            'model': model_name,
                            'measure': measure_name,
                            'value': measure.value,
                            'limit': limit,
                            'excess': measure.value - limit
                        })
        
        return violations
    
    def generate_risk_report(self, analysis_results: Dict[str, Any]) -> str:
        """生成风险报告"""
        report = []
        report.append("=" * 50)
        report.append("风险分析报告")
        report.append("=" * 50)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # VaR分析
        if 'var_analysis' in analysis_results:
            report.append("1. VaR分析结果")
            report.append("-" * 30)
            
            for model_name, measures in analysis_results['var_analysis'].items():
                report.append(f"\n{model_name.upper()}模型:")
                for measure_name, measure in measures.items():
                    report.append(f"  {measure_name}: {measure.value:.4f}")
        
        # 压力测试
        if 'stress_test' in analysis_results:
            report.append("\n\n2. 压力测试结果")
            report.append("-" * 30)
            
            for stress_result in analysis_results['stress_test']:
                report.append(f"\n场景: {stress_result['scenario']}")
                report.append(f"描述: {stress_result['description']}")
                report.append(f"组合影响: {stress_result['portfolio_impact']:.4f}")
                
                if stress_result['factor_contributions']:
                    report.append("因子贡献:")
                    for factor, contrib in stress_result['factor_contributions'].items():
                        report.append(f"  {factor}: {contrib:.4f}")
        
        # 风险归因
        if 'risk_attribution' in analysis_results:
            attribution = analysis_results['risk_attribution']
            report.append("\n\n3. 风险归因分析")
            report.append("-" * 30)
            report.append(f"组合波动率: {attribution['portfolio_volatility']:.4f}")
            report.append(f"因子风险占比: {attribution['factor_risk']/attribution['total_risk']*100:.2f}%")
            report.append(f"特异性风险占比: {attribution['specific_risk_percentage']:.2f}%")
        
        # 风险限制违反
        if 'risk_limit_violations' in analysis_results and analysis_results['risk_limit_violations']:
            report.append("\n\n4. 风险限制违反")
            report.append("-" * 30)
            
            for violation in analysis_results['risk_limit_violations']:
                report.append(f"违反项: {violation['model']}_{violation['measure']}")
                report.append(f"当前值: {violation['value']:.4f}")
                report.append(f"限制值: {violation['limit']:.4f}")
                report.append(f"超出幅度: {violation['excess']:.4f}")
                report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # 测试风险管理模块
    
    # 创建测试数据
    np.random.seed(42)
    n_assets = 10
    n_periods = 252
    
    # 生成随机收益率数据
    returns_data = pd.DataFrame(
        np.random.multivariate_normal(
            mean=np.random.normal(0.0005, 0.001, n_assets),
            cov=np.random.rand(n_assets, n_assets) * 0.0001 + np.eye(n_assets) * 0.0004,
            size=n_periods
        ),
        columns=[f'Asset_{i}' for i in range(n_assets)],
        index=pd.date_range('2023-01-01', periods=n_periods, freq='D')
    )
    
    # 创建组合权重
    portfolio_weights = pd.Series(
        np.random.dirichlet(np.ones(n_assets)),
        index=returns_data.columns
    )
    
    # 创建风险管理器
    risk_manager = RiskManager()
    
    # 设置风险限制
    risk_manager.set_risk_limit('historical_VaR_5%', 0.02)
    risk_manager.set_risk_limit('historical_Volatility', 0.15)
    
    # 运行综合风险分析
    results = risk_manager.run_comprehensive_risk_analysis(
        portfolio_weights, returns_data
    )
    
    # 生成风险报告
    report = risk_manager.generate_risk_report(results)
    print(report)