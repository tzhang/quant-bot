"""
多因子模型框架

实现因子合成、权重优化、风险控制等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import optimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FactorConfig:
    """因子配置"""
    name: str
    category: str  # 'technical', 'fundamental', 'risk', 'alternative'
    weight: float = 1.0
    enabled: bool = True
    neutralize: bool = False  # 是否进行行业中性化
    winsorize: bool = True   # 是否进行去极值处理
    standardize: bool = True  # 是否进行标准化
    lookback_window: int = 252  # 回看窗口
    decay_factor: float = 0.94  # 衰减因子
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("因子权重不能为负数")


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = "linear"  # 'linear', 'ridge', 'lasso', 'rf'
    rebalance_frequency: str = "monthly"  # 'daily', 'weekly', 'monthly', 'quarterly'
    lookback_period: int = 252  # 训练数据回看期
    min_periods: int = 60      # 最小训练期数
    alpha: float = 0.01        # 正则化参数
    max_weight: float = 0.1    # 单个因子最大权重
    risk_budget: float = 0.15  # 风险预算
    turnover_penalty: float = 0.001  # 换手率惩罚
    
    # 风险控制参数
    max_sector_exposure: float = 0.3   # 最大行业暴露
    max_stock_weight: float = 0.05     # 单股最大权重
    min_stock_weight: float = 0.001    # 单股最小权重
    max_tracking_error: float = 0.08   # 最大跟踪误差
    
    def validate(self):
        """验证配置参数"""
        if self.lookback_period < self.min_periods:
            raise ValueError("回看期不能小于最小训练期数")
        if not 0 < self.max_weight <= 1:
            raise ValueError("最大权重必须在(0,1]范围内")
        if self.risk_budget <= 0:
            raise ValueError("风险预算必须为正数")


@dataclass
class FactorExposure:
    """因子暴露度"""
    date: datetime
    symbol: str
    exposures: Dict[str, float]
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    
    def get_exposure(self, factor_name: str) -> float:
        """获取指定因子的暴露度"""
        return self.exposures.get(factor_name, 0.0)


@dataclass
class ModelOutput:
    """模型输出"""
    date: datetime
    factor_returns: Dict[str, float]
    factor_weights: Dict[str, float]
    stock_scores: Dict[str, float]
    portfolio_weights: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_attribution: Dict[str, float]


class FactorProcessor:
    """
    因子处理器
    
    负责因子数据的预处理、清洗、标准化等
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.logger = logger
    
    def winsorize(self, data: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
        """
        去极值处理
        
        Args:
            data: 输入数据
            lower: 下分位数
            upper: 上分位数
            
        Returns:
            pd.Series: 处理后的数据
        """
        lower_bound = data.quantile(lower)
        upper_bound = data.quantile(upper)
        
        return data.clip(lower=lower_bound, upper=upper_bound)
    
    def standardize(self, data: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """
        标准化处理
        
        Args:
            data: 输入数据
            method: 标准化方法 ('zscore', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        if method == "zscore":
            return (data - data.mean()) / data.std()
        elif method == "minmax":
            return (data - data.min()) / (data.max() - data.min())
        elif method == "robust":
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / (1.4826 * mad)
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    def neutralize_industry(self, factor_data: pd.DataFrame, 
                          industry_data: pd.DataFrame) -> pd.DataFrame:
        """
        行业中性化处理
        
        Args:
            factor_data: 因子数据
            industry_data: 行业数据
            
        Returns:
            pd.DataFrame: 中性化后的因子数据
        """
        neutralized_data = factor_data.copy()
        
        for date in factor_data.index:
            if date not in industry_data.index:
                continue
                
            # 获取当日数据
            factors = factor_data.loc[date]
            industries = industry_data.loc[date]
            
            # 对每个因子进行行业中性化
            for factor_name in factor_data.columns:
                if factor_name in factors and not factors[factor_name].isna().all():
                    # 创建行业哑变量
                    industry_dummies = pd.get_dummies(industries, prefix='industry')
                    
                    # 回归去除行业效应
                    valid_idx = ~(factors[factor_name].isna() | industries.isna())
                    
                    if valid_idx.sum() > 10:  # 至少需要10个有效观测
                        X = industry_dummies.loc[valid_idx]
                        y = factors[factor_name].loc[valid_idx]
                        
                        try:
                            reg = LinearRegression().fit(X, y)
                            residuals = y - reg.predict(X)
                            neutralized_data.loc[date, factor_name].loc[valid_idx] = residuals
                        except Exception as e:
                            self.logger.warning(f"行业中性化失败 {factor_name} on {date}: {e}")
        
        return neutralized_data
    
    def process_factors(self, factor_data: pd.DataFrame, 
                       factor_configs: Dict[str, FactorConfig],
                       industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        处理因子数据
        
        Args:
            factor_data: 原始因子数据
            factor_configs: 因子配置
            industry_data: 行业数据（用于中性化）
            
        Returns:
            pd.DataFrame: 处理后的因子数据
        """
        processed_data = factor_data.copy()
        
        for factor_name, config in factor_configs.items():
            if factor_name not in processed_data.columns or not config.enabled:
                continue
            
            factor_series = processed_data[factor_name]
            
            # 去极值
            if config.winsorize:
                for date in processed_data.index:
                    if date in factor_series.index:
                        daily_data = factor_series.loc[date]
                        if not daily_data.isna().all():
                            processed_data.loc[date, factor_name] = self.winsorize(daily_data)
            
            # 标准化
            if config.standardize:
                for date in processed_data.index:
                    if date in factor_series.index:
                        daily_data = processed_data.loc[date, factor_name]
                        if not daily_data.isna().all():
                            processed_data.loc[date, factor_name] = self.standardize(
                                daily_data.to_frame(), method="zscore"
                            ).iloc[:, 0]
        
        # 行业中性化
        if industry_data is not None:
            neutralize_factors = [
                name for name, config in factor_configs.items()
                if config.neutralize and config.enabled
            ]
            
            if neutralize_factors:
                neutralize_data = processed_data[neutralize_factors]
                neutralized = self.neutralize_industry(neutralize_data, industry_data)
                processed_data[neutralize_factors] = neutralized
        
        return processed_data


class RiskModel:
    """
    风险模型
    
    实现因子风险模型，计算因子协方差矩阵和特异性风险
    """
    
    def __init__(self, lookback_window: int = 252, decay_factor: float = 0.94):
        """
        初始化风险模型
        
        Args:
            lookback_window: 回看窗口
            decay_factor: 衰减因子
        """
        self.lookback_window = lookback_window
        self.decay_factor = decay_factor
        self.logger = logger
        
        # 风险模型组件
        self.factor_covariance: Optional[pd.DataFrame] = None
        self.specific_risk: Optional[pd.Series] = None
        self.factor_loadings: Optional[pd.DataFrame] = None
    
    def estimate_factor_returns(self, returns: pd.DataFrame, 
                              factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """
        估计因子收益率
        
        Args:
            returns: 股票收益率
            factor_exposures: 因子暴露度
            
        Returns:
            pd.DataFrame: 因子收益率
        """
        factor_returns = pd.DataFrame(index=returns.index, columns=factor_exposures.columns)
        
        for date in returns.index:
            if date not in factor_exposures.index:
                continue
            
            # 获取当日数据
            daily_returns = returns.loc[date].dropna()
            daily_exposures = factor_exposures.loc[date].reindex(daily_returns.index).dropna()
            
            # 确保数据对齐
            common_stocks = daily_returns.index.intersection(daily_exposures.index)
            
            if len(common_stocks) < 10:  # 至少需要10只股票
                continue
            
            y = daily_returns.loc[common_stocks]
            X = daily_exposures.loc[common_stocks]
            
            # 使用加权最小二乘法估计因子收益率
            try:
                # 市值加权（如果有市值数据）
                weights = np.ones(len(y))  # 等权重，实际应用中可以使用市值权重
                
                # 加权回归
                W = np.diag(weights)
                XtWX_inv = np.linalg.inv(X.T @ W @ X)
                factor_ret = XtWX_inv @ X.T @ W @ y
                
                factor_returns.loc[date] = factor_ret
                
            except Exception as e:
                self.logger.warning(f"因子收益率估计失败 {date}: {e}")
        
        return factor_returns.dropna(how='all')
    
    def estimate_factor_covariance(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        估计因子协方差矩阵
        
        Args:
            factor_returns: 因子收益率
            
        Returns:
            pd.DataFrame: 因子协方差矩阵
        """
        # 使用指数加权移动平均估计协方差
        weights = np.array([
            self.decay_factor ** i 
            for i in range(len(factor_returns))
        ])[::-1]
        
        # 标准化权重
        weights = weights / weights.sum()
        
        # 计算加权协方差矩阵
        factor_returns_centered = factor_returns - factor_returns.mean()
        
        cov_matrix = pd.DataFrame(
            index=factor_returns.columns,
            columns=factor_returns.columns,
            dtype=float
        )
        
        for i, factor1 in enumerate(factor_returns.columns):
            for j, factor2 in enumerate(factor_returns.columns):
                if i <= j:  # 利用对称性
                    series1 = factor_returns_centered[factor1].dropna()
                    series2 = factor_returns_centered[factor2].dropna()
                    
                    # 找到共同的日期
                    common_dates = series1.index.intersection(series2.index)
                    
                    if len(common_dates) >= 60:  # 至少需要60个观测
                        s1 = series1.loc[common_dates]
                        s2 = series2.loc[common_dates]
                        w = weights[-len(common_dates):]  # 取最近的权重
                        
                        covariance = np.sum(w * s1 * s2)
                        cov_matrix.loc[factor1, factor2] = covariance
                        cov_matrix.loc[factor2, factor1] = covariance
        
        return cov_matrix.astype(float)
    
    def estimate_specific_risk(self, returns: pd.DataFrame, 
                             factor_returns: pd.DataFrame,
                             factor_exposures: pd.DataFrame) -> pd.Series:
        """
        估计特异性风险
        
        Args:
            returns: 股票收益率
            factor_returns: 因子收益率
            factor_exposures: 因子暴露度
            
        Returns:
            pd.Series: 特异性风险
        """
        specific_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        # 计算特异性收益率
        for date in returns.index:
            if date not in factor_returns.index or date not in factor_exposures.index:
                continue
            
            daily_returns = returns.loc[date]
            daily_factor_returns = factor_returns.loc[date]
            daily_exposures = factor_exposures.loc[date]
            
            # 计算因子贡献的收益率
            factor_contribution = daily_exposures @ daily_factor_returns
            
            # 特异性收益率 = 总收益率 - 因子贡献收益率
            specific_returns.loc[date] = daily_returns - factor_contribution
        
        # 计算特异性风险（标准差）
        specific_risk = specific_returns.std(axis=0, skipna=True)
        
        # 使用指数加权移动平均平滑特异性风险
        if len(specific_returns) > self.lookback_window:
            weights = np.array([
                self.decay_factor ** i 
                for i in range(self.lookback_window)
            ])[::-1]
            weights = weights / weights.sum()
            
            recent_data = specific_returns.tail(self.lookback_window)
            specific_risk = np.sqrt(
                (weights[:, np.newaxis] * recent_data.fillna(0) ** 2).sum(axis=0)
            )
            specific_risk = pd.Series(specific_risk, index=returns.columns)
        
        return specific_risk.fillna(specific_risk.median())
    
    def build_risk_model(self, returns: pd.DataFrame, 
                        factor_exposures: pd.DataFrame) -> Dict[str, Any]:
        """
        构建完整的风险模型
        
        Args:
            returns: 股票收益率
            factor_exposures: 因子暴露度
            
        Returns:
            Dict[str, Any]: 风险模型组件
        """
        self.logger.info("开始构建风险模型...")
        
        # 估计因子收益率
        factor_returns = self.estimate_factor_returns(returns, factor_exposures)
        
        # 估计因子协方差矩阵
        self.factor_covariance = self.estimate_factor_covariance(factor_returns)
        
        # 估计特异性风险
        self.specific_risk = self.estimate_specific_risk(
            returns, factor_returns, factor_exposures
        )
        
        # 保存因子载荷
        self.factor_loadings = factor_exposures
        
        self.logger.info("风险模型构建完成")
        
        return {
            'factor_returns': factor_returns,
            'factor_covariance': self.factor_covariance,
            'specific_risk': self.specific_risk,
            'factor_loadings': self.factor_loadings
        }
    
    def calculate_portfolio_risk(self, weights: pd.Series, 
                               factor_exposures: pd.Series) -> float:
        """
        计算组合风险
        
        Args:
            weights: 组合权重
            factor_exposures: 因子暴露度
            
        Returns:
            float: 组合风险（年化波动率）
        """
        if self.factor_covariance is None or self.specific_risk is None:
            raise ValueError("风险模型尚未构建")
        
        # 组合的因子暴露度
        portfolio_exposures = factor_exposures @ weights
        
        # 因子风险贡献
        factor_risk = portfolio_exposures.T @ self.factor_covariance @ portfolio_exposures
        
        # 特异性风险贡献
        specific_risk_contrib = (weights ** 2 @ self.specific_risk ** 2)
        
        # 总风险
        total_risk = np.sqrt(factor_risk + specific_risk_contrib)
        
        # 年化
        return total_risk * np.sqrt(252)


class MultiFactorModel:
    """
    多因子模型
    
    实现因子合成、权重优化、风险控制等功能
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化多因子模型
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.config.validate()
        
        self.logger = logger
        
        # 模型组件
        self.factor_processor = FactorProcessor()
        self.risk_model = RiskModel()
        
        # 因子配置
        self.factor_configs: Dict[str, FactorConfig] = {}
        
        # 模型状态
        self.is_fitted = False
        self.last_rebalance_date: Optional[datetime] = None
        
        # 历史数据
        self.factor_returns_history: List[pd.DataFrame] = []
        self.portfolio_weights_history: List[Dict[str, float]] = []
        self.performance_history: List[Dict[str, float]] = []
        
        self.logger.info(f"多因子模型已初始化，模型类型: {config.model_type}")
    
    def add_factor(self, factor_config: FactorConfig):
        """
        添加因子
        
        Args:
            factor_config: 因子配置
        """
        self.factor_configs[factor_config.name] = factor_config
        self.logger.info(f"已添加因子: {factor_config.name} ({factor_config.category})")
    
    def remove_factor(self, factor_name: str):
        """
        移除因子
        
        Args:
            factor_name: 因子名称
        """
        if factor_name in self.factor_configs:
            del self.factor_configs[factor_name]
            self.logger.info(f"已移除因子: {factor_name}")
    
    def update_factor_weight(self, factor_name: str, weight: float):
        """
        更新因子权重
        
        Args:
            factor_name: 因子名称
            weight: 新权重
        """
        if factor_name in self.factor_configs:
            self.factor_configs[factor_name].weight = weight
            self.logger.info(f"已更新因子权重: {factor_name} = {weight}")
    
    def fit(self, returns: pd.DataFrame, factor_data: pd.DataFrame,
            industry_data: Optional[pd.DataFrame] = None,
            market_cap_data: Optional[pd.DataFrame] = None) -> 'MultiFactorModel':
        """
        训练模型
        
        Args:
            returns: 股票收益率数据
            factor_data: 因子数据
            industry_data: 行业数据
            market_cap_data: 市值数据
            
        Returns:
            MultiFactorModel: 训练后的模型
        """
        self.logger.info("开始训练多因子模型...")
        
        # 数据预处理
        processed_factors = self.factor_processor.process_factors(
            factor_data, self.factor_configs, industry_data
        )
        
        # 构建风险模型
        risk_model_components = self.risk_model.build_risk_model(
            returns, processed_factors
        )
        
        # 保存训练数据
        self.returns_data = returns
        self.factor_data = processed_factors
        self.industry_data = industry_data
        self.market_cap_data = market_cap_data
        
        self.is_fitted = True
        self.logger.info("多因子模型训练完成")
        
        return self
    
    def predict_returns(self, date: datetime, 
                       factor_exposures: pd.DataFrame) -> pd.Series:
        """
        预测股票收益率
        
        Args:
            date: 预测日期
            factor_exposures: 因子暴露度
            
        Returns:
            pd.Series: 预测收益率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 获取因子权重
        factor_weights = self._get_factor_weights(date)
        
        # 计算预测收益率
        predicted_returns = factor_exposures @ factor_weights
        
        return predicted_returns
    
    def _get_factor_weights(self, date: datetime) -> pd.Series:
        """
        获取因子权重
        
        Args:
            date: 日期
            
        Returns:
            pd.Series: 因子权重
        """
        # 获取训练数据
        end_date = date
        start_date = end_date - timedelta(days=self.config.lookback_period)
        
        # 筛选训练期数据
        train_returns = self.returns_data.loc[start_date:end_date]
        train_factors = self.factor_data.loc[start_date:end_date]
        
        if len(train_returns) < self.config.min_periods:
            self.logger.warning(f"训练数据不足: {len(train_returns)} < {self.config.min_periods}")
            # 使用等权重
            return pd.Series(
                [1.0 / len(self.factor_configs)] * len(self.factor_configs),
                index=train_factors.columns
            )
        
        # 根据模型类型选择算法
        if self.config.model_type == "linear":
            return self._fit_linear_model(train_returns, train_factors)
        elif self.config.model_type == "ridge":
            return self._fit_ridge_model(train_returns, train_factors)
        elif self.config.model_type == "lasso":
            return self._fit_lasso_model(train_returns, train_factors)
        elif self.config.model_type == "rf":
            return self._fit_random_forest_model(train_returns, train_factors)
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model_type}")
    
    def _fit_linear_model(self, returns: pd.DataFrame, 
                         factors: pd.DataFrame) -> pd.Series:
        """拟合线性模型"""
        # 准备数据
        y_data = []
        X_data = []
        
        for date in returns.index:
            if date in factors.index:
                daily_returns = returns.loc[date].dropna()
                daily_factors = factors.loc[date].reindex(daily_returns.index).dropna()
                
                common_stocks = daily_returns.index.intersection(daily_factors.index)
                
                if len(common_stocks) >= 10:
                    y_data.append(daily_returns.loc[common_stocks])
                    X_data.append(daily_factors.loc[common_stocks])
        
        if not y_data:
            return pd.Series(0.0, index=factors.columns)
        
        # 合并数据
        y = pd.concat(y_data)
        X = pd.concat(X_data)
        
        # 拟合模型
        model = LinearRegression()
        model.fit(X, y)
        
        return pd.Series(model.coef_, index=factors.columns)
    
    def _fit_ridge_model(self, returns: pd.DataFrame, 
                        factors: pd.DataFrame) -> pd.Series:
        """拟合Ridge回归模型"""
        # 准备数据（同线性模型）
        y_data = []
        X_data = []
        
        for date in returns.index:
            if date in factors.index:
                daily_returns = returns.loc[date].dropna()
                daily_factors = factors.loc[date].reindex(daily_returns.index).dropna()
                
                common_stocks = daily_returns.index.intersection(daily_factors.index)
                
                if len(common_stocks) >= 10:
                    y_data.append(daily_returns.loc[common_stocks])
                    X_data.append(daily_factors.loc[common_stocks])
        
        if not y_data:
            return pd.Series(0.0, index=factors.columns)
        
        # 合并数据
        y = pd.concat(y_data)
        X = pd.concat(X_data)
        
        # 拟合Ridge模型
        model = Ridge(alpha=self.config.alpha)
        model.fit(X, y)
        
        return pd.Series(model.coef_, index=factors.columns)
    
    def _fit_lasso_model(self, returns: pd.DataFrame, 
                        factors: pd.DataFrame) -> pd.Series:
        """拟合Lasso回归模型"""
        # 准备数据（同线性模型）
        y_data = []
        X_data = []
        
        for date in returns.index:
            if date in factors.index:
                daily_returns = returns.loc[date].dropna()
                daily_factors = factors.loc[date].reindex(daily_returns.index).dropna()
                
                common_stocks = daily_returns.index.intersection(daily_factors.index)
                
                if len(common_stocks) >= 10:
                    y_data.append(daily_returns.loc[common_stocks])
                    X_data.append(daily_factors.loc[common_stocks])
        
        if not y_data:
            return pd.Series(0.0, index=factors.columns)
        
        # 合并数据
        y = pd.concat(y_data)
        X = pd.concat(X_data)
        
        # 拟合Lasso模型
        model = Lasso(alpha=self.config.alpha)
        model.fit(X, y)
        
        return pd.Series(model.coef_, index=factors.columns)
    
    def _fit_random_forest_model(self, returns: pd.DataFrame, 
                               factors: pd.DataFrame) -> pd.Series:
        """拟合随机森林模型"""
        # 准备数据（同线性模型）
        y_data = []
        X_data = []
        
        for date in returns.index:
            if date in factors.index:
                daily_returns = returns.loc[date].dropna()
                daily_factors = factors.loc[date].reindex(daily_returns.index).dropna()
                
                common_stocks = daily_returns.index.intersection(daily_factors.index)
                
                if len(common_stocks) >= 10:
                    y_data.append(daily_returns.loc[common_stocks])
                    X_data.append(daily_factors.loc[common_stocks])
        
        if not y_data:
            return pd.Series(0.0, index=factors.columns)
        
        # 合并数据
        y = pd.concat(y_data)
        X = pd.concat(X_data)
        
        # 拟合随机森林模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 使用特征重要性作为权重
        return pd.Series(model.feature_importances_, index=factors.columns)
    
    def optimize_portfolio(self, date: datetime, 
                          expected_returns: pd.Series,
                          factor_exposures: pd.DataFrame,
                          benchmark_weights: Optional[pd.Series] = None) -> pd.Series:
        """
        优化组合权重
        
        Args:
            date: 优化日期
            expected_returns: 预期收益率
            factor_exposures: 因子暴露度
            benchmark_weights: 基准权重
            
        Returns:
            pd.Series: 优化后的组合权重
        """
        n_assets = len(expected_returns)
        
        # 初始权重（等权重或基准权重）
        if benchmark_weights is not None:
            initial_weights = benchmark_weights.reindex(expected_returns.index, fill_value=0)
        else:
            initial_weights = pd.Series(1.0 / n_assets, index=expected_returns.index)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 权重和为1
        ]
        
        # 权重边界
        bounds = [
            (self.config.min_stock_weight, self.config.max_stock_weight)
            for _ in range(n_assets)
        ]
        
        # 目标函数：最大化预期收益 - 风险惩罚 - 换手率惩罚
        def objective(weights):
            w = pd.Series(weights, index=expected_returns.index)
            
            # 预期收益
            expected_return = w @ expected_returns
            
            # 风险惩罚
            portfolio_risk = self.risk_model.calculate_portfolio_risk(
                w, factor_exposures.loc[date]
            )
            risk_penalty = self.config.risk_budget * portfolio_risk ** 2
            
            # 换手率惩罚
            if len(self.portfolio_weights_history) > 0:
                last_weights = pd.Series(self.portfolio_weights_history[-1])
                last_weights = last_weights.reindex(w.index, fill_value=0)
                turnover = np.sum(np.abs(w - last_weights))
                turnover_penalty = self.config.turnover_penalty * turnover
            else:
                turnover_penalty = 0
            
            # 最小化负收益 + 风险惩罚 + 换手率惩罚
            return -(expected_return - risk_penalty - turnover_penalty)
        
        # 优化
        try:
            result = optimize.minimize(
                objective,
                initial_weights.values,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
                
                # 确保权重满足约束
                optimal_weights = optimal_weights.clip(
                    lower=self.config.min_stock_weight,
                    upper=self.config.max_stock_weight
                )
                
                # 重新标准化
                optimal_weights = optimal_weights / optimal_weights.sum()
                
                return optimal_weights
            else:
                self.logger.warning(f"组合优化失败: {result.message}")
                return initial_weights
                
        except Exception as e:
            self.logger.error(f"组合优化异常: {e}")
            return initial_weights
    
    def generate_signals(self, date: datetime) -> ModelOutput:
        """
        生成交易信号
        
        Args:
            date: 信号日期
            
        Returns:
            ModelOutput: 模型输出
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 获取当日因子暴露度
        if date not in self.factor_data.index:
            raise ValueError(f"日期 {date} 的因子数据不存在")
        
        factor_exposures = self.factor_data.loc[date:date]
        
        # 预测收益率
        expected_returns = self.predict_returns(date, factor_exposures)
        
        # 获取因子权重
        factor_weights = self._get_factor_weights(date)
        
        # 优化组合权重
        portfolio_weights = self.optimize_portfolio(
            date, expected_returns, self.factor_data
        )
        
        # 计算风险指标
        portfolio_risk = self.risk_model.calculate_portfolio_risk(
            portfolio_weights, factor_exposures.iloc[0]
        )
        
        # 计算因子收益率（如果有历史数据）
        factor_returns = {}
        if len(self.factor_returns_history) > 0:
            recent_factor_returns = self.factor_returns_history[-1]
            if date in recent_factor_returns.index:
                factor_returns = recent_factor_returns.loc[date].to_dict()
        
        # 性能归因分析
        performance_attribution = self._calculate_performance_attribution(
            portfolio_weights, factor_exposures.iloc[0], factor_returns
        )
        
        # 创建模型输出
        output = ModelOutput(
            date=date,
            factor_returns=factor_returns,
            factor_weights=factor_weights.to_dict(),
            stock_scores=expected_returns.to_dict(),
            portfolio_weights=portfolio_weights.to_dict(),
            risk_metrics={
                'portfolio_risk': portfolio_risk,
                'tracking_error': portfolio_risk,  # 简化处理
                'max_weight': portfolio_weights.max(),
                'min_weight': portfolio_weights.min(),
                'concentration': (portfolio_weights ** 2).sum()
            },
            performance_attribution=performance_attribution
        )
        
        # 保存历史记录
        self.portfolio_weights_history.append(portfolio_weights.to_dict())
        
        return output
    
    def _calculate_performance_attribution(self, weights: pd.Series, 
                                         factor_exposures: pd.Series,
                                         factor_returns: Dict[str, float]) -> Dict[str, float]:
        """计算性能归因"""
        attribution = {}
        
        if not factor_returns:
            return attribution
        
        # 计算各因子的贡献
        for factor_name in factor_exposures.index:
            if factor_name in factor_returns:
                # 组合在该因子上的暴露度
                portfolio_exposure = weights @ factor_exposures
                
                # 该因子的收益贡献
                factor_contribution = portfolio_exposure * factor_returns[factor_name]
                attribution[f"{factor_name}_contribution"] = factor_contribution
        
        return attribution
    
    def backtest(self, start_date: datetime, end_date: datetime,
                initial_capital: float = 1000000) -> Dict[str, Any]:
        """
        回测模型
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        self.logger.info(f"开始回测: {start_date} 到 {end_date}")
        
        # 回测结果存储
        backtest_results = {
            'dates': [],
            'portfolio_values': [],
            'returns': [],
            'positions': [],
            'turnover': [],
            'risk_metrics': []
        }
        
        current_value = initial_capital
        current_positions = {}
        
        # 获取回测日期序列
        backtest_dates = pd.date_range(start_date, end_date, freq='D')
        backtest_dates = [d for d in backtest_dates if d in self.factor_data.index]
        
        for i, date in enumerate(backtest_dates):
            try:
                # 生成信号
                signals = self.generate_signals(date)
                
                # 计算当日收益率
                if i > 0 and current_positions:
                    daily_return = self._calculate_daily_return(
                        date, current_positions
                    )
                    current_value *= (1 + daily_return)
                    backtest_results['returns'].append(daily_return)
                else:
                    backtest_results['returns'].append(0.0)
                
                # 更新持仓
                new_positions = {
                    symbol: weight * current_value
                    for symbol, weight in signals.portfolio_weights.items()
                }
                
                # 计算换手率
                if current_positions:
                    turnover = self._calculate_turnover(current_positions, new_positions)
                else:
                    turnover = 1.0  # 初始建仓
                
                current_positions = new_positions
                
                # 记录结果
                backtest_results['dates'].append(date)
                backtest_results['portfolio_values'].append(current_value)
                backtest_results['positions'].append(current_positions.copy())
                backtest_results['turnover'].append(turnover)
                backtest_results['risk_metrics'].append(signals.risk_metrics)
                
                if i % 50 == 0:
                    self.logger.info(f"回测进度: {i+1}/{len(backtest_dates)}")
                    
            except Exception as e:
                self.logger.error(f"回测日期 {date} 处理失败: {e}")
                continue
        
        # 计算回测统计指标
        returns_series = pd.Series(backtest_results['returns'], index=backtest_results['dates'])
        
        backtest_stats = self._calculate_backtest_statistics(
            returns_series, backtest_results['portfolio_values']
        )
        
        self.logger.info("回测完成")
        
        return {
            'results': backtest_results,
            'statistics': backtest_stats
        }
    
    def _calculate_daily_return(self, date: datetime, 
                              positions: Dict[str, float]) -> float:
        """计算当日收益率"""
        if date not in self.returns_data.index:
            return 0.0
        
        daily_returns = self.returns_data.loc[date]
        total_return = 0.0
        total_value = sum(positions.values())
        
        for symbol, position_value in positions.items():
            if symbol in daily_returns:
                weight = position_value / total_value
                total_return += weight * daily_returns[symbol]
        
        return total_return
    
    def _calculate_turnover(self, old_positions: Dict[str, float],
                          new_positions: Dict[str, float]) -> float:
        """计算换手率"""
        old_total = sum(old_positions.values())
        new_total = sum(new_positions.values())
        
        if old_total == 0:
            return 1.0
        
        turnover = 0.0
        all_symbols = set(old_positions.keys()) | set(new_positions.keys())
        
        for symbol in all_symbols:
            old_weight = old_positions.get(symbol, 0) / old_total
            new_weight = new_positions.get(symbol, 0) / new_total
            turnover += abs(new_weight - old_weight)
        
        return turnover / 2  # 单边换手率
    
    def _calculate_backtest_statistics(self, returns: pd.Series, 
                                     portfolio_values: List[float]) -> Dict[str, float]:
        """计算回测统计指标"""
        if len(returns) == 0:
            return {}
        
        # 基础统计
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # Calmar比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(returns)
        }
    
    def get_factor_analysis(self) -> Dict[str, Any]:
        """
        获取因子分析报告
        
        Returns:
            Dict[str, Any]: 因子分析报告
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        analysis = {
            'factor_configs': {
                name: {
                    'category': config.category,
                    'weight': config.weight,
                    'enabled': config.enabled,
                    'neutralize': config.neutralize
                }
                for name, config in self.factor_configs.items()
            },
            'factor_statistics': {},
            'correlation_matrix': None,
            'factor_loadings': None
        }
        
        # 因子统计
        for factor_name in self.factor_data.columns:
            factor_series = self.factor_data[factor_name].stack()
            
            analysis['factor_statistics'][factor_name] = {
                'mean': factor_series.mean(),
                'std': factor_series.std(),
                'skewness': factor_series.skew(),
                'kurtosis': factor_series.kurtosis(),
                'min': factor_series.min(),
                'max': factor_series.max()
            }
        
        # 因子相关性矩阵
        if len(self.factor_data.columns) > 1:
            # 计算因子间的平均相关性
            correlations = []
            for date in self.factor_data.index:
                daily_factors = self.factor_data.loc[date].dropna()
                if len(daily_factors) > 1:
                    corr_matrix = daily_factors.T.corr()
                    correlations.append(corr_matrix)
            
            if correlations:
                avg_correlation = pd.concat(correlations).groupby(level=[0, 1]).mean()
                analysis['correlation_matrix'] = avg_correlation.to_dict()
        
        return analysis


if __name__ == "__main__":
    # 测试多因子模型
    
    # 创建模型配置
    config = ModelConfig(
        model_type="ridge",
        rebalance_frequency="monthly",
        lookback_period=252,
        alpha=0.01
    )
    
    # 创建模型
    model = MultiFactorModel(config)
    
    # 添加因子
    model.add_factor(FactorConfig(
        name="momentum",
        category="technical",
        weight=0.3,
        neutralize=True
    ))
    
    model.add_factor(FactorConfig(
        name="value",
        category="fundamental", 
        weight=0.4,
        neutralize=True
    ))
    
    model.add_factor(FactorConfig(
        name="quality",
        category="fundamental",
        weight=0.3,
        neutralize=False
    ))
    
    print("多因子模型配置完成")
    print(f"因子数量: {len(model.factor_configs)}")
    
    # 获取因子分析
    analysis = model.get_factor_analysis()
    print(f"因子分析: {analysis['factor_configs']}")