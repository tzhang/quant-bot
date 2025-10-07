"""
核心模块工具函数

提供通用的辅助功能和工具类
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_returns(returns: Union[pd.Series, np.ndarray]) -> bool:
        """验证收益率数据"""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        # 检查是否为空
        if len(returns) == 0:
            return False
        
        # 检查是否包含无效值
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            return False
        
        # 检查收益率范围（-100% 到 1000%）
        if np.any(returns < -1) or np.any(returns > 10):
            return False
        
        return True
    
    @staticmethod
    def validate_prices(prices: Union[pd.Series, np.ndarray]) -> bool:
        """验证价格数据"""
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        # 检查是否为空
        if len(prices) == 0:
            return False
        
        # 检查是否包含无效值
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return False
        
        # 检查价格是否为正数
        if np.any(prices <= 0):
            return False
        
        return True
    
    @staticmethod
    def validate_signals(signals: Union[pd.Series, np.ndarray]) -> bool:
        """验证信号数据"""
        if isinstance(signals, pd.Series):
            signals = signals.values
        
        # 检查是否为空
        if len(signals) == 0:
            return False
        
        # 检查是否包含无效值
        if np.any(np.isnan(signals)) or np.any(np.isinf(signals)):
            return False
        
        return True


class TimeSeriesUtils:
    """时间序列工具类"""
    
    @staticmethod
    def calculate_rolling_statistics(data: pd.Series, 
                                   window: int,
                                   statistics: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """计算滚动统计量"""
        result = pd.DataFrame(index=data.index)
        
        for stat in statistics:
            if stat == 'mean':
                result[f'rolling_{stat}_{window}'] = data.rolling(window).mean()
            elif stat == 'std':
                result[f'rolling_{stat}_{window}'] = data.rolling(window).std()
            elif stat == 'min':
                result[f'rolling_{stat}_{window}'] = data.rolling(window).min()
            elif stat == 'max':
                result[f'rolling_{stat}_{window}'] = data.rolling(window).max()
            elif stat == 'median':
                result[f'rolling_{stat}_{window}'] = data.rolling(window).median()
            elif stat == 'skew':
                result[f'rolling_{stat}_{window}'] = data.rolling(window).skew()
            elif stat == 'kurt':
                result[f'rolling_{stat}_{window}'] = data.rolling(window).kurt()
        
        return result
    
    @staticmethod
    def detect_outliers(data: pd.Series, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.Series:
        """检测异常值"""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold
        
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    @staticmethod
    def fill_missing_values(data: pd.Series, 
                          method: str = 'forward_fill') -> pd.Series:
        """填充缺失值"""
        if method == 'forward_fill':
            return data.fillna(method='ffill')
        elif method == 'backward_fill':
            return data.fillna(method='bfill')
        elif method == 'linear_interpolate':
            return data.interpolate(method='linear')
        elif method == 'mean':
            return data.fillna(data.mean())
        elif method == 'median':
            return data.fillna(data.median())
        elif method == 'zero':
            return data.fillna(0)
        else:
            raise ValueError(f"Unknown fill method: {method}")


class PerformanceUtils:
    """性能计算工具类"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - risk_free_rate / 252
        return excess_returns / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, 
                              risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns.mean() - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.inf
        
        return excess_returns / downside_std * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """计算卡尔玛比率"""
        if len(returns) == 0:
            return 0
        
        annual_return = returns.mean() * 252
        max_drawdown = PerformanceUtils.calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return np.inf
        
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 0
        
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())
    
    @staticmethod
    def calculate_var(returns: pd.Series, 
                     confidence_level: float = 0.05) -> float:
        """计算风险价值（VaR）"""
        if len(returns) == 0:
            return 0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, 
                      confidence_level: float = 0.05) -> float:
        """计算条件风险价值（CVaR）"""
        if len(returns) == 0:
            return 0
        
        var = PerformanceUtils.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return tail_returns.mean()
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> float:
        """计算信息比率"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0
        
        # 对齐时间序列
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns.iloc[-min_length:]
        benchmark_returns = benchmark_returns.iloc[-min_length:]
        
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return excess_returns.mean() / tracking_error * np.sqrt(252)


class RiskUtils:
    """风险计算工具类"""
    
    @staticmethod
    def calculate_beta(returns: pd.Series, 
                      market_returns: pd.Series) -> float:
        """计算贝塔系数"""
        if len(returns) == 0 or len(market_returns) == 0:
            return 0
        
        # 对齐时间序列
        min_length = min(len(returns), len(market_returns))
        returns = returns.iloc[-min_length:]
        market_returns = market_returns.iloc[-min_length:]
        
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 0
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """计算相关性矩阵"""
        return returns_df.corr()
    
    @staticmethod
    def calculate_portfolio_volatility(weights: np.ndarray, 
                                     cov_matrix: np.ndarray) -> float:
        """计算投资组合波动率"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    @staticmethod
    def calculate_portfolio_var(weights: np.ndarray,
                              returns_df: pd.DataFrame,
                              confidence_level: float = 0.05) -> float:
        """计算投资组合VaR"""
        portfolio_returns = (returns_df * weights).sum(axis=1)
        return PerformanceUtils.calculate_var(portfolio_returns, confidence_level)


class OptimizationUtils:
    """优化工具类"""
    
    @staticmethod
    def normalize_weights(weights: np.ndarray) -> np.ndarray:
        """归一化权重"""
        total_weight = np.sum(np.abs(weights))
        if total_weight == 0:
            return weights
        return weights / total_weight
    
    @staticmethod
    def apply_weight_constraints(weights: np.ndarray,
                               min_weight: float = 0.0,
                               max_weight: float = 1.0) -> np.ndarray:
        """应用权重约束"""
        weights = np.clip(weights, min_weight, max_weight)
        return OptimizationUtils.normalize_weights(weights)
    
    @staticmethod
    def calculate_turnover(old_weights: np.ndarray, 
                         new_weights: np.ndarray) -> float:
        """计算换手率"""
        return np.sum(np.abs(new_weights - old_weights)) / 2
    
    @staticmethod
    def apply_turnover_constraint(old_weights: np.ndarray,
                                new_weights: np.ndarray,
                                max_turnover: float) -> np.ndarray:
        """应用换手率约束"""
        current_turnover = OptimizationUtils.calculate_turnover(old_weights, new_weights)
        
        if current_turnover <= max_turnover:
            return new_weights
        
        # 简单的线性缩放方法
        scaling_factor = max_turnover / current_turnover
        adjusted_weights = old_weights + scaling_factor * (new_weights - old_weights)
        
        return OptimizationUtils.normalize_weights(adjusted_weights)


class SignalUtils:
    """信号处理工具类"""
    
    @staticmethod
    def normalize_signal(signal: pd.Series, 
                        method: str = 'zscore') -> pd.Series:
        """标准化信号"""
        if method == 'zscore':
            return (signal - signal.mean()) / signal.std()
        elif method == 'minmax':
            return (signal - signal.min()) / (signal.max() - signal.min())
        elif method == 'rank':
            return signal.rank(pct=True) - 0.5
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def smooth_signal(signal: pd.Series, 
                     method: str = 'ema',
                     window: int = 5) -> pd.Series:
        """平滑信号"""
        if method == 'sma':
            return signal.rolling(window).mean()
        elif method == 'ema':
            return signal.ewm(span=window).mean()
        elif method == 'median':
            return signal.rolling(window).median()
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    @staticmethod
    def combine_signals(signals: pd.DataFrame, 
                       weights: Optional[np.ndarray] = None,
                       method: str = 'weighted_average') -> pd.Series:
        """组合信号"""
        if weights is None:
            weights = np.ones(len(signals.columns)) / len(signals.columns)
        
        if method == 'weighted_average':
            return (signals * weights).sum(axis=1)
        elif method == 'rank_average':
            rank_signals = signals.rank(pct=True, axis=0)
            return (rank_signals * weights).sum(axis=1)
        elif method == 'median':
            return signals.median(axis=1)
        else:
            raise ValueError(f"Unknown combination method: {method}")


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config = {}
    
    def set_config(self, key: str, value: Any):
        """设置配置项"""
        self.config[key] = value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self.config.get(key, default)
    
    def update_config(self, config_dict: Dict[str, Any]):
        """批量更新配置"""
        self.config.update(config_dict)
    
    def load_from_dict(self, config_dict: Dict[str, Any]):
        """从字典加载配置"""
        self.config = config_dict.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()


# 全局配置实例
global_config = ConfigManager()

# 默认配置
DEFAULT_CONFIG = {
    'risk_free_rate': 0.02,
    'trading_cost': 0.001,
    'max_position_size': 0.1,
    'rebalance_frequency': 'daily',
    'lookback_window': 252,
    'min_observations': 30,
    'confidence_level': 0.05,
    'max_drawdown_threshold': 0.2,
    'min_sharpe_ratio': 0.5
}

# 初始化全局配置
global_config.load_from_dict(DEFAULT_CONFIG)