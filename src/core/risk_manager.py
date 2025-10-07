"""
风险管理模块

提供自适应风险管理、市场状态检测和波动率预测功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class AdaptiveRiskManager:
    """自适应风险管理器"""
    
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_portfolio_risk: float = 0.02,
                 lookback_period: int = 252):
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.lookback_period = lookback_period
        self.risk_metrics = {}
        
    def calculate_position_size(self, 
                              signal_strength: float,
                              volatility: float,
                              confidence: float = 1.0) -> float:
        """计算仓位大小"""
        # Kelly公式调整
        kelly_fraction = abs(signal_strength) * confidence / (volatility ** 2)
        
        # 应用最大仓位限制
        position_size = min(kelly_fraction, self.max_position_size)
        
        # 根据信号方向调整
        return position_size * np.sign(signal_strength)
    
    def calculate_portfolio_risk(self, 
                               positions: Dict[str, float],
                               returns_data: pd.DataFrame) -> Dict[str, float]:
        """计算投资组合风险"""
        # 计算协方差矩阵
        cov_matrix = returns_data.cov() * 252  # 年化
        
        # 转换仓位为向量
        assets = list(positions.keys())
        weights = np.array([positions.get(asset, 0) for asset in assets])
        
        # 计算投资组合方差
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # VaR计算 (95%置信度)
        var_95 = 1.645 * portfolio_volatility
        
        # CVaR计算
        cvar_95 = var_95 * 1.2  # 简化计算
        
        self.risk_metrics = {
            'portfolio_volatility': portfolio_volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': self._calculate_max_drawdown(returns_data, weights)
        }
        
        return self.risk_metrics
    
    def _calculate_max_drawdown(self, returns_data: pd.DataFrame, weights: np.ndarray) -> float:
        """计算最大回撤"""
        if len(weights) != len(returns_data.columns):
            return 0.0
            
        portfolio_returns = (returns_data * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # 计算回撤
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        return abs(drawdown.min())
    
    def adjust_positions_for_risk(self, 
                                positions: Dict[str, float],
                                returns_data: pd.DataFrame) -> Dict[str, float]:
        """根据风险调整仓位"""
        risk_metrics = self.calculate_portfolio_risk(positions, returns_data)
        
        # 如果投资组合风险超过限制，按比例缩减
        if risk_metrics['portfolio_volatility'] > self.max_portfolio_risk:
            scale_factor = self.max_portfolio_risk / risk_metrics['portfolio_volatility']
            positions = {k: v * scale_factor for k, v in positions.items()}
        
        return positions
    
    def get_risk_budget(self, 
                       current_positions: Dict[str, float],
                       returns_data: pd.DataFrame) -> Dict[str, float]:
        """计算风险预算"""
        risk_metrics = self.calculate_portfolio_risk(current_positions, returns_data)
        
        # 剩余风险预算
        remaining_risk = max(0, self.max_portfolio_risk - risk_metrics['portfolio_volatility'])
        
        return {
            'used_risk': risk_metrics['portfolio_volatility'],
            'remaining_risk': remaining_risk,
            'risk_utilization': risk_metrics['portfolio_volatility'] / self.max_portfolio_risk
        }


class MarketRegimeDetector:
    """市场状态检测器"""
    
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        self.regimes = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol']
        self.current_regime = None
        self.regime_history = []
        
    def detect_regime(self, data: pd.DataFrame) -> str:
        """检测当前市场状态"""
        if len(data) < self.lookback_period:
            return 'unknown'
        
        recent_data = data.tail(self.lookback_period)
        
        # 计算关键指标
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        trend = self._calculate_trend(recent_data['close'])
        
        # 状态判断逻辑
        regime = self._classify_regime(returns, volatility, trend)
        
        self.current_regime = regime
        self.regime_history.append({
            'timestamp': data.index[-1],
            'regime': regime,
            'volatility': volatility,
            'trend': trend
        })
        
        return regime
    
    def _calculate_trend(self, prices: pd.Series) -> float:
        """计算趋势强度"""
        # 使用线性回归斜率
        x = np.arange(len(prices))
        y = prices.values
        
        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]
        
        # 标准化斜率
        return slope / prices.mean()
    
    def _classify_regime(self, returns: pd.Series, volatility: float, trend: float) -> str:
        """分类市场状态"""
        # 波动率阈值
        vol_threshold_high = 0.25
        vol_threshold_low = 0.10
        
        # 趋势阈值
        trend_threshold = 0.001
        
        # 高波动率状态
        if volatility > vol_threshold_high:
            return 'high_vol'
        
        # 低波动率状态
        if volatility < vol_threshold_low:
            return 'low_vol'
        
        # 趋势状态
        if trend > trend_threshold:
            return 'bull'
        elif trend < -trend_threshold:
            return 'bear'
        else:
            return 'sideways'
    
    def get_regime_probabilities(self, data: pd.DataFrame) -> Dict[str, float]:
        """获取各状态的概率"""
        if len(data) < self.lookback_period:
            return {regime: 1.0/len(self.regimes) for regime in self.regimes}
        
        recent_data = data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        trend = self._calculate_trend(recent_data['close'])
        
        # 基于模糊逻辑的概率计算
        probabilities = {}
        
        # 牛市概率
        probabilities['bull'] = max(0, min(1, (trend + 0.002) / 0.004))
        
        # 熊市概率
        probabilities['bear'] = max(0, min(1, (-trend + 0.002) / 0.004))
        
        # 横盘概率
        probabilities['sideways'] = 1 - probabilities['bull'] - probabilities['bear']
        
        # 高波动率概率
        probabilities['high_vol'] = max(0, min(1, (volatility - 0.15) / 0.20))
        
        # 低波动率概率
        probabilities['low_vol'] = max(0, min(1, (0.20 - volatility) / 0.15))
        
        # 归一化
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return probabilities


class VolatilityPredictor:
    """波动率预测器"""
    
    def __init__(self, model_type: str = 'garch'):
        self.model_type = model_type
        self.model = None
        self.predictions = []
        
    def fit(self, returns: pd.Series):
        """训练波动率模型"""
        if self.model_type == 'garch':
            self._fit_garch(returns)
        elif self.model_type == 'ewma':
            self._fit_ewma(returns)
        elif self.model_type == 'rolling':
            self._fit_rolling(returns)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _fit_garch(self, returns: pd.Series):
        """拟合GARCH模型"""
        try:
            from arch import arch_model
            
            # 创建GARCH(1,1)模型
            self.model = arch_model(returns * 100, vol='Garch', p=1, q=1)
            self.model_fit = self.model.fit(disp='off')
            
        except ImportError:
            print("ARCH package not available, using EWMA instead")
            self._fit_ewma(returns)
    
    def _fit_ewma(self, returns: pd.Series):
        """拟合EWMA模型"""
        self.model = {
            'type': 'ewma',
            'lambda': 0.94,
            'returns': returns
        }
    
    def _fit_rolling(self, returns: pd.Series):
        """拟合滚动窗口模型"""
        self.model = {
            'type': 'rolling',
            'window': 30,
            'returns': returns
        }
    
    def predict(self, horizon: int = 1) -> np.ndarray:
        """预测未来波动率"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model_type == 'garch' and hasattr(self, 'model_fit'):
            # GARCH预测
            forecast = self.model_fit.forecast(horizon=horizon)
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100
            
        elif self.model['type'] == 'ewma':
            # EWMA预测
            returns = self.model['returns']
            lambda_param = self.model['lambda']
            
            # 计算当前方差
            current_var = returns.var()
            
            # EWMA递推预测
            volatility_forecast = []
            var_t = current_var
            
            for _ in range(horizon):
                var_t = lambda_param * var_t + (1 - lambda_param) * returns.iloc[-1]**2
                volatility_forecast.append(np.sqrt(var_t))
            
            volatility_forecast = np.array(volatility_forecast)
            
        elif self.model['type'] == 'rolling':
            # 滚动窗口预测
            returns = self.model['returns']
            window = self.model['window']
            
            current_vol = returns.rolling(window=window).std().iloc[-1]
            volatility_forecast = np.full(horizon, current_vol)
        
        else:
            raise ValueError("Invalid model configuration")
        
        self.predictions.append({
            'timestamp': pd.Timestamp.now(),
            'horizon': horizon,
            'forecast': volatility_forecast
        })
        
        return volatility_forecast
    
    def get_volatility_regime(self, current_vol: float) -> str:
        """判断波动率状态"""
        # 基于历史分位数的状态判断
        if hasattr(self, 'model') and self.model is not None:
            if self.model_type == 'garch' and hasattr(self, 'model_fit'):
                historical_vol = np.sqrt(self.model_fit.conditional_volatility) / 100
            else:
                returns = self.model['returns']
                historical_vol = returns.rolling(window=30).std()
            
            # 计算分位数
            vol_25 = historical_vol.quantile(0.25)
            vol_75 = historical_vol.quantile(0.75)
            
            if current_vol < vol_25:
                return 'low_vol'
            elif current_vol > vol_75:
                return 'high_vol'
            else:
                return 'normal_vol'
        
        return 'unknown'