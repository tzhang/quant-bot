"""
信号生成模块

提供多种技术指标计算、信号生成和信号融合功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class SignalGenerator:
    """通用信号生成器"""
    
    def __init__(self):
        self.signals = {}
        self.weights = {}
        
    def add_momentum_signals(self, data: pd.DataFrame, 
                           periods: List[int] = [5, 10, 20]) -> Dict[str, pd.Series]:
        """添加动量信号"""
        signals = {}
        
        for period in periods:
            # 价格动量
            signals[f'momentum_{period}'] = data['close'].pct_change(period)
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            signals[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            if period >= 12:
                ema_fast = data['close'].ewm(span=12).mean()
                ema_slow = data['close'].ewm(span=26).mean()
                macd = ema_fast - ema_slow
                signal_line = macd.ewm(span=9).mean()
                signals[f'macd_{period}'] = macd - signal_line
        
        self.signals.update(signals)
        return signals
    
    def add_mean_reversion_signals(self, data: pd.DataFrame,
                                 periods: List[int] = [10, 20, 50]) -> Dict[str, pd.Series]:
        """添加均值回归信号"""
        signals = {}
        
        for period in periods:
            # 布林带
            sma = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            signals[f'bollinger_{period}'] = (data['close'] - sma) / (2 * std)
            
            # Z-Score
            signals[f'zscore_{period}'] = (data['close'] - sma) / std
            
            # 价格相对位置
            high_max = data['high'].rolling(window=period).max()
            low_min = data['low'].rolling(window=period).min()
            signals[f'price_position_{period}'] = (data['close'] - low_min) / (high_max - low_min)
        
        self.signals.update(signals)
        return signals
    
    def add_volatility_signals(self, data: pd.DataFrame,
                             periods: List[int] = [10, 20]) -> Dict[str, pd.Series]:
        """添加波动率信号"""
        signals = {}
        
        for period in periods:
            # 历史波动率
            returns = data['close'].pct_change()
            signals[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            
            # ATR (Average True Range)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            signals[f'atr_{period}'] = true_range.rolling(window=period).mean()
            
            # 波动率突破
            vol_ma = signals[f'volatility_{period}'].rolling(window=period).mean()
            signals[f'vol_breakout_{period}'] = signals[f'volatility_{period}'] / vol_ma - 1
        
        self.signals.update(signals)
        return signals
    
    def add_microstructure_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """添加微观结构信号"""
        signals = {}
        
        # 价差信号
        if 'bid' in data.columns and 'ask' in data.columns:
            signals['bid_ask_spread'] = (data['ask'] - data['bid']) / data['close']
            signals['mid_price'] = (data['bid'] + data['ask']) / 2
            signals['price_impact'] = (data['close'] - signals['mid_price']) / signals['mid_price']
        
        # 成交量信号
        if 'volume' in data.columns:
            signals['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
            signals['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            signals['price_volume_trend'] = ((data['close'] - data['close'].shift()) / 
                                           data['close'].shift()) * data['volume']
        
        # 价格跳跃
        returns = data['close'].pct_change()
        signals['price_jump'] = np.abs(returns) > (returns.rolling(window=20).std() * 3)
        
        self.signals.update(signals)
        return signals
    
    def get_all_signals(self) -> Dict[str, pd.Series]:
        """获取所有信号"""
        return self.signals
    
    def clear_signals(self):
        """清空所有信号"""
        self.signals.clear()
        self.weights.clear()


class SignalFusion:
    """信号融合器"""
    
    def __init__(self, method: str = 'weighted_average'):
        self.method = method
        self.weights = {}
        
    def set_weights(self, weights: Dict[str, float]):
        """设置信号权重"""
        # 归一化权重
        total_weight = sum(weights.values())
        self.weights = {k: v/total_weight for k, v in weights.items()}
    
    def fuse_signals(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """融合多个信号"""
        if not signals:
            raise ValueError("No signals provided for fusion")
        
        if self.method == 'weighted_average':
            return self._weighted_average_fusion(signals)
        elif self.method == 'rank_based':
            return self._rank_based_fusion(signals)
        elif self.method == 'pca':
            return self._pca_fusion(signals)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
    
    def _weighted_average_fusion(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """加权平均融合"""
        if not self.weights:
            # 等权重
            weights = {k: 1.0/len(signals) for k in signals.keys()}
        else:
            weights = self.weights
        
        fused_signal = pd.Series(0, index=next(iter(signals.values())).index)
        
        for signal_name, signal_values in signals.items():
            if signal_name in weights:
                # 标准化信号
                normalized_signal = (signal_values - signal_values.mean()) / signal_values.std()
                fused_signal += weights[signal_name] * normalized_signal.fillna(0)
        
        return fused_signal
    
    def _rank_based_fusion(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """基于排名的融合"""
        signal_df = pd.DataFrame(signals)
        
        # 计算每个信号的排名
        rank_df = signal_df.rank(pct=True)
        
        # 平均排名作为融合信号
        fused_signal = rank_df.mean(axis=1)
        
        return fused_signal
    
    def _pca_fusion(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """PCA降维融合"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        signal_df = pd.DataFrame(signals).fillna(0)
        
        # 标准化
        scaler = StandardScaler()
        scaled_signals = scaler.fit_transform(signal_df)
        
        # PCA
        pca = PCA(n_components=1)
        fused_signal = pca.fit_transform(scaled_signals).flatten()
        
        return pd.Series(fused_signal, index=signal_df.index)


class SignalOptimizer:
    """信号优化器"""
    
    def __init__(self):
        self.best_params = {}
        self.optimization_history = []
    
    def optimize_signal_parameters(self, data: pd.DataFrame, 
                                 signal_func: callable,
                                 param_ranges: Dict[str, Tuple[int, int]],
                                 target_metric: str = 'sharpe_ratio',
                                 n_trials: int = 100) -> Dict[str, Any]:
        """优化信号参数"""
        try:
            import optuna
        except ImportError:
            raise ImportError("Please install optuna: pip install optuna")
        
        def objective(trial):
            # 采样参数
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            
            # 生成信号
            signal = signal_func(data, **params)
            
            # 计算目标指标
            if target_metric == 'sharpe_ratio':
                returns = signal.shift(1) * data['close'].pct_change()
                return returns.mean() / returns.std() if returns.std() > 0 else 0
            elif target_metric == 'information_ratio':
                returns = signal.shift(1) * data['close'].pct_change()
                benchmark_returns = data['close'].pct_change()
                excess_returns = returns - benchmark_returns
                return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            else:
                raise ValueError(f"Unknown target metric: {target_metric}")
        
        # 创建研究对象
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.optimization_history.append({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': n_trials
        })
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def get_optimized_signal(self, data: pd.DataFrame, 
                           signal_func: callable) -> pd.Series:
        """使用优化参数生成信号"""
        if not self.best_params:
            raise ValueError("No optimized parameters found. Run optimize_signal_parameters first.")
        
        return signal_func(data, **self.best_params)