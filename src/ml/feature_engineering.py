#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征工程模块

提供量化交易中的高级特征工程功能：
- 滞后特征
- 滚动窗口特征
- 差分特征
- 交互特征
- 多项式特征
- 时间序列特征
- 技术指标特征
- 统计特征

扩展功能（新增）：
- 高阶统计特征 (advanced_feature_engineering.py)
- 频域分析特征 (advanced_feature_engineering.py)
- 图论网络特征 (advanced_feature_engineering.py)
- 宏观经济特征 (advanced_feature_engineering.py)
- 非线性特征 (advanced_feature_engineering.py)
- 特征选择 (feature_selection.py)
- 特征验证 (feature_validation.py)

作者: Quant Team
日期: 2024-01-20
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import talib
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """高级特征工程器"""
    
    def __init__(self, 
                 scaling_method: str = 'robust',
                 feature_selection: bool = True,
                 n_features: int = 50):
        """
        初始化特征工程器
        
        Args:
            scaling_method: 缩放方法 ('standard', 'robust', 'minmax')
            feature_selection: 是否进行特征选择
            n_features: 选择的特征数量
        """
        self.scaling_method = scaling_method
        self.feature_selection = feature_selection
        self.n_features = n_features
        
        # 初始化缩放器
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            
        # 特征选择器
        self.feature_selector = None
        
    def create_lag_features(self, 
                           df: pd.DataFrame, 
                           columns: List[str], 
                           lags: List[int]) -> pd.DataFrame:
        """创建滞后特征"""
        result_df = df.copy()
        
        for col in columns:
            for lag in lags:
                result_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        return result_df
    
    def create_rolling_features(self, 
                               df: pd.DataFrame, 
                               columns: List[str], 
                               windows: List[int]) -> pd.DataFrame:
        """创建滚动窗口特征"""
        result_df = df.copy()
        
        for col in columns:
            for window in windows:
                # 滚动均值
                result_df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                # 滚动标准差
                result_df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                # 滚动最大值
                result_df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
                # 滚动最小值
                result_df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                # 滚动偏度
                result_df[f'{col}_rolling_skew_{window}'] = df[col].rolling(window).skew()
                # 滚动峰度
                result_df[f'{col}_rolling_kurt_{window}'] = df[col].rolling(window).kurt()
                
        return result_df
    
    def create_diff_features(self, 
                            df: pd.DataFrame, 
                            columns: List[str], 
                            periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """创建差分特征"""
        result_df = df.copy()
        
        for col in columns:
            for period in periods:
                # 绝对差分
                result_df[f'{col}_diff_{period}'] = df[col].diff(period)
                # 相对差分（百分比变化）
                result_df[f'{col}_pct_change_{period}'] = df[col].pct_change(period)
                
        return result_df
    
    def create_interaction_features(self, 
                                   df: pd.DataFrame, 
                                   columns: List[str]) -> pd.DataFrame:
        """创建交互特征"""
        result_df = df.copy()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # 乘积特征
                result_df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                # 比率特征
                result_df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
        return result_df
    
    def create_polynomial_features(self, 
                                  df: pd.DataFrame, 
                                  columns: List[str], 
                                  degree: int = 2) -> pd.DataFrame:
        """创建多项式特征"""
        result_df = df.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                result_df[f'{col}_poly_{d}'] = df[col] ** d
                
        return result_df
    
    def fit_transform(self, 
                     df: pd.DataFrame, 
                     target: Optional[pd.Series] = None) -> pd.DataFrame:
        """拟合并转换特征"""
        # 移除无限值和NaN
        df_clean = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        
        # 特征缩放
        if self.scaler is not None:
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_columns] = self.scaler.fit_transform(df_clean[numeric_columns])
        
        # 特征选择
        if self.feature_selection and target is not None:
            self.feature_selector = SelectKBest(
                score_func=f_regression, 
                k=min(self.n_features, df_clean.shape[1])
            )
            selected_features = self.feature_selector.fit_transform(df_clean, target)
            selected_columns = df_clean.columns[self.feature_selector.get_support()]
            df_clean = pd.DataFrame(selected_features, columns=selected_columns, index=df_clean.index)
        
        return df_clean
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换特征（用于测试集）"""
        # 移除无限值和NaN
        df_clean = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        
        # 特征缩放
        if self.scaler is not None:
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_columns] = self.scaler.transform(df_clean[numeric_columns])
        
        # 特征选择
        if self.feature_selector is not None:
            selected_features = self.feature_selector.transform(df_clean)
            selected_columns = df_clean.columns[self.feature_selector.get_support()]
            df_clean = pd.DataFrame(selected_features, columns=selected_columns, index=df_clean.index)
        
        return df_clean


class TimeSeriesFeatureExtractor:
    """时间序列特征提取器"""
    
    def __init__(self):
        pass
    
    def extract_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """提取时间特征"""
        result_df = df.copy()
        
        # 确保日期列是datetime类型
        result_df[date_column] = pd.to_datetime(result_df[date_column])
        
        # 基础时间特征
        result_df['year'] = result_df[date_column].dt.year
        result_df['month'] = result_df[date_column].dt.month
        result_df['day'] = result_df[date_column].dt.day
        result_df['dayofweek'] = result_df[date_column].dt.dayofweek
        result_df['dayofyear'] = result_df[date_column].dt.dayofyear
        result_df['quarter'] = result_df[date_column].dt.quarter
        result_df['week'] = result_df[date_column].dt.isocalendar().week
        
        # 周期性特征（正弦余弦编码）
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        result_df['day_sin'] = np.sin(2 * np.pi * result_df['day'] / 31)
        result_df['day_cos'] = np.cos(2 * np.pi * result_df['day'] / 31)
        result_df['dayofweek_sin'] = np.sin(2 * np.pi * result_df['dayofweek'] / 7)
        result_df['dayofweek_cos'] = np.cos(2 * np.pi * result_df['dayofweek'] / 7)
        
        # 是否为月初/月末
        result_df['is_month_start'] = result_df[date_column].dt.is_month_start.astype(int)
        result_df['is_month_end'] = result_df[date_column].dt.is_month_end.astype(int)
        result_df['is_quarter_start'] = result_df[date_column].dt.is_quarter_start.astype(int)
        result_df['is_quarter_end'] = result_df[date_column].dt.is_quarter_end.astype(int)
        
        return result_df
    
    def extract_seasonal_features(self, 
                                 df: pd.DataFrame, 
                                 value_column: str, 
                                 periods: List[int] = [12, 52, 252]) -> pd.DataFrame:
        """提取季节性特征"""
        result_df = df.copy()
        
        for period in periods:
            # 季节性分解
            result_df[f'{value_column}_seasonal_{period}'] = (
                result_df[value_column].rolling(period).mean()
            )
            
            # 去趋势
            result_df[f'{value_column}_detrend_{period}'] = (
                result_df[value_column] - result_df[f'{value_column}_seasonal_{period}']
            )
            
        return result_df


class TechnicalIndicatorFeatures:
    """技术指标特征"""
    
    def __init__(self):
        pass
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加动量指标"""
        result_df = df.copy()
        
        # 需要OHLCV数据
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return result_df
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # RSI
        result_df['rsi_14'] = talib.RSI(close, timeperiod=14)
        result_df['rsi_30'] = talib.RSI(close, timeperiod=30)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        result_df['macd'] = macd
        result_df['macd_signal'] = macd_signal
        result_df['macd_hist'] = macd_hist
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        result_df['stoch_k'] = slowk
        result_df['stoch_d'] = slowd
        
        # Williams %R
        result_df['willr'] = talib.WILLR(high, low, close)
        
        # ROC (Rate of Change)
        result_df['roc_10'] = talib.ROC(close, timeperiod=10)
        result_df['roc_20'] = talib.ROC(close, timeperiod=20)
        
        return result_df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率指标"""
        result_df = df.copy()
        
        if 'close' not in df.columns:
            return result_df
        
        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
        result_df['bb_upper'] = bb_upper
        result_df['bb_middle'] = bb_middle
        result_df['bb_lower'] = bb_lower
        result_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        result_df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Average True Range
        result_df['atr'] = talib.ATR(high, low, close)
        
        # Normalized ATR
        result_df['natr'] = talib.NATR(high, low, close)
        
        return result_df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量指标"""
        result_df = df.copy()
        
        if not all(col in df.columns for col in ['close', 'volume']):
            return result_df
        
        close = df['close'].values
        volume = df['volume'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        
        # On Balance Volume
        result_df['obv'] = talib.OBV(close, volume)
        
        # Chaikin A/D Line
        result_df['ad'] = talib.AD(high, low, close, volume)
        
        # Chaikin A/D Oscillator
        result_df['adosc'] = talib.ADOSC(high, low, close, volume)
        
        # Volume Rate of Change
        result_df['vroc'] = talib.ROC(volume, timeperiod=10)
        
        return result_df


class StatisticalFeatures:
    """统计特征"""
    
    def __init__(self):
        pass
    
    def add_distribution_features(self, 
                                 df: pd.DataFrame, 
                                 columns: List[str], 
                                 windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """添加分布特征"""
        result_df = df.copy()
        
        for col in columns:
            for window in windows:
                # 分位数特征
                result_df[f'{col}_q25_{window}'] = df[col].rolling(window).quantile(0.25)
                result_df[f'{col}_q75_{window}'] = df[col].rolling(window).quantile(0.75)
                result_df[f'{col}_iqr_{window}'] = (
                    result_df[f'{col}_q75_{window}'] - result_df[f'{col}_q25_{window}']
                )
                
                # 相对位置
                result_df[f'{col}_rank_{window}'] = (
                    df[col].rolling(window).rank(pct=True)
                )
                
        return result_df
    
    def add_correlation_features(self, 
                                df: pd.DataFrame, 
                                target_col: str, 
                                feature_cols: List[str], 
                                windows: List[int] = [20, 50]) -> pd.DataFrame:
        """添加相关性特征"""
        result_df = df.copy()
        
        for col in feature_cols:
            for window in windows:
                # 滚动相关系数
                result_df[f'{col}_corr_{target_col}_{window}'] = (
                    df[col].rolling(window).corr(df[target_col])
                )
                
        return result_df
    
    def add_regime_features(self, 
                           df: pd.DataFrame, 
                           price_col: str = 'close') -> pd.DataFrame:
        """添加市场状态特征"""
        result_df = df.copy()
        
        if price_col not in df.columns:
            return result_df
        
        # 趋势特征
        result_df['trend_5'] = np.where(
            df[price_col] > df[price_col].rolling(5).mean(), 1, 0
        )
        result_df['trend_20'] = np.where(
            df[price_col] > df[price_col].rolling(20).mean(), 1, 0
        )
        result_df['trend_50'] = np.where(
            df[price_col] > df[price_col].rolling(50).mean(), 1, 0
        )
        
        # 波动率状态
        returns = df[price_col].pct_change()
        vol_20 = returns.rolling(20).std()
        vol_50 = returns.rolling(50).std()
        
        result_df['high_vol'] = np.where(vol_20 > vol_50, 1, 0)
        
        # 动量状态
        momentum_10 = df[price_col].pct_change(10)
        momentum_20 = df[price_col].pct_change(20)
        
        result_df['momentum_regime'] = np.where(
            (momentum_10 > 0) & (momentum_20 > 0), 1,
            np.where((momentum_10 < 0) & (momentum_20 < 0), -1, 0)
        )
        
        return result_df