#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习因子计算器
包含特征工程、降维、聚类、异常检测等机器学习方法
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# 机器学习库导入
try:
    from sklearn.decomposition import PCA, ICA, FastICA, NMF
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("机器学习库未安装，将使用简化版本")

# 深度学习库导入
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class MLFactorCalculator:
    """机器学习因子计算器"""
    
    def __init__(self, lookback_window: int = 252, min_periods: int = 60):
        """
        初始化机器学习因子计算器
        
        Args:
            lookback_window: 回望窗口期
            min_periods: 最小计算周期
        """
        self.lookback_window = lookback_window
        self.min_periods = min_periods
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
    def prepare_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None, 
                        fundamental_data: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """
        准备机器学习特征
        
        Args:
            price_data: 价格数据 (OHLC)
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            
        Returns:
            特征数据框
        """
        features = pd.DataFrame(index=price_data.index)
        
        # 价格特征
        if 'close' in price_data.columns:
            close = price_data['close']
            features['returns_1d'] = close.pct_change()
            features['returns_5d'] = close.pct_change(5)
            features['returns_20d'] = close.pct_change(20)
            
            # 波动率特征
            features['volatility_5d'] = features['returns_1d'].rolling(5).std()
            features['volatility_20d'] = features['returns_1d'].rolling(20).std()
            features['volatility_60d'] = features['returns_1d'].rolling(60).std()
            
            # 价格位置特征
            features['price_position_20d'] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min())
            features['price_position_60d'] = (close - close.rolling(60).min()) / (close.rolling(60).max() - close.rolling(60).min())
            
            # 技术指标
            features['rsi_14'] = self._calculate_rsi(close, 14)
            features['rsi_30'] = self._calculate_rsi(close, 30)
            
            # 移动平均
            features['ma_5'] = close.rolling(5).mean()
            features['ma_20'] = close.rolling(20).mean()
            features['ma_60'] = close.rolling(60).mean()
            
            # 移动平均比率
            features['ma_ratio_5_20'] = features['ma_5'] / features['ma_20']
            features['ma_ratio_20_60'] = features['ma_20'] / features['ma_60']
        
        # 成交量特征
        if volume_data is not None and 'volume' in volume_data.columns:
            volume = volume_data['volume']
            features['volume_ma_5'] = volume.rolling(5).mean()
            features['volume_ma_20'] = volume.rolling(20).mean()
            features['volume_ratio'] = volume / features['volume_ma_20']
            features['volume_volatility'] = volume.rolling(20).std()
        
        # 基本面特征
        if fundamental_data:
            for key, series in fundamental_data.items():
                if len(series) > 0:
                    features[f'fundamental_{key}'] = series
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_feature_engineering_factors(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算特征工程因子"""
        engineering_factors = {}
        
        if not ML_AVAILABLE:
            return engineering_factors
        
        try:
            # PCA因子
            engineering_factors.update(self._calculate_pca_factors(features))
            
            # ICA因子
            engineering_factors.update(self._calculate_ica_factors(features))
            
            # 聚类因子
            engineering_factors.update(self._calculate_cluster_factors(features))
            
            # 特征选择因子
            engineering_factors.update(self._calculate_feature_selection_factors(features))
            
            # NMF因子
            engineering_factors.update(self._calculate_nmf_factors(features))
            
        except Exception as e:
            print(f"计算特征工程因子时出错: {e}")
        
        return engineering_factors
    
    def _calculate_pca_factors(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算PCA因子"""
        pca_factors = {}
        
        try:
            numeric_features = features.select_dtypes(include=[np.number]).dropna()
            if len(numeric_features) < 10:
                return pca_factors
            
            # 标准化
            scaled_features = self.scaler.fit_transform(numeric_features)
            
            # PCA降维
            pca = PCA(n_components=min(5, len(numeric_features.columns)))
            pca_result = pca.fit_transform(scaled_features)
            
            # 创建PCA因子
            for i in range(pca_result.shape[1]):
                pca_factors[f'pca_factor_{i+1}'] = pd.Series(
                    pca_result[:, i], 
                    index=numeric_features.index
                )
            
        except Exception as e:
            print(f"计算PCA因子时出错: {e}")
        
        return pca_factors
    
    def _calculate_ica_factors(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算ICA因子"""
        ica_factors = {}
        
        try:
            numeric_features = features.select_dtypes(include=[np.number]).dropna()
            if len(numeric_features) < 10:
                return ica_factors
            
            # 标准化
            scaled_features = self.scaler.fit_transform(numeric_features)
            
            # ICA分解
            ica = FastICA(n_components=min(3, len(numeric_features.columns)), random_state=42)
            ica_result = ica.fit_transform(scaled_features)
            
            # 创建ICA因子
            for i in range(ica_result.shape[1]):
                ica_factors[f'ica_factor_{i+1}'] = pd.Series(
                    ica_result[:, i], 
                    index=numeric_features.index
                )
        
        except Exception as e:
            print(f"计算ICA因子时出错: {e}")
        
        return ica_factors
    
    def _calculate_cluster_factors(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算聚类因子"""
        cluster_factors = {}
        
        try:
            numeric_features = features.select_dtypes(include=[np.number]).dropna()
            if len(numeric_features) < 20:
                return cluster_factors
            
            # 标准化
            scaled_features = self.scaler.fit_transform(numeric_features)
            
            # K-means聚类
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            cluster_factors['kmeans_cluster'] = pd.Series(cluster_labels, index=numeric_features.index)
            
            # 计算到聚类中心的距离
            distances = kmeans.transform(scaled_features)
            cluster_factors['cluster_distance'] = pd.Series(
                np.min(distances, axis=1), 
                index=numeric_features.index
            )
            
            # DBSCAN聚类
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(scaled_features)
            cluster_factors['dbscan_cluster'] = pd.Series(dbscan_labels, index=numeric_features.index)
            
        except Exception as e:
            print(f"计算聚类因子时出错: {e}")
        
        return cluster_factors
    
    def _calculate_feature_selection_factors(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算特征选择因子"""
        selection_factors = {}
        
        try:
            numeric_features = features.select_dtypes(include=[np.number]).dropna()
            if len(numeric_features) < 10 or 'returns_1d' not in numeric_features.columns:
                return selection_factors
            
            # 准备目标变量
            target = numeric_features['returns_1d'].shift(-1).dropna()
            feature_data = numeric_features.loc[target.index].drop('returns_1d', axis=1)
            
            if len(feature_data) < 30:
                return selection_factors
            
            # 基于F统计量的特征选择
            selector = SelectKBest(score_func=f_regression, k=min(5, len(feature_data.columns)))
            selected_features = selector.fit_transform(feature_data, target)
            
            # 创建选择的特征因子
            selected_indices = selector.get_support(indices=True)
            for i, idx in enumerate(selected_indices):
                col_name = feature_data.columns[idx]
                selection_factors[f'selected_{col_name}'] = feature_data[col_name]
            
        except Exception as e:
            print(f"计算特征选择因子时出错: {e}")
        
        return selection_factors
    
    def _calculate_nmf_factors(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算非负矩阵分解因子"""
        nmf_factors = {}
        
        try:
            numeric_features = features.select_dtypes(include=[np.number]).dropna()
            if len(numeric_features) < 10:
                return nmf_factors
            
            # 确保数据非负
            min_val = numeric_features.min().min()
            if min_val < 0:
                shifted_features = numeric_features - min_val + 0.01
            else:
                shifted_features = numeric_features
            
            # NMF分解
            nmf = NMF(n_components=min(3, len(shifted_features.columns)), random_state=42)
            nmf_result = nmf.fit_transform(shifted_features)
            
            # 创建NMF因子
            for i in range(nmf_result.shape[1]):
                nmf_factors[f'nmf_factor_{i+1}'] = pd.Series(
                    nmf_result[:, i], 
                    index=numeric_features.index
                )
        
        except Exception as e:
            print(f"计算NMF因子时出错: {e}")
        
        return nmf_factors
    
    def calculate_ensemble_factors(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算集成学习因子"""
        ensemble_factors = {}
        
        if not ML_AVAILABLE:
            return ensemble_factors
        
        try:
            # 滚动集成预测
            ensemble_factors.update(self._rolling_ensemble_prediction(features))
            
        except Exception as e:
            print(f"计算集成因子时出错: {e}")
        
        return ensemble_factors
    
    def _rolling_ensemble_prediction(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """滚动集成预测"""
        ensemble_factors = {}
        
        numeric_features = features.select_dtypes(include=[np.number]).dropna()
        if len(numeric_features) < 50 or 'returns_1d' not in numeric_features.columns:
            return ensemble_factors
        
        # 准备数据
        target = numeric_features['returns_1d'].shift(-1).dropna()
        feature_data = numeric_features.loc[target.index].drop('returns_1d', axis=1)
        
        if len(feature_data) < 100:
            return ensemble_factors
        
        # 定义模型
        models = {
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0),
            'lgb': lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        }
        
        # 滚动预测
        window_size = 60
        predictions = {name: [] for name in models.keys()}
        prediction_dates = []
        
        for i in range(window_size, len(feature_data)):
            train_start = max(0, i - window_size)
            train_end = i
            
            X_train = feature_data.iloc[train_start:train_end]
            y_train = target.iloc[train_start:train_end]
            X_test = feature_data.iloc[i:i+1]
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)[0]
                    predictions[name].append(pred)
                except:
                    predictions[name].append(np.nan)
            
            prediction_dates.append(feature_data.index[i])
        
        # 创建预测因子
        for name, preds in predictions.items():
            if len(preds) > 0:
                ensemble_factors[f'ensemble_{name}'] = pd.Series(preds, index=prediction_dates)
        
        # 集成预测
        if len(predictions) > 1:
            all_preds = pd.DataFrame(predictions, index=prediction_dates)
            ensemble_factors['ensemble_mean'] = all_preds.mean(axis=1)
            ensemble_factors['ensemble_median'] = all_preds.median(axis=1)
        
        return ensemble_factors
    
    def calculate_anomaly_detection_factors(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算异常检测因子"""
        anomaly_factors = {}
        
        if not ML_AVAILABLE:
            return anomaly_factors
        
        try:
            numeric_features = features.select_dtypes(include=[np.number]).dropna()
            if len(numeric_features) < 20:
                return anomaly_factors
            
            # 标准化
            scaled_features = self.scaler.fit_transform(numeric_features)
            
            # 孤立森林异常检测
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(scaled_features)
            anomaly_factors['isolation_anomaly'] = pd.Series(anomaly_scores, index=numeric_features.index)
            
            # 异常分数
            anomaly_scores_continuous = iso_forest.decision_function(scaled_features)
            anomaly_factors['anomaly_score'] = pd.Series(anomaly_scores_continuous, index=numeric_features.index)
            
            # 基于统计的异常检测
            if 'returns_1d' in numeric_features.columns:
                returns = numeric_features['returns_1d']
                rolling_mean = returns.rolling(20).mean()
                rolling_std = returns.rolling(20).std()
                z_score = (returns - rolling_mean) / rolling_std
                anomaly_factors['statistical_anomaly'] = (np.abs(z_score) > 2).astype(int)
                anomaly_factors['z_score'] = z_score
        
        except Exception as e:
            print(f"计算异常检测因子时出错: {e}")
        
        return anomaly_factors
    
    def calculate_time_series_features(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算时间序列特征"""
        ts_factors = {}
        
        if 'close' in price_data.columns:
            close = price_data['close']
            returns = close.pct_change().dropna()
            
            # 趋势强度
            ts_factors['trend_strength'] = self._calculate_trend_strength(close)
            
            # 季节性成分
            ts_factors['seasonal_component'] = self._calculate_seasonal_component(returns)
            
            # 波动率制度
            ts_factors['volatility_regime'] = self._calculate_volatility_regime(returns)
        
        return ts_factors
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算趋势强度"""
        # 使用线性回归斜率作为趋势强度
        def trend_slope(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return prices.rolling(window).apply(trend_slope, raw=False)
    
    def _calculate_seasonal_component(self, returns: pd.Series) -> pd.Series:
        """计算季节性成分"""
        # 简单的周期性检测
        seasonal = pd.Series(0, index=returns.index)
        
        # 周效应
        if hasattr(returns.index, 'dayofweek'):
            weekly_mean = returns.groupby(returns.index.dayofweek).mean()
            seasonal = returns.index.to_series().dt.dayofweek.map(weekly_mean)
        
        return seasonal
    
    def _calculate_volatility_regime(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """计算波动率制度"""
        volatility = returns.rolling(window).std()
        
        # 使用分位数定义制度
        high_vol_threshold = volatility.rolling(60).quantile(0.7)
        low_vol_threshold = volatility.rolling(60).quantile(0.3)
        
        regime = pd.Series(1, index=returns.index)  # 中等波动率
        regime[volatility > high_vol_threshold] = 2  # 高波动率
        regime[volatility < low_vol_threshold] = 0   # 低波动率
        
        return regime
    
    def calculate_all_factors(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None,
                            fundamental_data: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        计算所有机器学习因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            
        Returns:
            所有因子的字典
        """
        all_factors = {}
        
        # 准备特征
        features = self.prepare_features(price_data, volume_data, fundamental_data)
        
        if len(features) < self.min_periods:
            return all_factors
        
        # 计算各类因子
        all_factors.update(self.calculate_feature_engineering_factors(features))
        all_factors.update(self.calculate_ensemble_factors(features))
        all_factors.update(self.calculate_anomaly_detection_factors(features))
        all_factors.update(self.calculate_time_series_features(price_data))
        
        return all_factors
    
    def calculate_factor_scores(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """计算因子评分"""
        scores = {}
        
        for name, factor in factors.items():
            if len(factor.dropna()) > 0:
                # 标准化评分
                scores[f'{name}_score'] = (factor - factor.mean()) / factor.std()
        
        return scores
    
    def generate_factor_report(self, factors: Dict[str, pd.Series]) -> str:
        """生成因子报告"""
        report = "机器学习因子计算报告\n"
        report += "=" * 50 + "\n\n"
        
        for name, factor in factors.items():
            valid_data = factor.dropna()
            if len(valid_data) > 0:
                report += f"{name}:\n"
                report += f"  数据点数: {len(valid_data)}\n"
                report += f"  均值: {valid_data.mean():.4f}\n"
                report += f"  标准差: {valid_data.std():.4f}\n"
                report += f"  最小值: {valid_data.min():.4f}\n"
                report += f"  最大值: {valid_data.max():.4f}\n\n"
        
        return report


def main():
    """测试机器学习因子计算功能 - 仅用于测试和演示"""
    print("测试机器学习因子计算器...")
    
    # 创建测试数据 - 仅用于测试和演示
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')  # 生成日期范围 - 仅用于测试和演示
    np.random.seed(42)  # 设置随机种子确保结果可重现 - 仅用于测试和演示
    
    # 生成模拟价格数据 - 仅用于测试和演示
    price_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02),  # 开盘价：基于随机游走模型 - 仅用于测试和演示
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02) + np.random.rand(len(dates)),  # 最高价：开盘价基础上加随机值 - 仅用于测试和演示
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02) - np.random.rand(len(dates)),  # 最低价：开盘价基础上减随机值 - 仅用于测试和演示
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)  # 收盘价：基于随机游走模型 - 仅用于测试和演示
    }, index=dates)
    
    # 生成模拟成交量数据 - 仅用于测试和演示
    volume_data = pd.DataFrame({
        'volume': np.random.randint(1000000, 10000000, len(dates))  # 随机生成成交量 - 仅用于测试和演示
    }, index=dates)
    
    # 创建计算器 - 仅用于测试和演示
    calculator = MLFactorCalculator()
    
    # 计算因子 - 仅用于测试和演示
    factors = calculator.calculate_all_factors(price_data, volume_data)
    
    # 生成报告 - 仅用于测试和演示
    report = calculator.generate_factor_report(factors)
    print(report)
    
    print(f"成功计算了 {len(factors)} 个机器学习因子")


if __name__ == "__main__":
    main()