#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征选择模块

提供多种特征选择方法：
- 递归特征消除 (RFE)
- 基于重要性的选择
- 稳定性选择
- 信息增益选择
- 相关性分析选择
- 方差分析选择
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.feature_selection import (
    RFE, RFECV, SelectKBest, SelectPercentile, 
    mutual_info_regression, f_regression, chi2,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class RecursiveFeatureElimination:
    """递归特征消除"""
    
    def __init__(self, 
                 estimator=None, 
                 n_features_to_select=None, 
                 step=1, 
                 cv=5):
        self.estimator = estimator or RandomForestRegressor(n_estimators=100, random_state=42)
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.cv = cv
        self.selected_features_ = None
        self.feature_ranking_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RecursiveFeatureElimination':
        """拟合RFE选择器"""
        if self.n_features_to_select is None:
            # 使用交叉验证确定最优特征数
            rfe = RFECV(
                estimator=self.estimator,
                step=self.step,
                cv=self.cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
        else:
            rfe = RFE(
                estimator=self.estimator,
                n_features_to_select=self.n_features_to_select,
                step=self.step
            )
        
        rfe.fit(X, y)
        
        self.selected_features_ = X.columns[rfe.support_].tolist()
        self.feature_ranking_ = pd.Series(rfe.ranking_, index=X.columns)
        self.rfe_ = rfe
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据，只保留选中的特征"""
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性排名"""
        return self.feature_ranking_.sort_values()


class ImportanceBasedSelection:
    """基于重要性的特征选择"""
    
    def __init__(self, 
                 method='random_forest', 
                 threshold='median', 
                 n_estimators=100):
        self.method = method
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.selected_features_ = None
        self.feature_importance_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ImportanceBasedSelection':
        """拟合重要性选择器"""
        # 计算特征重要性
        if self.method == 'random_forest':
            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        elif self.method == 'lasso':
            model = LassoCV(cv=5, random_state=42)
        elif self.method == 'elastic_net':
            model = ElasticNetCV(cv=5, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        model.fit(X, y)
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            raise ValueError("Model doesn't have feature importance or coefficients")
        
        self.feature_importance_ = pd.Series(importance, index=X.columns)
        
        # 确定阈值
        if isinstance(self.threshold, str):
            if self.threshold == 'mean':
                threshold_value = self.feature_importance_.mean()
            elif self.threshold == 'median':
                threshold_value = self.feature_importance_.median()
            elif self.threshold == 'auto':
                # 使用肘部法则确定阈值
                sorted_importance = self.feature_importance_.sort_values(ascending=False)
                threshold_value = self._find_elbow_threshold(sorted_importance)
            else:
                raise ValueError(f"Unknown threshold method: {self.threshold}")
        else:
            threshold_value = self.threshold
        
        # 选择特征
        self.selected_features_ = self.feature_importance_[
            self.feature_importance_ >= threshold_value
        ].index.tolist()
        
        return self
    
    def _find_elbow_threshold(self, sorted_importance):
        """使用肘部法则找到阈值"""
        values = sorted_importance.values
        n = len(values)
        
        # 计算每个点到首尾连线的距离
        distances = []
        for i in range(1, n-1):
            # 点到直线的距离
            x1, y1 = 0, values[0]
            x2, y2 = n-1, values[-1]
            x0, y0 = i, values[i]
            
            distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
            distances.append(distance)
        
        # 找到最大距离对应的点
        elbow_idx = np.argmax(distances) + 1
        return values[elbow_idx]
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)


class StabilitySelection:
    """稳定性选择"""
    
    def __init__(self, 
                 base_estimator=None, 
                 n_bootstrap=100, 
                 threshold=0.6, 
                 sample_fraction=0.5,
                 feature_fraction=0.5):
        self.base_estimator = base_estimator or LassoCV(cv=3, random_state=42)
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.sample_fraction = sample_fraction
        self.feature_fraction = feature_fraction
        self.selected_features_ = None
        self.selection_probabilities_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StabilitySelection':
        """拟合稳定性选择器"""
        n_samples, n_features = X.shape
        n_sample_subset = int(n_samples * self.sample_fraction)
        n_feature_subset = int(n_features * self.feature_fraction)
        
        # 记录每个特征被选中的次数
        selection_counts = np.zeros(n_features)
        
        for i in range(self.n_bootstrap):
            # 随机采样样本
            sample_indices = np.random.choice(n_samples, n_sample_subset, replace=False)
            
            # 随机采样特征
            feature_indices = np.random.choice(n_features, n_feature_subset, replace=False)
            
            # 子集数据
            X_subset = X.iloc[sample_indices, feature_indices]
            y_subset = y.iloc[sample_indices]
            
            # 拟合模型
            estimator = self.base_estimator
            estimator.fit(X_subset, y_subset)
            
            # 获取选中的特征
            if hasattr(estimator, 'coef_'):
                selected = np.abs(estimator.coef_) > 1e-6
            elif hasattr(estimator, 'feature_importances_'):
                selected = estimator.feature_importances_ > 0
            else:
                # 使用默认选择（非零系数）
                selected = np.ones(len(feature_indices), dtype=bool)
            
            # 更新计数
            selection_counts[feature_indices[selected]] += 1
        
        # 计算选择概率
        self.selection_probabilities_ = pd.Series(
            selection_counts / self.n_bootstrap, 
            index=X.columns
        )
        
        # 选择稳定的特征
        self.selected_features_ = self.selection_probabilities_[
            self.selection_probabilities_ >= self.threshold
        ].index.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)


class InformationGainSelection:
    """信息增益特征选择"""
    
    def __init__(self, 
                 method='mutual_info', 
                 k=10, 
                 percentile=50):
        self.method = method
        self.k = k
        self.percentile = percentile
        self.selected_features_ = None
        self.scores_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'InformationGainSelection':
        """拟合信息增益选择器"""
        if self.method == 'mutual_info':
            scores = mutual_info_regression(X, y, random_state=42)
        elif self.method == 'f_regression':
            scores, _ = f_regression(X, y)
        elif self.method == 'chi2':
            # Chi2需要非负特征
            X_positive = X - X.min() + 1e-6
            scores, _ = chi2(X_positive, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.scores_ = pd.Series(scores, index=X.columns)
        
        # 选择特征
        if self.k is not None:
            # 选择top-k特征
            self.selected_features_ = self.scores_.nlargest(self.k).index.tolist()
        else:
            # 选择百分位数特征
            threshold = np.percentile(scores, self.percentile)
            self.selected_features_ = self.scores_[self.scores_ >= threshold].index.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)


class CorrelationBasedSelection:
    """基于相关性的特征选择"""
    
    def __init__(self, 
                 threshold=0.95, 
                 method='pearson'):
        self.threshold = threshold
        self.method = method
        self.selected_features_ = None
        self.correlation_matrix_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CorrelationBasedSelection':
        """拟合相关性选择器"""
        # 计算相关性矩阵
        self.correlation_matrix_ = X.corr(method=self.method)
        
        # 找到高相关性的特征对
        high_corr_pairs = []
        n_features = len(X.columns)
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(self.correlation_matrix_.iloc[i, j]) > self.threshold:
                    high_corr_pairs.append((X.columns[i], X.columns[j]))
        
        # 移除高相关性特征
        features_to_remove = set()
        
        for feat1, feat2 in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                # 如果提供了目标变量，保留与目标相关性更高的特征
                if y is not None:
                    corr1 = abs(X[feat1].corr(y))
                    corr2 = abs(X[feat2].corr(y))
                    if corr1 >= corr2:
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)
                else:
                    # 随机移除一个
                    features_to_remove.add(feat2)
        
        self.selected_features_ = [f for f in X.columns if f not in features_to_remove]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)


class VarianceBasedSelection:
    """基于方差的特征选择"""
    
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.selected_features_ = None
        self.variances_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'VarianceBasedSelection':
        """拟合方差选择器"""
        # 计算方差
        self.variances_ = X.var()
        
        # 选择方差大于阈值的特征
        self.selected_features_ = self.variances_[
            self.variances_ > self.threshold
        ].index.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)


class FeatureSelectionPipeline:
    """特征选择管道"""
    
    def __init__(self):
        self.selectors = {}
        self.selected_features_ = None
        self.selection_results_ = {}
    
    def add_selector(self, name: str, selector) -> 'FeatureSelectionPipeline':
        """添加选择器"""
        self.selectors[name] = selector
        return self
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelectionPipeline':
        """拟合所有选择器"""
        print(f"开始特征选择，原始特征数: {X.shape[1]}")
        
        for name, selector in self.selectors.items():
            print(f"应用 {name} 选择器...")
            
            try:
                selector.fit(X, y)
                selected_features = selector.selected_features_
                
                self.selection_results_[name] = {
                    'selected_features': selected_features,
                    'n_selected': len(selected_features),
                    'selection_rate': len(selected_features) / X.shape[1]
                }
                
                print(f"  {name}: 选择了 {len(selected_features)} 个特征 "
                      f"({len(selected_features)/X.shape[1]:.2%})")
                
            except Exception as e:
                print(f"  {name} 失败: {e}")
                self.selection_results_[name] = {
                    'selected_features': [],
                    'n_selected': 0,
                    'selection_rate': 0,
                    'error': str(e)
                }
        
        return self
    
    def get_consensus_features(self, 
                              min_votes: int = 2, 
                              voting_method: str = 'majority') -> List[str]:
        """获取共识特征"""
        all_features = set()
        feature_votes = {}
        
        # 收集所有特征的投票
        for name, result in self.selection_results_.items():
            if 'error' not in result:
                features = result['selected_features']
                all_features.update(features)
                
                for feature in features:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # 根据投票方法选择特征
        if voting_method == 'majority':
            threshold = len(self.selectors) / 2
        elif voting_method == 'unanimous':
            threshold = len(self.selectors)
        elif voting_method == 'min_votes':
            threshold = min_votes
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")
        
        consensus_features = [
            feature for feature, votes in feature_votes.items() 
            if votes >= threshold
        ]
        
        return consensus_features
    
    def transform(self, X: pd.DataFrame, method: str = 'consensus') -> pd.DataFrame:
        """转换数据"""
        if method == 'consensus':
            if self.selected_features_ is None:
                self.selected_features_ = self.get_consensus_features()
            return X[self.selected_features_]
        elif method in self.selectors:
            return self.selectors[method].transform(X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, method: str = 'consensus') -> pd.DataFrame:
        """拟合并转换数据"""
        self.fit(X, y)
        return self.transform(X, method)
    
    def get_selection_summary(self) -> pd.DataFrame:
        """获取选择结果摘要"""
        summary_data = []
        
        for name, result in self.selection_results_.items():
            summary_data.append({
                'selector': name,
                'n_selected': result['n_selected'],
                'selection_rate': result['selection_rate'],
                'status': 'success' if 'error' not in result else 'failed'
            })
        
        return pd.DataFrame(summary_data)


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据进行测试 - 仅用于测试和演示
    np.random.seed(42)  # 设置随机种子确保结果可重现 - 仅用于测试和演示
    n_samples, n_features = 1000, 50  # 设置样本数和特征数 - 仅用于测试和演示
    
    # 创建模拟特征数据 - 模拟数据仅用于演示
    X = pd.DataFrame(  # 创建特征DataFrame - 仅用于测试和演示
        np.random.randn(n_samples, n_features),  # 生成正态分布的模拟特征 - 仅用于测试和演示
        columns=[f'feature_{i}' for i in range(n_features)]  # 设置特征列名 - 仅用于测试和演示
    )
    
    # 创建模拟目标变量 - 仅用于测试
    y = (X.iloc[:, :10].sum(axis=1) +  # 使用前10个特征的和作为基础 - 仅用于测试和演示
         np.random.randn(n_samples) * 0.1)  # 添加噪声生成目标变量 - 仅用于测试和演示
    
    # 创建特征选择管道 - 仅用于测试和演示
    pipeline = FeatureSelectionPipeline()  # 初始化特征选择管道 - 仅用于测试和演示
    
    # 添加多个选择器 - 仅用于测试和演示
    pipeline.add_selector('rfe', RecursiveFeatureElimination(n_features_to_select=15))  # 添加递归特征消除 - 仅用于测试和演示
    pipeline.add_selector('importance', ImportanceBasedSelection(method='random_forest'))  # 添加重要性选择 - 仅用于测试和演示
    pipeline.add_selector('stability', StabilitySelection(threshold=0.5))  # 添加稳定性选择 - 仅用于测试和演示
    pipeline.add_selector('mutual_info', InformationGainSelection(method='mutual_info', k=20))  # 添加信息增益选择 - 仅用于测试和演示
    pipeline.add_selector('correlation', CorrelationBasedSelection(threshold=0.9))  # 添加相关性选择 - 仅用于测试和演示
    pipeline.add_selector('variance', VarianceBasedSelection(threshold=0.1))  # 添加方差选择 - 仅用于测试和演示
    
    # 拟合并选择特征 - 仅用于测试和演示
    X_selected = pipeline.fit_transform(X, y)  # 执行特征选择 - 仅用于测试和演示
    
    print(f"\n最终选择的特征数: {X_selected.shape[1]}")  # 输出选择的特征数 - 仅用于测试和演示
    print(f"选择率: {X_selected.shape[1]/X.shape[1]:.2%}")  # 输出特征选择率 - 仅用于测试和演示
    
    # 显示选择摘要 - 仅用于测试和演示
    summary = pipeline.get_selection_summary()  # 获取选择摘要 - 仅用于测试和演示
    print("\n选择器摘要:")  # 输出摘要标题 - 仅用于测试和演示
    print(summary)  # 输出摘要内容 - 仅用于测试和演示