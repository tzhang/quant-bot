#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型集成模块 - 为量化预测比赛优化

提供多种集成学习方法，包括Stacking、Blending和Voting等
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class AdvancedEnsemble(BaseEstimator, RegressorMixin):
    """高级集成模型基类"""
    
    def __init__(self, 
                 base_models: List[BaseEstimator],
                 meta_model: Optional[BaseEstimator] = None,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        初始化集成模型
        
        Args:
            base_models: 基础模型列表
            meta_model: 元模型（用于Stacking）
            cv_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else LinearRegression()
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.fitted_models = []
        
    def _get_cv_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """获取交叉验证预测"""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for i, model in enumerate(self.base_models):
                # 克隆模型
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train)
                cv_predictions[val_idx, i] = model_clone.predict(X_val)
                
        return cv_predictions
    
    def _clone_model(self, model: BaseEstimator) -> BaseEstimator:
        """克隆模型"""
        from sklearn.base import clone
        return clone(model)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """拟合集成模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # 训练基础模型
        self.fitted_models = []
        for model in self.base_models:
            fitted_model = self._clone_model(model)
            fitted_model.fit(X, y)
            self.fitted_models.append(fitted_model)
            
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # 获取基础模型预测
        base_predictions = np.column_stack([
            model.predict(X) for model in self.fitted_models
        ])
        
        return self._combine_predictions(base_predictions)
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """组合预测结果（子类实现）"""
        raise NotImplementedError


class StackingEnsemble(AdvancedEnsemble):
    """Stacking集成"""
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """拟合Stacking模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # 获取交叉验证预测作为元特征
        meta_features = self._get_cv_predictions(X, y)
        
        # 训练基础模型
        self.fitted_models = []
        for model in self.base_models:
            fitted_model = self._clone_model(model)
            fitted_model.fit(X, y)
            self.fitted_models.append(fitted_model)
            
        # 训练元模型
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """使用元模型组合预测"""
        return self.meta_model.predict(predictions)


class BlendingEnsemble(AdvancedEnsemble):
    """Blending集成"""
    
    def __init__(self, 
                 base_models: List[BaseEstimator],
                 meta_model: Optional[BaseEstimator] = None,
                 holdout_ratio: float = 0.2,
                 random_state: int = 42):
        """
        初始化Blending模型
        
        Args:
            holdout_ratio: 留出集比例
        """
        super().__init__(base_models, meta_model, random_state=random_state)
        self.holdout_ratio = holdout_ratio
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """拟合Blending模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # 分割数据
        n_samples = X.shape[0]
        n_holdout = int(n_samples * self.holdout_ratio)
        
        np.random.seed(self.random_state)
        indices = np.random.permutation(n_samples)
        train_idx = indices[n_holdout:]
        holdout_idx = indices[:n_holdout]
        
        X_train, X_holdout = X[train_idx], X[holdout_idx]
        y_train, y_holdout = y[train_idx], y[holdout_idx]
        
        # 训练基础模型
        self.fitted_models = []
        holdout_predictions = []
        
        for model in self.base_models:
            fitted_model = self._clone_model(model)
            fitted_model.fit(X_train, y_train)
            self.fitted_models.append(fitted_model)
            holdout_predictions.append(fitted_model.predict(X_holdout))
            
        # 使用留出集训练元模型
        holdout_features = np.column_stack(holdout_predictions)
        self.meta_model.fit(holdout_features, y_holdout)
        
        return self
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """使用元模型组合预测"""
        return self.meta_model.predict(predictions)


class VotingEnsemble(AdvancedEnsemble):
    """投票集成"""
    
    def __init__(self, 
                 base_models: List[BaseEstimator],
                 weights: Optional[List[float]] = None,
                 voting: str = 'soft'):
        """
        初始化投票集成
        
        Args:
            weights: 模型权重
            voting: 投票方式 ('hard' 或 'soft')
        """
        super().__init__(base_models)
        self.weights = weights if weights else [1.0] * len(base_models)
        self.voting = voting
        
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """加权平均组合预测"""
        weights = np.array(self.weights)
        weights = weights / weights.sum()  # 归一化权重
        
        return np.average(predictions, axis=1, weights=weights)


class DynamicEnsemble(AdvancedEnsemble):
    """动态权重集成"""
    
    def __init__(self, 
                 base_models: List[BaseEstimator],
                 window_size: int = 100,
                 update_frequency: int = 10):
        """
        初始化动态集成
        
        Args:
            window_size: 权重计算窗口大小
            update_frequency: 权重更新频率
        """
        super().__init__(base_models)
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.weights_history = []
        self.performance_history = []
        
    def _calculate_dynamic_weights(self, 
                                  predictions: np.ndarray, 
                                  actuals: np.ndarray) -> np.ndarray:
        """计算动态权重"""
        n_models = predictions.shape[1]
        weights = np.ones(n_models) / n_models
        
        if len(self.performance_history) > self.window_size:
            # 基于历史表现计算权重
            recent_performance = np.array(self.performance_history[-self.window_size:])
            model_scores = np.mean(recent_performance, axis=0)
            
            # 转换为权重（分数越高权重越大）
            weights = np.exp(model_scores) / np.sum(np.exp(model_scores))
            
        return weights
    
    def update_weights(self, 
                      predictions: np.ndarray, 
                      actuals: np.ndarray):
        """更新模型权重"""
        # 计算每个模型的表现
        model_performance = []
        for i in range(predictions.shape[1]):
            mse = mean_squared_error(actuals, predictions[:, i])
            model_performance.append(-mse)  # 负MSE，越大越好
            
        self.performance_history.append(model_performance)
        
        # 计算新权重
        new_weights = self._calculate_dynamic_weights(predictions, actuals)
        self.weights_history.append(new_weights)
        
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """使用最新权重组合预测"""
        if self.weights_history:
            weights = self.weights_history[-1]
        else:
            weights = np.ones(predictions.shape[1]) / predictions.shape[1]
            
        return np.average(predictions, axis=1, weights=weights)


class AutoEnsemble:
    """自动集成选择器"""
    
    def __init__(self, 
                 base_models: List[BaseEstimator],
                 ensemble_methods: List[str] = ['stacking', 'blending', 'voting'],
                 cv_folds: int = 5,
                 scoring: str = 'neg_mean_squared_error'):
        """
        初始化自动集成选择器
        
        Args:
            ensemble_methods: 要尝试的集成方法
            scoring: 评分方法
        """
        self.base_models = base_models
        self.ensemble_methods = ensemble_methods
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.best_ensemble = None
        self.ensemble_scores = {}
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """拟合并选择最佳集成方法"""
        from sklearn.model_selection import cross_val_score
        
        best_score = -np.inf
        
        for method in self.ensemble_methods:
            if method == 'stacking':
                ensemble = StackingEnsemble(self.base_models, cv_folds=self.cv_folds)
            elif method == 'blending':
                ensemble = BlendingEnsemble(self.base_models)
            elif method == 'voting':
                ensemble = VotingEnsemble(self.base_models)
            else:
                continue
                
            # 交叉验证评估
            scores = cross_val_score(
                ensemble, X, y, 
                cv=self.cv_folds, 
                scoring=self.scoring,
                n_jobs=-1
            )
            
            mean_score = np.mean(scores)
            self.ensemble_scores[method] = {
                'mean_score': mean_score,
                'std_score': np.std(scores),
                'scores': scores
            }
            
            if mean_score > best_score:
                best_score = mean_score
                self.best_ensemble = ensemble
                
        # 用全部数据训练最佳集成
        self.best_ensemble.fit(X, y)
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """使用最佳集成进行预测"""
        return self.best_ensemble.predict(X)
    
    def get_ensemble_comparison(self) -> pd.DataFrame:
        """获取集成方法比较结果"""
        comparison_data = []
        for method, scores in self.ensemble_scores.items():
            comparison_data.append({
                'Method': method,
                'Mean_Score': scores['mean_score'],
                'Std_Score': scores['std_score'],
                'Best': method == type(self.best_ensemble).__name__.replace('Ensemble', '').lower()
            })
            
        return pd.DataFrame(comparison_data).sort_values('Mean_Score', ascending=False)


def create_default_models() -> List[BaseEstimator]:
    """创建默认的基础模型集合"""
    models = [
        # 线性模型
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        
        # 树模型
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        
        # Boosting模型
        xgb.XGBRegressor(n_estimators=100, random_state=42),
        lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    ]
    
    return models


def create_competition_ensemble(X_train: np.ndarray, 
                               y_train: np.ndarray,
                               X_test: np.ndarray = None,
                               models: List[BaseEstimator] = None) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
    """
    为比赛创建优化的集成模型
    
    Returns:
        ensemble: 训练好的集成模型
        train_predictions: 训练集预测
        test_predictions: 测试集预测（如果提供了X_test）
    """
    if models is None:
        models = create_default_models()
    
    # 自动选择最佳集成方法
    auto_ensemble = AutoEnsemble(models)
    auto_ensemble.fit(X_train, y_train)
    
    # 获取预测
    train_predictions = auto_ensemble.predict(X_train)
    test_predictions = auto_ensemble.predict(X_test) if X_test is not None else None
    
    return auto_ensemble, train_predictions, test_predictions