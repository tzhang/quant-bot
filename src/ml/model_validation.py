#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型验证模块 - 为量化预测比赛优化

提供时间序列交叉验证、模型选择和超参数优化功能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, ParameterGrid, ParameterSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesCrossValidator(BaseCrossValidator):
    """时间序列交叉验证器"""
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: Optional[int] = None,
                 gap: int = 0,
                 expanding_window: bool = False):
        """
        初始化时间序列交叉验证器
        
        Args:
            n_splits: 分割数量
            test_size: 测试集大小
            gap: 训练集和测试集之间的间隔
            expanding_window: 是否使用扩展窗口（否则为滑动窗口）
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
        
    def split(self, X, y=None, groups=None):
        """生成训练/测试索引"""
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
            
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # 计算测试集结束位置
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            
            # 计算训练集结束位置（考虑gap）
            train_end = test_start - self.gap
            
            if self.expanding_window:
                # 扩展窗口：从开始到train_end
                train_start = 0
            else:
                # 滑动窗口：固定大小的训练窗口
                train_start = max(0, train_end - test_size * 3)  # 训练集是测试集的3倍
                
            if train_start >= train_end or test_start >= test_end:
                break
                
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """返回分割数量"""
        return self.n_splits


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """带清洗的分组时间序列分割"""
    
    def __init__(self, 
                 n_splits: int = 5,
                 group_gap: int = 1,
                 max_train_group_size: Optional[int] = None,
                 max_test_group_size: Optional[int] = None):
        """
        初始化分组时间序列分割器
        
        Args:
            n_splits: 分割数量
            group_gap: 组间间隔
            max_train_group_size: 最大训练组大小
            max_test_group_size: 最大测试组大小
        """
        self.n_splits = n_splits
        self.group_gap = group_gap
        self.max_train_group_size = max_train_group_size
        self.max_test_group_size = max_test_group_size
        
    def split(self, X, y=None, groups=None):
        """生成训练/测试索引"""
        if groups is None:
            raise ValueError("groups参数不能为None")
            
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        group_dict = {}
        for idx, group in enumerate(groups):
            if group not in group_dict:
                group_dict[group] = []
            group_dict[group].append(idx)
            
        indices = np.arange(len(X))
        
        for i in range(self.n_splits):
            # 计算测试组
            test_group_start = n_groups - (i + 1) * (n_groups // (self.n_splits + 1))
            test_group_end = test_group_start + (n_groups // (self.n_splits + 1))
            
            if self.max_test_group_size:
                test_group_end = min(test_group_end, test_group_start + self.max_test_group_size)
                
            # 计算训练组（考虑gap）
            train_group_end = test_group_start - self.group_gap
            train_group_start = 0
            
            if self.max_train_group_size:
                train_group_start = max(0, train_group_end - self.max_train_group_size)
                
            if train_group_start >= train_group_end:
                continue
                
            # 获取训练和测试索引
            train_groups = unique_groups[train_group_start:train_group_end]
            test_groups = unique_groups[test_group_start:test_group_end]
            
            train_indices = []
            test_indices = []
            
            for group in train_groups:
                train_indices.extend(group_dict[group])
                
            for group in test_groups:
                test_indices.extend(group_dict[group])
                
            yield np.array(train_indices), np.array(test_indices)
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """返回分割数量"""
        return self.n_splits


class ModelValidator:
    """模型验证器"""
    
    def __init__(self, 
                 cv: BaseCrossValidator,
                 scoring: Union[str, Callable] = 'neg_mean_squared_error',
                 return_train_score: bool = False):
        """
        初始化模型验证器
        
        Args:
            cv: 交叉验证器
            scoring: 评分函数
            return_train_score: 是否返回训练分数
        """
        self.cv = cv
        self.scoring = scoring
        self.return_train_score = return_train_score
        
    def validate_model(self, 
                      model: BaseEstimator,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """验证单个模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        test_scores = []
        train_scores = []
        predictions = []
        
        for train_idx, test_idx in self.cv.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 训练模型
            model_clone = self._clone_model(model)
            model_clone.fit(X_train, y_train)
            
            # 预测
            y_pred_test = model_clone.predict(X_test)
            predictions.append((test_idx, y_pred_test))
            
            # 计算分数
            test_score = self._calculate_score(y_test, y_pred_test)
            test_scores.append(test_score)
            
            if self.return_train_score:
                y_pred_train = model_clone.predict(X_train)
                train_score = self._calculate_score(y_train, y_pred_train)
                train_scores.append(train_score)
                
        result = {
            'test_scores': np.array(test_scores),
            'test_score_mean': np.mean(test_scores),
            'test_score_std': np.std(test_scores),
            'predictions': predictions
        }
        
        if self.return_train_score:
            result.update({
                'train_scores': np.array(train_scores),
                'train_score_mean': np.mean(train_scores),
                'train_score_std': np.std(train_scores)
            })
            
        return result
    
    def compare_models(self, 
                      models: Dict[str, BaseEstimator],
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      groups: Optional[np.ndarray] = None) -> pd.DataFrame:
        """比较多个模型"""
        results = []
        
        for name, model in models.items():
            print(f"验证模型: {name}")
            result = self.validate_model(model, X, y, groups)
            
            results.append({
                'Model': name,
                'Test_Score_Mean': result['test_score_mean'],
                'Test_Score_Std': result['test_score_std'],
                'Train_Score_Mean': result.get('train_score_mean', np.nan),
                'Train_Score_Std': result.get('train_score_std', np.nan)
            })
            
        return pd.DataFrame(results).sort_values('Test_Score_Mean', ascending=False)
    
    def _clone_model(self, model: BaseEstimator) -> BaseEstimator:
        """克隆模型"""
        from sklearn.base import clone
        return clone(model)
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算分数"""
        if callable(self.scoring):
            return self.scoring(y_true, y_pred)
        elif self.scoring == 'neg_mean_squared_error':
            return -mean_squared_error(y_true, y_pred)
        elif self.scoring == 'neg_mean_absolute_error':
            return -mean_absolute_error(y_true, y_pred)
        elif self.scoring == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"不支持的评分方法: {self.scoring}")


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, 
                 cv: BaseCrossValidator,
                 scoring: Union[str, Callable] = 'neg_mean_squared_error',
                 n_trials: int = 100,
                 random_state: int = 42):
        """
        初始化超参数优化器
        
        Args:
            cv: 交叉验证器
            scoring: 评分函数
            n_trials: 优化试验次数
            random_state: 随机种子
        """
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.random_state = random_state
        self.study = None
        self.best_params = None
        self.best_score = None
        
    def optimize_xgboost(self, 
                        X: Union[np.ndarray, pd.DataFrame],
                        y: Union[np.ndarray, pd.Series],
                        groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """优化XGBoost参数"""
        import xgboost as xgb
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state
            }
            
            model = xgb.XGBRegressor(**params)
            validator = ModelValidator(self.cv, self.scoring)
            result = validator.validate_model(model, X, y, groups)
            
            return result['test_score_mean']
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study
        }
    
    def optimize_lightgbm(self, 
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """优化LightGBM参数"""
        import lightgbm as lgb
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            validator = ModelValidator(self.cv, self.scoring)
            result = validator.validate_model(model, X, y, groups)
            
            return result['test_score_mean']
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study
        }
    
    def optimize_custom_model(self, 
                             model_class: type,
                             param_space: Dict[str, Any],
                             X: Union[np.ndarray, pd.DataFrame],
                             y: Union[np.ndarray, pd.Series],
                             groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """优化自定义模型参数"""
        
        def objective(trial):
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
                    
            model = model_class(**params)
            validator = ModelValidator(self.cv, self.scoring)
            result = validator.validate_model(model, X, y, groups)
            
            return result['test_score_mean']
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study
        }
    
    def plot_optimization_history(self):
        """绘制优化历史"""
        if self.study is None:
            raise ValueError("请先运行优化")
            
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 优化历史
        optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
        ax1.set_title('Optimization History')
        
        # 参数重要性
        optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
        ax2.set_title('Parameter Importances')
        
        plt.tight_layout()
        plt.show()


def create_time_series_cv(n_splits: int = 5, 
                         test_size: Optional[int] = None,
                         gap: int = 0) -> TimeSeriesCrossValidator:
    """创建时间序列交叉验证器"""
    return TimeSeriesCrossValidator(
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        expanding_window=True
    )


def quick_model_comparison(models: Dict[str, BaseEstimator],
                          X: Union[np.ndarray, pd.DataFrame],
                          y: Union[np.ndarray, pd.Series],
                          cv_splits: int = 5) -> pd.DataFrame:
    """快速模型比较"""
    cv = create_time_series_cv(n_splits=cv_splits)
    validator = ModelValidator(cv, scoring='neg_mean_squared_error', return_train_score=True)
    
    return validator.compare_models(models, X, y)