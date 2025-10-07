"""
机器学习引擎模块

提供特征分析、模型集成和时间序列验证功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


class MLFeatureAnalyzer:
    """机器学习特征分析器"""
    
    def __init__(self):
        self.feature_importance = {}
        self.feature_correlations = {}
        self.feature_stability = {}
        
    def analyze_feature_importance(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 method: str = 'random_forest') -> Dict[str, float]:
        """分析特征重要性"""
        if method == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X.fillna(0), y)
            importance = dict(zip(X.columns, model.feature_importances_))
            
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            importance_scores = mutual_info_regression(X.fillna(0), y)
            importance = dict(zip(X.columns, importance_scores))
            
        elif method == 'correlation':
            correlations = X.corrwith(y).abs()
            importance = correlations.to_dict()
            
        elif method == 'lasso':
            from sklearn.linear_model import LassoCV
            model = LassoCV(cv=5, random_state=42)
            model.fit(X.fillna(0), y)
            importance = dict(zip(X.columns, np.abs(model.coef_)))
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 归一化重要性
        max_importance = max(importance.values()) if importance.values() else 1
        if max_importance > 0:
            importance = {k: v/max_importance for k, v in importance.items()}
        
        self.feature_importance[method] = importance
        return importance
    
    def analyze_feature_correlations(self, X: pd.DataFrame) -> pd.DataFrame:
        """分析特征相关性"""
        correlation_matrix = X.corr()
        self.feature_correlations = correlation_matrix
        return correlation_matrix
    
    def detect_multicollinearity(self, X: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """检测多重共线性"""
        corr_matrix = self.analyze_feature_correlations(X)
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        return high_corr_pairs
    
    def analyze_feature_stability(self, 
                                X: pd.DataFrame,
                                time_periods: int = 5) -> Dict[str, float]:
        """分析特征稳定性"""
        n_samples = len(X)
        period_size = n_samples // time_periods
        
        stability_scores = {}
        
        for feature in X.columns:
            period_means = []
            
            for i in range(time_periods):
                start_idx = i * period_size
                end_idx = min((i + 1) * period_size, n_samples)
                
                if end_idx > start_idx:
                    period_data = X[feature].iloc[start_idx:end_idx]
                    period_means.append(period_data.mean())
            
            if len(period_means) > 1:
                # 计算变异系数作为稳定性指标
                mean_of_means = np.mean(period_means)
                std_of_means = np.std(period_means)
                cv = std_of_means / abs(mean_of_means) if mean_of_means != 0 else np.inf
                stability_scores[feature] = 1 / (1 + cv)  # 转换为稳定性得分
            else:
                stability_scores[feature] = 0.5
        
        self.feature_stability = stability_scores
        return stability_scores
    
    def select_features(self, 
                       X: pd.DataFrame,
                       y: pd.Series,
                       method: str = 'combined',
                       top_k: int = 20) -> List[str]:
        """特征选择"""
        if method == 'importance':
            importance = self.analyze_feature_importance(X, y)
            selected_features = sorted(importance.keys(), 
                                     key=lambda x: importance[x], 
                                     reverse=True)[:top_k]
            
        elif method == 'stability':
            stability = self.analyze_feature_stability(X)
            selected_features = sorted(stability.keys(),
                                     key=lambda x: stability[x],
                                     reverse=True)[:top_k]
            
        elif method == 'combined':
            importance = self.analyze_feature_importance(X, y)
            stability = self.analyze_feature_stability(X)
            
            # 组合得分
            combined_scores = {}
            for feature in X.columns:
                imp_score = importance.get(feature, 0)
                stab_score = stability.get(feature, 0)
                combined_scores[feature] = 0.7 * imp_score + 0.3 * stab_score
            
            selected_features = sorted(combined_scores.keys(),
                                     key=lambda x: combined_scores[x],
                                     reverse=True)[:top_k]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return selected_features


class ModelEnsemble:
    """模型集成器"""
    
    def __init__(self, models: Optional[List[BaseEstimator]] = None):
        self.models = models or self._get_default_models()
        self.model_weights = {}
        self.fitted_models = {}
        self.ensemble_method = 'weighted_average'
        
    def _get_default_models(self) -> List[BaseEstimator]:
        """获取默认模型集合"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.svm import SVR
        
        return [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            LinearRegression(),
            Ridge(alpha=1.0),
            SVR(kernel='rbf', C=1.0)
        ]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练集成模型"""
        self.fitted_models = {}
        
        for i, model in enumerate(self.models):
            try:
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X.fillna(0), y)
                self.fitted_models[f'model_{i}'] = model_copy
            except Exception as e:
                print(f"Error fitting model {i}: {e}")
                continue
        
        # 计算模型权重（基于交叉验证性能）
        self._calculate_model_weights(X, y)
    
    def _calculate_model_weights(self, X: pd.DataFrame, y: pd.Series):
        """计算模型权重"""
        from sklearn.model_selection import cross_val_score
        
        model_scores = {}
        
        for model_name, model in self.fitted_models.items():
            try:
                scores = cross_val_score(model, X.fillna(0), y, cv=5, 
                                       scoring='neg_mean_squared_error')
                model_scores[model_name] = -scores.mean()  # 转换为正值
            except Exception as e:
                print(f"Error calculating score for {model_name}: {e}")
                model_scores[model_name] = 1e6  # 大的惩罚值
        
        # 转换为权重（性能越好权重越大）
        if model_scores:
            # 使用倒数作为权重
            inverse_scores = {k: 1/v if v > 0 else 0 for k, v in model_scores.items()}
            total_weight = sum(inverse_scores.values())
            
            if total_weight > 0:
                self.model_weights = {k: v/total_weight for k, v in inverse_scores.items()}
            else:
                # 等权重
                self.model_weights = {k: 1/len(model_scores) for k in model_scores.keys()}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """集成预测"""
        if not self.fitted_models:
            raise ValueError("Models not fitted. Call fit() first.")
        
        predictions = {}
        
        for model_name, model in self.fitted_models.items():
            try:
                pred = model.predict(X.fillna(0))
                predictions[model_name] = pred
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # 集成预测
        if self.ensemble_method == 'weighted_average':
            ensemble_pred = np.zeros(len(X))
            
            for model_name, pred in predictions.items():
                weight = self.model_weights.get(model_name, 0)
                ensemble_pred += weight * pred
            
            return ensemble_pred
        
        elif self.ensemble_method == 'median':
            pred_matrix = np.column_stack(list(predictions.values()))
            return np.median(pred_matrix, axis=1)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """获取各模型性能"""
        performance = {}
        
        for model_name, model in self.fitted_models.items():
            try:
                pred = model.predict(X.fillna(0))
                mse = mean_squared_error(y, pred)
                performance[model_name] = mse
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                performance[model_name] = 1e6
        
        return performance


class TimeSeriesValidator:
    """时间序列验证器"""
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: Optional[int] = None,
                 gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.validation_results = []
        
    def validate_model(self, 
                      model: BaseEstimator,
                      X: pd.DataFrame,
                      y: pd.Series,
                      metrics: List[str] = ['mse', 'mae']) -> Dict[str, Any]:
        """时间序列交叉验证"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, 
                              test_size=self.test_size,
                              gap=self.gap)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # 分割数据
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 训练模型
            try:
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train.fillna(0), y_train)
                
                # 预测
                y_pred = model_copy.predict(X_test.fillna(0))
                
                # 计算指标
                fold_metrics = {}
                
                for metric in metrics:
                    if metric == 'mse':
                        fold_metrics[metric] = mean_squared_error(y_test, y_pred)
                    elif metric == 'mae':
                        from sklearn.metrics import mean_absolute_error
                        fold_metrics[metric] = mean_absolute_error(y_test, y_pred)
                    elif metric == 'rmse':
                        fold_metrics[metric] = np.sqrt(mean_squared_error(y_test, y_pred))
                    elif metric == 'r2':
                        from sklearn.metrics import r2_score
                        fold_metrics[metric] = r2_score(y_test, y_pred)
                
                fold_results.append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'metrics': fold_metrics
                })
                
            except Exception as e:
                print(f"Error in fold {fold}: {e}")
                continue
        
        # 汇总结果
        if fold_results:
            summary_metrics = {}
            for metric in metrics:
                metric_values = [fold['metrics'][metric] for fold in fold_results 
                               if metric in fold['metrics']]
                if metric_values:
                    summary_metrics[f'{metric}_mean'] = np.mean(metric_values)
                    summary_metrics[f'{metric}_std'] = np.std(metric_values)
            
            validation_result = {
                'fold_results': fold_results,
                'summary_metrics': summary_metrics,
                'n_folds': len(fold_results)
            }
            
            self.validation_results.append(validation_result)
            return validation_result
        
        else:
            return {'error': 'No successful validation folds'}
    
    def validate_strategy(self,
                         strategy_func: Callable,
                         data: pd.DataFrame,
                         **strategy_params) -> Dict[str, Any]:
        """验证交易策略"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits,
                              test_size=self.test_size,
                              gap=self.gap)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            try:
                # 分割数据
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # 在训练集上拟合策略参数（如果需要）
                # 在测试集上运行策略
                strategy_result = strategy_func(test_data, **strategy_params)
                
                # 计算策略指标
                if 'returns' in strategy_result:
                    returns = strategy_result['returns']
                    
                    fold_metrics = {
                        'total_return': (1 + returns).prod() - 1,
                        'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                        'max_drawdown': self._calculate_max_drawdown(returns),
                        'win_rate': (returns > 0).mean(),
                        'avg_return': returns.mean()
                    }
                    
                    fold_results.append({
                        'fold': fold,
                        'metrics': fold_metrics,
                        'returns': returns
                    })
                
            except Exception as e:
                print(f"Error in strategy validation fold {fold}: {e}")
                continue
        
        # 汇总结果
        if fold_results:
            summary_metrics = {}
            metric_names = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'avg_return']
            
            for metric in metric_names:
                metric_values = [fold['metrics'][metric] for fold in fold_results]
                summary_metrics[f'{metric}_mean'] = np.mean(metric_values)
                summary_metrics[f'{metric}_std'] = np.std(metric_values)
            
            return {
                'fold_results': fold_results,
                'summary_metrics': summary_metrics,
                'n_folds': len(fold_results)
            }
        
        else:
            return {'error': 'No successful strategy validation folds'}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())