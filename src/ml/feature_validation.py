#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征验证框架

提供特征质量验证功能：
- 特征稳定性测试
- 信息泄露检测
- 特征重要性分析
- 特征分布分析
- 时间序列特征验证
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FeatureStabilityTester:
    """特征稳定性测试"""
    
    def __init__(self, 
                 time_column: str = 'date',
                 test_periods: int = 5,
                 min_period_size: int = 100):
        self.time_column = time_column
        self.test_periods = test_periods
        self.min_period_size = min_period_size
        self.stability_results_ = {}
    
    def test_stability(self, 
                      df: pd.DataFrame, 
                      features: List[str]) -> Dict[str, Dict]:
        """测试特征稳定性"""
        print(f"开始特征稳定性测试，测试 {len(features)} 个特征...")
        
        # 按时间排序
        df_sorted = df.sort_values(self.time_column)
        
        # 分割时间段
        n_samples = len(df_sorted)
        period_size = max(n_samples // self.test_periods, self.min_period_size)
        
        periods = []
        for i in range(0, n_samples - period_size + 1, period_size):
            end_idx = min(i + period_size, n_samples)
            periods.append(df_sorted.iloc[i:end_idx])
        
        if len(periods) < 2:
            print("数据不足以进行稳定性测试")
            return {}
        
        # 测试每个特征的稳定性
        for feature in features:
            print(f"测试特征: {feature}")
            
            stability_metrics = {
                'mean_stability': [],
                'std_stability': [],
                'distribution_stability': [],
                'correlation_stability': []
            }
            
            # 计算各时期的统计量
            period_stats = []
            for period in periods:
                if feature in period.columns:
                    data = period[feature].dropna()
                    if len(data) > 0:
                        period_stats.append({
                            'mean': data.mean(),
                            'std': data.std(),
                            'skew': data.skew(),
                            'kurt': data.kurtosis(),
                            'data': data.values
                        })
            
            if len(period_stats) < 2:
                continue
            
            # 均值稳定性
            means = [stat['mean'] for stat in period_stats]
            stability_metrics['mean_stability'] = {
                'coefficient_of_variation': np.std(means) / np.mean(means) if np.mean(means) != 0 else np.inf,
                'trend_test': self._trend_test(means)
            }
            
            # 标准差稳定性
            stds = [stat['std'] for stat in period_stats]
            stability_metrics['std_stability'] = {
                'coefficient_of_variation': np.std(stds) / np.mean(stds) if np.mean(stds) != 0 else np.inf,
                'trend_test': self._trend_test(stds)
            }
            
            # 分布稳定性（KS检验）
            ks_results = []
            for i in range(len(period_stats) - 1):
                for j in range(i + 1, len(period_stats)):
                    ks_stat, p_value = ks_2samp(
                        period_stats[i]['data'], 
                        period_stats[j]['data']
                    )
                    ks_results.append({'ks_stat': ks_stat, 'p_value': p_value})
            
            stability_metrics['distribution_stability'] = {
                'mean_ks_stat': np.mean([r['ks_stat'] for r in ks_results]),
                'mean_p_value': np.mean([r['p_value'] for r in ks_results]),
                'stable_distribution': np.mean([r['p_value'] for r in ks_results]) > 0.05
            }
            
            # 计算稳定性评分
            stability_score = self._calculate_stability_score(stability_metrics)
            stability_metrics['overall_score'] = stability_score
            
            self.stability_results_[feature] = stability_metrics
        
        return self.stability_results_
    
    def _trend_test(self, values):
        """趋势检验"""
        try:
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            return {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'has_trend': p_value < 0.05
            }
        except:
            return {'slope': 0, 'r_squared': 0, 'p_value': 1, 'has_trend': False}
    
    def _calculate_stability_score(self, metrics):
        """计算综合稳定性评分"""
        score = 0
        
        # 均值稳定性 (25%)
        mean_cv = metrics['mean_stability']['coefficient_of_variation']
        if mean_cv < 0.1:
            score += 25
        elif mean_cv < 0.2:
            score += 20
        elif mean_cv < 0.5:
            score += 15
        else:
            score += 5
        
        # 标准差稳定性 (25%)
        std_cv = metrics['std_stability']['coefficient_of_variation']
        if std_cv < 0.1:
            score += 25
        elif std_cv < 0.2:
            score += 20
        elif std_cv < 0.5:
            score += 15
        else:
            score += 5
        
        # 分布稳定性 (50%)
        if metrics['distribution_stability']['stable_distribution']:
            score += 50
        else:
            # 根据p值给分
            p_val = metrics['distribution_stability']['mean_p_value']
            if p_val > 0.01:
                score += 30
            elif p_val > 0.001:
                score += 20
            else:
                score += 10
        
        return score
    
    def get_stability_report(self) -> pd.DataFrame:
        """获取稳定性报告"""
        report_data = []
        
        for feature, metrics in self.stability_results_.items():
            report_data.append({
                'feature': feature,
                'stability_score': metrics['overall_score'],
                'mean_cv': metrics['mean_stability']['coefficient_of_variation'],
                'std_cv': metrics['std_stability']['coefficient_of_variation'],
                'distribution_stable': metrics['distribution_stability']['stable_distribution'],
                'mean_trend': metrics['mean_stability']['trend_test']['has_trend'],
                'std_trend': metrics['std_stability']['trend_test']['has_trend']
            })
        
        df = pd.DataFrame(report_data)
        return df.sort_values('stability_score', ascending=False)


class InformationLeakageDetector:
    """信息泄露检测"""
    
    def __init__(self, 
                 time_column: str = 'date',
                 target_column: str = 'target',
                 lookahead_periods: List[int] = [1, 2, 3, 5]):
        self.time_column = time_column
        self.target_column = target_column
        self.lookahead_periods = lookahead_periods
        self.leakage_results_ = {}
    
    def detect_leakage(self, 
                      df: pd.DataFrame, 
                      features: List[str]) -> Dict[str, Dict]:
        """检测信息泄露"""
        print(f"开始信息泄露检测，检测 {len(features)} 个特征...")
        
        # 按时间排序
        df_sorted = df.sort_values(self.time_column).reset_index(drop=True)
        
        for feature in features:
            print(f"检测特征: {feature}")
            
            leakage_metrics = {
                'future_correlation': {},
                'prediction_power': {},
                'temporal_consistency': {}
            }
            
            # 检测与未来目标的相关性
            for period in self.lookahead_periods:
                future_target = df_sorted[self.target_column].shift(-period)
                current_feature = df_sorted[feature]
                
                # 计算相关性
                valid_mask = ~(current_feature.isna() | future_target.isna())
                if valid_mask.sum() > 10:
                    correlation = current_feature[valid_mask].corr(future_target[valid_mask])
                    leakage_metrics['future_correlation'][f'period_{period}'] = {
                        'correlation': correlation,
                        'abs_correlation': abs(correlation),
                        'suspicious': abs(correlation) > 0.3
                    }
            
            # 检测预测能力异常
            leakage_metrics['prediction_power'] = self._test_prediction_power(
                df_sorted, feature
            )
            
            # 检测时间一致性
            leakage_metrics['temporal_consistency'] = self._test_temporal_consistency(
                df_sorted, feature
            )
            
            # 计算泄露风险评分
            risk_score = self._calculate_leakage_risk(leakage_metrics)
            leakage_metrics['risk_score'] = risk_score
            
            self.leakage_results_[feature] = leakage_metrics
        
        return self.leakage_results_
    
    def _test_prediction_power(self, df, feature):
        """测试预测能力"""
        try:
            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            
            X = df[[feature]].fillna(df[feature].mean())
            y = df[self.target_column].fillna(df[self.target_column].mean())
            
            # 简单模型预测
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            
            return {
                'cv_score_mean': -scores.mean(),
                'cv_score_std': scores.std(),
                'suspiciously_good': -scores.mean() < 0.01  # 预测误差异常小
            }
        except:
            return {'cv_score_mean': np.inf, 'cv_score_std': 0, 'suspiciously_good': False}
    
    def _test_temporal_consistency(self, df, feature):
        """测试时间一致性"""
        try:
            # 检查特征值是否在时间上过于规律
            feature_values = df[feature].dropna()
            
            if len(feature_values) < 10:
                return {'consistent': True, 'pattern_detected': False}
            
            # 检测周期性模式
            autocorr_1 = feature_values.autocorr(lag=1)
            autocorr_7 = feature_values.autocorr(lag=7) if len(feature_values) > 7 else 0
            
            # 检测异常规律性
            pattern_detected = (abs(autocorr_1) > 0.9 or abs(autocorr_7) > 0.8)
            
            return {
                'autocorr_1': autocorr_1,
                'autocorr_7': autocorr_7,
                'pattern_detected': pattern_detected,
                'consistent': not pattern_detected
            }
        except:
            return {'consistent': True, 'pattern_detected': False}
    
    def _calculate_leakage_risk(self, metrics):
        """计算泄露风险评分"""
        risk_score = 0
        
        # 未来相关性风险 (50%)
        future_corrs = []
        for period_data in metrics['future_correlation'].values():
            if period_data['suspicious']:
                risk_score += 20
            future_corrs.append(period_data['abs_correlation'])
        
        if future_corrs and max(future_corrs) > 0.5:
            risk_score += 30
        
        # 预测能力风险 (30%)
        if metrics['prediction_power']['suspiciously_good']:
            risk_score += 30
        
        # 时间一致性风险 (20%)
        if metrics['temporal_consistency']['pattern_detected']:
            risk_score += 20
        
        return min(risk_score, 100)
    
    def get_leakage_report(self) -> pd.DataFrame:
        """获取泄露检测报告"""
        report_data = []
        
        for feature, metrics in self.leakage_results_.items():
            # 计算最大未来相关性
            max_future_corr = 0
            if metrics['future_correlation']:
                max_future_corr = max([
                    data['abs_correlation'] 
                    for data in metrics['future_correlation'].values()
                ])
            
            report_data.append({
                'feature': feature,
                'risk_score': metrics['risk_score'],
                'max_future_correlation': max_future_corr,
                'suspiciously_good_prediction': metrics['prediction_power']['suspiciously_good'],
                'pattern_detected': metrics['temporal_consistency']['pattern_detected'],
                'high_risk': metrics['risk_score'] > 50
            })
        
        df = pd.DataFrame(report_data)
        return df.sort_values('risk_score', ascending=False)


class FeatureImportanceAnalyzer:
    """特征重要性分析"""
    
    def __init__(self, 
                 models: Optional[List] = None,
                 cv_folds: int = 5):
        self.models = models or [
            RandomForestRegressor(n_estimators=100, random_state=42)
        ]
        self.cv_folds = cv_folds
        self.importance_results_ = {}
    
    def analyze_importance(self, 
                          X: pd.DataFrame, 
                          y: pd.Series) -> Dict[str, pd.DataFrame]:
        """分析特征重要性"""
        print(f"开始特征重要性分析，分析 {X.shape[1]} 个特征...")
        
        for i, model in enumerate(self.models):
            model_name = f"model_{i}_{type(model).__name__}"
            print(f"使用模型: {model_name}")
            
            # 交叉验证计算重要性
            importance_scores = []
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 获取特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance_scores.append(model.feature_importances_)
                elif hasattr(model, 'coef_'):
                    importance_scores.append(np.abs(model.coef_))
            
            # 计算平均重要性
            if importance_scores:
                mean_importance = np.mean(importance_scores, axis=0)
                std_importance = np.std(importance_scores, axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance_mean': mean_importance,
                    'importance_std': std_importance,
                    'importance_cv': std_importance / (mean_importance + 1e-8),
                    'rank': np.argsort(-mean_importance) + 1
                })
                
                self.importance_results_[model_name] = importance_df.sort_values(
                    'importance_mean', ascending=False
                )
        
        return self.importance_results_
    
    def get_consensus_importance(self) -> pd.DataFrame:
        """获取共识重要性排名"""
        if not self.importance_results_:
            return pd.DataFrame()
        
        # 收集所有模型的排名
        all_rankings = []
        features = None
        
        for model_name, importance_df in self.importance_results_.items():
            if features is None:
                features = importance_df['feature'].tolist()
            
            ranking = importance_df.set_index('feature')['rank']
            all_rankings.append(ranking)
        
        # 计算平均排名
        consensus_df = pd.DataFrame({
            'feature': features,
            'mean_rank': np.mean([ranking[features] for ranking in all_rankings], axis=0),
            'std_rank': np.std([ranking[features] for ranking in all_rankings], axis=0),
            'mean_importance': np.mean([
                self.importance_results_[model]['importance_mean'] 
                for model in self.importance_results_
            ], axis=0)
        })
        
        return consensus_df.sort_values('mean_rank')


class FeatureValidationPipeline:
    """特征验证管道"""
    
    def __init__(self, 
                 time_column: str = 'date',
                 target_column: str = 'target'):
        self.time_column = time_column
        self.target_column = target_column
        
        self.stability_tester = FeatureStabilityTester(time_column)
        self.leakage_detector = InformationLeakageDetector(time_column, target_column)
        self.importance_analyzer = FeatureImportanceAnalyzer()
        
        self.validation_results_ = {}
    
    def validate_features(self, 
                         df: pd.DataFrame, 
                         features: List[str]) -> Dict[str, pd.DataFrame]:
        """完整的特征验证"""
        print(f"开始完整特征验证，验证 {len(features)} 个特征...")
        
        # 准备数据
        X = df[features]
        y = df[self.target_column]
        
        # 1. 稳定性测试
        print("\n=== 稳定性测试 ===")
        self.stability_tester.test_stability(df, features)
        stability_report = self.stability_tester.get_stability_report()
        
        # 2. 信息泄露检测
        print("\n=== 信息泄露检测 ===")
        self.leakage_detector.detect_leakage(df, features)
        leakage_report = self.leakage_detector.get_leakage_report()
        
        # 3. 重要性分析
        print("\n=== 重要性分析 ===")
        self.importance_analyzer.analyze_importance(X, y)
        importance_report = self.importance_analyzer.get_consensus_importance()
        
        # 4. 综合报告
        comprehensive_report = self._create_comprehensive_report(
            stability_report, leakage_report, importance_report
        )
        
        self.validation_results_ = {
            'stability': stability_report,
            'leakage': leakage_report,
            'importance': importance_report,
            'comprehensive': comprehensive_report
        }
        
        return self.validation_results_
    
    def _create_comprehensive_report(self, 
                                   stability_report, 
                                   leakage_report, 
                                   importance_report):
        """创建综合报告"""
        # 合并所有报告
        comprehensive = stability_report.set_index('feature')
        
        if not leakage_report.empty:
            leakage_subset = leakage_report.set_index('feature')[['risk_score', 'high_risk']]
            comprehensive = comprehensive.join(leakage_subset, how='left')
        
        if not importance_report.empty:
            importance_subset = importance_report.set_index('feature')[['mean_rank', 'mean_importance']]
            comprehensive = comprehensive.join(importance_subset, how='left')
        
        # 计算综合质量评分
        comprehensive['quality_score'] = self._calculate_quality_score(comprehensive)
        
        # 添加推荐
        comprehensive['recommendation'] = comprehensive.apply(self._get_recommendation, axis=1)
        
        return comprehensive.reset_index().sort_values('quality_score', ascending=False)
    
    def _calculate_quality_score(self, row):
        """计算特征质量评分"""
        score = 0
        
        # 稳定性评分 (40%)
        if not pd.isna(row['stability_score']):
            score += row['stability_score'] * 0.4
        
        # 泄露风险评分 (30%)
        if not pd.isna(row.get('risk_score', np.nan)):
            score += (100 - row['risk_score']) * 0.3
        else:
            score += 30  # 默认给30分
        
        # 重要性评分 (30%)
        if not pd.isna(row.get('mean_importance', np.nan)):
            # 将重要性转换为0-30分
            max_importance = row.get('mean_importance', 0)
            importance_score = min(max_importance * 1000, 30)  # 假设重要性在0-0.03范围
            score += importance_score
        
        return min(score, 100)
    
    def _get_recommendation(self, row):
        """获取特征推荐"""
        quality_score = row['quality_score']
        high_risk = row.get('high_risk', False)
        
        if high_risk:
            return "REJECT - 高泄露风险"
        elif quality_score >= 80:
            return "ACCEPT - 高质量特征"
        elif quality_score >= 60:
            return "CONDITIONAL - 需要进一步验证"
        elif quality_score >= 40:
            return "CAUTION - 质量一般"
        else:
            return "REJECT - 质量较差"
    
    def get_feature_recommendations(self) -> Dict[str, List[str]]:
        """获取特征推荐分类"""
        if 'comprehensive' not in self.validation_results_:
            return {}
        
        report = self.validation_results_['comprehensive']
        
        recommendations = {
            'accept': [],
            'conditional': [],
            'caution': [],
            'reject': []
        }
        
        for _, row in report.iterrows():
            feature = row['feature']
            recommendation = row['recommendation']
            
            if 'ACCEPT' in recommendation:
                recommendations['accept'].append(feature)
            elif 'CONDITIONAL' in recommendation:
                recommendations['conditional'].append(feature)
            elif 'CAUTION' in recommendation:
                recommendations['caution'].append(feature)
            else:
                recommendations['reject'].append(feature)
        
        return recommendations


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # 创建正常特征
    normal_feature = np.cumsum(np.random.randn(n_samples) * 0.01)
    
    # 创建有泄露的特征（使用未来信息）
    target = np.cumsum(np.random.randn(n_samples) * 0.02)
    leaky_feature = np.roll(target, -1)  # 使用未来1期的目标值
    
    # 创建不稳定特征
    unstable_feature = np.random.randn(n_samples)
    unstable_feature[500:] += 5  # 后半段均值发生跳跃
    
    df = pd.DataFrame({
        'date': dates,
        'target': target,
        'normal_feature': normal_feature,
        'leaky_feature': leaky_feature,
        'unstable_feature': unstable_feature,
        'noise_feature': np.random.randn(n_samples)
    })
    
    # 特征验证
    validator = FeatureValidationPipeline()
    
    features_to_validate = ['normal_feature', 'leaky_feature', 'unstable_feature', 'noise_feature']
    
    results = validator.validate_features(df, features_to_validate)
    
    print("\n=== 综合验证报告 ===")
    print(results['comprehensive'][['feature', 'quality_score', 'recommendation']])
    
    print("\n=== 特征推荐 ===")
    recommendations = validator.get_feature_recommendations()
    for category, features in recommendations.items():
        print(f"{category.upper()}: {features}")