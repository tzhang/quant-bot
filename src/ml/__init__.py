#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习模块

提供特征工程、模型集成、模型验证和Kaggle比赛工具等功能
"""

# 特征工程
from .feature_engineering import (
    AdvancedFeatureEngineer,
    TimeSeriesFeatureExtractor,
    TechnicalIndicatorFeatures,
    StatisticalFeatures
)

# 模型集成
from .model_ensemble import (
    AdvancedEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    VotingEnsemble,
    DynamicEnsemble,
    AutoEnsemble,
    create_default_models,
    create_competition_ensemble
)

# 模型验证
from .model_validation import (
    TimeSeriesCrossValidator,
    PurgedGroupTimeSeriesSplit,
    ModelValidator,
    HyperparameterOptimizer,
    create_time_series_cv,
    quick_model_comparison
)

# Kaggle工具
from .kaggle_tools import (
    KaggleDataProcessor,
    KaggleSubmissionGenerator,
    KaggleModelTrainer,
    quick_kaggle_pipeline
)

__all__ = [
    # 特征工程
    'AdvancedFeatureEngineer',
    'TimeSeriesFeatureExtractor', 
    'TechnicalIndicatorFeatures',
    'StatisticalFeatures',
    
    # 模型集成
    'AdvancedEnsemble',
    'StackingEnsemble',
    'BlendingEnsemble', 
    'VotingEnsemble',
    'DynamicEnsemble',
    'AutoEnsemble',
    'create_default_models',
    'create_competition_ensemble',
    
    # 模型验证
    'TimeSeriesCrossValidator',
    'PurgedGroupTimeSeriesSplit',
    'ModelValidator',
    'HyperparameterOptimizer',
    'create_time_series_cv',
    'quick_model_comparison',
    
    # Kaggle工具
    'KaggleDataProcessor',
    'KaggleSubmissionGenerator', 
    'KaggleModelTrainer',
    'quick_kaggle_pipeline'
]