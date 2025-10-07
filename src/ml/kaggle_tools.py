#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle比赛专用工具包

提供数据预处理、特征工程、模型训练和提交文件生成等功能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import os
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class KaggleDataProcessor:
    """Kaggle数据处理器"""
    
    def __init__(self, target_col: str = 'target'):
        """
        初始化数据处理器
        
        Args:
            target_col: 目标列名
        """
        self.target_col = target_col
        self.feature_cols = None
        self.preprocessors = {}
        self.is_fitted = False
        
    def load_competition_data(self, 
                            train_path: str,
                            test_path: str,
                            sample_submission_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """加载比赛数据"""
        print("加载训练数据...")
        train_df = pd.read_csv(train_path)
        
        print("加载测试数据...")
        test_df = pd.read_csv(test_path)
        
        sample_submission = None
        if sample_submission_path and os.path.exists(sample_submission_path):
            print("加载样本提交文件...")
            sample_submission = pd.read_csv(sample_submission_path)
            
        print(f"训练数据形状: {train_df.shape}")
        print(f"测试数据形状: {test_df.shape}")
        if sample_submission is not None:
            print(f"样本提交文件形状: {sample_submission.shape}")
            
        return train_df, test_df, sample_submission
    
    def analyze_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据特征"""
        analysis = {
            'train_shape': train_df.shape,
            'test_shape': test_df.shape,
            'train_columns': list(train_df.columns),
            'test_columns': list(test_df.columns),
            'common_columns': list(set(train_df.columns) & set(test_df.columns)),
            'train_only_columns': list(set(train_df.columns) - set(test_df.columns)),
            'test_only_columns': list(set(test_df.columns) - set(train_df.columns)),
            'missing_values_train': train_df.isnull().sum().to_dict(),
            'missing_values_test': test_df.isnull().sum().to_dict(),
            'data_types_train': train_df.dtypes.to_dict(),
            'data_types_test': test_df.dtypes.to_dict()
        }
        
        # 数值列统计
        numeric_cols_train = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols_test = test_df.select_dtypes(include=[np.number]).columns
        
        analysis['numeric_columns_train'] = list(numeric_cols_train)
        analysis['numeric_columns_test'] = list(numeric_cols_test)
        
        if len(numeric_cols_train) > 0:
            analysis['numeric_stats_train'] = train_df[numeric_cols_train].describe().to_dict()
            
        if len(numeric_cols_test) > 0:
            analysis['numeric_stats_test'] = test_df[numeric_cols_test].describe().to_dict()
            
        # 分类列统计
        categorical_cols_train = train_df.select_dtypes(include=['object', 'category']).columns
        categorical_cols_test = test_df.select_dtypes(include=['object', 'category']).columns
        
        analysis['categorical_columns_train'] = list(categorical_cols_train)
        analysis['categorical_columns_test'] = list(categorical_cols_test)
        
        # 目标变量分析（如果存在）
        if self.target_col in train_df.columns:
            target_series = train_df[self.target_col]
            analysis['target_stats'] = {
                'mean': target_series.mean(),
                'std': target_series.std(),
                'min': target_series.min(),
                'max': target_series.max(),
                'skewness': target_series.skew(),
                'kurtosis': target_series.kurtosis(),
                'missing_count': target_series.isnull().sum()
            }
            
        return analysis
    
    def preprocess_data(self, 
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_engineering: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """预处理数据"""
        print("开始数据预处理...")
        
        # 合并数据进行统一处理
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # 标记训练集和测试集
        train_df['is_train'] = 1
        test_df['is_train'] = 0
        
        # 如果测试集没有目标列，添加占位符
        if self.target_col not in test_df.columns and self.target_col in train_df.columns:
            test_df[self.target_col] = np.nan
            
        # 合并数据
        all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        # 基础预处理
        all_data = self._handle_missing_values(all_data)
        all_data = self._encode_categorical_features(all_data)
        
        if feature_engineering:
            all_data = self._engineer_features(all_data)
            
        # 分离训练集和测试集
        train_processed = all_data[all_data['is_train'] == 1].drop('is_train', axis=1).reset_index(drop=True)
        test_processed = all_data[all_data['is_train'] == 0].drop('is_train', axis=1).reset_index(drop=True)
        
        # 移除测试集中的目标列（如果是占位符）
        if self.target_col in test_processed.columns and test_processed[self.target_col].isnull().all():
            test_processed = test_processed.drop(self.target_col, axis=1)
            
        self.is_fitted = True
        print("数据预处理完成")
        
        return train_processed, test_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        from sklearn.impute import SimpleImputer
        
        # 数值列：用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        if len(numeric_cols) > 0:
            if 'numeric_imputer' not in self.preprocessors:
                self.preprocessors['numeric_imputer'] = SimpleImputer(strategy='median')
                df[numeric_cols] = self.preprocessors['numeric_imputer'].fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.preprocessors['numeric_imputer'].transform(df[numeric_cols])
                
        # 分类列：用众数填充
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            if 'categorical_imputer' not in self.preprocessors:
                self.preprocessors['categorical_imputer'] = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = self.preprocessors['categorical_imputer'].fit_transform(df[categorical_cols])
            else:
                df[categorical_cols] = self.preprocessors['categorical_imputer'].transform(df[categorical_cols])
                
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.preprocessors:
                self.preprocessors[col] = LabelEncoder()
                df[col] = self.preprocessors[col].fit_transform(df[col].astype(str))
            else:
                # 处理新的类别
                known_classes = set(self.preprocessors[col].classes_)
                new_classes = set(df[col].astype(str).unique()) - known_classes
                
                if new_classes:
                    # 扩展编码器的类别
                    all_classes = list(known_classes) + list(new_classes)
                    self.preprocessors[col].classes_ = np.array(all_classes)
                    
                df[col] = df[col].astype(str).map(
                    lambda x: self.preprocessors[col].transform([x])[0] 
                    if x in self.preprocessors[col].classes_ 
                    else len(self.preprocessors[col].classes_) - 1
                )
                
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        # 数值特征统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in [self.target_col, 'is_train']]
        
        if len(numeric_cols) > 1:
            # 特征组合
            df['feature_sum'] = df[numeric_cols].sum(axis=1)
            df['feature_mean'] = df[numeric_cols].mean(axis=1)
            df['feature_std'] = df[numeric_cols].std(axis=1)
            df['feature_max'] = df[numeric_cols].max(axis=1)
            df['feature_min'] = df[numeric_cols].min(axis=1)
            
            # 特征比率
            if len(numeric_cols) >= 2:
                df['feature_ratio_1'] = df[numeric_cols[0]] / (df[numeric_cols[1]] + 1e-8)
                df['feature_ratio_2'] = df[numeric_cols[1]] / (df[numeric_cols[0]] + 1e-8)
                
        return df
    
    def save_preprocessors(self, filepath: str):
        """保存预处理器"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.preprocessors, f)
            
    def load_preprocessors(self, filepath: str):
        """加载预处理器"""
        with open(filepath, 'rb') as f:
            self.preprocessors = pickle.load(f)
        self.is_fitted = True


class KaggleSubmissionGenerator:
    """Kaggle提交文件生成器"""
    
    def __init__(self, competition_name: str = "competition"):
        """
        初始化提交文件生成器
        
        Args:
            competition_name: 比赛名称
        """
        self.competition_name = competition_name
        self.submissions_dir = f"submissions/{competition_name}"
        os.makedirs(self.submissions_dir, exist_ok=True)
        
    def create_submission(self, 
                         predictions: np.ndarray,
                         test_ids: Union[np.ndarray, pd.Series],
                         target_col: str = 'target',
                         id_col: str = 'id',
                         model_name: str = "model",
                         description: str = "") -> str:
        """创建提交文件"""
        # 创建提交DataFrame
        submission_df = pd.DataFrame({
            id_col: test_ids,
            target_col: predictions
        })
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.csv"
        filepath = os.path.join(self.submissions_dir, filename)
        
        # 保存文件
        submission_df.to_csv(filepath, index=False)
        
        # 保存元数据
        metadata = {
            'filename': filename,
            'model_name': model_name,
            'description': description,
            'timestamp': timestamp,
            'predictions_count': len(predictions),
            'predictions_mean': float(np.mean(predictions)),
            'predictions_std': float(np.std(predictions)),
            'predictions_min': float(np.min(predictions)),
            'predictions_max': float(np.max(predictions))
        }
        
        metadata_filepath = os.path.join(self.submissions_dir, f"{model_name}_{timestamp}_metadata.json")
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"提交文件已保存: {filepath}")
        print(f"元数据已保存: {metadata_filepath}")
        
        return filepath
    
    def blend_submissions(self, 
                         submission_files: List[str],
                         weights: Optional[List[float]] = None,
                         blend_name: str = "blend") -> str:
        """混合多个提交文件"""
        if weights is None:
            weights = [1.0 / len(submission_files)] * len(submission_files)
            
        if len(weights) != len(submission_files):
            raise ValueError("权重数量必须与提交文件数量相等")
            
        # 读取所有提交文件
        submissions = []
        for filepath in submission_files:
            df = pd.read_csv(filepath)
            submissions.append(df)
            
        # 检查ID列是否一致
        base_ids = submissions[0].iloc[:, 0]
        for i, sub in enumerate(submissions[1:], 1):
            if not base_ids.equals(sub.iloc[:, 0]):
                raise ValueError(f"提交文件 {i+1} 的ID列与第一个文件不匹配")
                
        # 混合预测
        blended_predictions = np.zeros(len(base_ids))
        for sub, weight in zip(submissions, weights):
            blended_predictions += weight * sub.iloc[:, 1].values
            
        # 创建混合提交文件
        return self.create_submission(
            predictions=blended_predictions,
            test_ids=base_ids,
            model_name=blend_name,
            description=f"Blend of {len(submission_files)} models with weights {weights}"
        )
    
    def analyze_submissions(self) -> pd.DataFrame:
        """分析所有提交文件"""
        metadata_files = [f for f in os.listdir(self.submissions_dir) if f.endswith('_metadata.json')]
        
        if not metadata_files:
            return pd.DataFrame()
            
        results = []
        for metadata_file in metadata_files:
            filepath = os.path.join(self.submissions_dir, metadata_file)
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            results.append(metadata)
            
        return pd.DataFrame(results).sort_values('timestamp', ascending=False)


class KaggleModelTrainer:
    """Kaggle模型训练器"""
    
    def __init__(self, 
                 competition_type: str = "regression",
                 random_state: int = 42):
        """
        初始化模型训练器
        
        Args:
            competition_type: 比赛类型 ('regression', 'classification', 'multiclass')
            random_state: 随机种子
        """
        self.competition_type = competition_type
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        
    def setup_models(self) -> Dict[str, Any]:
        """设置默认模型"""
        if self.competition_type == "regression":
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            import xgboost as xgb
            import lightgbm as lgb
            
            self.models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                'ridge': Ridge(alpha=1.0, random_state=self.random_state),
                'xgb': xgb.XGBRegressor(n_estimators=100, random_state=self.random_state),
                'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
            }
        elif self.competition_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            import xgboost as xgb
            import lightgbm as lgb
            
            self.models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                'lr': LogisticRegression(random_state=self.random_state),
                'xgb': xgb.XGBClassifier(n_estimators=100, random_state=self.random_state),
                'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1)
            }
            
        return self.models
    
    def train_models(self, 
                    X_train: Union[np.ndarray, pd.DataFrame],
                    y_train: Union[np.ndarray, pd.Series],
                    X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                    y_val: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """训练所有模型"""
        results = {}
        
        for name, model in self.models.items():
            print(f"训练模型: {name}")
            
            # 训练模型
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # 评估模型
            train_pred = model.predict(X_train)
            train_score = self._calculate_score(y_train, train_pred)
            
            result = {
                'model': model,
                'train_score': train_score
            }
            
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_score = self._calculate_score(y_val, val_pred)
                result['val_score'] = val_score
                print(f"{name} - 训练分数: {train_score:.4f}, 验证分数: {val_score:.4f}")
            else:
                print(f"{name} - 训练分数: {train_score:.4f}")
                
            results[name] = result
            
        return results
    
    def predict_test(self, 
                    X_test: Union[np.ndarray, pd.DataFrame],
                    model_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """对测试集进行预测"""
        if model_names is None:
            model_names = list(self.trained_models.keys())
            
        predictions = {}
        for name in model_names:
            if name in self.trained_models:
                pred = self.trained_models[name].predict(X_test)
                predictions[name] = pred
                
        return predictions
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算分数"""
        if self.competition_type == "regression":
            from sklearn.metrics import mean_squared_error
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.competition_type == "classification":
            from sklearn.metrics import accuracy_score
            return accuracy_score(y_true, y_pred)
        else:
            return 0.0


def quick_kaggle_pipeline(train_path: str,
                         test_path: str,
                         target_col: str = 'target',
                         competition_type: str = 'regression',
                         sample_submission_path: Optional[str] = None) -> Dict[str, Any]:
    """快速Kaggle比赛流水线"""
    
    # 1. 数据加载和预处理
    processor = KaggleDataProcessor(target_col=target_col)
    train_df, test_df, sample_submission = processor.load_competition_data(
        train_path, test_path, sample_submission_path
    )
    
    # 2. 数据分析
    analysis = processor.analyze_data(train_df, test_df)
    print("数据分析完成")
    
    # 3. 数据预处理
    train_processed, test_processed = processor.preprocess_data(train_df, test_df)
    
    # 4. 准备训练数据
    X_train = train_processed.drop(target_col, axis=1)
    y_train = train_processed[target_col]
    X_test = test_processed
    
    # 5. 模型训练
    trainer = KaggleModelTrainer(competition_type=competition_type)
    trainer.setup_models()
    
    # 简单的训练/验证分割
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    results = trainer.train_models(X_tr, y_tr, X_val, y_val)
    
    # 6. 测试集预测
    test_predictions = trainer.predict_test(X_test)
    
    # 7. 生成提交文件
    submission_generator = KaggleSubmissionGenerator()
    
    # 获取测试集ID（假设第一列是ID）
    if 'id' in test_df.columns:
        test_ids = test_df['id']
    else:
        test_ids = test_df.iloc[:, 0]
        
    submission_files = []
    for model_name, predictions in test_predictions.items():
        filepath = submission_generator.create_submission(
            predictions=predictions,
            test_ids=test_ids,
            target_col=target_col,
            model_name=model_name
        )
        submission_files.append(filepath)
        
    return {
        'analysis': analysis,
        'models': results,
        'predictions': test_predictions,
        'submission_files': submission_files,
        'processor': processor,
        'trainer': trainer,
        'submission_generator': submission_generator
    }