#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量管理系统

提供全面的数据质量管理功能：
1. 数据验证：格式、范围、一致性检查
2. 异常检测：统计异常、时间序列异常
3. 缺失值处理：检测、填充、插值
4. 数据清洗：重复值、异常值处理
5. 数据质量报告：生成质量评估报告
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class DataQualityManager:
    """
    数据质量管理器
    
    提供数据验证、清洗、异常检测和质量评估功能
    """
    
    def __init__(self, output_dir: str = "data_quality_reports"):
        """
        初始化数据质量管理器
        
        Args:
            output_dir: 质量报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据质量规则配置
        self.quality_rules = {
            'price_data': {
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'numeric_columns': ['open', 'high', 'low', 'close', 'volume'],
                'positive_columns': ['open', 'high', 'low', 'close', 'volume'],
                'price_consistency': True,  # high >= low, close在[low, high]范围内
                'volume_threshold': 0,  # 成交量应大于0
            },
            'fundamental_data': {
                'required_columns': ['totalRevenue', 'netIncome', 'totalAssets'],
                'numeric_columns': ['totalRevenue', 'netIncome', 'totalAssets', 'totalShareholderEquity'],
                'date_columns': ['fiscalDateEnding'],
            },
            'factor_data': {
                'required_columns': ['factor_value'],
                'numeric_columns': ['factor_value'],
                'outlier_threshold': 3,  # 标准差倍数
            }
        }
    
    def validate_data_format(self, df: pd.DataFrame, data_type: str = 'general') -> Dict[str, Any]:
        """
        验证数据格式
        
        Args:
            df: 待验证的DataFrame
            data_type: 数据类型（price_data, fundamental_data, factor_data等）
            
        Returns:
            验证结果字典
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            # 基本格式检查
            if df.empty:
                validation_result['errors'].append("数据为空")
                validation_result['is_valid'] = False
                return validation_result
            
            # 获取数据类型规则
            rules = self.quality_rules.get(data_type, {})
            
            # 检查必需列
            required_columns = rules.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_result['errors'].append(f"缺少必需列: {missing_columns}")
                validation_result['is_valid'] = False
            
            # 检查数值列
            numeric_columns = rules.get('numeric_columns', [])
            for col in numeric_columns:
                if col in df.columns:
                    non_numeric = df[col].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x)))
                    if non_numeric.any():
                        validation_result['warnings'].append(f"列 {col} 包含非数值数据")
            
            # 检查正值列
            positive_columns = rules.get('positive_columns', [])
            for col in positive_columns:
                if col in df.columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        validation_result['warnings'].append(f"列 {col} 包含 {negative_count} 个负值")
            
            # 检查日期列
            date_columns = rules.get('date_columns', [])
            for col in date_columns:
                if col in df.columns:
                    try:
                        pd.to_datetime(df[col])
                    except:
                        validation_result['errors'].append(f"列 {col} 日期格式无效")
                        validation_result['is_valid'] = False
            
            # 特定数据类型的验证
            if data_type == 'price_data':
                validation_result.update(self._validate_price_data(df))
            elif data_type == 'fundamental_data':
                validation_result.update(self._validate_fundamental_data(df))
            
            # 生成摘要
            validation_result['summary'] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict()
            }
            
            logger.info(f"数据格式验证完成: {data_type}, 有效性: {validation_result['is_valid']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"数据格式验证失败: {e}")
            validation_result['errors'].append(f"验证过程出错: {str(e)}")
            validation_result['is_valid'] = False
            return validation_result
    
    def _validate_price_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证价格数据特定规则"""
        result = {'errors': [], 'warnings': []}
        
        try:
            # 检查价格一致性：high >= low
            if 'high' in df.columns and 'low' in df.columns:
                inconsistent = df['high'] < df['low']
                if inconsistent.any():
                    count = inconsistent.sum()
                    result['errors'].append(f"发现 {count} 行最高价小于最低价")
            
            # 检查收盘价在合理范围内
            if all(col in df.columns for col in ['close', 'high', 'low']):
                out_of_range = (df['close'] > df['high']) | (df['close'] < df['low'])
                if out_of_range.any():
                    count = out_of_range.sum()
                    result['warnings'].append(f"发现 {count} 行收盘价超出最高最低价范围")
            
            # 检查成交量
            if 'volume' in df.columns:
                zero_volume = (df['volume'] == 0).sum()
                if zero_volume > 0:
                    result['warnings'].append(f"发现 {zero_volume} 行零成交量")
            
        except Exception as e:
            result['errors'].append(f"价格数据验证出错: {str(e)}")
        
        return result
    
    def _validate_fundamental_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证基本面数据特定规则"""
        result = {'errors': [], 'warnings': []}
        
        try:
            # 检查财务数据逻辑一致性
            if 'totalAssets' in df.columns and 'totalShareholderEquity' in df.columns:
                negative_equity = df['totalShareholderEquity'] < 0
                if negative_equity.any():
                    count = negative_equity.sum()
                    result['warnings'].append(f"发现 {count} 行负股东权益")
            
            # 检查收入和利润的合理性
            if 'totalRevenue' in df.columns and 'netIncome' in df.columns:
                # 净利润率超过100%可能有问题
                profit_margin = df['netIncome'] / df['totalRevenue']
                extreme_margin = (profit_margin > 1.0) | (profit_margin < -1.0)
                if extreme_margin.any():
                    count = extreme_margin.sum()
                    result['warnings'].append(f"发现 {count} 行极端净利润率")
            
        except Exception as e:
            result['errors'].append(f"基本面数据验证出错: {str(e)}")
        
        return result
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None, 
                       method: str = 'iqr') -> Dict[str, Any]:
        """
        检测异常值
        
        Args:
            df: 数据DataFrame
            columns: 要检测的列名列表，None表示所有数值列
            method: 检测方法（'iqr', 'zscore', 'isolation_forest'）
            
        Returns:
            异常值检测结果
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_result = {
            'method': method,
            'outliers': {},
            'summary': {},
            'recommendations': []
        }
        
        try:
            for col in columns:
                if col not in df.columns:
                    continue
                
                data = df[col].dropna()
                if len(data) == 0:
                    continue
                
                if method == 'iqr':
                    outliers = self._detect_outliers_iqr(data)
                elif method == 'zscore':
                    outliers = self._detect_outliers_zscore(data)
                elif method == 'isolation_forest':
                    outliers = self._detect_outliers_isolation_forest(data)
                else:
                    logger.warning(f"未知的异常值检测方法: {method}")
                    continue
                
                outlier_result['outliers'][col] = {
                    'indices': outliers.tolist(),
                    'count': len(outliers),
                    'percentage': len(outliers) / len(data) * 100
                }
            
            # 生成摘要
            total_outliers = sum(len(info['indices']) for info in outlier_result['outliers'].values())
            outlier_result['summary'] = {
                'total_outliers': total_outliers,
                'columns_with_outliers': len([col for col, info in outlier_result['outliers'].items() 
                                            if info['count'] > 0]),
                'average_outlier_percentage': np.mean([info['percentage'] 
                                                     for info in outlier_result['outliers'].values()])
            }
            
            # 生成建议
            if total_outliers > 0:
                outlier_result['recommendations'].append("发现异常值，建议进一步检查数据质量")
                if outlier_result['summary']['average_outlier_percentage'] > 5:
                    outlier_result['recommendations'].append("异常值比例较高，建议检查数据源")
            
            logger.info(f"异常值检测完成: 方法={method}, 总异常值={total_outliers}")
            return outlier_result
            
        except Exception as e:
            logger.error(f"异常值检测失败: {e}")
            outlier_result['error'] = str(e)
            return outlier_result
    
    def _detect_outliers_iqr(self, data: pd.Series) -> np.ndarray:
        """使用IQR方法检测异常值"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data < lower_bound) | (data > upper_bound)].index.values
    
    def _detect_outliers_zscore(self, data: pd.Series, threshold: float = 3) -> np.ndarray:
        """使用Z-score方法检测异常值"""
        z_scores = np.abs(stats.zscore(data))
        return data[z_scores > threshold].index.values
    
    def _detect_outliers_isolation_forest(self, data: pd.Series, contamination: float = 0.1) -> np.ndarray:
        """使用Isolation Forest方法检测异常值"""
        if len(data) < 10:  # 数据太少时不适用
            return np.array([])
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(data.values.reshape(-1, 1))
        return data[outliers == -1].index.values
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据DataFrame
            strategy: 处理策略（'auto', 'drop', 'forward_fill', 'backward_fill', 'interpolate', 'mean'）
            
        Returns:
            处理后的DataFrame
        """
        df_cleaned = df.copy()
        
        try:
            missing_info = df.isnull().sum()
            total_missing = missing_info.sum()
            
            if total_missing == 0:
                logger.info("数据无缺失值")
                return df_cleaned
            
            logger.info(f"发现 {total_missing} 个缺失值")
            
            if strategy == 'auto':
                # 自动选择策略
                missing_ratio = total_missing / (len(df) * len(df.columns))
                if missing_ratio > 0.5:
                    logger.warning("缺失值比例过高，建议检查数据源")
                    strategy = 'drop'
                elif missing_ratio > 0.1:
                    strategy = 'interpolate'
                else:
                    strategy = 'forward_fill'
                
                logger.info(f"自动选择缺失值处理策略: {strategy}")
            
            # 应用处理策略
            if strategy == 'drop':
                df_cleaned = df_cleaned.dropna()
            elif strategy == 'forward_fill':
                df_cleaned = df_cleaned.fillna(method='ffill')
            elif strategy == 'backward_fill':
                df_cleaned = df_cleaned.fillna(method='bfill')
            elif strategy == 'interpolate':
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].interpolate()
            elif strategy == 'mean':
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                    df_cleaned[numeric_columns].mean()
                )
            
            # 检查处理结果
            remaining_missing = df_cleaned.isnull().sum().sum()
            logger.info(f"缺失值处理完成: 剩余缺失值 {remaining_missing}")
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"缺失值处理失败: {e}")
            return df
    
    def clean_data(self, df: pd.DataFrame, remove_duplicates: bool = True,
                   handle_outliers: bool = True, outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 数据DataFrame
            remove_duplicates: 是否移除重复值
            handle_outliers: 是否处理异常值
            outlier_method: 异常值处理方法
            
        Returns:
            清洗后的DataFrame
        """
        df_cleaned = df.copy()
        
        try:
            original_shape = df_cleaned.shape
            
            # 移除重复值
            if remove_duplicates:
                duplicates_count = df_cleaned.duplicated().sum()
                if duplicates_count > 0:
                    df_cleaned = df_cleaned.drop_duplicates()
                    logger.info(f"移除 {duplicates_count} 行重复数据")
            
            # 处理异常值
            if handle_outliers:
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
                outlier_result = self.detect_outliers(df_cleaned, numeric_columns.tolist(), outlier_method)
                
                # 移除异常值（可选择其他处理方式）
                all_outlier_indices = set()
                for col_outliers in outlier_result['outliers'].values():
                    all_outlier_indices.update(col_outliers['indices'])
                
                if all_outlier_indices:
                    df_cleaned = df_cleaned.drop(index=list(all_outlier_indices))
                    logger.info(f"移除 {len(all_outlier_indices)} 行异常值")
            
            final_shape = df_cleaned.shape
            logger.info(f"数据清洗完成: {original_shape} -> {final_shape}")
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"数据清洗失败: {e}")
            return df
    
    def generate_quality_report(self, df: pd.DataFrame, data_type: str = 'general',
                              save_report: bool = True) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Args:
            df: 数据DataFrame
            data_type: 数据类型
            save_report: 是否保存报告
            
        Returns:
            质量报告字典
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'basic_info': {},
            'validation_result': {},
            'missing_values': {},
            'outliers': {},
            'recommendations': []
        }
        
        try:
            # 基本信息
            report['basic_info'] = {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'data_types': df.dtypes.to_dict(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # 数据验证
            report['validation_result'] = self.validate_data_format(df, data_type)
            
            # 缺失值分析
            missing_counts = df.isnull().sum()
            report['missing_values'] = {
                'total_missing': missing_counts.sum(),
                'missing_by_column': missing_counts.to_dict(),
                'missing_percentage': (missing_counts / len(df) * 100).to_dict()
            }
            
            # 异常值检测
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                report['outliers'] = self.detect_outliers(df, numeric_columns)
            
            # 生成建议
            if not report['validation_result']['is_valid']:
                report['recommendations'].append("数据格式验证失败，需要修复数据格式问题")
            
            if report['missing_values']['total_missing'] > 0:
                missing_ratio = report['missing_values']['total_missing'] / (df.shape[0] * df.shape[1])
                if missing_ratio > 0.1:
                    report['recommendations'].append("缺失值比例较高，建议进行缺失值处理")
            
            if 'total_outliers' in report.get('outliers', {}).get('summary', {}):
                if report['outliers']['summary']['total_outliers'] > 0:
                    report['recommendations'].append("发现异常值，建议进行异常值处理")
            
            # 保存报告
            if save_report:
                self._save_quality_report(report, data_type)
            
            logger.info(f"数据质量报告生成完成: {data_type}")
            return report
            
        except Exception as e:
            logger.error(f"生成数据质量报告失败: {e}")
            report['error'] = str(e)
            return report
    
    def _save_quality_report(self, report: Dict[str, Any], data_type: str):
        """保存质量报告到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{data_type}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"质量报告已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存质量报告失败: {e}")


# 创建全局实例
data_quality_manager = DataQualityManager()