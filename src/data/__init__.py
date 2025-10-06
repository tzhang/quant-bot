"""
数据管理模块

提供数据获取、缓存和管理功能
"""

from .data_manager import DataManager

# 占位符类，用于满足导入需求
class FundamentalDataManager:
    """基本面数据管理器占位符"""
    pass

class DataQualityManager:
    """数据质量管理器占位符"""
    pass

class MacroDataManager:
    """宏观数据管理器占位符"""
    pass

class SectorDataManager:
    """行业数据管理器占位符"""
    pass

class SentimentDataManager:
    """情绪数据管理器占位符"""
    pass

class AlternativeDataManager:
    """另类数据管理器占位符"""
    pass

__all__ = [
    'DataManager',
    'FundamentalDataManager',
    'DataQualityManager',
    'MacroDataManager',
    'SectorDataManager',
    'SentimentDataManager',
    'AlternativeDataManager'
]