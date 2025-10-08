"""
基础策略类
定义所有策略的基础接口和通用功能
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseStrategy(ABC):
    """策略基类，定义所有策略的通用接口"""
    
    def __init__(self, **kwargs):
        """
        初始化策略
        
        Args:
            **kwargs: 策略参数
        """
        self.params = kwargs
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号序列，0表示空仓，1表示满仓
        """
        pass
    
    def signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号的别名方法，兼容旧版本
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号序列，0表示空仓，1表示满仓
        """
        return self.generate_signal(data)
    
    def get_params(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.params.copy()
    
    def set_params(self, **params):
        """设置策略参数"""
        self.params.update(params)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据的有效性
        
        Args:
            data: 输入数据
            
        Returns:
            数据是否有效
        """
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        if data.empty:
            return False
            
        return True
    
    def __str__(self):
        return f"{self.name}({self.params})"
    
    def __repr__(self):
        return self.__str__()