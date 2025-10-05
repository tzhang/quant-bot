"""
技术指标计算工具

提供常用的技术分析指标计算功能
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple


class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """
        简单移动平均线
        
        Args:
            data: 价格数据
            window: 窗口期
            
        Returns:
            移动平均线数据
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """
        指数移动平均线
        
        Args:
            data: 价格数据
            window: 窗口期
            
        Returns:
            指数移动平均线数据
        """
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        相对强弱指标
        
        Args:
            data: 价格数据
            window: 窗口期
            
        Returns:
            RSI指标数据
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        布林带指标
        
        Args:
            data: 价格数据
            window: 窗口期
            num_std: 标准差倍数
            
        Returns:
            上轨、中轨、下轨
        """
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD指标
        
        Args:
            data: 价格数据
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            MACD线、信号线、柱状图
        """
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        随机指标
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            k_window: K线窗口期
            d_window: D线窗口期
            
        Returns:
            %K线、%D线
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        平均真实波幅
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            window: 窗口期
            
        Returns:
            ATR指标数据
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        威廉指标
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            window: 窗口期
            
        Returns:
            威廉指标数据
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr