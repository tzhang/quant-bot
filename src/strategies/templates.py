import pandas as pd
import numpy as np


class MeanReversionStrategy:
    """均值回归策略：基于价格相对于移动平均线的Z-score进行交易"""

    def __init__(self, lookback: int = 20, entry_z: float = -1.0, exit_z: float = 0.0):
        """
        初始化均值回归策略
        
        Args:
            lookback: 移动平均线和标准差的回望期
            entry_z: 进入信号的Z-score阈值（通常为负值）
            exit_z: 退出信号的Z-score阈值
        """
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号序列，0表示空仓，1表示满仓
        """
        sma = df["Close"].rolling(self.lookback).mean()
        std = df["Close"].rolling(self.lookback).std()
        z = (df["Close"] - sma) / (std + 1e-12)
        sig = pd.Series(0.0, index=df.index)
        sig = sig.where(z > self.entry_z, 1.0)
        sig = sig.where(z < self.exit_z, 0.0)
        return sig.clip(0.0, 1.0)


class MomentumStrategy:
    """动量策略：基于EMA交叉进行交易"""

    def __init__(self, fast: int = 12, slow: int = 26):
        """
        初始化动量策略
        
        Args:
            fast: 快速EMA周期
            slow: 慢速EMA周期
        """
        self.fast = fast
        self.slow = slow

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号序列，0表示空仓，1表示满仓
        """
        ema_fast = df["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.slow, adjust=False).mean()
        sig = (ema_fast > ema_slow).astype(float)
        return sig.clip(0.0, 1.0)


class RSIStrategy:
    """RSI策略：基于相对强弱指数的超买超卖策略"""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        初始化RSI策略
        
        Args:
            period: RSI计算周期
            oversold: 超卖阈值
            overbought: 超买阈值
        """
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / (loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def signal(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号序列，0表示空仓，1表示满仓
        """
        rsi = self._calculate_rsi(df["Close"])
        sig = pd.Series(0.0, index=df.index)
        # 当RSI低于超卖线时买入
        sig = sig.where(rsi > self.oversold, 1.0)
        # 当RSI高于超买线时卖出
        sig = sig.where(rsi < self.overbought, 0.0)
        return sig.clip(0.0, 1.0)


class BollingerBandsStrategy:
    """布林带策略：基于价格突破布林带进行交易"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, strategy_type: str = "mean_reversion"):
        """
        初始化布林带策略
        
        Args:
            period: 移动平均线周期
            std_dev: 标准差倍数
            strategy_type: 策略类型，"mean_reversion"（均值回归）或"breakout"（突破）
        """
        self.period = period
        self.std_dev = std_dev
        self.strategy_type = strategy_type
    
    def signal(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号序列，0表示空仓，1表示满仓
        """
        sma = df["Close"].rolling(self.period).mean()
        std = df["Close"].rolling(self.period).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        sig = pd.Series(0.0, index=df.index)
        
        if self.strategy_type == "mean_reversion":
            # 均值回归：价格触及下轨时买入，触及上轨时卖出
            sig = sig.where(df["Close"] > lower_band, 1.0)
            sig = sig.where(df["Close"] < upper_band, 0.0)
        else:  # breakout
            # 突破策略：价格突破上轨时买入，跌破下轨时卖出
            sig = sig.where(df["Close"] < upper_band, 1.0)
            sig = sig.where(df["Close"] > lower_band, 0.0)
        
        return sig.clip(0.0, 1.0)


class MACDStrategy:
    """MACD策略：基于MACD指标的金叉死叉进行交易"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        """
        初始化MACD策略
        
        Args:
            fast: 快速EMA周期
            slow: 慢速EMA周期
            signal_period: 信号线EMA周期
        """
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号序列，0表示空仓，1表示满仓
        """
        ema_fast = df["Close"].ewm(span=self.fast).mean()
        ema_slow = df["Close"].ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal_period).mean()  # 使用signal_period避免与方法名冲突
        
        # MACD金叉（MACD线上穿信号线）时买入
        sig = (macd > signal_line).astype(float)
        return sig.clip(0.0, 1.0)


class PairsTradingStrategy:
    """配对交易策略：基于两个相关资产的价差进行交易"""
    
    def __init__(self, lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5):
        """
        初始化配对交易策略
        
        Args:
            lookback: 价差统计的回望期
            entry_z: 进入交易的Z-score阈值
            exit_z: 退出交易的Z-score阈值
        """
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
    
    def signal(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
        """
        生成配对交易信号
        
        Args:
            df1: 第一个资产的OHLCV数据
            df2: 第二个资产的OHLCV数据
            
        Returns:
            交易信号序列，正值表示做多价差，负值表示做空价差，0表示空仓
        """
        # 计算价格比率（对数价差）
        ratio = np.log(df1["Close"] / df2["Close"])
        
        # 计算移动平均和标准差
        mean_ratio = ratio.rolling(self.lookback).mean()
        std_ratio = ratio.rolling(self.lookback).std()
        z_score = (ratio - mean_ratio) / (std_ratio + 1e-12)
        
        sig = pd.Series(0.0, index=df1.index)
        
        # 当Z-score超过阈值时进入交易
        sig = sig.where(abs(z_score) < self.entry_z, np.sign(-z_score))
        # 当Z-score回归到退出阈值时平仓
        sig = sig.where(abs(z_score) > self.exit_z, 0.0)
        
        return sig.clip(-1.0, 1.0)


class VolatilityBreakoutStrategy:
    """波动率突破策略：基于价格突破前期波动率范围进行交易"""
    
    def __init__(self, lookback: int = 20, multiplier: float = 1.5):
        """
        初始化波动率突破策略
        
        Args:
            lookback: 波动率计算的回望期
            multiplier: 波动率倍数
        """
        self.lookback = lookback
        self.multiplier = multiplier
    
    def signal(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号序列，0表示空仓，1表示满仓
        """
        # 计算真实波动率（ATR）
        high_low = df["High"] - df["Low"]
        high_close = abs(df["High"] - df["Close"].shift(1))
        low_close = abs(df["Low"] - df["Close"].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(self.lookback).mean()
        
        # 计算突破阈值
        prev_close = df["Close"].shift(1)
        upper_threshold = prev_close + (atr * self.multiplier)
        lower_threshold = prev_close - (atr * self.multiplier)
        
        sig = pd.Series(0.0, index=df.index)
        
        # 价格突破上阈值时买入
        sig = sig.where(df["Close"] < upper_threshold, 1.0)
        # 价格跌破下阈值时卖出
        sig = sig.where(df["Close"] > lower_threshold, 0.0)
        
        return sig.clip(0.0, 1.0)