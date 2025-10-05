"""
择时因子计算模块

实现各种择时相关的因子，包括：
1. 市场情绪因子
2. 宏观经济因子  
3. 技术择时因子
4. 波动率择时因子
5. 资金流向因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimingFactorCalculator:
    """
    择时因子计算器
    
    提供各种择时因子的计算功能，用于判断市场进入和退出的时机
    """
    
    def __init__(self):
        """初始化择时因子计算器"""
        self.logger = logging.getLogger(__name__)
    
    def calculate_market_sentiment_factors(self, market_data: pd.DataFrame, 
                                         vix_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        计算市场情绪因子
        
        Args:
            market_data: 市场指数数据，包含OHLCV
            vix_data: VIX恐慌指数数据（可选）
            
        Returns:
            Dict[str, pd.Series]: 市场情绪因子字典
        """
        factors = {}
        
        try:
            # 1. 涨跌比例因子
            factors['advance_decline_ratio'] = self._calculate_advance_decline_ratio(market_data)
            
            # 2. 新高新低比例
            factors['new_high_low_ratio'] = self._calculate_new_high_low_ratio(market_data)
            
            # 3. 成交量价格趋势 (VPT)
            factors['volume_price_trend'] = self._calculate_vpt(market_data)
            
            # 4. 资金流量指标 (MFI)
            factors['money_flow_index'] = self._calculate_mfi(market_data)
            
            # 5. 恐慌贪婪指数（基于VIX）
            if vix_data is not None:
                factors['fear_greed_index'] = self._calculate_fear_greed_index(vix_data)
            
            # 6. 市场强度指标
            factors['market_strength'] = self._calculate_market_strength(market_data)
            
            self.logger.info(f"成功计算 {len(factors)} 个市场情绪因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算市场情绪因子时出错: {str(e)}")
            return {}
    
    def calculate_technical_timing_factors(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算技术择时因子
        
        Args:
            price_data: 价格数据，包含OHLCV
            
        Returns:
            Dict[str, pd.Series]: 技术择时因子字典
        """
        factors = {}
        
        try:
            # 1. 趋势强度因子
            factors['trend_strength'] = self._calculate_trend_strength(price_data)
            
            # 2. 动量择时因子
            factors['momentum_timing'] = self._calculate_momentum_timing(price_data)
            
            # 3. 均线择时信号
            factors['ma_timing_signal'] = self._calculate_ma_timing_signal(price_data)
            
            # 4. 突破择时因子
            factors['breakout_timing'] = self._calculate_breakout_timing(price_data)
            
            # 5. 支撑阻力择时
            factors['support_resistance_timing'] = self._calculate_support_resistance_timing(price_data)
            
            # 6. 波段择时因子
            factors['swing_timing'] = self._calculate_swing_timing(price_data)
            
            self.logger.info(f"成功计算 {len(factors)} 个技术择时因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算技术择时因子时出错: {str(e)}")
            return {}
    
    def calculate_volatility_timing_factors(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算波动率择时因子
        
        Args:
            price_data: 价格数据
            
        Returns:
            Dict[str, pd.Series]: 波动率择时因子字典
        """
        factors = {}
        
        try:
            # 1. 波动率均值回归因子
            factors['volatility_mean_reversion'] = self._calculate_vol_mean_reversion(price_data)
            
            # 2. 波动率突破因子
            factors['volatility_breakout'] = self._calculate_vol_breakout(price_data)
            
            # 3. 波动率聚类因子
            factors['volatility_clustering'] = self._calculate_vol_clustering(price_data)
            
            # 4. 波动率择时信号
            factors['volatility_timing_signal'] = self._calculate_vol_timing_signal(price_data)
            
            self.logger.info(f"成功计算 {len(factors)} 个波动率择时因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算波动率择时因子时出错: {str(e)}")
            return {}
    
    def calculate_flow_timing_factors(self, price_data: pd.DataFrame, 
                                    volume_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        计算资金流向择时因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据（可选）
            
        Returns:
            Dict[str, pd.Series]: 资金流向择时因子字典
        """
        factors = {}
        
        try:
            # 1. 资金净流入因子
            factors['net_money_flow'] = self._calculate_net_money_flow(price_data)
            
            # 2. 大单净流入比例
            factors['large_order_flow_ratio'] = self._calculate_large_order_flow(price_data)
            
            # 3. 主力资金择时
            factors['institutional_flow_timing'] = self._calculate_institutional_flow_timing(price_data)
            
            # 4. 散户情绪择时
            factors['retail_sentiment_timing'] = self._calculate_retail_sentiment_timing(price_data)
            
            self.logger.info(f"成功计算 {len(factors)} 个资金流向择时因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算资金流向择时因子时出错: {str(e)}")
            return {}
    
    def calculate_all_timing_factors(self, market_data: pd.DataFrame,
                                   vix_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        计算所有择时因子
        
        Args:
            market_data: 市场数据
            vix_data: VIX数据（可选）
            
        Returns:
            Dict[str, pd.Series]: 所有择时因子字典
        """
        all_factors = {}
        
        # 计算各类择时因子
        sentiment_factors = self.calculate_market_sentiment_factors(market_data, vix_data)
        technical_factors = self.calculate_technical_timing_factors(market_data)
        volatility_factors = self.calculate_volatility_timing_factors(market_data)
        flow_factors = self.calculate_flow_timing_factors(market_data)
        
        # 合并所有因子
        all_factors.update(sentiment_factors)
        all_factors.update(technical_factors)
        all_factors.update(volatility_factors)
        all_factors.update(flow_factors)
        
        self.logger.info(f"总共计算了 {len(all_factors)} 个择时因子")
        return all_factors
    
    # ==================== 私有方法：具体因子计算实现 ====================
    
    def _calculate_advance_decline_ratio(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算涨跌比例因子"""
        returns = data['close'].pct_change()
        advances = (returns > 0).rolling(window=window).sum()
        declines = (returns < 0).rolling(window=window).sum()
        return advances / (declines + 1e-8)  # 避免除零
    
    def _calculate_new_high_low_ratio(self, data: pd.DataFrame, window: int = 252) -> pd.Series:
        """计算新高新低比例"""
        high_252 = data['high'].rolling(window=window).max()
        low_252 = data['low'].rolling(window=window).min()
        
        new_highs = (data['high'] >= high_252).astype(int)
        new_lows = (data['low'] <= low_252).astype(int)
        
        return new_highs.rolling(window=20).sum() / (new_lows.rolling(window=20).sum() + 1e-8)
    
    def _calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量价格趋势指标"""
        price_change = data['close'].pct_change()
        volume_weighted_change = price_change * data['volume']
        return volume_weighted_change.cumsum()
    
    def _calculate_mfi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算资金流量指标"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))
        return mfi
    
    def _calculate_fear_greed_index(self, vix_data: pd.DataFrame) -> pd.Series:
        """计算恐慌贪婪指数（基于VIX）"""
        vix = vix_data['close'] if 'close' in vix_data.columns else vix_data.iloc[:, 0]
        
        # VIX标准化到0-100区间，VIX越高越恐慌
        vix_normalized = (vix - vix.rolling(window=252).min()) / (
            vix.rolling(window=252).max() - vix.rolling(window=252).min() + 1e-8
        )
        
        # 转换为贪婪指数（VIX高时贪婪指数低）
        fear_greed = (1 - vix_normalized) * 100
        return fear_greed
    
    def _calculate_market_strength(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算市场强度指标"""
        returns = data['close'].pct_change()
        positive_returns = returns.where(returns > 0, 0)
        negative_returns = returns.where(returns < 0, 0).abs()
        
        avg_positive = positive_returns.rolling(window=window).mean()
        avg_negative = negative_returns.rolling(window=window).mean()
        
        return avg_positive / (avg_negative + 1e-8)
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """计算趋势强度因子"""
        # 使用ADX思想计算趋势强度
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算真实波幅
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算方向性移动
        dm_plus = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
        dm_minus = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
        
        # 计算平滑后的值
        atr = tr.rolling(window=14).mean()
        di_plus = (dm_plus.rolling(window=14).mean() / atr) * 100
        di_minus = (dm_minus.rolling(window=14).mean() / atr) * 100
        
        # 计算ADX
        dx = ((di_plus - di_minus).abs() / (di_plus + di_minus + 1e-8)) * 100
        adx = dx.rolling(window=14).mean()
        
        return adx
    
    def _calculate_momentum_timing(self, data: pd.DataFrame) -> pd.Series:
        """计算动量择时因子"""
        # 多周期动量组合
        mom_5 = data['close'].pct_change(5)
        mom_10 = data['close'].pct_change(10)
        mom_20 = data['close'].pct_change(20)
        
        # 加权组合
        momentum_timing = 0.5 * mom_5 + 0.3 * mom_10 + 0.2 * mom_20
        return momentum_timing
    
    def _calculate_ma_timing_signal(self, data: pd.DataFrame) -> pd.Series:
        """计算均线择时信号"""
        # 多条均线的排列
        ma_5 = data['close'].rolling(window=5).mean()
        ma_10 = data['close'].rolling(window=10).mean()
        ma_20 = data['close'].rolling(window=20).mean()
        ma_60 = data['close'].rolling(window=60).mean()
        
        # 均线多头排列得分
        score = 0
        score += (ma_5 > ma_10).astype(int)
        score += (ma_10 > ma_20).astype(int)
        score += (ma_20 > ma_60).astype(int)
        score += (data['close'] > ma_5).astype(int)
        
        return score / 4  # 标准化到0-1
    
    def _calculate_breakout_timing(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算突破择时因子"""
        # 布林带突破
        ma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        
        # 突破信号
        breakout_up = (data['close'] > upper_band).astype(int)
        breakout_down = (data['close'] < lower_band).astype(int) * -1
        
        return breakout_up + breakout_down
    
    def _calculate_support_resistance_timing(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算支撑阻力择时因子"""
        # 简化的支撑阻力计算
        high_resistance = data['high'].rolling(window=window).max()
        low_support = data['low'].rolling(window=window).min()
        
        # 价格在支撑阻力区间的位置
        price_position = (data['close'] - low_support) / (high_resistance - low_support + 1e-8)
        
        return price_position
    
    def _calculate_swing_timing(self, data: pd.DataFrame) -> pd.Series:
        """计算波段择时因子"""
        # 基于RSI的波段择时
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # RSI择时信号
        swing_signal = np.where(rsi < 30, 1,  # 超卖买入
                               np.where(rsi > 70, -1, 0))  # 超买卖出
        
        return pd.Series(swing_signal, index=data.index)
    
    def _calculate_vol_mean_reversion(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算波动率均值回归因子"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        vol_ma = volatility.rolling(window=60).mean()
        
        # 波动率偏离均值的程度
        vol_deviation = (volatility - vol_ma) / (vol_ma + 1e-8)
        return -vol_deviation  # 负号表示均值回归
    
    def _calculate_vol_breakout(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算波动率突破因子"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        vol_threshold = volatility.rolling(window=60).quantile(0.8)
        
        # 波动率突破信号
        vol_breakout = (volatility > vol_threshold).astype(int)
        return vol_breakout
    
    def _calculate_vol_clustering(self, data: pd.DataFrame, window: int = 5) -> pd.Series:
        """计算波动率聚类因子"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        # 波动率的自相关性
        vol_autocorr = volatility.rolling(window=20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        
        return vol_autocorr
    
    def _calculate_vol_timing_signal(self, data: pd.DataFrame) -> pd.Series:
        """计算波动率择时信号"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # 波动率分位数
        vol_percentile = volatility.rolling(window=252).rank(pct=True)
        
        # 低波动率买入，高波动率卖出
        timing_signal = np.where(vol_percentile < 0.2, 1,
                               np.where(vol_percentile > 0.8, -1, 0))
        
        return pd.Series(timing_signal, index=data.index)
    
    def _calculate_net_money_flow(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算资金净流入因子"""
        # 简化的资金流计算
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        # 上涨日和下跌日的资金流
        up_flow = money_flow.where(data['close'] > data['close'].shift(1), 0)
        down_flow = money_flow.where(data['close'] < data['close'].shift(1), 0)
        
        # 净流入
        net_flow = (up_flow - down_flow).rolling(window=window).sum()
        return net_flow
    
    def _calculate_large_order_flow(self, data: pd.DataFrame) -> pd.Series:
        """计算大单净流入比例（简化版）"""
        # 基于成交量异常来模拟大单
        volume_ma = data['volume'].rolling(window=20).mean()
        large_volume = data['volume'] > volume_ma * 1.5
        
        # 大单方向
        price_change = data['close'].pct_change()
        large_buy = large_volume & (price_change > 0)
        large_sell = large_volume & (price_change < 0)
        
        # 大单净流入比例
        net_large_flow = (large_buy.astype(int) - large_sell.astype(int)).rolling(window=10).sum()
        return net_large_flow
    
    def _calculate_institutional_flow_timing(self, data: pd.DataFrame) -> pd.Series:
        """计算主力资金择时因子（简化版）"""
        # 基于价量关系判断主力行为
        returns = data['close'].pct_change()
        volume_change = data['volume'].pct_change()
        
        # 价涨量增为主力买入，价跌量增为主力卖出
        institutional_buy = (returns > 0) & (volume_change > 0)
        institutional_sell = (returns < 0) & (volume_change > 0)
        
        # 主力净买入强度
        net_institutional = (institutional_buy.astype(int) - institutional_sell.astype(int)).rolling(window=10).sum()
        return net_institutional
    
    def _calculate_retail_sentiment_timing(self, data: pd.DataFrame) -> pd.Series:
        """计算散户情绪择时因子（简化版）"""
        # 基于小额交易的价量关系
        returns = data['close'].pct_change()
        volume_ma = data['volume'].rolling(window=20).mean()
        
        # 小成交量时的价格变化代表散户情绪
        retail_volume = data['volume'] < volume_ma * 0.8
        retail_sentiment = returns.where(retail_volume, 0).rolling(window=10).sum()
        
        return retail_sentiment