"""
技术因子计算模块

实现各种技术分析因子的计算，包括趋势、动量、波动率、成交量等因子
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
from scipy import stats
import talib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalFactorCalculator:
    """
    技术因子计算器
    
    提供各种技术分析因子的计算功能，包括：
    - 趋势因子：MA、EMA、MACD、ADX等
    - 动量因子：RSI、STOCH、Williams %R等
    - 波动率因子：ATR、Bollinger Bands、VIX等
    - 成交量因子：OBV、VWAP、CMF等
    - 价格形态因子：支撑阻力、缺口、K线形态等
    """
    
    def __init__(self):
        """初始化技术因子计算器"""
        self.logger = logger
        
    def calculate_trend_factors(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        计算趋势因子
        
        Args:
            price_data: 价格数据，包含open, high, low, close列
            volume_data: 成交量数据（可选）
            
        Returns:
            Dict[str, float]: 趋势因子字典
        """
        try:
            factors = {}
            
            if price_data.empty:
                return factors
                
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            
            # 移动平均线相关
            # MA5, MA10, MA20, MA60
            for period in [5, 10, 20, 60]:
                if len(close) >= period:
                    ma = talib.SMA(close, timeperiod=period)
                    factors[f'ma_{period}'] = ma[-1] if not np.isnan(ma[-1]) else 0
                    
                    # 价格相对MA的位置
                    if ma[-1] > 0:
                        factors[f'price_to_ma_{period}'] = (close[-1] / ma[-1] - 1) * 100
                    else:
                        factors[f'price_to_ma_{period}'] = 0
            
            # 指数移动平均线
            for period in [12, 26]:
                if len(close) >= period:
                    ema = talib.EMA(close, timeperiod=period)
                    factors[f'ema_{period}'] = ema[-1] if not np.isnan(ema[-1]) else 0
            
            # MACD
            if len(close) >= 26:
                macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                factors['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
                factors['macd_signal'] = macdsignal[-1] if not np.isnan(macdsignal[-1]) else 0
                factors['macd_hist'] = macdhist[-1] if not np.isnan(macdhist[-1]) else 0
                
                # MACD金叉死叉信号
                if len(macd) >= 2 and len(macdsignal) >= 2:
                    if macd[-2] <= macdsignal[-2] and macd[-1] > macdsignal[-1]:
                        factors['macd_golden_cross'] = 1
                    elif macd[-2] >= macdsignal[-2] and macd[-1] < macdsignal[-1]:
                        factors['macd_death_cross'] = 1
                    else:
                        factors['macd_golden_cross'] = 0
                        factors['macd_death_cross'] = 0
            
            # ADX (平均趋向指数)
            if len(close) >= 14:
                adx = talib.ADX(high, low, close, timeperiod=14)
                factors['adx'] = adx[-1] if not np.isnan(adx[-1]) else 0
                
                # +DI和-DI
                plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
                minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
                factors['plus_di'] = plus_di[-1] if not np.isnan(plus_di[-1]) else 0
                factors['minus_di'] = minus_di[-1] if not np.isnan(minus_di[-1]) else 0
            
            # 抛物线SAR
            if len(close) >= 2:
                sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
                factors['sar'] = sar[-1] if not np.isnan(sar[-1]) else 0
                factors['sar_signal'] = 1 if close[-1] > sar[-1] else -1
            
            # 趋势强度
            if len(close) >= 20:
                # 计算20日趋势强度
                slope, _, r_value, _, _ = stats.linregress(range(20), close[-20:])
                factors['trend_strength_20'] = slope
                factors['trend_r_squared_20'] = r_value ** 2
            
            # 价格通道
            if len(close) >= 20:
                highest_20 = np.max(high[-20:])
                lowest_20 = np.min(low[-20:])
                factors['price_channel_position'] = (close[-1] - lowest_20) / (highest_20 - lowest_20) * 100 if highest_20 > lowest_20 else 50
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算趋势因子时出错: {e}")
            return {}

    def calculate_momentum_factors(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算动量因子
        
        Args:
            price_data: 价格数据
            
        Returns:
            Dict[str, float]: 动量因子字典
        """
        try:
            factors = {}
            
            if price_data.empty:
                return factors
                
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            
            # RSI (相对强弱指数)
            for period in [6, 14, 24]:
                if len(close) >= period:
                    rsi = talib.RSI(close, timeperiod=period)
                    factors[f'rsi_{period}'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
            
            # 随机指标 (KDJ)
            if len(close) >= 14:
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
                factors['stoch_k'] = slowk[-1] if not np.isnan(slowk[-1]) else 50
                factors['stoch_d'] = slowd[-1] if not np.isnan(slowd[-1]) else 50
                
                # J值计算
                if not np.isnan(slowk[-1]) and not np.isnan(slowd[-1]):
                    factors['stoch_j'] = 3 * slowk[-1] - 2 * slowd[-1]
            
            # Williams %R
            if len(close) >= 14:
                willr = talib.WILLR(high, low, close, timeperiod=14)
                factors['williams_r'] = willr[-1] if not np.isnan(willr[-1]) else -50
            
            # CCI (商品通道指数)
            if len(close) >= 14:
                cci = talib.CCI(high, low, close, timeperiod=14)
                factors['cci'] = cci[-1] if not np.isnan(cci[-1]) else 0
            
            # 动量指标 (Momentum)
            for period in [10, 20]:
                if len(close) >= period:
                    mom = talib.MOM(close, timeperiod=period)
                    factors[f'momentum_{period}'] = mom[-1] if not np.isnan(mom[-1]) else 0
            
            # ROC (变动率指标)
            for period in [10, 20]:
                if len(close) >= period:
                    roc = talib.ROC(close, timeperiod=period)
                    factors[f'roc_{period}'] = roc[-1] if not np.isnan(roc[-1]) else 0
            
            # TRIX (三重指数平滑移动平均)
            if len(close) >= 30:
                trix = talib.TRIX(close, timeperiod=14)
                factors['trix'] = trix[-1] if not np.isnan(trix[-1]) else 0
            
            # 价格动量
            for period in [1, 5, 10, 20]:
                if len(close) > period:
                    factors[f'price_momentum_{period}'] = (close[-1] / close[-(period+1)] - 1) * 100
            
            # 相对动量
            if len(close) >= 20:
                # 计算相对于20日均线的动量
                ma20 = talib.SMA(close, timeperiod=20)
                if not np.isnan(ma20[-1]) and ma20[-1] > 0:
                    factors['relative_momentum'] = (close[-1] / ma20[-1] - 1) * 100
                else:
                    factors['relative_momentum'] = 0
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算动量因子时出错: {e}")
            return {}

    def calculate_volatility_factors(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算波动率因子
        
        Args:
            price_data: 价格数据
            
        Returns:
            Dict[str, float]: 波动率因子字典
        """
        try:
            factors = {}
            
            if price_data.empty:
                return factors
                
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            
            # ATR (真实波动幅度)
            for period in [14, 20]:
                if len(close) >= period:
                    atr = talib.ATR(high, low, close, timeperiod=period)
                    factors[f'atr_{period}'] = atr[-1] if not np.isnan(atr[-1]) else 0
                    
                    # ATR相对值
                    if close[-1] > 0:
                        factors[f'atr_{period}_pct'] = (atr[-1] / close[-1]) * 100
                    else:
                        factors[f'atr_{period}_pct'] = 0
            
            # 布林带
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                factors['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else 0
                factors['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else 0
                factors['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else 0
                
                # 布林带位置
                if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]) and bb_upper[-1] > bb_lower[-1]:
                    factors['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100
                    factors['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100 if bb_middle[-1] > 0 else 0
                else:
                    factors['bb_position'] = 50
                    factors['bb_width'] = 0
            
            # 历史波动率
            for period in [10, 20, 60]:
                if len(close) > period:
                    returns = np.diff(np.log(close[-period-1:]))
                    factors[f'volatility_{period}'] = np.std(returns) * np.sqrt(252) * 100  # 年化波动率
            
            # 价格振幅
            for period in [5, 10, 20]:
                if len(high) >= period and len(low) >= period:
                    high_max = np.max(high[-period:])
                    low_min = np.min(low[-period:])
                    if low_min > 0:
                        factors[f'price_range_{period}'] = (high_max - low_min) / low_min * 100
                    else:
                        factors[f'price_range_{period}'] = 0
            
            # 日内振幅
            if len(high) > 0 and len(low) > 0 and len(close) > 0:
                factors['intraday_range'] = (high[-1] - low[-1]) / close[-1] * 100 if close[-1] > 0 else 0
            
            # Keltner通道
            if len(close) >= 20:
                kc_middle = talib.EMA(close, timeperiod=20)
                atr_20 = talib.ATR(high, low, close, timeperiod=20)
                if not np.isnan(kc_middle[-1]) and not np.isnan(atr_20[-1]):
                    kc_upper = kc_middle[-1] + 2 * atr_20[-1]
                    kc_lower = kc_middle[-1] - 2 * atr_20[-1]
                    factors['kc_position'] = (close[-1] - kc_lower) / (kc_upper - kc_lower) * 100 if kc_upper > kc_lower else 50
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算波动率因子时出错: {e}")
            return {}

    def calculate_volume_factors(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算成交量因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            
        Returns:
            Dict[str, float]: 成交量因子字典
        """
        try:
            factors = {}
            
            if price_data.empty or volume_data.empty:
                return factors
                
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            volume = volume_data['volume'].values if 'volume' in volume_data.columns else volume_data.values
            
            # OBV (能量潮)
            if len(close) >= 2 and len(volume) >= 2:
                obv = talib.OBV(close, volume)
                factors['obv'] = obv[-1] if not np.isnan(obv[-1]) else 0
                
                # OBV变化率
                if len(obv) >= 10:
                    factors['obv_change_10'] = (obv[-1] - obv[-10]) / abs(obv[-10]) * 100 if obv[-10] != 0 else 0
            
            # 成交量移动平均
            for period in [5, 10, 20]:
                if len(volume) >= period:
                    vol_ma = talib.SMA(volume.astype(float), timeperiod=period)
                    factors[f'volume_ma_{period}'] = vol_ma[-1] if not np.isnan(vol_ma[-1]) else 0
                    
                    # 成交量相对比率
                    if vol_ma[-1] > 0:
                        factors[f'volume_ratio_{period}'] = volume[-1] / vol_ma[-1]
                    else:
                        factors[f'volume_ratio_{period}'] = 1
            
            # VWAP (成交量加权平均价)
            if len(close) >= 20 and len(volume) >= 20:
                typical_price = (high + low + close) / 3
                vwap = np.sum(typical_price[-20:] * volume[-20:]) / np.sum(volume[-20:])
                factors['vwap_20'] = vwap if not np.isnan(vwap) else 0
                
                # 价格相对VWAP位置
                if vwap > 0:
                    factors['price_to_vwap'] = (close[-1] / vwap - 1) * 100
                else:
                    factors['price_to_vwap'] = 0
            
            # CMF (资金流量指标)
            if len(close) >= 20:
                cmf_values = []
                for i in range(len(close)):
                    if high[i] != low[i]:
                        mfm = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                        cmf_values.append(mfm * volume[i])
                    else:
                        cmf_values.append(0)
                
                if len(cmf_values) >= 20:
                    cmf = sum(cmf_values[-20:]) / sum(volume[-20:]) if sum(volume[-20:]) > 0 else 0
                    factors['cmf_20'] = cmf
            
            # A/D Line (累积/派发线)
            if len(close) >= 2:
                ad = talib.AD(high, low, close, volume.astype(float))
                factors['ad_line'] = ad[-1] if not np.isnan(ad[-1]) else 0
            
            # 成交量价格趋势 (VPT)
            if len(close) >= 2 and len(volume) >= 2:
                vpt = 0
                for i in range(1, len(close)):
                    if close[i-1] != 0:
                        vpt += volume[i] * (close[i] - close[i-1]) / close[i-1]
                factors['vpt'] = vpt
            
            # 成交量震荡器
            for short_period, long_period in [(5, 10), (10, 20)]:
                if len(volume) >= long_period:
                    vol_short = talib.SMA(volume.astype(float), timeperiod=short_period)
                    vol_long = talib.SMA(volume.astype(float), timeperiod=long_period)
                    if not np.isnan(vol_short[-1]) and not np.isnan(vol_long[-1]) and vol_long[-1] > 0:
                        factors[f'volume_oscillator_{short_period}_{long_period}'] = (vol_short[-1] - vol_long[-1]) / vol_long[-1] * 100
                    else:
                        factors[f'volume_oscillator_{short_period}_{long_period}'] = 0
            
            # 成交量相对强度
            if len(volume) >= 20:
                avg_volume = np.mean(volume[-20:])
                factors['volume_strength'] = volume[-1] / avg_volume if avg_volume > 0 else 1
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算成交量因子时出错: {e}")
            return {}

    def calculate_pattern_factors(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算价格形态因子
        
        Args:
            price_data: 价格数据
            
        Returns:
            Dict[str, float]: 形态因子字典
        """
        try:
            factors = {}
            
            if price_data.empty:
                return factors
                
            open_price = price_data['open'].values
            high = price_data['high'].values
            low = price_data['low'].values
            close = price_data['close'].values
            
            # K线形态识别
            if len(close) >= 3:
                # 锤子线
                hammer = talib.CDLHAMMER(open_price, high, low, close)
                factors['hammer'] = 1 if hammer[-1] > 0 else 0
                
                # 上吊线
                hanging_man = talib.CDLHANGINGMAN(open_price, high, low, close)
                factors['hanging_man'] = 1 if hanging_man[-1] > 0 else 0
                
                # 吞没形态
                engulfing = talib.CDLENGULFING(open_price, high, low, close)
                factors['engulfing'] = 1 if engulfing[-1] != 0 else 0
                
                # 十字星
                doji = talib.CDLDOJI(open_price, high, low, close)
                factors['doji'] = 1 if doji[-1] > 0 else 0
                
                # 流星
                shooting_star = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
                factors['shooting_star'] = 1 if shooting_star[-1] > 0 else 0
            
            # 支撑阻力位
            if len(high) >= 20 and len(low) >= 20:
                # 近期高点和低点
                recent_high = np.max(high[-20:])
                recent_low = np.min(low[-20:])
                
                # 距离支撑阻力位的距离
                factors['distance_to_resistance'] = (recent_high - close[-1]) / close[-1] * 100 if close[-1] > 0 else 0
                factors['distance_to_support'] = (close[-1] - recent_low) / close[-1] * 100 if close[-1] > 0 else 0
            
            # 缺口分析
            if len(open_price) >= 2 and len(close) >= 2:
                # 向上缺口
                if open_price[-1] > close[-2]:
                    factors['gap_up'] = (open_price[-1] - close[-2]) / close[-2] * 100
                    factors['gap_down'] = 0
                # 向下缺口
                elif open_price[-1] < close[-2]:
                    factors['gap_down'] = (close[-2] - open_price[-1]) / close[-2] * 100
                    factors['gap_up'] = 0
                else:
                    factors['gap_up'] = 0
                    factors['gap_down'] = 0
            
            # 价格位置
            if len(high) >= 52 and len(low) >= 52:
                # 52周高低点位置
                high_52w = np.max(high[-252:]) if len(high) >= 252 else np.max(high)
                low_52w = np.min(low[-252:]) if len(low) >= 252 else np.min(low)
                
                if high_52w > low_52w:
                    factors['position_52w'] = (close[-1] - low_52w) / (high_52w - low_52w) * 100
                else:
                    factors['position_52w'] = 50
                
                # 距离52周高低点的距离
                factors['distance_to_52w_high'] = (high_52w - close[-1]) / close[-1] * 100 if close[-1] > 0 else 0
                factors['distance_to_52w_low'] = (close[-1] - low_52w) / close[-1] * 100 if close[-1] > 0 else 0
            
            # 价格通道突破
            if len(high) >= 20 and len(low) >= 20:
                channel_high = np.max(high[-20:-1])  # 排除当前K线
                channel_low = np.min(low[-20:-1])
                
                # 突破信号
                factors['breakout_up'] = 1 if high[-1] > channel_high else 0
                factors['breakout_down'] = 1 if low[-1] < channel_low else 0
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算形态因子时出错: {e}")
            return {}

    def calculate_cycle_factors(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算周期因子
        
        Args:
            price_data: 价格数据
            
        Returns:
            Dict[str, float]: 周期因子字典
        """
        try:
            factors = {}
            
            if price_data.empty:
                return factors
                
            close = price_data['close'].values
            
            # 希尔伯特变换 - 主导周期
            if len(close) >= 32:
                ht_dcperiod = talib.HT_DCPERIOD(close)
                factors['dominant_cycle'] = ht_dcperiod[-1] if not np.isnan(ht_dcperiod[-1]) else 0
            
            # 希尔伯特变换 - 趋势vs周期模式
            if len(close) >= 32:
                ht_trendmode = talib.HT_TRENDMODE(close)
                factors['trend_mode'] = ht_trendmode[-1] if not np.isnan(ht_trendmode[-1]) else 0
            
            # 希尔伯特变换 - 正弦波
            if len(close) >= 32:
                ht_sine, ht_leadsine = talib.HT_SINE(close)
                factors['ht_sine'] = ht_sine[-1] if not np.isnan(ht_sine[-1]) else 0
                factors['ht_leadsine'] = ht_leadsine[-1] if not np.isnan(ht_leadsine[-1]) else 0
            
            # 周期性强度分析
            if len(close) >= 60:
                # 使用FFT分析主要周期
                from scipy.fft import fft, fftfreq
                
                # 去趋势
                detrended = close - np.mean(close)
                
                # FFT分析
                fft_values = fft(detrended[-60:])
                freqs = fftfreq(60)
                
                # 找到主要频率
                power_spectrum = np.abs(fft_values)
                dominant_freq_idx = np.argmax(power_spectrum[1:30]) + 1  # 排除DC分量
                
                if freqs[dominant_freq_idx] != 0:
                    factors['dominant_period_fft'] = 1 / abs(freqs[dominant_freq_idx])
                else:
                    factors['dominant_period_fft'] = 0
                
                # 周期性强度
                factors['cyclical_strength'] = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算周期因子时出错: {e}")
            return {}

    def calculate_all_factors(self, 
                            symbol: str,
                            price_data: pd.DataFrame,
                            volume_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        计算所有技术因子
        
        Args:
            symbol: 股票代码
            price_data: 价格数据
            volume_data: 成交量数据（可选）
            
        Returns:
            Dict[str, Dict[str, float]]: 所有因子的字典
        """
        try:
            all_factors = {
                'trend': self.calculate_trend_factors(price_data, volume_data),
                'momentum': self.calculate_momentum_factors(price_data),
                'volatility': self.calculate_volatility_factors(price_data),
                'pattern': self.calculate_pattern_factors(price_data),
                'cycle': self.calculate_cycle_factors(price_data)
            }
            
            # 如果有成交量数据，计算成交量因子
            if volume_data is not None and not volume_data.empty:
                all_factors['volume'] = self.calculate_volume_factors(price_data, volume_data)
            
            return all_factors
            
        except Exception as e:
            self.logger.error(f"计算所有技术因子时出错: {e}")
            return {}

    def calculate_factor_scores(self, factors: Dict[str, float]) -> Dict[str, float]:
        """
        计算技术因子评分
        
        Args:
            factors: 因子值字典
            
        Returns:
            Dict[str, float]: 因子评分字典
        """
        try:
            scores = {}
            
            # RSI评分 (30-70为正常区间)
            for period in [6, 14, 24]:
                rsi_key = f'rsi_{period}'
                if rsi_key in factors:
                    rsi_value = factors[rsi_key]
                    if rsi_value <= 30:
                        scores[f'{rsi_key}_score'] = 100  # 超卖，看涨
                    elif rsi_value >= 70:
                        scores[f'{rsi_key}_score'] = 0   # 超买，看跌
                    else:
                        scores[f'{rsi_key}_score'] = 50  # 中性
            
            # MACD评分
            if 'macd' in factors and 'macd_signal' in factors:
                if factors['macd'] > factors['macd_signal']:
                    scores['macd_score'] = 100
                else:
                    scores['macd_score'] = 0
            
            # 布林带位置评分
            if 'bb_position' in factors:
                bb_pos = factors['bb_position']
                if bb_pos <= 20:
                    scores['bb_position_score'] = 100  # 接近下轨，看涨
                elif bb_pos >= 80:
                    scores['bb_position_score'] = 0   # 接近上轨，看跌
                else:
                    scores['bb_position_score'] = 50  # 中性
            
            # 趋势评分
            if 'adx' in factors:
                adx_value = factors['adx']
                if adx_value >= 25:
                    scores['trend_strength_score'] = 100  # 强趋势
                elif adx_value >= 20:
                    scores['trend_strength_score'] = 75   # 中等趋势
                else:
                    scores['trend_strength_score'] = 25   # 弱趋势
            
            # 成交量评分
            if 'volume_ratio_20' in factors:
                vol_ratio = factors['volume_ratio_20']
                if vol_ratio >= 2.0:
                    scores['volume_score'] = 100  # 放量
                elif vol_ratio >= 1.5:
                    scores['volume_score'] = 75   # 温和放量
                elif vol_ratio >= 0.5:
                    scores['volume_score'] = 50   # 正常
                else:
                    scores['volume_score'] = 25   # 缩量
            
            return scores
            
        except Exception as e:
            self.logger.error(f"计算技术因子评分时出错: {e}")
            return {}

    def generate_factor_report(self, symbol: str, all_factors: Dict[str, Dict[str, float]]) -> str:
        """
        生成技术因子分析报告
        
        Args:
            symbol: 股票代码
            all_factors: 所有因子数据
            
        Returns:
            str: 技术因子分析报告
        """
        try:
            report = f"\n{'='*60}\n"
            report += f"技术因子分析报告 - {symbol}\n"
            report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"{'='*60}\n\n"
            
            for category, factors in all_factors.items():
                if factors:  # 只显示有数据的类别
                    report += f"{category.upper()} 因子:\n"
                    report += f"{'-'*40}\n"
                    
                    for factor_name, factor_value in factors.items():
                        if isinstance(factor_value, (int, float)) and not np.isnan(factor_value):
                            if abs(factor_value) >= 1000:
                                report += f"{factor_name:25}: {factor_value:8.0f}\n"
                            elif abs(factor_value) >= 1:
                                report += f"{factor_name:25}: {factor_value:8.2f}\n"
                            else:
                                report += f"{factor_name:25}: {factor_value:8.4f}\n"
                    
                    report += "\n"
            
            # 添加技术分析总结
            report += "技术分析总结:\n"
            report += f"{'-'*40}\n"
            
            # 趋势分析
            if 'trend' in all_factors:
                trend_factors = all_factors['trend']
                if 'adx' in trend_factors:
                    adx = trend_factors['adx']
                    if adx >= 25:
                        report += f"趋势强度: 强趋势 (ADX={adx:.1f})\n"
                    elif adx >= 20:
                        report += f"趋势强度: 中等趋势 (ADX={adx:.1f})\n"
                    else:
                        report += f"趋势强度: 弱趋势 (ADX={adx:.1f})\n"
            
            # 动量分析
            if 'momentum' in all_factors:
                momentum_factors = all_factors['momentum']
                if 'rsi_14' in momentum_factors:
                    rsi = momentum_factors['rsi_14']
                    if rsi <= 30:
                        report += f"RSI信号: 超卖 (RSI={rsi:.1f})\n"
                    elif rsi >= 70:
                        report += f"RSI信号: 超买 (RSI={rsi:.1f})\n"
                    else:
                        report += f"RSI信号: 中性 (RSI={rsi:.1f})\n"
            
            # 波动率分析
            if 'volatility' in all_factors:
                vol_factors = all_factors['volatility']
                if 'volatility_20' in vol_factors:
                    vol = vol_factors['volatility_20']
                    if vol >= 30:
                        report += f"波动率: 高波动 ({vol:.1f}%)\n"
                    elif vol >= 20:
                        report += f"波动率: 中等波动 ({vol:.1f}%)\n"
                    else:
                        report += f"波动率: 低波动 ({vol:.1f}%)\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成技术因子报告时出错: {e}")
            return f"生成报告失败: {e}"


def main():
    """主函数 - 示例用法"""
    try:
        # 创建技术因子计算器实例
        calculator = TechnicalFactorCalculator()
        
        # 示例数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 生成模拟价格数据
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices[:-1]],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'close': prices[:-1]
        })
        
        volume_data = pd.DataFrame({
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # 计算所有技术因子
        all_factors = calculator.calculate_all_factors('AAPL', price_data, volume_data)
        
        # 生成报告
        report = calculator.generate_factor_report('AAPL', all_factors)
        print(report)
        
    except Exception as e:
        logger.error(f"主函数执行出错: {e}")


if __name__ == "__main__":
    main()