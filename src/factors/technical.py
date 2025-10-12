"""
技术因子计算模块 - 优化版

提供各种技术指标的计算，包括趋势、动量、波动率、成交量等因子
优化了计算方法以提高Alpha生成能力
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
import warnings
import functools
from concurrent.futures import ThreadPoolExecutor
import time
warnings.filterwarnings('ignore')

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("pandas-ta not available, using manual calculations")

# 缓存装饰器
def cache_result(func):
    """缓存计算结果的装饰器"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 创建缓存键
        key = str(hash((str(args), str(sorted(kwargs.items())))))
        
        if key in cache:
            return cache[key]
        
        result = func(self, *args, **kwargs)
        cache[key] = result
        return result
    
    return wrapper

class TechnicalFactors:
    """技术因子计算器 - 性能优化版"""
    
    def __init__(self, enable_parallel=True, max_workers=4):
        """初始化技术因子计算器"""
        self.factors = {}
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.calculation_cache = {}
        print("技术因子计算器初始化完成")
    
    @cache_result
    def calculate_enhanced_sma(self, data: pd.Series, windows: List[int] = [5, 10, 20, 50, 200]) -> Dict[str, pd.Series]:
        """
        计算增强版简单移动平均线 - 向量化优化
        """
        results = {}
        
        # 向量化计算所有窗口的SMA
        for window in windows:
            if len(data) >= window:
                sma = data.rolling(window=window, min_periods=window//2).mean()
                
                # 基础SMA
                results[f'sma_{window}'] = sma
                
                # 向量化计算SMA斜率
                results[f'sma_{window}_slope'] = sma.diff(5) / sma.shift(5)
                
                # 向量化计算价格相对SMA位置
                results[f'price_to_sma_{window}'] = (data / sma - 1) * 100
        
        # 批量计算SMA交叉信号
        sorted_windows = sorted(windows)
        for i in range(len(sorted_windows) - 1):
            short_window = sorted_windows[i]
            long_window = sorted_windows[i + 1]
            
            if f'sma_{short_window}' in results and f'sma_{long_window}' in results:
                short_sma = results[f'sma_{short_window}']
                long_sma = results[f'sma_{long_window}']
                results[f'sma_cross_{short_window}_{long_window}'] = (short_sma > long_sma).astype(int)
        
        return results

    @cache_result
    def calculate_enhanced_ema(self, data: pd.Series, windows: List[int] = [12, 26, 50]) -> Dict[str, pd.Series]:
        """
        计算增强版指数移动平均线 - 向量化优化
        """
        results = {}
        
        # 向量化计算所有EMA
        for window in windows:
            if len(data) >= window:
                ema = data.ewm(span=window, adjust=False).mean()
                results[f'ema_{window}'] = ema
                
                # 向量化计算EMA斜率和相对位置
                results[f'ema_{window}_slope'] = ema.diff(3) / ema.shift(3)
                results[f'price_to_ema_{window}'] = (data / ema - 1) * 100
        
        return results

    @cache_result
    def calculate_enhanced_rsi(self, data: pd.Series, periods: List[int] = [14, 21]) -> Dict[str, pd.Series]:
        """
        计算增强版RSI - 向量化优化
        """
        results = {}
        
        # 预计算价格变化
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        for period in periods:
            if len(data) >= period * 2:
                # 向量化计算RSI
                avg_gain = gain.rolling(window=period, min_periods=period).mean()
                avg_loss = loss.rolling(window=period, min_periods=period).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                results[f'rsi_{period}'] = rsi
                
                # RSI衍生指标
                results[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
                results[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
                results[f'rsi_{period}_momentum'] = rsi.diff(5)
        
        return results

    def _calculate_factor_group(self, func_name: str, *args, **kwargs) -> Dict[str, pd.Series]:
        """并行计算因子组的辅助方法"""
        method = getattr(self, func_name)
        return method(*args, **kwargs)

    def calculate_all_factors(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        计算所有技术因子 - 性能优化版
        
        Args:
            data: 包含价格和成交量数据的字典
        
        Returns:
            所有计算出的因子字典
        """
        start_time = time.time()
        all_factors = {}
        
        close = data['close']
        high = data.get('high', close)
        low = data.get('low', close)
        volume = data.get('volume')
        returns = data.get('returns', close.pct_change())
        
        print("开始计算技术因子（性能优化版）...")
        
        if self.enable_parallel:
            # 并行计算因子组
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                # 提交各个因子计算任务
                futures.append(executor.submit(self.calculate_enhanced_sma, close))
                futures.append(executor.submit(self.calculate_enhanced_ema, close))
                futures.append(executor.submit(self.calculate_enhanced_rsi, close))
                futures.append(executor.submit(self.calculate_enhanced_macd, close))
                futures.append(executor.submit(self.calculate_enhanced_bollinger_bands, close))
                futures.append(executor.submit(self.calculate_momentum_factors, close, high, low))
                futures.append(executor.submit(self.calculate_volatility_factors, close, returns))
                
                if volume is not None:
                    futures.append(executor.submit(self.calculate_volume_factors, volume, close))
                
                futures.append(executor.submit(self.calculate_pattern_factors, close, high, low))
                futures.append(executor.submit(self.calculate_cycle_factors, close))
                
                # 收集结果
                for future in futures:
                    try:
                        result = future.result()
                        all_factors.update(result)
                    except Exception as e:
                        print(f"并行计算出错: {e}")
        else:
            # 串行计算
            factor_groups = [
                self.calculate_enhanced_sma(close),
                self.calculate_enhanced_ema(close),
                self.calculate_enhanced_rsi(close),
                self.calculate_enhanced_macd(close),
                self.calculate_enhanced_bollinger_bands(close),
                self.calculate_momentum_factors(close, high, low),
                self.calculate_volatility_factors(close, returns),
                self.calculate_pattern_factors(close, high, low),
                self.calculate_cycle_factors(close)
            ]
            
            if volume is not None:
                factor_groups.append(self.calculate_volume_factors(volume, close))
            
            for factors in factor_groups:
                all_factors.update(factors)
        
        # 向量化清理因子
        cleaned_factors = self._vectorized_factor_cleaning(all_factors)
        
        calculation_time = time.time() - start_time
        print(f"技术因子计算完成，共生成 {len(cleaned_factors)} 个有效因子，耗时: {calculation_time:.3f}秒")
        
        return cleaned_factors
    
    def _vectorized_factor_cleaning(self, factors: Dict[str, pd.Series], min_valid_ratio: float = 0.5) -> Dict[str, pd.Series]:
        """向量化因子清理"""
        cleaned_factors = {}
        
        for name, factor in factors.items():
            if isinstance(factor, pd.Series):
                # 向量化计算有效比例
                valid_ratio = factor.notna().sum() / len(factor)
                if valid_ratio >= min_valid_ratio:
                    cleaned_factors[name] = factor
        
        return cleaned_factors

    def calculate_enhanced_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        计算增强版MACD指标
        """
        results = {}
        
        if len(data) >= slow * 2:
            if PANDAS_TA_AVAILABLE:
                macd_data = ta.macd(data, fast=fast, slow=slow, signal=signal)
                if macd_data is not None and not macd_data.empty:
                    macd_line = macd_data.iloc[:, 0]
                    macd_histogram = macd_data.iloc[:, 1]
                    macd_signal = macd_data.iloc[:, 2]
                else:
                    # 手动计算
                    ema_fast = data.ewm(span=fast).mean()
                    ema_slow = data.ewm(span=slow).mean()
                    macd_line = ema_fast - ema_slow
                    macd_signal = macd_line.ewm(span=signal).mean()
                    macd_histogram = macd_line - macd_signal
            else:
                ema_fast = data.ewm(span=fast).mean()
                ema_slow = data.ewm(span=slow).mean()
                macd_line = ema_fast - ema_slow
                macd_signal = macd_line.ewm(span=signal).mean()
                macd_histogram = macd_line - macd_signal
            
            results['macd_line'] = macd_line
            results['macd_signal'] = macd_signal
            results['macd_histogram'] = macd_histogram
            
            # MACD交叉信号
            results['macd_bullish_cross'] = ((macd_line > macd_signal) & 
                                           (macd_line.shift(1) <= macd_signal.shift(1))).astype(int)
            results['macd_bearish_cross'] = ((macd_line < macd_signal) & 
                                           (macd_line.shift(1) >= macd_signal.shift(1))).astype(int)
            
            # MACD动量
            results['macd_momentum'] = macd_histogram.diff(3)
            
            # MACD零轴穿越
            results['macd_zero_cross'] = ((macd_line > 0) & (macd_line.shift(1) <= 0)).astype(int)
        
        return results
    
    def calculate_enhanced_bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        计算增强版布林带指标
        """
        results = {}
        
        if len(data) >= window:
            if PANDAS_TA_AVAILABLE:
                bb_data = ta.bbands(data, length=window, std=std_dev)
                if bb_data is not None and not bb_data.empty:
                    bb_lower = bb_data.iloc[:, 0]
                    bb_middle = bb_data.iloc[:, 1]
                    bb_upper = bb_data.iloc[:, 2]
                else:
                    sma = data.rolling(window=window).mean()
                    std = data.rolling(window=window).std()
                    bb_upper = sma + (std * std_dev)
                    bb_lower = sma - (std * std_dev)
                    bb_middle = sma
            else:
                sma = data.rolling(window=window).mean()
                std = data.rolling(window=window).std()
                bb_upper = sma + (std * std_dev)
                bb_lower = sma - (std * std_dev)
                bb_middle = sma
            
            results['bb_upper'] = bb_upper
            results['bb_middle'] = bb_middle
            results['bb_lower'] = bb_lower
            
            # 布林带位置
            bb_width = bb_upper - bb_lower
            results['bb_position'] = (data - bb_lower) / bb_width
            
            # 布林带宽度 (波动率指标)
            results['bb_width'] = bb_width / bb_middle
            
            # 布林带挤压
            results['bb_squeeze'] = (bb_width < bb_width.rolling(20).mean() * 0.8).astype(int)
            
            # 布林带突破信号
            results['bb_upper_break'] = (data > bb_upper).astype(int)
            results['bb_lower_break'] = (data < bb_lower).astype(int)
            
            # 布林带回归信号
            results['bb_mean_reversion'] = ((data > bb_upper) | (data < bb_lower)).astype(int)
        
        return results
    
    def calculate_momentum_factors(self, data: pd.Series, high: pd.Series = None, low: pd.Series = None) -> Dict[str, pd.Series]:
        """
        计算动量因子
        """
        results = {}
        
        # 价格动量
        for period in [1, 3, 5, 10, 20, 60]:
            if len(data) > period:
                results[f'momentum_{period}d'] = (data / data.shift(period) - 1) * 100
        
        # 加速动量
        mom_5 = data / data.shift(5) - 1
        mom_20 = data / data.shift(20) - 1
        results['momentum_acceleration'] = mom_5 - mom_20
        
        # 相对强度
        if len(data) >= 252:
            results['relative_strength_1y'] = data / data.shift(252) - 1
        
        # 价格位置 (在一定期间内的相对位置)
        for period in [20, 60, 252]:
            if len(data) >= period:
                period_high = data.rolling(period).max()
                period_low = data.rolling(period).min()
                results[f'price_position_{period}d'] = (data - period_low) / (period_high - period_low)
        
        # 突破动量
        if high is not None and low is not None:
            for period in [20, 60]:
                if len(high) >= period:
                    period_high = high.rolling(period).max()
                    period_low = low.rolling(period).min()
                    results[f'breakout_momentum_{period}d'] = np.where(
                        data > period_high.shift(1), 1,
                        np.where(data < period_low.shift(1), -1, 0)
                    )
        
        return results
    
    def calculate_volatility_factors(self, data: pd.Series, returns: pd.Series = None) -> Dict[str, pd.Series]:
        """
        计算波动率因子
        """
        results = {}
        
        if returns is None:
            returns = data.pct_change()
        
        # 历史波动率
        for period in [5, 10, 20, 60]:
            if len(returns) >= period:
                results[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
        
        # 波动率比率
        if len(returns) >= 60:
            vol_short = returns.rolling(10).std()
            vol_long = returns.rolling(60).std()
            results['volatility_ratio'] = vol_short / vol_long
        
        # 真实波动幅度 (ATR)
        if len(data) >= 14:
            high_low = data.rolling(2).max() - data.rolling(2).min()
            high_close = abs(data - data.shift(1))
            low_close = abs(data.shift(1) - data)
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            results['atr_14'] = true_range.rolling(14).mean()
            results['atr_ratio'] = results['atr_14'] / data
        
        # 波动率突破
        for period in [20, 60]:
            if len(returns) >= period:
                vol = returns.rolling(period).std()
                vol_threshold = vol.rolling(period).quantile(0.8)
                results[f'volatility_breakout_{period}d'] = (vol > vol_threshold).astype(int)
        
        # Garman-Klass波动率估计 (如果有高低价数据)
        # 这里简化处理，使用收盘价估算
        results['gk_volatility'] = returns.rolling(20).std() * np.sqrt(252)
        
        return results
    
    def calculate_volume_factors(self, volume: pd.Series, price: pd.Series) -> Dict[str, pd.Series]:
        """
        计算成交量因子
        """
        results = {}
        
        # 成交量移动平均
        for period in [5, 10, 20, 60]:
            if len(volume) >= period:
                vol_ma = volume.rolling(period).mean()
                results[f'volume_ma_{period}'] = vol_ma
                results[f'volume_ratio_{period}'] = volume / vol_ma
        
        # 成交量价格趋势 (VPT)
        returns = price.pct_change()
        results['vpt'] = (returns * volume).cumsum()
        
        # 能量潮 (OBV)
        price_change = price.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0))
        results['obv'] = pd.Series(obv, index=volume.index).cumsum()
        
        # 成交量加权平均价格 (VWAP)
        if len(volume) >= 20:
            typical_price = price  # 简化，使用收盘价
            vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
            results['vwap_20'] = vwap
            results['price_to_vwap'] = (price / vwap - 1) * 100
        
        # 成交量突破
        for period in [20, 60]:
            if len(volume) >= period:
                vol_threshold = volume.rolling(period).quantile(0.8)
                results[f'volume_breakout_{period}d'] = (volume > vol_threshold).astype(int)
        
        # 成交量趋势
        for period in [10, 20]:
            if len(volume) >= period:
                vol_ma = volume.rolling(period).mean()
                results[f'volume_trend_{period}d'] = (vol_ma > vol_ma.shift(5)).astype(int)
        
        # 资金流向指标 (简化版MFI)
        if len(price) >= 14:
            typical_price = price
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(price > price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(price < price.shift(1), 0).rolling(14).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            results['mfi_14'] = mfi
        
        return results
    
    def calculate_pattern_factors(self, data: pd.Series, high: pd.Series = None, low: pd.Series = None) -> Dict[str, pd.Series]:
        """
        计算价格形态因子
        """
        results = {}
        
        # 价格缺口
        gap = (data - data.shift(1)) / data.shift(1)
        results['gap_up'] = (gap > 0.02).astype(int)  # 向上缺口
        results['gap_down'] = (gap < -0.02).astype(int)  # 向下缺口
        
        # 连续上涨/下跌天数
        price_change = data.diff()
        up_days = (price_change > 0).astype(int)
        down_days = (price_change < 0).astype(int)
        
        # 计算连续天数
        up_streak = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
        down_streak = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()
        
        results['consecutive_up_days'] = up_streak * up_days
        results['consecutive_down_days'] = down_streak * down_days
        
        # 价格反转信号
        for period in [3, 5, 10]:
            if len(data) >= period:
                period_high = data.rolling(period).max()
                period_low = data.rolling(period).min()
                
                # 顶部反转
                results[f'top_reversal_{period}d'] = ((data == period_high) & 
                                                     (data.shift(1) < data) & 
                                                     (data.shift(-1) < data)).astype(int)
                
                # 底部反转
                results[f'bottom_reversal_{period}d'] = ((data == period_low) & 
                                                        (data.shift(1) > data) & 
                                                        (data.shift(-1) > data)).astype(int)
        
        # 支撑阻力位
        for period in [20, 60]:
            if len(data) >= period:
                resistance = data.rolling(period).max()
                support = data.rolling(period).min()
                
                results[f'near_resistance_{period}d'] = (abs(data - resistance) / data < 0.02).astype(int)
                results[f'near_support_{period}d'] = (abs(data - support) / data < 0.02).astype(int)
        
        return results
    
    def calculate_cycle_factors(self, data: pd.Series) -> Dict[str, pd.Series]:
        """
        计算周期性因子
        """
        results = {}
        
        if len(data) >= 60:
            # 季节性趋势
            returns = data.pct_change()
            
            # 月度效应 (简化版)
            if hasattr(data.index, 'month'):
                monthly_returns = returns.groupby(data.index.month).mean()
                current_month = data.index[-1].month if hasattr(data.index[-1], 'month') else 1
                results['monthly_effect'] = pd.Series([monthly_returns.get(current_month, 0)] * len(data), index=data.index)
            
            # 周期性动量
            for cycle in [5, 10, 22]:  # 周、双周、月
                if len(data) >= cycle * 4:
                    cycle_returns = []
                    for i in range(len(data)):
                        if i >= cycle * 3:
                            recent_cycles = []
                            for j in range(3):
                                start_idx = i - (j + 1) * cycle
                                end_idx = i - j * cycle
                                if start_idx >= 0:
                                    cycle_return = (data.iloc[end_idx] / data.iloc[start_idx] - 1)
                                    recent_cycles.append(cycle_return)
                            
                            if recent_cycles:
                                cycle_returns.append(np.mean(recent_cycles))
                            else:
                                cycle_returns.append(0)
                        else:
                            cycle_returns.append(0)
                    
                    results[f'cycle_momentum_{cycle}d'] = pd.Series(cycle_returns, index=data.index)
        
        return results
    
    def calculate_advanced_trend_factors(self, data: pd.Series, high: pd.Series = None, low: pd.Series = None) -> Dict[str, pd.Series]:
        """
        计算高级趋势因子
        """
        results = {}
        
        # 自适应移动平均
        results['adaptive_ma'] = self._calculate_adaptive_ma(data)
        
        # 趋势强度指标
        results['trend_strength'] = self._calculate_trend_strength(data)
        
        # 趋势持续性指标
        results['trend_persistence'] = self._calculate_trend_persistence(data)
        
        # 价格通道指标
        if high is not None and low is not None:
            results.update(self._calculate_price_channels(high, low, data))
        
        # 支撑阻力指标
        results.update(self._calculate_support_resistance(data))
        
        return results
    
    def _calculate_adaptive_ma(self, data: pd.Series, window: int = 20) -> pd.Series:
        """计算自适应移动平均"""
        volatility = data.rolling(window).std()
        volatility_ma = volatility.rolling(window).mean()
        
        # 根据波动率调整平滑因子
        alpha = np.clip(volatility / volatility_ma, 0.1, 0.9)
        
        adaptive_ma = pd.Series(index=data.index, dtype=float)
        adaptive_ma.iloc[0] = data.iloc[0]
        
        for i in range(1, len(data)):
            adaptive_ma.iloc[i] = alpha.iloc[i] * data.iloc[i] + (1 - alpha.iloc[i]) * adaptive_ma.iloc[i-1]
        
        return adaptive_ma
    
    def _calculate_trend_strength(self, data: pd.Series, window: int = 20) -> pd.Series:
        """计算趋势强度"""
        returns = data.pct_change()
        
        # 计算正负收益率的比例
        positive_returns = returns[returns > 0].rolling(window).count()
        total_returns = returns.rolling(window).count()
        
        trend_strength = (positive_returns / total_returns - 0.5) * 2
        return trend_strength.fillna(0)
    
    def _calculate_trend_persistence(self, data: pd.Series, window: int = 20) -> pd.Series:
        """计算趋势持续性"""
        returns = data.pct_change()
        
        # 计算连续同向收益的长度
        sign_changes = (returns.shift(1) * returns < 0).astype(int)
        persistence = sign_changes.rolling(window).sum()
        
        return 1 - (persistence / window)
    
    def _calculate_price_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """计算价格通道指标"""
        results = {}
        
        # 唐奇安通道
        results['donchian_upper'] = high.rolling(window).max()
        results['donchian_lower'] = low.rolling(window).min()
        results['donchian_middle'] = (results['donchian_upper'] + results['donchian_lower']) / 2
        results['donchian_position'] = (close - results['donchian_lower']) / (results['donchian_upper'] - results['donchian_lower'])
        
        # 肯特纳通道
        typical_price = (high + low + close) / 3
        keltner_ma = typical_price.rolling(window).mean()
        atr = self._calculate_atr(high, low, close, window)
        
        results['keltner_upper'] = keltner_ma + 2 * atr
        results['keltner_lower'] = keltner_ma - 2 * atr
        results['keltner_position'] = (close - results['keltner_lower']) / (results['keltner_upper'] - results['keltner_lower'])
        
        return results
    
    def _calculate_support_resistance(self, data: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """计算支撑阻力指标"""
        results = {}
        
        # 局部高点和低点
        local_max = data.rolling(window, center=True).max() == data
        local_min = data.rolling(window, center=True).min() == data
        
        # 支撑阻力强度
        resistance_strength = local_max.rolling(window*2).sum()
        support_strength = local_min.rolling(window*2).sum()
        
        results['resistance_strength'] = resistance_strength
        results['support_strength'] = support_strength
        results['sr_ratio'] = resistance_strength / (support_strength + 1e-8)
        
        return results
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """计算平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window).mean()
    
    def calculate_market_microstructure_factors(self, data: pd.Series, volume: pd.Series = None) -> Dict[str, pd.Series]:
        """
        计算市场微观结构因子
        """
        results = {}
        
        # 价格跳跃检测
        results['price_jumps'] = self._detect_price_jumps(data)
        
        # 价格效率指标
        results['price_efficiency'] = self._calculate_price_efficiency(data)
        
        # 流动性指标
        if volume is not None:
            results.update(self._calculate_liquidity_factors(data, volume))
        
        # 市场冲击指标
        results['market_impact'] = self._calculate_market_impact(data)
        
        return results
    
    def _detect_price_jumps(self, data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """检测价格跳跃"""
        returns = data.pct_change()
        volatility = returns.rolling(20).std()
        
        # 标准化收益率
        standardized_returns = returns / volatility
        
        # 检测跳跃
        jumps = (abs(standardized_returns) > threshold).astype(int)
        return jumps
    
    def _calculate_price_efficiency(self, data: pd.Series, window: int = 20) -> pd.Series:
        """计算价格效率指标"""
        returns = data.pct_change()
        
        # 计算价格路径长度
        price_path = abs(returns).rolling(window).sum()
        
        # 计算直线距离
        straight_distance = abs(data - data.shift(window)) / data.shift(window)
        
        # 效率比率
        efficiency = straight_distance / (price_path + 1e-8)
        return efficiency
    
    def _calculate_liquidity_factors(self, price: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """计算流动性因子"""
        results = {}
        
        # Amihud非流动性指标
        returns = price.pct_change()
        dollar_volume = price * volume
        results['amihud_illiquidity'] = abs(returns) / (dollar_volume + 1e-8)
        
        # 成交量加权平均价格偏离
        vwap = (price * volume).rolling(20).sum() / volume.rolling(20).sum()
        results['vwap_deviation'] = (price - vwap) / vwap
        
        # 流动性比率
        results['liquidity_ratio'] = volume / volume.rolling(20).mean()
        
        return results
    
    def _calculate_market_impact(self, data: pd.Series, window: int = 5) -> pd.Series:
        """计算市场冲击指标"""
        returns = data.pct_change()
        
        # 计算价格冲击的持续性
        impact_persistence = returns.rolling(window).corr(returns.shift(1))
        
        return impact_persistence