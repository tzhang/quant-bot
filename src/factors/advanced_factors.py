"""
高级因子计算模块

包含更复杂和高级的量化因子，如：
1. 高频因子 - 基于高频数据的因子
2. 宏观因子 - 基于宏观经济数据的因子
3. 情绪因子 - 基于市场情绪的因子
4. 网络因子 - 基于股票关联网络的因子
5. 期权因子 - 基于期权数据的因子
6. 另类数据因子 - 基于另类数据源的因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# 网络分析库
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# 统计库
try:
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class AdvancedFactorCalculator:
    """高级因子计算器"""
    
    def __init__(self, lookback_window: int = 252, min_periods: int = 30):
        """
        初始化高级因子计算器
        
        Args:
            lookback_window: 回望窗口期
            min_periods: 最小计算周期
        """
        self.lookback_window = lookback_window
        self.min_periods = min_periods
        print("高级因子计算器初始化完成")
    
    def calculate_high_frequency_factors(self, price_data: pd.DataFrame, 
                                       volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算高频因子
        
        Args:
            price_data: 价格数据 (OHLC)
            volume_data: 成交量数据
        
        Returns:
            高频因子字典
        """
        results = {}
        
        if 'close' not in price_data.columns:
            return results
        
        close = price_data['close']
        
        try:
            # 1. 价格跳跃因子
            results['price_jump'] = self._calculate_price_jump(close)
            
            # 2. 已实现波动率因子
            results['realized_volatility'] = self._calculate_realized_volatility(close)
            
            # 3. 偏度因子
            results['realized_skewness'] = self._calculate_realized_skewness(close)
            
            # 4. 峰度因子
            results['realized_kurtosis'] = self._calculate_realized_kurtosis(close)
            
            # 5. 价格效率因子
            results['price_efficiency'] = self._calculate_price_efficiency(close)
            
            # 6. 微观结构噪声因子
            results['microstructure_noise'] = self._calculate_microstructure_noise(close)
            
            # 7. 流动性因子
            if volume_data is not None and 'volume' in volume_data.columns:
                volume = volume_data['volume']
                results['amihud_illiquidity'] = self._calculate_amihud_illiquidity(close, volume)
                results['volume_weighted_price'] = self._calculate_vwap_factor(price_data, volume)
                results['order_flow_imbalance'] = self._calculate_order_flow_imbalance(close, volume)
            
            # 8. 价格冲击因子
            results['price_impact'] = self._calculate_price_impact(close)
            
            # 9. 买卖价差估计因子
            results['bid_ask_spread'] = self._calculate_bid_ask_spread_estimate(price_data)
            
            # 10. 高频动量因子
            results['hf_momentum'] = self._calculate_hf_momentum(close)
            
        except Exception as e:
            print(f"高频因子计算失败: {e}")
        
        return results
    
    def _calculate_price_jump(self, prices: pd.Series, threshold: float = 3.0) -> pd.Series:
        """计算价格跳跃因子"""
        returns = prices.pct_change()
        rolling_std = returns.rolling(20).std()
        
        # 标准化收益率
        standardized_returns = returns / rolling_std
        
        # 识别跳跃
        jumps = (np.abs(standardized_returns) > threshold).astype(int)
        
        # 跳跃强度
        jump_intensity = jumps.rolling(20).sum()
        
        return jump_intensity
    
    def _calculate_realized_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算已实现波动率"""
        returns = prices.pct_change()
        realized_vol = returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2)))
        return realized_vol
    
    def _calculate_realized_skewness(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算已实现偏度"""
        returns = prices.pct_change()
        realized_skew = returns.rolling(window).skew()
        return realized_skew
    
    def _calculate_realized_kurtosis(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算已实现峰度"""
        returns = prices.pct_change()
        realized_kurt = returns.rolling(window).kurt()
        return realized_kurt
    
    def _calculate_price_efficiency(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算价格效率因子"""
        returns = prices.pct_change()
        
        # 计算价格路径的效率
        efficiency = pd.Series(np.nan, index=prices.index)
        
        for i in range(window, len(prices)):
            price_window = prices.iloc[i-window:i]
            
            # 实际价格变化
            actual_change = abs(price_window.iloc[-1] - price_window.iloc[0])
            
            # 累积价格变化
            cumulative_change = np.sum(np.abs(price_window.diff().dropna()))
            
            if cumulative_change > 0:
                efficiency.iloc[i] = actual_change / cumulative_change
            else:
                efficiency.iloc[i] = 0
        
        return efficiency
    
    def _calculate_microstructure_noise(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算微观结构噪声因子"""
        returns = prices.pct_change()
        
        # 计算一阶自相关系数作为噪声指标
        noise = returns.rolling(window).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
        
        return -noise  # 负自相关表示更多噪声
    
    def _calculate_amihud_illiquidity(self, prices: pd.Series, volume: pd.Series, 
                                    window: int = 20) -> pd.Series:
        """计算Amihud非流动性指标"""
        returns = prices.pct_change()
        dollar_volume = prices * volume
        
        # Amihud非流动性 = |收益率| / 成交额
        illiquidity = np.abs(returns) / dollar_volume
        
        # 滚动平均
        amihud = illiquidity.rolling(window).mean()
        
        return amihud
    
    def _calculate_vwap_factor(self, price_data: pd.DataFrame, volume: pd.Series) -> pd.Series:
        """计算VWAP相关因子"""
        if 'high' in price_data.columns and 'low' in price_data.columns and 'close' in price_data.columns:
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
        else:
            typical_price = price_data['close']
        
        # 成交量加权平均价格
        vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # 价格相对于VWAP的偏离
        vwap_deviation = (typical_price - vwap) / vwap
        
        return vwap_deviation
    
    def _calculate_order_flow_imbalance(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """计算订单流不平衡因子"""
        returns = prices.pct_change()
        
        # 简化的订单流不平衡：正收益时为买入，负收益时为卖出
        buy_volume = volume.where(returns > 0, 0)
        sell_volume = volume.where(returns < 0, 0)
        
        # 订单流不平衡
        ofi = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        return ofi.rolling(10).mean()
    
    def _calculate_price_impact(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算价格冲击因子"""
        returns = prices.pct_change()
        
        # 价格冲击的持续性
        impact = pd.Series(np.nan, index=prices.index)
        
        for i in range(window, len(returns)):
            recent_returns = returns.iloc[i-window:i]
            
            # 计算收益率的持续性（自相关）
            if len(recent_returns.dropna()) > 5:
                autocorr = recent_returns.autocorr(lag=1)
                impact.iloc[i] = autocorr if not np.isnan(autocorr) else 0
        
        return impact
    
    def _calculate_bid_ask_spread_estimate(self, price_data: pd.DataFrame) -> pd.Series:
        """估计买卖价差"""
        if 'high' in price_data.columns and 'low' in price_data.columns:
            # 使用高低价差作为买卖价差的代理
            spread = (price_data['high'] - price_data['low']) / price_data['close']
        else:
            # 使用价格变化的标准差估计
            returns = price_data['close'].pct_change()
            spread = returns.rolling(20).std() * 2  # 简化估计
        
        return spread
    
    def _calculate_hf_momentum(self, prices: pd.Series) -> pd.Series:
        """计算高频动量因子"""
        returns = prices.pct_change()
        
        # 多时间尺度动量
        momentum_1h = returns.rolling(5).sum()  # 假设5个周期为1小时
        momentum_1d = returns.rolling(20).sum()  # 假设20个周期为1天
        
        # 动量强度
        momentum_strength = momentum_1h / momentum_1d.rolling(5).std()
        
        return momentum_strength
    
    def calculate_sentiment_factors(self, price_data: pd.DataFrame, 
                                  volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算情绪因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
        
        Returns:
            情绪因子字典
        """
        results = {}
        
        if 'close' not in price_data.columns:
            return results
        
        close = price_data['close']
        
        try:
            # 1. VIX类波动率指数
            results['vix_like'] = self._calculate_vix_like(close)
            
            # 2. 恐慌指数
            results['fear_index'] = self._calculate_fear_index(price_data)
            
            # 3. 贪婪指数
            results['greed_index'] = self._calculate_greed_index(price_data, volume_data)
            
            # 4. 市场情绪摆动指标
            results['sentiment_oscillator'] = self._calculate_sentiment_oscillator(close)
            
            # 5. 投资者情绪指标
            results['investor_sentiment'] = self._calculate_investor_sentiment(price_data, volume_data)
            
            # 6. 市场压力指标
            results['market_stress'] = self._calculate_market_stress(close)
            
            # 7. 羊群效应指标
            results['herding_behavior'] = self._calculate_herding_behavior(close)
            
            # 8. 过度反应指标
            results['overreaction'] = self._calculate_overreaction(close)
            
        except Exception as e:
            print(f"情绪因子计算失败: {e}")
        
        return results
    
    def _calculate_vix_like(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算VIX类波动率指数"""
        returns = prices.pct_change()
        
        # 已实现波动率
        realized_vol = returns.rolling(window).std() * np.sqrt(252)
        
        # VIX类指数：年化波动率 * 100
        vix_like = realized_vol * 100
        
        return vix_like
    
    def _calculate_fear_index(self, price_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算恐慌指数"""
        close = price_data['close']
        
        # 价格下跌的频率和幅度
        returns = close.pct_change()
        negative_returns = returns.where(returns < 0, 0)
        
        # 恐慌指数：负收益的强度和频率
        fear_intensity = np.abs(negative_returns).rolling(window).mean()
        fear_frequency = (returns < 0).rolling(window).mean()
        
        fear_index = fear_intensity * fear_frequency * 100
        
        return fear_index
    
    def _calculate_greed_index(self, price_data: pd.DataFrame, 
                             volume_data: pd.DataFrame = None, window: int = 20) -> pd.Series:
        """计算贪婪指数"""
        close = price_data['close']
        returns = close.pct_change()
        
        # 价格上涨的频率和幅度
        positive_returns = returns.where(returns > 0, 0)
        
        greed_intensity = positive_returns.rolling(window).mean()
        greed_frequency = (returns > 0).rolling(window).mean()
        
        greed_base = greed_intensity * greed_frequency * 100
        
        # 如果有成交量数据，考虑成交量放大效应
        if volume_data is not None and 'volume' in volume_data.columns:
            volume = volume_data['volume']
            volume_ratio = volume / volume.rolling(window).mean()
            greed_index = greed_base * (1 + volume_ratio.rolling(5).mean())
        else:
            greed_index = greed_base
        
        return greed_index
    
    def _calculate_sentiment_oscillator(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算情绪摆动指标"""
        returns = prices.pct_change()
        
        # 类似RSI的情绪指标
        gain = returns.where(returns > 0, 0)
        loss = -returns.where(returns < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        sentiment_osc = 100 - (100 / (1 + rs))
        
        return sentiment_osc
    
    def _calculate_investor_sentiment(self, price_data: pd.DataFrame, 
                                    volume_data: pd.DataFrame = None) -> pd.Series:
        """计算投资者情绪指标"""
        close = price_data['close']
        
        # 多个情绪指标的综合
        # 1. 价格动量
        momentum = close / close.shift(20) - 1
        
        # 2. 波动率
        volatility = close.pct_change().rolling(20).std()
        
        # 3. 成交量情绪
        if volume_data is not None and 'volume' in volume_data.columns:
            volume = volume_data['volume']
            volume_sentiment = volume / volume.rolling(60).mean() - 1
        else:
            volume_sentiment = pd.Series(0, index=close.index)
        
        # 综合情绪指标
        sentiment = (momentum * 0.4 + 
                    (-volatility / volatility.rolling(60).mean()) * 0.3 + 
                    volume_sentiment * 0.3)
        
        return sentiment
    
    def _calculate_market_stress(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算市场压力指标"""
        returns = prices.pct_change()
        
        # 市场压力：极端收益的频率
        extreme_threshold = returns.rolling(60).std() * 2
        extreme_returns = np.abs(returns) > extreme_threshold
        
        stress_index = extreme_returns.rolling(window).mean() * 100
        
        return stress_index
    
    def _calculate_herding_behavior(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算羊群效应指标"""
        returns = prices.pct_change()
        
        # 羊群效应：收益率的一致性
        # 使用滚动相关系数的变化来衡量
        herding = pd.Series(np.nan, index=prices.index)
        
        for i in range(window * 2, len(returns)):
            recent_returns = returns.iloc[i-window:i]
            past_returns = returns.iloc[i-window*2:i-window]
            
            if len(recent_returns.dropna()) > 5 and len(past_returns.dropna()) > 5:
                # 计算收益率分布的相似性
                recent_std = recent_returns.std()
                past_std = past_returns.std()
                
                if past_std > 0:
                    herding.iloc[i] = 1 - abs(recent_std - past_std) / past_std
                else:
                    herding.iloc[i] = 0
        
        return herding
    
    def _calculate_overreaction(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算过度反应指标"""
        returns = prices.pct_change()
        
        # 过度反应：大幅变动后的反转
        large_moves = np.abs(returns) > returns.rolling(60).std() * 2
        
        overreaction = pd.Series(np.nan, index=prices.index)
        
        for i in range(window, len(returns)):
            if large_moves.iloc[i-1]:  # 前一期有大幅变动
                # 检查后续是否反转
                future_return = returns.iloc[i]
                past_return = returns.iloc[i-1]
                
                if past_return * future_return < 0:  # 方向相反
                    overreaction.iloc[i] = abs(future_return / past_return)
                else:
                    overreaction.iloc[i] = 0
            else:
                overreaction.iloc[i] = 0
        
        return overreaction.rolling(window).mean()
    
    def calculate_mathematical_factors(self, price_data: pd.DataFrame,
                                     volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算高级数学因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            
        Returns:
            数学因子字典
        """
        results = {}
        
        if 'close' not in price_data.columns:
            return results
            
        close = price_data['close']
        
        try:
            # 1. 分形维数因子
            results['fractal_dimension'] = self._calculate_fractal_dimension(close)
            
            # 2. 赫斯特指数因子
            results['hurst_exponent'] = self._calculate_hurst_exponent(close)
            
            # 3. 李雅普诺夫指数因子
            results['lyapunov_exponent'] = self._calculate_lyapunov_exponent(close)
            
            # 4. 熵因子
            results['shannon_entropy'] = self._calculate_shannon_entropy(close)
            results['approximate_entropy'] = self._calculate_approximate_entropy(close)
            
            # 5. 小波变换因子
            results.update(self._calculate_wavelet_factors(close))
            
            # 6. 傅里叶变换因子
            results.update(self._calculate_fourier_factors(close))
            
            # 7. 希尔伯特变换因子
            results['hilbert_transform'] = self._calculate_hilbert_transform(close)
            
            # 8. 相空间重构因子
            results.update(self._calculate_phase_space_factors(close))
            
            # 9. 多重分形因子
            results.update(self._calculate_multifractal_factors(close))
            
            # 10. 非线性动力学因子
            results.update(self._calculate_nonlinear_dynamics_factors(close))
            
        except Exception as e:
            print(f"数学因子计算失败: {e}")
            
        return results
    
    def _calculate_fractal_dimension(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """计算分形维数"""
        def box_counting_dimension(data):
            """盒计数法计算分形维数"""
            if len(data) < 10:
                return np.nan
                
            # 标准化数据
            data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)
            
            # 不同尺度的盒子大小
            scales = np.logspace(-2, 0, 10)
            counts = []
            
            for scale in scales:
                # 计算需要的盒子数量
                n_boxes = int(1 / scale)
                if n_boxes < 2:
                    continue
                    
                # 统计非空盒子数量
                boxes = np.zeros(n_boxes)
                for i, val in enumerate(data_norm):
                    box_idx = min(int(val * n_boxes), n_boxes - 1)
                    boxes[box_idx] = 1
                    
                counts.append(np.sum(boxes))
            
            if len(counts) < 3:
                return np.nan
                
            # 线性回归计算斜率
            log_scales = np.log(1 / np.array(scales[:len(counts)]))
            log_counts = np.log(np.array(counts))
            
            if len(log_scales) > 1:
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                return slope
            else:
                return np.nan
        
        return prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: box_counting_dimension(x.values), raw=False
        )
    
    def _calculate_hurst_exponent(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """计算赫斯特指数"""
        def hurst_rs(data):
            """R/S分析法计算赫斯特指数"""
            if len(data) < 10:
                return np.nan
                
            n = len(data)
            # 计算对数收益率
            log_returns = np.diff(np.log(data + 1e-10))
            
            # 不同时间尺度
            lags = range(2, min(n//4, 20))
            rs_values = []
            
            for lag in lags:
                # 分段计算R/S统计量
                segments = n // lag
                if segments < 2:
                    continue
                    
                rs_segment = []
                for i in range(segments):
                    start_idx = i * lag
                    end_idx = (i + 1) * lag
                    segment_data = log_returns[start_idx:end_idx]
                    
                    if len(segment_data) < 2:
                        continue
                        
                    # 计算累积偏差
                    mean_return = np.mean(segment_data)
                    cumulative_deviation = np.cumsum(segment_data - mean_return)
                    
                    # 计算范围R
                    R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                    
                    # 计算标准差S
                    S = np.std(segment_data)
                    
                    if S > 0:
                        rs_segment.append(R / S)
                
                if rs_segment:
                    rs_values.append(np.mean(rs_segment))
            
            if len(rs_values) < 3:
                return np.nan
                
            # 线性回归计算赫斯特指数
            log_lags = np.log(list(lags[:len(rs_values)]))
            log_rs = np.log(rs_values)
            
            hurst, _ = np.polyfit(log_lags, log_rs, 1)
            return hurst
        
        return prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: hurst_rs(x.values), raw=False
        )
    
    def _calculate_lyapunov_exponent(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """计算李雅普诺夫指数"""
        def lyapunov_exp(data):
            """计算最大李雅普诺夫指数"""
            if len(data) < 20:
                return np.nan
                
            # 相空间重构参数
            m = 3  # 嵌入维数
            tau = 1  # 时间延迟
            
            # 重构相空间
            n = len(data) - (m - 1) * tau
            if n < 10:
                return np.nan
                
            # 构建轨迹矩阵
            trajectory = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    trajectory[i, j] = data[i + j * tau]
            
            # 计算李雅普诺夫指数
            lyap_sum = 0
            count = 0
            
            for i in range(n - 1):
                # 找最近邻点
                distances = np.linalg.norm(trajectory - trajectory[i], axis=1)
                distances[i] = np.inf  # 排除自身
                
                nearest_idx = np.argmin(distances)
                if distances[nearest_idx] < 1e-10:
                    continue
                
                # 计算演化后的距离
                if i + 1 < n and nearest_idx + 1 < n:
                    d0 = distances[nearest_idx]
                    d1 = np.linalg.norm(trajectory[i + 1] - trajectory[nearest_idx + 1])
                    
                    if d1 > 1e-10 and d0 > 1e-10:
                        lyap_sum += np.log(d1 / d0)
                        count += 1
            
            return lyap_sum / count if count > 0 else np.nan
        
        return prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: lyapunov_exp(x.values), raw=False
        )
    
    def _calculate_shannon_entropy(self, prices: pd.Series, window: int = 50, bins: int = 10) -> pd.Series:
        """计算香农熵"""
        def shannon_entropy(data):
            if len(data) < bins:
                return np.nan
                
            # 计算收益率
            returns = np.diff(np.log(data + 1e-10))
            
            # 离散化
            hist, _ = np.histogram(returns, bins=bins, density=True)
            hist = hist / np.sum(hist)  # 归一化
            
            # 计算熵
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            return entropy
        
        return prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: shannon_entropy(x.values), raw=False
        )
    
    def _calculate_approximate_entropy(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """计算近似熵"""
        def approx_entropy(data, m=2, r=None):
            if len(data) < 20:
                return np.nan
                
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
                C = np.zeros(len(patterns))
                
                for i in range(len(patterns)):
                    template = patterns[i]
                    matches = sum([1 for pattern in patterns if _maxdist(template, pattern, m) <= r])
                    C[i] = matches / float(len(patterns))
                
                phi = np.mean([np.log(c) for c in C if c > 0])
                return phi
            
            return _phi(m) - _phi(m + 1)
        
        return prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: approx_entropy(x.values), raw=False
        )
    
    def _calculate_wavelet_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算小波变换因子"""
        try:
            import pywt
        except ImportError:
            print("PyWavelets未安装，跳过小波变换因子")
            return {}
        
        results = {}
        
        def wavelet_energy(data):
            """计算小波能量"""
            if len(data) < 16:
                return np.nan, np.nan, np.nan
                
            # 小波分解
            coeffs = pywt.wavedec(data, 'db4', level=3)
            
            # 计算各频带能量
            energies = [np.sum(c**2) for c in coeffs]
            total_energy = sum(energies)
            
            if total_energy == 0:
                return np.nan, np.nan, np.nan
                
            # 相对能量
            rel_energies = [e / total_energy for e in energies]
            
            return rel_energies[0], rel_energies[1], rel_energies[2] if len(rel_energies) > 2 else np.nan
        
        # 滚动计算小波因子
        wavelet_results = prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: wavelet_energy(x.values), raw=False
        )
        
        if not wavelet_results.empty:
            # 分离不同频带的能量
            results['wavelet_low_freq'] = wavelet_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['wavelet_mid_freq'] = wavelet_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
            results['wavelet_high_freq'] = wavelet_results.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
        
        return results
    
    def _calculate_fourier_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算傅里叶变换因子"""
        results = {}
        
        def fourier_analysis(data):
            """傅里叶频谱分析"""
            if len(data) < 16:
                return np.nan, np.nan, np.nan
                
            # FFT变换
            fft_vals = np.fft.fft(data - np.mean(data))
            power_spectrum = np.abs(fft_vals)**2
            
            # 频率分量
            freqs = np.fft.fftfreq(len(data))
            
            # 计算主要频率成分
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power_spectrum[:len(power_spectrum)//2]
            
            if len(positive_power) == 0:
                return np.nan, np.nan, np.nan
                
            # 主频率
            dominant_freq_idx = np.argmax(positive_power)
            dominant_freq = positive_freqs[dominant_freq_idx]
            
            # 频谱熵
            normalized_power = positive_power / np.sum(positive_power)
            spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-10))
            
            # 频谱质心
            spectral_centroid = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
            
            return dominant_freq, spectral_entropy, spectral_centroid
        
        fourier_results = prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: fourier_analysis(x.values), raw=False
        )
        
        if not fourier_results.empty:
            results['dominant_frequency'] = fourier_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['spectral_entropy'] = fourier_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
            results['spectral_centroid'] = fourier_results.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
        
        return results
    
    def _calculate_hilbert_transform(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """计算希尔伯特变换因子"""
        try:
            from scipy.signal import hilbert
        except ImportError:
            print("SciPy未安装，跳过希尔伯特变换")
            return pd.Series(index=prices.index, dtype=float)
        
        def hilbert_phase(data):
            """计算瞬时相位"""
            if len(data) < 10:
                return np.nan
                
            # 希尔伯特变换
            analytic_signal = hilbert(data - np.mean(data))
            instantaneous_phase = np.angle(analytic_signal)
            
            # 相位变化率
            phase_diff = np.diff(instantaneous_phase)
            return np.std(phase_diff)
        
        return prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: hilbert_phase(x.values), raw=False
        )
    
    def _calculate_phase_space_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算相空间重构因子"""
        results = {}
        
        def phase_space_analysis(data):
            """相空间分析"""
            if len(data) < 20:
                return np.nan, np.nan
                
            # 相空间重构参数
            m = 3  # 嵌入维数
            tau = 1  # 时间延迟
            
            # 重构相空间
            n = len(data) - (m - 1) * tau
            if n < 10:
                return np.nan, np.nan
                
            trajectory = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    trajectory[i, j] = data[i + j * tau]
            
            # 计算轨迹长度
            trajectory_length = 0
            for i in range(n - 1):
                trajectory_length += np.linalg.norm(trajectory[i + 1] - trajectory[i])
            
            # 计算轨迹复杂度
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    distances.append(np.linalg.norm(trajectory[i] - trajectory[j]))
            
            trajectory_complexity = np.std(distances) if distances else np.nan
            
            return trajectory_length, trajectory_complexity
        
        phase_results = prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: phase_space_analysis(x.values), raw=False
        )
        
        if not phase_results.empty:
            results['trajectory_length'] = phase_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['trajectory_complexity'] = phase_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        
        return results
    
    def _calculate_multifractal_factors(self, prices: pd.Series, window: int = 100) -> Dict[str, pd.Series]:
        """计算多重分形因子"""
        results = {}
        
        def multifractal_analysis(data):
            """多重分形去趋势波动分析"""
            if len(data) < 50:
                return np.nan, np.nan
                
            # 计算累积偏差
            y = np.cumsum(data - np.mean(data))
            
            # 不同尺度
            scales = np.unique(np.logspace(1, np.log10(len(y)//4), 10).astype(int))
            
            # 不同的q值
            q_values = [-2, 0, 2]
            fluctuations = {q: [] for q in q_values}
            
            for scale in scales:
                if scale < 4:
                    continue
                    
                # 分段
                segments = len(y) // scale
                if segments < 2:
                    continue
                
                segment_fluctuations = []
                for i in range(segments):
                    start_idx = i * scale
                    end_idx = (i + 1) * scale
                    segment = y[start_idx:end_idx]
                    
                    # 去趋势
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    detrended = segment - trend
                    
                    # 计算波动
                    fluctuation = np.sqrt(np.mean(detrended**2))
                    segment_fluctuations.append(fluctuation)
                
                if segment_fluctuations:
                    for q in q_values:
                        if q == 0:
                            # 对数平均
                            avg_fluct = np.exp(np.mean(np.log(np.array(segment_fluctuations) + 1e-10)))
                        else:
                            # q阶矩
                            avg_fluct = np.mean(np.array(segment_fluctuations)**q)**(1/q)
                        fluctuations[q].append(avg_fluct)
            
            # 计算Hurst指数
            hurst_exponents = {}
            for q in q_values:
                if len(fluctuations[q]) > 2:
                    log_scales = np.log(scales[:len(fluctuations[q])])
                    log_fluct = np.log(fluctuations[q])
                    hurst, _ = np.polyfit(log_scales, log_fluct, 1)
                    hurst_exponents[q] = hurst
            
            # 多重分形谱宽度
            if len(hurst_exponents) >= 2:
                h_max = max(hurst_exponents.values())
                h_min = min(hurst_exponents.values())
                spectrum_width = h_max - h_min
            else:
                spectrum_width = np.nan
            
            # 不对称性
            if 2 in hurst_exponents and -2 in hurst_exponents and 0 in hurst_exponents:
                asymmetry = (hurst_exponents[2] - hurst_exponents[0]) - (hurst_exponents[0] - hurst_exponents[-2])
            else:
                asymmetry = np.nan
            
            return spectrum_width, asymmetry
        
        multifractal_results = prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: multifractal_analysis(x.values), raw=False
        )
        
        if not multifractal_results.empty:
            results['multifractal_spectrum_width'] = multifractal_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['multifractal_asymmetry'] = multifractal_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        
        return results
    
    def _calculate_nonlinear_dynamics_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算非线性动力学因子"""
        results = {}
        
        def nonlinear_analysis(data):
            """非线性动力学分析"""
            if len(data) < 20:
                return np.nan, np.nan, np.nan
                
            returns = np.diff(np.log(data + 1e-10))
            
            # BDS统计量（简化版）
            def bds_statistic(x, m=2, eps=None):
                if eps is None:
                    eps = np.std(x)
                
                n = len(x)
                c_m = 0
                c_1 = 0
                
                # 计算相关积分
                for i in range(n - m + 1):
                    for j in range(i + 1, n - m + 1):
                        # m维距离
                        dist_m = max(abs(x[i + k] - x[j + k]) for k in range(m))
                        if dist_m < eps:
                            c_m += 1
                        
                        # 1维距离
                        if abs(x[i] - x[j]) < eps:
                            c_1 += 1
                
                n_pairs_m = (n - m + 1) * (n - m) / 2
                n_pairs_1 = n * (n - 1) / 2
                
                c_m = c_m / n_pairs_m if n_pairs_m > 0 else 0
                c_1 = c_1 / n_pairs_1 if n_pairs_1 > 0 else 0
                
                # BDS统计量
                if c_1 > 0:
                    bds = (c_m - c_1**m) / np.sqrt(c_1**(2*m))
                else:
                    bds = np.nan
                
                return bds
            
            bds_stat = bds_statistic(returns)
            
            # 递归图分析（简化）
            def recurrence_analysis(x, eps=None):
                if eps is None:
                    eps = 0.1 * np.std(x)
                
                n = len(x)
                recurrence_rate = 0
                determinism = 0
                
                # 构建递归矩阵
                for i in range(n):
                    for j in range(n):
                        if abs(x[i] - x[j]) < eps:
                            recurrence_rate += 1
                
                recurrence_rate = recurrence_rate / (n * n)
                
                # 简化的确定性度量
                diagonal_lines = 0
                for i in range(n - 2):
                    for j in range(n - 2):
                        if (abs(x[i] - x[j]) < eps and 
                            abs(x[i + 1] - x[j + 1]) < eps and 
                            abs(x[i + 2] - x[j + 2]) < eps):
                            diagonal_lines += 1
                
                determinism = diagonal_lines / (n * n) if n > 0 else 0
                
                return recurrence_rate, determinism
            
            rec_rate, determ = recurrence_analysis(returns)
            
            return bds_stat, rec_rate, determ
        
        nonlinear_results = prices.rolling(window=window, min_periods=window//2).apply(
            lambda x: nonlinear_analysis(x.values), raw=False
        )
        
        if not nonlinear_results.empty:
            results['bds_statistic'] = nonlinear_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['recurrence_rate'] = nonlinear_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
            results['determinism'] = nonlinear_results.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
        
        return results

    def calculate_statistical_factors(self, price_data: pd.DataFrame,
                                    volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算高级统计因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            
        Returns:
            统计因子字典
        """
        results = {}
        
        if 'close' not in price_data.columns:
            return results
            
        close = price_data['close']
        
        try:
            # 1. 高阶矩因子
            results.update(self._calculate_higher_moments(close))
            
            # 2. 分布拟合因子
            results.update(self._calculate_distribution_factors(close))
            
            # 3. 极值理论因子
            results.update(self._calculate_extreme_value_factors(close))
            
            # 4. 协整因子
            if len(price_data.columns) > 1:
                results.update(self._calculate_cointegration_factors(price_data))
            
            # 5. 因果关系因子
            results.update(self._calculate_causality_factors(close, volume_data))
            
            # 6. 状态空间因子
            results.update(self._calculate_state_space_factors(close))
            
            # 7. 贝叶斯因子
            results.update(self._calculate_bayesian_factors(close))
            
            # 8. 信息论因子
            results.update(self._calculate_information_theory_factors(close))
            
        except Exception as e:
            print(f"统计因子计算失败: {e}")
            
        return results
    
    def _calculate_higher_moments(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算高阶矩因子"""
        results = {}
        
        returns = prices.pct_change().dropna()
        
        # 滚动计算高阶矩
        results['skewness'] = returns.rolling(window=window).skew()
        results['kurtosis'] = returns.rolling(window=window).kurt()
        
        # 超额峰度
        results['excess_kurtosis'] = results['kurtosis'] - 3
        
        # Jarque-Bera统计量
        def jarque_bera_stat(x):
            if len(x) < 6:
                return np.nan
            n = len(x)
            s = x.skew()
            k = x.kurt()
            jb = n/6 * (s**2 + (k-3)**2/4)
            return jb
        
        results['jarque_bera'] = returns.rolling(window=window).apply(jarque_bera_stat)
        
        # 五阶矩和六阶矩
        def higher_moment(x, moment):
            if len(x) < 10:
                return np.nan
            centered = x - x.mean()
            return np.mean(centered**moment) / (x.std()**moment)
        
        results['fifth_moment'] = returns.rolling(window=window).apply(lambda x: higher_moment(x, 5))
        results['sixth_moment'] = returns.rolling(window=window).apply(lambda x: higher_moment(x, 6))
        
        return results
    
    def _calculate_distribution_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算分布拟合因子"""
        results = {}
        
        returns = prices.pct_change().dropna()
        
        def distribution_fit_quality(x):
            """分布拟合质量"""
            if len(x) < 20:
                return np.nan, np.nan, np.nan
                
            try:
                from scipy import stats as scipy_stats
                
                # 正态分布拟合
                normal_params = scipy_stats.norm.fit(x)
                normal_ks = scipy_stats.kstest(x, lambda y: scipy_stats.norm.cdf(y, *normal_params))[0]
                
                # t分布拟合
                t_params = scipy_stats.t.fit(x)
                t_ks = scipy_stats.kstest(x, lambda y: scipy_stats.t.cdf(y, *t_params))[0]
                
                # 拉普拉斯分布拟合
                laplace_params = scipy_stats.laplace.fit(x)
                laplace_ks = scipy_stats.kstest(x, lambda y: scipy_stats.laplace.cdf(y, *laplace_params))[0]
                
                return normal_ks, t_ks, laplace_ks
                
            except:
                return np.nan, np.nan, np.nan
        
        dist_results = returns.rolling(window=window, min_periods=window//2).apply(
            lambda x: distribution_fit_quality(x.values), raw=False
        )
        
        if not dist_results.empty:
            results['normal_fit_quality'] = dist_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['t_fit_quality'] = dist_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
            results['laplace_fit_quality'] = dist_results.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
        
        return results
    
    def _calculate_extreme_value_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算极值理论因子"""
        results = {}
        
        returns = prices.pct_change().dropna()
        
        def extreme_value_analysis(x, threshold_percentile=95):
            """极值分析"""
            if len(x) < 20:
                return np.nan, np.nan, np.nan
                
            # 阈值
            threshold = np.percentile(np.abs(x), threshold_percentile)
            
            # 超过阈值的观测
            exceedances = x[np.abs(x) > threshold]
            
            if len(exceedances) < 5:
                return np.nan, np.nan, np.nan
            
            # 极值指数（Hill估计量）
            sorted_exc = np.sort(np.abs(exceedances))[::-1]
            k = len(sorted_exc) // 2
            
            if k > 0:
                hill_estimator = np.mean(np.log(sorted_exc[:k])) - np.log(sorted_exc[k])
            else:
                hill_estimator = np.nan
            
            # 极值频率
            extreme_frequency = len(exceedances) / len(x)
            
            # 最大值与平均值比率
            max_to_mean_ratio = np.max(np.abs(x)) / (np.mean(np.abs(x)) + 1e-10)
            
            return hill_estimator, extreme_frequency, max_to_mean_ratio
        
        extreme_results = returns.rolling(window=window, min_periods=window//2).apply(
            lambda x: extreme_value_analysis(x.values), raw=False
        )
        
        if not extreme_results.empty:
            results['hill_estimator'] = extreme_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['extreme_frequency'] = extreme_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
            results['max_to_mean_ratio'] = extreme_results.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
        
        return results
    
    def _calculate_cointegration_factors(self, price_data: pd.DataFrame, window: int = 100) -> Dict[str, pd.Series]:
        """计算协整因子"""
        results = {}
        
        if len(price_data.columns) < 2:
            return results
        
        # 选择前两个价格序列进行协整分析
        price1 = price_data.iloc[:, 0]
        price2 = price_data.iloc[:, 1]
        
        def cointegration_analysis(p1, p2):
            """协整分析"""
            if len(p1) < 30 or len(p2) < 30:
                return np.nan, np.nan
                
            try:
                # 简单的协整检验
                # OLS回归
                X = np.column_stack([np.ones(len(p1)), p1])
                beta = np.linalg.lstsq(X, p2, rcond=None)[0]
                
                # 残差
                residuals = p2 - (beta[0] + beta[1] * p1)
                
                # ADF检验统计量（简化）
                residuals_diff = np.diff(residuals)
                residuals_lag = residuals[:-1]
                
                if len(residuals_diff) > 0 and len(residuals_lag) > 0:
                    # 简单回归
                    X_adf = np.column_stack([np.ones(len(residuals_lag)), residuals_lag])
                    adf_coef = np.linalg.lstsq(X_adf, residuals_diff, rcond=None)[0]
                    adf_stat = adf_coef[1]
                else:
                    adf_stat = np.nan
                
                # 半衰期
                if adf_coef[1] < 0:
                    half_life = -np.log(2) / adf_coef[1]
                else:
                    half_life = np.nan
                
                return adf_stat, half_life
                
            except:
                return np.nan, np.nan
        
        # 滚动协整分析
        cointegration_results = []
        for i in range(window, len(price1)):
            p1_window = price1.iloc[i-window:i]
            p2_window = price2.iloc[i-window:i]
            adf_stat, half_life = cointegration_analysis(p1_window.values, p2_window.values)
            cointegration_results.append((adf_stat, half_life))
        
        # 创建结果序列
        result_index = price1.index[window:]
        if cointegration_results:
            adf_stats = [r[0] for r in cointegration_results]
            half_lives = [r[1] for r in cointegration_results]
            
            results['cointegration_adf'] = pd.Series(adf_stats, index=result_index)
            results['cointegration_half_life'] = pd.Series(half_lives, index=result_index)
        
        return results
    
    def _calculate_causality_factors(self, prices: pd.Series, volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """计算因果关系因子"""
        results = {}
        
        if volume_data is None or 'volume' not in volume_data.columns:
            return results
        
        volume = volume_data['volume']
        price_returns = prices.pct_change().dropna()
        volume_changes = volume.pct_change().dropna()
        
        # 对齐数据
        common_index = price_returns.index.intersection(volume_changes.index)
        price_returns = price_returns.loc[common_index]
        volume_changes = volume_changes.loc[common_index]
        
        def granger_causality_test(x, y, max_lag=5, window=50):
            """格兰杰因果检验（简化版）"""
            if len(x) < window or len(y) < window:
                return np.nan
                
            # 滚动格兰杰因果检验
            causality_stats = []
            
            for i in range(window, len(x)):
                x_window = x.iloc[i-window:i].values
                y_window = y.iloc[i-window:i].values
                
                try:
                    # 构建滞后矩阵
                    n = len(x_window) - max_lag
                    if n < 10:
                        causality_stats.append(np.nan)
                        continue
                    
                    # 受限模型：y只依赖于自己的滞后
                    Y = y_window[max_lag:]
                    X_restricted = np.column_stack([
                        np.ones(n),
                        *[y_window[max_lag-j-1:-j-1] for j in range(max_lag)]
                    ])
                    
                    # 非受限模型：y依赖于自己和x的滞后
                    X_unrestricted = np.column_stack([
                        X_restricted,
                        *[x_window[max_lag-j-1:-j-1] for j in range(max_lag)]
                    ])
                    
                    # 回归
                    beta_restricted = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
                    beta_unrestricted = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
                    
                    # 残差平方和
                    rss_restricted = np.sum((Y - X_restricted @ beta_restricted)**2)
                    rss_unrestricted = np.sum((Y - X_unrestricted @ beta_unrestricted)**2)
                    
                    # F统计量
                    if rss_unrestricted > 0:
                        f_stat = ((rss_restricted - rss_unrestricted) / max_lag) / (rss_unrestricted / (n - 2*max_lag - 1))
                    else:
                        f_stat = np.nan
                    
                    causality_stats.append(f_stat)
                    
                except:
                    causality_stats.append(np.nan)
            
            return pd.Series(causality_stats, index=x.index[window:])
        
        # 价格对成交量的因果关系
        results['price_to_volume_causality'] = granger_causality_test(price_returns, volume_changes)
        
        # 成交量对价格的因果关系
        results['volume_to_price_causality'] = granger_causality_test(volume_changes, price_returns)
        
        return results
    
    def _calculate_state_space_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算状态空间因子"""
        results = {}
        
        returns = prices.pct_change().dropna()
        
        def kalman_filter_analysis(x):
            """卡尔曼滤波分析"""
            if len(x) < 20:
                return np.nan, np.nan
                
            # 简化的卡尔曼滤波
            # 状态：潜在收益率
            # 观测：实际收益率
            
            n = len(x)
            
            # 初始化
            state = 0  # 初始状态
            P = 1  # 初始协方差
            Q = 0.01  # 过程噪声
            R = 0.1  # 观测噪声
            
            states = []
            innovations = []
            
            for i in range(n):
                # 预测步骤
                state_pred = state
                P_pred = P + Q
                
                # 更新步骤
                K = P_pred / (P_pred + R)  # 卡尔曼增益
                innovation = x.iloc[i] - state_pred
                state = state_pred + K * innovation
                P = (1 - K) * P_pred
                
                states.append(state)
                innovations.append(innovation)
            
            # 状态平滑度
            state_smoothness = np.std(np.diff(states)) if len(states) > 1 else np.nan
            
            # 创新序列方差
            innovation_variance = np.var(innovations) if innovations else np.nan
            
            return state_smoothness, innovation_variance
        
        kalman_results = returns.rolling(window=window, min_periods=window//2).apply(
            lambda x: kalman_filter_analysis(x), raw=False
        )
        
        if not kalman_results.empty:
            results['state_smoothness'] = kalman_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['innovation_variance'] = kalman_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        
        return results
    
    def _calculate_bayesian_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算贝叶斯因子"""
        results = {}
        
        returns = prices.pct_change().dropna()
        
        def bayesian_analysis(x):
            """贝叶斯分析"""
            if len(x) < 20:
                return np.nan, np.nan
                
            # 贝叶斯变点检测（简化）
            n = len(x)
            
            # 计算不同变点位置的后验概率
            log_likelihoods = []
            
            for changepoint in range(5, n-5):
                # 分段计算似然
                segment1 = x.iloc[:changepoint]
                segment2 = x.iloc[changepoint:]
                
                # 假设正态分布
                if len(segment1) > 1 and len(segment2) > 1:
                    ll1 = -0.5 * len(segment1) * np.log(2 * np.pi * segment1.var()) - 0.5 * np.sum((segment1 - segment1.mean())**2) / segment1.var()
                    ll2 = -0.5 * len(segment2) * np.log(2 * np.pi * segment2.var()) - 0.5 * np.sum((segment2 - segment2.mean())**2) / segment2.var()
                    
                    total_ll = ll1 + ll2
                else:
                    total_ll = -np.inf
                
                log_likelihoods.append(total_ll)
            
            # 最优变点
            if log_likelihoods:
                best_changepoint = np.argmax(log_likelihoods) + 5
                changepoint_strength = max(log_likelihoods) - np.mean(log_likelihoods)
            else:
                changepoint_strength = np.nan
            
            # 贝叶斯信息准则
            # 单一模型
            single_model_bic = -2 * (-0.5 * n * np.log(2 * np.pi * x.var()) - 0.5 * np.sum((x - x.mean())**2) / x.var()) + 2 * np.log(n)
            
            # 变点模型
            if log_likelihoods:
                changepoint_model_bic = -2 * max(log_likelihoods) + 4 * np.log(n)
                bic_improvement = single_model_bic - changepoint_model_bic
            else:
                bic_improvement = np.nan
            
            return changepoint_strength, bic_improvement
        
        bayesian_results = returns.rolling(window=window, min_periods=window//2).apply(
            lambda x: bayesian_analysis(x), raw=False
        )
        
        if not bayesian_results.empty:
            results['changepoint_strength'] = bayesian_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['bic_improvement'] = bayesian_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        
        return results
    
    def _calculate_information_theory_factors(self, prices: pd.Series, window: int = 50) -> Dict[str, pd.Series]:
        """计算信息论因子"""
        results = {}
        
        returns = prices.pct_change().dropna()
        
        def information_theory_analysis(x, bins=10):
            """信息论分析"""
            if len(x) < 20:
                return np.nan, np.nan, np.nan
                
            # 离散化
            hist, bin_edges = np.histogram(x, bins=bins, density=True)
            hist = hist / np.sum(hist)  # 归一化
            
            # 香农熵
            shannon_entropy = -np.sum(hist * np.log(hist + 1e-10))
            
            # 相对熵（与正态分布比较）
            # 正态分布的理论概率
            normal_mean = x.mean()
            normal_std = x.std()
            
            theoretical_probs = []
            for i in range(len(bin_edges) - 1):
                left = bin_edges[i]
                right = bin_edges[i + 1]
                
                # 正态分布在区间内的概率
                from scipy.stats import norm
                prob = norm.cdf(right, normal_mean, normal_std) - norm.cdf(left, normal_mean, normal_std)
                theoretical_probs.append(prob)
            
            theoretical_probs = np.array(theoretical_probs)
            theoretical_probs = theoretical_probs / np.sum(theoretical_probs)  # 归一化
            
            # KL散度
            kl_divergence = np.sum(hist * np.log((hist + 1e-10) / (theoretical_probs + 1e-10)))
            
            # 互信息（滞后1期）
            if len(x) > 1:
                x_lag = x.iloc[:-1].values
                x_current = x.iloc[1:].values
                
                # 二维直方图
                hist_2d, _, _ = np.histogram2d(x_lag, x_current, bins=bins, density=True)
                hist_2d = hist_2d / np.sum(hist_2d)
                
                # 边际分布
                hist_lag = np.sum(hist_2d, axis=1)
                hist_current = np.sum(hist_2d, axis=0)
                
                # 互信息
                mutual_info = 0
                for i in range(len(hist_lag)):
                    for j in range(len(hist_current)):
                        if hist_2d[i, j] > 0 and hist_lag[i] > 0 and hist_current[j] > 0:
                            mutual_info += hist_2d[i, j] * np.log(hist_2d[i, j] / (hist_lag[i] * hist_current[j]))
            else:
                mutual_info = np.nan
            
            return shannon_entropy, kl_divergence, mutual_info
        
        info_results = returns.rolling(window=window, min_periods=window//2).apply(
            lambda x: information_theory_analysis(x), raw=False
        )
        
        if not info_results.empty:
            results['shannon_entropy_returns'] = info_results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            results['kl_divergence_normal'] = info_results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
            results['mutual_information_lag1'] = info_results.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
        
        return results

    def calculate_network_factors(self, price_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        计算网络因子（需要多只股票的数据）
        
        Args:
            price_data_dict: 多只股票的价格数据字典
        
        Returns:
            网络因子字典
        """
        results = {}
        
        if not NETWORKX_AVAILABLE or len(price_data_dict) < 2:
            print("网络分析库不可用或股票数量不足")
            return results
        
        try:
            # 构建相关性矩阵
            returns_dict = {}
            for symbol, data in price_data_dict.items():
                if 'close' in data.columns:
                    returns_dict[symbol] = data['close'].pct_change()
            
            if len(returns_dict) < 2:
                return results
            
            # 创建收益率矩阵
            returns_df = pd.DataFrame(returns_dict)
            
            # 计算网络因子
            results.update(self._calculate_centrality_factors(returns_df))
            results.update(self._calculate_clustering_factors(returns_df))
            results.update(self._calculate_contagion_factors(returns_df))
            
        except Exception as e:
            print(f"网络因子计算失败: {e}")
        
        return results
    
    def _calculate_centrality_factors(self, returns_df: pd.DataFrame, window: int = 60) -> Dict[str, pd.Series]:
        """计算中心性因子"""
        results = {}
        
        # 滚动计算网络中心性
        for symbol in returns_df.columns:
            centrality_scores = []
            
            for i in range(window, len(returns_df)):
                window_returns = returns_df.iloc[i-window:i]
                
                # 计算相关性矩阵
                corr_matrix = window_returns.corr()
                
                # 创建网络图
                G = nx.Graph()
                
                # 添加边（相关性大于阈值）
                threshold = 0.3
                for i_idx, stock1 in enumerate(corr_matrix.columns):
                    for j_idx, stock2 in enumerate(corr_matrix.columns):
                        if i_idx < j_idx and abs(corr_matrix.loc[stock1, stock2]) > threshold:
                            G.add_edge(stock1, stock2, weight=abs(corr_matrix.loc[stock1, stock2]))
                
                # 计算中心性
                if symbol in G.nodes():
                    centrality = nx.degree_centrality(G)[symbol]
                else:
                    centrality = 0
                
                centrality_scores.append(centrality)
            
            # 创建因子序列
            factor_index = returns_df.index[window:]
            results[f'{symbol}_centrality'] = pd.Series(centrality_scores, index=factor_index)
        
        return results
    
    def _calculate_clustering_factors(self, returns_df: pd.DataFrame, window: int = 60) -> Dict[str, pd.Series]:
        """计算聚类因子"""
        results = {}
        
        clustering_scores = []
        
        for i in range(window, len(returns_df)):
            window_returns = returns_df.iloc[i-window:i]
            corr_matrix = window_returns.corr()
            
            # 创建网络图
            G = nx.Graph()
            threshold = 0.3
            
            for i_idx, stock1 in enumerate(corr_matrix.columns):
                for j_idx, stock2 in enumerate(corr_matrix.columns):
                    if i_idx < j_idx and abs(corr_matrix.loc[stock1, stock2]) > threshold:
                        G.add_edge(stock1, stock2, weight=abs(corr_matrix.loc[stock1, stock2]))
            
            # 计算聚类系数
            if len(G.nodes()) > 0:
                clustering = nx.average_clustering(G)
            else:
                clustering = 0
            
            clustering_scores.append(clustering)
        
        factor_index = returns_df.index[window:]
        results['network_clustering'] = pd.Series(clustering_scores, index=factor_index)
        
        return results
    
    def _calculate_contagion_factors(self, returns_df: pd.DataFrame, window: int = 60) -> Dict[str, pd.Series]:
        """计算传染因子"""
        results = {}
        
        # 计算滚动相关性的变化
        for symbol in returns_df.columns:
            contagion_scores = []
            
            for i in range(window * 2, len(returns_df)):
                # 当前窗口和历史窗口的相关性
                current_window = returns_df.iloc[i-window:i]
                past_window = returns_df.iloc[i-window*2:i-window]
                
                current_corr = current_window[symbol].corr(current_window.drop(columns=[symbol]).mean(axis=1))
                past_corr = past_window[symbol].corr(past_window.drop(columns=[symbol]).mean(axis=1))
                
                # 传染效应：相关性的增加
                contagion = current_corr - past_corr if not (np.isnan(current_corr) or np.isnan(past_corr)) else 0
                contagion_scores.append(contagion)
            
            factor_index = returns_df.index[window * 2:]
            results[f'{symbol}_contagion'] = pd.Series(contagion_scores, index=factor_index)
        
        return results
    
    def calculate_alternative_data_factors(self, price_data: pd.DataFrame,
                                         news_sentiment: pd.Series = None,
                                         social_media_sentiment: pd.Series = None,
                                         search_volume: pd.Series = None) -> Dict[str, pd.Series]:
        """
        计算另类数据因子
        
        Args:
            price_data: 价格数据
            news_sentiment: 新闻情绪数据
            social_media_sentiment: 社交媒体情绪数据
            search_volume: 搜索量数据
        
        Returns:
            另类数据因子字典
        """
        results = {}
        
        try:
            # 1. 新闻情绪因子
            if news_sentiment is not None:
                results.update(self._calculate_news_sentiment_factors(price_data, news_sentiment))
            
            # 2. 社交媒体情绪因子
            if social_media_sentiment is not None:
                results.update(self._calculate_social_sentiment_factors(price_data, social_media_sentiment))
            
            # 3. 搜索量因子
            if search_volume is not None:
                results.update(self._calculate_search_volume_factors(price_data, search_volume))
            
            # 4. 综合另类数据因子
            if len(results) > 0:
                results.update(self._calculate_composite_alternative_factors(results))
            
        except Exception as e:
            print(f"另类数据因子计算失败: {e}")
        
        return results
    
    def _calculate_news_sentiment_factors(self, price_data: pd.DataFrame, 
                                        news_sentiment: pd.Series) -> Dict[str, pd.Series]:
        """计算新闻情绪因子"""
        results = {}
        
        # 对齐数据
        aligned_sentiment = news_sentiment.reindex(price_data.index, method='ffill')
        
        # 1. 情绪动量
        results['news_sentiment_momentum'] = aligned_sentiment.rolling(5).mean()
        
        # 2. 情绪反转
        results['news_sentiment_reversal'] = -aligned_sentiment.rolling(20).mean()
        
        # 3. 情绪波动率
        results['news_sentiment_volatility'] = aligned_sentiment.rolling(10).std()
        
        # 4. 情绪与价格的背离
        if 'close' in price_data.columns:
            price_momentum = price_data['close'].pct_change(5)
            sentiment_momentum = aligned_sentiment.rolling(5).mean()
            results['news_price_divergence'] = sentiment_momentum - price_momentum
        
        return results
    
    def _calculate_social_sentiment_factors(self, price_data: pd.DataFrame,
                                          social_sentiment: pd.Series) -> Dict[str, pd.Series]:
        """计算社交媒体情绪因子"""
        results = {}
        
        # 对齐数据
        aligned_sentiment = social_sentiment.reindex(price_data.index, method='ffill')
        
        # 1. 社交情绪强度
        results['social_sentiment_intensity'] = np.abs(aligned_sentiment)
        
        # 2. 社交情绪趋势
        results['social_sentiment_trend'] = aligned_sentiment.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # 3. 社交情绪极值
        results['social_sentiment_extreme'] = (
            (aligned_sentiment > aligned_sentiment.rolling(30).quantile(0.9)) |
            (aligned_sentiment < aligned_sentiment.rolling(30).quantile(0.1))
        ).astype(int)
        
        return results
    
    def _calculate_search_volume_factors(self, price_data: pd.DataFrame,
                                       search_volume: pd.Series) -> Dict[str, pd.Series]:
        """计算搜索量因子"""
        results = {}
        
        # 对齐数据
        aligned_search = search_volume.reindex(price_data.index, method='ffill')
        
        # 1. 搜索量异常
        search_mean = aligned_search.rolling(30).mean()
        search_std = aligned_search.rolling(30).std()
        results['search_volume_anomaly'] = (aligned_search - search_mean) / search_std
        
        # 2. 搜索量趋势
        results['search_volume_trend'] = aligned_search.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # 3. 搜索量与价格关系
        if 'close' in price_data.columns:
            price_change = price_data['close'].pct_change()
            search_change = aligned_search.pct_change()
            
            # 滚动相关性
            results['search_price_correlation'] = price_change.rolling(20).corr(search_change)
        
        return results
    
    def _calculate_composite_alternative_factors(self, alt_factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """计算综合另类数据因子"""
        results = {}
        
        # 将所有另类数据因子标准化后综合
        factor_df = pd.DataFrame(alt_factors)
        
        if not factor_df.empty:
            # 标准化
            standardized_factors = factor_df.apply(lambda x: (x - x.mean()) / x.std())
            
            # 综合因子
            results['composite_alternative_factor'] = standardized_factors.mean(axis=1)
            
            # 另类数据强度
            results['alternative_data_strength'] = standardized_factors.abs().mean(axis=1)
        
        return results
    
    def calculate_all_advanced_factors(self, price_data: pd.DataFrame,
                                     volume_data: pd.DataFrame = None,
                                     additional_data: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        计算所有高级因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            additional_data: 额外数据（新闻情绪、社交媒体情绪等）
        
        Returns:
            所有高级因子字典
        """
        all_factors = {}
        
        print("开始计算高级因子...")
        
        # 1. 高频因子
        hf_factors = self.calculate_high_frequency_factors(price_data, volume_data)
        all_factors.update(hf_factors)
        
        # 2. 情绪因子
        sentiment_factors = self.calculate_sentiment_factors(price_data, volume_data)
        all_factors.update(sentiment_factors)
        
        # 3. 另类数据因子
        if additional_data:
            alt_factors = self.calculate_alternative_data_factors(
                price_data,
                additional_data.get('news_sentiment'),
                additional_data.get('social_sentiment'),
                additional_data.get('search_volume')
            )
            all_factors.update(alt_factors)
        
        # 移除包含过多NaN的因子
        cleaned_factors = {}
        for name, factor in all_factors.items():
            if isinstance(factor, pd.Series):
                valid_ratio = factor.notna().sum() / len(factor)
                if valid_ratio >= 0.1:  # 至少10%的数据有效
                    cleaned_factors[name] = factor
        
        print(f"高级因子计算完成，共生成 {len(cleaned_factors)} 个有效因子")
        return cleaned_factors
    
    def generate_factor_report(self, factors: Dict[str, pd.Series]) -> str:
        """生成高级因子分析报告"""
        report = "=== 高级因子分析报告 ===\n\n"
        
        if not factors:
            return report + "无有效高级因子数据\n"
        
        # 统计信息
        report += f"总因子数量: {len(factors)}\n"
        
        # 按类别分组
        categories = {
            '高频因子': ['price_jump', 'realized_', 'hf_', 'amihud', 'vwap', 'microstructure'],
            '情绪因子': ['vix_like', 'fear_', 'greed_', 'sentiment_', 'stress', 'herding', 'overreaction'],
            '网络因子': ['centrality', 'clustering', 'contagion'],
            '另类数据因子': ['news_', 'social_', 'search_', 'alternative_']
        }
        
        for category, keywords in categories.items():
            category_factors = [f for f in factors.keys() if any(kw in f for kw in keywords)]
            if category_factors:
                report += f"\n{category} ({len(category_factors)}个):\n"
                for factor in category_factors[:5]:  # 显示前5个
                    latest_value = factors[factor].dropna().iloc[-1] if not factors[factor].dropna().empty else "N/A"
                    report += f"  - {factor}: {latest_value:.4f}\n"
        
        # 数据质量评估
        valid_ratios = []
        for factor in factors.values():
            if isinstance(factor, pd.Series):
                valid_ratio = factor.notna().sum() / len(factor)
                valid_ratios.append(valid_ratio)
        
        if valid_ratios:
            avg_valid_ratio = np.mean(valid_ratios)
            report += f"\n平均数据完整度: {avg_valid_ratio:.2%}\n"
        
        return report

def main():
    """示例用法"""
    print("高级因子计算器示例")
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_periods = len(dates)
    
    np.random.seed(42)
    
    # 模拟价格数据
    price_data = pd.DataFrame({
        'open': np.random.uniform(95, 105, n_periods),
        'high': np.random.uniform(100, 110, n_periods),
        'low': np.random.uniform(90, 100, n_periods),
        'close': np.random.uniform(95, 105, n_periods),
    }, index=dates)
    
    # 模拟成交量数据
    volume_data = pd.DataFrame({
        'volume': np.random.uniform(1e6, 5e6, n_periods)
    }, index=dates)
    
    # 模拟另类数据
    additional_data = {
        'news_sentiment': pd.Series(np.random.normal(0, 1, n_periods), index=dates),
        'social_sentiment': pd.Series(np.random.normal(0, 1, n_periods), index=dates),
        'search_volume': pd.Series(np.random.uniform(50, 150, n_periods), index=dates)
    }
    
    # 计算因子
    calculator = AdvancedFactorCalculator()
    factors = calculator.calculate_all_advanced_factors(price_data, volume_data, additional_data)
    
    # 生成报告
    report = calculator.generate_factor_report(factors)
    print(report)

if __name__ == "__main__":
    main()