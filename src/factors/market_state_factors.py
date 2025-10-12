"""
市场状态因子计算模块

基于市场环境和状态计算各种市场状态因子，包括：
1. 牛熊市状态因子 - 趋势状态、动量状态等
2. 波动率状态因子 - 高低波动率状态、波动率聚集等
3. 流动性状态因子 - 市场流动性状态、流动性风险等
4. 情绪状态因子 - 恐慌状态、贪婪状态等
5. 周期状态因子 - 经济周期状态、季节性状态等
6. 风险状态因子 - 系统性风险状态、尾部风险等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MarketStateFactorCalculator:
    """市场状态因子计算器"""
    
    def __init__(self, lookback_window: int = 252):
        """
        初始化市场状态因子计算器
        
        Args:
            lookback_window: 回望窗口长度
        """
        self.lookback_window = lookback_window
        self.logger = logging.getLogger(__name__)
        
        # 状态阈值
        self.bull_bear_threshold = 0.2  # 牛熊市判断阈值
        self.volatility_threshold = 0.02  # 波动率状态阈值
        self.liquidity_threshold = 0.5  # 流动性状态阈值
        
        print("市场状态因子计算器初始化完成")
    
    def calculate_bull_bear_state_factors(self, price_data: pd.DataFrame,
                                        market_index: Optional[pd.Series] = None) -> Dict[str, pd.DataFrame]:
        """
        计算牛熊市状态因子
        
        Args:
            price_data: 价格数据，列为股票代码
            market_index: 市场指数数据
            
        Returns:
            牛熊市状态因子字典
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        if market_index is None:
            market_index = self._generate_mock_market_index(price_data.index)
        
        print("计算牛熊市状态因子...")
        
        returns = price_data.pct_change()
        market_returns = market_index.pct_change()
        
        # 1. 趋势状态因子
        trend_state = self._calculate_trend_state(market_returns)
        factors['trend_state'] = pd.DataFrame(
            np.tile(trend_state.values.reshape(-1, 1), (1, len(price_data.columns))),
            index=price_data.index,
            columns=price_data.columns
        )
        
        # 2. 动量状态因子
        momentum_state = self._calculate_momentum_state(market_returns)
        factors['momentum_state'] = pd.DataFrame(
            np.tile(momentum_state.values.reshape(-1, 1), (1, len(price_data.columns))),
            index=price_data.index,
            columns=price_data.columns
        )
        
        # 3. 牛熊市强度因子
        bull_bear_strength = self._calculate_bull_bear_strength(market_returns)
        factors['bull_bear_strength'] = pd.DataFrame(
            np.tile(bull_bear_strength.values.reshape(-1, 1), (1, len(price_data.columns))),
            index=price_data.index,
            columns=price_data.columns
        )
        
        # 4. 市场宽度因子
        market_breadth = self._calculate_market_breadth(returns)
        factors['market_breadth'] = market_breadth
        
        # 5. 上涨股票比例
        up_stock_ratio = (returns > 0).sum(axis=1) / len(price_data.columns)
        factors['up_stock_ratio'] = pd.DataFrame(
            np.tile(up_stock_ratio.values.reshape(-1, 1), (1, len(price_data.columns))),
            index=price_data.index,
            columns=price_data.columns
        )
        
        # 6. 新高新低比率
        high_low_ratio = self._calculate_new_high_low_ratio(price_data)
        factors['new_high_low_ratio'] = pd.DataFrame(
            np.tile(high_low_ratio.values.reshape(-1, 1), (1, len(price_data.columns))),
            index=price_data.index,
            columns=price_data.columns
        )
        
        return factors
    
    def _generate_mock_price_data(self, n_stocks: int = 100, n_days: int = 500) -> pd.DataFrame:
        """生成模拟价格数据"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
        stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]
        
        # 生成具有市场状态变化的价格数据
        price_data = {}
        
        # 模拟市场状态变化
        market_states = np.random.choice([0, 1], size=n_days, p=[0.3, 0.7])  # 0=熊市, 1=牛市
        
        for i, stock in enumerate(stocks):
            initial_price = np.random.uniform(10, 200)
            prices = [initial_price]
            
            for day in range(1, n_days):
                if market_states[day] == 1:  # 牛市
                    mu = np.random.normal(0.001, 0.0005)  # 正向漂移
                    sigma = np.random.uniform(0.01, 0.02)  # 较低波动率
                else:  # 熊市
                    mu = np.random.normal(-0.001, 0.0005)  # 负向漂移
                    sigma = np.random.uniform(0.02, 0.04)  # 较高波动率
                
                return_val = np.random.normal(mu, sigma)
                new_price = prices[-1] * (1 + return_val)
                prices.append(new_price)
            
            price_data[stock] = prices
        
        return pd.DataFrame(price_data, index=dates)
    
    def _generate_mock_market_index(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成模拟市场指数"""
        np.random.seed(46)
        
        initial_value = 1000
        values = [initial_value]
        
        # 模拟市场指数走势
        for i in range(1, len(dates)):
            # 添加一些趋势和周期性
            trend = 0.0002 * np.sin(i / 50)  # 长期趋势
            cycle = 0.001 * np.sin(i / 10)   # 短期周期
            noise = np.random.normal(0, 0.015)  # 随机噪声
            
            return_val = trend + cycle + noise
            new_value = values[-1] * (1 + return_val)
            values.append(new_value)
        
        return pd.Series(values, index=dates, name='market_index')
    
    def _calculate_trend_state(self, market_returns: pd.Series, window: int = 50) -> pd.Series:
        """计算趋势状态"""
        # 使用移动平均线判断趋势
        ma_short = market_returns.rolling(window=window//2).mean()
        ma_long = market_returns.rolling(window=window).mean()
        
        # 趋势状态：1=上升趋势，0=下降趋势
        trend_state = (ma_short > ma_long).astype(int)
        
        return trend_state
    
    def _calculate_momentum_state(self, market_returns: pd.Series, window: int = 21) -> pd.Series:
        """计算动量状态"""
        # 计算滚动动量
        momentum = market_returns.rolling(window=window).sum()
        
        # 动量状态：基于动量的分位数
        momentum_percentile = momentum.rolling(window=252).rank(pct=True)
        
        # 将动量状态分为5个等级
        momentum_state = pd.cut(momentum_percentile, bins=5, labels=[0, 1, 2, 3, 4]).astype(float)
        
        return momentum_state
    
    def _calculate_bull_bear_strength(self, market_returns: pd.Series, window: int = 63) -> pd.Series:
        """计算牛熊市强度"""
        # 计算滚动收益率
        rolling_returns = market_returns.rolling(window=window).sum()
        
        # 计算强度（标准化）
        rolling_std = market_returns.rolling(window=window).std()
        strength = rolling_returns / rolling_std
        
        return strength.fillna(0)
    
    def _calculate_market_breadth(self, returns: pd.DataFrame) -> pd.DataFrame:
        """计算市场宽度"""
        # 上涨股票数量 - 下跌股票数量
        up_stocks = (returns > 0).sum(axis=1)
        down_stocks = (returns < 0).sum(axis=1)
        breadth = (up_stocks - down_stocks) / len(returns.columns)
        
        # 广播到所有股票
        market_breadth = pd.DataFrame(
            np.tile(breadth.values.reshape(-1, 1), (1, len(returns.columns))),
            index=returns.index,
            columns=returns.columns
        )
        
        return market_breadth
    
    def _calculate_new_high_low_ratio(self, price_data: pd.DataFrame, window: int = 252) -> pd.Series:
        """计算新高新低比率"""
        new_highs = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        new_lows = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        
        for i in range(window, len(price_data)):
            window_data = price_data.iloc[i-window:i+1]
            current_price = price_data.iloc[i]
            
            # 新高：当前价格是窗口期内最高价
            is_new_high = current_price == window_data.max()
            new_highs.iloc[i] = is_new_high
            
            # 新低：当前价格是窗口期内最低价
            is_new_low = current_price == window_data.min()
            new_lows.iloc[i] = is_new_low
        
        # 计算新高新低比率
        new_high_count = new_highs.sum(axis=1)
        new_low_count = new_lows.sum(axis=1)
        
        # 避免除零
        ratio = new_high_count / (new_low_count + 1e-6)
        
        return ratio
    
    def calculate_volatility_state_factors(self, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        计算波动率状态因子
        
        Args:
            price_data: 价格数据
            
        Returns:
            波动率状态因子字典
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        print("计算波动率状态因子...")
        
        returns = price_data.pct_change()
        
        # 1. 波动率状态
        volatility = returns.rolling(window=21).std()
        vol_state = self._calculate_volatility_state(volatility)
        factors['volatility_state'] = vol_state
        
        # 2. 波动率聚集状态
        vol_clustering = self._calculate_volatility_clustering(returns)
        factors['volatility_clustering'] = vol_clustering
        
        # 3. 波动率突破状态
        vol_breakout = self._calculate_volatility_breakout(volatility)
        factors['volatility_breakout'] = vol_breakout
        
        # 4. 相对波动率状态
        relative_vol = self._calculate_relative_volatility_state(volatility)
        factors['relative_volatility_state'] = relative_vol
        
        # 5. 波动率趋势状态
        vol_trend = self._calculate_volatility_trend_state(volatility)
        factors['volatility_trend_state'] = vol_trend
        
        return factors
    
    def _calculate_volatility_state(self, volatility: pd.DataFrame) -> pd.DataFrame:
        """计算波动率状态"""
        # 基于历史分位数判断波动率状态
        vol_percentile = volatility.rolling(window=252).rank(pct=True)
        
        # 波动率状态：0=低波动，1=中波动，2=高波动
        vol_state = pd.DataFrame(index=volatility.index, columns=volatility.columns)
        
        vol_state[vol_percentile <= 0.33] = 0  # 低波动
        vol_state[(vol_percentile > 0.33) & (vol_percentile <= 0.67)] = 1  # 中波动
        vol_state[vol_percentile > 0.67] = 2  # 高波动
        
        return vol_state.fillna(1)
    
    def _calculate_volatility_clustering(self, returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """计算波动率聚集状态"""
        # 计算滚动波动率
        volatility = returns.rolling(window=window).std()
        
        # 波动率的波动率（波动率聚集指标）
        vol_of_vol = volatility.rolling(window=window).std()
        
        # 标准化
        vol_clustering = vol_of_vol.rolling(window=252).rank(pct=True)
        
        return vol_clustering.fillna(0.5)
    
    def _calculate_volatility_breakout(self, volatility: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """计算波动率突破状态"""
        # 计算波动率的移动平均和标准差
        vol_ma = volatility.rolling(window=window).mean()
        vol_std = volatility.rolling(window=window).std()
        
        # 波动率突破：当前波动率超过均值+2倍标准差
        vol_breakout = (volatility > (vol_ma + 2 * vol_std)).astype(int)
        
        return vol_breakout.fillna(0)
    
    def _calculate_relative_volatility_state(self, volatility: pd.DataFrame) -> pd.DataFrame:
        """计算相对波动率状态"""
        # 计算每日的截面波动率中位数
        daily_vol_median = volatility.median(axis=1)
        
        # 相对波动率
        relative_vol = volatility.divide(daily_vol_median, axis=0)
        
        # 相对波动率状态
        relative_vol_state = relative_vol.rolling(window=252).rank(pct=True)
        
        return relative_vol_state.fillna(0.5)
    
    def _calculate_volatility_trend_state(self, volatility: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """计算波动率趋势状态"""
        # 波动率的短期和长期移动平均
        vol_ma_short = volatility.rolling(window=window).mean()
        vol_ma_long = volatility.rolling(window=window*2).mean()
        
        # 波动率趋势状态：1=上升，0=下降
        vol_trend_state = (vol_ma_short > vol_ma_long).astype(int)
        
        return vol_trend_state.fillna(0)
    
    def calculate_liquidity_state_factors(self, price_data: pd.DataFrame,
                                        volume_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        计算流动性状态因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            
        Returns:
            流动性状态因子字典
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        if volume_data is None:
            volume_data = self._generate_mock_volume_data(price_data.index, price_data.columns)
        
        print("计算流动性状态因子...")
        
        returns = price_data.pct_change()
        
        # 1. 流动性状态
        liquidity_state = self._calculate_liquidity_state(volume_data, returns)
        factors['liquidity_state'] = liquidity_state
        
        # 2. 流动性风险状态
        liquidity_risk = self._calculate_liquidity_risk_state(volume_data, returns)
        factors['liquidity_risk_state'] = liquidity_risk
        
        # 3. 市场深度状态
        market_depth = self._calculate_market_depth_state(volume_data, price_data)
        factors['market_depth_state'] = market_depth
        
        # 4. 流动性冲击状态
        liquidity_shock = self._calculate_liquidity_shock_state(volume_data)
        factors['liquidity_shock_state'] = liquidity_shock
        
        return factors
    
    def _generate_mock_volume_data(self, dates: pd.DatetimeIndex, stocks: List[str]) -> pd.DataFrame:
        """生成模拟成交量数据"""
        np.random.seed(47)
        
        volume_data = {}
        for stock in stocks:
            base_volume = np.random.uniform(1e5, 1e7)
            volumes = np.random.lognormal(np.log(base_volume), 0.5, len(dates))
            volume_data[stock] = volumes
        
        return pd.DataFrame(volume_data, index=dates)
    
    def _calculate_liquidity_state(self, volume_data: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """计算流动性状态"""
        # 使用成交量和价格变化计算流动性指标
        # Amihud流动性指标：|return| / volume
        amihud_illiquidity = returns.abs() / (volume_data + 1e-6)
        
        # 流动性状态（流动性越高，状态值越大）
        liquidity = 1 / (amihud_illiquidity + 1e-6)
        liquidity_state = liquidity.rolling(window=252).rank(pct=True)
        
        return liquidity_state.fillna(0.5)
    
    def _calculate_liquidity_risk_state(self, volume_data: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """计算流动性风险状态"""
        # 流动性风险：成交量波动率
        volume_volatility = volume_data.rolling(window=21).std()
        
        # 标准化流动性风险
        liquidity_risk_state = volume_volatility.rolling(window=252).rank(pct=True)
        
        return liquidity_risk_state.fillna(0.5)
    
    def _calculate_market_depth_state(self, volume_data: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """计算市场深度状态"""
        # 市场深度：成交量/价格变化幅度
        price_range = price_data.rolling(window=5).max() - price_data.rolling(window=5).min()
        market_depth = volume_data / (price_range + 1e-6)
        
        # 市场深度状态
        depth_state = market_depth.rolling(window=252).rank(pct=True)
        
        return depth_state.fillna(0.5)
    
    def _calculate_liquidity_shock_state(self, volume_data: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """计算流动性冲击状态"""
        # 成交量异常变化
        volume_ma = volume_data.rolling(window=window).mean()
        volume_std = volume_data.rolling(window=window).std()
        
        # 流动性冲击：成交量超过均值+2倍标准差
        liquidity_shock = (volume_data > (volume_ma + 2 * volume_std)).astype(int)
        
        return liquidity_shock.fillna(0)
    
    def calculate_risk_state_factors(self, price_data: pd.DataFrame,
                                   market_index: Optional[pd.Series] = None) -> Dict[str, pd.DataFrame]:
        """
        计算风险状态因子
        
        Args:
            price_data: 价格数据
            market_index: 市场指数
            
        Returns:
            风险状态因子字典
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        if market_index is None:
            market_index = self._generate_mock_market_index(price_data.index)
        
        print("计算风险状态因子...")
        
        returns = price_data.pct_change()
        market_returns = market_index.pct_change()
        
        # 1. 系统性风险状态
        systematic_risk = self._calculate_systematic_risk_state(returns, market_returns)
        factors['systematic_risk_state'] = systematic_risk
        
        # 2. 尾部风险状态
        tail_risk = self._calculate_tail_risk_state(returns)
        factors['tail_risk_state'] = tail_risk
        
        # 3. 下行风险状态
        downside_risk = self._calculate_downside_risk_state(returns)
        factors['downside_risk_state'] = downside_risk
        
        # 4. 风险传染状态
        contagion_risk = self._calculate_contagion_risk_state(returns)
        factors['contagion_risk_state'] = contagion_risk
        
        return factors
    
    def _calculate_systematic_risk_state(self, returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
        """计算系统性风险状态"""
        # 计算Beta
        window = 63
        betas = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            window_market = market_returns.iloc[i-window:i]
            
            for stock in returns.columns:
                stock_returns = window_returns[stock].dropna()
                aligned_market = window_market.loc[stock_returns.index]
                
                if len(stock_returns) > 10 and len(aligned_market) > 10:
                    covariance = np.cov(stock_returns, aligned_market)[0, 1]
                    market_variance = np.var(aligned_market)
                    if market_variance > 1e-6:
                        beta = covariance / market_variance
                        betas.iloc[i, betas.columns.get_loc(stock)] = beta
        
        # 系统性风险状态：基于Beta的分位数
        systematic_risk_state = betas.rolling(window=252).rank(pct=True)
        
        return systematic_risk_state.fillna(0.5)
    
    def _calculate_tail_risk_state(self, returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """计算尾部风险状态"""
        # 计算VaR (5%分位数)
        var_5 = returns.rolling(window=window).quantile(0.05)
        
        # 尾部风险状态：基于VaR的排名
        tail_risk_state = var_5.rolling(window=252).rank(pct=True)
        
        return tail_risk_state.fillna(0.5)
    
    def _calculate_downside_risk_state(self, returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """计算下行风险状态"""
        # 下行偏差
        downside_returns = returns.where(returns < 0, 0)
        downside_deviation = downside_returns.rolling(window=window).std()
        
        # 下行风险状态
        downside_risk_state = downside_deviation.rolling(window=252).rank(pct=True)
        
        return downside_risk_state.fillna(0.5)
    
    def _calculate_contagion_risk_state(self, returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """计算风险传染状态"""
        # 计算滚动相关性
        rolling_corr = returns.rolling(window=window).corr()
        
        # 平均相关性（风险传染指标）
        contagion_risk = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(window, len(returns)):
            corr_matrix = returns.iloc[i-window:i].corr()
            
            for stock in returns.columns:
                if stock in corr_matrix.columns:
                    # 该股票与其他股票的平均相关性
                    other_stocks = [s for s in corr_matrix.columns if s != stock]
                    if other_stocks:
                        avg_corr = corr_matrix.loc[stock, other_stocks].mean()
                        contagion_risk.iloc[i, contagion_risk.columns.get_loc(stock)] = avg_corr
        
        # 风险传染状态
        contagion_risk_state = contagion_risk.rolling(window=252).rank(pct=True)
        
        return contagion_risk_state.fillna(0.5)
    
    def calculate_all_factors(self, price_data: Optional[pd.DataFrame] = None,
                            volume_data: Optional[pd.DataFrame] = None,
                            market_index: Optional[pd.Series] = None) -> Dict[str, pd.DataFrame]:
        """
        计算所有市场状态因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            market_index: 市场指数
            
        Returns:
            所有市场状态因子字典
        """
        all_factors = {}
        
        print("开始计算市场状态因子...")
        
        if price_data is None:
            price_data = self._generate_mock_price_data()
        
        # 1. 牛熊市状态因子
        bull_bear_factors = self.calculate_bull_bear_state_factors(price_data, market_index)
        all_factors.update(bull_bear_factors)
        
        # 2. 波动率状态因子
        volatility_factors = self.calculate_volatility_state_factors(price_data)
        all_factors.update(volatility_factors)
        
        # 3. 流动性状态因子
        liquidity_factors = self.calculate_liquidity_state_factors(price_data, volume_data)
        all_factors.update(liquidity_factors)
        
        # 4. 风险状态因子
        risk_factors = self.calculate_risk_state_factors(price_data, market_index)
        all_factors.update(risk_factors)
        
        print(f"市场状态因子计算完成，共生成 {len(all_factors)} 个因子")
        return all_factors

def main():
    """测试市场状态因子计算器"""
    print("测试市场状态因子计算器...")
    
    # 初始化计算器
    calculator = MarketStateFactorCalculator()
    
    # 计算因子
    factors = calculator.calculate_all_factors()
    
    print(f"成功计算 {len(factors)} 个市场状态因子")
    for name, factor in list(factors.items())[:10]:  # 只显示前10个
        if isinstance(factor, pd.DataFrame) and not factor.empty:
            print(f"{name}: 形状={factor.shape}, 均值={factor.mean().mean():.4f}")

if __name__ == "__main__":
    main()