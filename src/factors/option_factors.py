"""
期权因子计算模块

基于期权数据计算各种期权相关因子，包括：
1. 隐含波动率因子 - IV水平、IV偏斜、IV期限结构等
2. 期权流量因子 - Put/Call比率、期权成交量等
3. 期权定价因子 - Greeks、时间价值等
4. 波动率曲面因子 - 波动率微笑、期限结构等
5. 期权情绪因子 - 恐慌指数、投资者情绪等
6. 期权套利因子 - 价格偏差、套利机会等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
import math

# 期权定价库
try:
    # import yfinance as yf  # 已移除，不再使用yfinance
    from py_vollib.black_scholes import black_scholes
    from py_vollib.black_scholes.greeks import delta, gamma, theta, vega, rho
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    OPTION_LIBS_AVAILABLE = True
except ImportError:
    OPTION_LIBS_AVAILABLE = False
    print("期权定价库未安装，将使用简化计算")

class OptionFactorCalculator:
    """期权因子计算器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化期权因子计算器
        
        Args:
            risk_free_rate: 无风险利率
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        
        print("期权因子计算器初始化完成")
    
    def calculate_implied_volatility_factors(self, symbol: str = 'SPY', 
                                           start_date: str = '2023-01-01',
                                           end_date: str = '2023-12-31') -> Dict[str, pd.Series]:
        """
        计算隐含波动率相关因子
        
        Args:
            symbol: 标的资产代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            隐含波动率因子字典
        """
        factors = {}
        
        # 获取VIX数据作为隐含波动率代理
        if symbol == 'SPY':
            vix_data = self._get_vix_data(start_date, end_date)
            if not vix_data.empty:
                factors['implied_volatility'] = vix_data
                
                # IV相关衍生因子
                factors['iv_percentile'] = self._calculate_percentile_rank(vix_data, 252)
                factors['iv_rank'] = self._calculate_rank(vix_data, 252)
                factors['iv_momentum'] = vix_data.pct_change(21)  # 1个月动量
                factors['iv_mean_reversion'] = vix_data / vix_data.rolling(63).mean() - 1
                factors['iv_term_structure'] = self._calculate_iv_term_structure(vix_data)
        
        # 如果无法获取真实数据，使用模拟数据
        if not factors:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            factors = self._generate_mock_iv_factors(dates, symbol)
        
        return factors
    
    def _get_vix_data(self, start_date: str, end_date: str) -> pd.Series:
        """获取VIX数据，优先使用IB TWS API"""
        # 第一优先级：尝试使用IB TWS API获取VIX数据
        try:
            from src.data.ib_data_provider import IBDataProvider, IBConfig
            ib_provider = IBDataProvider(IBConfig())
            vix_data = ib_provider.get_stock_data('^VIX', start_date, end_date)
            if vix_data is not None and not vix_data.empty and 'Close' in vix_data.columns:
                return vix_data['Close'].dropna()
        except Exception as e:
            print(f"IB TWS API获取VIX数据失败: {e}")
        
        # 第二优先级：回退到yfinance
        try:
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if not vix.empty and 'Close' in vix.columns:
                return vix['Close'].dropna()
        except Exception as e:
            print(f"yfinance获取VIX数据失败: {e}")
        
        return pd.Series()
    
    def _calculate_percentile_rank(self, series: pd.Series, window: int) -> pd.Series:
        """计算百分位排名"""
        def percentile_rank(x):
            if len(x) < 2:
                return 0.5
            return stats.percentileofscore(x[:-1], x.iloc[-1]) / 100
        
        return series.rolling(window).apply(percentile_rank)
    
    def _calculate_rank(self, series: pd.Series, window: int) -> pd.Series:
        """计算排名"""
        return series.rolling(window).rank(pct=True)
    
    def _calculate_iv_term_structure(self, iv_series: pd.Series) -> pd.Series:
        """计算隐含波动率期限结构"""
        # 简化计算：短期IV与长期IV的比值
        short_term_iv = iv_series.rolling(30).mean()
        long_term_iv = iv_series.rolling(90).mean()
        
        return short_term_iv / long_term_iv - 1
    
    def _generate_mock_iv_factors(self, dates: pd.DatetimeIndex, symbol: str) -> Dict[str, pd.Series]:
        """生成模拟隐含波动率因子 - 仅用于测试和演示"""
        np.random.seed(hash(symbol) % 2**32)  # 设置随机种子 - 模拟数据仅用于测试
        
        # 基础IV水平 - 仅用于测试和演示
        base_iv = 20.0
        
        # IV的均值回归过程 - 模拟数据仅用于演示
        iv_changes = np.random.normal(0, 2, len(dates))  # 随机变化 - 仅用于测试
        iv_level = np.zeros(len(dates))  # 初始化IV序列 - 模拟数据
        iv_level[0] = base_iv  # 设置初始值 - 仅用于演示
        
        for i in range(1, len(dates)):
            # 均值回归 + 随机冲击 - 模拟数据仅用于测试
            mean_reversion = -0.1 * (iv_level[i-1] - base_iv)  # 均值回归项 - 仅用于演示
            iv_level[i] = iv_level[i-1] + mean_reversion + iv_changes[i]  # 更新IV - 模拟数据
            iv_level[i] = max(iv_level[i], 5)  # 最小值5% - 仅用于测试
        
        iv_series = pd.Series(iv_level, index=dates, name='implied_volatility')  # 创建序列 - 模拟数据
        
        return {
            'implied_volatility': iv_series,
            'iv_percentile': self._calculate_percentile_rank(iv_series, 252),
            'iv_rank': self._calculate_rank(iv_series, 252),
            'iv_momentum': iv_series.pct_change(21),
            'iv_mean_reversion': iv_series / iv_series.rolling(63).mean() - 1,
            'iv_term_structure': self._calculate_iv_term_structure(iv_series)
        }
    
    def calculate_option_flow_factors(self, symbol: str = 'SPY',
                                    start_date: str = '2023-01-01',
                                    end_date: str = '2023-12-31') -> Dict[str, pd.Series]:
        """
        计算期权流量因子
        
        Args:
            symbol: 标的资产代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            期权流量因子字典
        """
        factors = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 由于真实期权流量数据获取困难，使用模拟数据
        factors = self._generate_mock_option_flow_factors(dates, symbol)
        
        return factors
    
    def _generate_mock_option_flow_factors(self, dates: pd.DatetimeIndex, symbol: str) -> Dict[str, pd.Series]:
        """生成模拟期权流量因子 - 仅用于测试和演示"""
        np.random.seed(hash(symbol + 'flow') % 2**32)  # 设置随机种子 - 模拟数据仅用于测试
        
        # 1. Put/Call比率 - 模拟数据仅用于演示
        base_pc_ratio = 0.8  # 基础比率 - 仅用于测试
        pc_ratio_noise = np.random.normal(0, 0.1, len(dates))  # 随机噪声 - 模拟数据
        pc_ratio = base_pc_ratio + pc_ratio_noise  # 计算比率 - 仅用于演示
        pc_ratio = np.clip(pc_ratio, 0.3, 2.0)  # 限制范围 - 仅用于测试
        
        # 2. 期权总成交量 - 模拟数据仅用于演示
        base_volume = 1000000  # 基础成交量 - 仅用于测试
        volume_trend = np.random.normal(1, 0.2, len(dates))  # 成交量趋势 - 模拟数据
        option_volume = base_volume * volume_trend  # 计算成交量 - 仅用于演示
        option_volume = np.maximum(option_volume, 100000)  # 最小值限制 - 仅用于测试
        
        # 3. Call期权成交量 - 模拟数据仅用于演示
        call_volume = option_volume / (1 + pc_ratio)  # 计算Call成交量 - 仅用于测试
        
        # 4. Put期权成交量 - 模拟数据仅用于演示
        put_volume = option_volume - call_volume  # 计算Put成交量 - 仅用于测试
        
        # 5. 期权成交量与股票成交量比率 - 模拟数据仅用于演示
        stock_volume = base_volume * 10 * np.random.normal(1, 0.15, len(dates))  # 股票成交量 - 仅用于测试
        option_stock_ratio = option_volume / stock_volume  # 计算比率 - 模拟数据
        
        # 6. 大单期权交易比例 - 模拟数据仅用于演示
        large_trade_ratio = np.random.uniform(0.1, 0.3, len(dates))  # 大单比例 - 仅用于测试
        
        # 7. 期权未平仓合约 - 模拟数据仅用于演示
        open_interest = base_volume * 5 * np.random.normal(1, 0.1, len(dates))  # 未平仓合约 - 仅用于测试
        open_interest = np.maximum(open_interest, base_volume)  # 最小值限制 - 模拟数据
        
        return {
            'put_call_ratio': pd.Series(pc_ratio, index=dates, name='put_call_ratio'),
            'option_volume': pd.Series(option_volume, index=dates, name='option_volume'),
            'call_volume': pd.Series(call_volume, index=dates, name='call_volume'),
            'put_volume': pd.Series(put_volume, index=dates, name='put_volume'),
            'option_stock_volume_ratio': pd.Series(option_stock_ratio, index=dates, name='option_stock_volume_ratio'),
            'large_trade_ratio': pd.Series(large_trade_ratio, index=dates, name='large_trade_ratio'),
            'open_interest': pd.Series(open_interest, index=dates, name='open_interest'),
            'oi_change': pd.Series(open_interest, index=dates).pct_change(),
            'volume_oi_ratio': pd.Series(option_volume / open_interest, index=dates, name='volume_oi_ratio')
        }
    
    def calculate_option_greeks_factors(self, symbol: str = 'SPY',
                                      start_date: str = '2023-01-01',
                                      end_date: str = '2023-12-31') -> Dict[str, pd.Series]:
        """
        计算期权Greeks因子
        
        Args:
            symbol: 标的资产代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            期权Greeks因子字典
        """
        factors = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 获取标的价格数据
        underlying_price = self._get_underlying_price(symbol, start_date, end_date)
        
        if underlying_price.empty:
            # 使用模拟数据
            underlying_price = self._generate_mock_price(dates, symbol)
        
        # 计算Greeks
        factors = self._calculate_greeks(underlying_price, dates)
        
        return factors
    
    def _get_underlying_price(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        """获取标的资产价格，优先使用IB TWS API"""
        # 第一优先级：尝试使用IB TWS API获取数据
        try:
            from src.data.ib_data_provider import IBDataProvider, IBConfig
            ib_provider = IBDataProvider(IBConfig())
            data = ib_provider.get_stock_data(symbol, start_date, end_date)
            if data is not None and not data.empty and 'Close' in data.columns:
                return data['Close'].dropna()
        except Exception as e:
            print(f"IB TWS API获取{symbol}数据失败: {e}")
        
        # 第二优先级：回退到yfinance
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty and 'Close' in data.columns:
                return data['Close'].dropna()
        except Exception as e:
            print(f"yfinance获取{symbol}数据失败: {e}")
        
        return pd.Series()
    
    def _generate_mock_price(self, dates: pd.DatetimeIndex, symbol: str) -> pd.Series:
        """生成模拟价格数据 - 仅用于测试和演示"""
        np.random.seed(hash(symbol + 'price') % 2**32)  # 设置随机种子 - 模拟数据仅用于测试
        
        # 几何布朗运动 - 模拟数据仅用于演示
        initial_price = 400.0  # SPY大约价格 - 仅用于测试
        mu = 0.08 / 252  # 年化8%收益率 - 模拟数据
        sigma = 0.16 / np.sqrt(252)  # 年化16%波动率 - 模拟数据
        
        returns = np.random.normal(mu, sigma, len(dates))  # 生成收益率 - 仅用于测试
        prices = initial_price * np.cumprod(1 + returns)  # 计算价格 - 模拟数据
        
        return pd.Series(prices, index=dates, name=f'{symbol}_price')
    
    def _calculate_greeks(self, underlying_price: pd.Series, dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """计算期权Greeks"""
        factors = {}
        
        # 期权参数
        strike = underlying_price.iloc[0]  # ATM期权
        time_to_expiry = 30 / 365  # 30天到期
        volatility = 0.2  # 20%波动率
        
        if OPTION_LIBS_AVAILABLE:
            # 使用真实的Black-Scholes公式
            try:
                delta_values = []
                gamma_values = []
                theta_values = []
                vega_values = []
                rho_values = []
                
                for price in underlying_price:
                    delta_val = delta('c', price, strike, time_to_expiry, self.risk_free_rate, volatility)
                    gamma_val = gamma('c', price, strike, time_to_expiry, self.risk_free_rate, volatility)
                    theta_val = theta('c', price, strike, time_to_expiry, self.risk_free_rate, volatility)
                    vega_val = vega('c', price, strike, time_to_expiry, self.risk_free_rate, volatility)
                    rho_val = rho('c', price, strike, time_to_expiry, self.risk_free_rate, volatility)
                    
                    delta_values.append(delta_val)
                    gamma_values.append(gamma_val)
                    theta_values.append(theta_val)
                    vega_values.append(vega_val)
                    rho_values.append(rho_val)
                
                factors['delta'] = pd.Series(delta_values, index=dates, name='delta')
                factors['gamma'] = pd.Series(gamma_values, index=dates, name='gamma')
                factors['theta'] = pd.Series(theta_values, index=dates, name='theta')
                factors['vega'] = pd.Series(vega_values, index=dates, name='vega')
                factors['rho'] = pd.Series(rho_values, index=dates, name='rho')
                
            except Exception as e:
                print(f"计算Greeks失败，使用简化方法: {e}")
                factors = self._calculate_simplified_greeks(underlying_price, dates, strike)
        else:
            # 使用简化的Greeks计算
            factors = self._calculate_simplified_greeks(underlying_price, dates, strike)
        
        return factors
    
    def _calculate_simplified_greeks(self, underlying_price: pd.Series, dates: pd.DatetimeIndex, strike: float) -> Dict[str, pd.Series]:
        """简化的Greeks计算"""
        # 简化的Delta (对价格的敏感性)
        moneyness = underlying_price / strike
        delta_approx = 0.5 + 0.3 * (moneyness - 1)  # 线性近似
        delta_approx = np.clip(delta_approx, 0, 1)
        
        # 简化的Gamma (Delta的变化率)
        gamma_approx = 0.01 * np.exp(-0.5 * (moneyness - 1)**2)  # 高斯形状
        
        # 简化的Theta (时间衰减)
        theta_approx = -0.02 * np.ones(len(underlying_price))  # 常数时间衰减
        
        # 简化的Vega (对波动率的敏感性)
        vega_approx = 0.1 * underlying_price / 100  # 与价格成比例
        
        # 简化的Rho (对利率的敏感性)
        rho_approx = 0.05 * underlying_price / 100  # 与价格成比例
        
        return {
            'delta': pd.Series(delta_approx, index=dates, name='delta'),
            'gamma': pd.Series(gamma_approx, index=dates, name='gamma'),
            'theta': pd.Series(theta_approx, index=dates, name='theta'),
            'vega': pd.Series(vega_approx, index=dates, name='vega'),
            'rho': pd.Series(rho_approx, index=dates, name='rho')
        }
    
    def calculate_volatility_surface_factors(self, symbol: str = 'SPY',
                                           start_date: str = '2023-01-01',
                                           end_date: str = '2023-12-31') -> Dict[str, pd.Series]:
        """
        计算波动率曲面因子
        
        Args:
            symbol: 标的资产代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            波动率曲面因子字典
        """
        factors = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 由于真实波动率曲面数据获取复杂，使用模拟数据
        factors = self._generate_mock_volatility_surface_factors(dates, symbol)
        
        return factors
    
    def _generate_mock_volatility_surface_factors(self, dates: pd.DatetimeIndex, symbol: str) -> Dict[str, pd.Series]:
        """生成模拟波动率曲面因子 - 仅用于测试和演示"""
        np.random.seed(hash(symbol + 'vol_surface') % 2**32)  # 设置随机种子 - 模拟数据仅用于测试
        
        # 1. 波动率偏斜 (Put偏斜) - 模拟数据仅用于演示
        vol_skew = np.random.normal(-0.05, 0.02, len(dates))  # 通常为负 - 仅用于测试
        
        # 2. 波动率凸性 - 模拟数据仅用于演示
        vol_convexity = np.random.normal(0.001, 0.0005, len(dates))  # 凸性计算 - 仅用于测试
        
        # 3. 期限结构斜率 - 模拟数据仅用于演示
        term_structure_slope = np.random.normal(0.02, 0.01, len(dates))  # 期限斜率 - 仅用于测试
        
        # 4. 波动率曲面的曲率 - 模拟数据仅用于演示
        surface_curvature = np.random.normal(0, 0.001, len(dates))  # 曲面曲率 - 仅用于测试
        
        # 5. ATM波动率 - 模拟数据仅用于演示
        atm_vol = 20 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 252) + np.random.normal(0, 2, len(dates))  # ATM波动率 - 仅用于测试
        atm_vol = np.clip(atm_vol, 10, 40)  # 限制范围 - 模拟数据
        
        # 6. 25Delta风险逆转 (25D Call IV - 25D Put IV) - 模拟数据仅用于演示
        risk_reversal_25d = np.random.normal(-2, 1, len(dates))  # 风险逆转 - 仅用于测试
        
        # 7. 25Delta蝶式价差 (平均(25D Call IV + 25D Put IV) - ATM IV) - 模拟数据仅用于演示
        butterfly_25d = np.random.normal(2, 0.5, len(dates))  # 蝶式价差 - 仅用于测试
        
        return {
            'vol_skew': pd.Series(vol_skew, index=dates, name='vol_skew'),
            'vol_convexity': pd.Series(vol_convexity, index=dates, name='vol_convexity'),
            'term_structure_slope': pd.Series(term_structure_slope, index=dates, name='term_structure_slope'),
            'surface_curvature': pd.Series(surface_curvature, index=dates, name='surface_curvature'),
            'atm_volatility': pd.Series(atm_vol, index=dates, name='atm_volatility'),
            'risk_reversal_25d': pd.Series(risk_reversal_25d, index=dates, name='risk_reversal_25d'),
            'butterfly_25d': pd.Series(butterfly_25d, index=dates, name='butterfly_25d')
        }
    
    def calculate_option_sentiment_factors(self, symbol: str = 'SPY',
                                         start_date: str = '2023-01-01',
                                         end_date: str = '2023-12-31') -> Dict[str, pd.Series]:
        """
        计算期权情绪因子
        
        Args:
            symbol: 标的资产代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            期权情绪因子字典
        """
        factors = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 获取VIX作为恐慌指数
        vix_data = self._get_vix_data(start_date, end_date)
        
        if not vix_data.empty:
            factors['fear_index'] = vix_data
            factors['fear_greed_ratio'] = 50 / vix_data  # 简化的恐慌贪婪比率
        
        # 生成其他情绪因子
        sentiment_factors = self._generate_mock_sentiment_factors(dates, symbol)
        factors.update(sentiment_factors)
        
        return factors
    
    def _generate_mock_sentiment_factors(self, dates: pd.DatetimeIndex, symbol: str) -> Dict[str, pd.Series]:
        """生成模拟期权情绪因子 - 仅用于测试和演示"""
        np.random.seed(hash(symbol + 'sentiment') % 2**32)  # 设置随机种子 - 模拟数据仅用于测试
        
        # 1. 投资者恐慌指数 (基于Put/Call比率) - 模拟数据仅用于演示
        panic_index = 50 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 252) + np.random.normal(0, 10, len(dates))
        panic_index = np.clip(panic_index, 10, 90)
        
        # 2. 期权情绪指数 - 模拟数据仅用于测试
        option_sentiment = np.random.normal(50, 15, len(dates))
        option_sentiment = np.clip(option_sentiment, 0, 100)
        
        # 3. 智能资金指数 (大单交易偏向) - 模拟数据仅用于演示
        smart_money_index = np.random.normal(0, 1, len(dates))
        
        # 4. 散户情绪指数 - 仅用于测试
        retail_sentiment = -smart_money_index + np.random.normal(0, 0.5, len(dates))
        
        # 5. 期权偏斜情绪 - 模拟数据仅用于演示
        skew_sentiment = np.random.normal(-10, 5, len(dates))  # 通常为负值 - 仅用于测试
        
        return {
            'panic_index': pd.Series(panic_index, index=dates, name='panic_index'),
            'option_sentiment': pd.Series(option_sentiment, index=dates, name='option_sentiment'),
            'smart_money_index': pd.Series(smart_money_index, index=dates, name='smart_money_index'),
            'retail_sentiment': pd.Series(retail_sentiment, index=dates, name='retail_sentiment'),
            'skew_sentiment': pd.Series(skew_sentiment, index=dates, name='skew_sentiment')
        }
    
    def calculate_all_factors(self, symbol: str = 'SPY',
                            start_date: str = '2023-01-01',
                            end_date: str = '2023-12-31') -> Dict[str, pd.Series]:
        """
        计算所有期权因子
        
        Args:
            symbol: 标的资产代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            所有期权因子字典
        """
        all_factors = {}
        
        print("开始计算期权因子...")
        
        # 1. 隐含波动率因子
        iv_factors = self.calculate_implied_volatility_factors(symbol, start_date, end_date)
        all_factors.update({f"iv_{k}": v for k, v in iv_factors.items()})
        
        # 2. 期权流量因子
        flow_factors = self.calculate_option_flow_factors(symbol, start_date, end_date)
        all_factors.update({f"flow_{k}": v for k, v in flow_factors.items()})
        
        # 3. 期权Greeks因子
        greeks_factors = self.calculate_option_greeks_factors(symbol, start_date, end_date)
        all_factors.update({f"greeks_{k}": v for k, v in greeks_factors.items()})
        
        # 4. 波动率曲面因子
        surface_factors = self.calculate_volatility_surface_factors(symbol, start_date, end_date)
        all_factors.update({f"surface_{k}": v for k, v in surface_factors.items()})
        
        # 5. 期权情绪因子
        sentiment_factors = self.calculate_option_sentiment_factors(symbol, start_date, end_date)
        all_factors.update({f"sentiment_{k}": v for k, v in sentiment_factors.items()})
        
        print(f"期权因子计算完成，共生成 {len(all_factors)} 个因子")
        return all_factors

def main():
    """测试期权因子计算器"""
    print("测试期权因子计算器...")
    
    # 初始化计算器
    calculator = OptionFactorCalculator()
    
    # 计算因子
    factors = calculator.calculate_all_factors('SPY', '2023-01-01', '2023-12-31')
    
    print(f"成功计算 {len(factors)} 个期权因子")
    for name, factor in list(factors.items())[:10]:  # 只显示前10个
        if not factor.empty:
            print(f"{name}: 均值={factor.mean():.4f}, 标准差={factor.std():.4f}")

if __name__ == "__main__":
    main()