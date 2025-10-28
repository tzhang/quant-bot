"""
宏观经济因子计算模块

实现各种宏观经济指标的计算和分析，包括：
- 利率因子：国债收益率、利率期限结构等
- 通胀因子：CPI、PPI、通胀预期等
- 货币政策因子：货币供应量、央行政策等
- 经济增长因子：GDP、工业增加值、PMI等
- 市场情绪因子：VIX、风险偏好等
- 汇率因子：美元指数、汇率波动等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
import requests
# import yfinance as yf  # 已移除，不再使用yfinance
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacroFactorCalculator:
    """
    宏观经济因子计算器
    
    提供各种宏观经济因子的计算功能，包括：
    - 利率环境因子
    - 通胀环境因子
    - 货币政策因子
    - 经济增长因子
    - 市场风险因子
    - 汇率环境因子
    """
    
    def __init__(self):
        """初始化宏观因子计算器"""
        self.logger = logger
        self.data_cache = {}
        
        # 常用的宏观经济指标代码
        self.macro_symbols = {
            # 利率相关
            'TNX': '^TNX',      # 10年期美债收益率
            'FVX': '^FVX',      # 5年期美债收益率
            'TYX': '^TYX',      # 30年期美债收益率
            'IRX': '^IRX',      # 3个月美债收益率
            
            # 股指相关
            'SPY': 'SPY',       # 标普500 ETF
            'VIX': '^VIX',      # 波动率指数
            'DXY': 'DX-Y.NYB',  # 美元指数
            
            # 商品相关
            'GLD': 'GLD',       # 黄金ETF
            'USO': 'USO',       # 原油ETF
            'UNG': 'UNG',       # 天然气ETF
            
            # 汇率相关
            'EURUSD': 'EURUSD=X',
            'USDJPY': 'USDJPY=X',
            'GBPUSD': 'GBPUSD=X',
            'USDCNY': 'USDCNY=X',
        }
    
    def fetch_macro_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        获取宏观经济数据
        
        数据源优先级：IB TWS API > yfinance
        
        Args:
            symbol: 数据符号（如 ^TNX, ^VIX等）
            period: 数据周期
            
        Returns:
            pd.DataFrame: 宏观经济数据
        """
        try:
            # 检查缓存
            cache_key = f"{symbol}_{period}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # 优先尝试使用 IB TWS API
            try:
                from src.data.ib_data_provider import IBDataProvider, IBConfig
                ib_provider = IBDataProvider(IBConfig())
                
                # 转换期间为天数
                days_map = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
                days = days_map.get(period, 365)
                
                data = ib_provider.get_historical_data(symbol, days=days, bar_size='1 day')
                if not data.empty:
                    self.data_cache[cache_key] = data
                    self.logger.info(f"成功从 IB TWS API 获取 {symbol} 的宏观数据")
                    return data
                    
            except Exception as e:
                self.logger.warning(f"IB TWS API 获取宏观数据 {symbol} 失败: {e}")
            
            # 回退到 yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                self.data_cache[cache_key] = data
                self.logger.info(f"成功从 yfinance 获取 {symbol} 的宏观数据")
                return data
            else:
                self.logger.warning(f"无法获取 {symbol} 的数据")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取宏观数据 {symbol} 时出错: {e}")
            return pd.DataFrame()
    
    def calculate_interest_rate_factors(self) -> Dict[str, float]:
        """
        计算利率环境因子
        
        Returns:
            Dict[str, float]: 利率因子字典
        """
        try:
            factors = {}
            
            # 获取各期限国债收益率
            tnx_data = self.fetch_macro_data(self.macro_symbols['TNX'])  # 10年期
            fvx_data = self.fetch_macro_data(self.macro_symbols['FVX'])  # 5年期
            tyx_data = self.fetch_macro_data(self.macro_symbols['TYX'])  # 30年期
            irx_data = self.fetch_macro_data(self.macro_symbols['IRX'])  # 3个月
            
            # 当前收益率水平
            if not tnx_data.empty:
                factors['yield_10y'] = tnx_data['Close'].iloc[-1]
                
                # 10年期收益率变化
                if len(tnx_data) >= 20:
                    factors['yield_10y_change_20d'] = tnx_data['Close'].iloc[-1] - tnx_data['Close'].iloc[-20]
                    factors['yield_10y_volatility_20d'] = tnx_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            if not fvx_data.empty:
                factors['yield_5y'] = fvx_data['Close'].iloc[-1]
            
            if not tyx_data.empty:
                factors['yield_30y'] = tyx_data['Close'].iloc[-1]
            
            if not irx_data.empty:
                factors['yield_3m'] = irx_data['Close'].iloc[-1]
            
            # 收益率曲线形状
            if 'yield_10y' in factors and 'yield_3m' in factors:
                factors['yield_curve_slope'] = factors['yield_10y'] - factors['yield_3m']
                
            if 'yield_30y' in factors and 'yield_10y' in factors:
                factors['yield_curve_curvature'] = factors['yield_30y'] - 2 * factors['yield_10y'] + factors['yield_3m'] if 'yield_3m' in factors else 0
            
            # 实际利率估算 (名义利率 - 通胀预期，这里简化为固定值2%)
            if 'yield_10y' in factors:
                factors['real_yield_10y'] = factors['yield_10y'] - 2.0  # 假设通胀预期2%
            
            # 利率环境评估
            if 'yield_10y' in factors:
                if factors['yield_10y'] < 2.0:
                    factors['rate_environment'] = 1  # 低利率环境
                elif factors['yield_10y'] < 4.0:
                    factors['rate_environment'] = 2  # 中等利率环境
                else:
                    factors['rate_environment'] = 3  # 高利率环境
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算利率因子时出错: {e}")
            return {}
    
    def calculate_inflation_factors(self) -> Dict[str, float]:
        """
        计算通胀环境因子
        
        Returns:
            Dict[str, float]: 通胀因子字典
        """
        try:
            factors = {}
            
            # 获取商品价格数据作为通胀代理指标
            gold_data = self.fetch_macro_data(self.macro_symbols['GLD'])
            oil_data = self.fetch_macro_data(self.macro_symbols['USO'])
            
            # 黄金价格变化 (通胀对冲指标)
            if not gold_data.empty:
                factors['gold_price'] = gold_data['Close'].iloc[-1]
                
                if len(gold_data) >= 60:
                    # 黄金价格动量
                    factors['gold_momentum_60d'] = (gold_data['Close'].iloc[-1] / gold_data['Close'].iloc[-60] - 1) * 100
                    factors['gold_volatility_60d'] = gold_data['Close'].pct_change().rolling(60).std().iloc[-1] * np.sqrt(252) * 100
            
            # 原油价格变化 (通胀先行指标)
            if not oil_data.empty:
                factors['oil_price'] = oil_data['Close'].iloc[-1]
                
                if len(oil_data) >= 60:
                    factors['oil_momentum_60d'] = (oil_data['Close'].iloc[-1] / oil_data['Close'].iloc[-60] - 1) * 100
                    factors['oil_volatility_60d'] = oil_data['Close'].pct_change().rolling(60).std().iloc[-1] * np.sqrt(252) * 100
            
            # 通胀预期指标 (基于TIPS收益率差，这里简化处理)
            tnx_data = self.fetch_macro_data(self.macro_symbols['TNX'])
            if not tnx_data.empty and len(tnx_data) >= 252:
                # 使用10年期收益率的历史分位数作为通胀预期代理
                current_yield = tnx_data['Close'].iloc[-1]
                historical_yields = tnx_data['Close'].iloc[-252:]
                percentile = stats.percentileofscore(historical_yields, current_yield)
                factors['inflation_expectation_proxy'] = percentile
            
            # 商品综合指数 (黄金和原油的加权平均)
            if 'gold_momentum_60d' in factors and 'oil_momentum_60d' in factors:
                factors['commodity_index'] = 0.6 * factors['gold_momentum_60d'] + 0.4 * factors['oil_momentum_60d']
            
            # 通胀环境评估
            if 'commodity_index' in factors:
                if factors['commodity_index'] > 10:
                    factors['inflation_environment'] = 3  # 高通胀环境
                elif factors['commodity_index'] > 0:
                    factors['inflation_environment'] = 2  # 温和通胀环境
                else:
                    factors['inflation_environment'] = 1  # 低通胀/通缩环境
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算通胀因子时出错: {e}")
            return {}
    
    def calculate_monetary_policy_factors(self) -> Dict[str, float]:
        """
        计算货币政策因子
        
        Returns:
            Dict[str, float]: 货币政策因子字典
        """
        try:
            factors = {}
            
            # 获取短期利率数据
            irx_data = self.fetch_macro_data(self.macro_symbols['IRX'])  # 3个月国债
            tnx_data = self.fetch_macro_data(self.macro_symbols['TNX'])  # 10年期国债
            
            # 政策利率变化趋势
            if not irx_data.empty and len(irx_data) >= 60:
                current_rate = irx_data['Close'].iloc[-1]
                rate_60d_ago = irx_data['Close'].iloc[-60]
                factors['policy_rate_change_60d'] = current_rate - rate_60d_ago
                
                # 政策利率趋势
                if factors['policy_rate_change_60d'] > 0.5:
                    factors['monetary_policy_stance'] = 3  # 紧缩
                elif factors['policy_rate_change_60d'] > -0.5:
                    factors['monetary_policy_stance'] = 2  # 中性
                else:
                    factors['monetary_policy_stance'] = 1  # 宽松
            
            # 收益率曲线控制指标
            if not irx_data.empty and not tnx_data.empty:
                short_rate = irx_data['Close'].iloc[-1]
                long_rate = tnx_data['Close'].iloc[-1]
                factors['yield_curve_control'] = long_rate - short_rate
                
                # 倒挂检测
                factors['yield_curve_inversion'] = 1 if factors['yield_curve_control'] < 0 else 0
            
            # 流动性环境 (使用VIX作为代理)
            vix_data = self.fetch_macro_data(self.macro_symbols['VIX'])
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                factors['market_liquidity_proxy'] = current_vix
                
                # 流动性环境评估
                if current_vix > 30:
                    factors['liquidity_environment'] = 1  # 流动性紧张
                elif current_vix > 20:
                    factors['liquidity_environment'] = 2  # 流动性正常
                else:
                    factors['liquidity_environment'] = 3  # 流动性充裕
            
            # 美元流动性 (美元指数变化)
            dxy_data = self.fetch_macro_data(self.macro_symbols['DXY'])
            if not dxy_data.empty and len(dxy_data) >= 60:
                factors['usd_strength'] = dxy_data['Close'].iloc[-1]
                factors['usd_momentum_60d'] = (dxy_data['Close'].iloc[-1] / dxy_data['Close'].iloc[-60] - 1) * 100
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算货币政策因子时出错: {e}")
            return {}
    
    def calculate_economic_growth_factors(self) -> Dict[str, float]:
        """
        计算经济增长因子
        
        Returns:
            Dict[str, float]: 经济增长因子字典
        """
        try:
            factors = {}
            
            # 使用股市表现作为经济增长代理指标
            spy_data = self.fetch_macro_data(self.macro_symbols['SPY'])
            
            if not spy_data.empty:
                # 股市动量 (经济增长预期)
                if len(spy_data) >= 60:
                    factors['equity_momentum_60d'] = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-60] - 1) * 100
                
                if len(spy_data) >= 252:
                    factors['equity_momentum_1y'] = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-252] - 1) * 100
                
                # 股市波动率 (经济不确定性)
                if len(spy_data) >= 60:
                    factors['equity_volatility_60d'] = spy_data['Close'].pct_change().rolling(60).std().iloc[-1] * np.sqrt(252) * 100
            
            # 风险偏好指标
            vix_data = self.fetch_macro_data(self.macro_symbols['VIX'])
            if not vix_data.empty:
                factors['risk_aversion'] = vix_data['Close'].iloc[-1]
                
                # 风险偏好变化
                if len(vix_data) >= 20:
                    factors['risk_aversion_change_20d'] = vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[-20]
            
            # 经济周期位置估算
            if 'equity_momentum_1y' in factors and 'risk_aversion' in factors:
                # 简化的经济周期判断
                if factors['equity_momentum_1y'] > 10 and factors['risk_aversion'] < 20:
                    factors['economic_cycle'] = 4  # 扩张期
                elif factors['equity_momentum_1y'] > 0 and factors['risk_aversion'] < 25:
                    factors['economic_cycle'] = 3  # 复苏期
                elif factors['equity_momentum_1y'] < -10 and factors['risk_aversion'] > 30:
                    factors['economic_cycle'] = 1  # 衰退期
                else:
                    factors['economic_cycle'] = 2  # 放缓期
            
            # 行业轮动指标 (这里简化处理)
            if 'equity_momentum_60d' in factors:
                # 基于股市动量判断行业轮动阶段
                if factors['equity_momentum_60d'] > 5:
                    factors['sector_rotation'] = 1  # 成长股主导
                elif factors['equity_momentum_60d'] > -5:
                    factors['sector_rotation'] = 2  # 平衡配置
                else:
                    factors['sector_rotation'] = 3  # 价值股主导
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算经济增长因子时出错: {e}")
            return {}
    
    def calculate_market_risk_factors(self) -> Dict[str, float]:
        """
        计算市场风险因子
        
        Returns:
            Dict[str, float]: 市场风险因子字典
        """
        try:
            factors = {}
            
            # VIX相关指标
            vix_data = self.fetch_macro_data(self.macro_symbols['VIX'])
            if not vix_data.empty:
                factors['vix_level'] = vix_data['Close'].iloc[-1]
                
                # VIX历史分位数
                if len(vix_data) >= 252:
                    historical_vix = vix_data['Close'].iloc[-252:]
                    factors['vix_percentile'] = stats.percentileofscore(historical_vix, factors['vix_level'])
                
                # VIX变化
                if len(vix_data) >= 5:
                    factors['vix_change_5d'] = vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[-5]
            
            # 股债相关性
            spy_data = self.fetch_macro_data(self.macro_symbols['SPY'])
            tnx_data = self.fetch_macro_data(self.macro_symbols['TNX'])
            
            if not spy_data.empty and not tnx_data.empty and len(spy_data) >= 60 and len(tnx_data) >= 60:
                # 计算股债相关性
                spy_returns = spy_data['Close'].pct_change().dropna()
                bond_returns = -tnx_data['Close'].pct_change().dropna()  # 债券收益率下降时债券价格上升
                
                # 对齐数据
                common_dates = spy_returns.index.intersection(bond_returns.index)
                if len(common_dates) >= 60:
                    spy_aligned = spy_returns.loc[common_dates]
                    bond_aligned = bond_returns.loc[common_dates]
                    
                    # 60日滚动相关性
                    correlation = spy_aligned.rolling(60).corr(bond_aligned).iloc[-1]
                    factors['stock_bond_correlation'] = correlation if not np.isnan(correlation) else 0
            
            # 避险情绪指标
            gold_data = self.fetch_macro_data(self.macro_symbols['GLD'])
            if not gold_data.empty and not spy_data.empty and len(gold_data) >= 20 and len(spy_data) >= 20:
                # 黄金与股市的相对表现
                gold_return_20d = (gold_data['Close'].iloc[-1] / gold_data['Close'].iloc[-20] - 1) * 100
                spy_return_20d = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-20] - 1) * 100
                factors['safe_haven_demand'] = gold_return_20d - spy_return_20d
            
            # 市场压力指标
            if 'vix_level' in factors and 'stock_bond_correlation' in factors:
                # 综合压力指数
                vix_score = min(factors['vix_level'] / 40, 1)  # VIX标准化到0-1
                corr_score = max(factors['stock_bond_correlation'], 0)  # 正相关表示压力
                factors['market_stress_index'] = (vix_score + corr_score) / 2 * 100
            
            # 流动性风险
            if not spy_data.empty and len(spy_data) >= 20:
                # 使用成交量变化作为流动性代理
                recent_volume = spy_data['Volume'].iloc[-5:].mean()
                historical_volume = spy_data['Volume'].iloc[-60:-5].mean()
                if historical_volume > 0:
                    factors['liquidity_risk'] = (recent_volume / historical_volume - 1) * 100
                else:
                    factors['liquidity_risk'] = 0
            
            # 尾部风险指标
            if not spy_data.empty and len(spy_data) >= 252:
                returns = spy_data['Close'].pct_change().dropna()
                if len(returns) >= 252:
                    # 计算VaR和CVaR
                    returns_1y = returns.iloc[-252:]
                    var_95 = np.percentile(returns_1y, 5) * 100
                    cvar_95 = returns_1y[returns_1y <= np.percentile(returns_1y, 5)].mean() * 100
                    factors['var_95'] = var_95
                    factors['cvar_95'] = cvar_95
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算市场风险因子时出错: {e}")
            return {}
    
    def calculate_currency_factors(self) -> Dict[str, float]:
        """
        计算汇率环境因子
        
        Returns:
            Dict[str, float]: 汇率因子字典
        """
        try:
            factors = {}
            
            # 美元指数
            dxy_data = self.fetch_macro_data(self.macro_symbols['DXY'])
            if not dxy_data.empty:
                factors['usd_index'] = dxy_data['Close'].iloc[-1]
                
                # 美元指数动量
                if len(dxy_data) >= 60:
                    factors['usd_momentum_60d'] = (dxy_data['Close'].iloc[-1] / dxy_data['Close'].iloc[-60] - 1) * 100
                
                # 美元指数波动率
                if len(dxy_data) >= 20:
                    factors['usd_volatility_20d'] = dxy_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            # 主要货币对
            currency_pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCNY']
            
            for pair in currency_pairs:
                if pair in self.macro_symbols:
                    pair_data = self.fetch_macro_data(self.macro_symbols[pair])
                    if not pair_data.empty:
                        factors[f'{pair.lower()}_rate'] = pair_data['Close'].iloc[-1]
                        
                        # 汇率动量
                        if len(pair_data) >= 30:
                            factors[f'{pair.lower()}_momentum_30d'] = (pair_data['Close'].iloc[-1] / pair_data['Close'].iloc[-30] - 1) * 100
                        
                        # 汇率波动率
                        if len(pair_data) >= 20:
                            factors[f'{pair.lower()}_volatility_20d'] = pair_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            # 汇率风险指标
            if 'usd_volatility_20d' in factors:
                if factors['usd_volatility_20d'] > 15:
                    factors['currency_risk_level'] = 3  # 高风险
                elif factors['usd_volatility_20d'] > 10:
                    factors['currency_risk_level'] = 2  # 中等风险
                else:
                    factors['currency_risk_level'] = 1  # 低风险
            
            # 美元强弱周期
            if 'usd_momentum_60d' in factors:
                if factors['usd_momentum_60d'] > 5:
                    factors['usd_cycle'] = 3  # 美元强势
                elif factors['usd_momentum_60d'] > -5:
                    factors['usd_cycle'] = 2  # 美元中性
                else:
                    factors['usd_cycle'] = 1  # 美元弱势
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算汇率因子时出错: {e}")
            return {}
    
    def calculate_all_macro_factors(self) -> Dict[str, Dict[str, float]]:
        """
        计算所有宏观经济因子
        
        Returns:
            Dict[str, Dict[str, float]]: 所有宏观因子的字典
        """
        try:
            all_factors = {
                'interest_rate': self.calculate_interest_rate_factors(),
                'inflation': self.calculate_inflation_factors(),
                'monetary_policy': self.calculate_monetary_policy_factors(),
                'economic_growth': self.calculate_economic_growth_factors(),
                'market_risk': self.calculate_market_risk_factors(),
                'currency': self.calculate_currency_factors()
            }
            
            return all_factors
            
        except Exception as e:
            self.logger.error(f"计算所有宏观因子时出错: {e}")
            return {}
    
    def calculate_macro_regime(self, all_factors: Dict[str, Dict[str, float]]) -> Dict[str, Union[int, str]]:
        """
        计算宏观经济制度/环境
        
        Args:
            all_factors: 所有宏观因子
            
        Returns:
            Dict[str, Union[int, str]]: 宏观制度判断
        """
        try:
            regime = {}
            
            # 利率制度
            if 'interest_rate' in all_factors and 'rate_environment' in all_factors['interest_rate']:
                regime['rate_regime'] = all_factors['interest_rate']['rate_environment']
                regime['rate_regime_desc'] = ['', '低利率', '中等利率', '高利率'][regime['rate_regime']]
            
            # 通胀制度
            if 'inflation' in all_factors and 'inflation_environment' in all_factors['inflation']:
                regime['inflation_regime'] = all_factors['inflation']['inflation_environment']
                regime['inflation_regime_desc'] = ['', '低通胀', '温和通胀', '高通胀'][regime['inflation_regime']]
            
            # 货币政策制度
            if 'monetary_policy' in all_factors and 'monetary_policy_stance' in all_factors['monetary_policy']:
                regime['monetary_regime'] = all_factors['monetary_policy']['monetary_policy_stance']
                regime['monetary_regime_desc'] = ['', '宽松', '中性', '紧缩'][regime['monetary_regime']]
            
            # 经济周期
            if 'economic_growth' in all_factors and 'economic_cycle' in all_factors['economic_growth']:
                regime['economic_cycle'] = all_factors['economic_growth']['economic_cycle']
                regime['economic_cycle_desc'] = ['', '衰退', '放缓', '复苏', '扩张'][regime['economic_cycle']]
            
            # 风险环境
            if 'market_risk' in all_factors and 'vix_level' in all_factors['market_risk']:
                vix = all_factors['market_risk']['vix_level']
                if vix > 30:
                    regime['risk_regime'] = 3
                    regime['risk_regime_desc'] = '高风险'
                elif vix > 20:
                    regime['risk_regime'] = 2
                    regime['risk_regime_desc'] = '中等风险'
                else:
                    regime['risk_regime'] = 1
                    regime['risk_regime_desc'] = '低风险'
            
            # 美元周期
            if 'currency' in all_factors and 'usd_cycle' in all_factors['currency']:
                regime['usd_regime'] = all_factors['currency']['usd_cycle']
                regime['usd_regime_desc'] = ['', '美元弱势', '美元中性', '美元强势'][regime['usd_regime']]
            
            return regime
            
        except Exception as e:
            self.logger.error(f"计算宏观制度时出错: {e}")
            return {}
    
    def generate_macro_report(self, all_factors: Dict[str, Dict[str, float]]) -> str:
        """
        生成宏观经济因子分析报告
        
        Args:
            all_factors: 所有宏观因子数据
            
        Returns:
            str: 宏观因子分析报告
        """
        try:
            report = f"\n{'='*60}\n"
            report += f"宏观经济因子分析报告\n"
            report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"{'='*60}\n\n"
            
            # 各类因子详情
            for category, factors in all_factors.items():
                if factors:  # 只显示有数据的类别
                    category_names = {
                        'interest_rate': '利率环境',
                        'inflation': '通胀环境',
                        'monetary_policy': '货币政策',
                        'economic_growth': '经济增长',
                        'market_risk': '市场风险',
                        'currency': '汇率环境'
                    }
                    
                    report += f"{category_names.get(category, category.upper())} 因子:\n"
                    report += f"{'-'*40}\n"
                    
                    for factor_name, factor_value in factors.items():
                        if isinstance(factor_value, (int, float)) and not np.isnan(factor_value):
                            if abs(factor_value) >= 1000:
                                report += f"{factor_name:30}: {factor_value:8.0f}\n"
                            elif abs(factor_value) >= 1:
                                report += f"{factor_name:30}: {factor_value:8.2f}\n"
                            else:
                                report += f"{factor_name:30}: {factor_value:8.4f}\n"
                    
                    report += "\n"
            
            # 宏观制度判断
            regime = self.calculate_macro_regime(all_factors)
            if regime:
                report += "宏观经济制度判断:\n"
                report += f"{'-'*40}\n"
                
                for key, value in regime.items():
                    if key.endswith('_desc'):
                        regime_type = key.replace('_desc', '').replace('_', ' ').title()
                        report += f"{regime_type:20}: {value}\n"
                
                report += "\n"
            
            # 投资建议
            report += "基于宏观环境的投资建议:\n"
            report += f"{'-'*40}\n"
            
            if regime:
                # 基于不同制度组合给出建议
                if regime.get('economic_cycle') == 4 and regime.get('risk_regime') == 1:
                    report += "• 经济扩张 + 低风险环境：适合配置成长股和周期股\n"
                elif regime.get('economic_cycle') == 1 and regime.get('risk_regime') == 3:
                    report += "• 经济衰退 + 高风险环境：建议防御性配置，增加债券和黄金\n"
                elif regime.get('monetary_regime') == 1:
                    report += "• 宽松货币政策：有利于风险资产，可增加股票配置\n"
                elif regime.get('monetary_regime') == 3:
                    report += "• 紧缩货币政策：谨慎配置风险资产，考虑现金和短期债券\n"
                
                if regime.get('inflation_regime') == 3:
                    report += "• 高通胀环境：考虑配置商品、REITs和通胀保护债券\n"
                elif regime.get('usd_regime') == 3:
                    report += "• 美元强势：有利于美国资产，谨慎配置新兴市场\n"
                elif regime.get('usd_regime') == 1:
                    report += "• 美元弱势：有利于商品和新兴市场资产\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成宏观因子报告时出错: {e}")
            return f"生成报告失败: {e}"


def main():
    """主函数 - 示例用法"""
    try:
        # 创建宏观因子计算器实例
        calculator = MacroFactorCalculator()
        
        print("正在获取宏观经济数据并计算因子...")
        
        # 计算所有宏观因子
        all_factors = calculator.calculate_all_macro_factors()
        
        # 生成报告
        report = calculator.generate_macro_report(all_factors)
        print(report)
        
        # 输出因子数量统计
        total_factors = sum(len(factors) for factors in all_factors.values())
        print(f"\n总计算出 {total_factors} 个宏观经济因子")
        
        for category, factors in all_factors.items():
            print(f"{category}: {len(factors)} 个因子")
        
    except Exception as e:
        logger.error(f"主函数执行出错: {e}")


if __name__ == "__main__":
    main()