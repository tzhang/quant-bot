"""
因子分析模块 - 集成技术和基本面因子分析
提供多维度因子计算、因子有效性评估、因子组合优化等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FactorValue:
    """因子值"""
    timestamp: datetime
    symbol: str
    factor_name: str
    value: float
    percentile: Optional[float] = None
    z_score: Optional[float] = None

@dataclass
class FactorPerformance:
    """因子表现"""
    factor_name: str
    ic: float  # 信息系数
    ic_ir: float  # IC信息比率
    rank_ic: float  # 排序IC
    win_rate: float  # 胜率
    annual_return: float  # 年化收益
    volatility: float  # 波动率
    sharpe_ratio: float  # 夏普比率
    max_drawdown: float  # 最大回撤

class TechnicalFactors:
    """技术因子计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_momentum_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算动量因子"""
        factors = {}
        
        try:
            # 价格动量
            factors['momentum_1d'] = df['close'].pct_change(1)
            factors['momentum_5d'] = df['close'].pct_change(5)
            factors['momentum_10d'] = df['close'].pct_change(10)
            factors['momentum_20d'] = df['close'].pct_change(20)
            
            # 相对强弱指数
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            factors['rsi'] = 100 - (100 / (1 + rs))
            
            # 威廉指标
            high_14 = df['high'].rolling(14).max()
            low_14 = df['low'].rolling(14).min()
            factors['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
            
            # 随机指标
            factors['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            factors['stoch_d'] = factors['stoch_k'].rolling(3).mean()
            
        except Exception as e:
            self.logger.error(f"计算动量因子失败: {e}")
            
        return factors
        
    def calculate_mean_reversion_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算均值回归因子"""
        factors = {}
        
        try:
            # 布林带位置
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            factors['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # 价格相对移动平均线位置
            factors['price_to_sma5'] = df['close'] / df['close'].rolling(5).mean() - 1
            factors['price_to_sma10'] = df['close'] / df['close'].rolling(10).mean() - 1
            factors['price_to_sma20'] = df['close'] / df['close'].rolling(20).mean() - 1
            
            # 移动平均线斜率
            factors['sma5_slope'] = df['close'].rolling(5).mean().diff(5)
            factors['sma10_slope'] = df['close'].rolling(10).mean().diff(5)
            factors['sma20_slope'] = df['close'].rolling(20).mean().diff(5)
            
            # 价格偏离度
            factors['price_deviation'] = (df['close'] - sma20) / std20
            
        except Exception as e:
            self.logger.error(f"计算均值回归因子失败: {e}")
            
        return factors
        
    def calculate_volatility_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算波动率因子"""
        factors = {}
        
        try:
            # 历史波动率
            returns = df['close'].pct_change()
            factors['volatility_5d'] = returns.rolling(5).std() * np.sqrt(252)
            factors['volatility_10d'] = returns.rolling(10).std() * np.sqrt(252)
            factors['volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
            
            # 真实波动率
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            factors['atr'] = true_range.rolling(14).mean()
            
            # 波动率比率
            factors['volatility_ratio'] = factors['volatility_5d'] / factors['volatility_20d']
            
            # Garman-Klass波动率
            factors['gk_volatility'] = np.sqrt(
                0.5 * (np.log(df['high'] / df['low'])) ** 2 - 
                (2 * np.log(2) - 1) * (np.log(df['close'] / df['open'])) ** 2
            ).rolling(20).mean()
            
        except Exception as e:
            self.logger.error(f"计算波动率因子失败: {e}")
            
        return factors
        
    def calculate_volume_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算成交量因子"""
        factors = {}
        
        try:
            # 成交量移动平均
            factors['volume_sma5'] = df['volume'].rolling(5).mean()
            factors['volume_sma10'] = df['volume'].rolling(10).mean()
            factors['volume_sma20'] = df['volume'].rolling(20).mean()
            
            # 成交量比率
            factors['volume_ratio'] = df['volume'] / factors['volume_sma20']
            
            # 价量关系
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            factors['price_volume_corr'] = price_change.rolling(20).corr(volume_change)
            
            # OBV (On Balance Volume)
            obv = (np.sign(price_change) * df['volume']).fillna(0).cumsum()
            factors['obv'] = obv
            factors['obv_sma'] = obv.rolling(20).mean()
            
            # 成交量加权平均价格偏离
            if 'vwap' in df.columns:
                factors['price_vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
            
        except Exception as e:
            self.logger.error(f"计算成交量因子失败: {e}")
            
        return factors
        
    def calculate_trend_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算趋势因子"""
        factors = {}
        
        try:
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            factors['macd'] = macd
            factors['macd_signal'] = signal
            factors['macd_histogram'] = macd - signal
            
            # ADX (Average Directional Index)
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            true_range = np.maximum(df['high'] - df['low'], 
                                  np.maximum(np.abs(df['high'] - df['close'].shift()),
                                           np.abs(df['low'] - df['close'].shift())))
            
            plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / pd.Series(true_range).rolling(14).sum()
            minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / pd.Series(true_range).rolling(14).sum()
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            factors['adx'] = dx.rolling(14).mean()
            
            # 抛物线SAR
            factors['sar'] = self._calculate_sar(df)
            
        except Exception as e:
            self.logger.error(f"计算趋势因子失败: {e}")
            
        return factors
        
    def _calculate_sar(self, df: pd.DataFrame, af_start=0.02, af_increment=0.02, af_max=0.2) -> pd.Series:
        """计算抛物线SAR"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            sar = np.zeros(len(df))
            trend = np.zeros(len(df))
            af = np.zeros(len(df))
            ep = np.zeros(len(df))
            
            # 初始化
            sar[0] = low[0]
            trend[0] = 1
            af[0] = af_start
            ep[0] = high[0]
            
            for i in range(1, len(df)):
                if trend[i-1] == 1:  # 上升趋势
                    sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                    
                    if low[i] <= sar[i]:
                        trend[i] = -1
                        sar[i] = ep[i-1]
                        af[i] = af_start
                        ep[i] = low[i]
                    else:
                        trend[i] = 1
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            af[i] = min(af[i-1] + af_increment, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                else:  # 下降趋势
                    sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                    
                    if high[i] >= sar[i]:
                        trend[i] = 1
                        sar[i] = ep[i-1]
                        af[i] = af_start
                        ep[i] = high[i]
                    else:
                        trend[i] = -1
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            af[i] = min(af[i-1] + af_increment, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                            
            return pd.Series(sar, index=df.index)
            
        except Exception as e:
            self.logger.error(f"计算SAR失败: {e}")
            return pd.Series(np.nan, index=df.index)

class FundamentalFactors:
    """基本面因子计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_valuation_factors(self, price_data: pd.DataFrame, fundamental_data: Dict) -> Dict[str, pd.Series]:
        """计算估值因子"""
        factors = {}
        
        try:
            # 市盈率
            if 'earnings_per_share' in fundamental_data:
                eps = fundamental_data['earnings_per_share']
                factors['pe_ratio'] = price_data['close'] / eps
                
            # 市净率
            if 'book_value_per_share' in fundamental_data:
                bvps = fundamental_data['book_value_per_share']
                factors['pb_ratio'] = price_data['close'] / bvps
                
            # 市销率
            if 'sales_per_share' in fundamental_data:
                sps = fundamental_data['sales_per_share']
                factors['ps_ratio'] = price_data['close'] / sps
                
            # 企业价值倍数
            if 'enterprise_value' in fundamental_data and 'ebitda' in fundamental_data:
                ev = fundamental_data['enterprise_value']
                ebitda = fundamental_data['ebitda']
                factors['ev_ebitda'] = ev / ebitda
                
        except Exception as e:
            self.logger.error(f"计算估值因子失败: {e}")
            
        return factors
        
    def calculate_quality_factors(self, fundamental_data: Dict) -> Dict[str, float]:
        """计算质量因子"""
        factors = {}
        
        try:
            # ROE
            if 'net_income' in fundamental_data and 'shareholders_equity' in fundamental_data:
                factors['roe'] = fundamental_data['net_income'] / fundamental_data['shareholders_equity']
                
            # ROA
            if 'net_income' in fundamental_data and 'total_assets' in fundamental_data:
                factors['roa'] = fundamental_data['net_income'] / fundamental_data['total_assets']
                
            # 毛利率
            if 'gross_profit' in fundamental_data and 'revenue' in fundamental_data:
                factors['gross_margin'] = fundamental_data['gross_profit'] / fundamental_data['revenue']
                
            # 净利率
            if 'net_income' in fundamental_data and 'revenue' in fundamental_data:
                factors['net_margin'] = fundamental_data['net_income'] / fundamental_data['revenue']
                
            # 资产负债率
            if 'total_debt' in fundamental_data and 'total_assets' in fundamental_data:
                factors['debt_to_assets'] = fundamental_data['total_debt'] / fundamental_data['total_assets']
                
            # 流动比率
            if 'current_assets' in fundamental_data and 'current_liabilities' in fundamental_data:
                factors['current_ratio'] = fundamental_data['current_assets'] / fundamental_data['current_liabilities']
                
        except Exception as e:
            self.logger.error(f"计算质量因子失败: {e}")
            
        return factors
        
    def calculate_growth_factors(self, historical_data: List[Dict]) -> Dict[str, float]:
        """计算成长因子"""
        factors = {}
        
        try:
            if len(historical_data) >= 2:
                current = historical_data[-1]
                previous = historical_data[-2]
                
                # 收入增长率
                if 'revenue' in current and 'revenue' in previous:
                    factors['revenue_growth'] = (current['revenue'] - previous['revenue']) / previous['revenue']
                    
                # 净利润增长率
                if 'net_income' in current and 'net_income' in previous:
                    factors['earnings_growth'] = (current['net_income'] - previous['net_income']) / previous['net_income']
                    
                # 每股收益增长率
                if 'earnings_per_share' in current and 'earnings_per_share' in previous:
                    factors['eps_growth'] = (current['earnings_per_share'] - previous['earnings_per_share']) / previous['earnings_per_share']
                    
        except Exception as e:
            self.logger.error(f"计算成长因子失败: {e}")
            
        return factors

class FactorAnalyzer:
    """因子分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.technical_factors = TechnicalFactors()
        self.fundamental_factors = FundamentalFactors()
        
        # 因子缓存
        self.factor_cache = {}
        self.performance_cache = {}
        
    def calculate_all_factors(self, symbol: str, price_data: pd.DataFrame, 
                            fundamental_data: Optional[Dict] = None) -> Dict[str, pd.Series]:
        """计算所有因子"""
        all_factors = {}
        
        try:
            # 技术因子
            momentum_factors = self.technical_factors.calculate_momentum_factors(price_data)
            mean_reversion_factors = self.technical_factors.calculate_mean_reversion_factors(price_data)
            volatility_factors = self.technical_factors.calculate_volatility_factors(price_data)
            volume_factors = self.technical_factors.calculate_volume_factors(price_data)
            trend_factors = self.technical_factors.calculate_trend_factors(price_data)
            
            all_factors.update(momentum_factors)
            all_factors.update(mean_reversion_factors)
            all_factors.update(volatility_factors)
            all_factors.update(volume_factors)
            all_factors.update(trend_factors)
            
            # 基本面因子
            if fundamental_data:
                valuation_factors = self.fundamental_factors.calculate_valuation_factors(price_data, fundamental_data)
                all_factors.update(valuation_factors)
                
            # 缓存因子
            self.factor_cache[symbol] = all_factors
            
        except Exception as e:
            self.logger.error(f"计算因子失败: {e}")
            
        return all_factors
        
    def evaluate_factor_performance(self, factor_values: pd.Series, returns: pd.Series, 
                                  periods: int = 20) -> FactorPerformance:
        """评估因子表现"""
        try:
            # 计算信息系数
            ic_series = []
            rank_ic_series = []
            
            for i in range(periods, len(factor_values)):
                factor_window = factor_values.iloc[i-periods:i]
                return_window = returns.iloc[i-periods:i]
                
                # 去除NaN值
                valid_idx = ~(factor_window.isna() | return_window.isna())
                if valid_idx.sum() > 10:  # 至少需要10个有效数据点
                    factor_clean = factor_window[valid_idx]
                    return_clean = return_window[valid_idx]
                    
                    # 信息系数
                    ic = factor_clean.corr(return_clean)
                    ic_series.append(ic)
                    
                    # 排序信息系数
                    rank_ic = factor_clean.rank().corr(return_clean.rank())
                    rank_ic_series.append(rank_ic)
                    
            ic_series = pd.Series(ic_series).dropna()
            rank_ic_series = pd.Series(rank_ic_series).dropna()
            
            # 计算性能指标
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            rank_ic_mean = rank_ic_series.mean()
            win_rate = (ic_series > 0).mean()
            
            # 构建因子组合收益
            factor_returns = self._calculate_factor_returns(factor_values, returns)
            annual_return = factor_returns.mean() * 252
            volatility = factor_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(factor_returns.cumsum())
            
            return FactorPerformance(
                factor_name="",
                ic=ic_mean,
                ic_ir=ic_ir,
                rank_ic=rank_ic_mean,
                win_rate=win_rate,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            self.logger.error(f"评估因子表现失败: {e}")
            return FactorPerformance("", 0, 0, 0, 0, 0, 0, 0, 0)
            
    def _calculate_factor_returns(self, factor_values: pd.Series, returns: pd.Series) -> pd.Series:
        """计算因子收益"""
        try:
            # 简单的因子收益计算：基于因子值的符号
            factor_signals = np.sign(factor_values.shift(1))  # 使用前一期因子值作为信号
            factor_returns = factor_signals * returns
            return factor_returns.dropna()
        except Exception as e:
            self.logger.error(f"计算因子收益失败: {e}")
            return pd.Series()
            
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """计算最大回撤"""
        try:
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            return drawdown.min()
        except Exception as e:
            self.logger.error(f"计算最大回撤失败: {e}")
            return 0.0
            
    def factor_selection(self, factors_df: pd.DataFrame, returns: pd.Series, 
                        method: str = 'ic') -> List[str]:
        """因子选择"""
        try:
            factor_scores = {}
            
            for factor_name in factors_df.columns:
                factor_values = factors_df[factor_name]
                performance = self.evaluate_factor_performance(factor_values, returns)
                
                if method == 'ic':
                    factor_scores[factor_name] = abs(performance.ic)
                elif method == 'ic_ir':
                    factor_scores[factor_name] = abs(performance.ic_ir)
                elif method == 'sharpe':
                    factor_scores[factor_name] = abs(performance.sharpe_ratio)
                    
            # 按得分排序
            sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
            return [factor[0] for factor in sorted_factors[:10]]  # 返回前10个因子
            
        except Exception as e:
            self.logger.error(f"因子选择失败: {e}")
            return []
            
    def factor_combination(self, factors_df: pd.DataFrame, returns: pd.Series, 
                          method: str = 'equal_weight') -> pd.Series:
        """因子组合"""
        try:
            if method == 'equal_weight':
                # 等权重组合
                combined_factor = factors_df.mean(axis=1)
            elif method == 'ic_weight':
                # IC加权组合
                weights = {}
                for factor_name in factors_df.columns:
                    factor_values = factors_df[factor_name]
                    performance = self.evaluate_factor_performance(factor_values, returns)
                    weights[factor_name] = abs(performance.ic)
                    
                # 归一化权重
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v/total_weight for k, v in weights.items()}
                    combined_factor = sum(factors_df[factor] * weight 
                                        for factor, weight in weights.items())
                else:
                    combined_factor = factors_df.mean(axis=1)
            elif method == 'pca':
                # 主成分分析
                scaler = StandardScaler()
                factors_scaled = scaler.fit_transform(factors_df.fillna(0))
                pca = PCA(n_components=1)
                combined_factor = pd.Series(pca.fit_transform(factors_scaled).flatten(), 
                                          index=factors_df.index)
            else:
                combined_factor = factors_df.mean(axis=1)
                
            return combined_factor
            
        except Exception as e:
            self.logger.error(f"因子组合失败: {e}")
            return pd.Series()
            
    def get_factor_summary(self, symbol: str) -> Dict[str, Any]:
        """获取因子摘要"""
        try:
            if symbol not in self.factor_cache:
                return {}
                
            factors = self.factor_cache[symbol]
            summary = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'factor_count': len(factors),
                'latest_values': {},
                'factor_categories': {
                    'momentum': [],
                    'mean_reversion': [],
                    'volatility': [],
                    'volume': [],
                    'trend': []
                }
            }
            
            # 最新因子值
            for factor_name, factor_series in factors.items():
                if not factor_series.empty:
                    summary['latest_values'][factor_name] = factor_series.iloc[-1]
                    
                    # 分类因子
                    if 'momentum' in factor_name or 'rsi' in factor_name:
                        summary['factor_categories']['momentum'].append(factor_name)
                    elif 'bb_' in factor_name or 'price_to_' in factor_name:
                        summary['factor_categories']['mean_reversion'].append(factor_name)
                    elif 'volatility' in factor_name or 'atr' in factor_name:
                        summary['factor_categories']['volatility'].append(factor_name)
                    elif 'volume' in factor_name or 'obv' in factor_name:
                        summary['factor_categories']['volume'].append(factor_name)
                    elif 'macd' in factor_name or 'adx' in factor_name:
                        summary['factor_categories']['trend'].append(factor_name)
                        
            return summary
            
        except Exception as e:
            self.logger.error(f"获取因子摘要失败: {e}")
            return {}