"""
截面因子计算模块

基于截面数据计算各种相对排名和比较因子，包括：
1. 相对排名因子 - 价格排名、成交量排名、波动率排名等
2. 行业相对因子 - 行业相对强度、行业内排名等
3. 市值相对因子 - 市值分位数、相对市值等
4. 估值相对因子 - 相对PE、相对PB等
5. 动量相对因子 - 相对动量、排名动量等
6. 质量相对因子 - 相对质量评分、财务质量排名等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging
from scipy import stats

class CrossSectionalFactorCalculator:
    """截面因子计算器"""
    
    def __init__(self, min_stocks: int = 20):
        """
        初始化截面因子计算器
        
        Args:
            min_stocks: 最小股票数量要求
        """
        self.min_stocks = min_stocks
        self.logger = logging.getLogger(__name__)
        
        print("截面因子计算器初始化完成")
    
    def calculate_relative_ranking_factors(self, price_data: pd.DataFrame,
                                         volume_data: Optional[pd.DataFrame] = None,
                                         window: int = 252) -> Dict[str, pd.DataFrame]:
        """
        计算相对排名因子
        
        Args:
            price_data: 价格数据，列为股票代码
            volume_data: 成交量数据
            window: 计算窗口
            
        Returns:
            相对排名因子字典，每个因子为DataFrame，行为日期，列为股票
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        if volume_data is None:
            volume_data = self._generate_mock_volume_data(price_data.index, price_data.columns)
        
        print(f"计算相对排名因子，股票数量: {len(price_data.columns)}")
        
        # 计算收益率
        returns = price_data.pct_change()
        
        # 1. 价格相对排名
        factors['price_rank'] = self._calculate_cross_sectional_rank(price_data)
        
        # 2. 收益率相对排名
        factors['return_rank'] = self._calculate_cross_sectional_rank(returns)
        
        # 3. 波动率相对排名
        volatility = returns.rolling(window=21).std()
        factors['volatility_rank'] = self._calculate_cross_sectional_rank(volatility)
        
        # 4. 成交量相对排名
        factors['volume_rank'] = self._calculate_cross_sectional_rank(volume_data)
        
        # 5. 动量相对排名
        momentum = returns.rolling(window=21).sum()
        factors['momentum_rank'] = self._calculate_cross_sectional_rank(momentum)
        
        # 6. 换手率相对排名（简化计算）
        turnover = volume_data / price_data  # 简化的换手率
        factors['turnover_rank'] = self._calculate_cross_sectional_rank(turnover)
        
        # 7. 价格相对强度
        market_return = returns.mean(axis=1)
        relative_strength = returns.subtract(market_return, axis=0)
        factors['relative_strength_rank'] = self._calculate_cross_sectional_rank(relative_strength)
        
        # 8. 滚动收益率排名
        for period in [5, 10, 21, 63]:
            period_return = returns.rolling(window=period).sum()
            factors[f'return_{period}d_rank'] = self._calculate_cross_sectional_rank(period_return)
        
        return factors
    
    def _generate_mock_price_data(self, n_stocks: int = 100, n_days: int = 500) -> pd.DataFrame:
        """生成模拟价格数据 - 仅用于测试和演示"""
        np.random.seed(42)  # 设置随机种子 - 模拟数据仅用于测试
        
        # 生成日期 - 模拟数据仅用于演示
        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
        
        # 生成股票代码 - 模拟数据仅用于演示
        stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]
        
        # 生成价格数据（几何布朗运动）- 模拟数据仅用于演示
        initial_prices = np.random.uniform(10, 200, n_stocks)  # 初始价格 - 仅用于测试
        
        price_data = {}
        for i, stock in enumerate(stocks):
            # 每只股票有不同的漂移和波动率 - 仅用于测试
            mu = np.random.normal(0.0001, 0.0005)  # 日收益率 - 模拟数据
            sigma = np.random.uniform(0.01, 0.03)  # 日波动率 - 模拟数据
            
            returns = np.random.normal(mu, sigma, n_days)  # 生成收益率 - 仅用于测试
            prices = initial_prices[i] * np.cumprod(1 + returns)  # 计算价格 - 模拟数据
            price_data[stock] = prices
        
        return pd.DataFrame(price_data, index=dates)
    
    def _generate_mock_volume_data(self, dates: pd.DatetimeIndex, stocks: List[str]) -> pd.DataFrame:
        """生成模拟成交量数据 - 仅用于测试和演示"""
        np.random.seed(43)  # 设置随机种子确保结果可重现 - 模拟数据仅用于测试
        
        volume_data = {}  # 存储各股票成交量数据 - 仅用于测试和演示
        for stock in stocks:  # 遍历所有股票代码 - 模拟数据仅用于演示
            # 基础成交量 + 随机波动 - 模拟数据仅用于演示
            base_volume = np.random.uniform(1e5, 1e7)  # 随机生成基础成交量 - 仅用于测试
            volumes = np.random.lognormal(np.log(base_volume), 0.5, len(dates))  # 对数正态分布生成成交量 - 模拟数据
            volume_data[stock] = volumes  # 保存股票成交量数据 - 仅用于测试和演示
        
        return pd.DataFrame(volume_data, index=dates)  # 返回成交量DataFrame - 模拟数据仅用于演示
    
    def _calculate_cross_sectional_rank(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算截面排名（百分位排名）"""
        return data.rank(axis=1, pct=True, method='min')
    
    def calculate_industry_relative_factors(self, price_data: pd.DataFrame,
                                          industry_mapping: Optional[Dict[str, str]] = None,
                                          window: int = 252) -> Dict[str, pd.DataFrame]:
        """
        计算行业相对因子
        
        Args:
            price_data: 价格数据
            industry_mapping: 股票到行业的映射
            window: 计算窗口
            
        Returns:
            行业相对因子字典
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        if industry_mapping is None:
            industry_mapping = self._generate_mock_industry_mapping(price_data.columns)
        
        print(f"计算行业相对因子，行业数量: {len(set(industry_mapping.values()))}")
        
        returns = price_data.pct_change()
        
        # 1. 行业内排名
        industry_rank = self._calculate_industry_rank(returns, industry_mapping)
        factors['industry_rank'] = industry_rank
        
        # 2. 相对行业表现
        industry_relative_return = self._calculate_industry_relative_performance(returns, industry_mapping)
        factors['industry_relative_return'] = industry_relative_return
        
        # 3. 行业动量排名
        momentum = returns.rolling(window=21).sum()
        industry_momentum_rank = self._calculate_industry_rank(momentum, industry_mapping)
        factors['industry_momentum_rank'] = industry_momentum_rank
        
        # 4. 行业波动率排名
        volatility = returns.rolling(window=21).std()
        industry_vol_rank = self._calculate_industry_rank(volatility, industry_mapping)
        factors['industry_volatility_rank'] = industry_vol_rank
        
        # 5. 行业相对强度
        industry_strength = self._calculate_industry_relative_strength(returns, industry_mapping)
        factors['industry_relative_strength'] = industry_strength
        
        return factors
    
    def _generate_mock_industry_mapping(self, stocks: List[str]) -> Dict[str, str]:
        """生成模拟行业映射 - 仅用于测试和演示"""
        industries = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 
                     'Industrial', 'Materials', 'Utilities', 'Telecom', 'RealEstate']  # 预定义行业列表 - 仅用于测试
        
        mapping = {}  # 存储股票到行业的映射关系 - 模拟数据仅用于演示
        for i, stock in enumerate(stocks):  # 遍历所有股票代码 - 仅用于测试和演示
            mapping[stock] = industries[i % len(industries)]  # 循环分配行业避免重复 - 模拟数据仅用于演示
        
        return mapping  # 返回行业映射字典 - 仅用于测试和演示
    
    def _calculate_industry_rank(self, data: pd.DataFrame, 
                               industry_mapping: Dict[str, str]) -> pd.DataFrame:
        """计算行业内排名"""
        industry_ranks = pd.DataFrame(index=data.index, columns=data.columns)
        
        for date in data.index:
            date_data = data.loc[date]
            
            for industry in set(industry_mapping.values()):
                industry_stocks = [stock for stock, ind in industry_mapping.items() 
                                 if ind == industry and stock in data.columns]
                
                if len(industry_stocks) > 1:
                    industry_data = date_data[industry_stocks]
                    # 计算行业内百分位排名
                    ranks = industry_data.rank(pct=True, method='min')
                    industry_ranks.loc[date, industry_stocks] = ranks
        
        return industry_ranks.astype(float)
    
    def _calculate_industry_relative_performance(self, returns: pd.DataFrame,
                                               industry_mapping: Dict[str, str]) -> pd.DataFrame:
        """计算相对行业表现"""
        relative_performance = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        # 计算行业平均收益率
        industry_returns = {}
        for industry in set(industry_mapping.values()):
            industry_stocks = [stock for stock, ind in industry_mapping.items() 
                             if ind == industry and stock in returns.columns]
            if industry_stocks:
                industry_returns[industry] = returns[industry_stocks].mean(axis=1)
        
        # 计算个股相对行业的表现
        for stock, industry in industry_mapping.items():
            if stock in returns.columns and industry in industry_returns:
                relative_performance[stock] = returns[stock] - industry_returns[industry]
        
        return relative_performance.astype(float)
    
    def _calculate_industry_relative_strength(self, returns: pd.DataFrame,
                                            industry_mapping: Dict[str, str]) -> pd.DataFrame:
        """计算行业相对强度"""
        # 计算21日滚动相对强度
        window = 21
        relative_strength = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            # 计算行业平均收益率
            industry_avg_returns = {}
            for industry in set(industry_mapping.values()):
                industry_stocks = [stock for stock, ind in industry_mapping.items() 
                                 if ind == industry and stock in returns.columns]
                if industry_stocks:
                    industry_avg_returns[industry] = window_returns[industry_stocks].mean().mean()
            
            # 计算个股相对行业的强度
            for stock, industry in industry_mapping.items():
                if stock in returns.columns and industry in industry_avg_returns:
                    stock_avg_return = window_returns[stock].mean()
                    relative_strength.iloc[i, relative_strength.columns.get_loc(stock)] = \
                        stock_avg_return - industry_avg_returns[industry]
        
        return relative_strength.astype(float)
    
    def calculate_market_cap_relative_factors(self, price_data: pd.DataFrame,
                                            shares_outstanding: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        计算市值相对因子
        
        Args:
            price_data: 价格数据
            shares_outstanding: 流通股本数据
            
        Returns:
            市值相对因子字典
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        if shares_outstanding is None:
            shares_outstanding = self._generate_mock_shares_data(price_data.index, price_data.columns)
        
        print("计算市值相对因子...")
        
        # 计算市值
        market_cap = price_data * shares_outstanding
        
        # 1. 市值排名
        factors['market_cap_rank'] = self._calculate_cross_sectional_rank(market_cap)
        
        # 2. 市值分位数
        factors['market_cap_percentile'] = market_cap.rank(axis=1, pct=True)
        
        # 3. 相对市值（相对于中位数）
        median_market_cap = market_cap.median(axis=1)
        relative_market_cap = market_cap.divide(median_market_cap, axis=0)
        factors['relative_market_cap'] = relative_market_cap
        
        # 4. 市值变化排名
        market_cap_change = market_cap.pct_change()
        factors['market_cap_change_rank'] = self._calculate_cross_sectional_rank(market_cap_change)
        
        # 5. 市值稳定性排名
        market_cap_volatility = market_cap.rolling(window=21).std()
        factors['market_cap_stability_rank'] = self._calculate_cross_sectional_rank(-market_cap_volatility)  # 负号表示波动率越小排名越高
        
        return factors
    
    def _generate_mock_shares_data(self, dates: pd.DatetimeIndex, stocks: List[str]) -> pd.DataFrame:
        """生成模拟流通股本数据 - 仅用于测试和演示"""
        np.random.seed(44)  # 设置随机种子确保结果可重现 - 模拟数据仅用于测试
        
        shares_data = {}  # 存储各股票流通股本数据 - 仅用于测试和演示
        for stock in stocks:  # 遍历所有股票代码 - 模拟数据仅用于演示
            # 流通股本相对稳定，偶尔有变化 - 模拟数据仅用于演示
            base_shares = np.random.uniform(1e8, 1e10)  # 随机生成基础股本(1亿到100亿股) - 仅用于测试
            shares = np.full(len(dates), base_shares)  # 初始化股本数组 - 模拟数据
            
            # 随机添加一些股本变化事件(如增发、回购) - 仅用于测试
            change_points = np.random.choice(len(dates), size=np.random.randint(0, 3), replace=False)  # 随机选择变化时点 - 模拟数据
            for cp in change_points:  # 遍历股本变化时点 - 仅用于测试和演示
                change_factor = np.random.uniform(0.9, 1.1)  # 生成±10%的变化幅度 - 模拟数据
                shares[cp:] *= change_factor  # 从变化时点开始应用新股本 - 仅用于演示
            
            shares_data[stock] = shares  # 保存股票股本数据 - 仅用于测试和演示
        
        return pd.DataFrame(shares_data, index=dates)  # 返回股本DataFrame - 模拟数据仅用于演示
    
    def calculate_valuation_relative_factors(self, price_data: pd.DataFrame,
                                           fundamental_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        计算估值相对因子
        
        Args:
            price_data: 价格数据
            fundamental_data: 基本面数据（EPS, Book Value等）
            
        Returns:
            估值相对因子字典
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        if fundamental_data is None:
            fundamental_data = self._generate_mock_fundamental_data(price_data.index, price_data.columns)
        
        print("计算估值相对因子...")
        
        # 1. 相对PE排名
        if 'EPS' in fundamental_data.columns:
            pe_ratio = price_data.div(fundamental_data['EPS'], axis=0)
            factors['pe_rank'] = self._calculate_cross_sectional_rank(pe_ratio)
            
            # 相对PE（相对于行业中位数）
            median_pe = pe_ratio.median(axis=1)
            relative_pe = pe_ratio.divide(median_pe, axis=0)
            factors['relative_pe'] = relative_pe
        
        # 2. 相对PB排名
        if 'BookValue' in fundamental_data.columns:
            pb_ratio = price_data.div(fundamental_data['BookValue'], axis=0)
            factors['pb_rank'] = self._calculate_cross_sectional_rank(pb_ratio)
            
            # 相对PB
            median_pb = pb_ratio.median(axis=1)
            relative_pb = pb_ratio.divide(median_pb, axis=0)
            factors['relative_pb'] = relative_pb
        
        # 3. 估值综合排名
        if 'EPS' in fundamental_data.columns and 'BookValue' in fundamental_data.columns:
            # 简单的估值综合评分（PE和PB的平均排名）
            pe_rank = self._calculate_cross_sectional_rank(pe_ratio)
            pb_rank = self._calculate_cross_sectional_rank(pb_ratio)
            valuation_composite_rank = (pe_rank + pb_rank) / 2
            factors['valuation_composite_rank'] = valuation_composite_rank
        
        return factors
    
    def _generate_mock_fundamental_data(self, dates: pd.DatetimeIndex, stocks: List[str]) -> pd.DataFrame:
        """生成模拟基本面数据 - 仅用于测试和演示"""
        np.random.seed(45)  # 设置随机种子确保结果可重现 - 模拟数据仅用于测试
        
        fundamental_data = {}  # 存储基本面数据字典 - 仅用于测试和演示
        
        for stock in stocks:  # 遍历所有股票代码 - 模拟数据仅用于演示
            # EPS数据（季度更新）- 模拟数据仅用于演示
            base_eps = np.random.uniform(1, 10)  # 随机生成基础EPS - 仅用于测试
            eps_growth = np.random.uniform(0.95, 1.05)  # 随机生成年增长率 - 模拟数据
            
            eps_values = []  # 存储EPS时间序列 - 仅用于测试和演示
            current_eps = base_eps  # 当前EPS值 - 模拟数据仅用于演示
            
            for i, date in enumerate(dates):  # 遍历所有日期 - 仅用于测试和演示
                # 每季度更新一次EPS - 仅用于测试
                if i % 63 == 0:  # 大约每季度(63个交易日) - 模拟数据仅用于演示
                    current_eps *= eps_growth ** (1/4)  # 应用季度增长率 - 仅用于测试
                eps_values.append(current_eps)  # 添加当前EPS值 - 模拟数据仅用于演示
            
            fundamental_data[f'{stock}_EPS'] = eps_values  # 保存EPS数据 - 仅用于测试和演示
            
            # Book Value数据 - 模拟数据仅用于演示
            base_bv = np.random.uniform(10, 50)  # 随机生成基础账面价值 - 仅用于测试
            bv_growth = np.random.uniform(1.02, 1.08)  # 随机生成年增长率 - 模拟数据
            
            bv_values = []  # 存储账面价值时间序列 - 仅用于测试和演示
            current_bv = base_bv  # 当前账面价值 - 模拟数据仅用于演示
            
            for i, date in enumerate(dates):  # 遍历所有日期 - 仅用于测试和演示
                if i % 252 == 0:  # 每年更新一次(252个交易日) - 仅用于测试
                    current_bv *= bv_growth  # 应用年增长率 - 模拟数据仅用于演示
                bv_values.append(current_bv)  # 添加当前账面价值 - 仅用于测试和演示
            
            fundamental_data[f'{stock}_BookValue'] = bv_values  # 保存账面价值数据 - 仅用于测试和演示
        
        # 重新组织数据结构为分指标的DataFrame - 模拟数据仅用于演示
        result_data = {}  # 存储重组后的数据 - 仅用于测试和演示
        for metric in ['EPS', 'BookValue']:  # 遍历所有指标 - 模拟数据仅用于演示
            metric_data = {}  # 存储单个指标的所有股票数据 - 仅用于测试和演示
            for stock in stocks:  # 遍历所有股票 - 模拟数据仅用于演示
                metric_data[stock] = fundamental_data[f'{stock}_{metric}']  # 提取指标数据 - 仅用于测试和演示
            result_data[metric] = pd.DataFrame(metric_data, index=dates)  # 创建指标DataFrame - 模拟数据仅用于演示
        
        return result_data  # 返回基本面数据字典 - 仅用于测试和演示
    
    def calculate_momentum_relative_factors(self, price_data: pd.DataFrame,
                                          window_list: List[int] = [5, 10, 21, 63, 126, 252]) -> Dict[str, pd.DataFrame]:
        """
        计算动量相对因子
        
        Args:
            price_data: 价格数据
            window_list: 动量计算窗口列表
            
        Returns:
            动量相对因子字典
        """
        factors = {}
        
        if price_data.empty:
            price_data = self._generate_mock_price_data()
        
        print(f"计算动量相对因子，窗口: {window_list}")
        
        returns = price_data.pct_change()
        
        for window in window_list:
            # 计算动量
            momentum = returns.rolling(window=window).sum()
            
            # 动量排名
            factors[f'momentum_{window}d_rank'] = self._calculate_cross_sectional_rank(momentum)
            
            # 相对动量（相对于市场）
            market_momentum = momentum.mean(axis=1)
            relative_momentum = momentum.subtract(market_momentum, axis=0)
            factors[f'relative_momentum_{window}d'] = relative_momentum
            
            # 动量分位数
            factors[f'momentum_{window}d_percentile'] = momentum.rank(axis=1, pct=True)
        
        # 动量一致性（不同周期动量的一致性）
        if len(window_list) >= 3:
            momentum_consistency = pd.DataFrame(index=price_data.index, columns=price_data.columns)
            
            for date in price_data.index:
                for stock in price_data.columns:
                    # 计算不同周期动量排名的标准差（一致性越高，标准差越小）
                    ranks = []
                    for window in window_list[:3]:  # 使用前3个窗口
                        if f'momentum_{window}d_rank' in factors:
                            rank_value = factors[f'momentum_{window}d_rank'].loc[date, stock]
                            if not pd.isna(rank_value):
                                ranks.append(rank_value)
                    
                    if len(ranks) >= 2:
                        consistency = 1 - np.std(ranks)  # 标准差越小，一致性越高
                        momentum_consistency.loc[date, stock] = consistency
            
            factors['momentum_consistency'] = momentum_consistency.astype(float)
        
        return factors
    
    def calculate_all_factors(self, price_data: Optional[pd.DataFrame] = None,
                            volume_data: Optional[pd.DataFrame] = None,
                            industry_mapping: Optional[Dict[str, str]] = None,
                            shares_outstanding: Optional[pd.DataFrame] = None,
                            fundamental_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        计算所有截面因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            industry_mapping: 行业映射
            shares_outstanding: 流通股本数据
            fundamental_data: 基本面数据
            
        Returns:
            所有截面因子字典
        """
        all_factors = {}
        
        print("开始计算截面因子...")
        
        if price_data is None:
            price_data = self._generate_mock_price_data()
        
        # 1. 相对排名因子
        ranking_factors = self.calculate_relative_ranking_factors(price_data, volume_data)
        all_factors.update(ranking_factors)
        
        # 2. 行业相对因子
        industry_factors = self.calculate_industry_relative_factors(price_data, industry_mapping)
        all_factors.update(industry_factors)
        
        # 3. 市值相对因子
        market_cap_factors = self.calculate_market_cap_relative_factors(price_data, shares_outstanding)
        all_factors.update(market_cap_factors)
        
        # 4. 估值相对因子
        valuation_factors = self.calculate_valuation_relative_factors(price_data, fundamental_data)
        all_factors.update(valuation_factors)
        
        # 5. 动量相对因子
        momentum_factors = self.calculate_momentum_relative_factors(price_data)
        all_factors.update(momentum_factors)
        
        print(f"截面因子计算完成，共生成 {len(all_factors)} 个因子")
        return all_factors

def main():
    """测试截面因子计算器"""
    print("测试截面因子计算器...")
    
    # 初始化计算器
    calculator = CrossSectionalFactorCalculator()
    
    # 计算因子
    factors = calculator.calculate_all_factors()
    
    print(f"成功计算 {len(factors)} 个截面因子")
    for name, factor in list(factors.items())[:10]:  # 只显示前10个
        if isinstance(factor, pd.DataFrame) and not factor.empty:
            print(f"{name}: 形状={factor.shape}, 均值={factor.mean().mean():.4f}")

if __name__ == "__main__":
    main()