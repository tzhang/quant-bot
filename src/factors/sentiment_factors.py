"""
情绪因子计算模块

该模块实现了多种情绪因子的计算，包括：
1. 市场情绪因子 - VIX、Put/Call比率、投资者情绪指数等
2. 技术情绪因子 - 超买超卖、背离、动量情绪等
3. 新闻情绪因子 - 新闻情绪分析、社交媒体情绪等
4. 资金流向因子 - 机构资金流向、散户情绪等
5. 期权情绪因子 - 期权偏斜、隐含波动率等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime, timedelta
import logging

# 尝试导入可选依赖
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    warnings.warn("yfinance not available. Some sentiment factors may not work.")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    warnings.warn("TextBlob not available. News sentiment analysis may not work.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    warnings.warn("requests not available. Some data fetching may not work.")

class SentimentFactorCalculator:
    """情绪因子计算器"""
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        初始化情绪因子计算器
        
        Args:
            lookback_periods: 各种因子的回看期设置
        """
        self.lookback_periods = lookback_periods or {
            'short': 5,
            'medium': 20,
            'long': 60,
            'sentiment': 10
        }
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_market_sentiment_factors(self, data: pd.DataFrame, 
                                         market_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        计算市场情绪因子
        
        Args:
            data: 股票价格数据，包含OHLCV
            market_data: 市场指数数据，包含VIX等
            
        Returns:
            Dict[str, pd.Series]: 市场情绪因子字典
        """
        factors = {}
        
        try:
            # 1. 恐慌指数相关因子
            if market_data is not None and 'VIX' in market_data.columns:
                factors['vix_level'] = market_data['VIX']
                factors['vix_change'] = market_data['VIX'].pct_change()
                factors['vix_zscore'] = (market_data['VIX'] - market_data['VIX'].rolling(60).mean()) / market_data['VIX'].rolling(60).std()
            
            # 2. 市场宽度因子
            if 'volume' in data.columns and 'close' in data.columns:
                # 成交量情绪
                volume_ma = data['volume'].rolling(self.lookback_periods['medium']).mean()
                factors['volume_sentiment'] = data['volume'] / volume_ma - 1
                
                # 价格动量情绪
                price_momentum = data['close'].pct_change(self.lookback_periods['sentiment'])
                factors['price_momentum_sentiment'] = price_momentum
                
            # 3. 市场参与度因子
            if all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                # 真实波动率
                tr = np.maximum(data['high'] - data['low'],
                               np.maximum(abs(data['high'] - data['close'].shift(1)),
                                        abs(data['low'] - data['close'].shift(1))))
                atr = tr.rolling(14).mean()
                factors['volatility_sentiment'] = atr / data['close']
                
                # 成交量加权情绪
                vwap = (data['high'] + data['low'] + data['close']) / 3 * data['volume']
                vwap_sum = vwap.rolling(self.lookback_periods['sentiment']).sum()
                volume_sum = data['volume'].rolling(self.lookback_periods['sentiment']).sum()
                factors['vwap_sentiment'] = (data['close'] - vwap_sum / volume_sum) / data['close']
                
        except Exception as e:
            self.logger.error(f"计算市场情绪因子时出错: {e}")
            
        return factors
    
    def calculate_technical_sentiment_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算技术情绪因子
        
        Args:
            data: 股票价格数据，包含OHLCV
            
        Returns:
            Dict[str, pd.Series]: 技术情绪因子字典
        """
        factors = {}
        
        try:
            if all(col in data.columns for col in ['high', 'low', 'close']):
                # 1. 超买超卖情绪
                # RSI情绪
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                factors['rsi_sentiment'] = np.where(rsi > 70, 1,  # 超买
                                                  np.where(rsi < 30, -1, 0))  # 超卖
                
                # 2. 随机指标情绪
                lowest_low = data['low'].rolling(14).min()
                highest_high = data['high'].rolling(14).max()
                k_percent = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
                d_percent = k_percent.rolling(3).mean()
                
                factors['stoch_sentiment'] = np.where(k_percent > 80, 1,
                                                    np.where(k_percent < 20, -1, 0))
                
                # 3. 布林带情绪
                sma_20 = data['close'].rolling(20).mean()
                std_20 = data['close'].rolling(20).std()
                upper_band = sma_20 + (std_20 * 2)
                lower_band = sma_20 - (std_20 * 2)
                
                factors['bollinger_sentiment'] = np.where(data['close'] > upper_band, 1,
                                                        np.where(data['close'] < lower_band, -1, 0))
                
                # 4. 威廉指标情绪
                williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
                factors['williams_sentiment'] = np.where(williams_r > -20, 1,
                                                       np.where(williams_r < -80, -1, 0))
                
            # 5. 动量情绪
            if 'close' in data.columns:
                # 短期动量情绪
                short_momentum = data['close'].pct_change(5)
                factors['short_momentum_sentiment'] = np.where(short_momentum > 0.05, 1,
                                                             np.where(short_momentum < -0.05, -1, 0))
                
                # 中期动量情绪
                medium_momentum = data['close'].pct_change(20)
                factors['medium_momentum_sentiment'] = np.where(medium_momentum > 0.1, 1,
                                                              np.where(medium_momentum < -0.1, -1, 0))
                
        except Exception as e:
            self.logger.error(f"计算技术情绪因子时出错: {e}")
            
        return factors
    
    def calculate_volume_sentiment_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算成交量情绪因子
        
        Args:
            data: 股票价格数据，包含OHLCV
            
        Returns:
            Dict[str, pd.Series]: 成交量情绪因子字典
        """
        factors = {}
        
        try:
            if all(col in data.columns for col in ['close', 'volume', 'high', 'low']):
                # 1. 成交量价格趋势 (VPT)
                vpt = (data['volume'] * data['close'].pct_change()).cumsum()
                factors['vpt_sentiment'] = vpt.pct_change(self.lookback_periods['sentiment'])
                
                # 2. 能量潮指标 (OBV)
                obv = (np.sign(data['close'].diff()) * data['volume']).cumsum()
                factors['obv_sentiment'] = obv.pct_change(self.lookback_periods['sentiment'])
                
                # 3. 成交量相对强弱
                volume_ma = data['volume'].rolling(self.lookback_periods['medium']).mean()
                factors['volume_strength'] = data['volume'] / volume_ma - 1
                
                # 4. 价量背离
                price_change = data['close'].pct_change(self.lookback_periods['sentiment'])
                volume_change = data['volume'].pct_change(self.lookback_periods['sentiment'])
                
                # 价涨量跌或价跌量涨为背离信号
                factors['price_volume_divergence'] = np.where(
                    (price_change > 0) & (volume_change < 0), -1,  # 价涨量跌，看跌背离
                    np.where((price_change < 0) & (volume_change > 0), 1, 0)  # 价跌量涨，看涨背离
                )
                
                # 5. 资金流量指数 (MFI)
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                money_flow = typical_price * data['volume']
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
                
                mfi = 100 - (100 / (1 + positive_flow / negative_flow))
                factors['mfi_sentiment'] = np.where(mfi > 80, 1,
                                                  np.where(mfi < 20, -1, 0))
                
        except Exception as e:
            self.logger.error(f"计算成交量情绪因子时出错: {e}")
            
        return factors
    
    def calculate_news_sentiment_factors(self, news_data: Optional[List[Dict]] = None,
                                       symbol: str = None) -> Dict[str, pd.Series]:
        """
        计算新闻情绪因子
        
        Args:
            news_data: 新闻数据列表，每个元素包含{'date', 'title', 'content', 'source'}
            symbol: 股票代码
            
        Returns:
            Dict[str, pd.Series]: 新闻情绪因子字典
        """
        factors = {}
        
        try:
            if news_data and HAS_TEXTBLOB:
                # 处理新闻情绪分析
                sentiment_scores = []
                dates = []
                
                for news in news_data:
                    try:
                        # 分析标题和内容的情绪
                        title_sentiment = TextBlob(news.get('title', '')).sentiment.polarity
                        content_sentiment = TextBlob(news.get('content', '')).sentiment.polarity
                        
                        # 综合情绪分数（标题权重更高）
                        combined_sentiment = 0.7 * title_sentiment + 0.3 * content_sentiment
                        
                        sentiment_scores.append(combined_sentiment)
                        dates.append(pd.to_datetime(news['date']))
                        
                    except Exception as e:
                        self.logger.warning(f"处理新闻情绪时出错: {e}")
                        continue
                
                if sentiment_scores:
                    # 创建情绪时间序列
                    sentiment_series = pd.Series(sentiment_scores, index=dates)
                    sentiment_daily = sentiment_series.resample('D').mean()
                    
                    factors['news_sentiment'] = sentiment_daily
                    factors['news_sentiment_ma'] = sentiment_daily.rolling(5).mean()
                    factors['news_sentiment_volatility'] = sentiment_daily.rolling(10).std()
                    
            else:
                # 如果没有新闻数据，创建模拟的情绪因子
                self.logger.warning("没有新闻数据或TextBlob不可用，创建模拟情绪因子")
                
                # 基于价格波动创建模拟情绪
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                mock_sentiment = np.random.normal(0, 0.3, len(dates))
                factors['mock_news_sentiment'] = pd.Series(mock_sentiment, index=dates)
                
        except Exception as e:
            self.logger.error(f"计算新闻情绪因子时出错: {e}")
            
        return factors
    
    def calculate_options_sentiment_factors(self, options_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        计算期权情绪因子
        
        Args:
            options_data: 期权数据，包含看涨看跌期权的成交量、未平仓合约等
            
        Returns:
            Dict[str, pd.Series]: 期权情绪因子字典
        """
        factors = {}
        
        try:
            if options_data is not None:
                # 1. Put/Call比率
                if all(col in options_data.columns for col in ['put_volume', 'call_volume']):
                    pc_ratio = options_data['put_volume'] / options_data['call_volume']
                    factors['put_call_ratio'] = pc_ratio
                    factors['put_call_ratio_ma'] = pc_ratio.rolling(10).mean()
                    
                    # Put/Call比率情绪信号
                    pc_ma = pc_ratio.rolling(20).mean()
                    pc_std = pc_ratio.rolling(20).std()
                    factors['pc_ratio_sentiment'] = np.where(pc_ratio > pc_ma + pc_std, -1,  # 看跌情绪
                                                           np.where(pc_ratio < pc_ma - pc_std, 1, 0))  # 看涨情绪
                
                # 2. 隐含波动率偏斜
                if all(col in options_data.columns for col in ['call_iv', 'put_iv']):
                    iv_skew = options_data['put_iv'] - options_data['call_iv']
                    factors['iv_skew'] = iv_skew
                    factors['iv_skew_sentiment'] = np.where(iv_skew > iv_skew.rolling(20).mean() + iv_skew.rolling(20).std(), -1, 1)
                
                # 3. 期权未平仓合约比率
                if all(col in options_data.columns for col in ['put_oi', 'call_oi']):
                    oi_ratio = options_data['put_oi'] / options_data['call_oi']
                    factors['oi_ratio'] = oi_ratio
                    factors['oi_ratio_sentiment'] = np.where(oi_ratio > oi_ratio.rolling(20).mean(), -1, 1)
                    
            else:
                # 创建模拟期权情绪因子
                self.logger.warning("没有期权数据，创建模拟期权情绪因子")
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                
                # 模拟Put/Call比率
                mock_pc_ratio = np.random.lognormal(0, 0.3, len(dates))
                factors['mock_put_call_ratio'] = pd.Series(mock_pc_ratio, index=dates)
                
        except Exception as e:
            self.logger.error(f"计算期权情绪因子时出错: {e}")
            
        return factors
    
    def calculate_social_sentiment_factors(self, social_data: Optional[List[Dict]] = None) -> Dict[str, pd.Series]:
        """
        计算社交媒体情绪因子
        
        Args:
            social_data: 社交媒体数据，包含提及次数、情绪分数等
            
        Returns:
            Dict[str, pd.Series]: 社交媒体情绪因子字典
        """
        factors = {}
        
        try:
            if social_data:
                # 处理社交媒体情绪
                mention_counts = []
                sentiment_scores = []
                dates = []
                
                for data in social_data:
                    mention_counts.append(data.get('mention_count', 0))
                    sentiment_scores.append(data.get('sentiment_score', 0))
                    dates.append(pd.to_datetime(data['date']))
                
                # 创建时间序列
                mentions_series = pd.Series(mention_counts, index=dates)
                sentiment_series = pd.Series(sentiment_scores, index=dates)
                
                factors['social_mentions'] = mentions_series
                factors['social_sentiment'] = sentiment_series
                factors['social_buzz'] = mentions_series.rolling(7).mean()
                factors['social_sentiment_trend'] = sentiment_series.rolling(7).mean()
                
            else:
                # 创建模拟社交情绪因子
                self.logger.warning("没有社交媒体数据，创建模拟社交情绪因子")
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                
                mock_mentions = np.random.poisson(100, len(dates))
                mock_sentiment = np.random.normal(0, 0.5, len(dates))
                
                factors['mock_social_mentions'] = pd.Series(mock_mentions, index=dates)
                factors['mock_social_sentiment'] = pd.Series(mock_sentiment, index=dates)
                
        except Exception as e:
            self.logger.error(f"计算社交媒体情绪因子时出错: {e}")
            
        return factors
    
    def calculate_all_factors(self, data: pd.DataFrame, 
                            market_data: Optional[pd.DataFrame] = None,
                            news_data: Optional[List[Dict]] = None,
                            options_data: Optional[pd.DataFrame] = None,
                            social_data: Optional[List[Dict]] = None) -> Dict[str, pd.Series]:
        """
        计算所有情绪因子
        
        Args:
            data: 股票价格数据
            market_data: 市场数据
            news_data: 新闻数据
            options_data: 期权数据
            social_data: 社交媒体数据
            
        Returns:
            Dict[str, pd.Series]: 所有情绪因子字典
        """
        all_factors = {}
        
        # 计算各类情绪因子
        market_factors = self.calculate_market_sentiment_factors(data, market_data)
        technical_factors = self.calculate_technical_sentiment_factors(data)
        volume_factors = self.calculate_volume_sentiment_factors(data)
        news_factors = self.calculate_news_sentiment_factors(news_data)
        options_factors = self.calculate_options_sentiment_factors(options_data)
        social_factors = self.calculate_social_sentiment_factors(social_data)
        
        # 合并所有因子
        all_factors.update(market_factors)
        all_factors.update(technical_factors)
        all_factors.update(volume_factors)
        all_factors.update(news_factors)
        all_factors.update(options_factors)
        all_factors.update(social_factors)
        
        return all_factors
    
    def calculate_factor_scores(self, factors: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        计算因子评分
        
        Args:
            factors: 因子字典
            
        Returns:
            Dict[str, float]: 因子评分字典
        """
        scores = {}
        
        for name, factor in factors.items():
            if factor is not None and len(factor) > 0:
                # 计算因子的统计特征
                mean_val = factor.mean()
                std_val = factor.std()
                skew_val = factor.skew() if hasattr(factor, 'skew') else 0
                
                # 综合评分（考虑均值、标准差和偏度）
                score = abs(mean_val) * 0.4 + std_val * 0.4 + abs(skew_val) * 0.2
                scores[name] = score
            else:
                scores[name] = 0.0
                
        return scores
    
    def generate_factor_report(self, factors: Dict[str, pd.Series], 
                             scores: Dict[str, float]) -> str:
        """
        生成情绪因子分析报告
        
        Args:
            factors: 因子字典
            scores: 因子评分字典
            
        Returns:
            str: 分析报告
        """
        report = []
        report.append("=" * 60)
        report.append("情绪因子分析报告")
        report.append("=" * 60)
        
        # 因子概览
        report.append(f"\n总因子数量: {len(factors)}")
        report.append(f"有效因子数量: {sum(1 for f in factors.values() if f is not None and len(f) > 0)}")
        
        # 按类别分组
        categories = {
            '市场情绪': ['vix_', 'volume_sentiment', 'price_momentum_sentiment', 'volatility_sentiment', 'vwap_sentiment'],
            '技术情绪': ['rsi_sentiment', 'stoch_sentiment', 'bollinger_sentiment', 'williams_sentiment', 'momentum_sentiment'],
            '成交量情绪': ['vpt_sentiment', 'obv_sentiment', 'volume_strength', 'price_volume_divergence', 'mfi_sentiment'],
            '新闻情绪': ['news_sentiment', 'mock_news_sentiment'],
            '期权情绪': ['put_call_ratio', 'iv_skew', 'oi_ratio', 'mock_put_call_ratio'],
            '社交情绪': ['social_mentions', 'social_sentiment', 'mock_social_mentions', 'mock_social_sentiment']
        }
        
        for category, factor_names in categories.items():
            category_factors = [name for name in factors.keys() 
                              if any(fn in name for fn in factor_names)]
            if category_factors:
                report.append(f"\n{category} ({len(category_factors)}个因子):")
                for factor_name in category_factors:
                    score = scores.get(factor_name, 0)
                    report.append(f"  - {factor_name}: {score:.4f}")
        
        # 顶级因子
        top_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        report.append(f"\n顶级情绪因子 (Top 10):")
        for i, (name, score) in enumerate(top_factors, 1):
            report.append(f"  {i}. {name}: {score:.4f}")
        
        # 统计摘要
        if scores:
            avg_score = np.mean(list(scores.values()))
            max_score = max(scores.values())
            min_score = min(scores.values())
            
            report.append(f"\n因子评分统计:")
            report.append(f"  平均评分: {avg_score:.4f}")
            report.append(f"  最高评分: {max_score:.4f}")
            report.append(f"  最低评分: {min_score:.4f}")
        
        return "\n".join(report)


def main():
    """示例用法"""
    # 创建示例数据
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # 模拟股票数据
    np.random.seed(42)
    price_base = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = price_base * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.random.uniform(0, 0.03, n_days)),
        'low': prices * (1 - np.random.uniform(0, 0.03, n_days)),
        'volume': np.random.lognormal(10, 0.5, n_days)
    }, index=dates)
    
    data['open'] = data['close'].shift(1)
    data = data.dropna()
    
    # 创建情绪因子计算器
    calculator = SentimentFactorCalculator()
    
    # 计算情绪因子
    print("计算情绪因子...")
    factors = calculator.calculate_all_factors(data)
    
    # 计算因子评分
    scores = calculator.calculate_factor_scores(factors)
    
    # 生成报告
    report = calculator.generate_factor_report(factors, scores)
    print(report)
    
    # 保存结果
    print(f"\n成功计算 {len(factors)} 个情绪因子")
    print("情绪因子计算完成！")


if __name__ == "__main__":
    main()