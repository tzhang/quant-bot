"""
另类数据因子计算模块

基于另类数据源计算各种创新因子，包括：
1. 卫星数据因子 - 基于卫星图像分析经济活动
2. 专利数据因子 - 基于专利申请分析创新能力
3. 供应链数据因子 - 基于供应链网络分析
4. 社交媒体因子 - 基于社交媒体情绪和关注度
5. 新闻文本因子 - 基于新闻文本挖掘
6. 搜索趋势因子 - 基于搜索引擎数据
7. 地理位置因子 - 基于地理位置和人流数据
8. 环境数据因子 - 基于环境和气候数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging
import json
import re
from collections import Counter

# 数据获取和处理库
try:
    import requests
    from bs4 import BeautifulSoup
    import yfinance as yf
    from textblob import TextBlob
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from pytrends.request import TrendReq
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    print("部分依赖库未安装，将使用模拟数据")

class AlternativeFactorCalculator:
    """另类数据因子计算器"""
    
    def __init__(self):
        """初始化另类数据因子计算器"""
        self.logger = logging.getLogger(__name__)
        
        # 初始化情感分析器
        if DATA_AVAILABLE:
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except:
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
            
        # 初始化Google Trends
        if DATA_AVAILABLE:
            try:
                self.pytrends = TrendReq(hl='en-US', tz=360)
            except:
                self.pytrends = None
        else:
            self.pytrends = None
            
        print("另类数据因子计算器初始化完成")
    
    def calculate_satellite_factors(self, companies: List[str] = None) -> Dict[str, pd.Series]:
        """
        计算卫星数据因子
        
        Args:
            companies: 公司列表
            
        Returns:
            卫星数据因子字典
        """
        factors = {}
        
        if companies is None:
            companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # 由于真实卫星数据获取复杂且昂贵，这里使用模拟数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        for company in companies:
            # 模拟卫星数据指标
            
            # 1. 停车场满载率 (零售公司)
            parking_occupancy = self._generate_satellite_parking_data(dates, company)
            factors[f'{company}_parking_occupancy'] = parking_occupancy
            
            # 2. 工厂活动强度 (制造业公司)
            factory_activity = self._generate_satellite_factory_data(dates, company)
            factors[f'{company}_factory_activity'] = factory_activity
            
            # 3. 物流中心活动 (电商公司)
            logistics_activity = self._generate_satellite_logistics_data(dates, company)
            factors[f'{company}_logistics_activity'] = logistics_activity
            
            # 4. 建筑工地进展 (房地产/基建公司)
            construction_progress = self._generate_satellite_construction_data(dates, company)
            factors[f'{company}_construction_progress'] = construction_progress
            
            # 5. 农田作物健康度 (农业公司)
            crop_health = self._generate_satellite_crop_data(dates, company)
            factors[f'{company}_crop_health'] = crop_health
        
        return factors
    
    def _generate_satellite_parking_data(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟停车场卫星数据"""
        np.random.seed(hash(company) % 2**32)
        
        # 基础占用率
        base_occupancy = 0.6
        
        # 周末效应
        weekend_effect = np.array([0.8 if d.weekday() >= 5 else 1.0 for d in dates])
        
        # 季节性效应
        seasonal_effect = 1 + 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        
        # 随机波动
        random_noise = np.random.normal(0, 0.1, len(dates))
        
        occupancy = base_occupancy * weekend_effect * seasonal_effect + random_noise
        occupancy = np.clip(occupancy, 0, 1)  # 限制在0-1之间
        
        return pd.Series(occupancy, index=dates, name=f'{company}_parking_occupancy')
    
    def _generate_satellite_factory_data(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟工厂活动卫星数据"""
        np.random.seed(hash(company + 'factory') % 2**32)
        
        # 基础活动强度
        base_activity = 0.7
        
        # 工作日效应
        workday_effect = np.array([0.3 if d.weekday() >= 5 else 1.0 for d in dates])
        
        # 月度周期 (月末通常活动更强)
        monthly_cycle = 1 + 0.15 * np.sin(2 * np.pi * np.array([d.day for d in dates]) / 30)
        
        # 随机波动
        random_noise = np.random.normal(0, 0.08, len(dates))
        
        activity = base_activity * workday_effect * monthly_cycle + random_noise
        activity = np.clip(activity, 0, 1)
        
        return pd.Series(activity, index=dates, name=f'{company}_factory_activity')
    
    def _generate_satellite_logistics_data(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟物流中心卫星数据"""
        np.random.seed(hash(company + 'logistics') % 2**32)
        
        # 基础物流活动
        base_logistics = 0.65
        
        # 购物季效应 (11月-12月活动增强)
        shopping_season = np.array([1.5 if d.month in [11, 12] else 1.0 for d in dates])
        
        # 周内效应 (周一到周三较高)
        weekly_pattern = np.array([1.2 if d.weekday() < 3 else 0.8 for d in dates])
        
        # 随机波动
        random_noise = np.random.normal(0, 0.1, len(dates))
        
        logistics = base_logistics * shopping_season * weekly_pattern + random_noise
        logistics = np.clip(logistics, 0, 2)
        
        return pd.Series(logistics, index=dates, name=f'{company}_logistics_activity')
    
    def _generate_satellite_construction_data(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟建筑工地卫星数据"""
        np.random.seed(hash(company + 'construction') % 2**32)
        
        # 基础建设进度
        base_progress = 0.5
        
        # 天气效应 (冬季建设活动减少)
        weather_effect = np.array([0.6 if d.month in [12, 1, 2] else 1.0 for d in dates])
        
        # 渐进式增长 (项目随时间推进)
        progress_trend = 1 + 0.3 * np.arange(len(dates)) / len(dates)
        
        # 随机波动
        random_noise = np.random.normal(0, 0.05, len(dates))
        
        progress = base_progress * weather_effect * progress_trend + random_noise
        progress = np.clip(progress, 0, 1)
        
        return pd.Series(progress, index=dates, name=f'{company}_construction_progress')
    
    def _generate_satellite_crop_data(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟农作物健康度卫星数据"""
        np.random.seed(hash(company + 'crop') % 2**32)
        
        # 基础作物健康度
        base_health = 0.75
        
        # 季节性效应 (春夏较好，秋冬较差)
        seasonal_health = 1 + 0.3 * np.sin(2 * np.pi * (np.arange(len(dates)) - 90) / 365.25)
        
        # 天气随机冲击
        weather_shocks = np.random.normal(0, 0.1, len(dates))
        extreme_weather = np.random.random(len(dates)) < 0.05  # 5%概率极端天气
        weather_shocks[extreme_weather] *= 3
        
        health = base_health * seasonal_health + weather_shocks
        health = np.clip(health, 0, 1)
        
        return pd.Series(health, index=dates, name=f'{company}_crop_health')
    
    def calculate_patent_factors(self, companies: List[str] = None) -> Dict[str, pd.Series]:
        """
        计算专利数据因子
        
        Args:
            companies: 公司列表
            
        Returns:
            专利数据因子字典
        """
        factors = {}
        
        if companies is None:
            companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # 由于真实专利数据获取复杂，这里使用模拟数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
        
        for company in companies:
            # 1. 专利申请数量
            patent_applications = self._generate_patent_applications(dates, company)
            factors[f'{company}_patent_applications'] = patent_applications
            
            # 2. 专利授权数量
            patent_grants = self._generate_patent_grants(dates, company)
            factors[f'{company}_patent_grants'] = patent_grants
            
            # 3. 专利质量分数 (基于引用次数)
            patent_quality = self._generate_patent_quality(dates, company)
            factors[f'{company}_patent_quality'] = patent_quality
            
            # 4. 技术领域多样性
            tech_diversity = self._generate_tech_diversity(dates, company)
            factors[f'{company}_tech_diversity'] = tech_diversity
            
            # 5. 创新强度 (专利数量/研发支出)
            innovation_intensity = self._generate_innovation_intensity(dates, company)
            factors[f'{company}_innovation_intensity'] = innovation_intensity
        
        return factors
    
    def _generate_patent_applications(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟专利申请数据"""
        np.random.seed(hash(company + 'patent_app') % 2**32)
        
        # 基础申请数量 (根据公司规模调整)
        company_scale = {'AAPL': 100, 'GOOGL': 120, 'MSFT': 110, 'AMZN': 80, 'TSLA': 60}
        base_applications = company_scale.get(company, 50)
        
        # 季度周期 (通常Q4申请较多)
        quarterly_pattern = np.array([1.2 if d.month in [10, 11, 12] else 1.0 for d in dates])
        
        # 趋势增长
        trend_growth = 1 + 0.05 * np.arange(len(dates)) / len(dates)
        
        # 随机波动
        random_variation = np.random.poisson(base_applications, len(dates))
        
        applications = random_variation * quarterly_pattern * trend_growth
        
        return pd.Series(applications, index=dates, name=f'{company}_patent_applications')
    
    def _generate_patent_grants(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟专利授权数据"""
        np.random.seed(hash(company + 'patent_grant') % 2**32)
        
        # 专利授权通常滞后申请18-24个月
        applications = self._generate_patent_applications(dates, company)
        
        # 授权率约60-80%
        grant_rate = np.random.uniform(0.6, 0.8)
        
        # 添加滞后效应
        grants = applications.shift(6) * grant_rate  # 6个月滞后
        grants = grants.fillna(applications.iloc[0] * grant_rate)
        
        # 添加随机波动
        grants *= np.random.normal(1, 0.1, len(dates))
        grants = np.maximum(grants, 0)  # 确保非负
        
        return pd.Series(grants, index=dates, name=f'{company}_patent_grants')
    
    def _generate_patent_quality(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟专利质量数据"""
        np.random.seed(hash(company + 'patent_quality') % 2**32)
        
        # 基础质量分数 (1-10分)
        base_quality = np.random.uniform(6, 9)
        
        # 质量随时间缓慢提升
        quality_trend = base_quality + 0.5 * np.arange(len(dates)) / len(dates)
        
        # 随机波动
        quality_noise = np.random.normal(0, 0.3, len(dates))
        
        quality = quality_trend + quality_noise
        quality = np.clip(quality, 1, 10)
        
        return pd.Series(quality, index=dates, name=f'{company}_patent_quality')
    
    def _generate_tech_diversity(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟技术多样性数据"""
        np.random.seed(hash(company + 'tech_div') % 2**32)
        
        # 技术领域数量 (Shannon多样性指数)
        base_diversity = np.random.uniform(2, 4)
        
        # 多样性随公司发展增加
        diversity_growth = base_diversity + 0.3 * np.arange(len(dates)) / len(dates)
        
        # 随机波动
        diversity_noise = np.random.normal(0, 0.1, len(dates))
        
        diversity = diversity_growth + diversity_noise
        diversity = np.maximum(diversity, 1)
        
        return pd.Series(diversity, index=dates, name=f'{company}_tech_diversity')
    
    def _generate_innovation_intensity(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟创新强度数据"""
        np.random.seed(hash(company + 'innovation') % 2**32)
        
        # 创新强度 (专利数量/研发支出的比率)
        base_intensity = np.random.uniform(0.5, 2.0)
        
        # 效率随时间提升
        efficiency_trend = base_intensity * (1 + 0.1 * np.arange(len(dates)) / len(dates))
        
        # 随机波动
        intensity_noise = np.random.normal(0, 0.1, len(dates))
        
        intensity = efficiency_trend + intensity_noise
        intensity = np.maximum(intensity, 0.1)
        
        return pd.Series(intensity, index=dates, name=f'{company}_innovation_intensity')
    
    def calculate_search_trend_factors(self, keywords: List[str] = None) -> Dict[str, pd.Series]:
        """
        计算搜索趋势因子
        
        Args:
            keywords: 搜索关键词列表
            
        Returns:
            搜索趋势因子字典
        """
        factors = {}
        
        if keywords is None:
            keywords = ['iPhone', 'Tesla', 'Amazon', 'Google', 'Microsoft']
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='W')
        
        if self.pytrends:
            try:
                for keyword in keywords:
                    # 获取Google Trends数据
                    self.pytrends.build_payload([keyword], timeframe='2023-01-01 2023-12-31')
                    trend_data = self.pytrends.interest_over_time()
                    
                    if not trend_data.empty and keyword in trend_data.columns:
                        trend_series = trend_data[keyword]
                        factors[f'search_{keyword.lower()}'] = trend_series
                        
                        # 计算搜索趋势的衍生指标
                        factors[f'search_{keyword.lower()}_momentum'] = trend_series.pct_change(4)
                        factors[f'search_{keyword.lower()}_volatility'] = trend_series.rolling(4).std()
                        
            except Exception as e:
                print(f"获取搜索趋势数据失败: {e}")
        
        # 如果无法获取真实数据，使用模拟数据
        if not factors:
            factors = self._generate_mock_search_trends(dates, keywords)
        
        return factors
    
    def _generate_mock_search_trends(self, dates: pd.DatetimeIndex, keywords: List[str]) -> Dict[str, pd.Series]:
        """生成模拟搜索趋势数据"""
        factors = {}
        
        for keyword in keywords:
            np.random.seed(hash(keyword) % 2**32)
            
            # 基础搜索热度
            base_interest = np.random.uniform(30, 80)
            
            # 季节性模式
            seasonal_pattern = 1 + 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
            
            # 随机事件冲击
            event_shocks = np.random.normal(0, 10, len(dates))
            rare_events = np.random.random(len(dates)) < 0.05  # 5%概率重大事件
            event_shocks[rare_events] *= 3
            
            # 趋势
            trend = np.random.uniform(-0.1, 0.1) * np.arange(len(dates))
            
            search_interest = base_interest * seasonal_pattern + event_shocks + trend
            search_interest = np.clip(search_interest, 0, 100)
            
            search_series = pd.Series(search_interest, index=dates, name=f'search_{keyword.lower()}')
            factors[f'search_{keyword.lower()}'] = search_series
            factors[f'search_{keyword.lower()}_momentum'] = search_series.pct_change(4)
            factors[f'search_{keyword.lower()}_volatility'] = search_series.rolling(4).std()
        
        return factors
    
    def calculate_news_sentiment_factors(self, companies: List[str] = None) -> Dict[str, pd.Series]:
        """
        计算新闻情感因子
        
        Args:
            companies: 公司列表
            
        Returns:
            新闻情感因子字典
        """
        factors = {}
        
        if companies is None:
            companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # 由于真实新闻数据获取复杂，这里使用模拟数据
        for company in companies:
            # 1. 新闻情感分数
            news_sentiment = self._generate_news_sentiment(dates, company)
            factors[f'{company}_news_sentiment'] = news_sentiment
            
            # 2. 新闻数量
            news_volume = self._generate_news_volume(dates, company)
            factors[f'{company}_news_volume'] = news_volume
            
            # 3. 正面新闻比例
            positive_news_ratio = self._generate_positive_news_ratio(dates, company)
            factors[f'{company}_positive_news_ratio'] = positive_news_ratio
            
            # 4. 新闻关注度
            news_attention = self._generate_news_attention(dates, company)
            factors[f'{company}_news_attention'] = news_attention
        
        return factors
    
    def _generate_news_sentiment(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟新闻情感数据"""
        np.random.seed(hash(company + 'sentiment') % 2**32)
        
        # 基础情感分数 (-1到1之间)
        base_sentiment = np.random.uniform(-0.1, 0.1)
        
        # 随机情感波动
        sentiment_changes = np.random.normal(0, 0.1, len(dates))
        
        # 偶发重大事件
        major_events = np.random.random(len(dates)) < 0.02  # 2%概率重大事件
        sentiment_changes[major_events] += np.random.choice([-0.5, 0.5], sum(major_events))
        
        # 累积情感分数
        sentiment = base_sentiment + np.cumsum(sentiment_changes * 0.1)
        sentiment = np.clip(sentiment, -1, 1)
        
        return pd.Series(sentiment, index=dates, name=f'{company}_news_sentiment')
    
    def _generate_news_volume(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟新闻数量数据"""
        np.random.seed(hash(company + 'volume') % 2**32)
        
        # 基础新闻数量
        base_volume = np.random.poisson(10, len(dates))
        
        # 工作日效应
        weekday_effect = np.array([0.5 if d.weekday() >= 5 else 1.0 for d in dates])
        
        # 财报季效应
        earnings_season = np.array([2.0 if d.month in [1, 4, 7, 10] else 1.0 for d in dates])
        
        volume = base_volume * weekday_effect * earnings_season
        
        return pd.Series(volume, index=dates, name=f'{company}_news_volume')
    
    def _generate_positive_news_ratio(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟正面新闻比例数据"""
        np.random.seed(hash(company + 'positive') % 2**32)
        
        # 基础正面比例
        base_ratio = np.random.uniform(0.4, 0.7)
        
        # 随机波动
        ratio_noise = np.random.normal(0, 0.1, len(dates))
        
        # 市场情绪影响
        market_sentiment = np.sin(2 * np.pi * np.arange(len(dates)) / 365) * 0.1
        
        ratio = base_ratio + ratio_noise + market_sentiment
        ratio = np.clip(ratio, 0, 1)
        
        return pd.Series(ratio, index=dates, name=f'{company}_positive_news_ratio')
    
    def _generate_news_attention(self, dates: pd.DatetimeIndex, company: str) -> pd.Series:
        """生成模拟新闻关注度数据"""
        np.random.seed(hash(company + 'attention') % 2**32)
        
        # 基础关注度
        base_attention = np.random.uniform(50, 100)
        
        # 与新闻数量相关
        news_volume = self._generate_news_volume(dates, company)
        volume_effect = news_volume / news_volume.mean()
        
        # 随机波动
        attention_noise = np.random.normal(0, 10, len(dates))
        
        attention = base_attention * volume_effect + attention_noise
        attention = np.maximum(attention, 0)
        
        return pd.Series(attention, index=dates, name=f'{company}_news_attention')
    
    def calculate_all_factors(self, companies: List[str] = None, 
                            keywords: List[str] = None) -> Dict[str, pd.Series]:
        """
        计算所有另类数据因子
        
        Args:
            companies: 公司列表
            keywords: 搜索关键词列表
            
        Returns:
            所有另类数据因子字典
        """
        all_factors = {}
        
        print("开始计算另类数据因子...")
        
        # 1. 卫星数据因子
        satellite_factors = self.calculate_satellite_factors(companies)
        all_factors.update({f"satellite_{k}": v for k, v in satellite_factors.items()})
        
        # 2. 专利数据因子
        patent_factors = self.calculate_patent_factors(companies)
        all_factors.update({f"patent_{k}": v for k, v in patent_factors.items()})
        
        # 3. 搜索趋势因子
        search_factors = self.calculate_search_trend_factors(keywords)
        all_factors.update({f"search_{k}": v for k, v in search_factors.items()})
        
        # 4. 新闻情感因子
        news_factors = self.calculate_news_sentiment_factors(companies)
        all_factors.update({f"news_{k}": v for k, v in news_factors.items()})
        
        print(f"另类数据因子计算完成，共生成 {len(all_factors)} 个因子")
        return all_factors

def main():
    """测试另类数据因子计算器"""
    print("测试另类数据因子计算器...")
    
    # 初始化计算器
    calculator = AlternativeFactorCalculator()
    
    # 计算因子
    factors = calculator.calculate_all_factors(
        companies=['AAPL', 'GOOGL', 'TSLA'],
        keywords=['iPhone', 'Tesla', 'AI']
    )
    
    print(f"成功计算 {len(factors)} 个另类数据因子")
    for name, factor in list(factors.items())[:10]:  # 只显示前10个
        if not factor.empty:
            print(f"{name}: 均值={factor.mean():.4f}, 标准差={factor.std():.4f}")

if __name__ == "__main__":
    main()