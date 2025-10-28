
# ==========================================
# 迁移说明 - 2025-10-10 23:06:36
# ==========================================
# 本文件已从yfinance迁移到IB TWS API
# 原始文件备份在: backup_before_ib_migration/src/data/sentiment_data.py
# 
# 主要变更:
# # - 替换yfinance导入为IB导入
# 
# 注意事项:
# 1. 需要启动IB TWS或Gateway
# 2. 确保API设置已正确配置
# 3. 某些yfinance特有功能可能需要手动调整
# ==========================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分析数据模块

提供市场情感数据的获取、处理和分析功能：
1. 新闻情感分析（财经新闻、公司新闻）
2. 社交媒体情绪分析（Twitter、Reddit等）
3. 分析师评级和目标价
4. 机构持仓变化
5. 恐慌指数（VIX）和市场情绪指标
6. 期权情绪指标（Put/Call比率）

数据源：
- 主要：Alpha Vantage、NewsAPI、Yahoo Finance
- 辅助：Reddit API、Twitter API（需要认证）
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import json
from pathlib import Path
import re

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.data.ib_data_provider import IBDataProvider, IBConfig
from textblob import TextBlob

logger = logging.getLogger(__name__)


class SentimentDataManager:
    """
    情感分析数据管理器
    
    统一管理各种市场情感数据源，提供情感分析和市场情绪指标
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, news_api_key: Optional[str] = None):
        """
        初始化情感数据管理器
        
        Args:
            alpha_vantage_key: Alpha Vantage API密钥
            news_api_key: NewsAPI密钥
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        
        # API基础URL
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.news_api_url = "https://newsapi.org/v2"
        
        # 设置缓存目录
        self.cache_dir = Path("data_cache/sentiment")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置HTTP会话
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # API调用限制
        self.api_call_interval = 1.0  # 秒
        self.last_api_call = 0
        
        # 市场情绪指标符号
        self.sentiment_indicators = {
            'VIX': '恐慌指数',
            'VXN': '纳斯达克波动率指数',
            'RVX': '罗素2000波动率指数',
            'SKEW': '偏度指数',
            'VVIX': 'VIX的波动率',
        }
        
        # 财经关键词（用于新闻过滤）
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'merger', 'acquisition', 'ipo', 'dividend', 'buyback',
            'fed', 'interest rate', 'inflation', 'gdp', 'unemployment',
            'bull market', 'bear market', 'recession', 'recovery'
        ]
    
    def _wait_for_api_limit(self):
        """等待API调用限制"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.api_call_interval:
            wait_time = self.api_call_interval - time_since_last_call
            time.sleep(wait_time)
        self.last_api_call = time.time()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str, max_age_hours: int = 4) -> Optional[Dict]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            # 检查缓存文件年龄
            file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if file_age.total_seconds() > max_age_hours * 3600:
                logger.debug(f"缓存文件过期: {cache_path}")
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"从缓存加载数据: {cache_key}")
                return data
                
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
                logger.debug(f"数据已缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        分析文本情感
        
        Args:
            text: 待分析的文本
            
        Returns:
            情感分析结果字典
        """
        try:
            # 使用TextBlob进行情感分析
            blob = TextBlob(text)
            
            # 极性：-1（负面）到 1（正面）
            polarity = blob.sentiment.polarity
            
            # 主观性：0（客观）到 1（主观）
            subjectivity = blob.sentiment.subjectivity
            
            # 分类情感
            if polarity > 0.1:
                sentiment_label = 'positive'
            elif polarity < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment_label,
                'confidence': abs(polarity)
            }
            
        except Exception as e:
            logger.error(f"文本情感分析失败: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0
            }
    
    def get_news_sentiment(self, symbol: str, days_back: int = 7, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取股票新闻情感分析
        
        Args:
            symbol: 股票代码
            days_back: 回溯天数
            use_cache: 是否使用缓存
            
        Returns:
            新闻情感分析结果
        """
        cache_key = f"news_sentiment_{symbol}_{days_back}"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_key, max_age_hours=2)
            if cached_data:
                return cached_data
        
        sentiment_data = {
            'symbol': symbol,
            'analysis_date': datetime.now().isoformat(),
            'period_days': days_back,
            'news_articles': [],
            'sentiment_summary': {
                'total_articles': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'avg_polarity': 0.0,
                'avg_subjectivity': 0.0,
                'sentiment_score': 0.0
            }
        }
        
        try:
            # 首先尝试从Alpha Vantage获取新闻
            news_articles = self._get_alpha_vantage_news(symbol)
            
            # 如果Alpha Vantage没有数据，尝试NewsAPI
            if not news_articles and self.news_api_key:
                news_articles = self._get_newsapi_articles(symbol, days_back)
            
            # 如果仍然没有数据，使用Yahoo Finance新闻
            if not news_articles:
                news_articles = self._get_yahoo_news(symbol)
            
            if not news_articles:
                logger.warning(f"未找到{symbol}的新闻数据")
                return sentiment_data
            
            # 分析每篇文章的情感
            polarities = []
            subjectivities = []
            
            for article in news_articles:
                # 分析标题和摘要的情感
                title = article.get('title', '')
                summary = article.get('summary', '') or article.get('description', '')
                
                # 合并标题和摘要进行分析
                text_to_analyze = f"{title}. {summary}"
                
                sentiment_result = self.analyze_text_sentiment(text_to_analyze)
                
                article_data = {
                    'title': title,
                    'summary': summary[:200] + '...' if len(summary) > 200 else summary,
                    'published_date': article.get('published_date') or article.get('publishedAt'),
                    'source': article.get('source', {}).get('name') if isinstance(article.get('source'), dict) else article.get('source'),
                    'url': article.get('url'),
                    'sentiment': sentiment_result
                }
                
                sentiment_data['news_articles'].append(article_data)
                polarities.append(sentiment_result['polarity'])
                subjectivities.append(sentiment_result['subjectivity'])
                
                # 统计情感分类
                if sentiment_result['sentiment'] == 'positive':
                    sentiment_data['sentiment_summary']['positive_count'] += 1
                elif sentiment_result['sentiment'] == 'negative':
                    sentiment_data['sentiment_summary']['negative_count'] += 1
                else:
                    sentiment_data['sentiment_summary']['neutral_count'] += 1
            
            # 计算总体情感指标
            sentiment_data['sentiment_summary']['total_articles'] = len(news_articles)
            
            if polarities:
                sentiment_data['sentiment_summary']['avg_polarity'] = np.mean(polarities)
                sentiment_data['sentiment_summary']['avg_subjectivity'] = np.mean(subjectivities)
                
                # 计算综合情感得分（考虑文章数量和平均极性）
                article_weight = min(len(news_articles) / 10, 1.0)  # 文章数量权重，最多为1
                sentiment_score = sentiment_data['sentiment_summary']['avg_polarity'] * article_weight
                sentiment_data['sentiment_summary']['sentiment_score'] = sentiment_score
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(cache_key, sentiment_data)
            
            logger.info(f"✅ 完成{symbol}新闻情感分析，共{len(news_articles)}篇文章")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"获取{symbol}新闻情感失败: {e}")
            sentiment_data['error'] = str(e)
            return sentiment_data
    
    def _get_alpha_vantage_news(self, symbol: str) -> List[Dict]:
        """从Alpha Vantage获取新闻"""
        if not self.alpha_vantage_key:
            return []
        
        try:
            self._wait_for_api_limit()
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 50
            }
            
            response = self.session.get(self.alpha_vantage_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' not in data:
                logger.warning("Alpha Vantage新闻数据格式异常")
                return []
            
            articles = []
            for item in data['feed'][:20]:  # 限制前20篇
                articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'published_date': item.get('time_published', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', '')
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Alpha Vantage新闻获取失败: {e}")
            return []
    
    def _get_newsapi_articles(self, symbol: str, days_back: int) -> List[Dict]:
        """从NewsAPI获取新闻"""
        if not self.news_api_key:
            return []
        
        try:
            self._wait_for_api_limit()
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'q': f'"{symbol}" OR "{symbol.replace(".", " ")}"',
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'pageSize': 20,
                'apiKey': self.news_api_key
            }
            
            response = self.session.get(f"{self.news_api_url}/everything", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.warning(f"NewsAPI错误: {data.get('message', '未知错误')}")
                return []
            
            articles = []
            for item in data.get('articles', []):
                # 过滤财经相关新闻
                title = item.get('title', '').lower()
                description = item.get('description', '').lower()
                
                if any(keyword in title or keyword in description for keyword in self.financial_keywords):
                    articles.append({
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'publishedAt': item.get('publishedAt', ''),
                        'source': item.get('source', {}),
                        'url': item.get('url', '')
                    })
            
            return articles
            
        except Exception as e:
            logger.error(f"NewsAPI新闻获取失败: {e}")
            return []
    
    def _get_yahoo_news(self, symbol: str) -> List[Dict]:
        """
        从Yahoo Finance获取新闻
        
        数据源优先级：IB TWS API > yfinance
        """
        try:
            # 优先尝试使用 IB TWS API 获取新闻数据
            try:
                ib_provider = IBDataProvider(IBConfig())
                # IB TWS API 可能不直接提供新闻数据，这里作为占位符
                # 实际实现可能需要使用其他新闻API
                logger.info(f"IB TWS API 暂不支持新闻数据获取，回退到 yfinance")
                raise NotImplementedError("IB TWS API 新闻功能待实现")
                
            except Exception as e:
                logger.warning(f"IB TWS API 获取新闻失败: {e}")
            
            # 回退到 yfinance
            # import yfinance as yf  # 已移除，不再使用yfinance
            logger.warning("yfinance已移除，无法获取Yahoo Finance新闻数据")
            return []  # 返回空列表
            # ticker = yf.Ticker(symbol)
            # news = ticker.news
            
            if not news:
                return []
            
            articles = []
            for item in news[:10]:  # 限制前10篇
                articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'published_date': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat() if item.get('providerPublishTime') else '',
                    'source': item.get('publisher', ''),
                    'url': item.get('link', '')
                })
            
            logger.info(f"从 yfinance 获取 {symbol} 新闻数据成功")
            return articles
            
        except Exception as e:
            logger.error(f"Yahoo Finance新闻获取失败: {e}")
            return []
    
    def get_market_sentiment_indicators(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取市场情绪指标
        
        数据源优先级：IB TWS API > yfinance
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            市场情绪指标字典
        """
        cache_key = "market_sentiment_indicators"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_key, max_age_hours=1)
            if cached_data:
                return cached_data
        
        sentiment_indicators = {
            'timestamp': datetime.now().isoformat(),
            'indicators': {},
            'summary': {}
        }
        
        try:
            # 获取各种情绪指标
            for symbol, description in self.sentiment_indicators.items():
                try:
                    # 优先尝试使用 IB TWS API
                    try:
                        ib_provider = IBDataProvider(IBConfig())
                        
                        # 转换符号格式（去掉^前缀）
                        ib_symbol = symbol
                        hist = ib_provider.get_historical_data(ib_symbol, days=5, bar_size='1 day')
                        
                        if not hist.empty:
                            current_value = hist['close'].iloc[-1]
                            prev_value = hist['close'].iloc[-2] if len(hist) > 1 else current_value
                            change = current_value - prev_value
                            change_pct = (change / prev_value) * 100 if prev_value != 0 else 0
                            
                            # 计算5日平均值
                            avg_5d = hist['close'].mean()
                            
                            sentiment_indicators['indicators'][symbol] = {
                                'name': description,
                                'current_value': float(current_value),
                                'previous_value': float(prev_value),
                                'change': float(change),
                                'change_percent': float(change_pct),
                                'avg_5d': float(avg_5d),
                                'vs_avg_5d': float((current_value / avg_5d - 1) * 100) if avg_5d != 0 else 0
                            }
                            
                            logger.info(f"✅ 从 IB TWS API 获取{description}({symbol})数据成功")
                            continue
                            
                    except Exception as e:
                        logger.warning(f"IB TWS API 获取情绪指标 {symbol} 失败: {e}")
                    
                    # 回退到 yfinance
            # import yfinance as yf  # 已移除，不再使用yfinance
            logger.warning("yfinance已移除，无法获取市场情绪指标数据")
            continue  # 跳过当前指标
            # ticker = yf.Ticker(f"^{symbol}")
            # hist = ticker.history(period='5d')
                    
                    if not hist.empty:
                        current_value = hist['Close'].iloc[-1]
                        prev_value = hist['Close'].iloc[-2] if len(hist) > 1 else current_value
                        change = current_value - prev_value
                        change_pct = (change / prev_value) * 100 if prev_value != 0 else 0
                        
                        # 计算5日平均值
                        avg_5d = hist['Close'].mean()
                        
                        sentiment_indicators['indicators'][symbol] = {
                            'name': description,
                            'current_value': float(current_value),
                            'previous_value': float(prev_value),
                            'change': float(change),
                            'change_percent': float(change_pct),
                            'avg_5d': float(avg_5d),
                            'vs_avg_5d': float((current_value / avg_5d - 1) * 100) if avg_5d != 0 else 0
                        }
                        
                        logger.info(f"✅ 从 yfinance 获取{description}({symbol})数据成功")
                    
                except Exception as e:
                    logger.error(f"❌ 获取{symbol}数据失败: {e}")
                    continue
            
            # 生成市场情绪摘要
            if 'VIX' in sentiment_indicators['indicators']:
                vix_data = sentiment_indicators['indicators']['VIX']
                vix_value = vix_data['current_value']
                
                # VIX情绪解读
                if vix_value < 15:
                    vix_sentiment = 'extremely_low_fear'
                    vix_description = '极低恐慌，市场过度乐观'
                elif vix_value < 20:
                    vix_sentiment = 'low_fear'
                    vix_description = '低恐慌，市场相对平静'
                elif vix_value < 30:
                    vix_sentiment = 'moderate_fear'
                    vix_description = '中等恐慌，市场有所担忧'
                elif vix_value < 40:
                    vix_sentiment = 'high_fear'
                    vix_description = '高恐慌，市场明显担忧'
                else:
                    vix_sentiment = 'extreme_fear'
                    vix_description = '极度恐慌，市场恐慌情绪严重'
                
                sentiment_indicators['summary']['vix_analysis'] = {
                    'sentiment': vix_sentiment,
                    'description': vix_description,
                    'value': vix_value,
                    'change_percent': vix_data['change_percent']
                }
            
            # 整体市场情绪评估
            fear_indicators = 0
            greed_indicators = 0
            
            for symbol, data in sentiment_indicators['indicators'].items():
                if symbol == 'VIX':
                    if data['current_value'] > 25:
                        fear_indicators += 1
                    elif data['current_value'] < 15:
                        greed_indicators += 1
                
                # 其他指标的变化趋势
                if data['change_percent'] > 5:
                    if symbol in ['VIX', 'VXN', 'RVX']:
                        fear_indicators += 1
                    else:
                        greed_indicators += 1
                elif data['change_percent'] < -5:
                    if symbol in ['VIX', 'VXN', 'RVX']:
                        greed_indicators += 1
                    else:
                        fear_indicators += 1
            
            # 综合情绪评分
            total_signals = fear_indicators + greed_indicators
            if total_signals > 0:
                fear_ratio = fear_indicators / total_signals
                if fear_ratio > 0.6:
                    overall_sentiment = 'fear'
                elif fear_ratio < 0.4:
                    overall_sentiment = 'greed'
                else:
                    overall_sentiment = 'neutral'
            else:
                overall_sentiment = 'neutral'
            
            sentiment_indicators['summary']['overall_sentiment'] = overall_sentiment
            sentiment_indicators['summary']['fear_indicators'] = fear_indicators
            sentiment_indicators['summary']['greed_indicators'] = greed_indicators
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(cache_key, sentiment_indicators)
            
            logger.info("市场情绪指标获取完成")
            return sentiment_indicators
            
        except Exception as e:
            logger.error(f"获取市场情绪指标失败: {e}")
            sentiment_indicators['error'] = str(e)
            return sentiment_indicators
    
    def get_analyst_ratings(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取分析师评级数据
        
        数据源优先级：IB TWS API > yfinance
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            Dict[str, Any]: 分析师评级数据
        """
        cache_key = f"analyst_ratings_{symbol}"
        
        # 检查缓存
        if use_cache:
            cached_data = self._load_from_cache(cache_key, max_age_hours=24)
            if cached_data:
                return cached_data
        
        ratings_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'recommendations': {},
            'target_price': {},
            'earnings_estimates': {},
            'summary': {}
        }
        
        try:
            # 优先尝试使用 IB TWS API
            try:
                ib_provider = IBDataProvider(IBConfig())
                # IB TWS API 可能不直接提供分析师评级数据
                logger.info(f"IB TWS API 暂不支持分析师评级数据，回退到 yfinance")
                raise NotImplementedError("IB TWS API 分析师评级功能待实现")
                
            except Exception as e:
                logger.warning(f"IB TWS API 获取分析师评级失败: {e}")
            
            # 回退到 yfinance
            # import yfinance as yf  # 已移除，不再使用yfinance
            logger.warning("yfinance已移除，无法获取分析师评级数据")
            return ratings_data  # 返回空的评级数据
            # ticker = yf.Ticker(symbol)
            
            # 获取分析师推荐
            try:
                recommendations = ticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    # 获取最新的推荐数据
                    latest_rec = recommendations.iloc[-1]
                    
                    ratings_data['recommendations'] = {
                        'strong_buy': int(latest_rec.get('strongBuy', 0)),
                        'buy': int(latest_rec.get('buy', 0)),
                        'hold': int(latest_rec.get('hold', 0)),
                        'sell': int(latest_rec.get('sell', 0)),
                        'strong_sell': int(latest_rec.get('strongSell', 0)),
                        'period': latest_rec.name.strftime('%Y-%m-%d') if hasattr(latest_rec.name, 'strftime') else str(latest_rec.name)
                    }
                    
                    # 计算推荐得分（1-5分，5分最好）
                    total_ratings = sum([
                        ratings_data['recommendations']['strong_buy'],
                        ratings_data['recommendations']['buy'],
                        ratings_data['recommendations']['hold'],
                        ratings_data['recommendations']['sell'],
                        ratings_data['recommendations']['strong_sell']
                    ])
                    
                    if total_ratings > 0:
                        weighted_score = (
                            ratings_data['recommendations']['strong_buy'] * 5 +
                            ratings_data['recommendations']['buy'] * 4 +
                            ratings_data['recommendations']['hold'] * 3 +
                            ratings_data['recommendations']['sell'] * 2 +
                            ratings_data['recommendations']['strong_sell'] * 1
                        ) / total_ratings
                        
                        ratings_data['summary']['recommendation_score'] = weighted_score
                        ratings_data['summary']['total_analysts'] = total_ratings
                        
                        # 推荐等级
                        if weighted_score >= 4.5:
                            ratings_data['summary']['recommendation'] = 'Strong Buy'
                        elif weighted_score >= 3.5:
                            ratings_data['summary']['recommendation'] = 'Buy'
                        elif weighted_score >= 2.5:
                            ratings_data['summary']['recommendation'] = 'Hold'
                        elif weighted_score >= 1.5:
                            ratings_data['summary']['recommendation'] = 'Sell'
                        else:
                            ratings_data['summary']['recommendation'] = 'Strong Sell'
                
            except Exception as e:
                logger.warning(f"获取{symbol}分析师推荐失败: {e}")
            
            # 获取目标价格
            try:
                info = ticker.info
                if info:
                    ratings_data['target_price'] = {
                        'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                        'target_mean': info.get('targetMeanPrice'),
                        'target_high': info.get('targetHighPrice'),
                        'target_low': info.get('targetLowPrice'),
                        'recommendation_key': info.get('recommendationKey'),
                        'number_of_analyst_opinions': info.get('numberOfAnalystOpinions')
                    }
                    
                    # 计算目标价格上涨空间
                    current_price = ratings_data['target_price']['current_price']
                    target_mean = ratings_data['target_price']['target_mean']
                    
                    if current_price and target_mean:
                        upside_potential = (target_mean / current_price - 1) * 100
                        ratings_data['summary']['upside_potential'] = upside_potential
                
            except Exception as e:
                logger.warning(f"获取{symbol}目标价格失败: {e}")
            
            # 获取盈利预估
            try:
                earnings_estimate = ticker.calendar
                if earnings_estimate is not None and not earnings_estimate.empty:
                    ratings_data['earnings_estimates'] = earnings_estimate.to_dict('records')
                
            except Exception as e:
                logger.warning(f"获取{symbol}盈利预估失败: {e}")
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(cache_key, ratings_data)
            
            logger.info(f"✅ 获取{symbol}分析师评级数据成功")
            return ratings_data
            
        except Exception as e:
            logger.error(f"获取{symbol}分析师评级失败: {e}")
            ratings_data['error'] = str(e)
            return ratings_data
    
    def generate_sentiment_report(self, symbols: List[str], days_back: int = 7) -> Dict[str, Any]:
        """
        生成综合情感分析报告
        
        Args:
            symbols: 股票代码列表
            days_back: 新闻回溯天数
            
        Returns:
            综合情感分析报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period': f"{days_back} days",
            'symbols_analyzed': symbols,
            'market_sentiment': {},
            'stock_sentiment': {},
            'analyst_ratings': {},
            'summary': {}
        }
        
        try:
            # 获取市场整体情绪
            market_sentiment = self.get_market_sentiment_indicators()
            report['market_sentiment'] = market_sentiment
            
            # 分析各股票情感
            stock_sentiments = {}
            analyst_ratings = {}
            
            for symbol in symbols:
                try:
                    # 新闻情感分析
                    news_sentiment = self.get_news_sentiment(symbol, days_back)
                    stock_sentiments[symbol] = news_sentiment
                    
                    # 分析师评级
                    ratings = self.get_analyst_ratings(symbol)
                    analyst_ratings[symbol] = ratings
                    
                    logger.info(f"✅ 完成{symbol}情感分析")
                    
                except Exception as e:
                    logger.error(f"❌ {symbol}情感分析失败: {e}")
                    continue
            
            report['stock_sentiment'] = stock_sentiments
            report['analyst_ratings'] = analyst_ratings
            
            # 生成摘要
            positive_stocks = []
            negative_stocks = []
            
            for symbol, sentiment_data in stock_sentiments.items():
                sentiment_score = sentiment_data.get('sentiment_summary', {}).get('sentiment_score', 0)
                if sentiment_score > 0.1:
                    positive_stocks.append({
                        'symbol': symbol,
                        'sentiment_score': sentiment_score,
                        'positive_articles': sentiment_data.get('sentiment_summary', {}).get('positive_count', 0)
                    })
                elif sentiment_score < -0.1:
                    negative_stocks.append({
                        'symbol': symbol,
                        'sentiment_score': sentiment_score,
                        'negative_articles': sentiment_data.get('sentiment_summary', {}).get('negative_count', 0)
                    })
            
            # 按情感得分排序
            positive_stocks.sort(key=lambda x: x['sentiment_score'], reverse=True)
            negative_stocks.sort(key=lambda x: x['sentiment_score'])
            
            report['summary'] = {
                'total_symbols': len(symbols),
                'positive_sentiment_count': len(positive_stocks),
                'negative_sentiment_count': len(negative_stocks),
                'neutral_sentiment_count': len(symbols) - len(positive_stocks) - len(negative_stocks),
                'most_positive': positive_stocks[:3] if positive_stocks else [],
                'most_negative': negative_stocks[:3] if negative_stocks else [],
                'market_fear_greed': market_sentiment.get('summary', {}).get('overall_sentiment', 'neutral'),
                'vix_level': market_sentiment.get('indicators', {}).get('VIX', {}).get('current_value', 0)
            }
            
            logger.info(f"情感分析报告生成完成，分析了{len(symbols)}只股票")
            return report
            
        except Exception as e:
            logger.error(f"生成情感分析报告失败: {e}")
            report['error'] = str(e)
            return report


# 创建全局实例
sentiment_data_manager = SentimentDataManager()