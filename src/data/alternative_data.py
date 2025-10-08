#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
另类数据源模块

提供传统股票数据之外的另类数据获取和分析功能：
1. 期权数据（期权链、隐含波动率、Put/Call比率）
2. 债券数据（国债收益率、信用利差、收益率曲线）
3. 商品数据（黄金、原油、农产品等）
4. 加密货币数据（主要数字货币价格和市值）
5. 外汇数据（主要货币对汇率）
6. 经济日历和事件数据
7. 高频数据指标（订单流、成交量分布等）

数据源：
- 主要：Yahoo Finance、Alpha Vantage、FRED
- 辅助：CoinGecko（加密货币）、Quandl
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import json
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

logger = logging.getLogger(__name__)


class AlternativeDataManager:
    """
    另类数据管理器
    
    统一管理各种另类数据源，为量化策略提供多元化数据支持
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, fred_key: Optional[str] = None):
        """
        初始化另类数据管理器
        
        Args:
            alpha_vantage_key: Alpha Vantage API密钥
            fred_key: FRED API密钥
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fred_key = fred_key or os.getenv('FRED_API_KEY')
        
        # API基础URL
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.fred_url = "https://api.stlouisfed.org/fred/series/observations"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        
        # 设置缓存目录
        self.cache_dir = Path("data_cache/alternative")
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
        
        # 预定义的另类数据符号
        self.bond_symbols = {
            '^TNX': '10年期美国国债收益率',
            '^FVX': '5年期美国国债收益率',
            '^TYX': '30年期美国国债收益率',
            '^IRX': '3个月期美国国债收益率',
            'TLT': '20年期国债ETF',
            'IEF': '7-10年期国债ETF',
            'SHY': '1-3年期国债ETF'
        }
        
        self.commodity_symbols = {
            'GC=F': '黄金期货',
            'SI=F': '白银期货',
            'CL=F': '原油期货',
            'NG=F': '天然气期货',
            'ZC=F': '玉米期货',
            'ZS=F': '大豆期货',
            'ZW=F': '小麦期货',
            'HG=F': '铜期货',
            'PL=F': '铂金期货'
        }
        
        self.forex_symbols = {
            'EURUSD=X': '欧元/美元',
            'GBPUSD=X': '英镑/美元',
            'USDJPY=X': '美元/日元',
            'USDCHF=X': '美元/瑞士法郎',
            'AUDUSD=X': '澳元/美元',
            'USDCAD=X': '美元/加元',
            'NZDUSD=X': '新西兰元/美元',
            'EURGBP=X': '欧元/英镑'
        }
        
        self.crypto_symbols = {
            'BTC-USD': '比特币',
            'ETH-USD': '以太坊',
            'BNB-USD': '币安币',
            'XRP-USD': '瑞波币',
            'ADA-USD': '卡尔达诺',
            'SOL-USD': '索拉纳',
            'DOGE-USD': '狗狗币',
            'DOT-USD': '波卡币'
        }
        
        # 期权相关ETF和指标
        self.options_indicators = {
            'VIX': '恐慌指数',
            'VXN': '纳斯达克波动率指数',
            'RVX': '罗素2000波动率指数',
            'VVIX': 'VIX的波动率'
        }
    
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
    
    def get_options_data(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取期权数据
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            期权数据字典
        """
        cache_key = f"options_data_{symbol}"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_key, max_age_hours=1)
            if cached_data:
                return cached_data
        
        options_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'options_chain': {},
            'implied_volatility': {},
            'put_call_ratio': {},
            'volume_analysis': {}
        }
        
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取期权到期日
            expiration_dates = ticker.options
            
            if not expiration_dates:
                logger.warning(f"{symbol}没有期权数据")
                return options_data
            
            # 获取最近的期权链数据（前3个到期日）
            for i, exp_date in enumerate(expiration_dates[:3]):
                try:
                    option_chain = ticker.option_chain(exp_date)
                    
                    calls = option_chain.calls
                    puts = option_chain.puts
                    
                    if calls.empty and puts.empty:
                        continue
                    
                    # 处理看涨期权数据
                    calls_data = []
                    if not calls.empty:
                        for _, row in calls.iterrows():
                            calls_data.append({
                                'strike': float(row['strike']),
                                'last_price': float(row['lastPrice']) if pd.notna(row['lastPrice']) else 0,
                                'bid': float(row['bid']) if pd.notna(row['bid']) else 0,
                                'ask': float(row['ask']) if pd.notna(row['ask']) else 0,
                                'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                                'open_interest': int(row['openInterest']) if pd.notna(row['openInterest']) else 0,
                                'implied_volatility': float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else 0
                            })
                    
                    # 处理看跌期权数据
                    puts_data = []
                    if not puts.empty:
                        for _, row in puts.iterrows():
                            puts_data.append({
                                'strike': float(row['strike']),
                                'last_price': float(row['lastPrice']) if pd.notna(row['lastPrice']) else 0,
                                'bid': float(row['bid']) if pd.notna(row['bid']) else 0,
                                'ask': float(row['ask']) if pd.notna(row['ask']) else 0,
                                'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                                'open_interest': int(row['openInterest']) if pd.notna(row['openInterest']) else 0,
                                'implied_volatility': float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else 0
                            })
                    
                    options_data['options_chain'][exp_date] = {
                        'calls': calls_data,
                        'puts': puts_data,
                        'expiration_date': exp_date
                    }
                    
                    # 计算Put/Call比率
                    total_call_volume = sum([c['volume'] for c in calls_data])
                    total_put_volume = sum([p['volume'] for p in puts_data])
                    
                    if total_call_volume > 0:
                        put_call_ratio = total_put_volume / total_call_volume
                    else:
                        put_call_ratio = 0
                    
                    options_data['put_call_ratio'][exp_date] = {
                        'ratio': put_call_ratio,
                        'call_volume': total_call_volume,
                        'put_volume': total_put_volume
                    }
                    
                    # 计算平均隐含波动率
                    call_ivs = [c['implied_volatility'] for c in calls_data if c['implied_volatility'] > 0]
                    put_ivs = [p['implied_volatility'] for p in puts_data if p['implied_volatility'] > 0]
                    
                    options_data['implied_volatility'][exp_date] = {
                        'avg_call_iv': np.mean(call_ivs) if call_ivs else 0,
                        'avg_put_iv': np.mean(put_ivs) if put_ivs else 0,
                        'avg_total_iv': np.mean(call_ivs + put_ivs) if call_ivs + put_ivs else 0
                    }
                    
                except Exception as e:
                    logger.error(f"处理{symbol}期权链{exp_date}失败: {e}")
                    continue
            
            # 计算整体期权指标
            if options_data['put_call_ratio']:
                avg_put_call_ratio = np.mean([data['ratio'] for data in options_data['put_call_ratio'].values()])
                total_call_volume = sum([data['call_volume'] for data in options_data['put_call_ratio'].values()])
                total_put_volume = sum([data['put_volume'] for data in options_data['put_call_ratio'].values()])
                
                options_data['volume_analysis'] = {
                    'avg_put_call_ratio': avg_put_call_ratio,
                    'total_call_volume': total_call_volume,
                    'total_put_volume': total_put_volume,
                    'sentiment': 'bearish' if avg_put_call_ratio > 1.0 else 'bullish'
                }
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(cache_key, options_data)
            
            logger.info(f"✅ 获取{symbol}期权数据成功")
            return options_data
            
        except Exception as e:
            logger.error(f"获取{symbol}期权数据失败: {e}")
            options_data['error'] = str(e)
            return options_data
    
    def get_bond_data(self, period: str = '1y', use_cache: bool = True) -> Dict[str, Any]:
        """
        获取债券市场数据
        
        Args:
            period: 数据周期
            use_cache: 是否使用缓存
            
        Returns:
            债券数据字典
        """
        cache_key = f"bond_data_{period}"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_key, max_age_hours=4)
            if cached_data:
                return cached_data
        
        bond_data = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'treasury_yields': {},
            'bond_etfs': {},
            'yield_curve': {},
            'credit_spreads': {}
        }
        
        try:
            # 获取国债收益率数据
            for symbol, description in self.bond_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        current_yield = hist['Close'].iloc[-1]
                        prev_yield = hist['Close'].iloc[-2] if len(hist) > 1 else current_yield
                        change = current_yield - prev_yield
                        
                        # 计算统计指标
                        avg_yield = hist['Close'].mean()
                        max_yield = hist['Close'].max()
                        min_yield = hist['Close'].min()
                        volatility = hist['Close'].std()
                        
                        bond_data['treasury_yields'][symbol] = {
                            'name': description,
                            'current_yield': float(current_yield),
                            'previous_yield': float(prev_yield),
                            'change': float(change),
                            'change_bps': float(change * 100),  # 基点变化
                            'avg_yield': float(avg_yield),
                            'max_yield': float(max_yield),
                            'min_yield': float(min_yield),
                            'volatility': float(volatility),
                            'data_points': len(hist)
                        }
                        
                        logger.info(f"✅ 获取{description}数据成功")
                    
                except Exception as e:
                    logger.error(f"❌ 获取{symbol}数据失败: {e}")
                    continue
            
            # 构建收益率曲线
            yield_curve_points = {}
            maturity_mapping = {
                '^IRX': 0.25,    # 3个月
                '^FVX': 5,       # 5年
                '^TNX': 10,      # 10年
                '^TYX': 30       # 30年
            }
            
            for symbol, maturity in maturity_mapping.items():
                if symbol in bond_data['treasury_yields']:
                    yield_curve_points[maturity] = bond_data['treasury_yields'][symbol]['current_yield']
            
            if len(yield_curve_points) >= 2:
                # 计算收益率曲线斜率
                maturities = sorted(yield_curve_points.keys())
                yields = [yield_curve_points[m] for m in maturities]
                
                # 2-10年期利差
                if 10 in yield_curve_points and 0.25 in yield_curve_points:
                    spread_10y_3m = yield_curve_points[10] - yield_curve_points[0.25]
                    bond_data['yield_curve']['10y_3m_spread'] = spread_10y_3m
                    bond_data['yield_curve']['curve_shape'] = 'normal' if spread_10y_3m > 0 else 'inverted'
                
                # 10-2年期利差（如果有2年期数据）
                if 10 in yield_curve_points and 5 in yield_curve_points:
                    spread_10y_5y = yield_curve_points[10] - yield_curve_points[5]
                    bond_data['yield_curve']['10y_5y_spread'] = spread_10y_5y
                
                bond_data['yield_curve']['points'] = yield_curve_points
                bond_data['yield_curve']['maturities'] = maturities
                bond_data['yield_curve']['yields'] = yields
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(cache_key, bond_data)
            
            logger.info("债券市场数据获取完成")
            return bond_data
            
        except Exception as e:
            logger.error(f"获取债券数据失败: {e}")
            bond_data['error'] = str(e)
            return bond_data
    
    def get_commodity_data(self, period: str = '1y', use_cache: bool = True) -> Dict[str, Any]:
        """
        获取商品数据
        
        Args:
            period: 数据周期
            use_cache: 是否使用缓存
            
        Returns:
            商品数据字典
        """
        cache_key = f"commodity_data_{period}"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_key, max_age_hours=4)
            if cached_data:
                return cached_data
        
        commodity_data = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'commodities': {},
            'sectors': {
                'precious_metals': [],
                'energy': [],
                'agriculture': [],
                'industrial_metals': []
            },
            'correlations': {}
        }
        
        try:
            # 获取各商品数据
            for symbol, description in self.commodity_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                        
                        # 计算技术指标
                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)  # 年化波动率
                        
                        # 计算移动平均线
                        ma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
                        ma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
                        
                        commodity_info = {
                            'name': description,
                            'current_price': float(current_price),
                            'previous_price': float(prev_price),
                            'change': float(change),
                            'change_percent': float(change_pct),
                            'volatility': float(volatility),
                            'ma_20': float(ma_20),
                            'ma_50': float(ma_50),
                            'trend': 'up' if current_price > ma_20 else 'down',
                            'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                            'high_52w': float(hist['Close'].max()),
                            'low_52w': float(hist['Close'].min())
                        }
                        
                        commodity_data['commodities'][symbol] = commodity_info
                        
                        # 按类别分组
                        if symbol in ['GC=F', 'SI=F', 'PL=F']:
                            commodity_data['sectors']['precious_metals'].append(symbol)
                        elif symbol in ['CL=F', 'NG=F']:
                            commodity_data['sectors']['energy'].append(symbol)
                        elif symbol in ['ZC=F', 'ZS=F', 'ZW=F']:
                            commodity_data['sectors']['agriculture'].append(symbol)
                        elif symbol in ['HG=F']:
                            commodity_data['sectors']['industrial_metals'].append(symbol)
                        
                        logger.info(f"✅ 获取{description}数据成功")
                    
                except Exception as e:
                    logger.error(f"❌ 获取{symbol}数据失败: {e}")
                    continue
            
            # 计算板块表现
            for sector, symbols in commodity_data['sectors'].items():
                if symbols:
                    sector_changes = [commodity_data['commodities'][s]['change_percent'] for s in symbols if s in commodity_data['commodities']]
                    if sector_changes:
                        commodity_data['sectors'][sector] = {
                            'symbols': symbols,
                            'avg_change': np.mean(sector_changes),
                            'best_performer': max(symbols, key=lambda s: commodity_data['commodities'].get(s, {}).get('change_percent', -999)),
                            'worst_performer': min(symbols, key=lambda s: commodity_data['commodities'].get(s, {}).get('change_percent', 999))
                        }
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(cache_key, commodity_data)
            
            logger.info("商品数据获取完成")
            return commodity_data
            
        except Exception as e:
            logger.error(f"获取商品数据失败: {e}")
            commodity_data['error'] = str(e)
            return commodity_data
    
    def get_crypto_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取加密货币数据
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            加密货币数据字典
        """
        cache_key = "crypto_data"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_key, max_age_hours=1)
            if cached_data:
                return cached_data
        
        crypto_data = {
            'timestamp': datetime.now().isoformat(),
            'cryptocurrencies': {},
            'market_summary': {},
            'dominance': {}
        }
        
        try:
            # 从Yahoo Finance获取加密货币数据
            for symbol, description in self.crypto_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='30d')
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                        
                        # 计算技术指标
                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(365)  # 年化波动率
                        
                        crypto_info = {
                            'name': description,
                            'current_price': float(current_price),
                            'previous_price': float(prev_price),
                            'change': float(change),
                            'change_percent': float(change_pct),
                            'volatility': float(volatility),
                            'volume_24h': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                            'market_cap': info.get('marketCap', 0) if info else 0,
                            'high_30d': float(hist['Close'].max()),
                            'low_30d': float(hist['Close'].min())
                        }
                        
                        crypto_data['cryptocurrencies'][symbol] = crypto_info
                        logger.info(f"✅ 获取{description}数据成功")
                    
                except Exception as e:
                    logger.error(f"❌ 获取{symbol}数据失败: {e}")
                    continue
            
            # 计算市场摘要
            if crypto_data['cryptocurrencies']:
                total_market_cap = sum([c.get('market_cap', 0) for c in crypto_data['cryptocurrencies'].values()])
                avg_change = np.mean([c['change_percent'] for c in crypto_data['cryptocurrencies'].values()])
                
                crypto_data['market_summary'] = {
                    'total_market_cap': total_market_cap,
                    'avg_change_percent': avg_change,
                    'coins_up': len([c for c in crypto_data['cryptocurrencies'].values() if c['change_percent'] > 0]),
                    'coins_down': len([c for c in crypto_data['cryptocurrencies'].values() if c['change_percent'] < 0]),
                    'most_volatile': max(crypto_data['cryptocurrencies'].keys(), 
                                       key=lambda k: crypto_data['cryptocurrencies'][k]['volatility'])
                }
                
                # 计算比特币市值占比
                btc_market_cap = crypto_data['cryptocurrencies'].get('BTC-USD', {}).get('market_cap', 0)
                if total_market_cap > 0 and btc_market_cap > 0:
                    btc_dominance = (btc_market_cap / total_market_cap) * 100
                    crypto_data['dominance']['btc_dominance'] = btc_dominance
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(cache_key, crypto_data)
            
            logger.info("加密货币数据获取完成")
            return crypto_data
            
        except Exception as e:
            logger.error(f"获取加密货币数据失败: {e}")
            crypto_data['error'] = str(e)
            return crypto_data
    
    def get_forex_data(self, period: str = '1mo', use_cache: bool = True) -> Dict[str, Any]:
        """
        获取外汇数据
        
        Args:
            period: 数据周期
            use_cache: 是否使用缓存
            
        Returns:
            外汇数据字典
        """
        cache_key = f"forex_data_{period}"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_key, max_age_hours=4)
            if cached_data:
                return cached_data
        
        forex_data = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'currency_pairs': {},
            'dollar_index': {},
            'volatility_ranking': []
        }
        
        try:
            # 获取各货币对数据
            for symbol, description in self.forex_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        current_rate = hist['Close'].iloc[-1]
                        prev_rate = hist['Close'].iloc[-2] if len(hist) > 1 else current_rate
                        change = current_rate - prev_rate
                        change_pct = (change / prev_rate) * 100 if prev_rate != 0 else 0
                        
                        # 计算技术指标
                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)  # 年化波动率
                        
                        # 计算移动平均线
                        ma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_rate
                        
                        forex_info = {
                            'name': description,
                            'current_rate': float(current_rate),
                            'previous_rate': float(prev_rate),
                            'change': float(change),
                            'change_percent': float(change_pct),
                            'volatility': float(volatility),
                            'ma_20': float(ma_20),
                            'trend': 'up' if current_rate > ma_20 else 'down',
                            'high_period': float(hist['Close'].max()),
                            'low_period': float(hist['Close'].min())
                        }
                        
                        forex_data['currency_pairs'][symbol] = forex_info
                        logger.info(f"✅ 获取{description}数据成功")
                    
                except Exception as e:
                    logger.error(f"❌ 获取{symbol}数据失败: {e}")
                    continue
            
            # 按波动率排序
            if forex_data['currency_pairs']:
                sorted_pairs = sorted(
                    forex_data['currency_pairs'].items(),
                    key=lambda x: x[1]['volatility'],
                    reverse=True
                )
                forex_data['volatility_ranking'] = [
                    {'symbol': symbol, 'volatility': data['volatility']}
                    for symbol, data in sorted_pairs
                ]
            
            # 尝试获取美元指数
            try:
                dxy_ticker = yf.Ticker('DX-Y.NYB')
                dxy_hist = dxy_ticker.history(period=period)
                
                if not dxy_hist.empty:
                    dxy_current = dxy_hist['Close'].iloc[-1]
                    dxy_prev = dxy_hist['Close'].iloc[-2] if len(dxy_hist) > 1 else dxy_current
                    dxy_change = dxy_current - dxy_prev
                    dxy_change_pct = (dxy_change / dxy_prev) * 100 if dxy_prev != 0 else 0
                    
                    forex_data['dollar_index'] = {
                        'current_value': float(dxy_current),
                        'change': float(dxy_change),
                        'change_percent': float(dxy_change_pct),
                        'strength': 'strong' if dxy_change_pct > 0.5 else 'weak' if dxy_change_pct < -0.5 else 'neutral'
                    }
                
            except Exception as e:
                logger.warning(f"获取美元指数失败: {e}")
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(cache_key, forex_data)
            
            logger.info("外汇数据获取完成")
            return forex_data
            
        except Exception as e:
            logger.error(f"获取外汇数据失败: {e}")
            forex_data['error'] = str(e)
            return forex_data
    
    def generate_alternative_data_report(self) -> Dict[str, Any]:
        """
        生成另类数据综合报告
        
        Returns:
            另类数据综合报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'bond_market': {},
            'commodity_market': {},
            'crypto_market': {},
            'forex_market': {},
            'options_sentiment': {},
            'cross_asset_analysis': {}
        }
        
        try:
            # 获取各类数据
            logger.info("开始生成另类数据综合报告...")
            
            # 债券市场
            bond_data = self.get_bond_data()
            report['bond_market'] = bond_data
            
            # 商品市场
            commodity_data = self.get_commodity_data()
            report['commodity_market'] = commodity_data
            
            # 加密货币市场
            crypto_data = self.get_crypto_data()
            report['crypto_market'] = crypto_data
            
            # 外汇市场
            forex_data = self.get_forex_data()
            report['forex_market'] = forex_data
            
            # 跨资产分析
            cross_asset = {}
            
            # 风险偏好分析
            risk_indicators = []
            
            # 从债券收益率看风险偏好
            if 'treasury_yields' in bond_data and '^TNX' in bond_data['treasury_yields']:
                tnx_change = bond_data['treasury_yields']['^TNX'].get('change_bps', 0)
                if tnx_change > 5:
                    risk_indicators.append('rising_yields')
                elif tnx_change < -5:
                    risk_indicators.append('falling_yields')
            
            # 从商品价格看通胀预期
            if 'commodities' in commodity_data and 'GC=F' in commodity_data['commodities']:
                gold_change = commodity_data['commodities']['GC=F'].get('change_percent', 0)
                if gold_change > 2:
                    risk_indicators.append('gold_strength')
                elif gold_change < -2:
                    risk_indicators.append('gold_weakness')
            
            # 从美元指数看全球风险偏好
            if 'dollar_index' in forex_data and forex_data['dollar_index']:
                dxy_change = forex_data['dollar_index'].get('change_percent', 0)
                if dxy_change > 1:
                    risk_indicators.append('dollar_strength')
                elif dxy_change < -1:
                    risk_indicators.append('dollar_weakness')
            
            cross_asset['risk_indicators'] = risk_indicators
            cross_asset['risk_sentiment'] = self._analyze_risk_sentiment(risk_indicators)
            
            report['cross_asset_analysis'] = cross_asset
            
            logger.info("另类数据综合报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"生成另类数据报告失败: {e}")
            report['error'] = str(e)
            return report
    
    def _analyze_risk_sentiment(self, risk_indicators: List[str]) -> str:
        """
        分析风险情绪
        
        Args:
            risk_indicators: 风险指标列表
            
        Returns:
            风险情绪评估
        """
        risk_on_signals = ['falling_yields', 'gold_weakness', 'dollar_weakness']
        risk_off_signals = ['rising_yields', 'gold_strength', 'dollar_strength']
        
        risk_on_count = sum(1 for signal in risk_indicators if signal in risk_on_signals)
        risk_off_count = sum(1 for signal in risk_indicators if signal in risk_off_signals)
        
        if risk_off_count > risk_on_count:
            return 'risk_off'
        elif risk_on_count > risk_off_count:
            return 'risk_on'
        else:
            return 'neutral'


# 创建全局实例
alternative_data_manager = AlternativeDataManager()