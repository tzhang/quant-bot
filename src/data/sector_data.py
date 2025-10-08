#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业板块数据模块

提供行业板块数据的获取、分析和轮动策略支持：
1. GICS行业分类数据
2. 板块ETF数据获取
3. 板块相对强弱分析
4. 行业估值指标
5. 板块轮动信号
6. 资金流向分析

数据源：
- 主要：Yahoo Finance (板块ETF)
- 辅助：Alpha Vantage、FRED
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
import yfinance as yf
from scipy import stats

logger = logging.getLogger(__name__)


class SectorDataManager:
    """
    行业板块数据管理器
    
    提供行业板块数据获取、分析和轮动策略支持
    """
    
    def __init__(self):
        """初始化行业板块数据管理器"""
        # 设置缓存目录
        self.cache_dir = Path("data_cache/sector")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # GICS行业分类和对应的ETF
        self.sector_etfs = {
            # SPDR行业ETF
            'XLK': {'name': '科技', 'gics_sector': 'Information Technology'},
            'XLF': {'name': '金融', 'gics_sector': 'Financials'},
            'XLV': {'name': '医疗保健', 'gics_sector': 'Health Care'},
            'XLE': {'name': '能源', 'gics_sector': 'Energy'},
            'XLI': {'name': '工业', 'gics_sector': 'Industrials'},
            'XLY': {'name': '可选消费', 'gics_sector': 'Consumer Discretionary'},
            'XLP': {'name': '必需消费', 'gics_sector': 'Consumer Staples'},
            'XLU': {'name': '公用事业', 'gics_sector': 'Utilities'},
            'XLB': {'name': '材料', 'gics_sector': 'Materials'},
            'XLRE': {'name': '房地产', 'gics_sector': 'Real Estate'},
            'XLC': {'name': '通信服务', 'gics_sector': 'Communication Services'},
        }
        
        # 市场基准
        self.market_benchmark = 'SPY'  # 标普500 ETF
        
        # 板块轮动相关参数
        self.momentum_periods = [20, 60, 120]  # 动量计算周期
        self.volatility_period = 60  # 波动率计算周期
        
        # 缓存过期时间（小时）
        self.cache_expiry_hours = 4
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            # 检查缓存文件年龄
            file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if file_age.total_seconds() > self.cache_expiry_hours * 3600:
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
                json.dump(data, f, indent=2, default=str)
                logger.debug(f"数据已缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def get_sector_data(self, period: str = '1y', use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        获取所有板块ETF数据
        
        Args:
            period: 数据周期 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            use_cache: 是否使用缓存
            
        Returns:
            包含各板块数据的字典
        """
        cache_key = f"sector_data_{period}"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                # 转换回DataFrame格式
                sector_data = {}
                for symbol, data in cached_data.items():
                    df = pd.DataFrame(data)
                    if not df.empty and 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.set_index('Date')
                        sector_data[symbol] = df
                return sector_data
        
        # 从Yahoo Finance获取数据
        sector_data = {}
        symbols = list(self.sector_etfs.keys()) + [self.market_benchmark]
        
        try:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        sector_data[symbol] = hist
                        logger.info(f"✅ 获取{symbol}数据成功，共{len(hist)}条记录")
                    else:
                        logger.warning(f"⚠️ {symbol}数据为空")
                        
                except Exception as e:
                    logger.error(f"❌ 获取{symbol}数据失败: {e}")
                    continue
            
            # 缓存数据
            if use_cache and sector_data:
                cache_data = {}
                for symbol, df in sector_data.items():
                    cache_data[symbol] = df.reset_index().to_dict('records')
                self._save_to_cache(cache_key, cache_data)
            
            logger.info(f"成功获取{len(sector_data)}个板块的数据")
            return sector_data
            
        except Exception as e:
            logger.error(f"获取板块数据失败: {e}")
            return {}
    
    def calculate_sector_performance(self, sector_data: Dict[str, pd.DataFrame], 
                                   periods: List[int] = None) -> pd.DataFrame:
        """
        计算板块表现
        
        Args:
            sector_data: 板块数据字典
            periods: 计算周期列表（天数）
            
        Returns:
            板块表现DataFrame
        """
        if periods is None:
            periods = [1, 5, 20, 60, 120, 252]  # 1天、1周、1月、3月、6月、1年
        
        performance_data = []
        
        for symbol, data in sector_data.items():
            if symbol == self.market_benchmark:
                continue
                
            if data.empty or 'Close' not in data.columns:
                continue
            
            sector_info = self.sector_etfs.get(symbol, {})
            sector_name = sector_info.get('name', symbol)
            
            row = {
                'Symbol': symbol,
                'Sector': sector_name,
                'Current_Price': data['Close'].iloc[-1],
                'Volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0,
            }
            
            # 计算不同周期的收益率
            for period in periods:
                if len(data) > period:
                    start_price = data['Close'].iloc[-period-1]
                    end_price = data['Close'].iloc[-1]
                    return_pct = (end_price / start_price - 1) * 100
                    row[f'Return_{period}d'] = return_pct
                else:
                    row[f'Return_{period}d'] = np.nan
            
            # 计算波动率（年化）
            if len(data) > 20:
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                row['Volatility_Ann'] = volatility
            else:
                row['Volatility_Ann'] = np.nan
            
            performance_data.append(row)
        
        performance_df = pd.DataFrame(performance_data)
        
        # 按1个月收益率排序
        if 'Return_20d' in performance_df.columns:
            performance_df = performance_df.sort_values('Return_20d', ascending=False)
        
        logger.info(f"计算了{len(performance_df)}个板块的表现")
        return performance_df
    
    def calculate_relative_strength(self, sector_data: Dict[str, pd.DataFrame], 
                                  benchmark_symbol: str = None) -> pd.DataFrame:
        """
        计算板块相对强弱
        
        Args:
            sector_data: 板块数据字典
            benchmark_symbol: 基准符号，默认为SPY
            
        Returns:
            相对强弱DataFrame
        """
        if benchmark_symbol is None:
            benchmark_symbol = self.market_benchmark
        
        if benchmark_symbol not in sector_data:
            logger.error(f"基准数据{benchmark_symbol}不存在")
            return pd.DataFrame()
        
        benchmark_data = sector_data[benchmark_symbol]['Close']
        relative_strength_data = {}
        
        for symbol, data in sector_data.items():
            if symbol == benchmark_symbol or data.empty:
                continue
            
            sector_prices = data['Close']
            
            # 确保数据对齐
            aligned_data = pd.concat([sector_prices, benchmark_data], axis=1, join='inner')
            if aligned_data.empty:
                continue
            
            sector_aligned = aligned_data.iloc[:, 0]
            benchmark_aligned = aligned_data.iloc[:, 1]
            
            # 计算相对强弱比率
            relative_ratio = sector_aligned / benchmark_aligned
            
            # 计算不同周期的相对强弱
            rs_metrics = {}
            for period in self.momentum_periods:
                if len(relative_ratio) > period:
                    # 相对强弱变化率
                    rs_change = (relative_ratio.iloc[-1] / relative_ratio.iloc[-period-1] - 1) * 100
                    rs_metrics[f'RS_{period}d'] = rs_change
                    
                    # 相对强弱排名（百分位）
                    recent_rs = relative_ratio.iloc[-period:]
                    current_percentile = stats.percentileofscore(recent_rs, relative_ratio.iloc[-1])
                    rs_metrics[f'RS_Percentile_{period}d'] = current_percentile
            
            # 当前相对强弱比率
            rs_metrics['Current_RS_Ratio'] = relative_ratio.iloc[-1]
            rs_metrics['Symbol'] = symbol
            rs_metrics['Sector'] = self.sector_etfs.get(symbol, {}).get('name', symbol)
            
            relative_strength_data[symbol] = rs_metrics
        
        rs_df = pd.DataFrame.from_dict(relative_strength_data, orient='index')
        
        # 按20天相对强弱排序
        if 'RS_20d' in rs_df.columns:
            rs_df = rs_df.sort_values('RS_20d', ascending=False)
        
        logger.info(f"计算了{len(rs_df)}个板块的相对强弱")
        return rs_df
    
    def identify_sector_rotation_signals(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        识别板块轮动信号
        
        Args:
            sector_data: 板块数据字典
            
        Returns:
            板块轮动信号字典
        """
        signals = {
            'timestamp': datetime.now().isoformat(),
            'rotation_signals': [],
            'momentum_leaders': [],
            'momentum_laggards': [],
            'volatility_analysis': {},
            'summary': {}
        }
        
        try:
            # 计算板块表现
            performance_df = self.calculate_sector_performance(sector_data)
            if performance_df.empty:
                return signals
            
            # 计算相对强弱
            rs_df = self.calculate_relative_strength(sector_data)
            
            # 识别动量领先板块（20天和60天收益率都为正且排名前3）
            if 'Return_20d' in performance_df.columns and 'Return_60d' in performance_df.columns:
                momentum_condition = (
                    (performance_df['Return_20d'] > 0) & 
                    (performance_df['Return_60d'] > 0)
                )
                momentum_leaders = performance_df[momentum_condition].head(3)
                
                for _, row in momentum_leaders.iterrows():
                    signals['momentum_leaders'].append({
                        'symbol': row['Symbol'],
                        'sector': row['Sector'],
                        'return_20d': row['Return_20d'],
                        'return_60d': row['Return_60d'],
                        'volatility': row.get('Volatility_Ann', 0)
                    })
            
            # 识别动量落后板块
            if 'Return_20d' in performance_df.columns:
                laggard_condition = performance_df['Return_20d'] < -5  # 20天跌幅超过5%
                momentum_laggards = performance_df[laggard_condition].tail(3)
                
                for _, row in momentum_laggards.iterrows():
                    signals['momentum_laggards'].append({
                        'symbol': row['Symbol'],
                        'sector': row['Sector'],
                        'return_20d': row['Return_20d'],
                        'return_60d': row.get('Return_60d', 0),
                        'volatility': row.get('Volatility_Ann', 0)
                    })
            
            # 板块轮动信号
            if not rs_df.empty and 'RS_20d' in rs_df.columns:
                # 强势板块：相对强弱排名前25%且20天相对强弱为正
                strong_sectors = rs_df[
                    (rs_df['RS_Percentile_20d'] > 75) & 
                    (rs_df['RS_20d'] > 0)
                ]
                
                # 弱势板块：相对强弱排名后25%且20天相对强弱为负
                weak_sectors = rs_df[
                    (rs_df['RS_Percentile_20d'] < 25) & 
                    (rs_df['RS_20d'] < 0)
                ]
                
                for _, row in strong_sectors.iterrows():
                    signals['rotation_signals'].append({
                        'type': 'BUY',
                        'symbol': row['Symbol'],
                        'sector': row['Sector'],
                        'rs_20d': row['RS_20d'],
                        'rs_percentile': row['RS_Percentile_20d'],
                        'confidence': 'HIGH' if row['RS_Percentile_20d'] > 90 else 'MEDIUM'
                    })
                
                for _, row in weak_sectors.iterrows():
                    signals['rotation_signals'].append({
                        'type': 'SELL',
                        'symbol': row['Symbol'],
                        'sector': row['Sector'],
                        'rs_20d': row['RS_20d'],
                        'rs_percentile': row['RS_Percentile_20d'],
                        'confidence': 'HIGH' if row['RS_Percentile_20d'] < 10 else 'MEDIUM'
                    })
            
            # 波动率分析
            if 'Volatility_Ann' in performance_df.columns:
                vol_stats = performance_df['Volatility_Ann'].describe()
                signals['volatility_analysis'] = {
                    'mean_volatility': vol_stats['mean'],
                    'median_volatility': vol_stats['50%'],
                    'high_vol_threshold': vol_stats['75%'],
                    'low_vol_threshold': vol_stats['25%']
                }
                
                # 高波动率板块
                high_vol_sectors = performance_df[
                    performance_df['Volatility_Ann'] > vol_stats['75%']
                ]['Sector'].tolist()
                signals['volatility_analysis']['high_vol_sectors'] = high_vol_sectors
            
            # 生成摘要
            signals['summary'] = {
                'total_sectors_analyzed': len(performance_df),
                'buy_signals': len([s for s in signals['rotation_signals'] if s['type'] == 'BUY']),
                'sell_signals': len([s for s in signals['rotation_signals'] if s['type'] == 'SELL']),
                'momentum_leaders_count': len(signals['momentum_leaders']),
                'momentum_laggards_count': len(signals['momentum_laggards'])
            }
            
            logger.info("板块轮动信号分析完成")
            return signals
            
        except Exception as e:
            logger.error(f"板块轮动信号分析失败: {e}")
            signals['error'] = str(e)
            return signals
    
    def get_sector_valuation_metrics(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        获取板块估值指标
        
        Args:
            symbols: 板块ETF符号列表，None表示所有板块
            
        Returns:
            估值指标DataFrame
        """
        if symbols is None:
            symbols = list(self.sector_etfs.keys())
        
        valuation_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if not info:
                    continue
                
                sector_info = self.sector_etfs.get(symbol, {})
                
                valuation_row = {
                    'Symbol': symbol,
                    'Sector': sector_info.get('name', symbol),
                    'PE_Ratio': info.get('trailingPE'),
                    'Forward_PE': info.get('forwardPE'),
                    'PB_Ratio': info.get('priceToBook'),
                    'Dividend_Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                    'Market_Cap': info.get('marketCap'),
                    'Beta': info.get('beta'),
                    '52W_High': info.get('fiftyTwoWeekHigh'),
                    '52W_Low': info.get('fiftyTwoWeekLow'),
                    'Current_Price': info.get('currentPrice') or info.get('regularMarketPrice'),
                }
                
                # 计算距离52周高点/低点的距离
                if valuation_row['Current_Price'] and valuation_row['52W_High']:
                    valuation_row['Distance_from_High'] = (
                        (valuation_row['Current_Price'] / valuation_row['52W_High'] - 1) * 100
                    )
                
                if valuation_row['Current_Price'] and valuation_row['52W_Low']:
                    valuation_row['Distance_from_Low'] = (
                        (valuation_row['Current_Price'] / valuation_row['52W_Low'] - 1) * 100
                    )
                
                valuation_data.append(valuation_row)
                logger.info(f"✅ 获取{symbol}估值数据成功")
                
            except Exception as e:
                logger.error(f"❌ 获取{symbol}估值数据失败: {e}")
                continue
        
        valuation_df = pd.DataFrame(valuation_data)
        
        if not valuation_df.empty:
            # 按PE比率排序
            if 'PE_Ratio' in valuation_df.columns:
                valuation_df = valuation_df.sort_values('PE_Ratio', na_last=True)
        
        logger.info(f"获取了{len(valuation_df)}个板块的估值数据")
        return valuation_df
    
    def generate_sector_report(self, period: str = '1y') -> Dict[str, Any]:
        """
        生成板块分析报告
        
        Args:
            period: 分析周期
            
        Returns:
            板块分析报告字典
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period': period,
            'sector_performance': {},
            'relative_strength': {},
            'rotation_signals': {},
            'valuation_metrics': {},
            'summary': {}
        }
        
        try:
            # 获取板块数据
            sector_data = self.get_sector_data(period=period)
            if not sector_data:
                report['error'] = "无法获取板块数据"
                return report
            
            # 板块表现分析
            performance_df = self.calculate_sector_performance(sector_data)
            if not performance_df.empty:
                report['sector_performance'] = performance_df.to_dict('records')
            
            # 相对强弱分析
            rs_df = self.calculate_relative_strength(sector_data)
            if not rs_df.empty:
                report['relative_strength'] = rs_df.to_dict('records')
            
            # 轮动信号分析
            rotation_signals = self.identify_sector_rotation_signals(sector_data)
            report['rotation_signals'] = rotation_signals
            
            # 估值指标
            valuation_df = self.get_sector_valuation_metrics()
            if not valuation_df.empty:
                report['valuation_metrics'] = valuation_df.to_dict('records')
            
            # 生成摘要
            if not performance_df.empty:
                best_performer = performance_df.iloc[0] if 'Return_20d' in performance_df.columns else None
                worst_performer = performance_df.iloc[-1] if 'Return_20d' in performance_df.columns else None
                
                report['summary'] = {
                    'best_performer': {
                        'sector': best_performer['Sector'] if best_performer is not None else None,
                        'return_20d': best_performer['Return_20d'] if best_performer is not None else None
                    },
                    'worst_performer': {
                        'sector': worst_performer['Sector'] if worst_performer is not None else None,
                        'return_20d': worst_performer['Return_20d'] if worst_performer is not None else None
                    },
                    'total_buy_signals': len([s for s in rotation_signals.get('rotation_signals', []) if s['type'] == 'BUY']),
                    'total_sell_signals': len([s for s in rotation_signals.get('rotation_signals', []) if s['type'] == 'SELL']),
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            logger.info("板块分析报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"生成板块分析报告失败: {e}")
            report['error'] = str(e)
            return report


# 创建全局实例
sector_data_manager = SectorDataManager()