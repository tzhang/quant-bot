
# ==========================================
# 迁移说明 - 2025-10-10 23:06:36
# ==========================================
# 本文件已从yfinance迁移到IB TWS API
# 原始文件备份在: backup_before_ib_migration/src/data/macro_data.py
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
宏观经济数据模块

提供宏观经济数据的获取、处理和分析功能：
1. 利率数据（联邦基金利率、国债收益率等）
2. 通胀数据（CPI、PCE等）
3. 经济增长数据（GDP、就业数据等）
4. 货币政策数据（货币供应量、央行政策等）
5. 国际经济数据（汇率、大宗商品等）

数据源：
- 主要：FRED (Federal Reserve Economic Data)
- 备用：Yahoo Finance、Alpha Vantage
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.data.ib_data_provider import IBDataProvider, IBConfig

logger = logging.getLogger(__name__)


class MacroDataManager:
    """
    宏观经济数据管理器
    
    统一管理各种宏观经济数据源，提供经济指标获取、缓存和分析功能
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        初始化宏观数据管理器
        
        Args:
            fred_api_key: FRED API密钥
        """
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        
        # 设置缓存目录
        self.cache_dir = Path("data_cache/macro")
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
        
        # API调用限制（FRED：120 calls per minute）
        self.api_call_interval = 0.5  # 秒
        self.last_api_call = 0
        
        # 预定义的经济指标
        self.economic_indicators = {
            # 利率指标
            'interest_rates': {
                'FEDFUNDS': '联邦基金利率',
                'DGS10': '10年期国债收益率',
                'DGS2': '2年期国债收益率',
                'DGS30': '30年期国债收益率',
                'TB3MS': '3个月国债利率',
                'AAA': 'AAA级公司债收益率',
                'BAA': 'BAA级公司债收益率',
            },
            # 通胀指标
            'inflation': {
                'CPIAUCSL': '消费者价格指数(CPI)',
                'CPILFESL': '核心CPI(除食品和能源)',
                'PCEPI': '个人消费支出价格指数(PCE)',
                'PCEPILFE': '核心PCE',
                'DFEDTARU': 'Fed目标通胀率',
            },
            # 经济增长指标
            'growth': {
                'GDP': '国内生产总值',
                'GDPC1': '实际GDP',
                'GDPPOT': '潜在GDP',
                'NYGDPMKTPCDWLD': '世界GDP增长率',
                'INDPRO': '工业生产指数',
                'PAYEMS': '非农就业人数',
                'UNRATE': '失业率',
                'CIVPART': '劳动参与率',
            },
            # 货币政策指标
            'monetary': {
                'M1SL': 'M1货币供应量',
                'M2SL': 'M2货币供应量',
                'BOGMBASE': '货币基础',
                'WALCL': 'Fed资产负债表总资产',
                'EFFR': '有效联邦基金利率',
            },
            # 国际经济指标
            'international': {
                'DEXUSEU': '美元/欧元汇率',
                'DEXJPUS': '日元/美元汇率',
                'DEXCHUS': '人民币/美元汇率',
                'GOLDAMGBD228NLBM': '黄金价格',
                'DCOILWTICO': 'WTI原油价格',
            }
        }
    
    def _wait_for_api_limit(self):
        """等待API调用限制"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.api_call_interval:
            wait_time = self.api_call_interval - time_since_last_call
            time.sleep(wait_time)
        self.last_api_call = time.time()
    
    def _get_cache_path(self, series_id: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{series_id}.json"
    
    def _load_from_cache(self, series_id: str, max_age_days: int = 1) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(series_id)
        
        if not cache_path.exists():
            return None
        
        try:
            # 检查缓存文件年龄
            file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if file_age.days > max_age_days:
                logger.debug(f"缓存文件过期: {cache_path}")
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                logger.debug(f"从缓存加载数据: {series_id}")
                return df
                
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None
    
    def _save_to_cache(self, series_id: str, data: pd.DataFrame):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(series_id)
        
        try:
            # 重置索引以便序列化
            df_to_save = data.reset_index()
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(df_to_save.to_dict('records'), f, indent=2, default=str)
                logger.debug(f"数据已缓存: {series_id}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def get_fred_series(self, series_id: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        从FRED获取经济数据序列
        
        Args:
            series_id: FRED序列ID
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            use_cache: 是否使用缓存
            
        Returns:
            包含经济数据的DataFrame
        """
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(series_id)
            if cached_data is not None:
                # 根据日期范围过滤
                if start_date:
                    cached_data = cached_data[cached_data.index >= start_date]
                if end_date:
                    cached_data = cached_data[cached_data.index <= end_date]
                return cached_data
        
        # 从FRED API获取数据
        if not self.fred_api_key:
            logger.warning("未设置FRED API密钥，尝试使用备用数据源")
            return self._get_series_fallback(series_id, start_date, end_date)
        
        try:
            self._wait_for_api_limit()
            
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json'
            }
            
            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date
            
            url = f"{self.fred_base_url}/series/observations"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查API响应
            if 'error_code' in data:
                logger.error(f"FRED API错误: {data.get('error_message', '未知错误')}")
                return None
            
            observations = data.get('observations', [])
            if not observations:
                logger.warning(f"未找到{series_id}的数据")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.set_index('date')
            df = df[['value']].dropna()
            df.columns = [series_id]
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(series_id, df)
            
            logger.info(f"成功获取FRED数据: {series_id}, 共{len(df)}个观测值")
            return df
            
        except Exception as e:
            logger.error(f"获取FRED数据失败: {e}")
            # 回退到备用数据源
            return self._get_series_fallback(series_id, start_date, end_date)
    
    def _get_series_fallback(self, series_id: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """备用数据源（使用预定义的Yahoo Finance映射）"""
        # FRED ID到Yahoo Finance符号的映射
        yahoo_mapping = {
            'DGS10': '^TNX',  # 10年期国债收益率
            'DGS2': '^IRX',   # 2年期国债收益率（近似）
            'GOLDAMGBD228NLBM': 'GC=F',  # 黄金期货
            'DCOILWTICO': 'CL=F',  # WTI原油期货
        }
        
        yahoo_symbol = yahoo_mapping.get(series_id)
        if not yahoo_symbol:
            logger.warning(f"无备用数据源: {series_id}")
            return None
        
        try:
            # 使用yfinance获取数据
            ticker = yf.Ticker(yahoo_symbol)
            
            if start_date and end_date:
                hist = ticker.history(start=start_date, end=end_date)
            else:
                hist = ticker.history(period='5y')  # 默认5年数据
            
            if hist.empty:
                return None
            
            # 使用收盘价作为值
            df = pd.DataFrame(hist['Close'])
            df.columns = [series_id]
            df.index.name = 'date'
            
            logger.info(f"从Yahoo Finance获取{series_id}数据")
            return df
            
        except Exception as e:
            logger.error(f"备用数据源获取失败: {e}")
            return None
    
    def get_interest_rates(self, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取利率数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含各种利率指标的字典
        """
        interest_data = {}
        
        for series_id, description in self.economic_indicators['interest_rates'].items():
            try:
                data = self.get_fred_series(series_id, start_date, end_date)
                if data is not None:
                    interest_data[series_id] = data
                    logger.info(f"✅ 获取{description}数据成功")
                else:
                    logger.warning(f"⚠️ 获取{description}数据失败")
            except Exception as e:
                logger.error(f"❌ 获取{description}数据出错: {e}")
        
        return interest_data
    
    def get_inflation_data(self, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取通胀数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含各种通胀指标的字典
        """
        inflation_data = {}
        
        for series_id, description in self.economic_indicators['inflation'].items():
            try:
                data = self.get_fred_series(series_id, start_date, end_date)
                if data is not None:
                    inflation_data[series_id] = data
                    logger.info(f"✅ 获取{description}数据成功")
                else:
                    logger.warning(f"⚠️ 获取{description}数据失败")
            except Exception as e:
                logger.error(f"❌ 获取{description}数据出错: {e}")
        
        return inflation_data
    
    def get_growth_indicators(self, start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取经济增长指标
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含各种增长指标的字典
        """
        growth_data = {}
        
        for series_id, description in self.economic_indicators['growth'].items():
            try:
                data = self.get_fred_series(series_id, start_date, end_date)
                if data is not None:
                    growth_data[series_id] = data
                    logger.info(f"✅ 获取{description}数据成功")
                else:
                    logger.warning(f"⚠️ 获取{description}数据失败")
            except Exception as e:
                logger.error(f"❌ 获取{description}数据出错: {e}")
        
        return growth_data
    
    def calculate_yield_curve(self, date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        计算收益率曲线
        
        Args:
            date: 指定日期，None表示最新数据
            
        Returns:
            收益率曲线DataFrame
        """
        try:
            # 获取不同期限的国债收益率
            yield_series = {
                '3M': 'TB3MS',
                '2Y': 'DGS2',
                '10Y': 'DGS10',
                '30Y': 'DGS30'
            }
            
            yield_data = {}
            for maturity, series_id in yield_series.items():
                data = self.get_fred_series(series_id)
                if data is not None:
                    if date:
                        # 获取指定日期的数据
                        closest_date = data.index[data.index <= date][-1] if any(data.index <= date) else data.index[0]
                        yield_data[maturity] = data.loc[closest_date, series_id]
                    else:
                        # 获取最新数据
                        yield_data[maturity] = data.iloc[-1, 0]
            
            if not yield_data:
                return None
            
            # 创建收益率曲线DataFrame
            curve_df = pd.DataFrame(list(yield_data.items()), columns=['Maturity', 'Yield'])
            curve_df['Maturity_Days'] = curve_df['Maturity'].map({
                '3M': 90, '2Y': 730, '10Y': 3650, '30Y': 10950
            })
            curve_df = curve_df.sort_values('Maturity_Days')
            
            logger.info("收益率曲线计算完成")
            return curve_df
            
        except Exception as e:
            logger.error(f"收益率曲线计算失败: {e}")
            return None
    
    def get_economic_summary(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取经济数据摘要
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            经济数据摘要字典
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'period': f"{start_date} to {end_date}" if start_date and end_date else "Latest available",
            'interest_rates': {},
            'inflation': {},
            'growth': {},
            'yield_curve': None
        }
        
        try:
            # 获取关键利率指标
            fed_funds = self.get_fred_series('FEDFUNDS', start_date, end_date)
            if fed_funds is not None:
                summary['interest_rates']['fed_funds_rate'] = {
                    'current': float(fed_funds.iloc[-1, 0]),
                    'change_1m': float(fed_funds.iloc[-1, 0] - fed_funds.iloc[-30, 0]) if len(fed_funds) > 30 else None
                }
            
            # 获取通胀数据
            cpi = self.get_fred_series('CPIAUCSL', start_date, end_date)
            if cpi is not None and len(cpi) > 12:
                # 计算年化通胀率
                current_cpi = cpi.iloc[-1, 0]
                year_ago_cpi = cpi.iloc[-12, 0]
                inflation_rate = (current_cpi / year_ago_cpi - 1) * 100
                summary['inflation']['cpi_yoy'] = float(inflation_rate)
            
            # 获取失业率
            unemployment = self.get_fred_series('UNRATE', start_date, end_date)
            if unemployment is not None:
                summary['growth']['unemployment_rate'] = {
                    'current': float(unemployment.iloc[-1, 0]),
                    'change_1m': float(unemployment.iloc[-1, 0] - unemployment.iloc[-2, 0]) if len(unemployment) > 1 else None
                }
            
            # 获取收益率曲线
            summary['yield_curve'] = self.calculate_yield_curve()
            
            logger.info("经济数据摘要生成完成")
            return summary
            
        except Exception as e:
            logger.error(f"生成经济数据摘要失败: {e}")
            summary['error'] = str(e)
            return summary


# 创建全局实例
macro_data_manager = MacroDataManager()