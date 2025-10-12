
# ==========================================
# 迁移说明 - 2025-10-10 23:06:36
# ==========================================
# 本文件已从yfinance迁移到IB TWS API
# 原始文件备份在: backup_before_ib_migration/src/data/fundamental_data.py
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
基本面数据获取模块

提供股票基本面数据的获取、处理和缓存功能：
1. 财务报表数据（资产负债表、利润表、现金流量表）
2. 估值指标（PE、PB、PS、EV/EBITDA等）
3. 盈利能力指标（ROE、ROA、毛利率、净利率等）
4. 成长性指标（营收增长率、净利润增长率等）

数据源：
- 主要：Alpha Vantage API
- 备用：Yahoo Finance、SEC EDGAR
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


class FundamentalDataManager:
    """
    基本面数据管理器
    
    统一管理各种基本面数据源，提供财务数据获取、缓存和处理功能
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        初始化基本面数据管理器
        
        Args:
            alpha_vantage_key: Alpha Vantage API密钥
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        
        # 设置缓存目录
        self.cache_dir = Path("data_cache/fundamental")
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
        
        # API调用限制（Alpha Vantage免费版：5 calls per minute）
        self.api_call_interval = 12  # 秒
        self.last_api_call = 0
    
    def _wait_for_api_limit(self):
        """等待API调用限制"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.api_call_interval:
            wait_time = self.api_call_interval - time_since_last_call
            logger.info(f"等待API限制: {wait_time:.1f}秒")
            time.sleep(wait_time)
        self.last_api_call = time.time()
    
    def _get_cache_path(self, symbol: str, data_type: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{data_type}_{symbol}.json"
    
    def _load_from_cache(self, symbol: str, data_type: str, max_age_days: int = 1) -> Optional[Dict]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(symbol, data_type)
        
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
                logger.debug(f"从缓存加载数据: {symbol} {data_type}")
                return data
                
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, data_type: str, data: Dict):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(symbol, data_type)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                logger.debug(f"数据已缓存: {symbol} {data_type}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def get_income_statement(self, symbol: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        获取利润表数据
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            包含利润表数据的DataFrame
        """
        data_type = "income_statement"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(symbol, data_type)
            if cached_data:
                return pd.DataFrame(cached_data)
        
        # 从Alpha Vantage获取数据
        if not self.alpha_vantage_key:
            logger.warning("未设置Alpha Vantage API密钥，尝试使用Yahoo Finance")
            return self._get_income_statement_yfinance(symbol)
        
        try:
            self._wait_for_api_limit()
            
            params = {
                'function': 'INCOME_STATEMENT',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查API响应
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage错误: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage限制: {data['Note']}")
                return None
            
            # 处理年度报告数据
            annual_reports = data.get('annualReports', [])
            if not annual_reports:
                logger.warning(f"未找到{symbol}的利润表数据")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(annual_reports)
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('fiscalDateEnding', ascending=False)
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(symbol, data_type, df.to_dict('records'))
            
            logger.info(f"成功获取{symbol}利润表数据，共{len(df)}年")
            return df
            
        except Exception as e:
            logger.error(f"获取利润表数据失败: {e}")
            # 回退到Yahoo Finance
            return self._get_income_statement_yfinance(symbol)
    
    def _get_income_statement_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """使用Yahoo Finance获取利润表数据（备用方案）"""
        try:
            ticker = yf.Ticker(symbol)
            income_stmt = ticker.financials
            
            if income_stmt.empty:
                return None
            
            # 转换为标准格式
            df = income_stmt.T
            df.index.name = 'fiscalDateEnding'
            df = df.reset_index()
            df = df.sort_values('fiscalDateEnding', ascending=False)
            
            logger.info(f"从Yahoo Finance获取{symbol}利润表数据")
            return df
            
        except Exception as e:
            logger.error(f"Yahoo Finance获取利润表失败: {e}")
            return None
    
    def get_balance_sheet(self, symbol: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        获取资产负债表数据
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            包含资产负债表数据的DataFrame
        """
        data_type = "balance_sheet"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(symbol, data_type)
            if cached_data:
                return pd.DataFrame(cached_data)
        
        # 从Alpha Vantage获取数据
        if not self.alpha_vantage_key:
            logger.warning("未设置Alpha Vantage API密钥，尝试使用Yahoo Finance")
            return self._get_balance_sheet_yfinance(symbol)
        
        try:
            self._wait_for_api_limit()
            
            params = {
                'function': 'BALANCE_SHEET',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查API响应
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage错误: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage限制: {data['Note']}")
                return None
            
            # 处理年度报告数据
            annual_reports = data.get('annualReports', [])
            if not annual_reports:
                logger.warning(f"未找到{symbol}的资产负债表数据")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(annual_reports)
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('fiscalDateEnding', ascending=False)
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(symbol, data_type, df.to_dict('records'))
            
            logger.info(f"成功获取{symbol}资产负债表数据，共{len(df)}年")
            return df
            
        except Exception as e:
            logger.error(f"获取资产负债表数据失败: {e}")
            # 回退到Yahoo Finance
            return self._get_balance_sheet_yfinance(symbol)
    
    def _get_balance_sheet_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """使用Yahoo Finance获取资产负债表数据（备用方案）"""
        try:
            ticker = yf.Ticker(symbol)
            balance_sheet = ticker.balance_sheet
            
            if balance_sheet.empty:
                return None
            
            # 转换为标准格式
            df = balance_sheet.T
            df.index.name = 'fiscalDateEnding'
            df = df.reset_index()
            df = df.sort_values('fiscalDateEnding', ascending=False)
            
            logger.info(f"从Yahoo Finance获取{symbol}资产负债表数据")
            return df
            
        except Exception as e:
            logger.error(f"Yahoo Finance获取资产负债表失败: {e}")
            return None
    
    def get_cash_flow(self, symbol: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        获取现金流量表数据
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            包含现金流量表数据的DataFrame
        """
        data_type = "cash_flow"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(symbol, data_type)
            if cached_data:
                return pd.DataFrame(cached_data)
        
        # 从Alpha Vantage获取数据
        if not self.alpha_vantage_key:
            logger.warning("未设置Alpha Vantage API密钥，尝试使用Yahoo Finance")
            return self._get_cash_flow_yfinance(symbol)
        
        try:
            self._wait_for_api_limit()
            
            params = {
                'function': 'CASH_FLOW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查API响应
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage错误: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage限制: {data['Note']}")
                return None
            
            # 处理年度报告数据
            annual_reports = data.get('annualReports', [])
            if not annual_reports:
                logger.warning(f"未找到{symbol}的现金流量表数据")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(annual_reports)
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('fiscalDateEnding', ascending=False)
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(symbol, data_type, df.to_dict('records'))
            
            logger.info(f"成功获取{symbol}现金流量表数据，共{len(df)}年")
            return df
            
        except Exception as e:
            logger.error(f"获取现金流量表数据失败: {e}")
            # 回退到Yahoo Finance
            return self._get_cash_flow_yfinance(symbol)
    
    def _get_cash_flow_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """使用Yahoo Finance获取现金流量表数据（备用方案）"""
        try:
            ticker = yf.Ticker(symbol)
            cash_flow = ticker.cashflow
            
            if cash_flow.empty:
                return None
            
            # 转换为标准格式
            df = cash_flow.T
            df.index.name = 'fiscalDateEnding'
            df = df.reset_index()
            df = df.sort_values('fiscalDateEnding', ascending=False)
            
            logger.info(f"从Yahoo Finance获取{symbol}现金流量表数据")
            return df
            
        except Exception as e:
            logger.error(f"Yahoo Finance获取现金流量表失败: {e}")
            return None
    
    def calculate_financial_ratios(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        计算关键财务比率
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含财务比率的字典
        """
        try:
            # 获取财务报表数据
            income_stmt = self.get_income_statement(symbol)
            balance_sheet = self.get_balance_sheet(symbol)
            
            if income_stmt is None or balance_sheet is None:
                logger.warning(f"无法获取{symbol}的完整财务数据")
                return None
            
            # 获取最新年度数据
            latest_income = income_stmt.iloc[0]
            latest_balance = balance_sheet.iloc[0]
            
            ratios = {}
            
            # 盈利能力指标
            try:
                total_revenue = float(latest_income.get('totalRevenue', 0))
                net_income = float(latest_income.get('netIncome', 0))
                total_assets = float(latest_balance.get('totalAssets', 0))
                shareholders_equity = float(latest_balance.get('totalShareholderEquity', 0))
                
                if total_revenue > 0:
                    ratios['net_profit_margin'] = net_income / total_revenue  # 净利率
                
                if total_assets > 0:
                    ratios['roa'] = net_income / total_assets  # 资产收益率
                
                if shareholders_equity > 0:
                    ratios['roe'] = net_income / shareholders_equity  # 净资产收益率
                
            except (ValueError, TypeError) as e:
                logger.warning(f"计算盈利能力指标失败: {e}")
            
            # 偿债能力指标
            try:
                current_assets = float(latest_balance.get('totalCurrentAssets', 0))
                current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 0))
                total_debt = float(latest_balance.get('shortLongTermDebtTotal', 0))
                
                if current_liabilities > 0:
                    ratios['current_ratio'] = current_assets / current_liabilities  # 流动比率
                
                if shareholders_equity > 0:
                    ratios['debt_to_equity'] = total_debt / shareholders_equity  # 负债权益比
                
            except (ValueError, TypeError) as e:
                logger.warning(f"计算偿债能力指标失败: {e}")
            
            logger.info(f"成功计算{symbol}的财务比率，共{len(ratios)}个指标")
            return ratios
            
        except Exception as e:
            logger.error(f"计算财务比率失败: {e}")
            return None
    
    def get_company_overview(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        获取公司概况数据
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            包含公司概况的字典
        """
        data_type = "company_overview"
        
        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(symbol, data_type)
            if cached_data:
                return cached_data
        
        # 从Alpha Vantage获取数据
        if not self.alpha_vantage_key:
            logger.warning("未设置Alpha Vantage API密钥，尝试使用Yahoo Finance")
            return self._get_company_overview_yfinance(symbol)
        
        try:
            self._wait_for_api_limit()
            
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查API响应
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage错误: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage限制: {data['Note']}")
                return None
            
            # 缓存数据
            if use_cache:
                self._save_to_cache(symbol, data_type, data)
            
            logger.info(f"成功获取{symbol}公司概况数据")
            return data
            
        except Exception as e:
            logger.error(f"获取公司概况失败: {e}")
            # 回退到Yahoo Finance
            return self._get_company_overview_yfinance(symbol)
    
    def _get_company_overview_yfinance(self, symbol: str) -> Optional[Dict]:
        """使用Yahoo Finance获取公司概况（备用方案）"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # 提取关键信息
            overview = {
                'Symbol': symbol,
                'Name': info.get('longName', ''),
                'Description': info.get('longBusinessSummary', ''),
                'Sector': info.get('sector', ''),
                'Industry': info.get('industry', ''),
                'MarketCapitalization': info.get('marketCap', 0),
                'PERatio': info.get('trailingPE', 0),
                'PEGRatio': info.get('pegRatio', 0),
                'BookValue': info.get('bookValue', 0),
                'DividendYield': info.get('dividendYield', 0),
                'EPS': info.get('trailingEps', 0),
                'Beta': info.get('beta', 0),
                '52WeekHigh': info.get('fiftyTwoWeekHigh', 0),
                '52WeekLow': info.get('fiftyTwoWeekLow', 0),
            }
            
            logger.info(f"从Yahoo Finance获取{symbol}公司概况")
            return overview
            
        except Exception as e:
            logger.error(f"Yahoo Finance获取公司概况失败: {e}")
            return None


# 创建全局实例
fundamental_data_manager = FundamentalDataManager()