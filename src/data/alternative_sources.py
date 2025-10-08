"""
备用数据源配置模块

提供多种股票数据源的配置和切换功能，当yfinance频率限制时可以切换到其他数据源。
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            symbol: 股票符号
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass


class AlphaVantageSource(DataSource):
    """Alpha Vantage数据源"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从Alpha Vantage获取数据"""
        if not self.is_available():
            raise ValueError("Alpha Vantage API密钥未配置")
        
        try:
            import requests
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logger.warning(f"Alpha Vantage返回异常数据: {data}")
                return pd.DataFrame()
            
            # 转换数据格式
            time_series = data['Time Series (Daily)']
            df_data = []
            
            for date_str, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # 过滤日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            return df
            
        except Exception as e:
            logger.error(f"Alpha Vantage数据获取失败: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """检查Alpha Vantage是否可用"""
        return self.api_key is not None and len(self.api_key.strip()) > 0


class TiingoSource(DataSource):
    """Tiingo数据源"""
    
    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")
        self.base_url = "https://api.tiingo.com/tiingo/daily"
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从Tiingo获取数据"""
        if not self.is_available():
            raise ValueError("Tiingo API密钥未配置")
        
        try:
            import requests
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Token {self.api_key}'
            }
            
            url = f"{self.base_url}/{symbol}/prices"
            params = {
                'startDate': start_date,
                'endDate': end_date,
                'format': 'json'
            }
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # 转换数据格式
            df_data = []
            for item in data:
                df_data.append({
                    'Date': pd.to_datetime(item['date']),
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': int(item['volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Tiingo数据获取失败: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """检查Tiingo是否可用"""
        return self.api_key is not None and len(self.api_key.strip()) > 0


class QuandlSource(DataSource):
    """Quandl数据源"""
    
    def __init__(self):
        self.api_key = os.getenv("QUANDL_API_KEY")
        self.base_url = "https://www.quandl.com/api/v3/datasets"
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从Quandl获取数据"""
        if not self.is_available():
            raise ValueError("Quandl API密钥未配置")
        
        try:
            import requests
            # 使用WIKI数据集（免费但已停止更新）
            dataset_code = f"WIKI/{symbol}"
            url = f"{self.base_url}/{dataset_code}/data.json"
            
            params = {
                'api_key': self.api_key,
                'start_date': start_date,
                'end_date': end_date,
                'order': 'asc'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'dataset_data' not in data or not data['dataset_data']['data']:
                return pd.DataFrame()
            
            # 转换数据格式
            columns = data['dataset_data']['column_names']
            rows = data['dataset_data']['data']
            
            df = pd.DataFrame(rows, columns=columns)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # 重命名列
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 选择需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Quandl数据获取失败: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """检查Quandl是否可用"""
        return self.api_key is not None and len(self.api_key.strip()) > 0


class DataSourceManager:
    """数据源管理器"""
    
    def __init__(self):
        self.sources = {
            'alphavantage': AlphaVantageSource(),
            'tiingo': TiingoSource(),
            'quandl': QuandlSource()
        }
        self.preferred_order = ['alphavantage', 'tiingo', 'quandl']
    
    def get_available_sources(self) -> Dict[str, DataSource]:
        """获取可用的数据源"""
        available = {}
        for name, source in self.sources.items():
            if source.is_available():
                available[name] = source
        return available
    
    def fetch_data_with_fallback(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        使用备用数据源获取数据
        
        按优先级顺序尝试各个数据源，直到成功获取数据或所有源都失败
        """
        available_sources = self.get_available_sources()
        
        if not available_sources:
            logger.warning("没有可用的备用数据源")
            return pd.DataFrame()
        
        for source_name in self.preferred_order:
            if source_name in available_sources:
                logger.info(f"尝试使用备用数据源: {source_name}")
                try:
                    df = available_sources[source_name].fetch_data(symbol, start_date, end_date)
                    if not df.empty:
                        logger.info(f"成功从 {source_name} 获取 {symbol} 数据")
                        return df
                except Exception as e:
                    logger.warning(f"备用数据源 {source_name} 失败: {e}")
                    continue
        
        logger.error(f"所有备用数据源都无法获取 {symbol} 数据")
        return pd.DataFrame()
    
    def get_source_status(self) -> Dict[str, bool]:
        """获取所有数据源的状态"""
        status = {}
        for name, source in self.sources.items():
            status[name] = source.is_available()
        return status


# 全局实例
alternative_data_manager = DataSourceManager()