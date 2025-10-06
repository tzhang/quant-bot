"""
数据管理器模块

提供统一的数据获取、缓存和管理接口
集成Qlib数据源以提供高质量的股票数据
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path
import pickle
import hashlib

# 导入Qlib数据适配器
try:
    from .data_adapter import DataAdapter, create_data_adapter
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataManager:
    """
    数据管理器
    
    提供统一的数据获取、缓存和管理接口，支持多数据源
    优先使用Qlib数据源，回退到yfinance
    """
    
    def __init__(self, cache_dir: str = "data_cache", enable_cache: bool = True):
        """
        初始化数据管理器
        
        Args:
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
        """
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.cache_dir.mkdir(exist_ok=True)
        
        # 初始化数据适配器
        if QLIB_AVAILABLE:
            try:
                self.data_adapter = create_data_adapter()
                logger.info("DataManager初始化完成，使用Qlib数据适配器")
            except Exception as e:
                logger.warning(f"Qlib数据适配器初始化失败: {e}，将使用yfinance")
                self.data_adapter = None
        else:
            logger.info("Qlib不可用，使用yfinance作为数据源")
            self.data_adapter = None
        
        # 缓存过期时间（小时）
        self.cache_expiry = {
            'daily': 24,      # 日线数据24小时过期
            'intraday': 1,    # 分钟数据1小时过期
            'fundamental': 168 # 基本面数据7天过期
        }
        
        logger.info(f"DataManager初始化完成，缓存目录: {self.cache_dir}")
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, 
                      data_type: str = 'daily') -> str:
        """生成缓存键"""
        key_str = f"{symbol}_{start_date}_{end_date}_{data_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path, data_type: str = 'daily') -> bool:
        """检查缓存是否有效"""
        if not cache_path.exists():
            return False
        
        # 检查文件修改时间
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_hours = self.cache_expiry.get(data_type, 24)
        
        return datetime.now() - file_time < timedelta(hours=expiry_hours)
    
    def _load_from_cache(self, cache_key: str, data_type: str = 'daily') -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        if not self.enable_cache:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, data_type):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"从缓存加载数据: {cache_key}")
                return data
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
        
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """保存数据到缓存"""
        if not self.enable_cache:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"数据已缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str,
                      data_type: str = 'daily') -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            data_type: 数据类型 ('daily', 'intraday')
            
        Returns:
            包含股票数据的DataFrame
        """
        # 生成缓存键
        cache_key = self._get_cache_key(symbol, start_date, end_date, data_type)
        
        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key, data_type)
        if cached_data is not None:
            return cached_data
        
        # 优先使用数据适配器（Qlib）
        if self.data_adapter and data_type == 'daily':
            try:
                data = self.data_adapter.get_stock_data(symbol, start_date, end_date)
                if not data.empty:
                    # 数据清洗
                    data = self._clean_data(data)
                    # 保存到缓存
                    self._save_to_cache(data, cache_key)
                    logger.info(f"成功从Qlib获取数据: {symbol}, {len(data)}条记录")
                    return data
                else:
                    logger.warning(f"Qlib未获取到数据: {symbol}，尝试yfinance")
            except Exception as e:
                logger.warning(f"Qlib获取数据失败 {symbol}: {e}，尝试yfinance")
        
        # 回退到yfinance获取数据
        try:
            ticker = yf.Ticker(symbol)
            
            if data_type == 'daily':
                data = ticker.history(start=start_date, end=end_date)
            elif data_type == 'intraday':
                # 获取1分钟数据（最近7天）
                data = ticker.history(period="7d", interval="1m")
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")
            
            if data.empty:
                logger.warning(f"未获取到数据: {symbol}")
                return pd.DataFrame()
            
            # 数据清洗
            data = self._clean_data(data)
            
            # 保存到缓存
            self._save_to_cache(data, cache_key)
            
            logger.info(f"成功从yfinance获取数据: {symbol}, {len(data)}条记录")
            return data
            
        except Exception as e:
            logger.error(f"获取股票数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: List[str], start_date: str, 
                               end_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            字典，键为股票代码，值为对应的DataFrame
        """
        results = {}
        
        # 优先使用数据适配器批量获取
        if self.data_adapter:
            try:
                batch_results = self.data_adapter.get_multiple_stocks_data(symbols, start_date, end_date)
                for symbol, data in batch_results.items():
                    if not data.empty:
                        results[symbol] = data
                        logger.info(f"从Qlib获取股票 {symbol} 数据: {len(data)}条记录")
                    else:
                        logger.warning(f"Qlib未获取到股票 {symbol} 数据")
                
                # 检查是否有未获取到的股票，使用yfinance补充
                missing_symbols = [s for s in symbols if s not in results]
                if missing_symbols:
                    logger.info(f"使用yfinance补充获取 {len(missing_symbols)} 只股票数据")
                    for symbol in missing_symbols:
                        try:
                            data = self.get_stock_data(symbol, start_date, end_date)
                            if not data.empty:
                                results[symbol] = data
                        except Exception as e:
                            logger.error(f"获取股票 {symbol} 数据失败: {e}")
                
            except Exception as e:
                logger.warning(f"批量获取数据失败: {e}，使用单独获取方式")
                # 回退到单独获取
                for symbol in symbols:
                    try:
                        data = self.get_stock_data(symbol, start_date, end_date)
                        if not data.empty:
                            results[symbol] = data
                        else:
                            logger.warning(f"股票 {symbol} 数据为空")
                    except Exception as e:
                        logger.error(f"获取股票 {symbol} 数据失败: {e}")
        else:
            # 没有数据适配器，使用原有方式
            for symbol in symbols:
                try:
                    data = self.get_stock_data(symbol, start_date, end_date)
                    if not data.empty:
                        results[symbol] = data
                    else:
                        logger.warning(f"股票 {symbol} 数据为空")
                except Exception as e:
                    logger.error(f"获取股票 {symbol} 数据失败: {e}")
        
        logger.info(f"批量获取完成，成功获取 {len(results)}/{len(symbols)} 只股票数据")
        return results
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取基本面数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含基本面数据的字典
        """
        cache_key = f"fundamental_{symbol}"
        
        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key, 'fundamental')
        if cached_data is not None:
            return cached_data.to_dict() if isinstance(cached_data, pd.DataFrame) else cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取基本面信息
            info = ticker.info
            
            # 提取关键指标
            fundamental_data = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'eps': info.get('trailingEps'),
                'revenue': info.get('totalRevenue'),
                'profit_margin': info.get('profitMargins'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
            # 保存到缓存
            df = pd.DataFrame([fundamental_data])
            self._save_to_cache(df, cache_key)
            
            logger.info(f"成功获取基本面数据: {symbol}")
            return fundamental_data
            
        except Exception as e:
            logger.error(f"获取基本面数据失败 {symbol}: {e}")
            return {}
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        if data.empty:
            return data
        
        # 删除重复行
        data = data.drop_duplicates()
        
        # 处理缺失值
        data = data.dropna()
        
        # 标准化列名为大写（保持向后兼容）
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        data = data.rename(columns=column_mapping)
        
        # 确保数值列为数值类型
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 删除异常值（价格为0或负数）
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        return data
    
    def clear_cache(self, symbol: str = None) -> None:
        """
        清理缓存
        
        Args:
            symbol: 指定股票代码，如果为None则清理所有缓存
        """
        if symbol:
            # 清理指定股票的缓存
            cache_files = list(self.cache_dir.glob(f"*{symbol}*.pkl"))
        else:
            # 清理所有缓存
            cache_files = list(self.cache_dir.glob("*.pkl"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.info(f"已删除缓存文件: {cache_file.name}")
            except Exception as e:
                logger.warning(f"删除缓存文件失败 {cache_file.name}: {e}")
        
        logger.info(f"缓存清理完成，删除了 {len(cache_files)} 个文件")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            缓存统计信息
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # 添加数据源信息
        data_source_info = {
            'qlib_available': QLIB_AVAILABLE and self.data_adapter is not None,
            'primary_source': 'Qlib' if (QLIB_AVAILABLE and self.data_adapter) else 'yfinance'
        }
        
        # 如果有数据适配器，获取其信息
        if self.data_adapter:
            try:
                adapter_info = self.data_adapter.get_data_info()
                data_source_info.update(adapter_info)
            except Exception as e:
                logger.warning(f"获取数据适配器信息失败: {e}")
        
        return {
            'cache_dir': str(self.cache_dir),
            'file_count': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'enable_cache': self.enable_cache,
            'data_source': data_source_info
        }
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的股票代码列表
        
        Returns:
            可用股票代码列表
        """
        if self.data_adapter:
            try:
                return self.data_adapter.get_available_symbols()
            except Exception as e:
                logger.warning(f"获取可用股票代码失败: {e}")
        
        # 如果没有数据适配器，返回常用股票代码
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
    def check_data_availability(self, symbol: str, start_date: str = None, end_date: str = None) -> bool:
        """
        检查数据可用性
        
        Args:
            symbol: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            数据是否可用
        """
        if self.data_adapter:
            try:
                return self.data_adapter.is_data_available(symbol)
            except Exception as e:
                logger.warning(f"检查数据可用性失败: {e}")
        
        # 回退到尝试获取数据
        try:
            # 如果没有提供日期，使用默认的短期范围
            if not start_date or not end_date:
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            data = self.get_stock_data(symbol, start_date, end_date)
            return not data.empty
        except Exception:
            return False