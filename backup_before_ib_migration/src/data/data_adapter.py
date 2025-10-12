"""
多数据源适配器

该模块提供统一的数据访问接口，集成多个数据源：
- Qlib: 本地量化数据
- OpenBB: 开源金融数据平台
- yfinance: Yahoo Finance数据

支持数据源优先级和回退机制。

作者: AI Assistant  
日期: 2024
"""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)

# 导入数据提供器
try:
    from .qlib_data_provider import QlibDataProvider, QLIB_AVAILABLE
except ImportError:
    QLIB_AVAILABLE = False
    QlibDataProvider = None

try:
    from .openbb_data_provider import OpenBBDataProvider
    OPENBB_AVAILABLE = True
except ImportError as e:
    OPENBB_AVAILABLE = False
    logger.warning(f"OpenBB provider not available: {e}")
    OpenBBDataProvider = None

try:
    from .ib_data_provider import IBDataProvider, IB_AVAILABLE
except ImportError as e:
    IB_AVAILABLE = False
    logger.warning(f"IB provider not available: {e}")
    IBDataProvider = None

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class DataAdapter:
    """
    多数据源适配器
    
    集成Qlib、OpenBB和yfinance数据源，提供统一的数据访问接口。
    支持数据源优先级和自动回退机制：
    1. 优先使用Qlib（本地数据，速度快）
    2. 回退到OpenBB（开源平台，数据丰富）
    3. 最后使用yfinance（备用数据源）
    """
    
    def __init__(self, 
                 prefer_qlib: bool = True,
                 qlib_data_dir: Optional[str] = None,
                 enable_openbb: bool = True,
                 enable_ib: bool = True,
                 fallback_to_yfinance: bool = True):
        """
        初始化数据适配器
        
        Args:
            prefer_qlib: 是否优先使用Qlib数据
            qlib_data_dir: Qlib数据目录
            enable_openbb: 是否启用OpenBB数据源
            enable_ib: 是否启用Interactive Brokers数据源
            fallback_to_yfinance: 当其他数据源不可用时是否回退到yfinance
        """
        self.prefer_qlib = prefer_qlib and QLIB_AVAILABLE
        self.enable_openbb = enable_openbb and OPENBB_AVAILABLE
        self.enable_ib = enable_ib and IB_AVAILABLE
        self.fallback_to_yfinance = fallback_to_yfinance and YFINANCE_AVAILABLE
        
        # 初始化Qlib提供器
        self.qlib_provider = None
        if self.prefer_qlib:
            try:
                self.qlib_provider = QlibDataProvider(qlib_data_dir)
                logger.info("Qlib data provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Qlib provider: {e}")
                self.prefer_qlib = False
        
        # 初始化OpenBB提供器
        self.openbb_provider = None
        if self.enable_openbb:
            try:
                self.openbb_provider = OpenBBDataProvider()
                logger.info("OpenBB data provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenBB provider: {e}")
                self.enable_openbb = False
        
        # 初始化IB提供器
        self.ib_provider = None
        if self.enable_ib:
            try:
                self.ib_provider = IBDataProvider()
                logger.info("Interactive Brokers data provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize IB provider: {e}")
                self.enable_ib = False
        
        # 检查可用的数据源
        self._log_available_sources()
    
    def _log_available_sources(self):
        """记录可用的数据源"""
        sources = []
        if self.prefer_qlib and self.qlib_provider:
            sources.append("Qlib (primary)")
        if self.enable_openbb and self.openbb_provider:
            sources.append("OpenBB (secondary)")
        if self.enable_ib and self.ib_provider:
            sources.append("Interactive Brokers (real-time)")
        if self.fallback_to_yfinance:
            sources.append("yfinance (fallback)")
        
        if sources:
            logger.info(f"Available data sources: {', '.join(sources)}")
        else:
            logger.error("No data sources available!")
    
    def get_stock_data(self, 
                      symbol: str, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      force_source: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            force_source: 强制使用指定数据源 ('qlib', 'openbb', 'ib', 'yfinance')
        
        Returns:
            股票数据DataFrame
        """
        # 标准化股票代码
        symbol = symbol.upper().strip()
        
        # 如果指定了强制数据源
        if force_source:
            return self._get_data_from_source(symbol, start_date, end_date, force_source)
        
        # 按优先级尝试各数据源
        
        # 1. 首先尝试Qlib
        if self.prefer_qlib and self.qlib_provider:
            try:
                data = self.qlib_provider.get_stock_data(
                    symbol.lower(), start_date, end_date
                )
                
                if not data.empty:
                    logger.info(f"Retrieved {len(data)} records for {symbol} from Qlib")
                    return data
                else:
                    logger.warning(f"No Qlib data found for {symbol}")
            except Exception as e:
                logger.warning(f"Qlib data retrieval failed for {symbol}: {e}")
        
        # 2. 回退到OpenBB
        if self.enable_openbb and self.openbb_provider:
            try:
                logger.info(f"Trying OpenBB for {symbol}")
                data = self.openbb_provider.get_stock_data(symbol, start_date, end_date)
                
                if not data.empty:
                    logger.info(f"Retrieved {len(data)} records for {symbol} from OpenBB")
                    return data
                else:
                    logger.warning(f"No OpenBB data found for {symbol}")
            except Exception as e:
                logger.warning(f"OpenBB data retrieval failed for {symbol}: {e}")
        
        # 3. 尝试Interactive Brokers
        if self.enable_ib and self.ib_provider:
            try:
                logger.info(f"Trying Interactive Brokers for {symbol}")
                data = self.ib_provider.get_stock_data(symbol, start_date, end_date)
                
                if not data.empty:
                    logger.info(f"Retrieved {len(data)} records for {symbol} from IB")
                    return data
                else:
                    logger.warning(f"No IB data found for {symbol}")
            except Exception as e:
                logger.warning(f"IB data retrieval failed for {symbol}: {e}")
        
        # 4. 最后回退到yfinance
        if self.fallback_to_yfinance:
            logger.info(f"Falling back to yfinance for {symbol}")
            return self._get_yfinance_data(symbol, start_date, end_date)
        
        logger.error(f"No data source available for {symbol}")
        return pd.DataFrame()
    
    def _get_data_from_source(self, 
                             symbol: str, 
                             start_date: Optional[str], 
                             end_date: Optional[str], 
                             source: str) -> pd.DataFrame:
        """
        从指定数据源获取数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源名称
        
        Returns:
            股票数据DataFrame
        """
        if source.lower() == "qlib" and self.qlib_provider:
            return self.qlib_provider.get_stock_data(symbol.lower(), start_date, end_date)
        elif source.lower() == "openbb" and self.openbb_provider:
            return self.openbb_provider.get_stock_data(symbol, start_date, end_date)
        elif source.lower() == "ib" and self.ib_provider:
            return self.ib_provider.get_stock_data(symbol, start_date, end_date)
        elif source.lower() == "yfinance":
            return self._get_yfinance_data(symbol, start_date, end_date)
        else:
            logger.error(f"Data source '{source}' not available")
            return pd.DataFrame()
    
    def _get_yfinance_data(self, 
                          symbol: str, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        使用yfinance获取数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            股票数据DataFrame
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance is not available")
            return pd.DataFrame()
        
        try:
            # 设置默认日期范围
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # 获取数据
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No yfinance data found for {symbol}")
                return pd.DataFrame()
            
            # 标准化列名
            data.columns = [col.lower() for col in data.columns]
            
            # 移除不需要的列
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
            data = data[[col for col in columns_to_keep if col in data.columns]]
            
            logger.info(f"Retrieved {len(data)} records for {symbol} from yfinance")
            return data
            
        except Exception as e:
            logger.error(f"yfinance data retrieval failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, 
                               symbols: List[str], 
                               start_date: Optional[str] = None, 
                               end_date: Optional[str] = None,
                               force_source: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取多个股票的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            force_source: 强制使用指定数据源
        
        Returns:
            字典，键为股票代码，值为DataFrame
        """
        result = {}
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, start_date, end_date, force_source)
                if not data.empty:
                    result[symbol] = data
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Error retrieving data for {symbol}: {e}")
        
        return result
    
    def get_available_symbols(self, source: str = "auto") -> List[str]:
        """
        获取可用的股票代码列表
        
        Args:
            source: 数据源 ("qlib", "openbb", "yfinance", "auto")
        
        Returns:
            股票代码列表
        """
        if source == "qlib" or (source == "auto" and self.prefer_qlib and self.qlib_provider):
            try:
                return self.qlib_provider.get_available_instruments()
            except Exception as e:
                logger.error(f"Error getting Qlib instruments: {e}")
        
        if source == "openbb" or (source == "auto" and self.enable_openbb and self.openbb_provider):
            try:
                return self.openbb_provider.get_available_instruments()
            except Exception as e:
                logger.error(f"Error getting OpenBB instruments: {e}")
        
        # yfinance和默认情况
        if source == "yfinance" or source == "auto":
            return [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "ADBE", "CRM", "ORCL", "IBM", "INTC", "AMD", "QCOM", "AVGO"
            ]
        
        return []
    
    def check_data_availability(self, symbol: str) -> Dict[str, Union[str, bool, int]]:
        """
        检查数据可用性
        
        Args:
            symbol: 股票代码
        
        Returns:
            数据可用性信息
        """
        result = {
            "symbol": symbol.upper(),
            "qlib_available": False,
            "openbb_available": False,
            "yfinance_available": False,
            "recommended_source": None
        }
        
        # 检查Qlib
        if self.qlib_provider:
            try:
                qlib_info = self.qlib_provider.check_data_availability(symbol.lower())
                result["qlib_available"] = qlib_info.get("available", False)
                if result["qlib_available"]:
                    result["qlib_records"] = qlib_info.get("records_count", 0)
                    result["qlib_date_range"] = qlib_info.get("date_range", "")
            except Exception as e:
                logger.error(f"Error checking Qlib availability for {symbol}: {e}")
        
        # 检查OpenBB
        if self.openbb_provider:
            try:
                openbb_info = self.openbb_provider.check_data_availability(symbol)
                result["openbb_available"] = openbb_info.get("available", False)
                if result["openbb_available"]:
                    result["openbb_records"] = openbb_info.get("records_count", 0)
                    result["openbb_date_range"] = openbb_info.get("date_range", "")
            except Exception as e:
                logger.error(f"Error checking OpenBB availability for {symbol}: {e}")
        
        # 检查yfinance
        if YFINANCE_AVAILABLE:
            try:
                test_data = self._get_yfinance_data(
                    symbol, 
                    (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d")
                )
                result["yfinance_available"] = not test_data.empty
                if result["yfinance_available"]:
                    result["yfinance_records"] = len(test_data)
            except Exception as e:
                logger.error(f"Error checking yfinance availability for {symbol}: {e}")
        
        # 推荐数据源
        if result["qlib_available"]:
            result["recommended_source"] = "qlib"
        elif result["openbb_available"]:
            result["recommended_source"] = "openbb"
        elif result["yfinance_available"]:
            result["recommended_source"] = "yfinance"
        else:
            result["recommended_source"] = "none"
        
        return result
    
    def is_data_available(self, symbol: str, source: str = "auto") -> bool:
        """
        检查指定数据源是否有数据
        
        Args:
            symbol: 股票代码
            source: 数据源 ("qlib", "openbb", "yfinance", "auto")
        
        Returns:
            是否有数据
        """
        availability = self.check_data_availability(symbol)
        
        if source == "qlib":
            return availability.get("qlib_available", False)
        elif source == "openbb":
            return availability.get("openbb_available", False)
        elif source == "yfinance":
            return availability.get("yfinance_available", False)
        elif source == "auto":
            return (availability.get("qlib_available", False) or 
                   availability.get("openbb_available", False) or
                   availability.get("yfinance_available", False))
        
        return False
    
    def get_data_source_info(self) -> Dict[str, Union[str, bool, int]]:
        """
        获取数据源信息
        
        Returns:
            数据源信息字典
        """
        info = {
            "qlib_available": bool(self.qlib_provider),
            "openbb_available": bool(self.openbb_provider),
            "yfinance_available": YFINANCE_AVAILABLE,
            "preferred_source": "qlib" if self.prefer_qlib else "openbb" if self.enable_openbb else "yfinance",
            "fallback_enabled": self.fallback_to_yfinance
        }
        
        # Qlib信息
        if self.qlib_provider:
            try:
                qlib_info = self.qlib_provider.get_data_info()
                info.update({
                    "qlib_instruments_count": qlib_info.get("total_instruments", 0),
                    "qlib_date_range": qlib_info.get("date_range", ""),
                    "qlib_data_directory": qlib_info.get("data_directory", "")
                })
            except Exception as e:
                logger.error(f"Error getting Qlib info: {e}")
        
        # OpenBB信息
        if self.openbb_provider:
            try:
                openbb_info = self.openbb_provider.get_data_info()
                info.update({
                    "openbb_instruments_count": openbb_info.get("total_instruments", 0),
                    "openbb_date_range": openbb_info.get("date_range", ""),
                    "openbb_provider_info": openbb_info.get("provider_info", "")
                })
            except Exception as e:
                logger.error(f"Error getting OpenBB info: {e}")
        
        return info


# 便捷函数
def create_data_adapter(prefer_qlib: bool = True, 
                       qlib_data_dir: Optional[str] = None,
                       enable_openbb: bool = True,
                       enable_ib: bool = True,
                       fallback_to_yfinance: bool = True) -> DataAdapter:
    """
    创建数据适配器实例
    
    Args:
        prefer_qlib: 是否优先使用Qlib
        qlib_data_dir: Qlib数据目录
        enable_openbb: 是否启用OpenBB
        enable_ib: 是否启用Interactive Brokers
        fallback_to_yfinance: 是否启用yfinance回退
    
    Returns:
        DataAdapter实例
    """
    return DataAdapter(
        prefer_qlib=prefer_qlib,
        qlib_data_dir=qlib_data_dir,
        enable_openbb=enable_openbb,
        enable_ib=enable_ib,
        fallback_to_yfinance=fallback_to_yfinance
    )


def get_stock_data(symbol: str, 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None,
                  prefer_qlib: bool = True) -> pd.DataFrame:
    """
    便捷函数：获取股票数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        prefer_qlib: 是否优先使用Qlib
    
    Returns:
        股票数据DataFrame
    """
    adapter = create_data_adapter(prefer_qlib=prefer_qlib)
    return adapter.get_stock_data(symbol, start_date, end_date)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据适配器
    adapter = create_data_adapter()
    
    # 获取数据源信息
    print("数据源信息:")
    info = adapter.get_data_source_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试获取AAPL数据
    print("\n测试获取AAPL数据:")
    aapl_data = adapter.get_stock_data("AAPL", "2020-01-01", "2020-03-01")
    if not aapl_data.empty:
        print(f"获取到 {len(aapl_data)} 条AAPL数据")
        print(aapl_data.head())
    else:
        print("未获取到AAPL数据")
    
    # 检查数据可用性
    print("\n检查AAPL数据可用性:")
    availability = adapter.check_data_availability("AAPL")
    for key, value in availability.items():
        print(f"  {key}: {value}")
    
    # 测试多个股票
    print("\n测试获取多个股票数据:")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    multi_data = adapter.get_multiple_stocks_data(symbols, "2020-01-01", "2020-02-01")
    for symbol, data in multi_data.items():
        print(f"  {symbol}: {len(data)} records")