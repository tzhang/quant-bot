"""
统一数据适配器
集成Qlib和yfinance数据源，提供统一的数据访问接口
"""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd

# 导入数据提供器
try:
    from .qlib_data_provider import QlibDataProvider, QLIB_AVAILABLE
except ImportError:
    QLIB_AVAILABLE = False
    QlibDataProvider = None

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataAdapter:
    """
    统一数据适配器
    
    优先使用Qlib数据（高性能、预处理），
    当Qlib数据不可用时回退到yfinance API
    """
    
    def __init__(self, 
                 prefer_qlib: bool = True,
                 qlib_data_dir: Optional[str] = None,
                 fallback_to_yfinance: bool = True):
        """
        初始化数据适配器
        
        Args:
            prefer_qlib: 是否优先使用Qlib数据
            qlib_data_dir: Qlib数据目录
            fallback_to_yfinance: 当Qlib不可用时是否回退到yfinance
        """
        self.prefer_qlib = prefer_qlib and QLIB_AVAILABLE
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
        
        # 检查可用的数据源
        self._log_available_sources()
    
    def _log_available_sources(self):
        """记录可用的数据源"""
        sources = []
        if self.prefer_qlib and self.qlib_provider:
            sources.append("Qlib (primary)")
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
                      force_yfinance: bool = False) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            force_yfinance: 强制使用yfinance
        
        Returns:
            股票数据DataFrame
        """
        # 标准化股票代码
        symbol = symbol.upper().strip()
        
        # 如果强制使用yfinance或Qlib不可用
        if force_yfinance or not self.prefer_qlib:
            return self._get_yfinance_data(symbol, start_date, end_date)
        
        # 首先尝试Qlib
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
        
        # 回退到yfinance
        if self.fallback_to_yfinance:
            logger.info(f"Falling back to yfinance for {symbol}")
            return self._get_yfinance_data(symbol, start_date, end_date)
        
        logger.error(f"No data source available for {symbol}")
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
                               force_yfinance: bool = False) -> Dict[str, pd.DataFrame]:
        """
        获取多个股票的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            force_yfinance: 强制使用yfinance
        
        Returns:
            字典，键为股票代码，值为DataFrame
        """
        result = {}
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, start_date, end_date, force_yfinance)
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
            source: 数据源 ("qlib", "yfinance", "auto")
        
        Returns:
            股票代码列表
        """
        if source == "qlib" or (source == "auto" and self.prefer_qlib):
            if self.qlib_provider:
                try:
                    return self.qlib_provider.get_available_instruments()
                except Exception as e:
                    logger.error(f"Error getting Qlib instruments: {e}")
        
        # yfinance没有直接的方法获取所有股票列表
        # 返回一些常见的股票代码作为示例
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
        elif result["yfinance_available"]:
            result["recommended_source"] = "yfinance"
        else:
            result["recommended_source"] = "none"
        
        return result
    
    def is_data_available(self, symbol: str) -> bool:
        """
        简单检查数据是否可用
        
        Args:
            symbol: 股票代码
            
        Returns:
            数据是否可用
        """
        availability = self.check_data_availability(symbol)
        return availability["recommended_source"] != "none"
    
    def get_data_source_info(self) -> Dict[str, Union[str, bool, int]]:
        """
        获取数据源信息
        
        Returns:
            数据源信息字典
        """
        info = {
            "qlib_available": bool(self.qlib_provider),
            "yfinance_available": YFINANCE_AVAILABLE,
            "preferred_source": "qlib" if self.prefer_qlib else "yfinance",
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
        
        return info


# 便捷函数
def create_data_adapter(prefer_qlib: bool = True, 
                       qlib_data_dir: Optional[str] = None,
                       fallback_to_yfinance: bool = True) -> DataAdapter:
    """
    创建数据适配器实例
    
    Args:
        prefer_qlib: 是否优先使用Qlib
        qlib_data_dir: Qlib数据目录
        fallback_to_yfinance: 是否启用yfinance回退
    
    Returns:
        DataAdapter实例
    """
    return DataAdapter(prefer_qlib, qlib_data_dir, fallback_to_yfinance)


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