
# ==========================================
# 迁移说明 - 2025-10-10 23:06:36
# ==========================================
# 本文件已从yfinance迁移到IB TWS API
# 原始文件备份在: backup_before_ib_migration/src/data/data_adapter.py
# 
# 主要变更:
# # - 替换yfinance导入为IB导入
# 
# 注意事项:
# 1. 需要启动IB TWS或Gateway
# 2. 确保API设置已正确配置
# 3. 某些yfinance特有功能可能需要手动调整
# ==========================================

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
    yf = None

try:
    from .alpaca_data_provider import AlpacaDataProvider
    ALPACA_AVAILABLE = True
except ImportError as e:
    ALPACA_AVAILABLE = False
    logger.warning(f"Alpaca provider not available: {e}")
    AlpacaDataProvider = None


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
                 enable_alpaca: bool = False,
                 fallback_to_yfinance: bool = False):  # 默认禁用yfinance
        """
        初始化数据适配器
        
        Args:
            prefer_qlib: 是否优先使用Qlib数据
            qlib_data_dir: Qlib数据目录
            enable_openbb: 是否启用OpenBB数据源
            enable_ib: 是否启用Interactive Brokers数据源
            enable_alpaca: 是否启用Alpaca数据源
            fallback_to_yfinance: 当其他数据源不可用时是否回退到yfinance (默认禁用)
        """
        self.prefer_qlib = prefer_qlib and QLIB_AVAILABLE
        self.enable_openbb = enable_openbb and OPENBB_AVAILABLE
        self.enable_ib = enable_ib and IB_AVAILABLE
        self.enable_alpaca = enable_alpaca and ALPACA_AVAILABLE
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
        
        # 初始化Alpaca提供器
        self.alpaca_provider = None
        if self.enable_alpaca:
            try:
                self.alpaca_provider = AlpacaDataProvider()
                logger.info("Alpaca data provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca provider: {e}")
                self.enable_alpaca = False
        
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
        if self.enable_alpaca and self.alpaca_provider:
            sources.append("Alpaca (real-time)")
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
        
        # 数据源优先级配置 (移除yfinance依赖)
        primary_sources = ['ib', 'qlib', 'openbb']  # 主要数据源
        fallback_sources = ['alpha_vantage', 'quandl']  # 备用数据源 (移除yahoo)
        
        # 按优先级尝试各数据源
        
        # 1. 首先尝试Alpaca (如果启用)
        if self.enable_alpaca and self.alpaca_provider:
            try:
                logger.info(f"Trying Alpaca for {symbol}")
                data = self.alpaca_provider.get_stock_data(symbol, start_date, end_date)
                
                if not data.empty:
                    logger.info(f"Retrieved {len(data)} records for {symbol} from Alpaca")
                    return data
                else:
                    logger.warning(f"No Alpaca data found for {symbol}")
            except Exception as e:
                logger.warning(f"Alpaca data retrieval failed for {symbol}: {e}")
        
        # 2. 尝试Interactive Brokers (实时数据优先)
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
        
        # 3. 回退到Qlib
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

        # 4. 回退到OpenBB
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
        
        # 5. 最后回退到yfinance (如果启用)
        if self.fallback_to_yfinance and YFINANCE_AVAILABLE:
            try:
                logger.info(f"Trying yfinance fallback for {symbol}")
                data = self._get_yfinance_data(symbol, start_date, end_date)
                
                if not data.empty:
                    logger.info(f"Retrieved {len(data)} records for {symbol} from yfinance")
                    return data
                else:
                    logger.warning(f"No yfinance data found for {symbol}")
            except Exception as e:
                logger.warning(f"yfinance data retrieval failed for {symbol}: {e}")
        
        # 6. 如果所有数据源都失败，返回空DataFrame并记录警告
        logger.warning(f"All data sources failed for {symbol}. Consider checking data source configurations.")
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
        elif source.lower() == "alpaca" and self.alpaca_provider:
            return self.alpaca_provider.get_stock_data(symbol, start_date, end_date)
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
        从yfinance获取股票数据，包含请求频率控制
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            股票数据DataFrame
        """
        if not YFINANCE_AVAILABLE or yf is None:
            logger.error("yfinance is not available")
            return pd.DataFrame()
        
        try:
            # 设置默认日期范围
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # 实现请求频率控制，避免API限速
            import time
            current_time = time.time()
            
            # 检查上次请求时间，确保至少间隔1秒
            if hasattr(self, '_last_yfinance_request'):
                time_diff = current_time - self._last_yfinance_request
                if time_diff < 1.0:  # 1秒间隔
                    sleep_time = 1.0 - time_diff
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            
            self._last_yfinance_request = time.time()
            
            logger.info(f"Fetching data from yfinance for {symbol}")
            
            # 获取历史数据，增加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 递增等待时间
                        logger.warning(f"yfinance request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        raise e
            
            if data.empty:
                logger.warning(f"No yfinance data found for {symbol}")
                return pd.DataFrame()
            
            # 标准化列名
            data.columns = [col.lower() for col in data.columns]
            
            # 确保包含必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns in yfinance data for {symbol}: {missing_columns}")
                return pd.DataFrame()
            
            # 移除不需要的列
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
            data = data[[col for col in columns_to_keep if col in data.columns]]
            
            logger.info(f"Successfully retrieved {len(data)} records from yfinance for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {e}")
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
                       enable_alpaca: bool = False,
                       fallback_to_yfinance: bool = True) -> DataAdapter:
    """
    创建数据适配器实例
    
    Args:
        prefer_qlib: 是否优先使用Qlib
        qlib_data_dir: Qlib数据目录
        enable_openbb: 是否启用OpenBB
        enable_ib: 是否启用Interactive Brokers
        enable_alpaca: 是否启用Alpaca
        fallback_to_yfinance: 是否启用yfinance回退
    
    Returns:
        DataAdapter实例
    """
    return DataAdapter(
        prefer_qlib=prefer_qlib,
        qlib_data_dir=qlib_data_dir,
        enable_openbb=enable_openbb,
        enable_ib=enable_ib,
        enable_alpaca=enable_alpaca,
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
    # 测试代码 - 仅用于测试和演示数据适配器功能
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据适配器 - 仅用于测试和演示
    adapter = create_data_adapter()
    
    # 获取数据源信息 - 仅用于测试和演示
    print("数据源信息:")
    info = adapter.get_data_source_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试获取AAPL数据 - 仅用于测试和演示
    print("\n测试获取AAPL数据:")
    aapl_data = adapter.get_stock_data("AAPL", "2020-01-01", "2020-03-01")  # 获取测试数据 - 仅用于测试和演示
    if not aapl_data.empty:
        print(f"获取到 {len(aapl_data)} 条AAPL数据")  # 显示数据条数 - 仅用于测试和演示
        print(aapl_data.head())  # 显示前几行数据 - 仅用于测试和演示
    else:
        print("未获取到AAPL数据")  # 数据获取失败提示 - 仅用于测试和演示
    
    # 检查数据可用性 - 仅用于测试和演示
    print("\n检查AAPL数据可用性:")
    availability = adapter.check_data_availability("AAPL")  # 检查数据源可用性 - 仅用于测试和演示
    for key, value in availability.items():
        print(f"  {key}: {value}")  # 显示可用性信息 - 仅用于测试和演示
    
    # 测试多个股票 - 仅用于测试和演示
    print("\n测试获取多个股票数据:")
    symbols = ["AAPL", "MSFT", "GOOGL"]  # 测试股票列表 - 仅用于测试和演示
    multi_data = adapter.get_multiple_stocks_data(symbols, "2020-01-01", "2020-02-01")  # 批量获取数据 - 仅用于测试和演示
    for symbol, data in multi_data.items():
        print(f"  {symbol}: {len(data)} records")  # 显示每个股票的数据条数 - 仅用于测试和演示