"""
OpenBB数据提供者

该模块提供基于OpenBB Platform的股票数据获取功能。
OpenBB是一个开源的金融数据平台，支持多种数据源。

作者: AI Assistant
日期: 2024
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Any
import warnings

# 配置日志
logger = logging.getLogger(__name__)

# 检查OpenBB是否可用
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
    logger.info("OpenBB Platform is available")
except ImportError as e:
    OPENBB_AVAILABLE = False
    logger.warning(f"OpenBB Platform is not available: {e}")
    obb = None


class OpenBBDataProvider:
    """
    OpenBB数据提供者类
    
    提供基于OpenBB Platform的股票数据获取功能，
    与项目中其他数据提供者保持一致的接口。
    """
    
    def __init__(self):
        """
        初始化OpenBB数据提供者
        """
        self.is_available = OPENBB_AVAILABLE
        
        if not self.is_available:
            logger.error("OpenBB Platform is not available. Please install it with: pip install openbb")
            return
        
        try:
            # 初始化OpenBB（如果需要的话）
            logger.info("OpenBB数据提供者初始化成功")
        except Exception as e:
            logger.error(f"OpenBB初始化失败: {e}")
            self.is_available = False
    
    def get_stock_data(self, 
                      symbol: str, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      **kwargs) -> pd.DataFrame:
        """
        获取股票历史数据
        
        Args:
            symbol: 股票代码 (如 'AAPL', 'MSFT')
            start_date: 开始日期，格式为 'YYYY-MM-DD'
            end_date: 结束日期，格式为 'YYYY-MM-DD'
            **kwargs: 其他参数
        
        Returns:
            包含股票数据的DataFrame，列包括: open, high, low, close, volume
        """
        if not self.is_available:
            logger.error("OpenBB is not available")
            return pd.DataFrame()
        
        try:
            # 设置默认日期范围
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # 使用OpenBB获取股票数据
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 获取历史股价数据
                data = obb.equity.price.historical(
                    symbol=symbol.upper(),
                    start_date=start_date,
                    end_date=end_date,
                    provider="yfinance"  # 可以根据需要更改数据提供商
                )
            
            if data is None or data.results is None:
                logger.warning(f"No OpenBB data found for {symbol}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = data.to_df()
            
            if df.empty:
                logger.warning(f"Empty OpenBB data for {symbol}")
                return pd.DataFrame()
            
            # 标准化列名（转换为小写）
            df.columns = [col.lower() for col in df.columns]
            
            # 确保包含必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in required_columns if col in df.columns]
            
            if not available_columns:
                logger.error(f"No required columns found in OpenBB data for {symbol}")
                return pd.DataFrame()
            
            # 只保留需要的列
            df = df[available_columns]
            
            # 确保索引是日期类型
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.warning(f"Could not convert index to datetime for {symbol}: {e}")
            
            logger.info(f"Retrieved {len(df)} records for {symbol} from OpenBB")
            return df
            
        except Exception as e:
            logger.error(f"OpenBB data retrieval failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, 
                               symbols: List[str], 
                               start_date: Optional[str] = None, 
                               end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取多个股票的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            字典，键为股票代码，值为DataFrame
        """
        result = {}
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, start_date, end_date)
                if not data.empty:
                    result[symbol] = data
                else:
                    logger.warning(f"No OpenBB data retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Error retrieving OpenBB data for {symbol}: {e}")
        
        return result
    
    def get_available_instruments(self) -> List[str]:
        """
        获取可用的股票代码列表
        
        注意：OpenBB没有直接的方法获取所有可用股票列表，
        这里返回一些常见的美股代码作为示例。
        
        Returns:
            股票代码列表
        """
        # 返回一些常见的美股代码
        common_stocks = [
            # 科技股
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", 
            "NFLX", "ADBE", "CRM", "ORCL", "IBM", "INTC", "AMD", "QCOM", "AVGO",
            
            # 金融股
            "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA", "PYPL",
            
            # 消费股
            "JNJ", "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "DIS",
            
            # 工业股
            "BA", "CAT", "GE", "MMM", "UPS", "FDX", "LMT", "RTX", "HON", "UNP",
            
            # 能源股
            "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "KMI", "OKE"
        ]
        
        return common_stocks
    
    def check_data_availability(self, symbol: str) -> Dict[str, Union[str, bool, int]]:
        """
        检查指定股票的数据可用性
        
        Args:
            symbol: 股票代码
        
        Returns:
            数据可用性信息字典
        """
        result = {
            "symbol": symbol.upper(),
            "available": False,
            "records_count": 0,
            "date_range": "",
            "provider": "openbb"
        }
        
        if not self.is_available:
            return result
        
        try:
            # 获取最近一周的数据来测试可用性
            test_end_date = datetime.now().strftime("%Y-%m-%d")
            test_start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            test_data = self.get_stock_data(symbol, test_start_date, test_end_date)
            
            if not test_data.empty:
                result["available"] = True
                result["records_count"] = len(test_data)
                
                # 获取更长时间范围的数据来确定日期范围
                long_start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                long_data = self.get_stock_data(symbol, long_start_date, test_end_date)
                
                if not long_data.empty:
                    start_date = long_data.index.min().strftime("%Y-%m-%d")
                    end_date = long_data.index.max().strftime("%Y-%m-%d")
                    result["date_range"] = f"{start_date} to {end_date}"
                    result["records_count"] = len(long_data)
            
        except Exception as e:
            logger.error(f"Error checking OpenBB data availability for {symbol}: {e}")
        
        return result
    
    def get_data_info(self) -> Dict[str, Union[str, int, bool]]:
        """
        获取OpenBB数据源的基本信息
        
        Returns:
            数据源信息字典
        """
        info = {
            "provider": "OpenBB Platform",
            "available": self.is_available,
            "description": "OpenBB Platform - Open source financial data platform",
            "supported_markets": ["US", "International"],
            "data_types": ["stocks", "options", "crypto", "forex", "economics"],
            "real_time": False,
            "historical": True
        }
        
        if self.is_available:
            try:
                # 获取一些基本统计信息
                sample_symbols = ["AAPL", "MSFT", "GOOGL"]
                available_count = 0
                
                for symbol in sample_symbols:
                    availability = self.check_data_availability(symbol)
                    if availability["available"]:
                        available_count += 1
                
                info["sample_availability"] = f"{available_count}/{len(sample_symbols)} symbols available"
                
            except Exception as e:
                logger.error(f"Error getting OpenBB data info: {e}")
                info["error"] = str(e)
        
        return info


# 便捷函数
def create_openbb_provider() -> OpenBBDataProvider:
    """
    创建OpenBB数据提供者实例
    
    Returns:
        OpenBBDataProvider实例
    """
    return OpenBBDataProvider()


def get_openbb_stock_data(symbol: str, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """
    便捷函数：使用OpenBB获取股票数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        股票数据DataFrame
    """
    provider = create_openbb_provider()
    return provider.get_stock_data(symbol, start_date, end_date)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== OpenBB数据提供者测试 ===")
    
    # 创建提供者
    provider = create_openbb_provider()
    
    # 获取数据源信息
    print("\n1. 数据源信息:")
    info = provider.get_data_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    if provider.is_available:
        # 测试单个股票数据获取
        print("\n2. 测试获取AAPL数据:")
        aapl_data = provider.get_stock_data("AAPL", "2024-01-01", "2024-01-10")
        if not aapl_data.empty:
            print(f"   获取到 {len(aapl_data)} 条记录")
            print(f"   列: {list(aapl_data.columns)}")
            print(f"   日期范围: {aapl_data.index.min()} 到 {aapl_data.index.max()}")
            print("   前3行数据:")
            print(aapl_data.head(3))
        else:
            print("   未获取到数据")
        
        # 测试数据可用性检查
        print("\n3. 检查AAPL数据可用性:")
        availability = provider.check_data_availability("AAPL")
        for key, value in availability.items():
            print(f"   {key}: {value}")
        
        # 测试多个股票
        print("\n4. 测试获取多个股票数据:")
        symbols = ["AAPL", "MSFT", "GOOGL"]
        multi_data = provider.get_multiple_stocks_data(symbols, "2024-01-01", "2024-01-05")
        for symbol, data in multi_data.items():
            print(f"   {symbol}: {len(data)} 条记录")
        
        # 测试可用股票列表
        print("\n5. 可用股票代码示例:")
        instruments = provider.get_available_instruments()
        print(f"   总共 {len(instruments)} 个股票代码")
        print(f"   前10个: {instruments[:10]}")
    
    else:
        print("\nOpenBB不可用，请安装: pip install openbb")