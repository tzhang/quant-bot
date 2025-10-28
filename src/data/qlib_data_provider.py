"""
Qlib数据提供器适配器
集成微软Qlib的数据包，提供高质量的股票数据
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import qlib
    from qlib.data import D
    from qlib.config import REG_US
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logging.warning("Qlib not installed. Please install with: pip install pyqlib")

logger = logging.getLogger(__name__)


class QlibDataProvider:
    """
    Qlib数据提供器
    
    提供统一的接口来访问Qlib预处理的股票数据，
    提供高质量、高性能的金融数据服务
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化Qlib数据提供器
        
        Args:
            data_dir: Qlib数据目录路径，默认使用 ~/.qlib/qlib_data/us_data
        """
        if not QLIB_AVAILABLE:
            raise ImportError("Qlib is not available. Please install with: pip install pyqlib")
        
        self.data_dir = data_dir or os.path.expanduser("~/.qlib/qlib_data/us_data")
        self.initialized = False
        self._initialize_qlib()
    
    def _initialize_qlib(self):
        """初始化Qlib"""
        try:
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Qlib data directory not found: {self.data_dir}")
            
            # 初始化Qlib
            qlib.init(provider_uri=self.data_dir, region=REG_US)
            self.initialized = True
            logger.info(f"Qlib initialized successfully with data from: {self.data_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qlib: {e}")
            raise
    
    def get_stock_data(self, 
                      symbol: str, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            symbol: 股票代码 (如 'AAPL')
            start_date: 开始日期 (YYYY-MM-DD格式)
            end_date: 结束日期 (YYYY-MM-DD格式)
            fields: 需要的字段列表，默认为 ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            包含股票数据的DataFrame
        """
        if not self.initialized:
            raise RuntimeError("Qlib not initialized")
        
        # 输入验证
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.strip()  # 清理空白字符
        
        # 默认字段
        if fields is None:
            fields = ['$open', '$high', '$low', '$close', '$volume']
        else:
            # 确保字段名以$开头（Qlib格式）
            fields = [f'${field}' if not field.startswith('$') else field for field in fields]
        
        # 验证和处理日期
        start_date, end_date = self._validate_and_process_dates(start_date, end_date)
        
        try:
            # 尝试不同的符号格式来获取数据
            symbol_variants = self._generate_symbol_variants(symbol)
            data = None
            used_symbol = None
            
            for variant in symbol_variants:
                try:
                    # 使用Qlib的D.features接口获取数据
                    data = D.features(
                        instruments=[variant],
                        fields=fields,
                        start_time=start_date,
                        end_time=end_date
                    )
                    
                    if data is not None and not data.empty:
                        used_symbol = variant
                        logger.debug(f"Successfully retrieved data using symbol variant: {variant}")
                        break
                        
                except Exception as e:
                    logger.debug(f"Failed to get data with symbol variant {variant}: {e}")
                    continue
            
            if data is None or data.empty:
                logger.warning(f"No data found for symbol: {symbol} (tried variants: {symbol_variants})")
                return pd.DataFrame()
            
            # 处理和验证数据
            df = self._process_multiindex_data(data, symbol, symbol_variants)
            
            if df is None or df.empty:
                logger.warning(f"No data extracted for symbol: {symbol}")
                return pd.DataFrame()
            
            # 数据后处理和验证
            df = self._post_process_data(df)
            
            # 数据质量检查
            df = self._validate_data_quality(df, symbol)
            
            logger.info(f"Retrieved {len(df)} records for {symbol} from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _validate_and_process_dates(self, start_date: Optional[str], end_date: Optional[str]) -> Tuple[str, str]:
        """验证和处理日期参数"""
        # 构建时间范围 - 使用Qlib数据的实际日期范围
        if start_date is None:
            start_date = "2020-01-01"
        if end_date is None:
            end_date = "2020-11-10"  # 使用Qlib数据的实际结束日期
        
        # 验证日期格式
        try:
            pd.to_datetime(start_date)
            pd.to_datetime(end_date)
        except Exception as e:
            logger.error(f"Invalid date format: {e}")
            raise ValueError(f"Invalid date format: {e}")
        
        return start_date, end_date
    
    def _generate_symbol_variants(self, symbol: str) -> List[str]:
        """生成符号变体列表"""
        variants = []
        
        # 基本变体
        variants.extend([symbol.upper(), symbol.lower(), symbol])
        
        # 去除重复
        return list(dict.fromkeys(variants))  # 保持顺序的去重
    
    def _process_multiindex_data(self, data: pd.DataFrame, symbol: str, symbol_variants: List[str]) -> Optional[pd.DataFrame]:
        """处理MultiIndex DataFrame"""
        df = None
        
        if isinstance(data.index, pd.MultiIndex):
            # 检查MultiIndex的层级名称
            level_names = data.index.names
            logger.debug(f"MultiIndex levels: {level_names}")
            
            # 尝试不同的层级名称和符号变体
            instrument_level = None
            for level_name in ['instrument', 'code', 'symbol']:
                if level_name in level_names:
                    instrument_level = level_name
                    break
            
            if instrument_level is not None:
                # 尝试使用不同的符号变体来提取数据
                for variant in symbol_variants:
                    try:
                        df = data.xs(variant, level=instrument_level)
                        logger.debug(f"Successfully extracted data using {variant} at level {instrument_level}")
                        break
                    except KeyError:
                        continue
                
                if df is None:
                    # 如果直接提取失败，尝试查看可用的instruments
                    df = self._find_matching_instrument(data, symbol, instrument_level)
            else:
                logger.error(f"Could not find instrument level in MultiIndex: {level_names}")
                return None
        else:
            df = data.copy()
        
        return df
    
    def _find_matching_instrument(self, data: pd.DataFrame, symbol: str, instrument_level: str) -> Optional[pd.DataFrame]:
        """在可用instruments中查找匹配的符号"""
        try:
            available_instruments = data.index.get_level_values(instrument_level).unique()
            logger.debug(f"Available instruments: {list(available_instruments)[:10]}...")  # 只显示前10个
            
            # 尝试找到匹配的instrument
            matching_instrument = None
            for instrument in available_instruments:
                if str(instrument).upper() == symbol.upper():
                    matching_instrument = instrument
                    break
            
            if matching_instrument is not None:
                df = data.xs(matching_instrument, level=instrument_level)
                logger.debug(f"Found matching instrument: {matching_instrument}")
                return df
            else:
                logger.warning(f"Symbol {symbol} not found in available instruments")
                return None
                
        except Exception as e:
            logger.error(f"Error processing MultiIndex data: {e}")
            return None
    
    def _post_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据后处理"""
        # 重命名列，移除$前缀
        column_mapping = {}
        for col in df.columns:
            if col.startswith('$'):
                new_col = col[1:].lower()  # 移除$并转为小写
                column_mapping[col] = new_col
        
        df = df.rename(columns=column_mapping)
        
        # 确保索引是日期格式
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 排序索引
        df = df.sort_index()
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """数据质量验证和清理"""
        if df.empty:
            return df
        
        original_length = len(df)
        
        # 移除全为NaN的行
        df = df.dropna(how='all')
        
        # 移除价格为负数或零的行（如果有价格列）
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # 移除成交量为负数的行
        if 'volume' in df.columns:
            df = df[df['volume'] >= 0]
        
        # 检查数据完整性
        if len(df) < original_length * 0.5:  # 如果丢失超过50%的数据
            logger.warning(f"Data quality issue for {symbol}: {original_length - len(df)} rows removed")
        
        # 检查是否有异常的价格跳跃（简单检查）
        if 'close' in df.columns and len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes > 0.5  # 50%以上的价格变化
            if extreme_changes.any():
                logger.warning(f"Detected extreme price changes for {symbol}: {extreme_changes.sum()} occurrences")
        
        return df
    
    def get_multiple_stocks_data(self, 
                               symbols: List[str], 
                               start_date: Optional[str] = None, 
                               end_date: Optional[str] = None,
                               fields: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        获取多个股票的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要的字段列表
        
        Returns:
            字典，键为股票代码，值为对应的DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, start_date, end_date, fields)
                if not data.empty:
                    result[symbol] = data
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Error retrieving data for {symbol}: {e}")
        
        return result
    
    def get_available_instruments(self) -> List[str]:
        """
        获取可用的股票代码列表
        
        Returns:
            可用股票代码列表
        """
        if not self.initialized:
            raise RuntimeError("Qlib not initialized")
        
        try:
            # 直接读取instruments文件
            instruments_file = os.path.join(self.data_dir, "instruments", "all.txt")
            if not os.path.exists(instruments_file):
                logger.error(f"Instruments file not found: {instruments_file}")
                return []
            
            instruments = []
            with open(instruments_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        instruments.append(parts[0])
            
            return sorted(instruments)
        except Exception as e:
            logger.error(f"Error retrieving available instruments: {e}")
            return []
    
    def get_trading_calendar(self, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            交易日期列表
        """
        if not self.initialized:
            raise RuntimeError("Qlib not initialized")
        
        try:
            if start_date is None:
                start_date = "2020-01-01"
            if end_date is None:
                end_date = "2020-11-10"  # 使用Qlib数据的实际结束日期
            
            # 获取交易日历
            calendar = D.calendar(start_time=start_date, end_time=end_date)
            return [date.strftime("%Y-%m-%d") for date in calendar]
        except Exception as e:
            logger.error(f"Error retrieving trading calendar: {e}")
            return []
    
    def check_data_availability(self, symbol: str) -> Dict[str, Union[str, int]]:
        """
        检查特定股票的数据可用性
        
        Args:
            symbol: 股票代码
        
        Returns:
            包含数据可用性信息的字典
        """
        try:
            # 获取最近一年的数据来检查可用性
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            data = self.get_stock_data(symbol, start_date, end_date)
            
            if data.empty:
                return {
                    "symbol": symbol,
                    "available": False,
                    "records_count": 0,
                    "date_range": "No data available"
                }
            
            return {
                "symbol": symbol,
                "available": True,
                "records_count": len(data),
                "date_range": f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
                "latest_date": data.index.max().strftime('%Y-%m-%d')
            }
        except Exception as e:
            logger.error(f"Error checking data availability for {symbol}: {e}")
            return {
                "symbol": symbol,
                "available": False,
                "error": str(e)
            }
    
    def get_data_info(self) -> Dict[str, Union[str, int, List[str]]]:
        """
        获取数据源信息
        
        Returns:
            包含数据源信息的字典
        """
        try:
            instruments = self.get_available_instruments()
            calendar = self.get_trading_calendar()
            
            return {
                "data_source": "Qlib (Microsoft)",
                "data_directory": self.data_dir,
                "total_instruments": len(instruments),
                "sample_instruments": instruments[:10] if instruments else [],
                "trading_days_count": len(calendar),
                "date_range": f"{calendar[0]} to {calendar[-1]}" if calendar else "No calendar data",
                "initialized": self.initialized
            }
        except Exception as e:
            logger.error(f"Error getting data info: {e}")
            return {
                "data_source": "Qlib (Microsoft)",
                "error": str(e),
                "initialized": self.initialized
            }


# 便捷函数
def create_qlib_provider(data_dir: Optional[str] = None) -> QlibDataProvider:
    """
    创建Qlib数据提供器实例
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        QlibDataProvider实例
    """
    return QlibDataProvider(data_dir)


def get_qlib_stock_data(symbol: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    便捷函数：获取单个股票的Qlib数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        data_dir: 数据目录路径
    
    Returns:
        股票数据DataFrame
    """
    provider = create_qlib_provider(data_dir)
    return provider.get_stock_data(symbol, start_date, end_date)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建数据提供器
        provider = create_qlib_provider()
        
        # 获取数据源信息
        info = provider.get_data_info()
        print("数据源信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 测试获取AAPL数据
        print("\n测试获取AAPL数据:")
        aapl_data = provider.get_stock_data("AAPL", "2024-01-01", "2024-12-31")
        if not aapl_data.empty:
            print(f"获取到 {len(aapl_data)} 条AAPL数据")
            print(aapl_data.head())
        else:
            print("未获取到AAPL数据")
        
        # 检查数据可用性
        print("\n检查AAPL数据可用性:")
        availability = provider.check_data_availability("AAPL")
        for key, value in availability.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"测试失败: {e}")