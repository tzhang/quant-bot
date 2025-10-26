"""
Alpaca 数据提供者

该模块提供基于Alpaca Markets API的股票数据获取功能。
支持历史数据获取，与项目中其他数据提供者保持一致的接口。

作者: AI Assistant
日期: 2024
"""

import logging
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Any
from dataclasses import dataclass
import os

# 配置日志
logger = logging.getLogger(__name__)

# 检查Alpaca API是否可用
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
    logger.info("Alpaca Trade API is available")
except ImportError as e:
    ALPACA_AVAILABLE = False
    logger.warning(f"Alpaca Trade API is not available: {e}")
    tradeapi = None

@dataclass
class AlpacaConfig:
    """Alpaca配置类"""
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        return cls(
            api_key=os.getenv('ALPACA_API_KEY', ''),
            secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )
    
    @classmethod
    def from_config_file(cls, config_path: str = "trading_config.yaml"):
        """从配置文件创建配置"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            alpaca_config = config.get('data_sources', {}).get('api_keys', {}).get('alpaca', {})
            return cls(
                api_key=alpaca_config.get('api_key', ''),
                secret_key=alpaca_config.get('secret_key', ''),
                base_url=alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')
            )
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls.from_env()

class AlpacaDataProvider:
    """
    Alpaca 数据提供者类
    
    提供基于Alpaca Markets API的股票数据获取功能，
    与项目中其他数据提供者保持一致的接口。
    """
    
    def __init__(self, config: AlpacaConfig = None):
        """
        初始化Alpaca数据提供者
        
        Args:
            config: Alpaca配置，如果为None则从配置文件或环境变量加载
        """
        self.config = config or AlpacaConfig.from_config_file()
        self.is_available = ALPACA_AVAILABLE and bool(self.config.api_key and self.config.secret_key)
        self.api = None
        
        if not ALPACA_AVAILABLE:
            logger.error("Alpaca Trade API is not available. Please install it with: pip install alpaca-trade-api")
            return
        
        if not self.config.api_key or not self.config.secret_key:
            logger.error("Alpaca API credentials not configured")
            self.is_available = False
            return
        
        try:
            self.api = tradeapi.REST(
                key_id=self.config.api_key,
                secret_key=self.config.secret_key,
                base_url=self.config.base_url,
                api_version='v2'
            )
            logger.info("Alpaca数据提供者初始化成功")
        except Exception as e:
            logger.error(f"Alpaca数据提供者初始化失败: {e}")
            self.is_available = False
    
    def get_stock_data(self, 
                      symbol: str, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        if not self.is_available:
            logger.error("Alpaca数据提供者不可用")
            return pd.DataFrame()
        
        try:
            # 标准化股票代码
            symbol = symbol.upper().strip()
            
            # 设置默认日期
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
                start_date = start_dt.strftime("%Y-%m-%d")
            
            logger.info(f"获取Alpaca数据: {symbol}, {start_date} 到 {end_date}")
            
            # 获取历史数据
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment='raw'
            ).df
            
            if bars.empty:
                logger.warning(f"未获取到Alpaca数据: {symbol}")
                return pd.DataFrame()
            
            # 标准化列名（转换为小写）
            bars.columns = [col.lower() for col in bars.columns]
            
            # 确保包含必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in required_columns if col in bars.columns]
            
            if not available_columns:
                logger.error(f"Alpaca数据缺少必要列: {symbol}")
                return pd.DataFrame()
            
            # 只保留需要的列
            data = bars[available_columns].copy()
            
            # 确保索引是日期类型
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception as e:
                    logger.warning(f"无法转换索引为日期时间: {e}")
            
            # 添加symbol列以保持一致性
            data['symbol'] = symbol.upper()
            
            logger.info(f"成功获取Alpaca数据: {symbol}, 数据量: {len(data)}")
            return data
            
        except Exception as e:
            logger.error(f"获取Alpaca股票数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, 
                               symbols: List[str], 
                               start_date: Optional[str] = None, 
                               end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个股票的历史数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            字典，键为股票代码，值为对应的DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, start_date, end_date)
                result[symbol] = data
                # 添加延迟以避免API限制
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"获取股票数据失败 {symbol}: {e}")
                result[symbol] = pd.DataFrame()
        
        return result
    
    def check_data_availability(self, symbol: str) -> Dict[str, Union[str, bool, int]]:
        """
        检查指定股票的数据可用性
        
        Args:
            symbol: 股票代码
        
        Returns:
            包含可用性信息的字典
        """
        if not self.is_available:
            return {
                "symbol": symbol,
                "available": False,
                "provider": "Alpaca",
                "error": "Alpaca API不可用"
            }
        
        try:
            # 尝试获取最近一天的数据来检查可用性
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            data = self.get_stock_data(symbol, start_date, end_date)
            
            return {
                "symbol": symbol,
                "available": not data.empty,
                "provider": "Alpaca",
                "records_count": len(data) if not data.empty else 0,
                "date_range": f"{data.index.min()} to {data.index.max()}" if not data.empty else "N/A",
                "columns": list(data.columns) if not data.empty else []
            }
            
        except Exception as e:
            return {
                "symbol": symbol,
                "available": False,
                "provider": "Alpaca",
                "error": str(e)
            }
    
    def is_data_available(self, symbol: str) -> bool:
        """
        简单检查数据是否可用
        
        Args:
            symbol: 股票代码
        
        Returns:
            布尔值，表示数据是否可用
        """
        availability = self.check_data_availability(symbol)
        return availability.get("available", False)
    
    def get_available_symbols(self) -> List[str]:
        """
        获取可用的股票代码列表
        
        Returns:
            股票代码列表
        """
        if not self.is_available:
            return []
        
        try:
            # Alpaca主要支持美股，返回一些常见的股票代码
            # 实际应用中可以通过API获取完整列表
            common_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND'
            ]
            return common_symbols
            
        except Exception as e:
            logger.error(f"获取可用股票代码失败: {e}")
            return []
    
    def get_data_info(self) -> Dict[str, Union[str, int, bool]]:
        """
        获取数据源信息
        
        Returns:
            包含数据源信息的字典
        """
        return {
            "provider": "Alpaca Markets",
            "version": "v2",
            "available": self.is_available,
            "api_key_configured": bool(self.config.api_key),
            "base_url": self.config.base_url,
            "supported_data_types": ["stocks", "historical_data", "real_time_data"],
            "update_frequency": "daily",
            "data_delay": "real-time (market hours)"
        }


def create_alpaca_provider(config: AlpacaConfig = None) -> AlpacaDataProvider:
    """
    创建Alpaca数据提供者实例
    
    Args:
        config: Alpaca配置
    
    Returns:
        AlpacaDataProvider实例
    """
    return AlpacaDataProvider(config)


def get_alpaca_stock_data(symbol: str, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None,
                         config: AlpacaConfig = None) -> pd.DataFrame:
    """
    便捷函数：获取Alpaca股票数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        config: Alpaca配置
    
    Returns:
        股票数据DataFrame
    """
    provider = create_alpaca_provider(config)
    return provider.get_stock_data(symbol, start_date, end_date)


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Alpaca 数据提供者测试 ===")
    
    # 创建提供者
    provider = create_alpaca_provider()
    
    print("\n1. 数据源信息:")
    info = provider.get_data_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    if provider.is_available:
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
        
        print("\n3. 检查AAPL数据可用性:")
        availability = provider.check_data_availability("AAPL")
        for key, value in availability.items():
            print(f"   {key}: {value}")
        
        print("\n4. 测试获取多个股票数据:")
        symbols = ["AAPL", "MSFT", "SPY"]
        multi_data = provider.get_multiple_stocks_data(symbols, "2024-01-01", "2024-01-05")
        for symbol, data in multi_data.items():
            print(f"   {symbol}: {len(data)} 条记录")
    else:
        print("\nAlpaca API不可用，请检查:")
        print("1. 是否安装了 alpaca-trade-api: pip install alpaca-trade-api")
        print("2. 是否配置了API密钥")
        print("3. 检查 trading_config.yaml 文件中的Alpaca配置")