"""
Interactive Brokers 数据提供者

该模块提供基于Interactive Brokers TWS API的股票数据获取功能。
支持历史数据获取，与项目中其他数据提供者保持一致的接口。

作者: AI Assistant
日期: 2024
"""

import logging
import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Any
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger(__name__)

# 检查IB API是否可用
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.common import TickerId, BarData
    IB_AVAILABLE = True
    logger.info("Interactive Brokers API is available")
except ImportError as e:
    IB_AVAILABLE = False
    logger.warning(f"Interactive Brokers API is not available: {e}")
    EClient = None
    EWrapper = None


@dataclass
class IBConfig:
    """IB配置类"""
    host: str = "127.0.0.1"
    port: int = 7497  # 模拟交易端口
    client_id: int = 1
    timeout: int = 30


class IBDataClient(EWrapper, EClient):
    """IB数据客户端"""
    
    def __init__(self, config: IBConfig):
        if IB_AVAILABLE:
            EClient.__init__(self, self)
        
        self.config = config
        self.connected = False
        self.historical_data: Dict[int, List[BarData]] = {}
        self.data_ready: Dict[int, bool] = {}
        self._req_id_counter = 1000
        
    def get_next_req_id(self) -> int:
        """获取下一个请求ID"""
        self._req_id_counter += 1
        return self._req_id_counter
    
    def connect_to_ib(self) -> bool:
        """连接到IB TWS"""
        if not IB_AVAILABLE:
            return False
            
        try:
            logger.info(f"正在连接到 IB TWS: {self.config.host}:{self.config.port}")
            self.connect(self.config.host, self.config.port, self.config.client_id)
            
            # 启动消息循环线程
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()
            
            # 等待连接建立
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < self.config.timeout:
                time.sleep(0.1)
                
            if self.connected:
                logger.info("✅ IB API 连接成功")
                return True
            else:
                logger.error("❌ IB API 连接超时")
                return False
                
        except Exception as e:
            logger.error(f"❌ IB API 连接失败: {e}")
            return False
    
    def disconnect_from_ib(self):
        """断开IB连接"""
        if IB_AVAILABLE and self.isConnected():
            self.disconnect()
            self.connected = False
            logger.info("🔌 IB API 连接已断开")
    
    # ========== IB API 回调函数 ==========
    
    def nextValidId(self, orderId: int):
        """接收下一个有效订单ID"""
        self.connected = True
        logger.info(f"✅ IB 连接成功，下一个订单ID: {orderId}")
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """错误处理"""
        # 过滤信息性消息
        if errorCode in [2104, 2106, 2158, 2119]:  # 连接状态信息
            logger.debug(f"IB 信息: {errorString}")
        elif errorCode == 162:  # 历史数据请求错误
            logger.warning(f"历史数据请求错误: {errorString}")
            if reqId in self.data_ready:
                self.data_ready[reqId] = True
        else:
            logger.error(f"IB 错误 [{errorCode}]: {errorString}")
    
    def historicalData(self, reqId: int, bar: BarData):
        """接收历史数据"""
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append(bar)
    
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """历史数据接收完成"""
        self.data_ready[reqId] = True
        logger.info(f"历史数据接收完成，请求ID: {reqId}, 数据量: {len(self.historical_data.get(reqId, []))}")
    
    def request_historical_data(self, symbol: str, start_date: str, end_date: str, 
                              bar_size: str = "1 day") -> Optional[pd.DataFrame]:
        """请求历史数据"""
        if not self.connected:
            logger.error("未连接到IB TWS")
            return None
        
        try:
            # 创建合约
            contract = Contract()
            contract.symbol = symbol.upper()
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # 计算持续时间
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            duration_days = (end_dt - start_dt).days
            
            # 设置持续时间字符串
            if duration_days <= 30:
                duration_str = f"{duration_days} D"
            elif duration_days <= 365:
                duration_str = f"{duration_days // 7} W"
            else:
                duration_str = f"{duration_days // 365} Y"
            
            # 请求历史数据
            req_id = self.get_next_req_id()
            self.data_ready[req_id] = False
            self.historical_data[req_id] = []
            
            logger.info(f"请求历史数据: {symbol}, 持续时间: {duration_str}, 结束日期: {end_date}")
            
            self.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime=end_date + " 23:59:59",
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=1,  # 只使用常规交易时间
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            
            # 等待数据接收完成
            start_time = time.time()
            while not self.data_ready.get(req_id, False) and (time.time() - start_time) < 30:
                time.sleep(0.1)
            
            if not self.data_ready.get(req_id, False):
                logger.error(f"历史数据请求超时: {symbol}")
                return None
            
            # 转换为DataFrame
            bars = self.historical_data.get(req_id, [])
            if not bars:
                logger.warning(f"未获取到历史数据: {symbol}")
                return pd.DataFrame()
            
            data = []
            for bar in bars:
                data.append({
                    'date': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 清理数据
            del self.historical_data[req_id]
            del self.data_ready[req_id]
            
            logger.info(f"成功获取历史数据: {symbol}, 数据量: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"请求历史数据失败 {symbol}: {e}")
            return None


class IBDataProvider:
    """
    Interactive Brokers 数据提供者类
    
    提供基于IB TWS API的股票数据获取功能，
    与项目中其他数据提供者保持一致的接口。
    """
    
    def __init__(self, config: IBConfig = None):
        """
        初始化IB数据提供者
        
        Args:
            config: IB配置，如果为None则使用默认配置
        """
        self.config = config or IBConfig()
        self.is_available = IB_AVAILABLE
        self.client: Optional[IBDataClient] = None
        
        if not self.is_available:
            logger.error("Interactive Brokers API is not available. Please install it with: pip install ibapi")
            return
        
        try:
            self.client = IBDataClient(self.config)
            logger.info("IB数据提供者初始化成功")
        except Exception as e:
            logger.error(f"IB数据提供者初始化失败: {e}")
            self.is_available = False
    
    def _ensure_connection(self) -> bool:
        """确保连接到IB"""
        if not self.is_available or not self.client:
            return False
        
        if not self.client.connected:
            return self.client.connect_to_ib()
        
        return True
    
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
            logger.error("IB API is not available")
            return pd.DataFrame()
        
        if not self._ensure_connection():
            logger.error("无法连接到IB TWS")
            return pd.DataFrame()
        
        try:
            # 设置默认日期范围
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # 获取历史数据
            data = self.client.request_historical_data(symbol, start_date, end_date)
            
            if data is None or data.empty:
                logger.warning(f"No IB data found for {symbol}")
                return pd.DataFrame()
            
            # 添加symbol列以保持一致性
            data['symbol'] = symbol.upper()
            
            logger.info(f"成功获取IB数据: {symbol}, 数据量: {len(data)}")
            return data
            
        except Exception as e:
            logger.error(f"获取IB股票数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, 
                                symbols: List[str], 
                                start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取多个股票的历史数据
        
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
                if not data.empty:
                    result[symbol] = data
                else:
                    logger.warning(f"未获取到数据: {symbol}")
                    result[symbol] = pd.DataFrame()
                
                # 添加延迟以避免请求过于频繁
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"获取股票数据失败 {symbol}: {e}")
                result[symbol] = pd.DataFrame()
        
        return result
    
    def get_available_instruments(self) -> List[str]:
        """
        获取可用的股票代码列表
        
        Returns:
            股票代码列表
        """
        # IB API不提供直接的股票列表查询，返回常见股票代码
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "BABA", "V", "JNJ", "WMT", "JPM", "PG", "UNH", "HD", "MA", "DIS",
            "PYPL", "ADBE", "CRM", "NFLX", "CMCSA", "PEP", "ABT", "COST",
            "TMO", "AVGO", "ACN", "NKE", "MRK", "TXN", "LLY", "QCOM", "NEE"
        ]
    
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
                "available": False,
                "source": "Interactive Brokers",
                "error": "IB API not available"
            }
        
        # 尝试获取少量数据来测试可用性
        try:
            test_data = self.get_stock_data(symbol, 
                                          (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                                          datetime.now().strftime("%Y-%m-%d"))
            
            if not test_data.empty:
                return {
                    "available": True,
                    "source": "Interactive Brokers",
                    "records_count": len(test_data),
                    "date_range": f"{test_data.index.min()} to {test_data.index.max()}"
                }
            else:
                return {
                    "available": False,
                    "source": "Interactive Brokers",
                    "error": "No data available"
                }
        except Exception as e:
            return {
                "available": False,
                "source": "Interactive Brokers",
                "error": str(e)
            }
    
    def get_data_info(self) -> Dict[str, Union[str, int, bool]]:
        """
        获取数据源信息
        
        Returns:
            包含数据源信息的字典
        """
        return {
            "provider": "Interactive Brokers",
            "version": "TWS API",
            "available": self.is_available,
            "connection_status": self.client.connected if self.client else False,
            "host": self.config.host,
            "port": self.config.port,
            "client_id": self.config.client_id,
            "supported_data_types": ["stocks", "historical_data"],
            "update_frequency": "real-time",
            "data_delay": "real-time (with subscription)"
        }
    
    def disconnect(self):
        """断开连接"""
        if self.client:
            self.client.disconnect_from_ib()


def create_ib_provider(config: IBConfig = None) -> IBDataProvider:
    """
    创建IB数据提供者实例
    
    Args:
        config: IB配置
    
    Returns:
        IBDataProvider实例
    """
    return IBDataProvider(config)


def get_ib_stock_data(symbol: str, 
                     start_date: Optional[str] = None, 
                     end_date: Optional[str] = None,
                     config: IBConfig = None) -> pd.DataFrame:
    """
    便捷函数：获取IB股票数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        config: IB配置
    
    Returns:
        股票数据DataFrame
    """
    provider = create_ib_provider(config)
    return provider.get_stock_data(symbol, start_date, end_date)


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Interactive Brokers 数据提供者测试 ===")
    
    # 创建提供者
    provider = create_ib_provider()
    
    print("\n1. 数据源信息:")
    info = provider.get_data_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    if provider.is_available:
        print("\n注意: 请确保IB TWS或Gateway已启动并配置API访问")
        
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
        
        # 断开连接
        provider.disconnect()
    else:
        print("\nIB API不可用，请安装: pip install ibapi")
        print("并确保IB TWS或Gateway已启动")