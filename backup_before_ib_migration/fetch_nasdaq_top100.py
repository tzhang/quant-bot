#!/usr/bin/env python3
"""
NASDAQ Top 100 股票数据获取脚本
获取NASDAQ 100指数成分股的历史数据并存储到数据库
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.dao import StockDataDAO
from src.database.connection import DatabaseManager
from src.utils.logger import get_logger
from src.data.ib_data_provider import IBDataProvider, create_ib_provider

# 设置日志
logger = get_logger(__name__)

# NASDAQ 100 股票列表 (基于2024年最新数据)
NASDAQ_100_STOCKS = [
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX',
    'PLTR', 'COST', 'ASML', 'AMD', 'CSCO', 'AZN', 'TMUS', 'MU', 'LIN', 'SHOP',
    'APP', 'PEP', 'PDD', 'INTC', 'QCOM', 'LRCX', 'INTC', 'ARM', 'AMAT', 'BKNG',
    'TXN', 'ISRG', 'AMGN', 'PANW', 'GILD', 'ADBE', 'KLAC', 'HON', 'CRWD', 'DASH',
    'ADP', 'ADI', 'CEG', 'CMCSA', 'MELI', 'VRTX', 'CDNS', 'MSTR', 'SBUX', 'SNPS',
    'ORLY', 'CTAS', 'MDLZ', 'MRVL', 'ABNB', 'MAR', 'PYPL', 'TRI', 'MNST', 'CSX',
    'ADSK', 'FTNT', 'WDAY', 'AEP', 'REGN', 'DDOG', 'AXON', 'NXPI', 'ROP', 'FAST',
    'IDXX', 'PCAR', 'EA', 'ZS', 'ROST', 'XEL', 'TTWO', 'BKR', 'EXC', 'PAYX',
    'WBD', 'CPRT', 'FANG', 'CHTR', 'CCEP', 'TEAM', 'MCHP', 'KDP', 'GEHC', 'VRSK',
    'CSGP', 'CTSH', 'ODFL', 'KHC', 'DXCM', 'TTD', 'BIIB', 'ON', 'LULU', 'CDW', 'GFS'
]

class NasdaqDataFetcher:
    """NASDAQ股票数据获取器"""
    
    def __init__(self):
        """初始化数据获取器"""
        self.db_manager = DatabaseManager()
        self.stock_data_dao = StockDataDAO()  # StockDataDAO不需要参数
        self.success_count = 0
        self.error_count = 0
        self.total_records = 0
        
        # 初始化IB数据提供者
        try:
            self.ib_provider = create_ib_provider()
            logger.info("IB数据提供者初始化成功")
        except Exception as e:
            logger.warning(f"IB数据提供者初始化失败: {e}")
            self.ib_provider = None
        
    def fetch_data_from_qlib(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从Qlib获取股票数据"""
        try:
            import qlib
            from qlib.data import D
            
            # 初始化qlib
            qlib.init(provider_uri='~/.qlib/qlib_data/us_data', region='us')
            
            # 获取数据
            data = D.features(
                [symbol], 
                ['$open', '$high', '$low', '$close', '$volume'],
                start_time=start_date,
                end_time=end_date
            )
            
            if data is not None and not data.empty:
                # 重置索引以获取日期列
                data = data.reset_index()
                data.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                data['date'] = pd.to_datetime(data['date'])
                return data
                
        except Exception as e:
            logger.warning(f"Qlib获取 {symbol} 数据失败: {e}")
            
        return None
    
    def fetch_data_from_ib(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从Interactive Brokers获取股票数据"""
        if self.ib_provider is None:
            logger.warning("IB数据提供者未初始化")
            return None
            
        try:
            data = self.ib_provider.get_stock_data(symbol, start_date, end_date)
            
            if not data.empty:
                # 确保数据格式一致
                if 'date' not in data.columns and data.index.name == 'date':
                    data = data.reset_index()
                
                # 统一列名
                column_mapping = {
                    'Date': 'date',
                    'Open': 'open', 'open': 'open',
                    'High': 'high', 'high': 'high', 
                    'Low': 'low', 'low': 'low',
                    'Close': 'close', 'close': 'close',
                    'Volume': 'volume', 'volume': 'volume'
                }
                
                data = data.rename(columns=column_mapping)
                
                # 确保必要的列存在
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                for col in required_columns:
                    if col not in data.columns:
                        if col == 'symbol':
                            data[col] = symbol
                        else:
                            logger.warning(f"缺少列 {col}，使用默认值")
                            data[col] = 0
                
                # 确保日期格式正确
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                
                return data
                
        except Exception as e:
            logger.warning(f"IB获取 {symbol} 数据失败: {e}")
            
        return None
    
    def fetch_data_from_yfinance(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从yfinance获取股票数据"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                # 重置索引以获取日期列
                data = data.reset_index()
                data['symbol'] = symbol
                
                # 统一列名
                column_mapping = {
                    'Date': 'date',
                    'Open': 'open', 'open': 'open',
                    'High': 'high', 'high': 'high', 
                    'Low': 'low', 'low': 'low',
                    'Close': 'close', 'close': 'close',
                    'Volume': 'volume', 'volume': 'volume'
                }
                
                data = data.rename(columns=column_mapping)
                data['date'] = pd.to_datetime(data['date'])
                
                # 选择需要的列
                required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                data = data[required_columns]
                
                return data
                
        except Exception as e:
            logger.warning(f"yfinance获取 {symbol} 数据失败: {e}")
            
        return None
    
    def fetch_data_from_openbb(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从OpenBB获取股票数据"""
        try:
            from openbb import obb
            
            data = obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                provider='yfinance'
            )
            
            if data is not None and hasattr(data, 'results') and data.results:
                df = pd.DataFrame([item.__dict__ for item in data.results])
                df['symbol'] = symbol
                
                # 统一列名
                column_mapping = {
                    'date': 'date',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low', 
                    'close': 'close',
                    'volume': 'volume'
                }
                
                df = df.rename(columns=column_mapping)
                df['date'] = pd.to_datetime(df['date'])
                
                # 选择需要的列
                required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                df = df[required_columns]
                
                return df
                
        except Exception as e:
            logger.warning(f"OpenBB获取 {symbol} 数据失败: {e}")
            
        return None
    
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取单只股票数据，尝试多个数据源"""
        logger.info(f"开始获取 {symbol} 的数据...")
        
        # 尝试不同的数据源
        data_sources = [
            ('ib', self.fetch_data_from_ib),
            ('yfinance', self.fetch_data_from_yfinance),
            ('qlib', self.fetch_data_from_qlib),
            ('openbb', self.fetch_data_from_openbb)
        ]
        
        for source_name, fetch_func in data_sources:
            try:
                data = fetch_func(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    logger.info(f"成功从 {source_name} 获取 {symbol} 数据，共 {len(data)} 条记录")
                    return data
            except Exception as e:
                logger.warning(f"从 {source_name} 获取 {symbol} 数据失败: {e}")
                continue
        
        logger.error(f"所有数据源都无法获取 {symbol} 的数据")
        return None
    
    def store_stock_data(self, data: pd.DataFrame) -> bool:
        """存储股票数据到数据库"""
        try:
            for _, row in data.iterrows():
                try:
                    # 创建股票数据记录
                    self.stock_data_dao.create(
                        symbol=row['symbol'],
                        date=row['date'].date(),
                        open_price=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume']) if pd.notna(row['volume']) else 0
                    )
                    self.total_records += 1
                except Exception as e:
                    logger.error(f"存储数据失败: {e}")
                    continue
            
            return True
            
        except Exception as e:
            logger.error(f"批量存储数据失败: {e}")
            return False
    
    def fetch_all_nasdaq_stocks(self, start_date: str = None, end_date: str = None):
        """获取所有NASDAQ 100股票数据"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"开始获取NASDAQ 100股票数据，时间范围: {start_date} 到 {end_date}")
        logger.info(f"股票总数: {len(NASDAQ_100_STOCKS)}")
        
        for i, symbol in enumerate(NASDAQ_100_STOCKS, 1):
            try:
                logger.info(f"处理第 {i}/{len(NASDAQ_100_STOCKS)} 只股票: {symbol}")
                
                # 获取股票数据
                data = self.fetch_stock_data(symbol, start_date, end_date)
                
                if data is not None and not data.empty:
                    # 存储数据
                    if self.store_stock_data(data):
                        self.success_count += 1
                        logger.info(f"✅ {symbol} 数据获取和存储成功")
                    else:
                        self.error_count += 1
                        logger.error(f"❌ {symbol} 数据存储失败")
                else:
                    self.error_count += 1
                    logger.error(f"❌ {symbol} 数据获取失败")
                    
            except Exception as e:
                self.error_count += 1
                logger.error(f"❌ 处理 {symbol} 时发生错误: {e}")
                continue
        
        # 打印总结
        self.print_summary()
    
    def print_summary(self):
        """打印获取结果摘要"""
        total_stocks = len(NASDAQ_100_STOCKS)
        success_rate = (self.success_count / total_stocks) * 100 if total_stocks > 0 else 0
        
        print("\n" + "="*60)
        print("📊 NASDAQ 100 股票数据获取完成")
        print("="*60)
        print(f"总股票数量: {total_stocks}")
        print(f"成功获取: {self.success_count}")
        print(f"获取失败: {self.error_count}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"总记录数: {self.total_records}")
        print("="*60)
        
        # 验证数据库中的数据
        self.verify_database_data()
    
    def verify_database_data(self):
        """验证数据库中的数据"""
        try:
            print("\n🔍 验证数据库中的股票数据...")
            
            # 获取所有股票代码
            symbols = self.stock_data_dao.get_symbols()
            print(f"数据库中的股票代码: {symbols}")
            
            # 检查每只股票的数据
            for symbol in symbols:
                try:
                    # 获取最新数据
                    latest_data = self.stock_data_dao.get_by_symbol_and_date_range(
                        symbol=symbol,
                        start_date=(datetime.now() - timedelta(days=60)).date(),
                        end_date=datetime.now().date()
                    )
                    
                    if latest_data:
                        latest_record = latest_data[-1]  # 获取最新记录
                        print(f"  {symbol}: {len(latest_data)} 条记录, 最新日期: {latest_record.date}, 收盘价: ${latest_record.close:.2f}")
                    else:
                        print(f"  {symbol}: 查询失败")
                        
                except Exception as e:
                    print(f"  {symbol}: 查询失败 - {e}")
                    
        except Exception as e:
            logger.error(f"验证数据库数据失败: {e}")

def main():
    """主函数"""
    try:
        # 创建数据获取器
        fetcher = NasdaqDataFetcher()
        
        # 设置时间范围（最近5年）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        logger.info(f"开始获取NASDAQ 100股票数据，时间范围: {start_date} 到 {end_date}")
        
        # 获取所有NASDAQ 100股票数据
        fetcher.fetch_all_nasdaq_stocks(start_date, end_date)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()