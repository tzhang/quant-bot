#!/usr/bin/env python3
"""
使用Interactive Brokers TWS API获取NASDAQ全量股票的近5年历史数据
替代yfinance，提供更稳定的数据获取服务

作者: AI Assistant
日期: 2024
"""

import ftplib
import io
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
import time
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.dao import StockDataDAO
from src.database.models import StockData
from src.data.ib_data_provider import IBDataProvider, IBConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nasdaq_all_stocks_ib.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NASDAQAllStocksFetcherIB:
    """使用IB TWS API获取NASDAQ全量股票数据"""
    
    def __init__(self, ib_config: IBConfig = None):
        self.dao = StockDataDAO()
        self.ftp_host = 'ftp.nasdaqtrader.com'
        self.ftp_directory = 'symboldirectory'
        
        # 初始化IB数据提供者
        self.ib_config = ib_config or IBConfig()
        self.ib_provider = IBDataProvider(self.ib_config)
        
        # 统计信息
        self.stats = {
            'total_symbols': 0,
            'processed_symbols': 0,
            'successful_symbols': 0,
            'failed_symbols': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None
        }
        
    def fetch_nasdaq_symbols(self) -> Set[str]:
        """从NASDAQ FTP服务器获取所有股票代码"""
        symbols = set()
        
        try:
            with ftplib.FTP(self.ftp_host) as ftp:
                ftp.login()  # 匿名登录
                ftp.cwd(self.ftp_directory)
                
                # 获取NASDAQ上市股票
                nasdaq_data = io.BytesIO()
                ftp.retrbinary('RETR nasdaqlisted.txt', nasdaq_data.write)
                nasdaq_content = nasdaq_data.getvalue().decode('utf-8')
                
                # 获取其他交易所上市股票
                other_data = io.BytesIO()
                ftp.retrbinary('RETR otherlisted.txt', other_data.write)
                other_content = other_data.getvalue().decode('utf-8')
                
                # 解析NASDAQ上市股票
                nasdaq_symbols = self._parse_nasdaq_listed(nasdaq_content)
                logger.info(f"从nasdaqlisted.txt获取到 {len(nasdaq_symbols)} 只股票")
                
                # 解析其他交易所股票
                other_symbols = self._parse_other_listed(other_content)
                logger.info(f"从otherlisted.txt获取到 {len(other_symbols)} 只股票")
                
                symbols.update(nasdaq_symbols)
                symbols.update(other_symbols)
                
        except Exception as e:
            logger.error(f"从NASDAQ FTP服务器获取股票列表失败: {e}")
            raise
            
        logger.info(f"总共获取到 {len(symbols)} 只股票代码")
        return symbols
    
    def _parse_nasdaq_listed(self, content: str) -> Set[str]:
        """解析nasdaqlisted.txt文件"""
        symbols = set()
        lines = content.strip().split('\n')
        
        # 跳过头部和尾部
        for line in lines[1:-1]:  # 跳过头部和文件创建时间行
            if line.strip() and '|' in line:
                parts = line.split('|')
                if len(parts) >= 8:
                    symbol = parts[0].strip()
                    test_issue = parts[3].strip()  # 测试股票标识
                    financial_status = parts[4].strip()  # 财务状态
                    etf = parts[6].strip()  # ETF标识
                    
                    # 过滤条件：
                    # 1. 不是测试股票 (Test Issue != 'Y')
                    # 2. 财务状态正常 (Financial Status == 'N')
                    # 3. 不是ETF (ETF != 'Y')
                    # 4. 只包含字母的股票代码
                    if (test_issue != 'Y' and 
                        financial_status == 'N' and 
                        etf != 'Y' and 
                        symbol.isalpha() and 
                        len(symbol) <= 5):
                        symbols.add(symbol)
                        
        return symbols
    
    def _parse_other_listed(self, content: str) -> Set[str]:
        """解析otherlisted.txt文件"""
        symbols = set()
        lines = content.strip().split('\n')
        
        # 跳过头部和尾部
        for line in lines[1:-1]:  # 跳过头部和文件创建时间行
            if line.strip() and '|' in line:
                parts = line.split('|')
                if len(parts) >= 8:
                    symbol = parts[0].strip()
                    test_issue = parts[4].strip()  # 测试股票标识
                    etf = parts[5].strip()  # ETF标识
                    
                    # 过滤条件：
                    # 1. 不是测试股票 (Test Issue != 'Y')
                    # 2. 不是ETF (ETF != 'Y')
                    # 3. 只包含字母的股票代码
                    if (test_issue != 'Y' and 
                        etf != 'Y' and 
                        symbol.isalpha() and 
                        len(symbol) <= 5):
                        symbols.add(symbol)
                        
        return symbols
    
    def fetch_stock_data_ib(self, symbol: str, start_date: datetime, end_date: datetime) -> List[StockData]:
        """使用IB API获取单只股票的历史数据"""
        try:
            # 格式化日期
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"正在获取 {symbol} 的数据 ({start_str} 到 {end_str})")
            
            # 使用IB API获取数据
            df = self.ib_provider.get_stock_data(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str
            )
            
            if df.empty:
                logger.warning(f"未获取到 {symbol} 的数据")
                return []
            
            # 转换为StockData对象列表
            stock_data_list = []
            for date, row in df.iterrows():
                try:
                    stock_data = StockData(
                        symbol=symbol,
                        date=date.date() if hasattr(date, 'date') else date,
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=int(row['Volume']) if pd.notna(row['Volume']) else 0
                    )
                    stock_data_list.append(stock_data)
                except Exception as e:
                    logger.warning(f"转换 {symbol} 在 {date} 的数据时出错: {e}")
                    continue
            
            logger.info(f"✅ {symbol}: 获取到 {len(stock_data_list)} 条记录")
            return stock_data_list
            
        except Exception as e:
            logger.error(f"❌ 获取 {symbol} 数据失败: {e}")
            return []
    
    def save_stock_data(self, stock_data_list: List[StockData]) -> int:
        """批量保存股票数据到数据库"""
        if not stock_data_list:
            return 0
            
        try:
            # 使用批量创建方法
            created_count = self.dao.batch_create(stock_data_list)
            logger.info(f"✅ 成功保存 {created_count} 条记录到数据库")
            return created_count
        except Exception as e:
            logger.error(f"❌ 保存数据到数据库失败: {e}")
            return 0
    
    def process_single_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """处理单个股票符号"""
        result = {
            'symbol': symbol,
            'success': False,
            'records_count': 0,
            'error': None
        }
        
        try:
            # 检查数据库中是否已存在该股票的数据
            existing_data = self.dao.get_by_symbol_and_date_range(symbol, start_date.date(), end_date.date())
            if existing_data:
                logger.info(f"⏭️  {symbol}: 数据库中已存在 {len(existing_data)} 条记录，跳过")
                result['success'] = True
                result['records_count'] = len(existing_data)
                return result
            
            # 获取股票数据
            stock_data_list = self.fetch_stock_data_ib(symbol, start_date, end_date)
            
            if stock_data_list:
                # 保存到数据库
                saved_count = self.save_stock_data(stock_data_list)
                result['success'] = saved_count > 0
                result['records_count'] = saved_count
                
                # 添加延迟以避免API限制
                time.sleep(0.5)  # IB API通常有更严格的限制
            else:
                result['error'] = "未获取到数据"
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ 处理 {symbol} 时出错: {e}")
        
        return result
    
    def run(self, max_workers: int = 1, test_mode: bool = False, test_symbols: int = 10):
        """
        运行数据获取任务
        
        Args:
            max_workers: 最大并发数（IB API建议使用1）
            test_mode: 测试模式，只处理少量股票
            test_symbols: 测试模式下处理的股票数量
        """
        self.stats['start_time'] = datetime.now()
        
        try:
            # 检查IB连接
            if not self.ib_provider.is_available:
                logger.error("❌ IB TWS API不可用，请检查：")
                logger.error("   1. 是否已安装 ibapi: pip install ibapi")
                logger.error("   2. IB TWS或Gateway是否已启动")
                logger.error("   3. API设置是否正确配置")
                return
            
            logger.info("🚀 开始获取NASDAQ全量股票数据 (使用IB TWS API)")
            
            # 获取股票列表
            logger.info("📋 正在获取股票列表...")
            symbols = self.fetch_nasdaq_symbols()
            symbols_list = sorted(list(symbols))
            
            if test_mode:
                symbols_list = symbols_list[:test_symbols]
                logger.info(f"🧪 测试模式：只处理前 {len(symbols_list)} 只股票")
            
            self.stats['total_symbols'] = len(symbols_list)
            
            # 设置日期范围（近5年）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)
            
            logger.info(f"📅 数据时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"📊 待处理股票数量: {len(symbols_list)}")
            
            # 使用线程池处理（IB API建议单线程）
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_symbol = {
                    executor.submit(self.process_single_symbol, symbol, start_date, end_date): symbol
                    for symbol in symbols_list
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        self.stats['processed_symbols'] += 1
                        
                        if result['success']:
                            self.stats['successful_symbols'] += 1
                            self.stats['total_records'] += result['records_count']
                        else:
                            self.stats['failed_symbols'] += 1
                        
                        # 每处理10只股票输出一次进度
                        if self.stats['processed_symbols'] % 10 == 0:
                            progress = (self.stats['processed_symbols'] / self.stats['total_symbols']) * 100
                            logger.info(f"📈 进度: {self.stats['processed_symbols']}/{self.stats['total_symbols']} "
                                      f"({progress:.1f}%) - 成功: {self.stats['successful_symbols']}, "
                                      f"失败: {self.stats['failed_symbols']}")
                        
                    except Exception as e:
                        logger.error(f"❌ 处理 {symbol} 的future时出错: {e}")
                        self.stats['failed_symbols'] += 1
            
        except Exception as e:
            logger.error(f"❌ 运行过程中出现错误: {e}")
        finally:
            self.stats['end_time'] = datetime.now()
            self._print_final_stats()
            
            # 断开IB连接
            try:
                self.ib_provider.disconnect()
                logger.info("🔌 已断开IB连接")
            except Exception as e:
                logger.warning(f"断开IB连接时出错: {e}")
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("=" * 60)
        logger.info("📊 最终统计信息")
        logger.info("=" * 60)
        logger.info(f"总股票数量: {self.stats['total_symbols']}")
        logger.info(f"已处理股票: {self.stats['processed_symbols']}")
        logger.info(f"成功股票数: {self.stats['successful_symbols']}")
        logger.info(f"失败股票数: {self.stats['failed_symbols']}")
        logger.info(f"总记录数量: {self.stats['total_records']}")
        logger.info(f"运行时长: {duration}")
        logger.info(f"成功率: {(self.stats['successful_symbols']/max(self.stats['processed_symbols'], 1)*100):.1f}%")
        logger.info("=" * 60)

def main():
    """主函数"""
    # 创建IB配置
    ib_config = IBConfig(
        host="127.0.0.1",
        port=7497,  # 模拟交易端口，实盘使用7496
        timeout=30
    )
    
    # 创建获取器实例
    fetcher = NASDAQAllStocksFetcherIB(ib_config)
    
    # 运行数据获取（全量模式）
    fetcher.run(max_workers=1, test_mode=False, test_symbols=10)

if __name__ == "__main__":
    main()