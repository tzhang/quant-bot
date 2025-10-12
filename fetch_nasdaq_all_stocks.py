#!/usr/bin/env python3
"""
获取NASDAQ全量股票的近5年历史数据
包括NASDAQ、NYSE、AMEX等所有交易所的股票
"""

import ftplib
import io
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Set
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.dao import StockDataDAO
from src.database.models import StockData

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nasdaq_all_stocks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NASDAQAllStocksFetcher:
    def __init__(self):
        self.dao = StockDataDAO()
        self.ftp_host = 'ftp.nasdaqtrader.com'
        self.ftp_directory = 'symboldirectory'
        
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
                    etf = parts[6].strip()  # ETF标识
                    
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
    
    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[StockData]:
        """获取单只股票的历史数据"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"股票 {symbol} 没有历史数据")
                return []
            
            stock_data_list = []
            for date, row in hist.iterrows():
                stock_data = StockData(
                    symbol=symbol,
                    date=date.date(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                )
                stock_data_list.append(stock_data)
            
            logger.info(f"股票 {symbol} 获取到 {len(stock_data_list)} 条历史数据")
            return stock_data_list
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 数据失败: {e}")
            return []
    
    def save_stock_data(self, stock_data_list: List[StockData]) -> int:
        """批量保存股票数据到数据库"""
        if not stock_data_list:
            return 0
            
        try:
            saved_count = 0
            for stock_data in stock_data_list:
                # 检查数据是否已存在 - 使用正确的方法名
                existing = self.dao.get_by_symbol_and_date_range(
                    stock_data.symbol, 
                    stock_data.date, 
                    stock_data.date
                )
                if not existing:
                    # 使用正确的create方法参数
                    self.dao.create(
                        symbol=stock_data.symbol,
                        date=stock_data.date,
                        open_price=stock_data.open,
                        high=stock_data.high,
                        low=stock_data.low,
                        close=stock_data.close,
                        volume=stock_data.volume
                    )
                    saved_count += 1
            
            return saved_count
            
        except Exception as e:
            logger.error(f"保存股票数据失败: {e}")
            return 0
    
    def run(self):
        """主执行函数"""
        logger.info("开始获取NASDAQ全量股票的近5年历史数据")
        
        # 计算时间范围（近5年）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5年
        
        logger.info(f"数据时间范围: {start_date.date()} 到 {end_date.date()}")
        
        try:
            # 1. 获取所有股票代码
            logger.info("正在从NASDAQ FTP服务器获取股票列表...")
            all_symbols = self.fetch_nasdaq_symbols()
            
            if not all_symbols:
                logger.error("未能获取到任何股票代码")
                return
            
            logger.info(f"准备获取 {len(all_symbols)} 只股票的历史数据")
            
            # 2. 获取每只股票的历史数据
            total_processed = 0
            total_saved = 0
            failed_symbols = []
            
            # 将股票列表转换为排序列表以便跟踪进度
            symbols_list = sorted(list(all_symbols))
            
            for i, symbol in enumerate(symbols_list, 1):
                logger.info(f"处理股票 {symbol} ({i}/{len(symbols_list)})")
                
                # 获取股票数据
                stock_data_list = self.fetch_stock_data(symbol, start_date, end_date)
                
                if stock_data_list:
                    # 保存到数据库
                    saved_count = self.save_stock_data(stock_data_list)
                    total_saved += saved_count
                    logger.info(f"股票 {symbol} 保存了 {saved_count} 条新数据")
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"股票 {symbol} 未获取到数据")
                
                total_processed += 1
                
                # 每处理10只股票休息一下，避免被限流
                if i % 10 == 0:
                    logger.info(f"已处理 {i} 只股票，休息2秒...")
                    time.sleep(2)
                
                # 每处理100只股票输出进度
                if i % 100 == 0:
                    logger.info(f"进度: {i}/{len(symbols_list)} ({i/len(symbols_list)*100:.1f}%)")
            
            # 输出最终统计
            logger.info("=" * 50)
            logger.info("数据获取完成！")
            logger.info(f"总处理股票数: {total_processed}")
            logger.info(f"总保存数据条数: {total_saved}")
            logger.info(f"失败股票数: {len(failed_symbols)}")
            
            if failed_symbols:
                logger.info(f"失败的股票代码: {', '.join(failed_symbols[:20])}")  # 只显示前20个
                if len(failed_symbols) > 20:
                    logger.info(f"... 还有 {len(failed_symbols) - 20} 个失败的股票")
            
        except Exception as e:
            logger.error(f"执行过程中发生错误: {e}")
            raise

def main():
    """主函数"""
    fetcher = NASDAQAllStocksFetcher()
    fetcher.run()

if __name__ == "__main__":
    main()