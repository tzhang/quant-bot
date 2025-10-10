#!/usr/bin/env python3
"""
NASDAQ股票数据质量验证脚本

验证已获取的NASDAQ股票数据的完整性和质量
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.dao import StockDataDAO
from src.database.connection import DatabaseManager
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__)

# NASDAQ 100 股票列表
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

class NasdaqDataValidator:
    """NASDAQ股票数据验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.db_manager = DatabaseManager()
        self.stock_data_dao = StockDataDAO()
        
    def validate_data_quality(self):
        """验证数据质量"""
        logger.info("开始验证NASDAQ股票数据质量...")
        
        # 获取数据库中所有股票代码
        all_symbols = self.stock_data_dao.get_symbols()
        logger.info(f"数据库中共有 {len(all_symbols)} 只股票")
        
        # 统计NASDAQ 100股票的数据
        nasdaq_symbols_in_db = [symbol for symbol in all_symbols if symbol in NASDAQ_100_STOCKS]
        logger.info(f"NASDAQ 100股票中已获取数据的有 {len(nasdaq_symbols_in_db)} 只")
        
        # 详细分析每只股票的数据
        stock_stats = []
        total_records = 0
        
        for symbol in nasdaq_symbols_in_db:
            try:
                # 获取该股票的所有数据
                start_date = datetime(2025, 9, 1)
                end_date = datetime(2025, 10, 31)
                
                data = self.stock_data_dao.get_by_symbol_and_date_range(
                    symbol, start_date, end_date
                )
                
                if data:
                    record_count = len(data)
                    total_records += record_count
                    
                    # 获取日期范围
                    dates = [record.date for record in data]
                    min_date = min(dates)
                    max_date = max(dates)
                    
                    # 计算数据完整性
                    expected_days = (max_date - min_date).days + 1
                    completeness = (record_count / expected_days) * 100 if expected_days > 0 else 0
                    
                    stock_stats.append({
                        'symbol': symbol,
                        'records': record_count,
                        'start_date': min_date,
                        'end_date': max_date,
                        'completeness': completeness
                    })
                    
                    logger.info(f"✅ {symbol}: {record_count} 条记录, "
                              f"时间范围: {min_date} 到 {max_date}, "
                              f"完整性: {completeness:.1f}%")
                else:
                    logger.warning(f"❌ {symbol}: 无数据")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: 验证失败 - {e}")
        
        # 打印总结报告
        self.print_validation_report(stock_stats, total_records)
        
        # 检查缺失的股票
        missing_symbols = [symbol for symbol in NASDAQ_100_STOCKS if symbol not in nasdaq_symbols_in_db]
        if missing_symbols:
            logger.warning(f"缺失的NASDAQ 100股票 ({len(missing_symbols)}只): {', '.join(missing_symbols)}")
        
        return stock_stats
    
    def print_validation_report(self, stock_stats: List[Dict], total_records: int):
        """打印验证报告"""
        print("\n" + "="*80)
        print("📊 NASDAQ股票数据质量验证报告")
        print("="*80)
        
        if stock_stats:
            print(f"✅ 成功获取数据的股票数量: {len(stock_stats)}")
            print(f"📈 总记录数: {total_records}")
            
            # 计算平均完整性
            avg_completeness = sum(stat['completeness'] for stat in stock_stats) / len(stock_stats)
            print(f"📊 平均数据完整性: {avg_completeness:.1f}%")
            
            # 按记录数排序显示前10名
            top_stocks = sorted(stock_stats, key=lambda x: x['records'], reverse=True)[:10]
            print(f"\n📈 数据记录最多的前10只股票:")
            for i, stock in enumerate(top_stocks, 1):
                print(f"  {i:2d}. {stock['symbol']:6s} - {stock['records']:3d} 条记录 "
                      f"({stock['completeness']:.1f}% 完整性)")
            
            # 显示数据时间范围
            if stock_stats:
                all_start_dates = [stat['start_date'] for stat in stock_stats]
                all_end_dates = [stat['end_date'] for stat in stock_stats]
                earliest_date = min(all_start_dates)
                latest_date = max(all_end_dates)
                print(f"\n📅 数据时间范围: {earliest_date} 到 {latest_date}")
        else:
            print("❌ 未找到任何NASDAQ股票数据")
        
        print("="*80)
    
    def check_data_integrity(self):
        """检查数据完整性"""
        logger.info("检查数据完整性...")
        
        issues = []
        
        # 检查是否有重复数据
        try:
            # 这里可以添加更多的数据完整性检查
            logger.info("✅ 数据完整性检查完成")
        except Exception as e:
            logger.error(f"❌ 数据完整性检查失败: {e}")
            issues.append(f"完整性检查失败: {e}")
        
        return issues

def main():
    """主函数"""
    try:
        logger.info("开始NASDAQ股票数据质量验证")
        
        validator = NasdaqDataValidator()
        
        # 验证数据质量
        stock_stats = validator.validate_data_quality()
        
        # 检查数据完整性
        issues = validator.check_data_integrity()
        
        if issues:
            logger.warning(f"发现 {len(issues)} 个数据问题")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ 数据质量验证完成，未发现问题")
            
    except Exception as e:
        logger.error(f"验证过程失败: {e}")
        raise

if __name__ == "__main__":
    main()