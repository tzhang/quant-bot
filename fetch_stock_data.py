#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据批量获取脚本

该脚本用于批量获取股票数据并存储到PostgreSQL数据库中
支持多种数据源：Qlib、yfinance、OpenBB等
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.data.data_manager import DataManager
from src.database.dao import stock_data_dao
from src.database.connection import get_db_session
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认股票列表
DEFAULT_SYMBOLS = [
    # 科技股
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    # 金融股
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL',
    # 消费股
    'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE',
    # 工业股
    'BA', 'CAT', 'GE', 'MMM', 'UPS', 'HON', 'LMT', 'RTX',
    # 医疗股
    'UNH', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD'
]

def fetch_and_store_stock_data(
    symbols: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    batch_size: int = 5
) -> Dict[str, Any]:
    """
    批量获取并存储股票数据
    
    Args:
        symbols: 股票代码列表，默认使用DEFAULT_SYMBOLS
        start_date: 开始日期，格式'YYYY-MM-DD'，默认为1年前
        end_date: 结束日期，格式'YYYY-MM-DD'，默认为今天
        batch_size: 批处理大小，避免API限制
    
    Returns:
        包含处理结果的字典
    """
    # 设置默认参数
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始批量获取股票数据")
    logger.info(f"股票数量: {len(symbols)}")
    logger.info(f"时间范围: {start_date} 到 {end_date}")
    logger.info(f"批处理大小: {batch_size}")
    
    # 初始化组件
    data_manager = DataManager()
    
    # 统计信息
    results = {
        'total_symbols': len(symbols),
        'successful': 0,
        'failed': 0,
        'total_records': 0,
        'failed_symbols': [],
        'processing_time': 0
    }
    
    start_time = time.time()
    
    try:
        # 获取数据库会话
        with get_db_session() as session:
            
            # 分批处理股票
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}: {batch_symbols}")
                
                for symbol in batch_symbols:
                    try:
                        logger.info(f"获取 {symbol} 数据...")
                        
                        # 获取股票数据
                        stock_data = data_manager.get_stock_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if stock_data is not None and not stock_data.empty:
                            # 准备数据库记录
                            records = []
                            for date, row in stock_data.iterrows():
                                record_data = {
                                    'symbol': symbol,
                                    'date': date.date() if hasattr(date, 'date') else date,
                                    'open_price': float(row.get('Open', row.get('open', 0))),
                                    'high_price': float(row.get('High', row.get('high', 0))),
                                    'low_price': float(row.get('Low', row.get('low', 0))),
                                    'close_price': float(row.get('Close', row.get('close', 0))),
                                    'volume': int(row.get('Volume', row.get('volume', 0))),
                                    'adjusted_close': float(row.get('Adj Close', row.get('adj_close', row.get('Close', row.get('close', 0)))))
                                }
                                records.append(record_data)
                            
                            # 批量插入数据库
                            if records:
                                stock_data_dao.batch_create(records)
                                results['total_records'] += len(records)
                                logger.info(f"✅ {symbol}: 成功存储 {len(records)} 条记录")
                                results['successful'] += 1
                            else:
                                logger.warning(f"⚠️ {symbol}: 无有效数据")
                                results['failed'] += 1
                                results['failed_symbols'].append(symbol)
                        else:
                            logger.warning(f"⚠️ {symbol}: 获取数据失败或为空")
                            results['failed'] += 1
                            results['failed_symbols'].append(symbol)
                    
                    except Exception as e:
                        logger.error(f"❌ {symbol}: 处理失败 - {str(e)}")
                        results['failed'] += 1
                        results['failed_symbols'].append(symbol)
                    
                    # 添加延迟避免API限制
                    time.sleep(0.5)
                
                # 批次间延迟
                if i + batch_size < len(symbols):
                    logger.info("批次间休息2秒...")
                    time.sleep(2)
    
    except Exception as e:
        logger.error(f"数据库操作失败: {str(e)}")
        raise
    
    # 计算处理时间
    results['processing_time'] = time.time() - start_time
    
    return results

def print_summary(results: Dict[str, Any]):
    """打印处理结果摘要"""
    print("\n" + "="*60)
    print("📊 股票数据获取结果摘要")
    print("="*60)
    print(f"📈 总股票数量: {results['total_symbols']}")
    print(f"✅ 成功获取: {results['successful']}")
    print(f"❌ 获取失败: {results['failed']}")
    print(f"📄 总记录数: {results['total_records']:,}")
    print(f"⏱️ 处理时间: {results['processing_time']:.2f} 秒")
    
    if results['failed_symbols']:
        print(f"\n❌ 失败的股票代码:")
        for symbol in results['failed_symbols']:
            print(f"   - {symbol}")
    
    success_rate = (results['successful'] / results['total_symbols']) * 100
    print(f"\n🎯 成功率: {success_rate:.1f}%")
    
    if results['total_records'] > 0:
        avg_records = results['total_records'] / results['successful']
        print(f"📊 平均每股记录数: {avg_records:.0f}")

def main():
    """主函数"""
    print("🚀 启动股票数据批量获取流程...")
    
    try:
        # 执行数据获取
        results = fetch_and_store_stock_data(
            symbols=DEFAULT_SYMBOLS[:10],  # 先测试前10只股票
            start_date='2023-01-01',
            end_date='2024-01-01'
        )
        
        # 打印结果摘要
        print_summary(results)
        
        print("\n🎉 股票数据获取流程完成！")
        
    except Exception as e:
        logger.error(f"数据获取流程失败: {str(e)}")
        print(f"\n❌ 数据获取流程失败: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)