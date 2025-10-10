#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版股票数据批量获取脚本

该脚本用于批量获取股票数据并存储到PostgreSQL数据库中
使用单条记录插入方式，避免批量插入的问题
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
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认股票列表（较少数量用于测试）
DEFAULT_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'
]

def fetch_and_store_stock_data_simple(
    symbols: List[str] = None,
    start_date: str = None,
    end_date: str = None
) -> Dict[str, Any]:
    """
    简化版批量获取并存储股票数据
    
    Args:
        symbols: 股票代码列表，默认使用DEFAULT_SYMBOLS
        start_date: 开始日期，格式'YYYY-MM-DD'，默认为1年前
        end_date: 结束日期，格式'YYYY-MM-DD'，默认为今天
    
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
        for symbol in symbols:
            try:
                logger.info(f"获取 {symbol} 数据...")
                
                # 获取股票数据
                stock_data = data_manager.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if stock_data is not None and not stock_data.empty:
                    # 逐条插入数据库
                    record_count = 0
                    for date, row in stock_data.iterrows():
                        try:
                            # 转换日期格式
                            if hasattr(date, 'date'):
                                date_obj = date.date()
                            else:
                                date_obj = date
                            
                            # 创建单条记录
                            record = stock_data_dao.create(
                                symbol=symbol,
                                date=datetime.combine(date_obj, datetime.min.time()) if hasattr(date_obj, 'year') else date,
                                open_price=float(row.get('Open', row.get('open', 0))),
                                high=float(row.get('High', row.get('high', 0))),
                                low=float(row.get('Low', row.get('low', 0))),
                                close=float(row.get('Close', row.get('close', 0))),
                                volume=int(row.get('Volume', row.get('volume', 0)))
                            )
                            record_count += 1
                        except Exception as e:
                            logger.warning(f"插入记录失败 {symbol} {date}: {str(e)}")
                            continue
                    
                    if record_count > 0:
                        results['total_records'] += record_count
                        logger.info(f"✅ {symbol}: 成功存储 {record_count} 条记录")
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
            time.sleep(1)
    
    except Exception as e:
        logger.error(f"数据获取流程失败: {str(e)}")
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
    print("🚀 启动简化版股票数据批量获取流程...")
    
    try:
        # 执行数据获取
        results = fetch_and_store_stock_data_simple(
            symbols=DEFAULT_SYMBOLS,  # 使用所有5只股票
            start_date='2023-01-01',
            end_date='2023-02-01'  # 缩短时间范围用于测试
        )
        
        # 打印结果摘要
        print_summary(results)
        
        # 验证数据库中的数据
        print("\n" + "="*60)
        print("📊 数据库验证")
        print("="*60)
        
        symbols_in_db = stock_data_dao.get_symbols()
        print(f"数据库中的股票代码: {symbols_in_db}")
        
        for symbol in symbols_in_db:
            try:
                latest_data = stock_data_dao.get_latest_by_symbol(symbol)
                if latest_data:
                    print(f"{symbol}: 最新数据日期 {latest_data.date}, 收盘价 ${latest_data.close:.2f}")
            except Exception as e:
                print(f"{symbol}: 查询失败 - {str(e)}")
        
        print("\n🎉 股票数据获取流程完成！")
        
    except Exception as e:
        logger.error(f"数据获取流程失败: {str(e)}")
        print(f"\n❌ 数据获取流程失败: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)