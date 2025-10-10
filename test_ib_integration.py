#!/usr/bin/env python3
"""
测试Interactive Brokers TWS API集成
验证IB数据源的连接和数据获取功能
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.ib_data_provider import IBDataProvider, create_ib_provider
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__)

def test_ib_connection():
    """测试IB连接"""
    logger.info("开始测试IB连接...")
    
    try:
        # 创建IB数据提供者
        ib_provider = create_ib_provider()
        
        if ib_provider is None:
            logger.error("无法创建IB数据提供者")
            return False
            
        if not ib_provider.is_available:
            logger.error("IB API不可用")
            return False
            
        logger.info("IB数据提供者创建成功")
        return True
        
    except Exception as e:
        logger.error(f"IB连接测试失败: {e}")
        return False

def test_ib_data_fetch():
    """测试IB数据获取"""
    logger.info("开始测试IB数据获取...")
    
    try:
        # 创建IB数据提供者
        ib_provider = create_ib_provider()
        
        if not ib_provider or not ib_provider.is_available:
            logger.error("IB数据提供者不可用")
            return False
        
        # 测试股票列表
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # 设置日期范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        logger.info(f"测试日期范围: {start_date} 到 {end_date}")
        
        success_count = 0
        
        for symbol in test_symbols:
            try:
                logger.info(f"测试获取 {symbol} 数据...")
                
                data = ib_provider.get_stock_data(symbol, start_date, end_date)
                
                if data is not None and not data.empty:
                    logger.info(f"成功获取 {symbol} 数据，共 {len(data)} 条记录")
                    logger.info(f"数据列: {list(data.columns)}")
                    logger.info(f"数据样本:\n{data.head()}")
                    success_count += 1
                else:
                    logger.warning(f"未获取到 {symbol} 数据")
                    
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
        
        logger.info(f"数据获取测试完成，成功获取 {success_count}/{len(test_symbols)} 只股票数据")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"IB数据获取测试失败: {e}")
        return False

def test_ib_data_availability():
    """测试IB数据可用性检查"""
    logger.info("开始测试IB数据可用性检查...")
    
    try:
        ib_provider = create_ib_provider()
        
        if not ib_provider or not ib_provider.is_available:
            logger.error("IB数据提供者不可用")
            return False
        
        test_symbols = ['AAPL', 'INVALID_SYMBOL']
        
        for symbol in test_symbols:
            try:
                availability = ib_provider.check_data_availability(symbol)
                logger.info(f"{symbol} 数据可用性: {availability}")
                
            except Exception as e:
                logger.error(f"检查 {symbol} 数据可用性失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"IB数据可用性测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始IB TWS API集成测试")
    
    # 测试结果
    results = {}
    
    # 1. 测试连接
    results['connection'] = test_ib_connection()
    
    # 2. 测试数据获取
    results['data_fetch'] = test_ib_data_fetch()
    
    # 3. 测试数据可用性
    results['data_availability'] = test_ib_data_availability()
    
    # 输出测试结果
    logger.info("\n" + "="*50)
    logger.info("IB TWS API集成测试结果:")
    logger.info("="*50)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
    
    # 总体结果
    all_passed = all(results.values())
    overall_status = "✓ 全部通过" if all_passed else "✗ 部分失败"
    logger.info(f"\n总体结果: {overall_status}")
    
    if all_passed:
        logger.info("IB TWS API集成测试成功！可以在数据获取脚本中使用IB数据源。")
    else:
        logger.warning("IB TWS API集成测试存在问题，请检查配置和连接。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)