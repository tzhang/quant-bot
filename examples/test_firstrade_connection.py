#!/usr/bin/env python3
"""
测试Firstrade连接的独立脚本
"""

import logging
import traceback
from datetime import datetime

# 设置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_firstrade_connection():
    """测试Firstrade连接"""
    try:
        logger.info("开始测试Firstrade连接...")
        
        # 导入FirstradeTradingSystem
        from firstrade_trading_system import FirstradeTradingSystem
        logger.info("成功导入FirstradeTradingSystem")
        
        # 创建实例
        trading_system = FirstradeTradingSystem()
        logger.info("成功创建FirstradeTradingSystem实例")
        
        # 测试获取投资组合状态
        logger.info("测试获取投资组合状态...")
        portfolio_status = trading_system.get_portfolio_status()
        logger.info(f"投资组合状态结果: {portfolio_status}")
        
        # 测试获取持仓
        logger.info("测试获取持仓...")
        positions = trading_system.get_detailed_positions()
        logger.info(f"持仓结果: {positions}")
        
        # 测试计算绩效
        logger.info("测试计算投资组合绩效...")
        performance = trading_system.calculate_portfolio_performance()
        logger.info(f"绩效结果: {performance}")
        
        # 测试获取市场数据
        logger.info("测试获取市场数据...")
        watchlist = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        for symbol in watchlist:
            try:
                market_data = trading_system.get_market_data(symbol)
                logger.info(f"{symbol} 市场数据: {market_data}")
            except Exception as e:
                logger.error(f"获取 {symbol} 市场数据失败: {e}")
        
        logger.info("Firstrade连接测试完成")
        
    except ImportError as e:
        logger.error(f"导入FirstradeTradingSystem失败: {e}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {type(e).__name__}: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")

if __name__ == "__main__":
    test_firstrade_connection()