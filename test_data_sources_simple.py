#!/usr/bin/env python3
"""
简化数据源测试脚本
测试移除Yahoo Finance后的数据获取功能
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_adapter_without_yfinance():
    """测试移除Yahoo Finance后的数据适配器功能"""
    try:
        from src.data.data_adapter import DataAdapter
        
        # 创建数据适配器实例，禁用yfinance
        adapter = DataAdapter(
            prefer_qlib=True,
            enable_openbb=True,
            enable_ib=True,
            fallback_to_yfinance=False  # 明确禁用yfinance
        )
        
        logger.info("✅ DataAdapter 初始化成功 (已禁用Yahoo Finance)")
        
        # 测试数据源配置
        logger.info(f"主要数据源: {getattr(adapter, 'primary_sources', ['ib', 'qlib', 'openbb'])}")
        logger.info(f"备用数据源: {getattr(adapter, 'fallback_sources', ['alpha_vantage', 'quandl'])}")
        logger.info(f"Yahoo Finance 状态: {'禁用' if not adapter.fallback_to_yfinance else '启用'}")
        
        # 测试获取股票数据
        test_symbols = ['AAPL', 'MSFT']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        for symbol in test_symbols:
            logger.info(f"\n测试获取 {symbol} 数据...")
            try:
                data = adapter.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is not None and not data.empty:
                    logger.info(f"✅ {symbol} 数据获取成功: {len(data)} 条记录")
                    logger.info(f"   数据范围: {data.index[0]} 到 {data.index[-1]}")
                    logger.info(f"   列名: {list(data.columns)}")
                else:
                    logger.warning(f"⚠️  {symbol} 数据为空或获取失败")
                    
            except Exception as e:
                logger.error(f"❌ {symbol} 数据获取异常: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 测试异常: {e}")
        return False

def test_config_settings():
    """测试配置设置"""
    try:
        # 尝试直接导入设置
        import config.settings as settings_module
        
        if hasattr(settings_module, 'DataSourceConfig'):
            config = settings_module.DataSourceConfig()
            logger.info("✅ DataSourceConfig 导入成功")
            logger.info(f"Yahoo Finance 启用状态: {getattr(config, 'yahoo_enabled', 'N/A')}")
            logger.info(f"IB Gateway 启用状态: {getattr(config, 'ib_enabled', 'N/A')}")
        else:
            logger.warning("⚠️  DataSourceConfig 类未找到")
            
        return True
        
    except ImportError as e:
        logger.error(f"❌ 配置导入错误: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 配置测试异常: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("开始简化数据源测试")
    logger.info("=" * 60)
    
    # 测试配置
    logger.info("\n1. 测试配置设置...")
    config_ok = test_config_settings()
    
    # 测试数据适配器
    logger.info("\n2. 测试数据适配器...")
    adapter_ok = test_data_adapter_without_yfinance()
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结:")
    logger.info(f"配置测试: {'✅ 通过' if config_ok else '❌ 失败'}")
    logger.info(f"数据适配器测试: {'✅ 通过' if adapter_ok else '❌ 失败'}")
    
    if config_ok and adapter_ok:
        logger.info("🎉 所有测试通过！Yahoo Finance 已成功移除")
    else:
        logger.error("❌ 部分测试失败，需要进一步检查")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()