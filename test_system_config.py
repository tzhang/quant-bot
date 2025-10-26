#!/usr/bin/env python3
"""
系统配置验证脚本
验证交易系统配置和数据源连接
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_import():
    """测试配置文件导入"""
    try:
        from config.settings import DataSourceConfig, TradingConfig
        logger.info("✅ 配置文件导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 配置文件导入失败: {e}")
        return False

def test_data_source_config():
    """测试数据源配置"""
    try:
        from config.settings import DataSourceConfig
        
        config = DataSourceConfig()
        
        logger.info("📊 数据源配置检查:")
        logger.info(f"  - IB启用: {config.ib_enabled}")
        logger.info(f"  - IB主机: {config.ib_host}")
        logger.info(f"  - IB端口: {config.ib_port}")
        logger.info(f"  - Yahoo Finance启用: {config.yahoo_enabled}")
        logger.info(f"  - Qlib启用: {config.qlib_enabled}")
        logger.info(f"  - OpenBB启用: {config.openbb_enabled}")
        logger.info(f"  - 主要数据源: {config.primary_sources}")
        logger.info(f"  - 备用数据源: {config.fallback_sources}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 数据源配置检查失败: {e}")
        return False

def test_trading_config():
    """测试交易配置"""
    try:
        from config.settings import TradingConfig
        
        config = TradingConfig()
        
        logger.info("💰 交易配置检查:")
        logger.info(f"  - 初始资金: ${config.initial_capital:,.2f}")
        logger.info(f"  - 佣金费率: {config.commission_rate:.4f}")
        logger.info(f"  - 滑点费率: {config.slippage_rate:.4f}")
        logger.info(f"  - 最大持仓比例: {config.max_position_size:.2%}")
        logger.info(f"  - 最大回撤: {config.max_drawdown:.2%}")
        logger.info(f"  - 止损比例: {config.stop_loss:.2%}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 交易配置检查失败: {e}")
        return False

def test_yaml_config():
    """测试YAML配置文件"""
    try:
        import yaml
        
        config_file = os.path.join(project_root, 'trading_config.yaml')
        if not os.path.exists(config_file):
            logger.error(f"❌ 配置文件不存在: {config_file}")
            return False
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("📄 YAML配置文件检查:")
        logger.info(f"  - 主要数据源: {config.get('data_sources', {}).get('primary', 'N/A')}")
        logger.info(f"  - 备用数据源: {config.get('data_sources', {}).get('backup', 'N/A')}")
        logger.info(f"  - 交易模式: {config.get('trading', {}).get('mode', 'N/A')}")
        logger.info(f"  - 初始资金: ${config.get('trading', {}).get('initial_capital', 0):,.2f}")
        logger.info(f"  - 交易标的: {config.get('trading', {}).get('symbols', [])}")
        
        # 检查数据源是否为IB
        primary_source = config.get('data_sources', {}).get('primary', '')
        if primary_source == 'ib':
            logger.info("✅ 主要数据源已设置为IB")
        else:
            logger.warning(f"⚠️ 主要数据源为: {primary_source}")
        
        return True
    except Exception as e:
        logger.error(f"❌ YAML配置文件检查失败: {e}")
        return False

def test_data_adapter():
    """测试数据适配器初始化"""
    try:
        from src.data.data_adapter import DataAdapter
        
        logger.info("🔌 测试数据适配器初始化...")
        
        # 创建数据适配器实例（禁用yfinance）
        adapter = DataAdapter(fallback_to_yfinance=False)
        
        logger.info("✅ 数据适配器初始化成功")
        
        # 检查可用的数据提供器
        available_providers = []
        if hasattr(adapter, 'qlib_provider') and adapter.qlib_provider:
            available_providers.append('Qlib')
        if hasattr(adapter, 'openbb_provider') and adapter.openbb_provider:
            available_providers.append('OpenBB')
        if hasattr(adapter, 'ib_provider') and adapter.ib_provider:
            available_providers.append('Interactive Brokers')
        
        logger.info(f"  - 可用数据提供器: {available_providers}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 数据适配器初始化失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始系统配置验证...")
    logger.info(f"项目根目录: {project_root}")
    
    tests = [
        ("配置文件导入", test_config_import),
        ("数据源配置", test_data_source_config),
        ("交易配置", test_trading_config),
        ("YAML配置文件", test_yaml_config),
        ("数据适配器", test_data_adapter),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 测试: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} - 通过")
            else:
                logger.error(f"❌ {test_name} - 失败")
        except Exception as e:
            logger.error(f"❌ {test_name} - 异常: {e}")
    
    logger.info(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有配置验证通过，系统准备就绪！")
        return True
    else:
        logger.error("⚠️ 部分配置验证失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)