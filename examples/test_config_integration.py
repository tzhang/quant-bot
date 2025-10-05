#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置集成测试脚本
测试新券商配置的加载和使用
"""

import os
import logging
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_loading():
    """测试配置加载"""
    logger.info("开始测试配置加载...")
    
    try:
        # 创建配置实例
        config = Config()
        logger.info("✓ 配置实例创建成功")
        
        # 检查新券商配置是否存在
        brokers_to_test = [
            ('TD Ameritrade', 'td_ameritrade'),
            ('Charles Schwab', 'charles_schwab'),
            ('E*TRADE', 'etrade'),
            ('Robinhood', 'robinhood')
        ]
        
        for broker_name, attr_name in brokers_to_test:
            if hasattr(config, attr_name):
                broker_config = getattr(config, attr_name)
                logger.info(f"✓ {broker_name} 配置加载成功")
                logger.info(f"  启用状态: {broker_config.enabled}")
                logger.info(f"  沙盒模式: {broker_config.sandbox}")
                logger.info(f"  干运行模式: {broker_config.dry_run}")
            else:
                logger.error(f"✗ {broker_name} 配置不存在")
        
        logger.info("配置加载测试完成")
        
    except Exception as e:
        logger.error(f"配置加载测试失败: {e}")

def test_environment_variables():
    """测试环境变量加载"""
    logger.info("开始测试环境变量加载...")
    
    # 设置测试环境变量
    test_env_vars = {
        # TD Ameritrade
        'TD_AMERITRADE_CONSUMER_KEY': 'test_td_key',
        'TD_AMERITRADE_CONSUMER_SECRET': 'test_td_secret',
        'TD_AMERITRADE_ACCESS_TOKEN': 'test_td_token',
        'TD_AMERITRADE_ACCESS_SECRET': 'test_td_access_secret',
        'TD_AMERITRADE_SANDBOX': 'true',
        'TD_AMERITRADE_DRY_RUN': 'true',
        'TD_AMERITRADE_ENABLED': 'true',
        
        # Charles Schwab
        'CHARLES_SCHWAB_APP_KEY': 'test_schwab_key',
        'CHARLES_SCHWAB_APP_SECRET': 'test_schwab_secret',
        'CHARLES_SCHWAB_ACCESS_TOKEN': 'test_schwab_token',
        'CHARLES_SCHWAB_REFRESH_TOKEN': 'test_schwab_refresh',
        'CHARLES_SCHWAB_SANDBOX': 'true',
        'CHARLES_SCHWAB_DRY_RUN': 'true',
        'CHARLES_SCHWAB_ENABLED': 'true',
        
        # E*TRADE
        'ETRADE_CONSUMER_KEY': 'test_etrade_key',
        'ETRADE_CONSUMER_SECRET': 'test_etrade_secret',
        'ETRADE_ACCESS_TOKEN': 'test_etrade_token',
        'ETRADE_ACCESS_SECRET': 'test_etrade_access_secret',
        'ETRADE_SANDBOX': 'true',
        'ETRADE_DRY_RUN': 'true',
        'ETRADE_ENABLED': 'true',
        
        # Robinhood
        'ROBINHOOD_USERNAME': 'test_rh_user',
        'ROBINHOOD_PASSWORD': 'test_rh_pass',
        'ROBINHOOD_DEVICE_TOKEN': 'test_rh_device',
        'ROBINHOOD_CHALLENGE_TYPE': 'sms',
        'ROBINHOOD_SANDBOX': 'true',
        'ROBINHOOD_DRY_RUN': 'true',
        'ROBINHOOD_ENABLED': 'true',
    }
    
    # 设置环境变量
    for key, value in test_env_vars.items():
        os.environ[key] = value
    
    try:
        # 重新加载配置
        config = Config()
        config._load_from_env()
        
        # 验证TD Ameritrade配置
        if hasattr(config, 'td_ameritrade'):
            td_config = config.td_ameritrade
            assert td_config.consumer_key == 'test_td_key'
            assert td_config.consumer_secret == 'test_td_secret'
            assert td_config.access_token == 'test_td_token'
            assert td_config.access_secret == 'test_td_access_secret'
            assert td_config.sandbox == True
            assert td_config.dry_run == True
            assert td_config.enabled == True
            logger.info("✓ TD Ameritrade 环境变量加载成功")
        
        # 验证Charles Schwab配置
        if hasattr(config, 'charles_schwab'):
            schwab_config = config.charles_schwab
            assert schwab_config.app_key == 'test_schwab_key'
            assert schwab_config.app_secret == 'test_schwab_secret'
            assert schwab_config.access_token == 'test_schwab_token'
            assert schwab_config.refresh_token == 'test_schwab_refresh'
            assert schwab_config.sandbox == True
            assert schwab_config.dry_run == True
            assert schwab_config.enabled == True
            logger.info("✓ Charles Schwab 环境变量加载成功")
        
        # 验证E*TRADE配置
        if hasattr(config, 'etrade'):
            etrade_config = config.etrade
            assert etrade_config.consumer_key == 'test_etrade_key'
            assert etrade_config.consumer_secret == 'test_etrade_secret'
            assert etrade_config.access_token == 'test_etrade_token'
            assert etrade_config.access_secret == 'test_etrade_access_secret'
            assert etrade_config.sandbox == True
            assert etrade_config.dry_run == True
            assert etrade_config.enabled == True
            logger.info("✓ E*TRADE 环境变量加载成功")
        
        # 验证Robinhood配置
        if hasattr(config, 'robinhood'):
            rh_config = config.robinhood
            assert rh_config.username == 'test_rh_user'
            assert rh_config.password == 'test_rh_pass'
            assert rh_config.device_token == 'test_rh_device'
            assert rh_config.challenge_type == 'sms'
            assert rh_config.sandbox == True
            assert rh_config.dry_run == True
            assert rh_config.enabled == True
            logger.info("✓ Robinhood 环境变量加载成功")
        
        logger.info("环境变量加载测试完成")
        
    except Exception as e:
        logger.error(f"环境变量加载测试失败: {e}")
    finally:
        # 清理环境变量
        for key in test_env_vars.keys():
            if key in os.environ:
                del os.environ[key]

def test_config_broker_factory_integration():
    """测试配置与券商工厂集成"""
    logger.info("开始测试配置与券商工厂集成...")
    
    try:
        from broker_factory import BrokerFactory
        
        # 创建配置
        config = Config()
        
        # 测试使用配置创建券商
        brokers_config = [
            ('td_ameritrade', config.td_ameritrade if hasattr(config, 'td_ameritrade') else None),
            ('charles_schwab', config.charles_schwab if hasattr(config, 'charles_schwab') else None),
            ('etrade', config.etrade if hasattr(config, 'etrade') else None),
            ('robinhood', config.robinhood if hasattr(config, 'robinhood') else None),
        ]
        
        for broker_type, broker_config in brokers_config:
            if broker_config and broker_config.enabled:
                try:
                    if broker_type == 'td_ameritrade':
                        broker = BrokerFactory.create_broker(
                            broker_type,
                            consumer_key=broker_config.consumer_key,
                            consumer_secret=broker_config.consumer_secret,
                            access_token=broker_config.access_token,
                            access_secret=broker_config.access_secret,
                            sandbox=broker_config.sandbox,
                            dry_run=broker_config.dry_run
                        )
                    elif broker_type == 'charles_schwab':
                        broker = BrokerFactory.create_broker(
                            broker_type,
                            app_key=broker_config.app_key,
                            app_secret=broker_config.app_secret,
                            access_token=broker_config.access_token,
                            refresh_token=broker_config.refresh_token,
                            sandbox=broker_config.sandbox,
                            dry_run=broker_config.dry_run
                        )
                    elif broker_type == 'etrade':
                        broker = BrokerFactory.create_broker(
                            broker_type,
                            consumer_key=broker_config.consumer_key,
                            consumer_secret=broker_config.consumer_secret,
                            access_token=broker_config.access_token,
                            access_secret=broker_config.access_secret,
                            sandbox=broker_config.sandbox,
                            dry_run=broker_config.dry_run
                        )
                    elif broker_type == 'robinhood':
                        broker = BrokerFactory.create_broker(
                            broker_type,
                            username=broker_config.username,
                            password=broker_config.password,
                            device_token=broker_config.device_token,
                            challenge_type=broker_config.challenge_type,
                            sandbox=broker_config.sandbox,
                            dry_run=broker_config.dry_run
                        )
                    
                    logger.info(f"✓ {broker_type} 券商使用配置创建成功")
                    
                except Exception as e:
                    logger.error(f"✗ {broker_type} 券商使用配置创建失败: {e}")
            else:
                logger.info(f"- {broker_type} 券商未启用或配置不存在")
        
        logger.info("配置与券商工厂集成测试完成")
        
    except Exception as e:
        logger.error(f"配置与券商工厂集成测试失败: {e}")

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("配置集成测试开始")
    logger.info("=" * 60)
    
    # 运行所有测试
    test_config_loading()
    print()
    
    test_environment_variables()
    print()
    
    test_config_broker_factory_integration()
    print()
    
    logger.info("=" * 60)
    logger.info("配置集成测试完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()