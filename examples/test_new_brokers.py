#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新券商功能测试脚本
测试TD Ameritrade、Charles Schwab、E*TRADE和Robinhood适配器
"""

import logging
import time
from typing import Dict, Any, List
from broker_factory import BrokerFactory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_broker_creation():
    """测试新券商适配器创建"""
    logger.info("开始测试新券商适配器创建...")
    
    # 测试TD Ameritrade
    try:
        td_broker = BrokerFactory.create_broker(
            'td_ameritrade',
            consumer_key='test_key',
            consumer_secret='test_secret',
            access_token='test_token',
            access_secret='test_secret',
            sandbox=True,
            dry_run=True
        )
        logger.info("✓ TD Ameritrade 适配器创建成功")
        logger.info(f"  券商名称: {td_broker.broker_name}")
    except Exception as e:
        logger.error(f"✗ TD Ameritrade 适配器创建失败: {e}")
    
    # 测试Charles Schwab
    try:
        schwab_broker = BrokerFactory.create_broker(
            'charles_schwab',
            app_key='test_key',
            app_secret='test_secret',
            access_token='test_token',
            refresh_token='test_refresh',
            sandbox=True,
            dry_run=True
        )
        logger.info("✓ Charles Schwab 适配器创建成功")
        logger.info(f"  券商名称: {schwab_broker.broker_name}")
    except Exception as e:
        logger.error(f"✗ Charles Schwab 适配器创建失败: {e}")
    
    # 测试E*TRADE
    try:
        etrade_broker = BrokerFactory.create_broker(
            'etrade',
            consumer_key='test_key',
            consumer_secret='test_secret',
            access_token='test_token',
            access_secret='test_secret',
            sandbox=True,
            dry_run=True
        )
        logger.info("✓ E*TRADE 适配器创建成功")
        logger.info(f"  券商名称: {etrade_broker.broker_name}")
    except Exception as e:
        logger.error(f"✗ E*TRADE 适配器创建失败: {e}")
    
    # 测试Robinhood
    try:
        robinhood_broker = BrokerFactory.create_broker(
            'robinhood',
            username='test_user',
            password='test_pass',
            device_token='test_device',
            challenge_type='sms',
            sandbox=True,
            dry_run=True
        )
        logger.info("✓ Robinhood 适配器创建成功")
        logger.info(f"  券商名称: {robinhood_broker.broker_name}")
    except Exception as e:
        logger.error(f"✗ Robinhood 适配器创建失败: {e}")

def test_broker_interface():
    """测试券商接口功能"""
    logger.info("开始测试券商接口功能...")
    
    # 创建测试券商
    test_brokers = []
    
    try:
        # TD Ameritrade
        td_broker = BrokerFactory.create_broker(
            'td_ameritrade',
            consumer_key='test_key',
            consumer_secret='test_secret',
            dry_run=True
        )
        test_brokers.append(('TD Ameritrade', td_broker))
        
        # Charles Schwab
        schwab_broker = BrokerFactory.create_broker(
            'schwab',
            app_key='test_key',
            app_secret='test_secret',
            dry_run=True
        )
        test_brokers.append(('Charles Schwab', schwab_broker))
        
        # E*TRADE
        etrade_broker = BrokerFactory.create_broker(
            'etrade',
            consumer_key='test_key',
            consumer_secret='test_secret',
            dry_run=True
        )
        test_brokers.append(('E*TRADE', etrade_broker))
        
        # Robinhood
        robinhood_broker = BrokerFactory.create_broker(
            'rh',
            username='test_user',
            password='test_pass',
            dry_run=True
        )
        test_brokers.append(('Robinhood', robinhood_broker))
        
    except Exception as e:
        logger.error(f"创建测试券商失败: {e}")
        return
    
    # 测试每个券商的接口
    for broker_name, broker in test_brokers:
        logger.info(f"\n测试 {broker_name} 接口...")
        
        try:
            # 测试连接
            connected = broker.connect()
            logger.info(f"  连接状态: {'成功' if connected else '失败'}")
            
            # 测试连接检查
            is_connected = broker.is_connected()
            logger.info(f"  连接检查: {'已连接' if is_connected else '未连接'}")
            
            if is_connected:
                # 测试投资组合状态
                portfolio = broker.get_portfolio_status()
                logger.info(f"  投资组合价值: ${portfolio.get('total_value', 0):,.2f}")
                
                # 测试持仓
                positions = broker.get_positions()
                logger.info(f"  持仓数量: {len(positions) if positions else 0}")
                
                # 测试交易表现
                performance = broker.get_performance()
                logger.info(f"  总交易次数: {performance.get('total_trades', 0)}")
                
                # 测试详细持仓
                detailed_positions = broker.get_detailed_positions()
                logger.info(f"  详细持仓数量: {len(detailed_positions) if detailed_positions else 0}")
                
                # 测试投资组合表现计算
                portfolio_perf = broker.calculate_portfolio_performance(30)
                logger.info(f"  30天收益率: {portfolio_perf.get('total_return_percent', 0):.2f}%")
            
            # 断开连接
            broker.disconnect()
            logger.info(f"  {broker_name} 测试完成")
            
        except Exception as e:
            logger.error(f"  {broker_name} 测试失败: {e}")

def test_broker_factory_integration():
    """测试券商工厂集成"""
    logger.info("开始测试券商工厂集成...")
    
    try:
        # 创建券商工厂实例
        factory = BrokerFactory()
        logger.info("✓ 券商工厂创建成功")
        
        # 测试支持的券商类型
        supported_brokers = [
            'firstrade', 'alpaca', 'interactive_brokers',
            'td_ameritrade', 'charles_schwab', 'etrade', 'robinhood'
        ]
        
        logger.info(f"支持的券商类型: {', '.join(supported_brokers)}")
        
        # 测试创建不同券商
        for broker_type in ['td_ameritrade', 'charles_schwab', 'etrade', 'robinhood']:
            try:
                if broker_type == 'td_ameritrade':
                    broker = BrokerFactory.create_broker(
                        broker_type,
                        consumer_key='test',
                        consumer_secret='test',
                        dry_run=True
                    )
                elif broker_type == 'charles_schwab':
                    broker = BrokerFactory.create_broker(
                        broker_type,
                        app_key='test',
                        app_secret='test',
                        dry_run=True
                    )
                elif broker_type == 'etrade':
                    broker = BrokerFactory.create_broker(
                        broker_type,
                        consumer_key='test',
                        consumer_secret='test',
                        dry_run=True
                    )
                elif broker_type == 'robinhood':
                    broker = BrokerFactory.create_broker(
                        broker_type,
                        username='test',
                        password='test',
                        dry_run=True
                    )
                
                logger.info(f"✓ {broker_type} 券商创建成功")
                
            except Exception as e:
                logger.error(f"✗ {broker_type} 券商创建失败: {e}")
        
        logger.info("券商工厂集成测试完成")
        
    except Exception as e:
        logger.error(f"券商工厂集成测试失败: {e}")

def test_error_handling():
    """测试错误处理"""
    logger.info("开始测试错误处理...")
    
    # 测试不支持的券商类型
    try:
        BrokerFactory.create_broker('unsupported_broker')
        logger.error("✗ 应该抛出不支持券商类型的异常")
    except ValueError as e:
        logger.info(f"✓ 正确处理不支持的券商类型: {e}")
    except Exception as e:
        logger.error(f"✗ 意外的异常类型: {e}")
    
    # 测试缺少必要参数
    try:
        BrokerFactory.create_broker('td_ameritrade')  # 缺少必要参数
        logger.info("✓ 使用默认参数创建券商")
    except Exception as e:
        logger.info(f"✓ 正确处理缺少参数的情况: {e}")

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("新券商功能测试开始")
    logger.info("=" * 60)
    
    # 运行所有测试
    test_broker_creation()
    print()
    
    test_broker_interface()
    print()
    
    test_broker_factory_integration()
    print()
    
    test_error_handling()
    print()
    
    logger.info("=" * 60)
    logger.info("新券商功能测试完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()