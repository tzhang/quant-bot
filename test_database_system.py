#!/usr/bin/env python3
"""
数据库系统完整测试脚本

该脚本测试数据库系统的所有核心功能：
1. 数据库连接测试
2. 表结构验证
3. CRUD操作测试
4. Redis缓存测试
5. DAO层功能测试
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.connection import db_manager
from src.database.dao import stock_data_dao, strategy_performance_dao, factor_data_dao
from src.database.models import StockData, StrategyPerformance, FactorData

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_connections():
    """测试数据库连接"""
    logger.info("🔍 测试数据库连接...")
    
    results = db_manager.test_connections()
    
    for db_type, success in results.items():
        if success:
            logger.info(f"✅ {db_type.upper()} 连接成功")
        else:
            logger.error(f"❌ {db_type.upper()} 连接失败")
            return False
    
    return all(results.values())


def test_redis_cache():
    """测试Redis缓存功能"""
    logger.info("🔍 测试Redis缓存功能...")
    
    try:
        redis_client = db_manager.get_redis_client()
        
        # 基本缓存测试
        test_key = "test:cache:basic"
        test_value = "Hello Redis Cache!"
        
        redis_client.set(test_key, test_value, ex=60)
        cached_value = redis_client.get(test_key)
        
        if cached_value == test_value:
            logger.info("✅ Redis基本缓存功能正常")
        else:
            logger.error("❌ Redis基本缓存功能异常")
            return False
        
        # 清理测试数据
        redis_client.delete(test_key)
        
        # 测试缓存过期
        redis_client.set("test:expire", "expire_test", ex=1)
        import time
        time.sleep(2)
        expired_value = redis_client.get("test:expire")
        
        if expired_value is None:
            logger.info("✅ Redis缓存过期功能正常")
        else:
            logger.error("❌ Redis缓存过期功能异常")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis缓存测试失败: {e}")
        return False


def test_stock_data_operations():
    """测试股票数据CRUD操作"""
    logger.info("🔍 测试股票数据CRUD操作...")
    
    try:
        # 创建测试数据
        test_data = StockData(
            symbol="TEST",
            date=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000
        )
        
        # 创建股票数据
        created_stock = stock_data_dao.create(
            symbol="TEST",
            date=datetime.now(),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000
        )
        
        logger.info(f"创建的股票数据ID: {created_stock.id}")
        
        # 查询股票数据
        stock_id = created_stock.id
        retrieved_stock = stock_data_dao.get_by_id(stock_id)
        if retrieved_stock is None:
            logger.error("❌ 应该能够查询到创建的股票数据")
            return False
        if retrieved_stock.symbol != "TEST":
            logger.error("❌ 股票代码应该匹配")
            return False
        logger.info("✅ 股票数据查询成功")
        
        # 更新股票数据
        retrieved_stock.close = 103.0
        updated_stock = stock_data_dao.update(retrieved_stock)
        if updated_stock.close != 103.0:
            logger.error("❌ 股票收盘价应该被更新")
            return False
        logger.info("✅ 股票数据更新成功")
        
        # 删除股票数据
        delete_result = stock_data_dao.delete(stock_id)
        if delete_result != True:
            logger.error("❌ 应该能够删除股票数据")
            return False
        logger.info("✅ 股票数据删除成功")
        
        # 验证删除
        deleted_stock = stock_data_dao.get_by_id(stock_id)
        if deleted_stock is not None:
            logger.error("❌ 删除后应该查询不到数据")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 股票数据操作测试失败: {e}")
        return False


def test_strategy_performance_operations():
    """测试策略绩效数据操作"""
    logger.info("🔍 测试策略绩效数据操作...")
    
    try:
        # 创建测试数据
        test_data = StrategyPerformance(
            strategy_name="test_strategy",
            date=datetime.now(),
            returns=0.15,
            cumulative_returns=0.25,
            drawdown=0.05,
            positions='{"AAPL": 100, "MSFT": 50}'
        )
        
        # 创建策略绩效数据
        created_performance = strategy_performance_dao.create(
            strategy_name="test_strategy",
            date=datetime.now(),
            returns=0.05,
            cumulative_returns=0.15,
            drawdown=-0.02,
            positions={"AAPL": 100, "GOOGL": 50}
        )
        
        logger.info(f"创建的策略绩效数据ID: {created_performance.id}")
        
        # 查询策略绩效数据
        performance_id = created_performance.id
        retrieved_performance = strategy_performance_dao.get_by_id(performance_id)
        if retrieved_performance is None:
            logger.error("❌ 应该能够查询到创建的策略绩效数据")
            return False
        if retrieved_performance.strategy_name != "test_strategy":
            logger.error("❌ 策略名称应该匹配")
            return False
        logger.info("✅ 策略绩效数据查询成功")
        
        # 更新策略绩效数据
        retrieved_performance.returns = 0.06
        updated_performance = strategy_performance_dao.update(retrieved_performance)
        if updated_performance.returns != 0.06:
            logger.error("❌ 策略收益率应该被更新")
            return False
        logger.info("✅ 策略绩效数据更新成功")
        
        # 删除策略绩效数据
        delete_result = strategy_performance_dao.delete(performance_id)
        if delete_result != True:
            logger.error("❌ 应该能够删除策略绩效数据")
            return False
        logger.info("✅ 策略绩效数据删除成功")
        
        # 验证删除
        deleted_performance = strategy_performance_dao.get_by_id(performance_id)
        if deleted_performance is not None:
            logger.error("❌ 删除后应该查询不到数据")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 策略绩效数据操作测试失败: {e}")
        return False


def test_factor_data_operations():
    """测试因子数据操作"""
    logger.info("🔍 测试因子数据操作...")
    
    try:
        # 创建测试数据
        test_data = FactorData(
            symbol="TEST",
            date=datetime.now(),
            factor_name="PE_ratio",
            factor_value=15.5
        )
        
        # 创建因子数据
        created_factor = factor_data_dao.create(
            symbol="TEST",
            date=datetime.now(),
            factor_name="PE_ratio",
            factor_value=15.5
        )
        
        logger.info(f"创建的因子数据ID: {created_factor.id}")
        
        # 查询因子数据
        factor_id = created_factor.id
        retrieved_factor = factor_data_dao.get_by_id(factor_id)
        if retrieved_factor is None:
            logger.error("❌ 应该能够查询到创建的因子数据")
            return False
        if retrieved_factor.factor_name != "PE_ratio":
            logger.error("❌ 因子名称应该匹配")
            return False
        logger.info("✅ 因子数据查询成功")
        
        # 更新因子数据
        retrieved_factor.factor_value = 16.0
        updated_factor = factor_data_dao.update(retrieved_factor)
        if updated_factor.factor_value != 16.0:
            logger.error("❌ 因子值应该被更新")
            return False
        logger.info("✅ 因子数据更新成功")
        
        # 删除因子数据
        delete_result = factor_data_dao.delete(factor_id)
        if delete_result != True:
            logger.error("❌ 应该能够删除因子数据")
            return False
        logger.info("✅ 因子数据删除成功")
        
        # 验证删除
        deleted_factor = factor_data_dao.get_by_id(factor_id)
        if deleted_factor is not None:
            logger.error("❌ 删除后应该查询不到数据")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 因子数据操作测试失败: {e}")
        return False


def test_dao_cache_functionality():
    """测试DAO层缓存功能"""
    logger.info("🔍 测试DAO层缓存功能...")
    
    try:
        # 创建测试数据
        created_data = stock_data_dao.create(
            symbol="CACHE_TEST",
            date=datetime.now(),
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000
        )
        
        # 测试缓存功能
        stock_id = created_data.id
        
        # 第一次查询（应该从数据库获取）
        start_time = datetime.now()
        data1 = stock_data_dao.get_by_id(stock_id)
        first_query_time = (datetime.now() - start_time).total_seconds()
        
        # 第二次查询（应该从缓存获取）
        start_time = datetime.now()
        data2 = stock_data_dao.get_by_id(stock_id)
        second_query_time = (datetime.now() - start_time).total_seconds()
        
        if data1 and data2 and data1.symbol == data2.symbol:
            logger.info(f"✅ DAO缓存功能正常 (第一次: {first_query_time:.4f}s, 第二次: {second_query_time:.4f}s)")
        else:
            logger.error("❌ DAO缓存功能异常")
            return False
        
        # 清理测试数据
        stock_data_dao.delete(stock_id)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DAO缓存功能测试失败: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("🚀 开始数据库系统完整测试...")
    
    # 加载环境变量
    env_file = project_root / '.env'
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        logger.info(f"已加载环境变量文件: {env_file}")
    
    test_results = []
    
    # 执行所有测试
    tests = [
        ("数据库连接测试", test_database_connections),
        ("Redis缓存测试", test_redis_cache),
        ("股票数据CRUD测试", test_stock_data_operations),
        ("策略绩效数据测试", test_strategy_performance_operations),
        ("因子数据测试", test_factor_data_operations),
        ("DAO缓存功能测试", test_dao_cache_functionality),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name}执行异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试结果
    logger.info("\n" + "="*50)
    logger.info("📊 测试结果汇总:")
    logger.info("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("="*50)
    logger.info(f"总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！数据库系统运行正常")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    try:
        success = main()
    finally:
        # 关闭数据库连接
        db_manager.close_connections()
    
    sys.exit(0 if success else 1)