#!/usr/bin/env python3
"""
Interactive Brokers API 集成测试程序
测试所有IB API相关组件的功能和集成
"""

import unittest
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 导入测试模块
from enhanced_ib_trading_system import EnhancedIBTradingSystem, TradingMode, TradingConfig
from ib_risk_manager import IBRiskManager, RiskLimit, RiskLevel
from ib_order_manager import IBOrderManager, OrderRequest, OrderType, TimeInForce
from test_ib_connection import test_connection

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IBConnectionTest(unittest.TestCase):
    """IB连接测试"""
    
    def test_basic_connection(self):
        """测试基本连接"""
        logger.info("测试IB基本连接...")
        result = test_connection()
        self.assertTrue(result, "IB连接测试失败")
        logger.info("✓ IB基本连接测试通过")

class IBTradingSystemTest(unittest.TestCase):
    """IB交易系统测试"""
    
    def setUp(self):
        """测试设置"""
        self.config = TradingConfig(
            host="127.0.0.1",
            port=7497,
            client_id=10,
            mode=TradingMode.PAPER,
            enable_risk_management=True
        )
        self.trading_system = EnhancedIBTradingSystem(self.config)
    
    def tearDown(self):
        """测试清理"""
        if self.trading_system:
            self.trading_system.disconnect()
    
    def test_connection(self):
        """测试交易系统连接"""
        logger.info("测试交易系统连接...")
        result = self.trading_system.connect()
        self.assertTrue(result, "交易系统连接失败")
        
        # 等待连接稳定
        time.sleep(3)
        
        # 测试连接状态
        self.assertTrue(self.trading_system.is_connected(), "连接状态检查失败")
        logger.info("✓ 交易系统连接测试通过")
    
    def test_account_info(self):
        """测试账户信息获取"""
        logger.info("测试账户信息获取...")
        
        if not self.trading_system.connect():
            self.skipTest("无法连接到IB")
        
        time.sleep(3)
        
        # 获取账户信息
        account_info = self.trading_system.get_account_info()
        self.assertIsNotNone(account_info, "账户信息获取失败")
        
        # 检查必要字段
        self.assertIn('NetLiquidation', account_info, "缺少净清算值")
        self.assertIn('TotalCashValue', account_info, "缺少现金价值")
        
        logger.info(f"✓ 账户信息获取成功: 净值=${account_info.get('NetLiquidation', 0)}")
    
    def test_market_data_subscription(self):
        """测试市场数据订阅"""
        logger.info("测试市场数据订阅...")
        
        if not self.trading_system.connect():
            self.skipTest("无法连接到IB")
        
        time.sleep(3)
        
        # 订阅市场数据
        symbol = "AAPL"
        result = self.trading_system.subscribe_market_data(symbol)
        self.assertTrue(result, f"订阅{symbol}市场数据失败")
        
        # 等待数据更新
        time.sleep(5)
        
        # 获取市场数据
        market_data = self.trading_system.get_market_data(symbol)
        # 注意：在测试环境中可能无法获取实时数据
        logger.info(f"✓ 市场数据订阅测试完成: {symbol}")

class IBRiskManagerTest(unittest.TestCase):
    """IB风险管理器测试"""
    
    def setUp(self):
        """测试设置"""
        self.risk_limits = RiskLimit(
            max_position_value=10000.0,
            max_symbol_exposure=5000.0,
            max_daily_trades=10,
            max_order_size=100,
            max_daily_loss=1000.0
        )
        self.risk_manager = IBRiskManager(self.risk_limits)
    
    def test_risk_limit_validation(self):
        """测试风险限制验证"""
        logger.info("测试风险限制验证...")
        
        # 测试正常订单
        allow, alerts = self.risk_manager.check_order_risk("AAPL", "BUY", 50, 150.0)
        self.assertTrue(allow, "正常订单应该被允许")
        self.assertEqual(len(alerts), 0, "正常订单不应有警报")
        
        # 测试超限订单
        allow, alerts = self.risk_manager.check_order_risk("AAPL", "BUY", 200, 150.0)
        self.assertFalse(allow, "超限订单应该被拒绝")
        self.assertGreater(len(alerts), 0, "超限订单应该有警报")
        
        logger.info("✓ 风险限制验证测试通过")
    
    def test_position_tracking(self):
        """测试持仓跟踪"""
        logger.info("测试持仓跟踪...")
        
        # 模拟持仓更新
        self.risk_manager.update_position("AAPL", 100, 150.0)
        self.risk_manager.update_position("MSFT", 50, 300.0)
        
        # 获取风险摘要
        risk_summary = self.risk_manager.get_risk_summary()
        metrics = risk_summary['metrics']
        
        # 验证持仓计算
        expected_value = 100 * 150.0 + 50 * 300.0  # 30000
        self.assertAlmostEqual(metrics.total_position_value, expected_value, places=2)
        
        logger.info(f"✓ 持仓跟踪测试通过: 总持仓价值=${metrics.total_position_value}")
    
    def test_risk_monitoring(self):
        """测试风险监控"""
        logger.info("测试风险监控...")
        
        # 启动监控
        self.risk_manager.start_monitoring()
        
        # 等待监控启动
        time.sleep(2)
        
        # 检查监控状态
        self.assertTrue(self.risk_manager.is_monitoring, "风险监控应该已启动")
        
        # 停止监控
        self.risk_manager.stop_monitoring()
        
        # 等待监控停止
        time.sleep(1)
        
        self.assertFalse(self.risk_manager.is_monitoring, "风险监控应该已停止")
        
        logger.info("✓ 风险监控测试通过")

class IBOrderManagerTest(unittest.TestCase):
    """IB订单管理器测试"""
    
    def setUp(self):
        """测试设置"""
        self.order_manager = IBOrderManager(
            host="127.0.0.1",
            port=7497,
            client_id=11
        )
        
        # 设置风险管理器
        risk_limits = RiskLimit(max_order_size=1000, max_daily_trades=100)
        self.risk_manager = IBRiskManager(risk_limits)
        self.order_manager.set_risk_manager(self.risk_manager)
    
    def tearDown(self):
        """测试清理"""
        if self.order_manager:
            self.order_manager.disconnect_from_ib()
    
    def test_connection(self):
        """测试订单管理器连接"""
        logger.info("测试订单管理器连接...")
        
        result = self.order_manager.connect_to_ib()
        self.assertTrue(result, "订单管理器连接失败")
        
        # 等待连接稳定
        time.sleep(3)
        
        logger.info("✓ 订单管理器连接测试通过")
    
    def test_order_creation(self):
        """测试订单创建"""
        logger.info("测试订单创建...")
        
        # 创建市价订单
        market_order = OrderRequest(
            symbol="AAPL",
            action="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
            strategy_id="test_strategy"
        )
        
        # 验证订单属性
        self.assertEqual(market_order.symbol, "AAPL")
        self.assertEqual(market_order.action, "BUY")
        self.assertEqual(market_order.quantity, 10)
        self.assertEqual(market_order.order_type, OrderType.MARKET)
        
        # 创建限价订单
        limit_order = OrderRequest(
            symbol="MSFT",
            action="SELL",
            quantity=5,
            order_type=OrderType.LIMIT,
            price=350.0,
            time_in_force=TimeInForce.DAY
        )
        
        self.assertEqual(limit_order.price, 350.0)
        self.assertEqual(limit_order.time_in_force, TimeInForce.DAY)
        
        logger.info("✓ 订单创建测试通过")
    
    def test_order_validation(self):
        """测试订单验证"""
        logger.info("测试订单验证...")
        
        # 有效订单
        valid_order = OrderRequest(
            symbol="AAPL",
            action="BUY",
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        is_valid, error = self.order_manager._validate_order(valid_order)
        self.assertTrue(is_valid, f"有效订单验证失败: {error}")
        
        # 无效订单 - 缺少价格的限价订单
        invalid_order = OrderRequest(
            symbol="AAPL",
            action="BUY",
            quantity=10,
            order_type=OrderType.LIMIT
            # 缺少price
        )
        
        is_valid, error = self.order_manager._validate_order(invalid_order)
        self.assertFalse(is_valid, "无效订单应该验证失败")
        self.assertIsNotNone(error, "应该有错误信息")
        
        logger.info("✓ 订单验证测试通过")

class IBIntegrationTest(unittest.TestCase):
    """IB集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.config = TradingConfig(
            host="127.0.0.1",
            port=7497,
            client_id=12,
            mode=TradingMode.PAPER
        )
        
        # 创建组件
        self.trading_system = EnhancedIBTradingSystem(self.config)
        
        risk_limits = RiskLimit(
            max_position_value=50000.0,
            max_order_size=500
        )
        self.risk_manager = IBRiskManager(risk_limits)
        
        self.order_manager = IBOrderManager(
            host=self.config.host,
            port=self.config.port,
            client_id=13
        )
        self.order_manager.set_risk_manager(self.risk_manager)
    
    def tearDown(self):
        """测试清理"""
        if self.trading_system:
            self.trading_system.disconnect()
        if self.order_manager:
            self.order_manager.disconnect_from_ib()
    
    def test_full_integration(self):
        """测试完整集成"""
        logger.info("测试完整集成...")
        
        # 连接所有组件
        trading_connected = self.trading_system.connect()
        order_connected = self.order_manager.connect_to_ib()
        
        if not (trading_connected and order_connected):
            self.skipTest("无法建立完整连接")
        
        # 等待连接稳定
        time.sleep(5)
        
        # 启动风险监控
        self.risk_manager.start_monitoring()
        
        # 订阅市场数据
        symbol = "AAPL"
        self.trading_system.subscribe_market_data(symbol)
        
        # 等待数据
        time.sleep(3)
        
        # 创建测试订单（小量，避免实际执行）
        test_order = OrderRequest(
            symbol=symbol,
            action="BUY",
            quantity=1,  # 最小数量
            order_type=OrderType.LIMIT,
            price=1.0,  # 极低价格，不会成交
            strategy_id="integration_test"
        )
        
        # 检查风险
        allow, alerts = self.risk_manager.check_order_risk(
            test_order.symbol, test_order.action, test_order.quantity, test_order.price
        )
        
        self.assertTrue(allow, "测试订单应该通过风险检查")
        
        # 提交订单（但不会成交）
        order_id = self.order_manager.submit_order(test_order)
        
        if order_id:
            logger.info(f"测试订单已提交: {order_id}")
            
            # 等待订单状态更新
            time.sleep(3)
            
            # 取消订单
            cancelled = self.order_manager.cancel_order(order_id)
            self.assertTrue(cancelled, "订单取消失败")
        
        # 停止监控
        self.risk_manager.stop_monitoring()
        
        logger.info("✓ 完整集成测试通过")

class IBPerformanceTest(unittest.TestCase):
    """IB性能测试"""
    
    def test_connection_performance(self):
        """测试连接性能"""
        logger.info("测试连接性能...")
        
        start_time = time.time()
        
        trading_system = EnhancedIBTradingSystem(TradingConfig(
            host="127.0.0.1",
            port=7497,
            client_id=20,
            mode=TradingMode.PAPER
        ))
        
        connected = trading_system.connect()
        
        if connected:
            connection_time = time.time() - start_time
            logger.info(f"连接时间: {connection_time:.2f}秒")
            
            # 连接时间应该在合理范围内
            self.assertLess(connection_time, 10.0, "连接时间过长")
            
            trading_system.disconnect()
        else:
            self.skipTest("无法连接到IB")
        
        logger.info("✓ 连接性能测试通过")
    
    def test_order_processing_performance(self):
        """测试订单处理性能"""
        logger.info("测试订单处理性能...")
        
        order_manager = IBOrderManager(
            host="127.0.0.1",
            port=7497,
            client_id=21
        )
        
        if not order_manager.connect_to_ib():
            self.skipTest("无法连接到IB")
        
        time.sleep(3)
        
        # 测试订单创建性能
        start_time = time.time()
        
        orders = []
        for i in range(10):
            order = OrderRequest(
                symbol="AAPL",
                action="BUY",
                quantity=1,
                order_type=OrderType.LIMIT,
                price=1.0,  # 不会成交的价格
                strategy_id=f"perf_test_{i}"
            )
            orders.append(order)
        
        creation_time = time.time() - start_time
        logger.info(f"10个订单创建时间: {creation_time:.3f}秒")
        
        # 测试订单验证性能
        start_time = time.time()
        
        for order in orders:
            is_valid, _ = order_manager._validate_order(order)
            self.assertTrue(is_valid)
        
        validation_time = time.time() - start_time
        logger.info(f"10个订单验证时间: {validation_time:.3f}秒")
        
        order_manager.disconnect_from_ib()
        
        logger.info("✓ 订单处理性能测试通过")

def run_all_tests():
    """运行所有测试"""
    logger.info("开始IB API集成测试...")
    logger.info("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        IBConnectionTest,
        IBTradingSystemTest,
        IBRiskManagerTest,
        IBOrderManagerTest,
        IBIntegrationTest,
        IBPerformanceTest
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印结果摘要
    logger.info("=" * 60)
    logger.info("测试结果摘要:")
    logger.info(f"运行测试: {result.testsRun}")
    logger.info(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"失败: {len(result.failures)}")
    logger.info(f"错误: {len(result.errors)}")
    
    if result.failures:
        logger.error("失败的测试:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("错误的测试:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    logger.info(f"成功率: {success_rate:.1f}%")
    logger.info("=" * 60)
    
    return result.wasSuccessful()

def main():
    """主函数"""
    print("Interactive Brokers API 集成测试")
    print("=" * 60)
    print("注意: 请确保TWS或IB Gateway已启动并配置为Paper Trading模式")
    print("默认连接: 127.0.0.1:7497")
    print("=" * 60)
    
    # 等待用户确认
    input("按Enter键开始测试...")
    
    try:
        success = run_all_tests()
        
        if success:
            print("\n✓ 所有测试通过！IB API集成正常工作。")
            return 0
        else:
            print("\n✗ 部分测试失败，请检查配置和连接。")
            return 1
            
    except Exception as e:
        logger.error(f"测试运行错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())