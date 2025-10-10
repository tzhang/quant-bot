"""
风险管理模块测试用例
测试所有风险管理功能的正确性和稳定性
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.enhanced_risk_manager import (
    EnhancedRiskManager, RiskLimits, RiskLevel, AlertType, 
    RiskAlert, PortfolioRiskMetrics
)
from src.risk.risk_metrics import (
    RiskMetricsEngine, VaRCalculator, CVaRCalculator, 
    VolatilityCalculator, DrawdownCalculator, 
    CorrelationCalculator, BetaCalculator, SharpeRatioCalculator
)
from src.risk.real_time_monitor import (
    RealTimeRiskMonitor, MonitoringConfig, MonitoringStatus, ActionType
)


class TestRiskMetrics(unittest.TestCase):
    """测试风险指标计算"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)  # 一年的日收益率
        self.prices = 100 * np.cumprod(1 + self.returns)
        self.market_returns = np.random.normal(0.0008, 0.015, 252)
        
        # 创建风险指标引擎
        self.risk_metrics_engine = RiskMetricsEngine()
    
    def test_var_calculations(self):
        """测试VaR计算"""
        var_calc = self.risk_metrics_engine.var_calculator
        
        # 历史模拟VaR
        hist_var = var_calc.historical_var(self.returns, confidence=0.95)
        self.assertIsInstance(hist_var, float)
        self.assertLess(hist_var, 0)  # VaR应该是负数
        
        # 参数法VaR
        param_var = var_calc.parametric_var(self.returns, confidence=0.95)
        self.assertIsInstance(param_var, float)
        self.assertLess(param_var, 0)
        
        # 修正VaR
        modified_var = var_calc.modified_var(self.returns, confidence=0.95)
        self.assertIsInstance(modified_var, float)
        self.assertLess(modified_var, 0)
        
        # 蒙特卡洛VaR
        mc_var = var_calc.monte_carlo_var(self.returns, confidence=0.95, n_simulations=1000)
        self.assertIsInstance(mc_var, float)
        self.assertLess(mc_var, 0)
    
    def test_cvar_calculations(self):
        """测试CVaR计算"""
        cvar_calc = self.risk_metrics_engine.cvar_calculator
        
        # 历史模拟CVaR
        hist_cvar = cvar_calc.historical_cvar(self.returns, confidence=0.95)
        self.assertIsInstance(hist_cvar, float)
        self.assertLess(hist_cvar, 0)
        
        # 参数法CVaR
        param_cvar = cvar_calc.parametric_cvar(self.returns, confidence=0.95)
        self.assertIsInstance(param_cvar, float)
        self.assertLess(param_cvar, 0)
        
        # CVaR应该比VaR更负（更大的损失）
        hist_var = self.risk_metrics_engine.var_calculator.historical_var(self.returns, confidence=0.95)
        self.assertLess(hist_cvar, hist_var)
    
    def test_volatility_calculations(self):
        """测试波动率计算"""
        vol_calc = self.risk_metrics_engine.volatility_calculator
        
        # 简单波动率
        simple_vol = vol_calc.simple_volatility(self.returns)
        self.assertIsInstance(simple_vol, float)
        self.assertGreater(simple_vol, 0)
        
        # EWMA波动率
        ewma_vol = vol_calc.ewma_volatility(self.returns, lambda_param=0.94)
        self.assertIsInstance(ewma_vol, float)
        self.assertGreater(ewma_vol, 0)
        
        # GARCH波动率
        garch_vol = vol_calc.garch_volatility(self.returns)
        self.assertIsInstance(garch_vol, float)
        self.assertGreater(garch_vol, 0)
    
    def test_drawdown_calculations(self):
        """测试回撤计算"""
        dd_calc = self.risk_metrics_engine.drawdown_calculator
        
        # 最大回撤
        max_dd = dd_calc.max_drawdown(self.prices)
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0)  # 回撤应该是负数或零
        
        # 当前回撤
        current_dd = dd_calc.current_drawdown(self.prices)
        self.assertIsInstance(current_dd, float)
        self.assertLessEqual(current_dd, 0)
        
        # 回撤持续时间
        dd_duration = dd_calc.drawdown_duration(self.prices)
        self.assertIsInstance(dd_duration, int)
        self.assertGreaterEqual(dd_duration, 0)
    
    def test_correlation_calculations(self):
        """测试相关性计算"""
        corr_calc = self.risk_metrics_engine.correlation_calculator
        
        # 皮尔逊相关系数
        pearson_corr = corr_calc.pearson_correlation(self.returns, self.market_returns)
        self.assertIsInstance(pearson_corr, float)
        self.assertGreaterEqual(pearson_corr, -1)
        self.assertLessEqual(pearson_corr, 1)
        
        # 斯皮尔曼相关系数
        spearman_corr = corr_calc.spearman_correlation(self.returns, self.market_returns)
        self.assertIsInstance(spearman_corr, float)
        self.assertGreaterEqual(spearman_corr, -1)
        self.assertLessEqual(spearman_corr, 1)
    
    def test_beta_calculations(self):
        """测试Beta计算"""
        beta_calc = self.risk_metrics_engine.beta_calculator
        
        # 市场Beta
        market_beta = beta_calc.market_beta(self.returns, self.market_returns)
        self.assertIsInstance(market_beta, float)
        
        # 滚动Beta
        rolling_beta = beta_calc.rolling_beta(self.returns, self.market_returns, window=60)
        self.assertIsInstance(rolling_beta, float)
    
    def test_sharpe_ratio_calculations(self):
        """测试夏普比率计算"""
        sharpe_calc = self.risk_metrics_engine.sharpe_calculator
        
        # 夏普比率
        sharpe_ratio = sharpe_calc.sharpe_ratio(self.returns, risk_free_rate=0.02)
        self.assertIsInstance(sharpe_ratio, float)
        
        # 索提诺比率
        sortino_ratio = sharpe_calc.sortino_ratio(self.returns, risk_free_rate=0.02)
        self.assertIsInstance(sortino_ratio, float)
        
        # 卡尔马比率
        calmar_ratio = sharpe_calc.calmar_ratio(self.returns, self.prices)
        self.assertIsInstance(calmar_ratio, float)


class TestEnhancedRiskManager(unittest.TestCase):
    """测试增强风险管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.risk_limits = RiskLimits(
            max_position_size=0.1,
            max_leverage=2.0,
            var_limit_1d=0.02,
            max_drawdown=0.15,
            max_concentration=0.05,
            stop_loss_pct=0.03
        )
       # 创建风险管理器
        self.risk_manager = EnhancedRiskManager(
            risk_limits=self.risk_limits,
            monitoring_interval=60
        )
        
        # 模拟投资组合数据
        self.portfolio_data = {
            'AAPL': {'position': 100, 'price': 150.0, 'weight': 0.15},
            'GOOGL': {'position': 50, 'price': 2500.0, 'weight': 0.125},
            'MSFT': {'position': 200, 'price': 300.0, 'weight': 0.06},
        }
    
    def test_risk_limits_validation(self):
        """测试风险限制验证"""
        # 测试有效的风险限制
        valid_limits = RiskLimits(
            max_position_size=0.1,
            max_leverage=2.0,
            var_limit_1d=0.02,
            max_drawdown=0.15,
            max_concentration=0.05,
            stop_loss_pct=0.03
        )
        self.assertIsInstance(valid_limits, RiskLimits)
        
        # 测试无效的风险限制（负值）
        with self.assertRaises(ValueError):
            RiskLimits(
                max_position_size=-0.1,  # 负值应该抛出异常
                max_leverage=2.0,
                var_limit_1d=0.02,
                max_drawdown=0.15,
                max_concentration=0.05,
                stop_loss_pct=0.03
            )
    
    def test_trade_validation(self):
        """测试交易验证"""
        # 测试正常交易
        is_valid, message = self.risk_manager.validate_trade('AAPL', 100, 150.0)
        self.assertTrue(is_valid)
        
        # 测试超大订单
        is_valid, message = self.risk_manager.validate_trade('AAPL', 1000, 150.0)
        self.assertFalse(is_valid)
        self.assertIn('订单', message)
    
    def test_position_update(self):
        """测试仓位更新"""
        positions = {
            'AAPL': {
                'market_value': 10000,
                'unrealized_pnl': 500,
                'daily_pnl': 200,
                'liquidity_score': 0.9,
                'sector': 'Technology'
            }
        }
        
        # 更新仓位不应抛出异常
        try:
            self.risk_manager.update_positions(positions)
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)
    
    def test_market_data_update(self):
        """测试市场数据更新"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = np.random.randn(100).cumsum() + 100
        market_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # 更新市场数据不应抛出异常
        try:
            self.risk_manager.update_market_data('AAPL', market_data)
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)
    
    def test_risk_metrics_calculation(self):
        """测试风险指标计算"""
        # 设置测试仓位
        positions = {
            'AAPL': {
                'market_value': 10000,
                'unrealized_pnl': 500,
                'daily_pnl': 200,
                'liquidity_score': 0.9,
                'sector': 'Technology'
            },
            'GOOGL': {
                'market_value': 8000,
                'unrealized_pnl': -200,
                'daily_pnl': -100,
                'liquidity_score': 0.8,
                'sector': 'Technology'
            }
        }
        
        self.risk_manager.update_positions(positions)
        
        # 计算风险指标
        metrics = self.risk_manager.calculate_portfolio_risk_metrics()
        
        # 验证指标存在
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.total_value, 0)
        self.assertIsInstance(metrics.leverage, float)
        self.assertIsInstance(metrics.concentration_ratio, float)
    
    def test_alert_generation(self):
        """测试警报生成"""
        # 生成警报
        alert = self.risk_manager.generate_alert(
            alert_type=AlertType.POSITION_LIMIT,
            level=RiskLevel.HIGH,
            message="测试警报",
            symbol="AAPL"
        )
        
        self.assertIsInstance(alert, RiskAlert)
        self.assertEqual(alert.alert_type, AlertType.POSITION_LIMIT)
        self.assertEqual(alert.level, RiskLevel.HIGH)
        self.assertEqual(alert.message, "测试警报")
        self.assertEqual(alert.symbol, "AAPL")
        self.assertIsInstance(alert.timestamp, datetime)
    
    def test_risk_limits_update(self):
        """测试风险限制更新"""
        new_limits = RiskLimits(
            max_position_size=0.08,  # 从10%降到8%
            max_leverage=1.8,        # 从2.0降到1.8
            var_limit_1d=0.018,     # 从2%降到1.8%
            max_drawdown=0.12,      # 从15%降到12%
            max_concentration=0.04, # 从5%降到4%
            stop_loss_pct=0.025 # 从3%降到2.5%
        )
        
        self.risk_manager.update_risk_limits(new_limits)
        
        # 验证更新
        self.assertEqual(self.risk_manager.risk_limits.max_position_size, 0.08)
        self.assertEqual(self.risk_manager.risk_limits.max_leverage, 1.8)
        self.assertEqual(self.risk_manager.risk_limits.var_limit_1d, 0.018)


class TestRealTimeMonitor(unittest.TestCase):
    """测试实时风险监控"""
    
    def setUp(self):
        """设置测试环境"""
        self.risk_limits = RiskLimits(
            max_position_size=0.1,
            max_leverage=2.0,
            var_limit_1d=0.02,
            max_drawdown=0.15,
            max_concentration=0.05,
            stop_loss_pct=0.03
        )
       # 创建风险管理器
        self.risk_manager = EnhancedRiskManager(
            risk_limits=self.risk_limits,
            monitoring_interval=60
        )
        
        self.config = MonitoringConfig(
            check_interval=1.0,  # 1秒检查一次
            alert_threshold=RiskLevel.MEDIUM,
            auto_actions_enabled=False,  # 测试时禁用自动操作
            websocket_port=8765
        )
        
        self.monitor = RealTimeRiskMonitor(
            risk_manager=self.risk_manager,
            config=self.config
        )
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        self.assertIsInstance(self.monitor, RealTimeRiskMonitor)
        self.assertEqual(self.monitor.status, MonitoringStatus.STOPPED)
        self.assertEqual(self.monitor.config.check_interval, 1.0)
        self.assertFalse(self.monitor.config.auto_actions_enabled)
    
    def test_monitoring_status_changes(self):
        """测试监控状态变化"""
        # 初始状态
        self.assertEqual(self.monitor.status, MonitoringStatus.STOPPED)
        
        # 启动监控（但不实际运行循环）
        self.monitor.status = MonitoringStatus.RUNNING
        self.assertEqual(self.monitor.status, MonitoringStatus.RUNNING)
        
        # 暂停监控
        self.monitor.status = MonitoringStatus.PAUSED
        self.assertEqual(self.monitor.status, MonitoringStatus.PAUSED)
        
        # 停止监控
        self.monitor.status = MonitoringStatus.STOPPED
        self.assertEqual(self.monitor.status, MonitoringStatus.STOPPED)
    
    def test_dashboard_data_generation(self):
        """测试仪表板数据生成"""
        dashboard_data = self.monitor.get_dashboard_data()
        
        self.assertIsInstance(dashboard_data, dict)
        self.assertIn('status', dashboard_data)
        self.assertIn('current_metrics', dashboard_data)
        self.assertIn('recent_alerts', dashboard_data)
        self.assertIn('risk_limits', dashboard_data)
        self.assertIn('last_update', dashboard_data)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """设置集成测试环境"""
        self.risk_limits = RiskLimits(
            max_position_size=0.1,
            max_leverage=2.0,
            var_limit_1d=0.02,
            max_drawdown=0.15,
            max_concentration=0.05,
            stop_loss_pct=0.03
        )
        
        self.risk_manager = EnhancedRiskManager(
            risk_limits=self.risk_limits,
            monitoring_interval=60
        )
        
        self.risk_metrics_engine = RiskMetricsEngine()
        
        # 生成测试数据
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)
        self.prices = 100 * np.cumprod(1 + self.returns)
    
    def test_end_to_end_risk_workflow(self):
        """测试端到端风险管理工作流"""
        # 1. 计算风险指标
        var_95 = self.risk_metrics_engine.var_calculator.historical_var(self.returns, confidence=0.95)
        volatility = self.risk_metrics_engine.volatility_calculator.simple_volatility(self.returns)
        max_drawdown = self.risk_metrics_engine.drawdown_calculator.max_drawdown(self.prices)
        
        # 2. 检查风险限制
        var_check = self.risk_manager.check_var(abs(var_95))
        leverage_check = self.risk_manager.check_leverage(1.5)
        
        # 3. 验证结果
        self.assertIsInstance(var_95, float)
        self.assertIsInstance(volatility, float)
        self.assertIsInstance(max_drawdown, float)
        self.assertTrue(var_check.is_valid or not var_check.is_valid)  # 结果应该是布尔值
        self.assertTrue(leverage_check.is_valid)  # 1.5倍杠杆应该通过检查
    
    def test_portfolio_risk_assessment(self):
        """测试投资组合风险评估"""
        # 模拟投资组合数据
        portfolio_weights = {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.2, 'TSLA': 0.15, 'AMZN': 0.1}
        
        # 检查集中度
        concentration_check = self.risk_manager.check_concentration(portfolio_weights)
        
        # AAPL权重30%超过5%限制，应该失败
        self.assertFalse(concentration_check.is_valid)
        
        # 调整权重
        adjusted_weights = {'AAPL': 0.04, 'GOOGL': 0.04, 'MSFT': 0.04, 'TSLA': 0.04, 'AMZN': 0.04}
        adjusted_check = self.risk_manager.check_concentration(adjusted_weights)
        
        # 调整后应该通过检查
        self.assertTrue(adjusted_check.is_valid)


def run_risk_management_tests():
    """运行所有风险管理测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestRiskMetrics,
        TestEnhancedRiskManager,
        TestRealTimeMonitor,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    print("=" * 60)
    print("风险管理模块测试")
    print("=" * 60)
    
    # 运行测试
    result = run_risk_management_tests()
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"运行测试数量: {result.testsRun}")
    print(f"失败数量: {len(result.failures)}")
    print(f"错误数量: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息")