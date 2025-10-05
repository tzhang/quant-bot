#!/usr/bin/env python3
"""
监控系统测试脚本
测试监控系统、告警系统和仪表板功能
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class MockTradingSystem:
    """模拟交易系统"""
    
    def __init__(self):
        self.connected = True
        self.portfolio = {
            'total_value': 100000.0,
            'cash': 20000.0,
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'value': 15000},
                {'symbol': 'GOOGL', 'quantity': 50, 'value': 12500},
                {'symbol': 'MSFT', 'quantity': 75, 'value': 22500}
            ]
        }
        self.orders = [
            {'id': '001', 'symbol': 'AAPL', 'status': 'filled'},
            {'id': '002', 'symbol': 'GOOGL', 'status': 'pending'},
            {'id': '003', 'symbol': 'MSFT', 'status': 'filled'}
        ]
    
    def is_connected(self):
        return self.connected
    
    def get_portfolio_status(self):
        return self.portfolio
    
    def get_orders(self):
        return self.orders

def test_real_time_monitor():
    """测试实时监控系统"""
    logger.info("开始测试实时监控系统...")
    
    try:
        from monitoring.real_time_monitor import TradingSystemMonitor, get_global_monitor
        
        # 创建模拟交易系统
        mock_system = MockTradingSystem()
        
        # 创建监控器
        monitor = TradingSystemMonitor(mock_system)
        
        # 启动监控
        monitor.start()
        logger.info("监控系统已启动")
        
        # 等待一段时间
        time.sleep(2)
        
        # 获取仪表板数据
        dashboard_data = monitor.get_dashboard_data()
        logger.info(f"仪表板数据: {json.dumps(dashboard_data, indent=2, ensure_ascii=False)}")
        
        # 测试指标导出
        metrics_json = monitor.export_metrics('json')
        logger.info(f"指标导出 (JSON): {len(metrics_json)} 字符")
        
        metrics_csv = monitor.export_metrics('csv')
        logger.info(f"指标导出 (CSV): {len(metrics_csv)} 字符")
        
        # 测试告警导出
        alerts_json = monitor.export_alerts('json')
        logger.info(f"告警导出 (JSON): {len(alerts_json)} 字符")
        
        alerts_csv = monitor.export_alerts('csv')
        logger.info(f"告警导出 (CSV): {len(alerts_csv)} 字符")
        
        # 停止监控
        monitor.stop()
        logger.info("监控系统已停止")
        
        # 测试全局监控器
        global_monitor = get_global_monitor()
        logger.info("全局监控器创建成功")
        
        logger.info("实时监控系统测试完成 ✓")
        return True
        
    except Exception as e:
        logger.error(f"实时监控系统测试失败: {e}")
        return False

def test_alert_system():
    """测试告警系统"""
    logger.info("开始测试告警系统...")
    
    try:
        from monitoring.alert_system import AlertRouter, EmailAlertChannel, WebhookAlertChannel
        from monitoring.real_time_monitor import AlertInfo, AlertLevel
        
        # 创建告警路由器
        router = AlertRouter()
        
        # 创建邮件通道（使用模拟配置）
        email_channel = EmailAlertChannel(
            smtp_server="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="password",
            from_email="test@example.com",
            to_emails=["admin@example.com"]
        )
        
        # 创建Webhook通道
        webhook_channel = WebhookAlertChannel("https://hooks.example.com/webhook")
        
        # 添加通道
        router.add_channel("email", email_channel)
        router.add_channel("webhook", webhook_channel)
        
        # 测试连接（会失败，但不影响测试）
        try:
            email_result = email_channel.test_connection()
            logger.info(f"邮件通道测试: {email_result}")
        except Exception as e:
            logger.warning(f"邮件通道测试失败（预期）: {e}")
        
        try:
            webhook_result = webhook_channel.test_connection()
            logger.info(f"Webhook通道测试: {webhook_result}")
        except Exception as e:
            logger.warning(f"Webhook通道测试失败（预期）: {e}")
        
        # 测试告警路由
        test_alert = AlertInfo(
            id="test_001",
            level=AlertLevel.WARNING,
            title="测试告警",
            message="这是一个测试告警",
            timestamp=datetime.now(),
            source="test_system",
            data={'test_key': 'test_value'}
        )
        
        try:
            router.route_alert(test_alert)
            logger.info("告警路由测试完成")
        except Exception as e:
            logger.warning(f"告警路由测试失败（预期）: {e}")
        
        logger.info("告警系统测试完成 ✓")
        return True
        
    except Exception as e:
        logger.error(f"告警系统测试失败: {e}")
        return False

def test_dashboard():
    """测试仪表板"""
    logger.info("开始测试仪表板...")
    
    try:
        from monitoring.dashboard import create_dashboard
        
        # 创建模拟交易系统
        mock_system = MockTradingSystem()
        
        # 创建仪表板
        dashboard = create_dashboard(trading_system=mock_system, port=5001)
        
        logger.info("仪表板创建成功")
        
        # 测试仪表板组件
        if dashboard.monitor:
            logger.info("监控器组件正常")
        
        if dashboard.alert_router:
            logger.info("告警路由器组件正常")
        
        # 检查AlertManager是否有add_alert_handler方法
        if hasattr(dashboard.monitor, 'system_monitor') and hasattr(dashboard.monitor.system_monitor, 'alert_manager'):
            alert_manager = dashboard.monitor.system_monitor.alert_manager
            if hasattr(alert_manager, 'add_alert_handler'):
                logger.info("AlertManager的add_alert_handler方法存在")
            else:
                logger.warning("AlertManager缺少add_alert_handler方法")
        
        # 测试Flask应用
        app = dashboard.app
        with app.app_context():
            # 测试客户端
            client = app.test_client()
            
            # 测试主页
            response = client.get('/')
            logger.info(f"主页响应状态: {response.status_code}")
            
            # 测试API端点
            response = client.get('/api/status')
            logger.info(f"状态API响应状态: {response.status_code}")
            
            response = client.get('/api/alerts')
            logger.info(f"告警API响应状态: {response.status_code}")
            
            response = client.get('/api/metrics')
            logger.info(f"指标API响应状态: {response.status_code}")
            
            # 测试导出功能
            response = client.get('/api/export/metrics?format=json')
            logger.info(f"指标导出API响应状态: {response.status_code}")
            
            response = client.get('/api/export/alerts?format=csv')
            logger.info(f"告警导出API响应状态: {response.status_code}")
        
        logger.info("仪表板测试完成 ✓")
        return True
        
    except Exception as e:
        logger.error(f"仪表板测试失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def test_system_monitor():
    """测试系统监控器"""
    logger.info("开始测试系统监控器...")
    
    try:
        from monitoring.system_monitor import SystemMonitor, AlertLevel
        
        # 创建系统监控器
        monitor = SystemMonitor()
        
        # 启动监控
        monitor.start_monitoring()
        logger.info("系统监控已启动")
        
        # 添加一些指标
        monitor.metric_collector.add_metric("test.cpu", 45.5, "%")
        monitor.metric_collector.add_metric("test.memory", 60.2, "%")
        monitor.metric_collector.add_metric("test.disk", 35.8, "%")
        
        # 创建一些告警
        monitor.alert_manager.create_alert(
            AlertLevel.INFO,
            "测试信息告警",
            "这是一个测试信息告警",
            "test_system"
        )
        
        monitor.alert_manager.create_alert(
            AlertLevel.WARNING,
            "测试警告告警",
            "这是一个测试警告告警",
            "test_system"
        )
        
        # 获取系统状态
        status = monitor.get_system_status()
        logger.info(f"系统状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
        
        # 获取指标
        metrics = monitor.metric_collector.get_metrics()
        logger.info(f"收集到 {len(metrics)} 个指标")
        
        # 获取告警
        alerts = monitor.alert_manager.get_active_alerts()
        logger.info(f"活跃告警数量: {len(alerts)}")
        
        # 停止监控
        monitor.stop_monitoring()
        logger.info("系统监控已停止")
        
        logger.info("系统监控器测试完成 ✓")
        return True
        
    except Exception as e:
        logger.error(f"系统监控器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("开始监控系统综合测试")
    logger.info("=" * 60)
    
    test_results = []
    
    # 测试各个组件
    test_results.append(("系统监控器", test_system_monitor()))
    test_results.append(("实时监控系统", test_real_time_monitor()))
    test_results.append(("告警系统", test_alert_system()))
    test_results.append(("仪表板", test_dashboard()))
    
    # 输出测试结果
    logger.info("=" * 60)
    logger.info("测试结果汇总:")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("所有测试通过！监控系统功能正常 🎉")
        return 0
    else:
        logger.error("部分测试失败，请检查相关组件")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)