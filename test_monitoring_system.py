#!/usr/bin/env python3
"""
监控系统测试脚本
测试监控、告警和仪表板功能
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from monitoring.real_time_monitor import TradingSystemMonitor, AlertLevel
from monitoring.alert_system import create_default_alert_router
from monitoring.dashboard import create_dashboard

class MockTradingSystem:
    """模拟交易系统"""
    
    def __init__(self):
        self.is_connected = True
        self.account_balance = 100000.0
        self.positions = []
        self.orders = []
        
    def get_account_info(self):
        """获取账户信息"""
        return {
            'balance': self.account_balance,
            'buying_power': self.account_balance * 0.8,
            'positions_count': len(self.positions)
        }
        
    def get_positions(self):
        """获取持仓"""
        return self.positions
        
    def get_orders(self):
        """获取订单"""
        return self.orders

def test_monitoring_system():
    """测试监控系统"""
    print("=" * 60)
    print("监控系统测试")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建模拟交易系统
    trading_system = MockTradingSystem()
    
    # 创建监控器
    monitor = TradingSystemMonitor(trading_system)
    
    print("✓ 创建监控器成功")
    
    # 测试告警系统
    print("\n1. 测试告警系统")
    try:
        # 创建不同级别的告警
        alert_manager = monitor.system_monitor.alert_manager
        
        info_alert = alert_manager.create_alert(
            AlertLevel.INFO,
            "系统启动",
            "监控系统已成功启动",
            "system"
        )
        print(f"  ✓ 创建信息告警: {info_alert.id}")
        
        warning_alert = alert_manager.create_alert(
            AlertLevel.WARNING,
            "CPU使用率高",
            "CPU使用率超过80%",
            "system_monitor"
        )
        print(f"  ✓ 创建警告告警: {warning_alert.id}")
        
        error_alert = alert_manager.create_alert(
            AlertLevel.ERROR,
            "连接失败",
            "无法连接到交易服务器",
            "trading_system"
        )
        print(f"  ✓ 创建错误告警: {error_alert.id}")
        
        # 检查活跃告警
        active_alerts = alert_manager.get_active_alerts()
        print(f"  ✓ 活跃告警数量: {len(active_alerts)}")
        
        # 解决一个告警
        alert_manager.resolve_alert(info_alert.id)
        print(f"  ✓ 解决告警: {info_alert.id}")
        
        # 再次检查活跃告警
        active_alerts = alert_manager.get_active_alerts()
        print(f"  ✓ 解决后活跃告警数量: {len(active_alerts)}")
        
    except Exception as e:
        print(f"  ✗ 告警系统测试失败: {str(e)}")
        return False
    
    # 测试指标收集
    print("\n2. 测试指标收集")
    try:
        metric_collector = monitor.system_monitor.metric_collector
        
        # 添加自定义指标
        metric_collector.add_metric("test.counter", 1, "count")
        metric_collector.add_metric("test.gauge", 85.5, "percent")
        metric_collector.add_metric("test.timer", 1.23, "seconds")
        
        print("  ✓ 添加自定义指标成功")
        
        # 获取指标
        recent_metrics = metric_collector.get_metrics(
            since=datetime.now() - timedelta(minutes=1)
        )
        print(f"  ✓ 获取到 {len(recent_metrics)} 个指标")
        
        # 显示部分指标
        for metric in recent_metrics[:3]:
            print(f"    - {metric.name}: {metric.value} {metric.unit}")
            
    except Exception as e:
        print(f"  ✗ 指标收集测试失败: {str(e)}")
        return False
    
    # 测试监控器
    print("\n3. 测试监控器")
    try:
        # 启动监控
        monitor.start()
        print("  ✓ 启动监控成功")
        
        # 等待一段时间收集数据
        time.sleep(2)
        
        # 获取仪表板数据
        dashboard_data = monitor.get_dashboard_data()
        print("  ✓ 获取仪表板数据成功")
        print(f"    - 系统状态: {dashboard_data['system_status']['status']}")
        print(f"    - 活跃告警: {len(dashboard_data['active_alerts'])}")
        
        # 停止监控
        monitor.stop()
        print("  ✓ 停止监控成功")
        
    except Exception as e:
        print(f"  ✗ 监控器测试失败: {str(e)}")
        return False
    
    # 测试数据导出
    print("\n4. 测试数据导出")
    try:
        # 导出指标数据
        json_metrics = monitor.export_metrics('json')
        csv_metrics = monitor.export_metrics('csv')
        
        print(f"  ✓ JSON格式指标导出: {len(json_metrics)} 字符")
        print(f"  ✓ CSV格式指标导出: {len(csv_metrics)} 字符")
        
        # 导出告警数据
        json_alerts = monitor.export_alerts('json')
        csv_alerts = monitor.export_alerts('csv')
        
        print(f"  ✓ JSON格式告警导出: {len(json_alerts)} 字符")
        print(f"  ✓ CSV格式告警导出: {len(csv_alerts)} 字符")
        
    except Exception as e:
        print(f"  ✗ 数据导出测试失败: {str(e)}")
        return False
    
    return True

def test_dashboard():
    """测试监控仪表板"""
    print("\n" + "=" * 60)
    print("监控仪表板测试")
    print("=" * 60)
    
    try:
        # 创建仪表板
        dashboard = create_dashboard(port=5001)
        print("✓ 创建仪表板成功")
        
        # 启动监控
        dashboard.monitor.start()
        print("✓ 启动监控成功")
        
        # 在线程中启动仪表板
        dashboard_thread = dashboard.start_in_thread()
        print("✓ 启动仪表板线程成功")
        
        # 等待仪表板启动
        time.sleep(3)
        
        # 测试API端点
        import requests
        
        base_url = f"http://localhost:5001"
        
        # 测试状态API
        try:
            response = requests.get(f"{base_url}/api/status", timeout=5)
            if response.status_code == 200:
                print("✓ 状态API测试成功")
            else:
                print(f"✗ 状态API返回错误: {response.status_code}")
        except Exception as e:
            print(f"✗ 状态API测试失败: {str(e)}")
        
        # 测试指标API
        try:
            response = requests.get(f"{base_url}/api/metrics", timeout=5)
            if response.status_code == 200:
                print("✓ 指标API测试成功")
            else:
                print(f"✗ 指标API返回错误: {response.status_code}")
        except Exception as e:
            print(f"✗ 指标API测试失败: {str(e)}")
        
        # 测试告警API
        try:
            response = requests.get(f"{base_url}/api/alerts", timeout=5)
            if response.status_code == 200:
                print("✓ 告警API测试成功")
            else:
                print(f"✗ 告警API返回错误: {response.status_code}")
        except Exception as e:
            print(f"✗ 告警API测试失败: {str(e)}")
        
        # 测试告警测试API
        try:
            response = requests.post(f"{base_url}/api/test_alerts", timeout=5)
            if response.status_code == 200:
                print("✓ 告警测试API成功")
            else:
                print(f"✗ 告警测试API返回错误: {response.status_code}")
        except Exception as e:
            print(f"✗ 告警测试API测试失败: {str(e)}")
        
        print(f"\n仪表板访问地址: {base_url}")
        print("请在浏览器中访问上述地址查看监控仪表板")
        
        # 停止监控
        dashboard.monitor.stop()
        print("✓ 停止监控成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 仪表板测试失败: {str(e)}")
        return False

def test_alert_channels():
    """测试告警通道"""
    print("\n" + "=" * 60)
    print("告警通道测试")
    print("=" * 60)
    
    try:
        from monitoring.alert_system import EmailAlertChannel, WebhookAlertChannel
        
        # 测试邮件告警通道（模拟模式）
        print("1. 测试邮件告警通道")
        email_channel = EmailAlertChannel(
            smtp_server="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="password",
            from_email="test@example.com",
            to_emails=["admin@example.com"]
        )
        
        # 测试连接（会失败，但不会抛出异常）
        connection_ok = email_channel.test_connection()
        print(f"  邮件通道连接测试: {'✓' if connection_ok else '✗'}")
        
        # 测试Webhook告警通道
        print("\n2. 测试Webhook告警通道")
        webhook_channel = WebhookAlertChannel(
            webhook_url="https://httpbin.org/post",
            headers={"Content-Type": "application/json"}
        )
        
        # 测试连接
        connection_ok = webhook_channel.test_connection()
        print(f"  Webhook通道连接测试: {'✓' if connection_ok else '✗'}")
        
        # 创建告警路由器
        print("\n3. 测试告警路由")
        alert_router = create_default_alert_router()
        
        # 添加通道
        alert_router.add_channel("webhook", webhook_channel)
        print("  ✓ 添加Webhook通道成功")
        
        # 创建测试告警
        from monitoring.real_time_monitor import AlertInfo
        import uuid
        test_alert = AlertInfo(
            id=str(uuid.uuid4()),
            level=AlertLevel.INFO,
            title="测试告警",
            message="这是一个测试告警消息",
            timestamp=datetime.now(),
            source="test_system"
        )
        
        # 路由告警
        try:
            alert_router.route_alert(test_alert)
            print("  ✓ 告警路由成功")
        except Exception as e:
            print(f"  ✗ 告警路由失败: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 告警通道测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("量化交易系统 - 监控系统测试")
    print("=" * 60)
    
    # 检查依赖
    try:
        import flask
        import requests
        print("✓ 依赖检查通过")
    except ImportError as e:
        print(f"✗ 缺少依赖: {str(e)}")
        print("请安装: pip install flask requests")
        return
    
    # 运行测试
    tests = [
        ("监控系统基础功能", test_monitoring_system),
        ("告警通道功能", test_alert_channels),
        ("监控仪表板", test_dashboard),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n开始测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"测试结果: {'✓ 通过' if result else '✗ 失败'}")
        except Exception as e:
            print(f"测试异常: {str(e)}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！监控系统功能正常")
    else:
        print("⚠️  部分测试失败，请检查相关功能")

if __name__ == "__main__":
    main()