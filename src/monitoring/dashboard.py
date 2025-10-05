"""
监控仪表板
提供Web界面展示监控数据和告警信息
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template_string, jsonify, request
import threading
import time
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.real_time_monitor import TradingSystemMonitor, AlertLevel
from monitoring.alert_system import create_default_alert_router

class MonitoringDashboard:
    """监控仪表板"""
    
    def __init__(self, monitor: TradingSystemMonitor, port: int = 5000):
        self.monitor = monitor
        self.port = port
        self.app = Flask(__name__)
        self.app.secret_key = 'monitoring_dashboard_secret_key'
        self.logger = logging.getLogger(__name__)
        
        # 设置路由
        self._setup_routes()
        
        # 告警路由器
        self.alert_router = create_default_alert_router()
        
        # 添加告警处理器
        self.monitor.system_monitor.alert_manager.add_alert_handler(
            self._handle_alert
        )
        
    def _handle_alert(self, alert):
        """处理告警"""
        try:
            self.alert_router.route_alert(alert)
        except Exception as e:
            self.logger.error(f"告警处理失败: {str(e)}")
        
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template_string(self._get_dashboard_template())
            
        @self.app.route('/api/status')
        def api_status():
            """获取系统状态"""
            try:
                return jsonify(self.monitor.get_dashboard_data())
            except Exception as e:
                self.logger.error(f"获取状态失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/metrics')
        def api_metrics():
            """获取指标数据"""
            try:
                # 获取时间范围
                hours = request.args.get('hours', 1, type=int)
                since = datetime.now() - timedelta(hours=hours)
                
                metrics = self.monitor.system_monitor.metric_collector.get_metrics(since=since)
                
                # 按指标名称分组
                grouped_metrics = {}
                for metric in metrics:
                    if metric.name not in grouped_metrics:
                        grouped_metrics[metric.name] = []
                    grouped_metrics[metric.name].append({
                        'timestamp': metric.timestamp.isoformat(),
                        'value': metric.value,
                        'unit': metric.unit
                    })
                
                return jsonify(grouped_metrics)
                
            except Exception as e:
                self.logger.error(f"获取指标失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/alerts')
        def api_alerts():
            """获取告警数据"""
            try:
                # 获取参数
                active_only = request.args.get('active_only', 'false').lower() == 'true'
                level = request.args.get('level')
                
                if active_only:
                    alerts = self.monitor.system_monitor.alert_manager.get_active_alerts()
                else:
                    alerts = self.monitor.system_monitor.alert_manager.alerts
                
                if level:
                    try:
                        alert_level = AlertLevel(level.lower())
                        alerts = [a for a in alerts if a.level == alert_level]
                    except ValueError:
                        pass
                
                return jsonify([alert.to_dict() for alert in alerts])
                
            except Exception as e:
                self.logger.error(f"获取告警失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
        def api_resolve_alert(alert_id):
            """解决告警"""
            try:
                self.monitor.system_monitor.alert_manager.resolve_alert(alert_id)
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"解决告警失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/test_alerts', methods=['POST'])
        def api_test_alerts():
            """测试告警系统"""
            try:
                # 创建测试告警
                test_alert = self.monitor.system_monitor.alert_manager.create_alert(
                    AlertLevel.INFO,
                    "测试告警",
                    "这是一个测试告警消息",
                    "dashboard_test"
                )
                
                return jsonify({
                    'success': True,
                    'alert_id': test_alert.id
                })
                
            except Exception as e:
                self.logger.error(f"测试告警失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/export/metrics')
        def api_export_metrics():
            """导出指标"""
            try:
                format_type = request.args.get('format', 'json')
                data = self.monitor.system_monitor.export_metrics(format_type)
                
                response = self.app.response_class(
                    response=data,
                    status=200,
                    mimetype='application/json' if format_type == 'json' else 'text/plain'
                )
                
                filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
                response.headers['Content-Disposition'] = f'attachment; filename={filename}'
                
                return response
                
            except Exception as e:
                self.logger.error(f"导出指标失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/export/alerts')
        def api_export_alerts():
            """导出告警"""
            try:
                format_type = request.args.get('format', 'json')
                data = self.monitor.system_monitor.export_alerts(format_type)
                
                response = self.app.response_class(
                    response=data,
                    status=200,
                    mimetype='application/json' if format_type == 'json' else 'text/plain'
                )
                
                filename = f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
                response.headers['Content-Disposition'] = f'attachment; filename={filename}'
                
                return response
                
            except Exception as e:
                self.logger.error(f"导出告警失败: {str(e)}")
                return jsonify({'error': str(e)}), 500
    
    def _get_dashboard_template(self) -> str:
        """获取仪表板模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化交易系统监控仪表板</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .status-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .status-card h3 {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-value {
            font-size: 2rem;
            font-weight: 700;
            color: #333;
        }
        
        .status-running { border-left-color: #10b981; }
        .status-warning { border-left-color: #f59e0b; }
        .status-error { border-left-color: #ef4444; }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .chart-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .chart-card h3 {
            margin-bottom: 1rem;
            color: #333;
            font-size: 1.1rem;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        .alerts-section {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .alerts-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .alerts-header h3 {
            color: #333;
            font-size: 1.1rem;
        }
        
        .alert-item {
            border-left: 4px solid #ccc;
            padding: 1rem;
            margin-bottom: 0.5rem;
            background: #f9f9f9;
            border-radius: 0 4px 4px 0;
        }
        
        .alert-info { border-left-color: #3b82f6; }
        .alert-warning { border-left-color: #f59e0b; }
        .alert-error { border-left-color: #ef4444; }
        .alert-critical { border-left-color: #7c3aed; }
        
        .alert-title {
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .alert-message {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        
        .alert-meta {
            font-size: 0.8rem;
            color: #999;
        }
        
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background-color 0.2s;
        }
        
        .btn:hover {
            background: #5a67d8;
        }
        
        .btn-small {
            padding: 0.25rem 0.5rem;
            font-size: 0.8rem;
        }
        
        .loading {
            text-align: center;
            color: #666;
            padding: 2rem;
        }
        
        .error {
            color: #ef4444;
            background: #fef2f2;
            padding: 1rem;
            border-radius: 4px;
            border: 1px solid #fecaca;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .status-grid {
                grid-template-columns: 1fr;
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>量化交易系统监控仪表板</h1>
    </div>
    
    <div class="container">
        <!-- 状态卡片 -->
        <div class="status-grid">
            <div class="status-card" id="system-status">
                <h3>系统状态</h3>
                <div class="status-value" id="status-value">加载中...</div>
            </div>
            <div class="status-card">
                <h3>活跃告警</h3>
                <div class="status-value" id="alerts-count">0</div>
            </div>
            <div class="status-card">
                <h3>关键告警</h3>
                <div class="status-value" id="critical-alerts">0</div>
            </div>
            <div class="status-card">
                <h3>监控器数量</h3>
                <div class="status-value" id="monitors-count">0</div>
            </div>
        </div>
        
        <!-- 图表 -->
        <div class="charts-grid">
            <div class="chart-card">
                <h3>系统资源使用率</h3>
                <div class="chart-container">
                    <canvas id="resourceChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h3>告警趋势</h3>
                <div class="chart-container">
                    <canvas id="alertChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- 告警列表 -->
        <div class="alerts-section">
            <div class="alerts-header">
                <h3>最近告警</h3>
                <div>
                    <button class="btn btn-small" onclick="testAlert()">测试告警</button>
                    <button class="btn btn-small" onclick="refreshData()">刷新</button>
                </div>
            </div>
            <div id="alerts-list">
                <div class="loading">加载告警数据...</div>
            </div>
        </div>
    </div>
    
    <script>
        let resourceChart, alertChart;
        
        // 初始化图表
        function initCharts() {
            // 资源使用率图表
            const resourceCtx = document.getElementById('resourceChart').getContext('2d');
            resourceChart = new Chart(resourceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU (%)',
                            data: [],
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: '内存 (%)',
                            data: [],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: '磁盘 (%)',
                            data: [],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // 告警趋势图表
            const alertCtx = document.getElementById('alertChart').getContext('2d');
            alertChart = new Chart(alertCtx, {
                type: 'bar',
                data: {
                    labels: ['信息', '警告', '错误', '关键'],
                    datasets: [{
                        label: '告警数量',
                        data: [0, 0, 0, 0],
                        backgroundColor: [
                            '#3b82f6',
                            '#f59e0b',
                            '#ef4444',
                            '#7c3aed'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        // 获取系统状态
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                updateStatusCards(data.system_status);
                updateAlerts(data.active_alerts);
                
            } catch (error) {
                console.error('获取状态失败:', error);
                document.getElementById('status-value').textContent = '错误';
                document.getElementById('system-status').className = 'status-card status-error';
            }
        }
        
        // 获取指标数据
        async function fetchMetrics() {
            try {
                const response = await fetch('/api/metrics?hours=1');
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                updateResourceChart(data);
                
            } catch (error) {
                console.error('获取指标失败:', error);
            }
        }
        
        // 获取告警数据
        async function fetchAlerts() {
            try {
                const response = await fetch('/api/alerts');
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                updateAlertChart(data);
                
            } catch (error) {
                console.error('获取告警失败:', error);
            }
        }
        
        // 更新状态卡片
        function updateStatusCards(status) {
            document.getElementById('status-value').textContent = 
                status.status === 'running' ? '运行中' : '已停止';
            
            const statusCard = document.getElementById('system-status');
            statusCard.className = 'status-card ' + 
                (status.status === 'running' ? 'status-running' : 'status-error');
            
            document.getElementById('alerts-count').textContent = status.active_alerts_count || 0;
            document.getElementById('critical-alerts').textContent = status.critical_alerts_count || 0;
            document.getElementById('monitors-count').textContent = status.monitors_count || 0;
        }
        
        // 更新资源图表
        function updateResourceChart(metrics) {
            const cpuData = metrics['system.cpu_percent'] || [];
            const memoryData = metrics['system.memory_percent'] || [];
            const diskData = metrics['system.disk_percent'] || [];
            
            if (cpuData.length === 0) return;
            
            const labels = cpuData.map(item => 
                new Date(item.timestamp).toLocaleTimeString()
            ).slice(-20);
            
            resourceChart.data.labels = labels;
            resourceChart.data.datasets[0].data = cpuData.map(item => item.value).slice(-20);
            resourceChart.data.datasets[1].data = memoryData.map(item => item.value).slice(-20);
            resourceChart.data.datasets[2].data = diskData.map(item => item.value).slice(-20);
            
            resourceChart.update();
        }
        
        // 更新告警图表
        function updateAlertChart(alerts) {
            const counts = { info: 0, warning: 0, error: 0, critical: 0 };
            
            alerts.forEach(alert => {
                if (counts.hasOwnProperty(alert.level)) {
                    counts[alert.level]++;
                }
            });
            
            alertChart.data.datasets[0].data = [
                counts.info,
                counts.warning,
                counts.error,
                counts.critical
            ];
            
            alertChart.update();
        }
        
        // 更新告警列表
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alerts-list');
            
            if (alerts.length === 0) {
                alertsList.innerHTML = '<div class="loading">暂无活跃告警</div>';
                return;
            }
            
            const html = alerts.slice(0, 10).map(alert => `
                <div class="alert-item alert-${alert.level}">
                    <div class="alert-title">${alert.title}</div>
                    <div class="alert-message">${alert.message}</div>
                    <div class="alert-meta">
                        ${alert.source} • ${new Date(alert.timestamp).toLocaleString()}
                        <button class="btn btn-small" onclick="resolveAlert('${alert.id}')" style="float: right;">
                            解决
                        </button>
                    </div>
                </div>
            `).join('');
            
            alertsList.innerHTML = html;
        }
        
        // 解决告警
        async function resolveAlert(alertId) {
            try {
                const response = await fetch(`/api/alerts/${alertId}/resolve`, {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    refreshData();
                } else {
                    alert('解决告警失败: ' + (result.error || '未知错误'));
                }
                
            } catch (error) {
                console.error('解决告警失败:', error);
                alert('解决告警失败: ' + error.message);
            }
        }
        
        // 测试告警
        async function testAlert() {
            try {
                const response = await fetch('/api/test_alerts', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('测试告警已发送');
                    setTimeout(refreshData, 1000);
                } else {
                    alert('测试告警失败: ' + (result.error || '未知错误'));
                }
                
            } catch (error) {
                console.error('测试告警失败:', error);
                alert('测试告警失败: ' + error.message);
            }
        }
        
        // 刷新数据
        function refreshData() {
            fetchStatus();
            fetchMetrics();
            fetchAlerts();
        }
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            refreshData();
            
            // 定时刷新
            setInterval(refreshData, 30000); // 30秒刷新一次
        });
    </script>
</body>
</html>
        """
    
    def start(self, debug: bool = False, threaded: bool = True):
        """启动仪表板"""
        try:
            self.logger.info(f"启动监控仪表板，端口: {self.port}")
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=debug,
                threaded=threaded,
                use_reloader=False
            )
        except Exception as e:
            self.logger.error(f"启动仪表板失败: {str(e)}")
            raise
    
    def start_in_thread(self):
        """在线程中启动仪表板"""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread

def create_dashboard(trading_system=None, port: int = 5000) -> MonitoringDashboard:
    """创建监控仪表板"""
    monitor = TradingSystemMonitor(trading_system)
    dashboard = MonitoringDashboard(monitor, port)
    return dashboard

if __name__ == "__main__":
    # 测试仪表板
    logging.basicConfig(level=logging.INFO)
    
    dashboard = create_dashboard()
    
    # 启动监控
    dashboard.monitor.start()
    
    try:
        # 启动仪表板
        print(f"监控仪表板启动中...")
        print(f"访问地址: http://localhost:{dashboard.port}")
        dashboard.start(debug=True)
    except KeyboardInterrupt:
        print("\n停止监控仪表板...")
        dashboard.monitor.stop()
    except Exception as e:
        print(f"仪表板启动失败: {str(e)}")
        dashboard.monitor.stop()