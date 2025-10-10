"""
实时风险监控模块

提供持续的风险监控、预警和自动化响应功能
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import websockets
import pandas as pd
import numpy as np

from .enhanced_risk_manager import (
    EnhancedRiskManager, RiskAlert, RiskLevel, AlertType, 
    PortfolioRiskMetrics, RiskLimits
)
from .risk_metrics import RiskMetricsEngine, RiskMetricResult

# 配置日志
logger = logging.getLogger(__name__)


class MonitoringStatus(Enum):
    """监控状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class ActionType(Enum):
    """自动化响应动作类型"""
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    HEDGE_POSITION = "hedge_position"
    STOP_TRADING = "stop_trading"
    SEND_NOTIFICATION = "send_notification"
    REBALANCE_PORTFOLIO = "rebalance_portfolio"


@dataclass
class MonitoringConfig:
    """监控配置"""
    # 监控间隔
    check_interval: int = 30  # 秒
    metrics_update_interval: int = 60  # 秒
    
    # 数据保留
    max_alerts_history: int = 1000
    max_metrics_history: int = 2000
    
    # 通知设置
    enable_email_alerts: bool = False
    enable_sms_alerts: bool = False
    enable_webhook_alerts: bool = True
    
    # 自动化响应
    enable_auto_actions: bool = False
    auto_action_cooldown: int = 300  # 秒
    
    # WebSocket设置
    websocket_port: int = 8765
    enable_websocket: bool = True


@dataclass
class AutoAction:
    """自动化响应动作"""
    action_type: ActionType
    trigger_conditions: Dict[str, Any]
    parameters: Dict[str, Any]
    cooldown_seconds: int = 300
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.last_executed is None:
            return True
        
        time_since_last = datetime.now() - self.last_executed
        return time_since_last.total_seconds() >= self.cooldown_seconds
    
    def mark_executed(self):
        """标记为已执行"""
        self.last_executed = datetime.now()
        self.execution_count += 1


class RealTimeRiskMonitor:
    """实时风险监控器"""
    
    def __init__(self, 
                 risk_manager: EnhancedRiskManager,
                 config: Optional[MonitoringConfig] = None):
        """
        初始化实时风险监控器
        
        Args:
            risk_manager: 增强版风险管理器
            config: 监控配置
        """
        self.risk_manager = risk_manager
        self.config = config or MonitoringConfig()
        
        # 监控状态
        self.status = MonitoringStatus.STOPPED
        self.start_time: Optional[datetime] = None
        
        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics_thread: Optional[threading.Thread] = None
        self._websocket_server = None
        
        # 数据存储
        self.alerts_history: deque = deque(maxlen=self.config.max_alerts_history)
        self.metrics_history: deque = deque(maxlen=self.config.max_metrics_history)
        self.performance_stats: Dict[str, Any] = {}
        
        # 自动化响应
        self.auto_actions: List[AutoAction] = []
        self.action_callbacks: Dict[ActionType, Callable] = {}
        
        # WebSocket连接
        self.websocket_clients: Set = set()
        
        # 事件循环
        self._loop = None
        self._running = False
        
        # 风险指标引擎
        self.metrics_engine = RiskMetricsEngine()
        
        self.logger = logger
        self.logger.info("实时风险监控器初始化完成")
    
    def start(self):
        """启动监控"""
        if self.status == MonitoringStatus.RUNNING:
            self.logger.warning("监控已在运行")
            return
        
        self.status = MonitoringStatus.STARTING
        self.start_time = datetime.now()
        self._running = True
        
        # 启动风险管理器监控
        self.risk_manager.start_monitoring()
        
        # 启动监控线程
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        # 启动指标更新线程
        self._metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
        self._metrics_thread.start()
        
        # 启动WebSocket服务器
        if self.config.enable_websocket:
            self._start_websocket_server()
        
        self.status = MonitoringStatus.RUNNING
        self.logger.info("实时风险监控已启动")
    
    def stop(self):
        """停止监控"""
        self._running = False
        self.status = MonitoringStatus.STOPPED
        
        # 停止风险管理器监控
        self.risk_manager.stop_monitoring()
        
        # 等待线程结束
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        if self._metrics_thread:
            self._metrics_thread.join(timeout=5)
        
        # 停止WebSocket服务器
        if self._websocket_server:
            self._websocket_server.close()
        
        self.logger.info("实时风险监控已停止")
    
    def pause(self):
        """暂停监控"""
        if self.status == MonitoringStatus.RUNNING:
            self.status = MonitoringStatus.PAUSED
            self.logger.info("实时风险监控已暂停")
    
    def resume(self):
        """恢复监控"""
        if self.status == MonitoringStatus.PAUSED:
            self.status = MonitoringStatus.RUNNING
            self.logger.info("实时风险监控已恢复")
    
    def _monitor_loop(self):
        """主监控循环"""
        while self._running:
            try:
                if self.status == MonitoringStatus.RUNNING:
                    self._perform_monitoring_cycle()
                
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                self.status = MonitoringStatus.ERROR
                time.sleep(self.config.check_interval)
    
    def _metrics_loop(self):
        """指标更新循环"""
        while self._running:
            try:
                if self.status == MonitoringStatus.RUNNING:
                    self._update_performance_metrics()
                
                time.sleep(self.config.metrics_update_interval)
                
            except Exception as e:
                self.logger.error(f"指标更新错误: {e}")
                time.sleep(self.config.metrics_update_interval)
    
    def _perform_monitoring_cycle(self):
        """执行一次监控周期"""
        # 获取当前风险指标
        current_metrics = self.risk_manager.get_current_risk_metrics()
        if current_metrics:
            self.metrics_history.append(current_metrics)
        
        # 获取新警报
        recent_alerts = self.risk_manager.get_recent_alerts(hours=1)
        new_alerts = [alert for alert in recent_alerts 
                     if alert not in self.alerts_history]
        
        # 处理新警报
        for alert in new_alerts:
            self._process_alert(alert)
            self.alerts_history.append(alert)
        
        # 检查自动化响应
        if self.config.enable_auto_actions:
            self._check_auto_actions(current_metrics)
        
        # 广播更新
        if self.config.enable_websocket and current_metrics:
            self._broadcast_update(current_metrics, new_alerts)
    
    def _process_alert(self, alert: RiskAlert):
        """处理警报"""
        self.logger.warning(f"处理风险警报: {alert.message}")
        
        # 发送通知
        if self.config.enable_webhook_alerts:
            self._send_webhook_notification(alert)
        
        if self.config.enable_email_alerts:
            self._send_email_notification(alert)
        
        if self.config.enable_sms_alerts:
            self._send_sms_notification(alert)
    
    def _check_auto_actions(self, metrics: PortfolioRiskMetrics):
        """检查并执行自动化响应"""
        for action in self.auto_actions:
            if not action.can_execute():
                continue
            
            # 检查触发条件
            if self._evaluate_trigger_conditions(action.trigger_conditions, metrics):
                self._execute_auto_action(action, metrics)
    
    def _evaluate_trigger_conditions(self, conditions: Dict[str, Any], 
                                   metrics: PortfolioRiskMetrics) -> bool:
        """评估触发条件"""
        try:
            for condition, threshold in conditions.items():
                metric_value = getattr(metrics, condition, None)
                if metric_value is None:
                    continue
                
                if isinstance(threshold, dict):
                    # 复杂条件
                    operator = threshold.get('operator', '>')
                    value = threshold.get('value', 0)
                    
                    if operator == '>' and metric_value <= value:
                        return False
                    elif operator == '<' and metric_value >= value:
                        return False
                    elif operator == '>=' and metric_value < value:
                        return False
                    elif operator == '<=' and metric_value > value:
                        return False
                    elif operator == '==' and metric_value != value:
                        return False
                else:
                    # 简单条件（默认大于）
                    if metric_value <= threshold:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"评估触发条件错误: {e}")
            return False
    
    def _execute_auto_action(self, action: AutoAction, metrics: PortfolioRiskMetrics):
        """执行自动化响应"""
        try:
            callback = self.action_callbacks.get(action.action_type)
            if callback:
                callback(action, metrics)
                action.mark_executed()
                self.logger.info(f"执行自动化响应: {action.action_type.value}")
            else:
                self.logger.warning(f"未找到动作回调: {action.action_type.value}")
                
        except Exception as e:
            self.logger.error(f"执行自动化响应错误: {e}")
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        if not self.metrics_history:
            return
        
        try:
            # 计算统计指标
            recent_metrics = list(self.metrics_history)[-100:]  # 最近100个数据点
            
            if len(recent_metrics) > 1:
                # VaR趋势
                var_values = [m.var_1d for m in recent_metrics]
                var_trend = np.polyfit(range(len(var_values)), var_values, 1)[0]
                
                # 波动率趋势
                vol_values = [m.volatility for m in recent_metrics]
                vol_trend = np.polyfit(range(len(vol_values)), vol_values, 1)[0]
                
                # 杠杆趋势
                leverage_values = [m.leverage for m in recent_metrics]
                leverage_trend = np.polyfit(range(len(leverage_values)), leverage_values, 1)[0]
                
                self.performance_stats = {
                    'var_trend': var_trend,
                    'volatility_trend': vol_trend,
                    'leverage_trend': leverage_trend,
                    'avg_var': np.mean(var_values),
                    'avg_volatility': np.mean(vol_values),
                    'avg_leverage': np.mean(leverage_values),
                    'max_var': np.max(var_values),
                    'max_volatility': np.max(vol_values),
                    'max_leverage': np.max(leverage_values),
                    'alert_count_24h': len([a for a in self.alerts_history 
                                          if a.timestamp >= datetime.now() - timedelta(hours=24)]),
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 
                                   if self.start_time else 0
                }
            
        except Exception as e:
            self.logger.error(f"更新性能指标错误: {e}")
    
    def _start_websocket_server(self):
        """启动WebSocket服务器"""
        try:
            async def handle_client(websocket, path):
                self.websocket_clients.add(websocket)
                self.logger.info(f"WebSocket客户端连接: {websocket.remote_address}")
                
                try:
                    # 发送初始数据
                    initial_data = {
                        'type': 'initial',
                        'status': self.status.value,
                        'config': {
                            'check_interval': self.config.check_interval,
                            'metrics_update_interval': self.config.metrics_update_interval
                        }
                    }
                    await websocket.send(json.dumps(initial_data))
                    
                    # 保持连接
                    await websocket.wait_closed()
                    
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.websocket_clients.discard(websocket)
                    self.logger.info("WebSocket客户端断开连接")
            
            # 在新线程中启动WebSocket服务器
            def run_server():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                
                start_server = websockets.serve(
                    handle_client, 
                    "localhost", 
                    self.config.websocket_port
                )
                
                self._websocket_server = loop.run_until_complete(start_server)
                self.logger.info(f"WebSocket服务器启动: localhost:{self.config.websocket_port}")
                
                loop.run_forever()
            
            websocket_thread = threading.Thread(target=run_server, daemon=True)
            websocket_thread.start()
            
        except Exception as e:
            self.logger.error(f"启动WebSocket服务器错误: {e}")
    
    def _broadcast_update(self, metrics: PortfolioRiskMetrics, alerts: List[RiskAlert]):
        """广播更新到WebSocket客户端"""
        if not self.websocket_clients or not self._loop:
            return
        
        try:
            update_data = {
                'type': 'update',
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics.to_dict(),
                'alerts': [alert.to_dict() for alert in alerts],
                'performance_stats': self.performance_stats
            }
            
            message = json.dumps(update_data)
            
            # 在事件循环中发送消息
            def send_to_clients():
                disconnected_clients = set()
                for client in self.websocket_clients:
                    try:
                        asyncio.create_task(client.send(message))
                    except Exception:
                        disconnected_clients.add(client)
                
                # 清理断开的连接
                self.websocket_clients -= disconnected_clients
            
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(send_to_clients)
                
        except Exception as e:
            self.logger.error(f"广播更新错误: {e}")
    
    def _send_webhook_notification(self, alert: RiskAlert):
        """发送Webhook通知"""
        # 简化实现，实际应该发送HTTP请求
        self.logger.info(f"Webhook通知: {alert.message}")
    
    def _send_email_notification(self, alert: RiskAlert):
        """发送邮件通知"""
        # 简化实现，实际应该发送邮件
        self.logger.info(f"邮件通知: {alert.message}")
    
    def _send_sms_notification(self, alert: RiskAlert):
        """发送短信通知"""
        # 简化实现，实际应该发送短信
        self.logger.info(f"短信通知: {alert.message}")
    
    def add_auto_action(self, action: AutoAction):
        """添加自动化响应动作"""
        self.auto_actions.append(action)
        self.logger.info(f"添加自动化响应: {action.action_type.value}")
    
    def register_action_callback(self, action_type: ActionType, callback: Callable):
        """注册动作回调函数"""
        self.action_callbacks[action_type] = callback
        self.logger.info(f"注册动作回调: {action_type.value}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() 
                             if self.start_time else 0,
            'alerts_count': len(self.alerts_history),
            'metrics_count': len(self.metrics_history),
            'websocket_clients': len(self.websocket_clients),
            'auto_actions_count': len(self.auto_actions),
            'performance_stats': self.performance_stats
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        current_metrics = self.risk_manager.get_current_risk_metrics()
        recent_alerts = list(self.alerts_history)[-10:]  # 最近10个警报
        
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': self.get_monitoring_status(),
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'recent_alerts': [alert.to_dict() for alert in recent_alerts],
            'performance_stats': self.performance_stats,
            'risk_limits': {
                'max_position_size': self.risk_manager.risk_limits.max_position_size,
                'max_leverage': self.risk_manager.risk_limits.max_leverage,
                'var_limit_1d': self.risk_manager.risk_limits.var_limit_1d,
                'max_drawdown': self.risk_manager.risk_limits.max_drawdown
            }
        }


# 使用示例
if __name__ == "__main__":
    from .enhanced_risk_manager import EnhancedRiskManager, RiskLimits
    
    # 创建风险管理器
    risk_limits = RiskLimits(max_leverage=2.0, var_limit_1d=0.02)
    risk_manager = EnhancedRiskManager(risk_limits=risk_limits)
    
    # 创建监控配置
    config = MonitoringConfig(
        check_interval=10,
        enable_auto_actions=True,
        enable_websocket=True
    )
    
    # 创建实时监控器
    monitor = RealTimeRiskMonitor(risk_manager, config)
    
    # 添加自动化响应
    auto_action = AutoAction(
        action_type=ActionType.REDUCE_POSITION,
        trigger_conditions={'leverage': 2.5},
        parameters={'reduction_ratio': 0.2}
    )
    monitor.add_auto_action(auto_action)
    
    # 注册动作回调
    def reduce_position_callback(action: AutoAction, metrics: PortfolioRiskMetrics):
        print(f"执行减仓操作: 当前杠杆 {metrics.leverage:.2f}")
    
    monitor.register_action_callback(ActionType.REDUCE_POSITION, reduce_position_callback)
    
    # 启动监控
    monitor.start()
    
    # 模拟运行
    try:
        time.sleep(30)
        
        # 获取仪表板数据
        dashboard_data = monitor.get_dashboard_data()
        print(f"监控状态: {dashboard_data['monitoring_status']['status']}")
        
    finally:
        monitor.stop()