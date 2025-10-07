"""
实时监控与预警系统

提供性能监控、风险预警和系统健康检查功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import time
import threading
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, 
                 window_size: int = 100,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'sharpe_ratio': 0.5,
            'max_drawdown': 0.1,
            'win_rate': 0.4,
            'volatility': 0.3
        }
        
        # 性能指标历史
        self.returns_history = deque(maxlen=window_size)
        self.positions_history = deque(maxlen=window_size)
        self.trades_history = deque(maxlen=window_size)
        
        # 实时指标
        self.current_metrics = {}
        self.alerts = []
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
    def update_performance(self, 
                          returns: float,
                          position: float,
                          trade_info: Optional[Dict] = None):
        """更新性能数据"""
        timestamp = datetime.now()
        
        # 更新历史数据
        self.returns_history.append({
            'timestamp': timestamp,
            'return': returns
        })
        
        self.positions_history.append({
            'timestamp': timestamp,
            'position': position
        })
        
        if trade_info:
            trade_info['timestamp'] = timestamp
            self.trades_history.append(trade_info)
        
        # 计算实时指标
        self._calculate_metrics()
        
        # 检查预警
        self._check_alerts()
    
    def _calculate_metrics(self):
        """计算性能指标"""
        if len(self.returns_history) < 2:
            return
        
        # 提取收益率序列
        returns = [r['return'] for r in self.returns_history]
        returns_series = pd.Series(returns)
        
        # 计算基本指标
        self.current_metrics = {
            'total_return': (1 + returns_series).prod() - 1,
            'avg_return': returns_series.mean(),
            'volatility': returns_series.std(),
            'sharpe_ratio': returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns_series),
            'win_rate': (returns_series > 0).mean(),
            'profit_factor': self._calculate_profit_factor(returns_series),
            'calmar_ratio': self._calculate_calmar_ratio(returns_series),
            'sortino_ratio': self._calculate_sortino_ratio(returns_series)
        }
        
        # 计算交易相关指标
        if self.trades_history:
            self.current_metrics.update(self._calculate_trade_metrics())
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """计算盈利因子"""
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return profits / losses if losses > 0 else np.inf
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """计算卡玛比率"""
        annual_return = returns.mean() * 252  # 假设252个交易日
        max_dd = self._calculate_max_drawdown(returns)
        return annual_return / max_dd if max_dd > 0 else np.inf
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """计算索提诺比率"""
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        return returns.mean() / downside_std if downside_std > 0 else np.inf
    
    def _calculate_trade_metrics(self) -> Dict[str, float]:
        """计算交易指标"""
        trades = list(self.trades_history)
        
        if not trades:
            return {}
        
        # 提取交易信息
        trade_returns = [t.get('return', 0) for t in trades if 'return' in t]
        trade_sizes = [t.get('size', 0) for t in trades if 'size' in t]
        
        metrics = {}
        
        if trade_returns:
            metrics.update({
                'avg_trade_return': np.mean(trade_returns),
                'trade_win_rate': sum(1 for r in trade_returns if r > 0) / len(trade_returns),
                'avg_winning_trade': np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0,
                'avg_losing_trade': np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0,
                'largest_win': max(trade_returns) if trade_returns else 0,
                'largest_loss': min(trade_returns) if trade_returns else 0
            })
        
        if trade_sizes:
            metrics.update({
                'avg_trade_size': np.mean(trade_sizes),
                'total_volume': sum(abs(s) for s in trade_sizes)
            })
        
        return metrics
    
    def _check_alerts(self):
        """检查预警条件"""
        current_time = datetime.now()
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in self.current_metrics:
                value = self.current_metrics[metric]
                
                # 根据指标类型判断预警条件
                alert_triggered = False
                
                if metric in ['sharpe_ratio', 'win_rate', 'profit_factor']:
                    # 这些指标越高越好
                    alert_triggered = value < threshold
                elif metric in ['max_drawdown', 'volatility']:
                    # 这些指标越低越好
                    alert_triggered = value > threshold
                
                if alert_triggered:
                    alert = {
                        'timestamp': current_time,
                        'type': 'performance_alert',
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'message': f"{metric} ({value:.4f}) crossed threshold ({threshold:.4f})"
                    }
                    self.alerts.append(alert)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'timestamp': datetime.now(),
            'metrics': self.current_metrics.copy(),
            'recent_alerts': self.alerts[-10:],  # 最近10个预警
            'data_points': len(self.returns_history)
        }
    
    def start_monitoring(self, update_interval: float = 1.0):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(update_interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitoring_loop(self, update_interval: float):
        """监控循环"""
        while self.is_monitoring:
            # 这里可以添加定期检查逻辑
            time.sleep(update_interval)


class RiskMonitor:
    """风险监控器"""
    
    def __init__(self, 
                 risk_limits: Optional[Dict[str, float]] = None,
                 position_limits: Optional[Dict[str, float]] = None):
        self.risk_limits = risk_limits or {
            'max_position_size': 0.1,  # 最大仓位比例
            'max_leverage': 3.0,       # 最大杠杆
            'max_correlation': 0.8,    # 最大相关性
            'var_limit': 0.05,         # VaR限制
            'max_sector_exposure': 0.3  # 最大行业暴露
        }
        
        self.position_limits = position_limits or {
            'single_stock': 0.05,      # 单只股票最大仓位
            'sector': 0.2,             # 单个行业最大仓位
            'total_long': 1.0,         # 总多头仓位
            'total_short': 0.5         # 总空头仓位
        }
        
        # 当前持仓和风险状态
        self.current_positions = {}
        self.risk_metrics = {}
        self.risk_alerts = []
        
    def update_positions(self, positions: Dict[str, float]):
        """更新持仓信息"""
        self.current_positions = positions.copy()
        self._calculate_risk_metrics()
        self._check_risk_limits()
    
    def _calculate_risk_metrics(self):
        """计算风险指标"""
        if not self.current_positions:
            return
        
        positions = pd.Series(self.current_positions)
        
        # 基本风险指标
        self.risk_metrics = {
            'total_exposure': positions.abs().sum(),
            'net_exposure': positions.sum(),
            'long_exposure': positions[positions > 0].sum(),
            'short_exposure': abs(positions[positions < 0].sum()),
            'num_positions': len(positions[positions != 0]),
            'max_position': positions.abs().max(),
            'position_concentration': self._calculate_concentration(positions)
        }
        
        # 计算杠杆
        if 'portfolio_value' in self.current_positions:
            portfolio_value = self.current_positions['portfolio_value']
            if portfolio_value > 0:
                self.risk_metrics['leverage'] = self.risk_metrics['total_exposure'] / portfolio_value
    
    def _calculate_concentration(self, positions: pd.Series) -> float:
        """计算仓位集中度（HHI指数）"""
        if len(positions) == 0:
            return 0
        
        weights = positions.abs() / positions.abs().sum()
        hhi = (weights ** 2).sum()
        return hhi
    
    def calculate_var(self, 
                     returns_data: pd.DataFrame,
                     confidence_level: float = 0.05,
                     method: str = 'historical') -> float:
        """计算风险价值（VaR）"""
        if returns_data.empty or not self.current_positions:
            return 0
        
        # 获取持仓权重
        position_weights = pd.Series(self.current_positions)
        
        # 确保资产匹配
        common_assets = set(position_weights.index) & set(returns_data.columns)
        if not common_assets:
            return 0
        
        position_weights = position_weights[list(common_assets)]
        returns_data = returns_data[list(common_assets)]
        
        if method == 'historical':
            # 历史模拟法
            portfolio_returns = (returns_data * position_weights).sum(axis=1)
            var = np.percentile(portfolio_returns, confidence_level * 100)
            
        elif method == 'parametric':
            # 参数法（假设正态分布）
            portfolio_mean = (returns_data.mean() * position_weights).sum()
            portfolio_var = np.dot(position_weights, np.dot(returns_data.cov(), position_weights))
            portfolio_std = np.sqrt(portfolio_var)
            
            from scipy.stats import norm
            var = portfolio_mean + norm.ppf(confidence_level) * portfolio_std
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        self.risk_metrics['var'] = abs(var)
        return abs(var)
    
    def _check_risk_limits(self):
        """检查风险限制"""
        current_time = datetime.now()
        
        # 检查仓位限制
        for limit_name, limit_value in self.position_limits.items():
            alert_triggered = False
            current_value = 0
            
            if limit_name == 'single_stock':
                if self.current_positions:
                    max_single_position = max(abs(pos) for pos in self.current_positions.values())
                    current_value = max_single_position
                    alert_triggered = current_value > limit_value
                    
            elif limit_name == 'total_long':
                current_value = self.risk_metrics.get('long_exposure', 0)
                alert_triggered = current_value > limit_value
                
            elif limit_name == 'total_short':
                current_value = self.risk_metrics.get('short_exposure', 0)
                alert_triggered = current_value > limit_value
            
            if alert_triggered:
                alert = {
                    'timestamp': current_time,
                    'type': 'position_limit_alert',
                    'limit': limit_name,
                    'current_value': current_value,
                    'limit_value': limit_value,
                    'message': f"{limit_name} limit exceeded: {current_value:.4f} > {limit_value:.4f}"
                }
                self.risk_alerts.append(alert)
        
        # 检查风险限制
        for limit_name, limit_value in self.risk_limits.items():
            alert_triggered = False
            current_value = 0
            
            if limit_name == 'max_leverage':
                current_value = self.risk_metrics.get('leverage', 0)
                alert_triggered = current_value > limit_value
                
            elif limit_name == 'var_limit':
                current_value = self.risk_metrics.get('var', 0)
                alert_triggered = current_value > limit_value
                
            elif limit_name == 'max_position_size':
                current_value = self.risk_metrics.get('max_position', 0)
                alert_triggered = current_value > limit_value
            
            if alert_triggered:
                alert = {
                    'timestamp': current_time,
                    'type': 'risk_limit_alert',
                    'limit': limit_name,
                    'current_value': current_value,
                    'limit_value': limit_value,
                    'message': f"{limit_name} limit exceeded: {current_value:.4f} > {limit_value:.4f}"
                }
                self.risk_alerts.append(alert)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """获取风险报告"""
        return {
            'timestamp': datetime.now(),
            'positions': self.current_positions.copy(),
            'risk_metrics': self.risk_metrics.copy(),
            'recent_alerts': self.risk_alerts[-10:],
            'risk_limits': self.risk_limits.copy(),
            'position_limits': self.position_limits.copy()
        }


class SystemHealthMonitor:
    """系统健康监控器"""
    
    def __init__(self):
        self.health_metrics = {}
        self.system_alerts = []
        self.component_status = {}
        
        # 性能指标历史
        self.latency_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.cpu_history = deque(maxlen=100)
        
    def update_system_metrics(self, 
                            latency: Optional[float] = None,
                            memory_usage: Optional[float] = None,
                            cpu_usage: Optional[float] = None):
        """更新系统指标"""
        timestamp = datetime.now()
        
        if latency is not None:
            self.latency_history.append({
                'timestamp': timestamp,
                'latency': latency
            })
        
        if memory_usage is not None:
            self.memory_history.append({
                'timestamp': timestamp,
                'memory': memory_usage
            })
        
        if cpu_usage is not None:
            self.cpu_history.append({
                'timestamp': timestamp,
                'cpu': cpu_usage
            })
        
        self._calculate_health_metrics()
        self._check_system_health()
    
    def _calculate_health_metrics(self):
        """计算健康指标"""
        self.health_metrics = {}
        
        # 延迟指标
        if self.latency_history:
            latencies = [l['latency'] for l in self.latency_history]
            self.health_metrics.update({
                'avg_latency': np.mean(latencies),
                'max_latency': np.max(latencies),
                'latency_p95': np.percentile(latencies, 95),
                'latency_p99': np.percentile(latencies, 99)
            })
        
        # 内存指标
        if self.memory_history:
            memory_usage = [m['memory'] for m in self.memory_history]
            self.health_metrics.update({
                'avg_memory': np.mean(memory_usage),
                'max_memory': np.max(memory_usage),
                'memory_trend': self._calculate_trend(memory_usage)
            })
        
        # CPU指标
        if self.cpu_history:
            cpu_usage = [c['cpu'] for c in self.cpu_history]
            self.health_metrics.update({
                'avg_cpu': np.mean(cpu_usage),
                'max_cpu': np.max(cpu_usage),
                'cpu_trend': self._calculate_trend(cpu_usage)
            })
    
    def _calculate_trend(self, data: List[float]) -> str:
        """计算趋势"""
        if len(data) < 2:
            return 'stable'
        
        # 简单线性回归计算趋势
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _check_system_health(self):
        """检查系统健康状态"""
        current_time = datetime.now()
        
        # 检查延迟
        if 'avg_latency' in self.health_metrics:
            if self.health_metrics['avg_latency'] > 1000:  # 1秒
                alert = {
                    'timestamp': current_time,
                    'type': 'latency_alert',
                    'message': f"High average latency: {self.health_metrics['avg_latency']:.2f}ms",
                    'severity': 'high' if self.health_metrics['avg_latency'] > 5000 else 'medium'
                }
                self.system_alerts.append(alert)
        
        # 检查内存使用
        if 'avg_memory' in self.health_metrics:
            if self.health_metrics['avg_memory'] > 80:  # 80%
                alert = {
                    'timestamp': current_time,
                    'type': 'memory_alert',
                    'message': f"High memory usage: {self.health_metrics['avg_memory']:.1f}%",
                    'severity': 'high' if self.health_metrics['avg_memory'] > 90 else 'medium'
                }
                self.system_alerts.append(alert)
        
        # 检查CPU使用
        if 'avg_cpu' in self.health_metrics:
            if self.health_metrics['avg_cpu'] > 80:  # 80%
                alert = {
                    'timestamp': current_time,
                    'type': 'cpu_alert',
                    'message': f"High CPU usage: {self.health_metrics['avg_cpu']:.1f}%",
                    'severity': 'high' if self.health_metrics['avg_cpu'] > 90 else 'medium'
                }
                self.system_alerts.append(alert)
    
    def register_component(self, component_name: str, health_check_func: Callable):
        """注册组件健康检查"""
        self.component_status[component_name] = {
            'health_check': health_check_func,
            'last_check': None,
            'status': 'unknown'
        }
    
    def check_component_health(self, component_name: str) -> Dict[str, Any]:
        """检查组件健康状态"""
        if component_name not in self.component_status:
            return {'status': 'not_registered'}
        
        component = self.component_status[component_name]
        
        try:
            health_result = component['health_check']()
            component['last_check'] = datetime.now()
            component['status'] = 'healthy' if health_result else 'unhealthy'
            
            return {
                'component': component_name,
                'status': component['status'],
                'last_check': component['last_check'],
                'details': health_result
            }
            
        except Exception as e:
            component['status'] = 'error'
            component['last_check'] = datetime.now()
            
            return {
                'component': component_name,
                'status': 'error',
                'last_check': component['last_check'],
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 检查所有组件
        component_statuses = {}
        for component_name in self.component_status:
            component_statuses[component_name] = self.check_component_health(component_name)
        
        # 计算整体健康状态
        overall_status = 'healthy'
        if any(status['status'] in ['unhealthy', 'error'] for status in component_statuses.values()):
            overall_status = 'degraded'
        
        return {
            'timestamp': datetime.now(),
            'overall_status': overall_status,
            'health_metrics': self.health_metrics.copy(),
            'component_statuses': component_statuses,
            'recent_alerts': self.system_alerts[-10:]
        }