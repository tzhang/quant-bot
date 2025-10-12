"""
因子监控体系
用于实时监控因子表现，及时发现因子失效和异常情况
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """告警级别"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "严重"

@dataclass
class FactorAlert:
    """因子告警信息"""
    factor_name: str
    alert_level: AlertLevel
    alert_type: str
    message: str
    timestamp: datetime
    metric_value: float
    threshold: float

class FactorMonitor:
    """
    因子监控系统
    
    功能：
    1. 因子表现监控
    2. 因子失效检测
    3. 异常值检测
    4. 相关性监控
    5. 稳定性监控
    6. 告警系统
    """
    
    def __init__(self, 
                 lookback_window: int = 252,
                 ic_threshold: float = 0.02,
                 stability_threshold: float = 0.3,
                 correlation_threshold: float = 0.8):
        """
        初始化因子监控器
        
        Args:
            lookback_window: 回看窗口期
            ic_threshold: IC阈值
            stability_threshold: 稳定性阈值
            correlation_threshold: 相关性阈值
        """
        self.lookback_window = lookback_window
        self.ic_threshold = ic_threshold
        self.stability_threshold = stability_threshold
        self.correlation_threshold = correlation_threshold
        
        # 监控历史
        self.factor_history = {}
        self.performance_history = {}
        self.alerts_history = []
        
        # 监控指标
        self.monitoring_metrics = {
            'ic': [],
            'ic_ir': [],
            'turnover': [],
            'coverage': [],
            'stability': [],
            'correlation': []
        }
    
    def update_factor_data(self, factor_data: Dict[str, pd.Series], returns: pd.Series):
        """
        更新因子数据
        
        Args:
            factor_data: 因子数据字典
            returns: 收益率数据
        """
        timestamp = datetime.now()
        
        # 存储因子数据
        for factor_name, factor_values in factor_data.items():
            if factor_name not in self.factor_history:
                self.factor_history[factor_name] = []
            
            self.factor_history[factor_name].append({
                'timestamp': timestamp,
                'values': factor_values,
                'returns': returns
            })
            
            # 保持历史数据在合理范围内
            if len(self.factor_history[factor_name]) > self.lookback_window:
                self.factor_history[factor_name] = self.factor_history[factor_name][-self.lookback_window:]
    
    def calculate_ic_metrics(self, factor_values: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """
        计算IC相关指标
        
        Args:
            factor_values: 因子值
            returns: 收益率
            
        Returns:
            IC指标字典
        """
        # 确保数据对齐
        aligned_data = pd.concat([factor_values, returns], axis=1, join='inner').dropna()
        if len(aligned_data) < 10:
            return {'ic': 0.0, 'ic_abs': 0.0, 'ic_ir': 0.0}
        
        factor_clean = aligned_data.iloc[:, 0]
        returns_clean = aligned_data.iloc[:, 1]
        
        # 计算IC
        ic = factor_clean.corr(returns_clean)
        ic_abs = abs(ic) if not np.isnan(ic) else 0.0
        
        # 计算IC IR (需要历史IC数据)
        ic_ir = 0.0
        if hasattr(self, '_ic_history') and len(self._ic_history) > 1:
            ic_std = np.std(self._ic_history)
            if ic_std > 0:
                ic_ir = np.mean(self._ic_history) / ic_std
        
        return {
            'ic': ic if not np.isnan(ic) else 0.0,
            'ic_abs': ic_abs,
            'ic_ir': ic_ir
        }
    
    def calculate_turnover(self, current_factor: pd.Series, previous_factor: pd.Series) -> float:
        """
        计算因子换手率
        
        Args:
            current_factor: 当前因子值
            previous_factor: 前期因子值
            
        Returns:
            换手率
        """
        if previous_factor is None or len(previous_factor) == 0:
            return 0.0
        
        # 对齐数据
        aligned_data = pd.concat([current_factor, previous_factor], axis=1, join='inner').dropna()
        if len(aligned_data) < 5:
            return 0.0
        
        current_clean = aligned_data.iloc[:, 0]
        previous_clean = aligned_data.iloc[:, 1]
        
        # 计算排名相关性
        rank_corr = current_clean.rank().corr(previous_clean.rank())
        turnover = 1 - rank_corr if not np.isnan(rank_corr) else 1.0
        
        return max(0.0, min(1.0, turnover))
    
    def calculate_coverage(self, factor_values: pd.Series, total_universe: int) -> float:
        """
        计算因子覆盖率
        
        Args:
            factor_values: 因子值
            total_universe: 总股票数量
            
        Returns:
            覆盖率
        """
        valid_count = factor_values.dropna().shape[0]
        coverage = valid_count / max(total_universe, 1)
        return min(1.0, coverage)
    
    def calculate_stability(self, factor_name: str) -> float:
        """
        计算因子稳定性
        
        Args:
            factor_name: 因子名称
            
        Returns:
            稳定性指标
        """
        if factor_name not in self.factor_history or len(self.factor_history[factor_name]) < 5:
            return 0.0
        
        # 获取最近的IC值
        recent_ics = []
        for i in range(min(20, len(self.factor_history[factor_name]))):
            data_point = self.factor_history[factor_name][-(i+1)]
            ic_metrics = self.calculate_ic_metrics(data_point['values'], data_point['returns'])
            recent_ics.append(ic_metrics['ic'])
        
        if len(recent_ics) < 3:
            return 0.0
        
        # 计算IC的稳定性（负的变异系数）
        ic_mean = np.mean(recent_ics)
        ic_std = np.std(recent_ics)
        
        if ic_std == 0 or ic_mean == 0:
            return 1.0 if ic_mean != 0 else 0.0
        
        stability = 1 - (ic_std / abs(ic_mean))
        return max(0.0, min(1.0, stability))
    
    def detect_factor_decay(self, factor_name: str) -> Tuple[bool, float]:
        """
        检测因子衰减
        
        Args:
            factor_name: 因子名称
            
        Returns:
            (是否衰减, 衰减程度)
        """
        if factor_name not in self.factor_history or len(self.factor_history[factor_name]) < 10:
            return False, 0.0
        
        # 获取最近的IC值
        recent_ics = []
        for i in range(min(20, len(self.factor_history[factor_name]))):
            data_point = self.factor_history[factor_name][-(i+1)]
            ic_metrics = self.calculate_ic_metrics(data_point['values'], data_point['returns'])
            recent_ics.append(ic_metrics['ic_abs'])
        
        if len(recent_ics) < 10:
            return False, 0.0
        
        # 计算趋势
        x = np.arange(len(recent_ics))
        slope = np.polyfit(x, recent_ics, 1)[0]
        
        # 判断是否衰减
        decay_threshold = -0.001  # 每期衰减0.1%
        is_decaying = slope < decay_threshold
        decay_magnitude = abs(slope) if is_decaying else 0.0
        
        return is_decaying, decay_magnitude
    
    def check_correlation_risk(self, factor_data: Dict[str, pd.Series]) -> List[Tuple[str, str, float]]:
        """
        检查因子间相关性风险
        
        Args:
            factor_data: 因子数据字典
            
        Returns:
            高相关性因子对列表
        """
        high_corr_pairs = []
        factor_names = list(factor_data.keys())
        
        for i in range(len(factor_names)):
            for j in range(i+1, len(factor_names)):
                factor1 = factor_data[factor_names[i]]
                factor2 = factor_data[factor_names[j]]
                
                # 对齐数据
                aligned_data = pd.concat([factor1, factor2], axis=1, join='inner').dropna()
                if len(aligned_data) < 10:
                    continue
                
                corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                if not np.isnan(corr) and abs(corr) > self.correlation_threshold:
                    high_corr_pairs.append((factor_names[i], factor_names[j], corr))
        
        return high_corr_pairs
    
    def generate_alerts(self, factor_data: Dict[str, pd.Series], returns: pd.Series) -> List[FactorAlert]:
        """
        生成告警信息
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            
        Returns:
            告警列表
        """
        alerts = []
        timestamp = datetime.now()
        
        for factor_name, factor_values in factor_data.items():
            # 1. IC告警
            ic_metrics = self.calculate_ic_metrics(factor_values, returns)
            if abs(ic_metrics['ic']) < self.ic_threshold:
                alerts.append(FactorAlert(
                    factor_name=factor_name,
                    alert_level=AlertLevel.MEDIUM,
                    alert_type="IC过低",
                    message=f"因子{factor_name}的IC值({ic_metrics['ic']:.4f})低于阈值({self.ic_threshold})",
                    timestamp=timestamp,
                    metric_value=abs(ic_metrics['ic']),
                    threshold=self.ic_threshold
                ))
            
            # 2. 稳定性告警
            stability = self.calculate_stability(factor_name)
            if stability < self.stability_threshold:
                alerts.append(FactorAlert(
                    factor_name=factor_name,
                    alert_level=AlertLevel.HIGH,
                    alert_type="稳定性不足",
                    message=f"因子{factor_name}的稳定性({stability:.4f})低于阈值({self.stability_threshold})",
                    timestamp=timestamp,
                    metric_value=stability,
                    threshold=self.stability_threshold
                ))
            
            # 3. 因子衰减告警
            is_decaying, decay_magnitude = self.detect_factor_decay(factor_name)
            if is_decaying:
                alert_level = AlertLevel.CRITICAL if decay_magnitude > 0.01 else AlertLevel.HIGH
                alerts.append(FactorAlert(
                    factor_name=factor_name,
                    alert_level=alert_level,
                    alert_type="因子衰减",
                    message=f"因子{factor_name}出现衰减趋势，衰减幅度：{decay_magnitude:.4f}",
                    timestamp=timestamp,
                    metric_value=decay_magnitude,
                    threshold=0.001
                ))
            
            # 4. 覆盖率告警
            coverage = self.calculate_coverage(factor_values, len(factor_values))
            if coverage < 0.8:
                alerts.append(FactorAlert(
                    factor_name=factor_name,
                    alert_level=AlertLevel.MEDIUM,
                    alert_type="覆盖率不足",
                    message=f"因子{factor_name}的覆盖率({coverage:.2%})过低",
                    timestamp=timestamp,
                    metric_value=coverage,
                    threshold=0.8
                ))
        
        # 5. 相关性告警
        high_corr_pairs = self.check_correlation_risk(factor_data)
        for factor1, factor2, corr in high_corr_pairs:
            alerts.append(FactorAlert(
                factor_name=f"{factor1}-{factor2}",
                alert_level=AlertLevel.MEDIUM,
                alert_type="高相关性",
                message=f"因子{factor1}和{factor2}相关性过高：{corr:.4f}",
                timestamp=timestamp,
                metric_value=abs(corr),
                threshold=self.correlation_threshold
            ))
        
        # 存储告警历史
        self.alerts_history.extend(alerts)
        
        return alerts
    
    def run_monitoring(self, factor_data: Dict[str, pd.Series], returns: pd.Series) -> Dict[str, Any]:
        """
        运行完整的监控流程
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            
        Returns:
            监控报告
        """
        # 更新数据
        self.update_factor_data(factor_data, returns)
        
        # 生成告警
        alerts = self.generate_alerts(factor_data, returns)
        
        # 计算监控指标
        monitoring_results = {}
        for factor_name, factor_values in factor_data.items():
            ic_metrics = self.calculate_ic_metrics(factor_values, returns)
            stability = self.calculate_stability(factor_name)
            coverage = self.calculate_coverage(factor_values, len(factor_values))
            
            # 计算换手率（如果有历史数据）
            turnover = 0.0
            if (factor_name in self.factor_history and 
                len(self.factor_history[factor_name]) > 1):
                previous_values = self.factor_history[factor_name][-2]['values']
                turnover = self.calculate_turnover(factor_values, previous_values)
            
            monitoring_results[factor_name] = {
                'ic': ic_metrics['ic'],
                'ic_abs': ic_metrics['ic_abs'],
                'ic_ir': ic_metrics['ic_ir'],
                'stability': stability,
                'coverage': coverage,
                'turnover': turnover,
                'is_decaying': self.detect_factor_decay(factor_name)[0]
            }
        
        # 生成报告
        report = {
            'timestamp': datetime.now(),
            'factor_metrics': monitoring_results,
            'alerts': alerts,
            'summary': {
                'total_factors': len(factor_data),
                'active_alerts': len(alerts),
                'critical_alerts': len([a for a in alerts if a.alert_level == AlertLevel.CRITICAL]),
                'decaying_factors': len([f for f, m in monitoring_results.items() if m['is_decaying']]),
                'avg_ic': np.mean([m['ic_abs'] for m in monitoring_results.values()]),
                'avg_stability': np.mean([m['stability'] for m in monitoring_results.values()])
            }
        }
        
        return report
    
    def generate_monitoring_report(self, report: Dict[str, Any]) -> str:
        """
        生成监控报告
        
        Args:
            report: 监控结果
            
        Returns:
            格式化的报告字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("因子监控报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 摘要信息
        summary = report['summary']
        lines.append("监控摘要:")
        lines.append(f"  总因子数量: {summary['total_factors']}")
        lines.append(f"  活跃告警数: {summary['active_alerts']}")
        lines.append(f"  严重告警数: {summary['critical_alerts']}")
        lines.append(f"  衰减因子数: {summary['decaying_factors']}")
        lines.append(f"  平均IC: {summary['avg_ic']:.4f}")
        lines.append(f"  平均稳定性: {summary['avg_stability']:.4f}")
        lines.append("")
        
        # 因子详细指标
        lines.append("因子详细指标:")
        lines.append("-" * 60)
        for factor_name, metrics in report['factor_metrics'].items():
            lines.append(f"因子: {factor_name}")
            lines.append(f"  IC: {metrics['ic']:.4f}")
            lines.append(f"  |IC|: {metrics['ic_abs']:.4f}")
            lines.append(f"  IC_IR: {metrics['ic_ir']:.4f}")
            lines.append(f"  稳定性: {metrics['stability']:.4f}")
            lines.append(f"  覆盖率: {metrics['coverage']:.2%}")
            lines.append(f"  换手率: {metrics['turnover']:.4f}")
            lines.append(f"  是否衰减: {'是' if metrics['is_decaying'] else '否'}")
            lines.append("")
        
        # 告警信息
        if report['alerts']:
            lines.append("告警信息:")
            lines.append("-" * 60)
            for alert in report['alerts']:
                lines.append(f"[{alert.alert_level.value}] {alert.alert_type}")
                lines.append(f"  因子: {alert.factor_name}")
                lines.append(f"  消息: {alert.message}")
                lines.append(f"  时间: {alert.timestamp.strftime('%H:%M:%S')}")
                lines.append("")
        else:
            lines.append("无告警信息")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_monitoring_report(self, report: Dict[str, Any], filename: str = None):
        """
        保存监控报告到文件
        
        Args:
            report: 监控结果
            filename: 文件名
        """
        if filename is None:
            timestamp = report['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"factor_monitoring_report_{timestamp}.txt"
        
        report_text = self.generate_monitoring_report(report)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"监控报告已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存监控报告失败: {e}")

def main():
    """示例用法"""
    # 创建监控器
    monitor = FactorMonitor(
        lookback_window=100,
        ic_threshold=0.02,
        stability_threshold=0.3,
        correlation_threshold=0.8
    )
    
    # 生成模拟数据
    np.random.seed(42)
    n_stocks = 100
    n_periods = 50
    
    print("开始因子监控测试...")
    
    # 模拟多期监控
    for period in range(n_periods):
        # 生成模拟因子数据
        factor_data = {
            'momentum': pd.Series(np.random.randn(n_stocks), 
                                index=[f'stock_{i}' for i in range(n_stocks)]),
            'value': pd.Series(np.random.randn(n_stocks), 
                             index=[f'stock_{i}' for i in range(n_stocks)]),
            'quality': pd.Series(np.random.randn(n_stocks), 
                               index=[f'stock_{i}' for i in range(n_stocks)])
        }
        
        # 生成模拟收益率（添加一些与因子的相关性）
        returns = pd.Series(
            0.1 * factor_data['momentum'] + 
            0.05 * factor_data['value'] + 
            np.random.randn(n_stocks) * 0.02,
            index=[f'stock_{i}' for i in range(n_stocks)]
        )
        
        # 运行监控
        report = monitor.run_monitoring(factor_data, returns)
        
        # 每10期输出一次报告
        if period % 10 == 0:
            print(f"\n第{period+1}期监控报告:")
            print(monitor.generate_monitoring_report(report))
    
    # 保存最终报告
    monitor.save_monitoring_report(report, "final_monitoring_report.txt")
    print("\n因子监控测试完成！")

if __name__ == "__main__":
    main()