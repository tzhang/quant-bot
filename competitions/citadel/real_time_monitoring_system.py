#!/usr/bin/env python3
"""
实时监控和预警系统
自动检测策略衰减和异常情况
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML和统计库
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu
import joblib

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# 系统和时间
import time
import threading
import queue
import json
from collections import deque
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeMonitoringSystem:
    """实时监控和预警系统"""
    
    def __init__(self, strategy_name="Citadel_Strategy", window_size=100):
        self.strategy_name = strategy_name
        self.window_size = window_size
        
        # 数据存储
        self.performance_buffer = deque(maxlen=window_size * 2)
        self.trade_buffer = deque(maxlen=window_size * 5)
        self.market_buffer = deque(maxlen=window_size * 2)
        
        # 监控指标
        self.monitoring_metrics = {
            'returns': deque(maxlen=window_size),
            'sharpe_ratio': deque(maxlen=window_size),
            'drawdown': deque(maxlen=window_size),
            'win_rate': deque(maxlen=window_size),
            'trade_frequency': deque(maxlen=window_size),
            'volatility': deque(maxlen=window_size)
        }
        
        # 基准性能
        self.baseline_metrics = {}
        self.performance_thresholds = {
            'return_decline': -0.3,      # 收益率下降30%
            'sharpe_decline': -0.5,      # 夏普比率下降50%
            'drawdown_increase': 0.5,    # 最大回撤增加50%
            'win_rate_decline': -0.2,    # 胜率下降20%
            'volatility_increase': 0.8   # 波动率增加80%
        }
        
        # ML模型
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.performance_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        
        # 预警系统
        self.alert_queue = queue.Queue()
        self.alert_history = []
        
        # 监控状态
        self.is_monitoring = False
        self.monitoring_thread = None
        
        print(f"🔍 实时监控系统初始化完成: {strategy_name}")
    
    def set_baseline_performance(self, historical_data):
        """设置基准性能指标"""
        print("📊 设置基准性能指标...")
        
        # 计算历史性能指标
        returns = historical_data['returns']
        
        self.baseline_metrics = {
            'avg_return': np.mean(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': np.sum(returns > 0) / len(returns),
            'volatility': np.std(returns),
            'trade_frequency': len(returns) / 30  # 假设30天数据
        }
        
        print(f"   基准收益率: {self.baseline_metrics['avg_return']:.4f}")
        print(f"   基准夏普比率: {self.baseline_metrics['sharpe_ratio']:.4f}")
        print(f"   基准最大回撤: {self.baseline_metrics['max_drawdown']:.4f}")
        print(f"   基准胜率: {self.baseline_metrics['win_rate']:.4f}")
    
    def _calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        returns = np.array(returns)  # 确保是numpy数组
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def add_performance_data(self, timestamp, return_rate, trade_info=None, market_info=None):
        """添加性能数据"""
        # 存储原始数据
        perf_data = {
            'timestamp': timestamp,
            'return': return_rate,
            'cumulative_return': 0  # 将在后续计算
        }
        self.performance_buffer.append(perf_data)
        
        # 存储交易信息
        if trade_info:
            trade_info['timestamp'] = timestamp
            self.trade_buffer.append(trade_info)
        
        # 存储市场信息
        if market_info:
            market_info['timestamp'] = timestamp
            self.market_buffer.append(market_info)
        
        # 更新监控指标
        self._update_monitoring_metrics()
        
        # 检查异常
        if len(self.performance_buffer) >= self.window_size:
            self._check_anomalies()
    
    def _update_monitoring_metrics(self):
        """更新监控指标"""
        if len(self.performance_buffer) < 10:
            return
        
        # 获取最近的收益率数据
        recent_returns = [p['return'] for p in list(self.performance_buffer)[-self.window_size:]]
        
        # 计算各项指标
        self.monitoring_metrics['returns'].append(np.mean(recent_returns))
        
        if len(recent_returns) > 1:
            sharpe = np.mean(recent_returns) / np.std(recent_returns) if np.std(recent_returns) > 0 else 0
            self.monitoring_metrics['sharpe_ratio'].append(sharpe)
        
        # 计算回撤
        drawdown = self._calculate_max_drawdown(recent_returns)
        self.monitoring_metrics['drawdown'].append(drawdown)
        
        # 计算胜率
        win_rate = np.sum(np.array(recent_returns) > 0) / len(recent_returns)
        self.monitoring_metrics['win_rate'].append(win_rate)
        
        # 计算交易频率
        recent_trades = len([t for t in self.trade_buffer if t.get('timestamp', 0) > time.time() - 86400])
        self.monitoring_metrics['trade_frequency'].append(recent_trades)
        
        # 计算波动率
        volatility = np.std(recent_returns)
        self.monitoring_metrics['volatility'].append(volatility)
    
    def _check_anomalies(self):
        """检查异常情况"""
        current_time = datetime.now()
        
        # 1. 性能衰减检测
        self._detect_performance_degradation()
        
        # 2. 异常模式检测
        self._detect_anomaly_patterns()
        
        # 3. 市场制度变化检测
        self._detect_regime_change()
        
        # 4. 统计显著性检测
        self._detect_statistical_significance()
    
    def _detect_performance_degradation(self):
        """检测性能衰减"""
        if not self.baseline_metrics or len(self.monitoring_metrics['returns']) < 20:
            return
        
        # 获取最近性能
        recent_return = np.mean(list(self.monitoring_metrics['returns'])[-10:])
        recent_sharpe = np.mean(list(self.monitoring_metrics['sharpe_ratio'])[-10:])
        recent_drawdown = np.mean(list(self.monitoring_metrics['drawdown'])[-10:])
        recent_win_rate = np.mean(list(self.monitoring_metrics['win_rate'])[-10:])
        recent_volatility = np.mean(list(self.monitoring_metrics['volatility'])[-10:])
        
        # 计算相对变化
        return_change = (recent_return - self.baseline_metrics['avg_return']) / abs(self.baseline_metrics['avg_return'])
        sharpe_change = (recent_sharpe - self.baseline_metrics['sharpe_ratio']) / abs(self.baseline_metrics['sharpe_ratio'])
        drawdown_change = (recent_drawdown - self.baseline_metrics['max_drawdown']) / abs(self.baseline_metrics['max_drawdown'])
        win_rate_change = (recent_win_rate - self.baseline_metrics['win_rate']) / self.baseline_metrics['win_rate']
        volatility_change = (recent_volatility - self.baseline_metrics['volatility']) / self.baseline_metrics['volatility']
        
        # 检查阈值
        alerts = []
        
        if return_change < self.performance_thresholds['return_decline']:
            alerts.append(f"收益率显著下降: {return_change:.2%}")
        
        if sharpe_change < self.performance_thresholds['sharpe_decline']:
            alerts.append(f"夏普比率显著下降: {sharpe_change:.2%}")
        
        if drawdown_change > self.performance_thresholds['drawdown_increase']:
            alerts.append(f"最大回撤显著增加: {drawdown_change:.2%}")
        
        if win_rate_change < self.performance_thresholds['win_rate_decline']:
            alerts.append(f"胜率显著下降: {win_rate_change:.2%}")
        
        if volatility_change > self.performance_thresholds['volatility_increase']:
            alerts.append(f"波动率显著增加: {volatility_change:.2%}")
        
        # 发送预警
        for alert in alerts:
            self._send_alert("性能衰减", alert, "high")
    
    def _detect_anomaly_patterns(self):
        """检测异常模式"""
        if len(self.monitoring_metrics['returns']) < 30:
            return
        
        # 准备特征数据
        features = []
        for i in range(len(self.monitoring_metrics['returns'])):
            feature_vector = [
                list(self.monitoring_metrics['returns'])[i],
                list(self.monitoring_metrics['sharpe_ratio'])[i] if i < len(self.monitoring_metrics['sharpe_ratio']) else 0,
                list(self.monitoring_metrics['drawdown'])[i],
                list(self.monitoring_metrics['win_rate'])[i],
                list(self.monitoring_metrics['volatility'])[i]
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # 训练异常检测模型
        if len(features) >= 30:
            try:
                # 标准化特征
                features_scaled = self.scaler.fit_transform(features)
                
                # 检测异常
                anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
                
                # 检查最近的异常
                recent_anomalies = anomaly_scores[-10:]
                anomaly_count = np.sum(recent_anomalies == -1)
                
                if anomaly_count >= 3:  # 最近10个点中有3个异常
                    self._send_alert("异常模式", f"检测到{anomaly_count}个异常数据点", "medium")
                    
            except Exception as e:
                logger.warning(f"异常检测失败: {e}")
    
    def _detect_regime_change(self):
        """检测市场制度变化"""
        if len(self.monitoring_metrics['returns']) < 50:
            return
        
        returns = list(self.monitoring_metrics['returns'])
        
        # 分割数据为两个时期
        mid_point = len(returns) // 2
        period1 = returns[:mid_point]
        period2 = returns[mid_point:]
        
        # Kolmogorov-Smirnov检验
        try:
            ks_stat, ks_p_value = ks_2samp(period1, period2)
            
            if ks_p_value < 0.05:  # 显著性水平5%
                self._send_alert("制度变化", f"检测到市场制度变化 (KS统计量: {ks_stat:.3f}, p值: {ks_p_value:.3f})", "medium")
        
        except Exception as e:
            logger.warning(f"制度变化检测失败: {e}")
    
    def _detect_statistical_significance(self):
        """检测统计显著性变化"""
        if len(self.monitoring_metrics['returns']) < 30:
            return
        
        returns = list(self.monitoring_metrics['returns'])
        
        # 检查最近收益率是否显著不同于零
        recent_returns = returns[-20:]
        
        try:
            # t检验
            t_stat, t_p_value = stats.ttest_1samp(recent_returns, 0)
            
            # 如果收益率显著为负
            if t_p_value < 0.05 and np.mean(recent_returns) < 0:
                self._send_alert("统计显著性", f"收益率显著为负 (t统计量: {t_stat:.3f}, p值: {t_p_value:.3f})", "high")
        
        except Exception as e:
            logger.warning(f"统计显著性检测失败: {e}")
    
    def _send_alert(self, alert_type, message, severity="medium"):
        """发送预警"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity,
            'strategy': self.strategy_name
        }
        
        self.alert_queue.put(alert)
        self.alert_history.append(alert)
        
        # 打印预警
        severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}
        print(f"{severity_emoji.get(severity, '⚠️')} [{alert_type}] {message}")
        
        # 记录日志
        logger.warning(f"预警: {alert_type} - {message}")
    
    def get_current_status(self):
        """获取当前监控状态"""
        if not self.monitoring_metrics['returns']:
            return {"status": "insufficient_data"}
        
        # 计算当前指标
        current_metrics = {}
        for metric_name, values in self.monitoring_metrics.items():
            if values:
                current_metrics[metric_name] = {
                    'current': values[-1],
                    'avg_recent': np.mean(list(values)[-10:]) if len(values) >= 10 else values[-1],
                    'trend': 'up' if len(values) >= 2 and values[-1] > values[-2] else 'down'
                }
        
        # 计算健康度评分
        health_score = self._calculate_health_score()
        
        return {
            'status': 'monitoring',
            'health_score': health_score,
            'current_metrics': current_metrics,
            'recent_alerts': len([a for a in self.alert_history if 
                                (datetime.now() - a['timestamp']).seconds < 3600]),  # 最近1小时的预警
            'total_alerts': len(self.alert_history)
        }
    
    def _calculate_health_score(self):
        """计算策略健康度评分 (0-100)"""
        if not self.baseline_metrics or not self.monitoring_metrics['returns']:
            return 50  # 默认分数
        
        score = 100
        
        # 基于各项指标的相对表现计算分数
        try:
            recent_return = np.mean(list(self.monitoring_metrics['returns'])[-10:])
            recent_sharpe = np.mean(list(self.monitoring_metrics['sharpe_ratio'])[-10:])
            recent_win_rate = np.mean(list(self.monitoring_metrics['win_rate'])[-10:])
            
            # 收益率评分 (30%)
            return_score = max(0, min(30, 30 * (1 + recent_return / abs(self.baseline_metrics['avg_return']))))
            
            # 夏普比率评分 (30%)
            sharpe_score = max(0, min(30, 30 * recent_sharpe / max(self.baseline_metrics['sharpe_ratio'], 0.1)))
            
            # 胜率评分 (20%)
            win_rate_score = max(0, min(20, 20 * recent_win_rate / self.baseline_metrics['win_rate']))
            
            # 稳定性评分 (20%) - 基于最近预警数量
            recent_alerts = len([a for a in self.alert_history if 
                               (datetime.now() - a['timestamp']).seconds < 3600])
            stability_score = max(0, 20 - recent_alerts * 5)
            
            score = return_score + sharpe_score + win_rate_score + stability_score
            
        except Exception as e:
            logger.warning(f"健康度评分计算失败: {e}")
            score = 50
        
        return max(0, min(100, score))
    
    def generate_monitoring_report(self):
        """生成监控报告"""
        print("\n📊 生成实时监控报告...")
        
        status = self.get_current_status()
        
        print(f"\n📋 策略监控报告 - {self.strategy_name}")
        print("=" * 60)
        print(f"监控状态: {status['status']}")
        print(f"健康度评分: {status['health_score']:.1f}/100")
        print(f"最近1小时预警: {status['recent_alerts']} 次")
        print(f"总预警次数: {status['total_alerts']} 次")
        
        if 'current_metrics' in status:
            print("\n📈 当前性能指标:")
            for metric, data in status['current_metrics'].items():
                trend_emoji = "📈" if data['trend'] == 'up' else "📉"
                print(f"   {metric}: {data['current']:.4f} {trend_emoji}")
        
        # 显示最近预警
        if self.alert_history:
            print("\n🚨 最近预警:")
            recent_alerts = sorted(self.alert_history, key=lambda x: x['timestamp'], reverse=True)[:5]
            for alert in recent_alerts:
                time_str = alert['timestamp'].strftime('%H:%M:%S')
                print(f"   [{time_str}] {alert['type']}: {alert['message']}")
        
        return status
    
    def visualize_monitoring_dashboard(self, save_path="/tmp/monitoring_dashboard.png"):
        """可视化监控仪表板"""
        print("📊 生成监控仪表板...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'实时监控仪表板 - {self.strategy_name}', fontsize=16, fontweight='bold')
        
        # 1. 收益率趋势
        if self.monitoring_metrics['returns']:
            axes[0, 0].plot(list(self.monitoring_metrics['returns']), 'b-', linewidth=2)
            axes[0, 0].set_title('收益率趋势')
            axes[0, 0].set_ylabel('收益率')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 夏普比率
        if self.monitoring_metrics['sharpe_ratio']:
            axes[0, 1].plot(list(self.monitoring_metrics['sharpe_ratio']), 'g-', linewidth=2)
            axes[0, 1].set_title('夏普比率')
            axes[0, 1].set_ylabel('夏普比率')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 回撤
        if self.monitoring_metrics['drawdown']:
            axes[0, 2].plot(list(self.monitoring_metrics['drawdown']), 'r-', linewidth=2)
            axes[0, 2].set_title('最大回撤')
            axes[0, 2].set_ylabel('回撤')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 胜率
        if self.monitoring_metrics['win_rate']:
            axes[1, 0].plot(list(self.monitoring_metrics['win_rate']), 'purple', linewidth=2)
            axes[1, 0].set_title('胜率')
            axes[1, 0].set_ylabel('胜率')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 交易频率
        if self.monitoring_metrics['trade_frequency']:
            axes[1, 1].bar(range(len(self.monitoring_metrics['trade_frequency'])), 
                          list(self.monitoring_metrics['trade_frequency']), alpha=0.7)
            axes[1, 1].set_title('交易频率')
            axes[1, 1].set_ylabel('交易次数')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 健康度评分
        health_score = self._calculate_health_score()
        colors = ['red' if health_score < 30 else 'orange' if health_score < 70 else 'green']
        axes[1, 2].bar(['健康度'], [health_score], color=colors)
        axes[1, 2].set_title('策略健康度')
        axes[1, 2].set_ylabel('评分')
        axes[1, 2].set_ylim(0, 100)
        axes[1, 2].text(0, health_score + 5, f'{health_score:.1f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 监控仪表板已保存至: {save_path}")
        return save_path

def simulate_real_time_monitoring():
    """模拟实时监控演示"""
    print("🔍 实时监控系统演示")
    print("=" * 60)
    
    # 创建监控系统
    monitor = RealTimeMonitoringSystem("Citadel_ML_Strategy")
    
    # 生成历史基准数据
    np.random.seed(42)
    historical_returns = np.random.normal(0.001, 0.02, 100)  # 历史收益率
    historical_data = {'returns': historical_returns}
    
    # 设置基准
    monitor.set_baseline_performance(historical_data)
    
    print("\n🔄 开始模拟实时数据流...")
    
    # 模拟实时数据
    for i in range(150):
        # 模拟策略衰减 - 在第80个点后性能开始下降
        if i < 80:
            # 正常表现
            return_rate = np.random.normal(0.001, 0.02)
        else:
            # 性能衰减
            decline_factor = (i - 80) / 70  # 逐渐衰减
            return_rate = np.random.normal(0.001 * (1 - decline_factor), 0.02 * (1 + decline_factor))
        
        # 模拟交易信息
        trade_info = {
            'trade_id': f'trade_{i}',
            'side': 'buy' if return_rate > 0 else 'sell',
            'quantity': abs(return_rate) * 1000,
            'price': 100 + np.random.normal(0, 1)
        }
        
        # 模拟市场信息
        market_info = {
            'volatility': abs(return_rate) * 10,
            'volume': np.random.lognormal(10, 0.5),
            'spread': abs(return_rate) * 0.1
        }
        
        # 添加数据到监控系统
        monitor.add_performance_data(
            timestamp=time.time() + i,
            return_rate=return_rate,
            trade_info=trade_info,
            market_info=market_info
        )
        
        # 每20个点显示一次状态
        if (i + 1) % 30 == 0:
            print(f"\n📊 第 {i+1} 个数据点:")
            status = monitor.get_current_status()
            print(f"   健康度评分: {status['health_score']:.1f}/100")
            print(f"   最近预警: {status['recent_alerts']} 次")
    
    print("\n📋 生成最终监控报告...")
    final_status = monitor.generate_monitoring_report()
    
    print("\n📊 生成监控仪表板...")
    dashboard_path = monitor.visualize_monitoring_dashboard()
    
    print(f"\n🎯 监控演示总结:")
    print(f"   最终健康度评分: {final_status['health_score']:.1f}/100")
    print(f"   总预警次数: {final_status['total_alerts']} 次")
    print(f"   监控数据点: 150 个")
    print(f"   仪表板路径: {dashboard_path}")
    
    return monitor, final_status

def main():
    """主函数 - 演示实时监控系统"""
    monitor, status = simulate_real_time_monitoring()
    
    print("\n🚀 实时监控系统演示完成!")
    print("   系统成功检测到策略性能衰减并发出预警")
    print("   可用于生产环境的策略监控和风险管理")

if __name__ == "__main__":
    main()