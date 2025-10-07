#!/usr/bin/env python3
"""
Terminal AI 工具模块
专为Citadel Terminal AI Competition等AI交易竞赛设计

主要功能:
1. 实时数据处理
2. 高频交易策略
3. 算法优化
4. 性能监控
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
import threading
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class RealTimeDataProcessor:
    """实时数据处理器"""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.feature_cache = {}
        self.last_update = time.time()
    
    def add_tick_data(self, tick_data: Dict) -> None:
        """
        添加tick数据
        
        Args:
            tick_data: tick数据字典
        """
        tick_data['timestamp'] = time.time()
        self.data_buffer.append(tick_data)
        self.last_update = time.time()
        
        # 更新特征缓存
        self._update_feature_cache(tick_data)
    
    def get_latest_features(self, lookback: int = 100) -> Dict:
        """
        获取最新特征
        
        Args:
            lookback: 回看期数
            
        Returns:
            最新特征字典
        """
        if len(self.data_buffer) < lookback:
            return {}
        
        recent_data = list(self.data_buffer)[-lookback:]
        df = pd.DataFrame(recent_data)
        
        features = {}
        
        if 'price' in df.columns:
            # 价格特征
            features['price_current'] = df['price'].iloc[-1]
            features['price_change'] = df['price'].iloc[-1] - df['price'].iloc[0]
            features['price_change_pct'] = features['price_change'] / df['price'].iloc[0] * 100
            
            # 移动平均
            features['sma_5'] = df['price'].tail(5).mean()
            features['sma_20'] = df['price'].tail(20).mean() if len(df) >= 20 else df['price'].mean()
            
            # 波动率
            returns = df['price'].pct_change().dropna()
            features['volatility'] = returns.std() * np.sqrt(len(returns))
            
            # 动量指标
            if len(returns) > 0:
                features['momentum'] = returns.tail(10).mean()
                features['rsi'] = self._calculate_rsi(df['price'])
        
        if 'volume' in df.columns:
            # 成交量特征
            features['volume_current'] = df['volume'].iloc[-1]
            features['volume_ma'] = df['volume'].mean()
            features['volume_ratio'] = features['volume_current'] / features['volume_ma'] if features['volume_ma'] > 0 else 1
        
        # 时间特征
        features['time_since_update'] = time.time() - self.last_update
        features['data_freshness'] = len(self.data_buffer) / self.buffer_size
        
        return features
    
    def _update_feature_cache(self, tick_data: Dict) -> None:
        """更新特征缓存"""
        # 缓存关键特征以提高性能
        if 'price' in tick_data:
            self.feature_cache['last_price'] = tick_data['price']
        
        if 'volume' in tick_data:
            self.feature_cache['last_volume'] = tick_data['volume']
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

class HighFrequencyStrategy:
    """高频交易策略"""
    
    def __init__(self, strategy_name: str = "HFT_Strategy"):
        self.strategy_name = strategy_name
        self.positions = {}
        self.orders = {}
        self.pnl = 0.0
        self.trade_count = 0
        self.last_signal_time = 0
        self.min_signal_interval = 0.1  # 最小信号间隔(秒)
    
    def generate_signal(self, features: Dict) -> Dict:
        """
        生成交易信号
        
        Args:
            features: 特征字典
            
        Returns:
            交易信号
        """
        current_time = time.time()
        
        # 防止信号过于频繁
        if current_time - self.last_signal_time < self.min_signal_interval:
            return {'action': 'hold', 'confidence': 0.0}
        
        signal = {'action': 'hold', 'confidence': 0.0, 'quantity': 0}
        
        if not features:
            return signal
        
        # 动量策略
        momentum_signal = self._momentum_strategy(features)
        
        # 均值回归策略
        mean_reversion_signal = self._mean_reversion_strategy(features)
        
        # 成交量策略
        volume_signal = self._volume_strategy(features)
        
        # 组合信号
        combined_confidence = (
            momentum_signal['confidence'] * 0.4 +
            mean_reversion_signal['confidence'] * 0.4 +
            volume_signal['confidence'] * 0.2
        )
        
        if combined_confidence > 0.6:
            signal['action'] = 'buy'
            signal['confidence'] = combined_confidence
            signal['quantity'] = self._calculate_position_size(features, combined_confidence)
        elif combined_confidence < -0.6:
            signal['action'] = 'sell'
            signal['confidence'] = abs(combined_confidence)
            signal['quantity'] = self._calculate_position_size(features, abs(combined_confidence))
        
        self.last_signal_time = current_time
        return signal
    
    def _momentum_strategy(self, features: Dict) -> Dict:
        """动量策略"""
        if 'momentum' not in features or 'volatility' not in features:
            return {'confidence': 0.0}
        
        momentum = features['momentum']
        volatility = features['volatility']
        
        # 标准化动量
        if volatility > 0:
            normalized_momentum = momentum / volatility
            confidence = np.tanh(normalized_momentum * 5)  # 使用tanh函数限制范围
        else:
            confidence = 0.0
        
        return {'confidence': confidence}
    
    def _mean_reversion_strategy(self, features: Dict) -> Dict:
        """均值回归策略"""
        if 'price_current' not in features or 'sma_20' not in features:
            return {'confidence': 0.0}
        
        current_price = features['price_current']
        sma = features['sma_20']
        
        if sma > 0:
            deviation = (current_price - sma) / sma
            # 均值回归信号与价格偏离方向相反
            confidence = -np.tanh(deviation * 10)
        else:
            confidence = 0.0
        
        return {'confidence': confidence}
    
    def _volume_strategy(self, features: Dict) -> Dict:
        """成交量策略"""
        if 'volume_ratio' not in features:
            return {'confidence': 0.0}
        
        volume_ratio = features['volume_ratio']
        
        # 成交量异常时的信号
        if volume_ratio > 2.0:  # 成交量异常放大
            confidence = 0.3
        elif volume_ratio < 0.5:  # 成交量异常缩小
            confidence = -0.2
        else:
            confidence = 0.0
        
        return {'confidence': confidence}
    
    def _calculate_position_size(self, features: Dict, confidence: float) -> float:
        """计算仓位大小"""
        base_size = 100  # 基础仓位
        
        # 根据信心水平调整仓位
        size_multiplier = confidence
        
        # 根据波动率调整仓位
        if 'volatility' in features and features['volatility'] > 0:
            volatility_adjustment = 1.0 / (1.0 + features['volatility'])
            size_multiplier *= volatility_adjustment
        
        position_size = base_size * size_multiplier
        return max(0, min(position_size, 1000))  # 限制仓位范围

class AlgorithmOptimizer:
    """算法优化器"""
    
    def __init__(self):
        self.optimization_history = []
        self.current_params = {}
        self.performance_metrics = {}
    
    def optimize_strategy_parameters(self, 
                                   strategy: HighFrequencyStrategy,
                                   historical_data: List[Dict],
                                   optimization_target: str = 'sharpe_ratio') -> Dict:
        """
        优化策略参数
        
        Args:
            strategy: 策略对象
            historical_data: 历史数据
            optimization_target: 优化目标
            
        Returns:
            优化结果
        """
        best_params = {}
        best_score = -np.inf
        
        # 参数搜索空间
        param_space = {
            'min_signal_interval': [0.05, 0.1, 0.2, 0.5],
            'momentum_weight': [0.2, 0.4, 0.6, 0.8],
            'mean_reversion_weight': [0.2, 0.4, 0.6, 0.8],
            'volume_weight': [0.1, 0.2, 0.3, 0.4]
        }
        
        # 网格搜索
        for min_interval in param_space['min_signal_interval']:
            for mom_weight in param_space['momentum_weight']:
                for mr_weight in param_space['mean_reversion_weight']:
                    for vol_weight in param_space['volume_weight']:
                        
                        # 确保权重和为1
                        total_weight = mom_weight + mr_weight + vol_weight
                        if abs(total_weight - 1.0) > 0.1:
                            continue
                        
                        # 测试参数组合
                        test_params = {
                            'min_signal_interval': min_interval,
                            'momentum_weight': mom_weight / total_weight,
                            'mean_reversion_weight': mr_weight / total_weight,
                            'volume_weight': vol_weight / total_weight
                        }
                        
                        score = self._evaluate_parameters(strategy, historical_data, test_params, optimization_target)
                        
                        if score > best_score:
                            best_score = score
                            best_params = test_params.copy()
        
        # 记录优化历史
        optimization_result = {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_target': optimization_target,
            'timestamp': time.time()
        }
        
        self.optimization_history.append(optimization_result)
        self.current_params = best_params
        
        return optimization_result
    
    def _evaluate_parameters(self, 
                           strategy: HighFrequencyStrategy,
                           historical_data: List[Dict],
                           params: Dict,
                           target: str) -> float:
        """评估参数组合"""
        # 应用参数
        original_interval = strategy.min_signal_interval
        strategy.min_signal_interval = params['min_signal_interval']
        
        # 模拟交易
        returns = []
        positions = 0
        
        for data_point in historical_data:
            # 生成特征
            features = self._extract_features_from_data(data_point)
            
            # 生成信号
            signal = strategy.generate_signal(features)
            
            # 模拟执行
            if signal['action'] == 'buy' and positions <= 0:
                positions = signal['quantity']
            elif signal['action'] == 'sell' and positions >= 0:
                positions = -signal['quantity']
            
            # 计算收益
            if 'price_change_pct' in features and positions != 0:
                returns.append(positions * features['price_change_pct'] / 100)
        
        # 恢复原始参数
        strategy.min_signal_interval = original_interval
        
        # 计算目标指标
        if not returns:
            return -np.inf
        
        returns_array = np.array(returns)
        
        if target == 'sharpe_ratio':
            if returns_array.std() == 0:
                return 0
            return returns_array.mean() / returns_array.std()
        elif target == 'total_return':
            return returns_array.sum()
        elif target == 'max_drawdown':
            cumulative = np.cumsum(returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return -drawdown.min()  # 返回负值，因为我们要最小化回撤
        else:
            return returns_array.mean()
    
    def _extract_features_from_data(self, data_point: Dict) -> Dict:
        """从数据点提取特征"""
        # 简化的特征提取
        features = {}
        
        if 'price' in data_point:
            features['price_current'] = data_point['price']
            features['price_change_pct'] = data_point.get('price_change_pct', 0)
        
        if 'volume' in data_point:
            features['volume_ratio'] = data_point.get('volume_ratio', 1.0)
        
        features['momentum'] = data_point.get('momentum', 0)
        features['volatility'] = data_point.get('volatility', 0.01)
        features['sma_20'] = data_point.get('sma_20', features.get('price_current', 0))
        
        return features

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self, 
                        strategy: HighFrequencyStrategy,
                        data_processor: RealTimeDataProcessor,
                        update_interval: float = 1.0) -> None:
        """
        开始性能监控
        
        Args:
            strategy: 策略对象
            data_processor: 数据处理器
            update_interval: 更新间隔(秒)
        """
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(strategy, data_processor, update_interval)
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, 
                     strategy: HighFrequencyStrategy,
                     data_processor: RealTimeDataProcessor,
                     update_interval: float) -> None:
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集性能指标
                metrics = self._collect_metrics(strategy, data_processor)
                self.metrics_history.append(metrics)
                
                # 检查告警条件
                self._check_alerts(metrics)
                
                # 限制历史记录长度
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(update_interval)
    
    def _collect_metrics(self, 
                        strategy: HighFrequencyStrategy,
                        data_processor: RealTimeDataProcessor) -> Dict:
        """收集性能指标"""
        current_time = time.time()
        
        metrics = {
            'timestamp': current_time,
            'strategy_pnl': strategy.pnl,
            'trade_count': strategy.trade_count,
            'data_buffer_size': len(data_processor.data_buffer),
            'data_freshness': current_time - data_processor.last_update,
            'positions': len(strategy.positions),
            'active_orders': len(strategy.orders)
        }
        
        # 计算收益率指标
        if len(self.metrics_history) > 0:
            previous_pnl = self.metrics_history[-1]['strategy_pnl']
            metrics['pnl_change'] = strategy.pnl - previous_pnl
            
            # 计算最近的夏普比率
            if len(self.metrics_history) >= 30:
                recent_pnl_changes = [m.get('pnl_change', 0) for m in self.metrics_history[-30:]]
                pnl_array = np.array(recent_pnl_changes)
                if pnl_array.std() > 0:
                    metrics['recent_sharpe'] = pnl_array.mean() / pnl_array.std()
                else:
                    metrics['recent_sharpe'] = 0
        
        return metrics
    
    def _check_alerts(self, metrics: Dict) -> None:
        """检查告警条件"""
        alerts = []
        
        # 数据新鲜度告警
        if metrics['data_freshness'] > 5.0:  # 5秒没有新数据
            alerts.append({
                'type': 'data_stale',
                'message': f"数据已过期 {metrics['data_freshness']:.1f} 秒",
                'severity': 'warning',
                'timestamp': metrics['timestamp']
            })
        
        # PnL告警
        if metrics['strategy_pnl'] < -1000:  # 损失超过1000
            alerts.append({
                'type': 'high_loss',
                'message': f"策略损失过大: {metrics['strategy_pnl']:.2f}",
                'severity': 'critical',
                'timestamp': metrics['timestamp']
            })
        
        # 夏普比率告警
        if 'recent_sharpe' in metrics and metrics['recent_sharpe'] < -1.0:
            alerts.append({
                'type': 'poor_performance',
                'message': f"近期夏普比率过低: {metrics['recent_sharpe']:.2f}",
                'severity': 'warning',
                'timestamp': metrics['timestamp']
            })
        
        # 添加到告警列表
        self.alerts.extend(alerts)
        
        # 限制告警历史长度
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        # 计算统计指标
        pnl_history = [m['strategy_pnl'] for m in self.metrics_history]
        trade_counts = [m['trade_count'] for m in self.metrics_history]
        
        summary = {
            'current_pnl': latest_metrics['strategy_pnl'],
            'total_trades': latest_metrics['trade_count'],
            'max_pnl': max(pnl_history),
            'min_pnl': min(pnl_history),
            'pnl_volatility': np.std(pnl_history),
            'avg_trades_per_period': np.mean(np.diff(trade_counts)) if len(trade_counts) > 1 else 0,
            'data_quality': 'good' if latest_metrics['data_freshness'] < 2.0 else 'poor',
            'active_alerts': len([a for a in self.alerts if a['timestamp'] > time.time() - 300]),  # 5分钟内的告警
            'monitoring_duration': time.time() - self.metrics_history[0]['timestamp'] if self.metrics_history else 0
        }
        
        return summary

# 便捷函数
def create_terminal_ai_system(buffer_size: int = 10000) -> Dict:
    """
    创建Terminal AI系统
    
    Args:
        buffer_size: 数据缓冲区大小
        
    Returns:
        完整的Terminal AI系统
    """
    data_processor = RealTimeDataProcessor(buffer_size)
    strategy = HighFrequencyStrategy("Terminal_AI_Strategy")
    optimizer = AlgorithmOptimizer()
    monitor = PerformanceMonitor()
    
    return {
        'data_processor': data_processor,
        'strategy': strategy,
        'optimizer': optimizer,
        'monitor': monitor
    }

def run_terminal_ai_simulation(historical_data: List[Dict],
                              system_config: Dict = None) -> Dict:
    """
    运行Terminal AI模拟
    
    Args:
        historical_data: 历史数据
        system_config: 系统配置
        
    Returns:
        模拟结果
    """
    if system_config is None:
        system_config = {}
    
    # 创建系统
    system = create_terminal_ai_system(
        buffer_size=system_config.get('buffer_size', 10000)
    )
    
    data_processor = system['data_processor']
    strategy = system['strategy']
    optimizer = system['optimizer']
    monitor = system['monitor']
    
    # 优化策略参数
    if len(historical_data) > 100:
        optimization_result = optimizer.optimize_strategy_parameters(
            strategy, 
            historical_data[:100],  # 使用前100个数据点进行优化
            system_config.get('optimization_target', 'sharpe_ratio')
        )
    else:
        optimization_result = {'best_params': {}, 'best_score': 0}
    
    # 开始监控
    monitor.start_monitoring(strategy, data_processor, 0.1)
    
    # 模拟实时交易
    simulation_results = []
    
    try:
        for i, data_point in enumerate(historical_data):
            # 添加数据到处理器
            data_processor.add_tick_data(data_point)
            
            # 获取特征
            features = data_processor.get_latest_features()
            
            # 生成信号
            signal = strategy.generate_signal(features)
            
            # 记录结果
            result = {
                'timestamp': data_point.get('timestamp', i),
                'features': features,
                'signal': signal,
                'strategy_pnl': strategy.pnl
            }
            simulation_results.append(result)
            
            # 模拟延迟
            if system_config.get('simulate_latency', False):
                time.sleep(0.001)  # 1ms延迟
    
    finally:
        # 停止监控
        monitor.stop_monitoring()
    
    # 获取性能摘要
    performance_summary = monitor.get_performance_summary()
    
    return {
        'optimization_result': optimization_result,
        'simulation_results': simulation_results,
        'performance_summary': performance_summary,
        'system_components': system
    }