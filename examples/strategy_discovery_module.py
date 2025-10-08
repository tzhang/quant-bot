"""
策略发现模块 - 自动化策略生成和评估
提供策略模板、参数优化、回测评估、策略组合等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import logging
from abc import ABC, abstractmethod
import itertools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategySignal:
    """策略信号"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 信号强度 0-1
    confidence: float  # 信号置信度 0-1
    factors: Dict[str, float]  # 相关因子值
    metadata: Dict[str, Any] = None

@dataclass
class StrategyPerformance:
    """策略表现"""
    strategy_name: str
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    trade_count: int
    avg_trade_duration: float

@dataclass
class StrategyParameters:
    """策略参数"""
    name: str
    parameters: Dict[str, Any]
    optimization_bounds: Dict[str, Tuple[float, float]]
    
class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[StrategySignal]:
        """生成交易信号"""
        pass
        
    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取参数优化边界"""
        pass

class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'lookback_period': 20,
            'momentum_threshold': 0.02,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Momentum Strategy", default_params)
        
    def generate_signals(self, data: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[StrategySignal]:
        """生成动量策略信号"""
        signals = []
        
        try:
            # 获取因子
            momentum = factors.get('momentum_20d', pd.Series())
            rsi = factors.get('rsi', pd.Series())
            volume_ratio = factors.get('volume_ratio', pd.Series())
            
            for i in range(len(data)):
                timestamp = data.index[i]
                
                # 检查数据有效性
                if (i < self.parameters['lookback_period'] or 
                    pd.isna(momentum.iloc[i]) or 
                    pd.isna(rsi.iloc[i])):
                    continue
                    
                # 动量信号
                momentum_val = momentum.iloc[i]
                rsi_val = rsi.iloc[i]
                volume_val = volume_ratio.iloc[i] if not pd.isna(volume_ratio.iloc[i]) else 1.0
                
                signal_type = 'hold'
                strength = 0.0
                confidence = 0.0
                
                # 买入信号
                if (momentum_val > self.parameters['momentum_threshold'] and 
                    rsi_val < self.parameters['rsi_overbought'] and
                    volume_val > self.parameters['volume_threshold']):
                    signal_type = 'buy'
                    strength = min(momentum_val / self.parameters['momentum_threshold'], 1.0)
                    confidence = (100 - rsi_val) / 100 * 0.7 + min(volume_val / 2, 1.0) * 0.3
                    
                # 卖出信号
                elif (momentum_val < -self.parameters['momentum_threshold'] or 
                      rsi_val > self.parameters['rsi_overbought']):
                    signal_type = 'sell'
                    strength = min(abs(momentum_val) / self.parameters['momentum_threshold'], 1.0)
                    confidence = max(rsi_val - 50, 0) / 50 * 0.8 + 0.2
                    
                if signal_type != 'hold':
                    signal = StrategySignal(
                        timestamp=timestamp,
                        symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                        signal_type=signal_type,
                        strength=strength,
                        confidence=confidence,
                        factors={
                            'momentum': momentum_val,
                            'rsi': rsi_val,
                            'volume_ratio': volume_val
                        }
                    )
                    signals.append(signal)
                    
        except Exception as e:
            self.logger.error(f"生成动量策略信号失败: {e}")
            
        return signals
        
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取参数优化边界"""
        return {
            'lookback_period': (5, 50),
            'momentum_threshold': (0.005, 0.05),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'volume_threshold': (1.0, 3.0)
        }

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'bb_threshold': 0.1,  # 布林带阈值
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'price_deviation_threshold': 2.0,
            'volume_confirmation': True
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Mean Reversion Strategy", default_params)
        
    def generate_signals(self, data: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[StrategySignal]:
        """生成均值回归策略信号"""
        signals = []
        
        try:
            # 获取因子
            bb_position = factors.get('bb_position', pd.Series())
            rsi = factors.get('rsi', pd.Series())
            price_deviation = factors.get('price_deviation', pd.Series())
            volume_ratio = factors.get('volume_ratio', pd.Series())
            
            for i in range(len(data)):
                timestamp = data.index[i]
                
                # 检查数据有效性
                if (pd.isna(bb_position.iloc[i]) or 
                    pd.isna(rsi.iloc[i]) or
                    pd.isna(price_deviation.iloc[i])):
                    continue
                    
                bb_val = bb_position.iloc[i]
                rsi_val = rsi.iloc[i]
                deviation_val = price_deviation.iloc[i]
                volume_val = volume_ratio.iloc[i] if not pd.isna(volume_ratio.iloc[i]) else 1.0
                
                signal_type = 'hold'
                strength = 0.0
                confidence = 0.0
                
                # 超卖买入信号
                if (bb_val < self.parameters['bb_threshold'] and 
                    rsi_val < self.parameters['rsi_oversold'] and
                    deviation_val < -self.parameters['price_deviation_threshold']):
                    
                    volume_confirm = not self.parameters['volume_confirmation'] or volume_val > 1.2
                    if volume_confirm:
                        signal_type = 'buy'
                        strength = (self.parameters['rsi_oversold'] - rsi_val) / self.parameters['rsi_oversold']
                        confidence = (1 - bb_val) * 0.4 + (self.parameters['rsi_oversold'] - rsi_val) / 50 * 0.6
                        
                # 超买卖出信号
                elif (bb_val > (1 - self.parameters['bb_threshold']) and 
                      rsi_val > self.parameters['rsi_overbought'] and
                      deviation_val > self.parameters['price_deviation_threshold']):
                    
                    volume_confirm = not self.parameters['volume_confirmation'] or volume_val > 1.2
                    if volume_confirm:
                        signal_type = 'sell'
                        strength = (rsi_val - self.parameters['rsi_overbought']) / (100 - self.parameters['rsi_overbought'])
                        confidence = bb_val * 0.4 + (rsi_val - 50) / 50 * 0.6
                        
                if signal_type != 'hold':
                    signal = StrategySignal(
                        timestamp=timestamp,
                        symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                        signal_type=signal_type,
                        strength=strength,
                        confidence=confidence,
                        factors={
                            'bb_position': bb_val,
                            'rsi': rsi_val,
                            'price_deviation': deviation_val,
                            'volume_ratio': volume_val
                        }
                    )
                    signals.append(signal)
                    
        except Exception as e:
            self.logger.error(f"生成均值回归策略信号失败: {e}")
            
        return signals
        
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取参数优化边界"""
        return {
            'bb_threshold': (0.05, 0.2),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'price_deviation_threshold': (1.0, 3.0)
        }

class MLStrategy(BaseStrategy):
    """机器学习策略"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'model_type': 'random_forest',  # 'random_forest', 'gradient_boosting'
            'lookback_window': 20,
            'prediction_horizon': 5,
            'feature_importance_threshold': 0.01,
            'signal_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("ML Strategy", default_params)
        self.model = None
        self.feature_names = []
        
    def train_model(self, data: pd.DataFrame, factors: Dict[str, pd.Series], returns: pd.Series):
        """训练机器学习模型"""
        try:
            # 准备特征数据
            features_df = pd.DataFrame(factors)
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            # 准备标签数据（未来收益方向）
            future_returns = returns.shift(-self.parameters['prediction_horizon'])
            labels = (future_returns > 0).astype(int)
            
            # 移除无效数据
            valid_idx = ~(labels.isna() | features_df.isna().any(axis=1))
            X = features_df[valid_idx]
            y = labels[valid_idx]
            
            if len(X) < 100:  # 数据量太少
                self.logger.warning("训练数据不足，无法训练模型")
                return False
                
            # 选择模型
            if self.parameters['model_type'] == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
                
            # 训练模型
            self.model.fit(X, y)
            self.feature_names = X.columns.tolist()
            
            # 评估模型
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            self.logger.info(f"模型训练完成，准确率: {accuracy:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练ML模型失败: {e}")
            return False
            
    def generate_signals(self, data: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[StrategySignal]:
        """生成ML策略信号"""
        signals = []
        
        try:
            if self.model is None:
                self.logger.warning("模型未训练，无法生成信号")
                return signals
                
            # 准备特征数据
            features_df = pd.DataFrame(factors)
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            # 确保特征顺序一致
            if not all(col in features_df.columns for col in self.feature_names):
                self.logger.warning("特征不匹配，无法生成信号")
                return signals
                
            X = features_df[self.feature_names]
            
            # 预测
            predictions = self.model.predict_proba(X)
            
            for i in range(len(data)):
                timestamp = data.index[i]
                
                if i >= len(predictions):
                    break
                    
                # 获取预测概率
                prob_up = predictions[i][1] if len(predictions[i]) > 1 else 0.5
                prob_down = predictions[i][0] if len(predictions[i]) > 0 else 0.5
                
                signal_type = 'hold'
                strength = 0.0
                confidence = 0.0
                
                # 生成信号
                if prob_up > self.parameters['signal_threshold']:
                    signal_type = 'buy'
                    strength = prob_up
                    confidence = prob_up
                elif prob_down > self.parameters['signal_threshold']:
                    signal_type = 'sell'
                    strength = prob_down
                    confidence = prob_down
                    
                if signal_type != 'hold':
                    # 获取重要特征值
                    important_factors = {}
                    if hasattr(self.model, 'feature_importances_'):
                        importances = self.model.feature_importances_
                        for j, feature in enumerate(self.feature_names):
                            if importances[j] > self.parameters['feature_importance_threshold']:
                                important_factors[feature] = X.iloc[i, j]
                                
                    signal = StrategySignal(
                        timestamp=timestamp,
                        symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                        signal_type=signal_type,
                        strength=strength,
                        confidence=confidence,
                        factors=important_factors,
                        metadata={'prob_up': prob_up, 'prob_down': prob_down}
                    )
                    signals.append(signal)
                    
        except Exception as e:
            self.logger.error(f"生成ML策略信号失败: {e}")
            
        return signals
        
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取参数优化边界"""
        return {
            'lookback_window': (10, 50),
            'prediction_horizon': (1, 10),
            'signal_threshold': (0.5, 0.8)
        }

class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def grid_search_optimization(self, strategy: BaseStrategy, data: pd.DataFrame, 
                               factors: Dict[str, pd.Series], returns: pd.Series,
                               optimization_steps: int = 10) -> Dict[str, Any]:
        """网格搜索优化"""
        try:
            bounds = strategy.get_parameter_bounds()
            best_params = strategy.parameters.copy()
            best_performance = -np.inf
            
            # 生成参数网格
            param_grids = {}
            for param, (min_val, max_val) in bounds.items():
                if isinstance(strategy.parameters.get(param), int):
                    param_grids[param] = np.linspace(min_val, max_val, optimization_steps, dtype=int)
                else:
                    param_grids[param] = np.linspace(min_val, max_val, optimization_steps)
                    
            # 网格搜索
            param_combinations = list(itertools.product(*param_grids.values()))
            
            for combination in param_combinations[:100]:  # 限制搜索次数
                # 设置参数
                test_params = dict(zip(param_grids.keys(), combination))
                strategy.parameters.update(test_params)
                
                # 生成信号并评估
                signals = strategy.generate_signals(data, factors)
                performance = self._evaluate_signals(signals, data, returns)
                
                if performance > best_performance:
                    best_performance = performance
                    best_params.update(test_params)
                    
            # 恢复最佳参数
            strategy.parameters = best_params
            
            return {
                'best_parameters': best_params,
                'best_performance': best_performance,
                'optimization_method': 'grid_search'
            }
            
        except Exception as e:
            self.logger.error(f"网格搜索优化失败: {e}")
            return {}
            
    def _evaluate_signals(self, signals: List[StrategySignal], data: pd.DataFrame, 
                         returns: pd.Series) -> float:
        """评估信号质量"""
        try:
            if not signals:
                return -1.0
                
            # 简单的信号评估：计算信号后的平均收益
            signal_returns = []
            
            for signal in signals:
                # 找到信号时间点
                signal_idx = data.index.get_loc(signal.timestamp) if signal.timestamp in data.index else None
                if signal_idx is None or signal_idx >= len(returns) - 1:
                    continue
                    
                # 计算信号后的收益
                future_return = returns.iloc[signal_idx + 1]
                
                if signal.signal_type == 'buy':
                    signal_returns.append(future_return * signal.strength)
                elif signal.signal_type == 'sell':
                    signal_returns.append(-future_return * signal.strength)
                    
            if signal_returns:
                return np.mean(signal_returns)
            else:
                return -1.0
                
        except Exception as e:
            self.logger.error(f"评估信号失败: {e}")
            return -1.0

class StrategyBacktester:
    """策略回测器"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
    def backtest_strategy(self, strategy: BaseStrategy, data: pd.DataFrame, 
                         factors: Dict[str, pd.Series], 
                         transaction_cost: float = 0.001) -> StrategyPerformance:
        """回测策略"""
        try:
            # 生成信号
            signals = strategy.generate_signals(data, factors)
            
            if not signals:
                return StrategyPerformance(
                    strategy_name=strategy.name,
                    total_return=0, annual_return=0, volatility=0,
                    sharpe_ratio=0, max_drawdown=0, win_rate=0,
                    profit_factor=0, calmar_ratio=0, sortino_ratio=0,
                    trade_count=0, avg_trade_duration=0
                )
                
            # 模拟交易
            portfolio_value = [self.initial_capital]
            positions = {}  # {symbol: position_size}
            trades = []
            
            for signal in signals:
                # 找到信号时间点的价格
                if signal.timestamp not in data.index:
                    continue
                    
                price = data.loc[signal.timestamp, 'close']
                symbol = signal.symbol
                
                # 计算仓位大小
                position_size = self._calculate_position_size(
                    signal, portfolio_value[-1], price
                )
                
                # 执行交易
                if signal.signal_type == 'buy':
                    if symbol not in positions:
                        positions[symbol] = 0
                    positions[symbol] += position_size
                    cost = position_size * price * (1 + transaction_cost)
                    portfolio_value.append(portfolio_value[-1] - cost)
                    
                elif signal.signal_type == 'sell':
                    if symbol in positions and positions[symbol] > 0:
                        sell_size = min(position_size, positions[symbol])
                        positions[symbol] -= sell_size
                        revenue = sell_size * price * (1 - transaction_cost)
                        portfolio_value.append(portfolio_value[-1] + revenue)
                        
                        # 记录交易
                        trades.append({
                            'timestamp': signal.timestamp,
                            'symbol': symbol,
                            'type': signal.signal_type,
                            'size': sell_size,
                            'price': price
                        })
                        
            # 计算最终组合价值（平仓所有持仓）
            final_value = portfolio_value[-1]
            for symbol, position in positions.items():
                if position > 0:
                    final_price = data.loc[data.index[-1], 'close']
                    final_value += position * final_price
                    
            # 计算性能指标
            returns = pd.Series(portfolio_value).pct_change().dropna()
            
            total_return = (final_value - self.initial_capital) / self.initial_capital
            annual_return = total_return * (252 / len(data)) if len(data) > 0 else 0
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            # 胜率
            win_rate = (returns > 0).mean() if len(returns) > 0 else 0
            
            # 盈亏比
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            profit_factor = (winning_trades.sum() / abs(losing_trades.sum()) 
                           if len(losing_trades) > 0 and losing_trades.sum() < 0 else 0)
            
            # Calmar比率
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Sortino比率
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
            
            return StrategyPerformance(
                strategy_name=strategy.name,
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                trade_count=len(trades),
                avg_trade_duration=0  # 简化处理
            )
            
        except Exception as e:
            self.logger.error(f"策略回测失败: {e}")
            return StrategyPerformance(
                strategy_name=strategy.name,
                total_return=0, annual_return=0, volatility=0,
                sharpe_ratio=0, max_drawdown=0, win_rate=0,
                profit_factor=0, calmar_ratio=0, sortino_ratio=0,
                trade_count=0, avg_trade_duration=0
            )
            
    def _calculate_position_size(self, signal: StrategySignal, portfolio_value: float, 
                               price: float) -> float:
        """计算仓位大小"""
        try:
            # 基于信号强度和置信度的仓位管理
            base_position_ratio = 0.1  # 基础仓位比例
            max_position_ratio = 0.3   # 最大仓位比例
            
            position_ratio = base_position_ratio * signal.strength * signal.confidence
            position_ratio = min(position_ratio, max_position_ratio)
            
            position_value = portfolio_value * position_ratio
            position_size = position_value / price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return 0.0

class StrategyDiscoveryEngine:
    """策略发现引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategies = {}
        self.optimizer = StrategyOptimizer()
        self.backtester = StrategyBacktester()
        
        # 注册默认策略
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """注册默认策略"""
        self.strategies['momentum'] = MomentumStrategy()
        self.strategies['mean_reversion'] = MeanReversionStrategy()
        self.strategies['ml'] = MLStrategy()
        
    def discover_strategies(self, data: pd.DataFrame, factors: Dict[str, pd.Series], 
                          returns: pd.Series) -> List[Tuple[str, StrategyPerformance]]:
        """发现最佳策略"""
        try:
            strategy_performances = []
            
            for strategy_name, strategy in self.strategies.items():
                self.logger.info(f"测试策略: {strategy_name}")
                
                # 优化策略参数
                optimization_result = self.optimizer.grid_search_optimization(
                    strategy, data, factors, returns
                )
                
                # 回测策略
                performance = self.backtester.backtest_strategy(strategy, data, factors)
                
                strategy_performances.append((strategy_name, performance))
                
                self.logger.info(f"策略 {strategy_name} - 夏普比率: {performance.sharpe_ratio:.3f}, "
                               f"年化收益: {performance.annual_return:.3f}")
                
            # 按夏普比率排序
            strategy_performances.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)
            
            return strategy_performances
            
        except Exception as e:
            self.logger.error(f"策略发现失败: {e}")
            return []
            
    def create_ensemble_strategy(self, strategies: List[str], weights: List[float] = None) -> 'EnsembleStrategy':
        """创建集成策略"""
        try:
            if weights is None:
                weights = [1.0 / len(strategies)] * len(strategies)
                
            selected_strategies = [self.strategies[name] for name in strategies if name in self.strategies]
            
            return EnsembleStrategy(selected_strategies, weights)
            
        except Exception as e:
            self.logger.error(f"创建集成策略失败: {e}")
            return None
            
    def get_strategy_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        try:
            summary = {
                'total_strategies': len(self.strategies),
                'strategy_types': list(self.strategies.keys()),
                'timestamp': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取策略摘要失败: {e}")
            return {}

class EnsembleStrategy(BaseStrategy):
    """集成策略"""
    
    def __init__(self, strategies: List[BaseStrategy], weights: List[float]):
        self.strategies = strategies
        self.weights = weights
        super().__init__("Ensemble Strategy", {})
        
    def generate_signals(self, data: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[StrategySignal]:
        """生成集成策略信号"""
        try:
            all_signals = {}  # {timestamp: [signals]}
            
            # 收集所有策略的信号
            for strategy in self.strategies:
                signals = strategy.generate_signals(data, factors)
                for signal in signals:
                    if signal.timestamp not in all_signals:
                        all_signals[signal.timestamp] = []
                    all_signals[signal.timestamp].append(signal)
                    
            # 合并信号
            ensemble_signals = []
            for timestamp, signals in all_signals.items():
                if len(signals) >= 2:  # 至少需要2个策略同意
                    ensemble_signal = self._combine_signals(signals, timestamp)
                    if ensemble_signal:
                        ensemble_signals.append(ensemble_signal)
                        
            return ensemble_signals
            
        except Exception as e:
            self.logger.error(f"生成集成策略信号失败: {e}")
            return []
            
    def _combine_signals(self, signals: List[StrategySignal], timestamp: datetime) -> Optional[StrategySignal]:
        """合并信号"""
        try:
            # 按信号类型分组
            buy_signals = [s for s in signals if s.signal_type == 'buy']
            sell_signals = [s for s in signals if s.signal_type == 'sell']
            
            # 计算加权投票
            buy_weight = sum(s.strength * s.confidence for s in buy_signals)
            sell_weight = sum(s.strength * s.confidence for s in sell_signals)
            
            if buy_weight > sell_weight and buy_weight > 0.5:
                signal_type = 'buy'
                strength = buy_weight / len(buy_signals)
                confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            elif sell_weight > buy_weight and sell_weight > 0.5:
                signal_type = 'sell'
                strength = sell_weight / len(sell_signals)
                confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
            else:
                return None
                
            # 合并因子
            combined_factors = {}
            for signal in signals:
                combined_factors.update(signal.factors)
                
            return StrategySignal(
                timestamp=timestamp,
                symbol=signals[0].symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                factors=combined_factors,
                metadata={'component_strategies': len(signals)}
            )
            
        except Exception as e:
            self.logger.error(f"合并信号失败: {e}")
            return None
            
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取参数优化边界"""
        return {}  # 集成策略不需要参数优化