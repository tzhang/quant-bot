#!/usr/bin/env python3
"""
策略管理器
整合多种交易策略，提供统一的信号生成接口
支持策略组合、权重分配和动态调整
"""

import sys
import os
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
from enum import Enum
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.trading.enhanced_trading_engine import TradingSignal, OrderAction

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """策略类型"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PAIRS_TRADING = "pairs_trading"
    ML_PREDICTION = "ml_prediction"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT = "sentiment"

@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    strategy_type: StrategyType
    weight: float = 1.0
    enabled: bool = True
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
class BaseStrategy:
    """基础策略类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.strategy_type = config.strategy_type
        self.weight = config.weight
        self.enabled = config.enabled
        self.parameters = config.parameters
        
        # 历史数据
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.volume_history: Dict[str, List[int]] = defaultdict(list)
        self.signal_history: List[TradingSignal] = []
        
        # 性能统计
        self.stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0
        }
        
        logger.info(f"策略初始化: {self.name} ({self.strategy_type.value})")
    
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """生成交易信号 - 子类需要实现"""
        raise NotImplementedError("子类必须实现 generate_signals 方法")
    
    def update_data(self, symbol: str, data: MarketData):
        """更新市场数据"""
        self.price_history[symbol].append(data.close)
        self.volume_history[symbol].append(data.volume)
        
        # 保持历史数据长度
        max_history = self.parameters.get('max_history', 1000)
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
    
    def calculate_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """计算技术指标"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return {}
        
        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        
        indicators = {}
        
        try:
            # 移动平均线
            if len(prices) >= 5:
                indicators['sma_5'] = talib.SMA(prices, timeperiod=5)[-1]
            if len(prices) >= 10:
                indicators['sma_10'] = talib.SMA(prices, timeperiod=10)[-1]
            if len(prices) >= 20:
                indicators['sma_20'] = talib.SMA(prices, timeperiod=20)[-1]
                indicators['ema_20'] = talib.EMA(prices, timeperiod=20)[-1]
            
            # RSI
            if len(prices) >= 14:
                indicators['rsi'] = talib.RSI(prices, timeperiod=14)[-1]
            
            # MACD
            if len(prices) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(prices)
                indicators['macd'] = macd[-1]
                indicators['macd_signal'] = macd_signal[-1]
                indicators['macd_hist'] = macd_hist[-1]
            
            # 布林带
            if len(prices) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices, timeperiod=20)
                indicators['bb_upper'] = bb_upper[-1]
                indicators['bb_middle'] = bb_middle[-1]
                indicators['bb_lower'] = bb_lower[-1]
                indicators['bb_position'] = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # 成交量指标
            if len(volumes) >= 10:
                indicators['volume_sma'] = np.mean(volumes[-10:])
                indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma']
            
            # 波动率
            if len(prices) >= 20:
                returns = np.diff(prices) / prices[:-1]
                indicators['volatility'] = np.std(returns[-20:]) * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"计算技术指标错误: {e}")
        
        return indicators

class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        signals = []
        
        for symbol, data in market_data.items():
            self.update_data(symbol, data)
            
            if len(self.price_history[symbol]) < 20:
                continue
            
            indicators = self.calculate_technical_indicators(symbol)
            if not indicators:
                continue
            
            # 动量信号逻辑
            signal_strength = 0.0
            confidence = "LOW"
            
            # 价格动量
            if 'sma_5' in indicators and 'sma_20' in indicators:
                if indicators['sma_5'] > indicators['sma_20']:
                    signal_strength += 0.3
                else:
                    signal_strength -= 0.3
            
            # RSI动量
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi > 70:
                    signal_strength -= 0.2  # 超买
                elif rsi < 30:
                    signal_strength += 0.2  # 超卖
                elif 50 < rsi < 70:
                    signal_strength += 0.1  # 上升动量
            
            # MACD动量
            if 'macd' in indicators and 'macd_signal' in indicators:
                if indicators['macd'] > indicators['macd_signal']:
                    signal_strength += 0.2
                else:
                    signal_strength -= 0.2
            
            # 成交量确认
            if 'volume_ratio' in indicators:
                if indicators['volume_ratio'] > 1.5:
                    signal_strength *= 1.2  # 放量确认
            
            # 确定信号强度和置信度
            abs_strength = abs(signal_strength)
            if abs_strength > 0.6:
                confidence = "HIGH"
            elif abs_strength > 0.3:
                confidence = "MEDIUM"
            
            # 生成信号
            if signal_strength > 0.3:
                action = OrderAction.BUY
                quantity = self._calculate_position_size(symbol, data.close, signal_strength)
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    signal_strength=signal_strength,
                    confidence=confidence,
                    strategy_type=self.strategy_type.value
                )
                signals.append(signal)
                
            elif signal_strength < -0.3:
                # 卖出信号（如果有持仓）
                action = OrderAction.SELL
                quantity = self._calculate_position_size(symbol, data.close, abs(signal_strength))
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    signal_strength=abs(signal_strength),
                    confidence=confidence,
                    strategy_type=self.strategy_type.value
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_position_size(self, symbol: str, price: float, strength: float) -> int:
        """计算仓位大小"""
        base_size = self.parameters.get('base_position_size', 100)
        max_size = self.parameters.get('max_position_size', 500)
        
        # 根据信号强度调整仓位
        size = int(base_size * strength * 2)
        return min(size, max_size)

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        signals = []
        
        for symbol, data in market_data.items():
            self.update_data(symbol, data)
            
            if len(self.price_history[symbol]) < 20:
                continue
            
            indicators = self.calculate_technical_indicators(symbol)
            if not indicators:
                continue
            
            signal_strength = 0.0
            confidence = "LOW"
            
            # 布林带均值回归
            if 'bb_position' in indicators:
                bb_pos = indicators['bb_position']
                if bb_pos < 0.1:  # 接近下轨
                    signal_strength += 0.4
                elif bb_pos > 0.9:  # 接近上轨
                    signal_strength -= 0.4
            
            # RSI均值回归
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi < 25:  # 严重超卖
                    signal_strength += 0.3
                elif rsi > 75:  # 严重超买
                    signal_strength -= 0.3
            
            # 价格偏离均线程度
            if 'sma_20' in indicators:
                price_deviation = (data.close - indicators['sma_20']) / indicators['sma_20']
                if price_deviation < -0.05:  # 价格低于均线5%
                    signal_strength += 0.2
                elif price_deviation > 0.05:  # 价格高于均线5%
                    signal_strength -= 0.2
            
            # 确定信号强度和置信度
            abs_strength = abs(signal_strength)
            if abs_strength > 0.6:
                confidence = "HIGH"
            elif abs_strength > 0.3:
                confidence = "MEDIUM"
            
            # 生成信号
            if abs_strength > 0.3:
                action = OrderAction.BUY if signal_strength > 0 else OrderAction.SELL
                quantity = self._calculate_position_size(symbol, data.close, abs_strength)
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    signal_strength=abs_strength,
                    confidence=confidence,
                    strategy_type=self.strategy_type.value
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_position_size(self, symbol: str, price: float, strength: float) -> int:
        """计算仓位大小"""
        base_size = self.parameters.get('base_position_size', 100)
        max_size = self.parameters.get('max_position_size', 300)
        
        size = int(base_size * strength * 1.5)
        return min(size, max_size)

class BreakoutStrategy(BaseStrategy):
    """突破策略"""
    
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        signals = []
        
        for symbol, data in market_data.items():
            self.update_data(symbol, data)
            
            if len(self.price_history[symbol]) < 50:
                continue
            
            indicators = self.calculate_technical_indicators(symbol)
            if not indicators:
                continue
            
            signal_strength = 0.0
            confidence = "LOW"
            
            # 价格突破
            prices = self.price_history[symbol]
            recent_high = max(prices[-20:])
            recent_low = min(prices[-20:])
            
            if data.close > recent_high * 1.02:  # 向上突破
                signal_strength += 0.5
            elif data.close < recent_low * 0.98:  # 向下突破
                signal_strength -= 0.5
            
            # 成交量确认
            if 'volume_ratio' in indicators:
                if indicators['volume_ratio'] > 2.0:  # 放量突破
                    signal_strength *= 1.5
            
            # 布林带突破
            if 'bb_position' in indicators:
                bb_pos = indicators['bb_position']
                if bb_pos > 1.0:  # 突破上轨
                    signal_strength += 0.3
                elif bb_pos < 0.0:  # 突破下轨
                    signal_strength -= 0.3
            
            # 确定信号强度和置信度
            abs_strength = abs(signal_strength)
            if abs_strength > 0.8:
                confidence = "HIGH"
            elif abs_strength > 0.4:
                confidence = "MEDIUM"
            
            # 生成信号
            if abs_strength > 0.4:
                action = OrderAction.BUY if signal_strength > 0 else OrderAction.SELL
                quantity = self._calculate_position_size(symbol, data.close, abs_strength)
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    signal_strength=abs_strength,
                    confidence=confidence,
                    strategy_type=self.strategy_type.value
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_position_size(self, symbol: str, price: float, strength: float) -> int:
        """计算仓位大小"""
        base_size = self.parameters.get('base_position_size', 150)
        max_size = self.parameters.get('max_position_size', 600)
        
        size = int(base_size * strength * 2)
        return min(size, max_size)

class MLPredictionStrategy(BaseStrategy):
    """机器学习预测策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_history: Dict[str, List[List[float]]] = defaultdict(list)
        self.label_history: Dict[str, List[int]] = defaultdict(list)
    
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        signals = []
        
        for symbol, data in market_data.items():
            self.update_data(symbol, data)
            
            if len(self.price_history[symbol]) < 50:
                continue
            
            # 准备特征
            features = self._prepare_features(symbol)
            if features is None:
                continue
            
            # 训练模型（如果需要）
            if not self.is_trained and len(self.feature_history[symbol]) > 100:
                self._train_model(symbol)
            
            if not self.is_trained:
                continue
            
            # 预测
            try:
                features_scaled = self.scaler.transform([features])
                prediction = self.model.predict_proba(features_scaled)[0]
                
                # 预测概率转换为信号
                buy_prob = prediction[1] if len(prediction) > 1 else 0
                sell_prob = prediction[0] if len(prediction) > 0 else 0
                
                signal_strength = 0.0
                confidence = "LOW"
                
                if buy_prob > 0.7:
                    signal_strength = buy_prob
                    confidence = "HIGH" if buy_prob > 0.8 else "MEDIUM"
                    action = OrderAction.BUY
                elif sell_prob > 0.7:
                    signal_strength = sell_prob
                    confidence = "HIGH" if sell_prob > 0.8 else "MEDIUM"
                    action = OrderAction.SELL
                else:
                    continue
                
                quantity = self._calculate_position_size(symbol, data.close, signal_strength)
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    signal_strength=signal_strength,
                    confidence=confidence,
                    strategy_type=self.strategy_type.value
                )
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"ML预测错误: {e}")
        
        return signals
    
    def _prepare_features(self, symbol: str) -> Optional[List[float]]:
        """准备机器学习特征"""
        if len(self.price_history[symbol]) < 30:
            return None
        
        indicators = self.calculate_technical_indicators(symbol)
        if not indicators:
            return None
        
        features = []
        
        # 价格特征
        prices = self.price_history[symbol]
        returns = np.diff(prices) / prices[:-1]
        
        features.extend([
            returns[-1] if len(returns) > 0 else 0,  # 最新收益率
            np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # 5日平均收益率
            np.std(returns[-10:]) if len(returns) >= 10 else 0,  # 10日收益率标准差
        ])
        
        # 技术指标特征
        features.extend([
            indicators.get('rsi', 50) / 100,  # 归一化RSI
            indicators.get('bb_position', 0.5),  # 布林带位置
            indicators.get('macd', 0),  # MACD
            indicators.get('volume_ratio', 1),  # 成交量比率
        ])
        
        # 趋势特征
        if 'sma_5' in indicators and 'sma_20' in indicators:
            features.append((indicators['sma_5'] - indicators['sma_20']) / indicators['sma_20'])
        else:
            features.append(0)
        
        return features
    
    def _train_model(self, symbol: str):
        """训练机器学习模型"""
        if len(self.feature_history[symbol]) < 100:
            return
        
        try:
            X = np.array(self.feature_history[symbol])
            y = np.array(self.label_history[symbol])
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"ML模型训练完成: {symbol}, 样本数: {len(X)}")
            
        except Exception as e:
            logger.error(f"ML模型训练错误: {e}")
    
    def _calculate_position_size(self, symbol: str, price: float, strength: float) -> int:
        """计算仓位大小"""
        base_size = self.parameters.get('base_position_size', 100)
        max_size = self.parameters.get('max_position_size', 400)
        
        size = int(base_size * strength * 1.5)
        return min(size, max_size)

class StrategyManager:
    """策略管理器"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.market_data_cache: Dict[str, MarketData] = {}
        
        # 信号聚合设置
        self.signal_aggregation_method = "weighted_average"  # weighted_average, majority_vote, ensemble
        self.min_signal_strength = 0.3
        self.min_confidence_level = "MEDIUM"
        
        logger.info("策略管理器初始化完成")
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """添加策略"""
        self.strategies[strategy.name] = strategy
        self.strategy_weights[strategy.name] = weight
        logger.info(f"添加策略: {strategy.name}, 权重: {weight}")
    
    def remove_strategy(self, strategy_name: str):
        """移除策略"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            del self.strategy_weights[strategy_name]
            logger.info(f"移除策略: {strategy_name}")
    
    def update_strategy_weight(self, strategy_name: str, weight: float):
        """更新策略权重"""
        if strategy_name in self.strategy_weights:
            self.strategy_weights[strategy_name] = weight
            logger.info(f"更新策略权重: {strategy_name} -> {weight}")
    
    def update_market_data(self, symbol: str, data: MarketData):
        """更新市场数据"""
        self.market_data_cache[symbol] = data
        
        # 更新所有策略的数据
        for strategy in self.strategies.values():
            if strategy.enabled:
                strategy.update_data(symbol, data)
    
    def generate_aggregated_signals(self, symbols: List[str] = None) -> List[TradingSignal]:
        """生成聚合交易信号"""
        if symbols is None:
            symbols = list(self.market_data_cache.keys())
        
        # 收集所有策略的信号
        all_signals: Dict[str, List[TradingSignal]] = defaultdict(list)
        
        for strategy_name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue
            
            try:
                # 过滤市场数据
                filtered_data = {
                    symbol: data for symbol, data in self.market_data_cache.items()
                    if symbol in symbols
                }
                
                signals = strategy.generate_signals(filtered_data)
                
                for signal in signals:
                    # 应用策略权重
                    signal.signal_strength *= self.strategy_weights[strategy_name]
                    all_signals[signal.symbol].append(signal)
                
            except Exception as e:
                logger.error(f"策略 {strategy_name} 生成信号错误: {e}")
        
        # 聚合信号
        aggregated_signals = []
        
        for symbol, signals in all_signals.items():
            if not signals:
                continue
            
            aggregated_signal = self._aggregate_signals(symbol, signals)
            if aggregated_signal:
                aggregated_signals.append(aggregated_signal)
        
        return aggregated_signals
    
    def _aggregate_signals(self, symbol: str, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """聚合单个标的的信号"""
        if not signals:
            return None
        
        if self.signal_aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(symbol, signals)
        elif self.signal_aggregation_method == "majority_vote":
            return self._majority_vote_aggregation(symbol, signals)
        elif self.signal_aggregation_method == "ensemble":
            return self._ensemble_aggregation(symbol, signals)
        else:
            return signals[0]  # 默认返回第一个信号
    
    def _weighted_average_aggregation(self, symbol: str, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """加权平均聚合"""
        buy_signals = [s for s in signals if s.action == OrderAction.BUY]
        sell_signals = [s for s in signals if s.action == OrderAction.SELL]
        
        buy_strength = sum(s.signal_strength for s in buy_signals)
        sell_strength = sum(s.signal_strength for s in sell_signals)
        
        net_strength = buy_strength - sell_strength
        
        if abs(net_strength) < self.min_signal_strength:
            return None
        
        # 确定最终动作和强度
        action = OrderAction.BUY if net_strength > 0 else OrderAction.SELL
        final_strength = abs(net_strength)
        
        # 确定置信度
        confidence = "LOW"
        if final_strength > 0.8:
            confidence = "HIGH"
        elif final_strength > 0.5:
            confidence = "MEDIUM"
        
        if confidence == "LOW" and self.min_confidence_level in ["MEDIUM", "HIGH"]:
            return None
        if confidence == "MEDIUM" and self.min_confidence_level == "HIGH":
            return None
        
        # 计算平均数量
        avg_quantity = int(np.mean([s.quantity for s in signals]))
        
        # 选择主要策略类型
        strategy_types = [s.strategy_type for s in signals]
        main_strategy = max(set(strategy_types), key=strategy_types.count)
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            quantity=avg_quantity,
            signal_strength=final_strength,
            confidence=confidence,
            strategy_type=f"AGGREGATED_{main_strategy}"
        )
    
    def _majority_vote_aggregation(self, symbol: str, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """多数投票聚合"""
        buy_votes = len([s for s in signals if s.action == OrderAction.BUY])
        sell_votes = len([s for s in signals if s.action == OrderAction.SELL])
        
        if buy_votes == sell_votes:
            return None  # 平票，不生成信号
        
        action = OrderAction.BUY if buy_votes > sell_votes else OrderAction.SELL
        vote_ratio = max(buy_votes, sell_votes) / len(signals)
        
        # 根据投票比例确定信号强度
        signal_strength = vote_ratio
        
        if signal_strength < self.min_signal_strength:
            return None
        
        # 确定置信度
        confidence = "LOW"
        if vote_ratio > 0.8:
            confidence = "HIGH"
        elif vote_ratio > 0.6:
            confidence = "MEDIUM"
        
        # 计算平均数量
        relevant_signals = [s for s in signals if s.action == action]
        avg_quantity = int(np.mean([s.quantity for s in relevant_signals]))
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            quantity=avg_quantity,
            signal_strength=signal_strength,
            confidence=confidence,
            strategy_type="MAJORITY_VOTE"
        )
    
    def _ensemble_aggregation(self, symbol: str, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """集成聚合（结合加权平均和多数投票）"""
        # 先进行多数投票
        majority_signal = self._majority_vote_aggregation(symbol, signals)
        
        # 再进行加权平均
        weighted_signal = self._weighted_average_aggregation(symbol, signals)
        
        # 如果两种方法都没有信号，返回None
        if not majority_signal and not weighted_signal:
            return None
        
        # 如果只有一种方法有信号，返回该信号
        if not majority_signal:
            return weighted_signal
        if not weighted_signal:
            return majority_signal
        
        # 如果两种方法都有信号且方向一致，增强信号
        if majority_signal.action == weighted_signal.action:
            final_strength = (majority_signal.signal_strength + weighted_signal.signal_strength) / 2
            final_quantity = (majority_signal.quantity + weighted_signal.quantity) // 2
            
            confidence = "HIGH" if final_strength > 0.7 else "MEDIUM"
            
            return TradingSignal(
                symbol=symbol,
                action=majority_signal.action,
                quantity=final_quantity,
                signal_strength=final_strength,
                confidence=confidence,
                strategy_type="ENSEMBLE"
            )
        
        # 如果方向不一致，返回强度更高的信号
        return majority_signal if majority_signal.signal_strength > weighted_signal.signal_strength else weighted_signal
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """获取策略性能统计"""
        performance = {}
        
        for name, strategy in self.strategies.items():
            performance[name] = {
                'name': name,
                'type': strategy.strategy_type.value,
                'weight': self.strategy_weights[name],
                'enabled': strategy.enabled,
                'stats': strategy.stats.copy()
            }
        
        return performance
    
    def optimize_strategy_weights(self, performance_data: Dict[str, float]):
        """根据性能数据优化策略权重"""
        total_performance = sum(performance_data.values())
        
        if total_performance <= 0:
            return
        
        # 根据相对性能调整权重
        for strategy_name, performance in performance_data.items():
            if strategy_name in self.strategy_weights:
                relative_performance = performance / total_performance
                new_weight = max(0.1, min(2.0, relative_performance * len(performance_data)))
                self.strategy_weights[strategy_name] = new_weight
                
                logger.info(f"优化策略权重: {strategy_name} -> {new_weight:.2f}")

def create_default_strategies() -> List[BaseStrategy]:
    """创建默认策略组合"""
    strategies = []
    
    # 动量策略
    momentum_config = StrategyConfig(
        name="momentum_strategy",
        strategy_type=StrategyType.MOMENTUM,
        weight=1.0,
        parameters={
            'base_position_size': 100,
            'max_position_size': 500,
            'max_history': 1000
        }
    )
    strategies.append(MomentumStrategy(momentum_config))
    
    # 均值回归策略
    mean_reversion_config = StrategyConfig(
        name="mean_reversion_strategy",
        strategy_type=StrategyType.MEAN_REVERSION,
        weight=0.8,
        parameters={
            'base_position_size': 100,
            'max_position_size': 300,
            'max_history': 1000
        }
    )
    strategies.append(MeanReversionStrategy(mean_reversion_config))
    
    # 突破策略
    breakout_config = StrategyConfig(
        name="breakout_strategy",
        strategy_type=StrategyType.BREAKOUT,
        weight=1.2,
        parameters={
            'base_position_size': 150,
            'max_position_size': 600,
            'max_history': 1000
        }
    )
    strategies.append(BreakoutStrategy(breakout_config))
    
    # 机器学习策略
    ml_config = StrategyConfig(
        name="ml_prediction_strategy",
        strategy_type=StrategyType.ML_PREDICTION,
        weight=0.9,
        parameters={
            'base_position_size': 100,
            'max_position_size': 400,
            'max_history': 1000
        }
    )
    strategies.append(MLPredictionStrategy(ml_config))
    
    return strategies

def main():
    """示例用法"""
    # 创建策略管理器
    manager = StrategyManager()
    
    # 添加默认策略
    strategies = create_default_strategies()
    for strategy in strategies:
        manager.add_strategy(strategy)
    
    # 模拟市场数据
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    for i in range(100):
        for symbol in symbols:
            # 生成模拟数据
            base_price = 100 + i * 0.1
            price_change = np.random.normal(0, 0.02)
            current_price = base_price * (1 + price_change)
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=current_price * 0.99,
                high=current_price * 1.01,
                low=current_price * 0.98,
                close=current_price,
                volume=int(np.random.normal(1000000, 200000))
            )
            
            manager.update_market_data(symbol, market_data)
        
        # 每10次生成一次信号
        if i % 10 == 0:
            signals = manager.generate_aggregated_signals()
            
            print(f"\n=== 第 {i//10 + 1} 轮信号 ===")
            for signal in signals:
                print(f"{signal.symbol}: {signal.action.value} {signal.quantity} "
                      f"(强度: {signal.signal_strength:.2f}, 置信度: {signal.confidence}, "
                      f"策略: {signal.strategy_type})")
        
        time.sleep(0.1)
    
    # 显示策略性能
    performance = manager.get_strategy_performance()
    print("\n=== 策略性能 ===")
    for name, stats in performance.items():
        print(f"{name}: 权重={stats['weight']:.2f}, 启用={stats['enabled']}")

if __name__ == "__main__":
    main()