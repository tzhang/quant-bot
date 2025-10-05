#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标模块
提供各种技术分析指标和交易策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    strength: float  # 信号强度 0-1
    price: float
    timestamp: str
    indicators: Dict[str, float]
    reason: str

class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        简单移动平均线 (Simple Moving Average)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            SMA值
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        指数移动平均线 (Exponential Moving Average)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            EMA值
        """
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        相对强弱指数 (Relative Strength Index)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            RSI值
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD指标 (Moving Average Convergence Divergence)
        
        Args:
            data: 价格数据
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            包含MACD线、信号线和柱状图的字典
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        布林带 (Bollinger Bands)
        
        Args:
            data: 价格数据
            period: 周期
            std_dev: 标准差倍数
            
        Returns:
            包含上轨、中轨、下轨的字典
        """
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        随机指标 (Stochastic Oscillator)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            k_period: K线周期
            d_period: D线周期
            
        Returns:
            包含%K和%D的字典
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        平均真实波幅 (Average True Range)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            ATR值
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        威廉指标 (Williams %R)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            Williams %R值
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        商品通道指数 (Commodity Channel Index)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            CCI值
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        能量潮指标 (On-Balance Volume)
        
        Args:
            close: 收盘价
            volume: 成交量
            
        Returns:
            OBV值
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

class TradingStrategies:
    """交易策略"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def moving_average_crossover(self, data: pd.DataFrame, fast_period: int = 10, 
                                slow_period: int = 20) -> List[TradingSignal]:
        """
        移动平均线交叉策略
        
        Args:
            data: 包含OHLCV数据的DataFrame
            fast_period: 快线周期
            slow_period: 慢线周期
            
        Returns:
            交易信号列表
        """
        signals = []
        close = data['close']
        
        fast_ma = self.indicators.sma(close, fast_period)
        slow_ma = self.indicators.sma(close, slow_period)
        
        for i in range(1, len(data)):
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
                
            # 金叉：快线上穿慢线
            if (fast_ma.iloc[i] > slow_ma.iloc[i] and 
                fast_ma.iloc[i-1] <= slow_ma.iloc[i-1]):
                
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    strength=0.7,
                    price=close.iloc[i],
                    timestamp=str(data.index[i]),
                    indicators={
                        'fast_ma': fast_ma.iloc[i],
                        'slow_ma': slow_ma.iloc[i]
                    },
                    reason=f"快线({fast_period})上穿慢线({slow_period})"
                )
                signals.append(signal)
            
            # 死叉：快线下穿慢线
            elif (fast_ma.iloc[i] < slow_ma.iloc[i] and 
                  fast_ma.iloc[i-1] >= slow_ma.iloc[i-1]):
                
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    strength=0.7,
                    price=close.iloc[i],
                    timestamp=str(data.index[i]),
                    indicators={
                        'fast_ma': fast_ma.iloc[i],
                        'slow_ma': slow_ma.iloc[i]
                    },
                    reason=f"快线({fast_period})下穿慢线({slow_period})"
                )
                signals.append(signal)
        
        return signals
    
    def rsi_strategy(self, data: pd.DataFrame, period: int = 14, 
                     oversold: float = 30, overbought: float = 70) -> List[TradingSignal]:
        """
        RSI策略
        
        Args:
            data: 包含OHLCV数据的DataFrame
            period: RSI周期
            oversold: 超卖阈值
            overbought: 超买阈值
            
        Returns:
            交易信号列表
        """
        signals = []
        close = data['close']
        rsi = self.indicators.rsi(close, period)
        
        for i in range(1, len(data)):
            if pd.isna(rsi.iloc[i]):
                continue
            
            # RSI从超卖区域向上突破
            if rsi.iloc[i] > oversold and rsi.iloc[i-1] <= oversold:
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    strength=min(1.0, (oversold - rsi.iloc[i-1]) / 10 + 0.5),
                    price=close.iloc[i],
                    timestamp=str(data.index[i]),
                    indicators={'rsi': rsi.iloc[i]},
                    reason=f"RSI从超卖区域({oversold})向上突破"
                )
                signals.append(signal)
            
            # RSI从超买区域向下突破
            elif rsi.iloc[i] < overbought and rsi.iloc[i-1] >= overbought:
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    strength=min(1.0, (rsi.iloc[i-1] - overbought) / 10 + 0.5),
                    price=close.iloc[i],
                    timestamp=str(data.index[i]),
                    indicators={'rsi': rsi.iloc[i]},
                    reason=f"RSI从超买区域({overbought})向下突破"
                )
                signals.append(signal)
        
        return signals
    
    def macd_strategy(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        MACD策略
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号列表
        """
        signals = []
        close = data['close']
        macd_data = self.indicators.macd(close)
        
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        histogram = macd_data['histogram']
        
        for i in range(1, len(data)):
            if (pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]) or 
                pd.isna(histogram.iloc[i])):
                continue
            
            # MACD线上穿信号线
            if (macd_line.iloc[i] > signal_line.iloc[i] and 
                macd_line.iloc[i-1] <= signal_line.iloc[i-1]):
                
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    strength=min(1.0, abs(histogram.iloc[i]) / 2 + 0.5),
                    price=close.iloc[i],
                    timestamp=str(data.index[i]),
                    indicators={
                        'macd': macd_line.iloc[i],
                        'signal': signal_line.iloc[i],
                        'histogram': histogram.iloc[i]
                    },
                    reason="MACD线上穿信号线"
                )
                signals.append(signal)
            
            # MACD线下穿信号线
            elif (macd_line.iloc[i] < signal_line.iloc[i] and 
                  macd_line.iloc[i-1] >= signal_line.iloc[i-1]):
                
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    strength=min(1.0, abs(histogram.iloc[i]) / 2 + 0.5),
                    price=close.iloc[i],
                    timestamp=str(data.index[i]),
                    indicators={
                        'macd': macd_line.iloc[i],
                        'signal': signal_line.iloc[i],
                        'histogram': histogram.iloc[i]
                    },
                    reason="MACD线下穿信号线"
                )
                signals.append(signal)
        
        return signals
    
    def bollinger_bands_strategy(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        布林带策略
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号列表
        """
        signals = []
        close = data['close']
        bb_data = self.indicators.bollinger_bands(close)
        
        upper_band = bb_data['upper']
        middle_band = bb_data['middle']
        lower_band = bb_data['lower']
        
        for i in range(1, len(data)):
            if (pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]) or 
                pd.isna(middle_band.iloc[i])):
                continue
            
            # 价格从下轨反弹
            if (close.iloc[i] > lower_band.iloc[i] and 
                close.iloc[i-1] <= lower_band.iloc[i-1]):
                
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    strength=0.6,
                    price=close.iloc[i],
                    timestamp=str(data.index[i]),
                    indicators={
                        'price': close.iloc[i],
                        'upper_band': upper_band.iloc[i],
                        'middle_band': middle_band.iloc[i],
                        'lower_band': lower_band.iloc[i]
                    },
                    reason="价格从布林带下轨反弹"
                )
                signals.append(signal)
            
            # 价格从上轨回落
            elif (close.iloc[i] < upper_band.iloc[i] and 
                  close.iloc[i-1] >= upper_band.iloc[i-1]):
                
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    strength=0.6,
                    price=close.iloc[i],
                    timestamp=str(data.index[i]),
                    indicators={
                        'price': close.iloc[i],
                        'upper_band': upper_band.iloc[i],
                        'middle_band': middle_band.iloc[i],
                        'lower_band': lower_band.iloc[i]
                    },
                    reason="价格从布林带上轨回落"
                )
                signals.append(signal)
        
        return signals
    
    def multi_indicator_strategy(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        多指标综合策略
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            交易信号列表
        """
        # 获取各个策略的信号
        ma_signals = self.moving_average_crossover(data)
        rsi_signals = self.rsi_strategy(data)
        macd_signals = self.macd_strategy(data)
        bb_signals = self.bollinger_bands_strategy(data)
        
        # 合并所有信号
        all_signals = ma_signals + rsi_signals + macd_signals + bb_signals
        
        # 按时间排序
        all_signals.sort(key=lambda x: x.timestamp)
        
        # 信号过滤和强化
        filtered_signals = []
        signal_window = {}  # 用于存储时间窗口内的信号
        
        for signal in all_signals:
            timestamp = signal.timestamp
            
            # 清理过期信号（超过1小时）
            expired_keys = []
            for key in signal_window:
                if abs(pd.to_datetime(timestamp) - pd.to_datetime(key)) > pd.Timedelta(hours=1):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del signal_window[key]
            
            # 添加当前信号
            if timestamp not in signal_window:
                signal_window[timestamp] = []
            signal_window[timestamp].append(signal)
            
            # 检查是否有多个指标确认
            buy_count = sum(1 for s in signal_window[timestamp] if s.signal_type == SignalType.BUY)
            sell_count = sum(1 for s in signal_window[timestamp] if s.signal_type == SignalType.SELL)
            
            # 如果有多个指标确认，增强信号强度
            if buy_count >= 2:
                enhanced_signal = TradingSignal(
                    symbol=signal.symbol,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, signal.strength + 0.3),
                    price=signal.price,
                    timestamp=timestamp,
                    indicators=signal.indicators,
                    reason=f"多指标确认买入信号({buy_count}个)"
                )
                filtered_signals.append(enhanced_signal)
            
            elif sell_count >= 2:
                enhanced_signal = TradingSignal(
                    symbol=signal.symbol,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, signal.strength + 0.3),
                    price=signal.price,
                    timestamp=timestamp,
                    indicators=signal.indicators,
                    reason=f"多指标确认卖出信号({sell_count}个)"
                )
                filtered_signals.append(enhanced_signal)
        
        return filtered_signals

class StrategyBacktester:
    """策略回测器"""
    
    def __init__(self, initial_capital: float = 100000):
        """
        初始化回测器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def backtest_strategy(self, data: pd.DataFrame, signals: List[TradingSignal]) -> Dict:
        """
        回测策略
        
        Args:
            data: 历史数据
            signals: 交易信号列表
            
        Returns:
            回测结果
        """
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        for signal in signals:
            self._execute_signal(signal)
        
        # 计算回测指标
        return self._calculate_metrics()
    
    def _execute_signal(self, signal: TradingSignal):
        """执行交易信号"""
        symbol = signal.symbol
        price = signal.price
        
        if signal.signal_type == SignalType.BUY:
            # 买入信号
            if symbol not in self.positions:
                # 根据信号强度决定仓位大小
                position_size = int(self.capital * signal.strength * 0.1 / price)
                if position_size > 0:
                    cost = position_size * price
                    if cost <= self.capital:
                        self.positions[symbol] = {
                            'quantity': position_size,
                            'avg_price': price,
                            'total_cost': cost
                        }
                        self.capital -= cost
                        
                        self.trades.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': position_size,
                            'price': price,
                            'timestamp': signal.timestamp,
                            'reason': signal.reason
                        })
        
        elif signal.signal_type == SignalType.SELL:
            # 卖出信号
            if symbol in self.positions:
                position = self.positions[symbol]
                quantity = position['quantity']
                revenue = quantity * price
                
                self.capital += revenue
                profit = revenue - position['total_cost']
                
                self.trades.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': signal.timestamp,
                    'profit': profit,
                    'reason': signal.reason
                })
                
                del self.positions[symbol]
        
        # 记录权益曲线
        total_value = self.capital + sum(
            pos['quantity'] * price for pos in self.positions.values()
        )
        self.equity_curve.append({
            'timestamp': signal.timestamp,
            'total_value': total_value,
            'cash': self.capital,
            'positions_value': total_value - self.capital
        })
    
    def _calculate_metrics(self) -> Dict:
        """计算回测指标"""
        if not self.equity_curve:
            return {}
        
        # 计算总收益
        final_value = self.equity_curve[-1]['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 计算交易统计
        profitable_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit', 0) < 0]
        
        win_rate = len(profitable_trades) / len(self.trades) if self.trades else 0
        
        avg_profit = np.mean([t['profit'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # 生成模拟股价数据
    price = 100
    prices = [price]
    for _ in range(99):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    data.set_index('date', inplace=True)
    data['symbol'] = 'TEST'
    
    # 创建策略实例
    strategy = TradingStrategies()
    
    # 生成交易信号
    signals = strategy.multi_indicator_strategy(data)
    
    print(f"生成了 {len(signals)} 个交易信号")
    for signal in signals[:5]:  # 显示前5个信号
        print(f"{signal.timestamp}: {signal.signal_type.value} - {signal.reason} (强度: {signal.strength:.2f})")
    
    # 回测策略
    backtester = StrategyBacktester(initial_capital=100000)
    results = backtester.backtest_strategy(data, signals)
    
    print("\n回测结果:")
    print(f"初始资金: ${results['initial_capital']:,.2f}")
    print(f"最终价值: ${results['final_value']:,.2f}")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"盈亏比: {results['profit_factor']:.2f}")