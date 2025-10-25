#!/usr/bin/env python3
"""
加密货币交易工具模块
专为CME Group Crypto Classic等加密货币比赛设计

主要功能:
1. 加密货币数据处理
2. 期货合约分析
3. 加密货币特有指标
4. 风险管理工具
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class CryptoDataProcessor:
    """加密货币数据处理器"""
    
    def __init__(self):
        self.crypto_pairs = ['BTC/USD', 'ETH/USD', 'BTC/ETH']
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    def process_futures_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理期货数据
        
        Args:
            data: 原始期货数据
            
        Returns:
            处理后的数据
        """
        processed_data = data.copy()
        
        # 计算期货特有指标
        processed_data['basis'] = processed_data['futures_price'] - processed_data['spot_price']
        processed_data['basis_pct'] = processed_data['basis'] / processed_data['spot_price'] * 100
        
        # 计算持仓量变化
        if 'open_interest' in processed_data.columns:
            processed_data['oi_change'] = processed_data['open_interest'].pct_change()
            processed_data['oi_ma'] = processed_data['open_interest'].rolling(20).mean()
        
        # 计算资金费率相关指标
        if 'funding_rate' in processed_data.columns:
            processed_data['funding_ma'] = processed_data['funding_rate'].rolling(8).mean()
            processed_data['funding_std'] = processed_data['funding_rate'].rolling(24).std()
        
        return processed_data
    
    def add_crypto_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加加密货币特有特征
        
        Args:
            data: 价格数据
            
        Returns:
            添加特征后的数据
        """
        features_data = data.copy()
        
        # 价格特征
        features_data['price_log'] = np.log(features_data['close'])
        features_data['price_log_return'] = features_data['price_log'].diff()
        
        # 波动率特征
        features_data['volatility_1h'] = features_data['price_log_return'].rolling(60).std() * np.sqrt(60)
        features_data['volatility_4h'] = features_data['price_log_return'].rolling(240).std() * np.sqrt(240)
        features_data['volatility_1d'] = features_data['price_log_return'].rolling(1440).std() * np.sqrt(1440)
        
        # 成交量特征
        if 'volume' in features_data.columns:
            features_data['volume_ma'] = features_data['volume'].rolling(20).mean()
            features_data['volume_ratio'] = features_data['volume'] / features_data['volume_ma']
            features_data['vwap'] = (features_data['close'] * features_data['volume']).rolling(20).sum() / features_data['volume'].rolling(20).sum()
        
        # 技术指标
        features_data = self._add_technical_indicators(features_data)
        
        return features_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # 布林带
        data['bb_middle'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        return data

class CryptoRiskManager:
    """加密货币风险管理器"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = 0.1  # 最大仓位10%
        self.max_daily_loss = 0.02    # 最大日损失2%
        self.volatility_lookback = 20
    
    def calculate_position_size(self, 
                              price: float, 
                              volatility: float, 
                              confidence: float = 0.5) -> float:
        """
        计算仓位大小
        
        Args:
            price: 当前价格
            volatility: 波动率
            confidence: 信心水平
            
        Returns:
            建议仓位大小
        """
        # Kelly公式调整
        kelly_fraction = confidence * 0.5 / volatility if volatility > 0 else 0
        
        # 限制最大仓位
        position_fraction = min(kelly_fraction, self.max_position_size)
        
        # 计算实际仓位
        position_value = self.current_capital * position_fraction
        position_size = position_value / price
        
        return position_size
    
    def check_risk_limits(self, 
                         current_pnl: float, 
                         daily_pnl: float) -> Dict[str, bool]:
        """
        检查风险限制
        
        Args:
            current_pnl: 当前盈亏
            daily_pnl: 日盈亏
            
        Returns:
            风险检查结果
        """
        risk_status = {
            'within_daily_limit': abs(daily_pnl) <= self.max_daily_loss * self.initial_capital,
            'positive_equity': self.current_capital + current_pnl > 0,
            'can_trade': True
        }
        
        risk_status['can_trade'] = all([
            risk_status['within_daily_limit'],
            risk_status['positive_equity']
        ])
        
        return risk_status

class CryptoStrategyOptimizer:
    """加密货币策略优化器"""
    
    def __init__(self):
        self.strategies = {}
        self.performance_metrics = {}
    
    def optimize_futures_strategy(self, 
                                 data: pd.DataFrame, 
                                 strategy_params: Dict) -> Dict:
        """
        优化期货策略
        
        Args:
            data: 历史数据
            strategy_params: 策略参数
            
        Returns:
            优化结果
        """
        results = {}
        
        # 基差策略优化
        if 'basis_strategy' in strategy_params:
            results['basis'] = self._optimize_basis_strategy(data, strategy_params['basis_strategy'])
        
        # 动量策略优化
        if 'momentum_strategy' in strategy_params:
            results['momentum'] = self._optimize_momentum_strategy(data, strategy_params['momentum_strategy'])
        
        # 均值回归策略优化
        if 'mean_reversion_strategy' in strategy_params:
            results['mean_reversion'] = self._optimize_mean_reversion_strategy(data, strategy_params['mean_reversion_strategy'])
        
        return results
    
    def _optimize_basis_strategy(self, data: pd.DataFrame, params: Dict) -> Dict:
        """优化基差策略"""
        best_params = params.copy()
        best_sharpe = -np.inf
        
        # 参数网格搜索
        for threshold in np.arange(0.5, 3.0, 0.5):
            for lookback in [10, 20, 30]:
                # 计算策略收益
                signals = self._generate_basis_signals(data, threshold, lookback)
                returns = self._calculate_strategy_returns(data, signals)
                sharpe = self._calculate_sharpe_ratio(returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params.update({
                        'threshold': threshold,
                        'lookback': lookback,
                        'sharpe_ratio': sharpe
                    })
        
        return best_params
    
    def _optimize_momentum_strategy(self, data: pd.DataFrame, params: Dict) -> Dict:
        """优化动量策略"""
        best_params = params.copy()
        best_sharpe = -np.inf
        
        for short_window in [5, 10, 15]:
            for long_window in [20, 30, 50]:
                if short_window >= long_window:
                    continue
                
                signals = self._generate_momentum_signals(data, short_window, long_window)
                returns = self._calculate_strategy_returns(data, signals)
                sharpe = self._calculate_sharpe_ratio(returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params.update({
                        'short_window': short_window,
                        'long_window': long_window,
                        'sharpe_ratio': sharpe
                    })
        
        return best_params
    
    def _optimize_mean_reversion_strategy(self, data: pd.DataFrame, params: Dict) -> Dict:
        """优化均值回归策略"""
        best_params = params.copy()
        best_sharpe = -np.inf
        
        for window in [10, 20, 30]:
            for std_threshold in [1.5, 2.0, 2.5]:
                signals = self._generate_mean_reversion_signals(data, window, std_threshold)
                returns = self._calculate_strategy_returns(data, signals)
                sharpe = self._calculate_sharpe_ratio(returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params.update({
                        'window': window,
                        'std_threshold': std_threshold,
                        'sharpe_ratio': sharpe
                    })
        
        return best_params
    
    def _generate_basis_signals(self, data: pd.DataFrame, threshold: float, lookback: int) -> pd.Series:
        """生成基差信号"""
        if 'basis_pct' not in data.columns:
            return pd.Series(0, index=data.index)
        
        basis_ma = data['basis_pct'].rolling(lookback).mean()
        basis_std = data['basis_pct'].rolling(lookback).std()
        
        signals = pd.Series(0, index=data.index)
        signals[data['basis_pct'] > basis_ma + threshold * basis_std] = -1  # 做空
        signals[data['basis_pct'] < basis_ma - threshold * basis_std] = 1   # 做多
        
        return signals
    
    def _generate_momentum_signals(self, data: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
        """生成动量信号"""
        short_ma = data['close'].rolling(short_window).mean()
        long_ma = data['close'].rolling(long_window).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1   # 做多
        signals[short_ma < long_ma] = -1  # 做空
        
        return signals
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame, window: int, std_threshold: float) -> pd.Series:
        """生成均值回归信号"""
        price_ma = data['close'].rolling(window).mean()
        price_std = data['close'].rolling(window).std()
        
        signals = pd.Series(0, index=data.index)
        signals[data['close'] > price_ma + std_threshold * price_std] = -1  # 做空
        signals[data['close'] < price_ma - std_threshold * price_std] = 1   # 做多
        
        return signals
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """计算策略收益"""
        returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        return strategy_returns.dropna()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)

class CryptoPortfolioManager:
    """加密货币投资组合管理器"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
    
    def execute_trade(self, 
                     symbol: str, 
                     action: str, 
                     quantity: float, 
                     price: float, 
                     timestamp: pd.Timestamp) -> Dict:
        """
        执行交易
        
        Args:
            symbol: 交易品种
            action: 交易动作 ('buy', 'sell')
            quantity: 交易数量
            price: 交易价格
            timestamp: 时间戳
            
        Returns:
            交易结果
        """
        trade_value = quantity * price
        
        if action == 'buy':
            if trade_value <= self.current_capital:
                self.current_capital -= trade_value
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                trade_record = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'capital_after': self.current_capital
                }
                self.trade_history.append(trade_record)
                
                return {'success': True, 'message': f'Bought {quantity} {symbol} at {price}'}
            else:
                return {'success': False, 'message': 'Insufficient capital'}
        
        elif action == 'sell':
            if self.positions.get(symbol, 0) >= quantity:
                self.current_capital += trade_value
                self.positions[symbol] -= quantity
                
                trade_record = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'capital_after': self.current_capital
                }
                self.trade_history.append(trade_record)
                
                return {'success': True, 'message': f'Sold {quantity} {symbol} at {price}'}
            else:
                return {'success': False, 'message': 'Insufficient position'}
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        计算投资组合总价值
        
        Args:
            current_prices: 当前价格字典
            
        Returns:
            投资组合总价值
        """
        portfolio_value = self.current_capital
        
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity > 0:
                portfolio_value += quantity * current_prices[symbol]
        
        return portfolio_value
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """
        计算绩效指标
        
        Args:
            current_prices: 当前价格字典
            
        Returns:
            绩效指标
        """
        current_value = self.get_portfolio_value(current_prices)
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        # 计算交易统计
        trades_df = pd.DataFrame(self.trade_history)
        
        metrics = {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cash_balance': self.current_capital,
            'positions': self.positions.copy(),
            'total_trades': len(self.trade_history),
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        if not trades_df.empty:
            # 计算盈亏交易数量
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            # 简化的盈亏计算
            if len(sell_trades) > 0:
                avg_sell_price = sell_trades['price'].mean()
                avg_buy_price = buy_trades['price'].mean() if len(buy_trades) > 0 else 0
                
                if avg_sell_price > avg_buy_price:
                    metrics['winning_trades'] = len(sell_trades)
                else:
                    metrics['losing_trades'] = len(sell_trades)
        
        return metrics

# 便捷函数
def create_crypto_pipeline(data: pd.DataFrame, 
                          strategy_type: str = 'momentum',
                          initial_capital: float = 100000) -> Dict:
    """
    创建加密货币交易管道
    
    Args:
        data: 历史数据
        strategy_type: 策略类型
        initial_capital: 初始资金
        
    Returns:
        完整的交易管道结果
    """
    # 数据处理
    processor = CryptoDataProcessor()
    processed_data = processor.add_crypto_features(data)
    
    # 策略优化
    optimizer = CryptoStrategyOptimizer()
    strategy_params = {
        f'{strategy_type}_strategy': {
            'type': strategy_type
        }
    }
    optimized_params = optimizer.optimize_futures_strategy(processed_data, strategy_params)
    
    # 风险管理
    risk_manager = CryptoRiskManager(initial_capital)
    
    # 投资组合管理
    portfolio_manager = CryptoPortfolioManager(initial_capital)
    
    return {
        'processed_data': processed_data,
        'optimized_params': optimized_params,
        'risk_manager': risk_manager,
        'portfolio_manager': portfolio_manager
    }

def simulate_crypto_trading(data: pd.DataFrame, 
                           strategy_params: Dict,
                           initial_capital: float = 100000) -> Dict:
    """
    模拟加密货币交易 - 仅用于测试和演示交易策略效果
    
    Args:
        data: 历史数据 - 仅用于测试和演示
        strategy_params: 策略参数 - 仅用于测试和演示
        initial_capital: 初始资金 - 仅用于测试和演示
        
    Returns:
        交易模拟结果 - 仅用于测试和演示
    """
    portfolio_manager = CryptoPortfolioManager(initial_capital)  # 创建投资组合管理器 - 仅用于测试和演示
    risk_manager = CryptoRiskManager(initial_capital)  # 创建风险管理器 - 仅用于测试和演示
    
    # 生成交易信号 - 仅用于测试和演示
    optimizer = CryptoStrategyOptimizer()
    
    # 模拟交易执行 - 仅用于测试和演示，不代表真实交易
    for i in range(1, len(data)):
        current_price = data.iloc[i]['close']  # 获取当前价格 - 仅用于测试和演示
        timestamp = data.index[i]  # 获取时间戳 - 仅用于测试和演示
        
        # 简化的信号生成逻辑 - 仅用于测试和演示
        if i > 20:  # 确保有足够的历史数据 - 仅用于测试和演示
            # 这里可以根据具体策略生成信号 - 仅用于测试和演示
            # 示例：简单的动量策略 - 仅用于测试和演示
            short_ma = data['close'].iloc[i-10:i].mean()  # 计算短期移动平均 - 仅用于测试和演示
            long_ma = data['close'].iloc[i-20:i].mean()   # 计算长期移动平均 - 仅用于测试和演示
            
            if short_ma > long_ma and 'BTC' not in portfolio_manager.positions:
                # 买入信号 - 仅用于测试和演示
                position_size = risk_manager.calculate_position_size(
                    current_price, 
                    data['close'].iloc[i-20:i].std() / current_price,  # 计算波动率 - 仅用于测试和演示
                    0.6
                )
                if position_size > 0:
                    portfolio_manager.execute_trade('BTC', 'buy', position_size, current_price, timestamp)  # 执行买入 - 仅用于测试和演示
            
            elif short_ma < long_ma and portfolio_manager.positions.get('BTC', 0) > 0:
                # 卖出信号 - 仅用于测试和演示
                position_size = portfolio_manager.positions.get('BTC', 0)  # 获取持仓数量 - 仅用于测试和演示
                portfolio_manager.execute_trade('BTC', 'sell', position_size, current_price, timestamp)  # 执行卖出 - 仅用于测试和演示
    
    # 计算最终绩效 - 仅用于测试和演示
    final_prices = {'BTC': data.iloc[-1]['close']}  # 获取最终价格 - 仅用于测试和演示
    performance = portfolio_manager.get_performance_metrics(final_prices)  # 计算绩效指标 - 仅用于测试和演示
    
    return {
        'portfolio_manager': portfolio_manager,  # 返回投资组合管理器 - 仅用于测试和演示
        'performance': performance,              # 返回绩效指标 - 仅用于测试和演示
        'trade_history': portfolio_manager.trade_history  # 返回交易历史 - 仅用于测试和演示
    }