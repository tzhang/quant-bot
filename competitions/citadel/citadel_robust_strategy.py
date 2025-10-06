#!/usr/bin/env python3
"""
Citadel 稳健优化高频交易策略
修复交易逻辑问题，确保稳定收益
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.risk.risk_manager import RiskManager
from src.utils.logger import get_logger

class RobustCitadelHFTStrategy:
    """稳健优化的Citadel高频交易策略"""
    
    def __init__(self, config_path: str = None):
        """初始化策略"""
        self.logger = get_logger(__name__)
        
        # 稳健优化的策略参数
        self.config = {
            "strategy_name": "RobustCitadelHFT",
            "version": "1.0",
            "description": "稳健优化的Citadel高频交易策略",
            "signal_parameters": {
                "lookback_period": 20,  # 增加回看期
                "signal_threshold": 0.6,  # 提高信号阈值
                "position_limit": 0.05,  # 降低仓位限制
                "max_trade_size": 5000,  # 降低交易规模
                "min_signal_strength": 0.7,  # 最小信号强度
                "signal_decay": 0.95  # 信号衰减因子
            },
            "risk_management": {
                "stop_loss": 0.01,  # 1% 止损
                "take_profit": 0.025,  # 2.5% 止盈
                "max_portfolio_risk": 0.01,  # 降低组合风险
                "max_single_position": 0.05,
                "max_daily_trades": 50,  # 限制每日交易次数
                "cooldown_period": 10  # 交易冷却期（分钟）
            },
            "technical_indicators": {
                "rsi_period": 14,
                "bb_period": 20,
                "bb_std_multiplier": 2.0,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "volatility_window": 20,
                "volume_window": 20
            },
            "signal_weights": {
                "momentum": 0.25,
                "mean_reversion": 0.30,
                "volatility": 0.20,
                "volume": 0.15,
                "microstructure": 0.10
            },
            "market_conditions": {
                "min_volume_threshold": 5000,  # 提高最小成交量要求
                "max_spread_threshold": 0.005,  # 降低最大价差
                "volatility_filter": True,
                "market_hours_only": False,
                "min_price": 10.0,  # 最小价格过滤
                "max_volatility": 0.05  # 最大波动率过滤
            },
            "optimization_settings": {
                "adaptive_thresholds": True,
                "dynamic_position_sizing": True,
                "regime_detection": True,
                "correlation_filter": True
            },
            "execution_settings": {
                "slippage": 0.0002,  # 增加滑点成本
                "commission": 0.0002,  # 增加手续费
                "market_impact": 0.0001
            }
        }
        
        # 初始化风险管理器
        self.risk_manager = RiskManager()
        
        # 初始化交易状态
        self.initial_capital = 1000000  # 初始资金100万
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_trades = 0
        self.last_trade_time = None
        
        self.logger.info("🚀 稳健优化Citadel高频交易策略初始化完成")
    
    def load_market_data(self, data_path: str = None) -> pd.DataFrame:
        """加载市场数据"""
        if data_path is None:
            data_path = "competitions/citadel/data/sample_market_data.csv"
        
        try:
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                self.logger.info(f"✅ 数据文件加载成功: {data_path}")
            else:
                # 生成模拟高频数据
                self.logger.info("📊 生成模拟高频市场数据...")
                data = self._generate_sample_data()
                
                # 确保目录存在
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                data.to_csv(data_path, index=False)
                self.logger.info(f"💾 模拟数据已保存到: {data_path}")
            
            # 数据预处理
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # 数据质量过滤
            data = data[data['close'] >= self.config['market_conditions']['min_price']]
            data = data.reset_index(drop=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    def _generate_sample_data(self, n_points: int = 10000) -> pd.DataFrame:
        """生成模拟高频交易数据（减少数据量）"""
        np.random.seed(42)
        
        # 生成时间序列
        start_time = datetime.now() - timedelta(days=1)
        timestamps = pd.date_range(start=start_time, periods=n_points, freq='30S')
        
        # 生成价格数据（几何布朗运动）
        initial_price = 100.0
        dt = 1/252/24/60/2  # 30秒间隔
        mu = 0.05  # 年化收益率
        sigma = 0.15  # 降低波动率
        
        # 生成价格路径
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_points)
        prices = [initial_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        # 生成其他市场数据
        data = pd.DataFrame({
            'timestamp': timestamps,
            'asset_id': ['ASSET_000'] * n_points,
            'open': prices,
            'high': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.001, n_points))),
            'low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.001, n_points))),
            'close': prices,
            'volume': np.random.lognormal(9, 0.5, n_points).astype(int),  # 增加成交量
            'bid_price': np.array(prices) * (1 - np.random.uniform(0.0001, 0.0005, n_points)),
            'ask_price': np.array(prices) * (1 + np.random.uniform(0.0001, 0.0005, n_points)),
            'bid_size': np.random.lognormal(6, 0.5, n_points).astype(int),
            'ask_size': np.random.lognormal(6, 0.5, n_points).astype(int)
        })
        
        return data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # RSI
        rsi_period = self.config['technical_indicators']['rsi_period']
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        bb_period = self.config['technical_indicators']['bb_period']
        bb_std = self.config['technical_indicators']['bb_std_multiplier']
        data['bb_middle'] = data['close'].rolling(window=bb_period).mean()
        bb_std_dev = data['close'].rolling(window=bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std_dev * bb_std)
        data['bb_lower'] = data['bb_middle'] - (bb_std_dev * bb_std)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # MACD
        macd_fast = self.config['technical_indicators']['macd_fast']
        macd_slow = self.config['technical_indicators']['macd_slow']
        macd_signal = self.config['technical_indicators']['macd_signal']
        
        ema_fast = data['close'].ewm(span=macd_fast).mean()
        ema_slow = data['close'].ewm(span=macd_slow).mean()
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=macd_signal).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # 移动平均线
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['ema_10'] = data['close'].ewm(span=10).mean()
        
        # 波动率
        vol_window = self.config['technical_indicators']['volatility_window']
        data['volatility'] = data['close'].pct_change().rolling(window=vol_window).std()
        
        # 成交量指标
        vol_window = self.config['technical_indicators']['volume_window']
        data['volume_sma'] = data['volume'].rolling(window=vol_window).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # 价差
        if 'bid_price' in data.columns and 'ask_price' in data.columns:
            data['spread'] = (data['ask_price'] - data['bid_price']) / data['close']
        else:
            data['spread'] = 0.001
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        signals = []
        lookback = self.config['signal_parameters']['lookback_period']
        
        for idx in range(lookback, len(data)):
            current_row = data.iloc[idx]
            
            # 检查数据完整性
            required_fields = ['rsi', 'bb_position', 'macd_histogram', 'volatility', 'volume_ratio']
            if any(pd.isna(current_row[field]) for field in required_fields):
                signals.append(0)
                continue
            
            # 市场条件预过滤
            if (current_row['volume'] < self.config['market_conditions']['min_volume_threshold'] or
                current_row['spread'] > self.config['market_conditions']['max_spread_threshold'] or
                current_row['volatility'] > self.config['market_conditions']['max_volatility']):
                signals.append(0)
                continue
            
            # 1. 动量信号（更保守）
            momentum_score = 0
            if current_row['macd_histogram'] > 0 and current_row['macd'] > current_row['macd_signal']:
                momentum_score += 0.4
            if current_row['close'] > current_row['sma_20']:
                momentum_score += 0.3
            
            # 2. 均值回归信号（更严格）
            mean_reversion_score = 0
            if current_row['rsi'] < 25:  # 更严格的超卖条件
                mean_reversion_score += 0.5
            elif current_row['rsi'] > 75:  # 更严格的超买条件
                mean_reversion_score -= 0.5
            
            if current_row['bb_position'] < 0.1:  # 更接近下轨
                mean_reversion_score += 0.4
            elif current_row['bb_position'] > 0.9:  # 更接近上轨
                mean_reversion_score -= 0.4
            
            # 3. 波动率信号
            volatility_score = 0
            if idx >= 40:
                vol_percentile = np.percentile(data['volatility'].iloc[idx-40:idx], 30)
                if current_row['volatility'] > vol_percentile:
                    volatility_score += 0.3
            
            # 4. 成交量信号（更严格）
            volume_score = 0
            if current_row['volume_ratio'] > 2.0:  # 更高的成交量要求
                volume_score += 0.4
            elif current_row['volume_ratio'] > 1.5:
                volume_score += 0.2
            
            # 5. 微观结构信号
            microstructure_score = 0
            if current_row['spread'] < 0.002:  # 更低的价差要求
                microstructure_score += 0.3
            
            # 综合信号计算
            weights = self.config['signal_weights']
            total_signal = (
                momentum_score * weights['momentum'] +
                mean_reversion_score * weights['mean_reversion'] +
                volatility_score * weights['volatility'] +
                volume_score * weights['volume'] +
                microstructure_score * weights['microstructure']
            )
            
            # 信号强度过滤
            min_strength = self.config['signal_parameters']['min_signal_strength']
            if abs(total_signal) < min_strength:
                total_signal = 0
            
            signals.append(total_signal)
        
        # 添加前面的空值
        data['signal'] = [0] * lookback + signals
        return data
    
    def can_trade(self, timestamp: pd.Timestamp) -> bool:
        """检查是否可以交易"""
        # 检查每日交易次数限制
        if self.daily_trades >= self.config['risk_management']['max_daily_trades']:
            return False
        
        # 检查冷却期
        if self.last_trade_time is not None:
            cooldown = timedelta(minutes=self.config['risk_management']['cooldown_period'])
            if timestamp - self.last_trade_time < cooldown:
                return False
        
        return True
    
    def execute_trades(self, data: pd.DataFrame) -> List[Dict]:
        """执行交易（修复交易逻辑）"""
        trades = []
        signal_threshold = self.config['signal_parameters']['signal_threshold']
        max_trade_size = self.config['signal_parameters']['max_trade_size']
        
        current_position = 0  # 当前持仓
        entry_price = 0
        entry_time = None
        
        for idx, row in data.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            
            # 检查是否可以交易
            if not self.can_trade(timestamp):
                continue
            
            # 检查信号强度
            if abs(row['signal']) <= signal_threshold:
                continue
            
            price = row['close']
            slippage = self.config['execution_settings']['slippage']
            commission = self.config['execution_settings']['commission']
            
            # 如果没有持仓，考虑开仓
            if current_position == 0:
                if row['signal'] > signal_threshold:
                    # 买入开仓
                    size = min(max_trade_size, self.cash * 0.1)  # 限制单笔交易规模
                    execution_price = price * (1 + slippage)
                    cost = size * execution_price
                    commission_cost = cost * commission
                    
                    if self.cash >= cost + commission_cost:
                        current_position = size / execution_price  # 股数
                        entry_price = execution_price
                        entry_time = timestamp
                        self.cash -= (cost + commission_cost)
                        
                        trade = {
                            'timestamp': timestamp,
                            'side': 'buy',
                            'size': current_position,
                            'price': execution_price,
                            'signal_strength': row['signal'],
                            'commission': commission_cost,
                            'type': 'open'
                        }
                        trades.append(trade)
                        self.daily_trades += 1
                        self.last_trade_time = timestamp
                
                elif row['signal'] < -signal_threshold:
                    # 卖空开仓（简化处理，假设可以卖空）
                    size = min(max_trade_size, self.cash * 0.1)
                    execution_price = price * (1 - slippage)
                    proceeds = size * execution_price
                    commission_cost = proceeds * commission
                    
                    current_position = -(size / execution_price)  # 负股数表示空头
                    entry_price = execution_price
                    entry_time = timestamp
                    self.cash += (proceeds - commission_cost)
                    
                    trade = {
                        'timestamp': timestamp,
                        'side': 'sell',
                        'size': abs(current_position),
                        'price': execution_price,
                        'signal_strength': row['signal'],
                        'commission': commission_cost,
                        'type': 'open'
                    }
                    trades.append(trade)
                    self.daily_trades += 1
                    self.last_trade_time = timestamp
            
            # 如果有持仓，考虑平仓
            elif current_position != 0:
                should_close = False
                close_reason = ""
                
                # 止盈止损检查
                if current_position > 0:  # 多头持仓
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct >= self.config['risk_management']['take_profit']:
                        should_close = True
                        close_reason = "take_profit"
                    elif pnl_pct <= -self.config['risk_management']['stop_loss']:
                        should_close = True
                        close_reason = "stop_loss"
                    elif row['signal'] < -signal_threshold:  # 反向信号
                        should_close = True
                        close_reason = "signal_reversal"
                
                elif current_position < 0:  # 空头持仓
                    pnl_pct = (entry_price - price) / entry_price
                    if pnl_pct >= self.config['risk_management']['take_profit']:
                        should_close = True
                        close_reason = "take_profit"
                    elif pnl_pct <= -self.config['risk_management']['stop_loss']:
                        should_close = True
                        close_reason = "stop_loss"
                    elif row['signal'] > signal_threshold:  # 反向信号
                        should_close = True
                        close_reason = "signal_reversal"
                
                # 执行平仓
                if should_close:
                    if current_position > 0:
                        # 卖出平仓
                        execution_price = price * (1 - slippage)
                        proceeds = current_position * execution_price
                        commission_cost = proceeds * commission
                        self.cash += (proceeds - commission_cost)
                        
                        trade = {
                            'timestamp': timestamp,
                            'side': 'sell',
                            'size': current_position,
                            'price': execution_price,
                            'signal_strength': row['signal'],
                            'commission': commission_cost,
                            'type': 'close',
                            'reason': close_reason,
                            'pnl': proceeds - (current_position * entry_price)
                        }
                    else:
                        # 买入平仓
                        execution_price = price * (1 + slippage)
                        cost = abs(current_position) * execution_price
                        commission_cost = cost * commission
                        self.cash -= (cost + commission_cost)
                        
                        trade = {
                            'timestamp': timestamp,
                            'side': 'buy',
                            'size': abs(current_position),
                            'price': execution_price,
                            'signal_strength': row['signal'],
                            'commission': commission_cost,
                            'type': 'close',
                            'reason': close_reason,
                            'pnl': (abs(current_position) * entry_price) - cost
                        }
                    
                    trades.append(trade)
                    current_position = 0
                    entry_price = 0
                    entry_time = None
                    self.daily_trades += 1
                    self.last_trade_time = timestamp
        
        # 更新最终组合价值
        if current_position != 0:
            # 如果还有持仓，按最后价格计算
            final_price = data.iloc[-1]['close']
            if current_position > 0:
                self.portfolio_value = self.cash + (current_position * final_price)
            else:
                self.portfolio_value = self.cash - (abs(current_position) * final_price)
        else:
            self.portfolio_value = self.cash
        
        return trades
    
    def calculate_performance_metrics(self, trades: List[Dict], data: pd.DataFrame) -> Dict:
        """计算性能指标"""
        if not trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'win_rate': 0,
                'final_portfolio_value': self.portfolio_value,
                'average_trade_return': 0
            }
        
        # 计算交易收益
        trade_pnls = []
        for trade in trades:
            if trade.get('type') == 'close' and 'pnl' in trade:
                trade_pnls.append(trade['pnl'])
        
        if not trade_pnls:
            total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            return {
                'total_return': total_return,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': len(trades),
                'win_rate': 0,
                'final_portfolio_value': self.portfolio_value,
                'average_trade_return': 0
            }
        
        # 计算指标
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        if len(trade_pnls) > 1:
            sharpe_ratio = np.mean(trade_pnls) / np.std(trade_pnls) if np.std(trade_pnls) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 胜率
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0
        
        # 最大回撤（简化计算）
        portfolio_values = [self.initial_capital]
        running_pnl = 0
        for pnl in trade_pnls:
            running_pnl += pnl
            portfolio_values.append(self.initial_capital + running_pnl)
        
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'final_portfolio_value': self.portfolio_value,
            'average_trade_return': np.mean(trade_pnls) if trade_pnls else 0
        }
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """运行回测"""
        self.logger.info("🚀 开始运行稳健优化策略回测...")
        
        # 计算技术指标
        data = self.calculate_technical_indicators(data)
        
        # 生成交易信号
        data = self.generate_signals(data)
        
        # 执行交易
        trades = self.execute_trades(data)
        
        # 计算性能指标
        performance = self.calculate_performance_metrics(trades, data)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'strategy': 'RobustCitadelHFT',
            'config': self.config,
            'performance_metrics': performance,
            'trades_summary': {
                'total_trades': len(trades),
                'final_portfolio_value': self.portfolio_value,
                'final_cash': self.cash
            }
        }
        
        # 保存到文件
        results_path = f"competitions/citadel/citadel_robust_backtest_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存交易记录
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_path = f"competitions/citadel/citadel_robust_trades_{timestamp}.csv"
            trades_df.to_csv(trades_path, index=False)
            self.logger.info(f"交易记录已保存到: {trades_path}")
        
        self.logger.info(f"回测结果已保存到: {results_path}")
        
        return results

def main():
    """主函数"""
    print("🚀 Citadel 稳健优化高频交易策略")
    print("=" * 60)
    
    try:
        # 初始化策略
        strategy = RobustCitadelHFTStrategy()
        
        # 加载数据
        data = strategy.load_market_data()
        print(f"数据加载完成，共 {len(data)} 条记录")
        print(f"数据时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
        
        # 运行回测
        results = strategy.run_backtest(data)
        
        # 显示结果
        metrics = results['performance_metrics']
        print(f"\n📊 回测结果:")
        print("-" * 40)
        print(f"总收益率: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
        print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"胜率: {metrics['win_rate']:.4f} ({metrics['win_rate']*100:.2f}%)")
        print(f"最终组合价值: ${metrics['final_portfolio_value']:,.2f}")
        print(f"平均交易收益: {metrics['average_trade_return']:.2f}")
        
        print(f"\n🎉 稳健优化策略回测完成!")
        
    except Exception as e:
        print(f"❌ 策略运行失败: {e}")
        raise

if __name__ == "__main__":
    main()