#!/usr/bin/env python3
"""
🏛️ Citadel 改进版高频交易策略
修复信号生成逻辑，确保产生平衡的买卖信号
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger
from src.ml.terminal_ai_tools import run_terminal_ai_simulation
from src.ml.feature_engineering import AdvancedFeatureEngineer
from src.risk.risk_manager import RiskManager

class ImprovedCitadelHFTStrategy:
    """改进版Citadel高频交易策略"""
    
    def __init__(self, config_file=None):
        self.logger = get_logger(__name__)
        self.config = self.load_config(config_file)
        
        # 策略参数 - 降低阈值以产生更多交易
        self.lookback_period = self.config.get('lookback_period', 8)
        self.signal_threshold = self.config.get('signal_threshold', 0.15)  # 降低阈值
        self.position_limit = self.config.get('position_limit', 0.12)
        self.max_trade_size = self.config.get('max_trade_size', 6000)
        self.stop_loss = self.config.get('stop_loss', 0.008)
        self.take_profit = self.config.get('take_profit', 0.012)
        
        # 交易状态
        self.positions = {}
        self.cash = 1000000
        self.portfolio_value = 1000000
        self.trades = []
        self.entry_prices = {}  # 记录入场价格
        
        # 信号权重 - 重新平衡以产生更多卖出信号
        self.signal_weights = {
            'momentum': 0.25,
            'mean_reversion': 0.35,  # 增加均值回归权重
            'volatility': 0.2,
            'microstructure': 0.2
        }
        
        # 风险管理
        self.risk_manager = RiskManager()
        
    def load_config(self, config_file):
        """加载配置"""
        default_config = {
            "strategy_name": "improved_citadel_hft",
            "version": "1.0",
            "description": "改进版Citadel高频交易策略，修复信号生成逻辑",
            "signal_parameters": {
                "lookback_period": 10,
                "signal_threshold": 0.2,
                "position_limit": 0.1,
                "max_trade_size": 5000,
                "min_signal_strength": 0.25,
                "signal_decay": 0.95
            },
            "risk_management": {
                "stop_loss": 0.01,
                "take_profit": 0.015,
                "max_portfolio_risk": 0.02,
                "max_single_position": 0.1,
                "max_correlation": 0.7
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        # 移动平均线
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['ema_5'] = data['close'].ewm(span=5).mean()
        data['ema_10'] = data['close'].ewm(span=10).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # 布林带
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # 价格位置
        data['price_position'] = (data['close'] - data['close'].rolling(window=20).min()) / \
                                (data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min())
        
        # 波动率
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        data['volatility_ratio'] = data['volatility'] / data['volatility'].rolling(window=50).mean()
        
        # 价差（如果有bid_price和ask_price）
        if 'bid_price' in data.columns and 'ask_price' in data.columns:
            data['spread'] = data['ask_price'] - data['bid_price']
            data['spread_pct'] = data['spread'] / data['close']
        else:
            data['spread'] = 0.001  # 默认价差
            data['spread_pct'] = 0.001
        
        return data
    
    def generate_signals(self, data, idx):
        """生成交易信号 - 改进版本，确保产生平衡的买卖信号"""
        if idx < self.lookback_period:
            return 0
        
        signals = {}
        current_row = data.iloc[idx]
        
        # 动量信号 - 更平衡的逻辑
        momentum_score = 0
        
        # 价格相对于移动平均线
        if current_row['close'] > current_row['sma_10']:
            momentum_score += 0.2
        else:
            momentum_score -= 0.2
            
        # EMA交叉
        if current_row['ema_5'] > current_row['ema_10']:
            momentum_score += 0.2
        else:
            momentum_score -= 0.2
            
        # MACD
        if current_row['macd'] > current_row['macd_signal']:
            momentum_score += 0.2
        else:
            momentum_score -= 0.2
            
        signals['momentum'] = momentum_score
        
        # 均值回归信号 - 更强的反转逻辑
        mean_reversion_score = 0
        
        # RSI
        rsi = current_row['rsi']
        if not pd.isna(rsi):
            if rsi < 30:  # 超卖，买入信号
                mean_reversion_score += 0.4
            elif rsi > 70:  # 超买，卖出信号
                mean_reversion_score -= 0.4
            elif rsi < 40:
                mean_reversion_score += 0.2
            elif rsi > 60:
                mean_reversion_score -= 0.2
        
        # 布林带位置
        bb_pos = current_row['bb_position']
        if not pd.isna(bb_pos):
            if bb_pos < 0.2:  # 接近下轨，买入
                mean_reversion_score += 0.3
            elif bb_pos > 0.8:  # 接近上轨，卖出
                mean_reversion_score -= 0.3
            elif bb_pos < 0.3:
                mean_reversion_score += 0.1
            elif bb_pos > 0.7:
                mean_reversion_score -= 0.1
        
        # 价格位置
        price_pos = current_row['price_position']
        if not pd.isna(price_pos):
            if price_pos < 0.2:
                mean_reversion_score += 0.2
            elif price_pos > 0.8:
                mean_reversion_score -= 0.2
        
        signals['mean_reversion'] = mean_reversion_score
        
        # 波动率信号
        vol_ratio = current_row['volatility_ratio']
        if not pd.isna(vol_ratio):
            if vol_ratio > 1.5:  # 高波动，谨慎
                volatility_score = -0.1
            elif vol_ratio < 0.7:  # 低波动，积极
                volatility_score = 0.1
            else:
                volatility_score = 0
        else:
            volatility_score = 0
        signals['volatility'] = volatility_score
        
        # 微观结构信号
        microstructure_score = 0
        if 'vwap' in current_row.index and not pd.isna(current_row['vwap']):
            close = current_row['close']
            vwap = current_row['vwap']
            if close > vwap * 1.002:
                microstructure_score = 0.15
            elif close < vwap * 0.998:
                microstructure_score = -0.15
        else:
            # 使用价格相对于SMA的位置
            if current_row['close'] > current_row['sma_10'] * 1.001:
                microstructure_score = 0.1
            elif current_row['close'] < current_row['sma_10'] * 0.999:
                microstructure_score = -0.1
        
        signals['microstructure'] = microstructure_score
        
        # 加权组合信号
        final_signal = sum(signals[key] * self.signal_weights[key] for key in signals)
        
        return final_signal
    
    def execute_trade(self, symbol, signal, price, timestamp, volume):
        """执行交易"""
        if abs(signal) < self.signal_threshold:
            return
        
        # 计算交易规模
        current_position = self.positions.get(symbol, 0)
        max_position_value = self.portfolio_value * self.position_limit
        max_shares = int(max_position_value / price)
        
        # 根据信号强度调整交易规模
        signal_strength = min(abs(signal), 1.0)
        trade_size = int(self.max_trade_size * signal_strength)
        trade_size = min(trade_size, max_shares)
        
        if signal > 0:  # 买入信号
            if current_position < max_shares:
                shares_to_buy = min(trade_size, max_shares - current_position)
                cost = shares_to_buy * price * (1 + 0.001)  # 包含手续费
                
                if self.cash >= cost:
                    self.positions[symbol] = current_position + shares_to_buy
                    self.cash -= cost
                    self.entry_prices[symbol] = price  # 记录入场价格
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': price,
                        'value': cost,
                        'signal': signal,
                        'portfolio_value': self.calculate_portfolio_value(price)
                    })
        
        elif signal < 0:  # 卖出信号
            if current_position > 0:
                shares_to_sell = min(trade_size, current_position)
                proceeds = shares_to_sell * price * (1 - 0.001)  # 扣除手续费
                
                self.positions[symbol] = current_position - shares_to_sell
                self.cash += proceeds
                
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'value': proceeds,
                    'signal': signal,
                    'portfolio_value': self.calculate_portfolio_value(price)
                })
    
    def check_stop_loss_take_profit(self, symbol, current_price):
        """检查止损止盈"""
        if symbol not in self.positions or self.positions[symbol] == 0:
            return
        
        if symbol not in self.entry_prices:
            return
        
        entry_price = self.entry_prices[symbol]
        current_position = self.positions[symbol]
        
        # 计算收益率
        if current_position > 0:  # 多头仓位
            return_pct = (current_price - entry_price) / entry_price
            
            # 止损
            if return_pct <= -self.stop_loss:
                self.close_position(symbol, current_price, 'STOP_LOSS')
            # 止盈
            elif return_pct >= self.take_profit:
                self.close_position(symbol, current_price, 'TAKE_PROFIT')
    
    def close_position(self, symbol, price, reason):
        """平仓"""
        if symbol not in self.positions or self.positions[symbol] == 0:
            return
        
        shares = self.positions[symbol]
        proceeds = shares * price * (1 - 0.001)  # 扣除手续费
        
        self.positions[symbol] = 0
        self.cash += proceeds
        
        self.trades.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': f'SELL_{reason}',
            'shares': shares,
            'price': price,
            'value': proceeds,
            'signal': 0,
            'portfolio_value': self.calculate_portfolio_value(price)
        })
    
    def calculate_portfolio_value(self, current_price):
        """计算组合价值"""
        total_value = self.cash
        for symbol, shares in self.positions.items():
            total_value += shares * current_price
        return total_value
    
    def run_backtest(self, data):
        """运行回测"""
        results = []
        
        for idx in range(len(data)):
            row = data.iloc[idx]
            timestamp = row['timestamp']
            price = row['close']
            volume = row['volume']
            
            # 检查止损止盈
            self.check_stop_loss_take_profit('SYMBOL', price)
            
            # 生成信号
            signal = self.generate_signals(data, idx)
            
            # 执行交易
            self.execute_trade('SYMBOL', signal, price, timestamp, volume)
            
            # 更新组合价值
            self.portfolio_value = self.calculate_portfolio_value(price)
            
            # 记录结果
            results.append({
                'timestamp': timestamp,
                'price': price,
                'signal': signal,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'position': self.positions.get('SYMBOL', 0)
            })
        
        return results
    
    def calculate_performance_metrics(self, results):
        """计算性能指标"""
        portfolio_values = [r['portfolio_value'] for r in results]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)  # 年化
        else:
            sharpe_ratio = 0
        
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        # 交易统计
        total_trades = len(self.trades)
        if total_trades > 0:
            winning_trades = sum(1 for trade in self.trades if 'SELL' in trade['action'])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            trade_returns = []
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if 'SELL' in t['action']]
            
            for sell_trade in sell_trades:
                # 简化的收益计算
                if buy_trades:
                    buy_price = buy_trades[0]['price']  # 简化处理
                    sell_price = sell_trade['price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_returns.append(trade_return)
            
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        else:
            win_rate = 0
            avg_trade_return = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values[-1],
            'avg_trade_return': avg_trade_return
        }
    
    def calculate_max_drawdown(self, portfolio_values):
        """计算最大回撤"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def save_results(self, results, metrics):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存交易记录
        trades_df = pd.DataFrame(self.trades)
        trades_file = f"competitions/citadel/citadel_improved_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        self.logger.info(f"交易记录已保存到: {trades_file}")
        
        # 保存回测结果
        results_data = {
            'config': self.config,
            'summary': metrics,
            'results': results
        }
        
        results_file = f"competitions/citadel/citadel_improved_backtest_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        return results_file

def load_market_data():
    """加载市场数据"""
    data_file = "competitions/citadel/data/sample_market_data.csv"
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("生成模拟数据...")
        
        # 生成模拟数据
        np.random.seed(42)
        n_points = 50000
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=4),
            end=datetime.now(),
            periods=n_points
        )
        
        # 生成价格数据
        base_price = 100
        returns = np.random.normal(0, 0.001, n_points)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 生成其他数据
        volumes = np.random.lognormal(10, 1, n_points)
        spreads = np.random.uniform(0.001, 0.005, n_points)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
            'close': prices,
            'volume': volumes,
            'vwap': [p * (1 + np.random.normal(0, 0.0005)) for p in prices],
            'bid_price': [p - s/2 for p, s in zip(prices, spreads)],
            'ask_price': [p + s/2 for p, s in zip(prices, spreads)],
            'asset_id': ['SYMBOL'] * n_points,
            'bid_size': np.random.uniform(100, 1000, n_points),
            'ask_size': np.random.uniform(100, 1000, n_points)
        })
        
        # 确保目录存在
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        data.to_csv(data_file, index=False)
        print(f"模拟数据已保存到: {data_file}")
    
    data = pd.read_csv(data_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    return data

def main():
    """主函数"""
    print("🏛️ Citadel 改进版高频交易策略")
    print("=" * 60)
    
    # 加载数据
    data = load_market_data()
    print(f"数据加载完成，共 {len(data)} 条记录")
    print(f"数据时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    
    # 初始化策略
    strategy = ImprovedCitadelHFTStrategy()
    
    # 计算技术指标
    data = strategy.calculate_technical_indicators(data)
    
    # 运行回测
    strategy.logger.info("开始运行改进版策略回测...")
    results = strategy.run_backtest(data)
    
    # 计算性能指标
    metrics = strategy.calculate_performance_metrics(results)
    
    # 保存结果
    results_file = strategy.save_results(results, metrics)
    
    # 打印结果
    print("\n📊 回测结果:")
    print("-" * 40)
    print(f"总收益率: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"胜率: {metrics['win_rate']:.4f} ({metrics['win_rate']*100:.2f}%)")
    print(f"最终组合价值: ${metrics['final_portfolio_value']:,.2f}")
    print(f"平均交易收益: {metrics['avg_trade_return']*100:.2f}%")
    
    print(f"\n📁 详细结果已保存到: {results_file}")
    print("\n🎉 改进版策略回测完成!")

if __name__ == "__main__":
    main()