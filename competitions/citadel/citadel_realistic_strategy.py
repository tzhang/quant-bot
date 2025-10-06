#!/usr/bin/env python3
"""
Citadel 现实版策略 - 修复高频交易问题
解决超高收益率和异常交易频率问题
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CitadelRealisticStrategy:
    def __init__(self, config=None):
        """初始化策略"""
        self.config = config or {}
        
        # 基本参数 - 更保守的设置
        self.initial_capital = self.config.get('initial_capital', 1000000)
        self.lookback_period = self.config.get('lookback_period', 20)  # 增加到20天
        self.signal_threshold = self.config.get('signal_threshold', 0.05)  # 降低阈值到0.05
        self.position_limit = self.config.get('position_limit', 0.3)  # 降低到30%
        self.max_trade_size = self.config.get('max_trade_size', 0.1)  # 降低到10%
        
        # 风险管理参数 - 更严格的设置
        self.stop_loss = self.config.get('stop_loss', 0.02)  # 2%止损
        self.take_profit = self.config.get('take_profit', 0.03)  # 3%止盈
        self.max_daily_trades = self.config.get('max_daily_trades', 10)  # 增加每日最大交易次数
        self.min_trade_interval = self.config.get('min_trade_interval', 60)  # 减少最小交易间隔到1分钟
        
        # 信号权重 - 平衡设置
        self.momentum_weight = self.config.get('momentum_weight', 0.25)
        self.mean_reversion_weight = self.config.get('mean_reversion_weight', 0.25)
        self.volatility_weight = self.config.get('volatility_weight', 0.25)
        self.microstructure_weight = self.config.get('microstructure_weight', 0.25)
        
        # 状态变量
        self.portfolio_value = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_trades = {}
        self.last_trade_time = {}
        
        print(f"🚀 Citadel现实版策略初始化完成")
        print(f"   初始资金: ${self.initial_capital:,.2f}")
        print(f"   信号阈值: {self.signal_threshold}")
        print(f"   最大仓位: {self.position_limit*100}%")
        print(f"   止损/止盈: {self.stop_loss*100}%/{self.take_profit*100}%")
        print(f"   每日最大交易: {self.max_daily_trades}笔")
    
    def load_data(self, data_path):
        """加载数据"""
        print(f"📊 加载数据: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 读取数据，跳过注释行
        self.data = pd.read_csv(data_path, comment='#')
        
        # 数据预处理
        # 重命名列以匹配预期格式
        column_mapping = {
            'Price': 'timestamp',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        self.data = self.data.rename(columns=column_mapping)
        
        # 处理时间戳
        if 'timestamp' in self.data.columns:
            # 如果timestamp是日期格式，转换为datetime
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        print(f"   数据行数: {len(self.data)}")
        print(f"   时间范围: {self.data['timestamp'].min()} 到 {self.data['timestamp'].max()}")
        
        return self.data
    
    def calculate_technical_indicators(self):
        """计算技术指标"""
        print("🔧 计算技术指标...")
        
        # 移动平均线
        self.data['sma_5'] = self.data['close'].rolling(5).mean()
        self.data['sma_10'] = self.data['close'].rolling(10).mean()
        self.data['sma_20'] = self.data['close'].rolling(20).mean()
        self.data['ema_12'] = self.data['close'].ewm(span=12).mean()
        self.data['ema_26'] = self.data['close'].ewm(span=26).mean()
        
        # MACD
        self.data['macd'] = self.data['ema_12'] - self.data['ema_26']
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
        
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        self.data['bb_middle'] = self.data['close'].rolling(20).mean()
        bb_std = self.data['close'].rolling(20).std()
        self.data['bb_upper'] = self.data['bb_middle'] + (bb_std * 2)
        self.data['bb_lower'] = self.data['bb_middle'] - (bb_std * 2)
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
        
        # 波动率
        self.data['volatility'] = self.data['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # 成交量指标
        self.data['volume_sma'] = self.data['volume'].rolling(20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_sma']
        
        # 价格变化
        self.data['returns'] = self.data['close'].pct_change()
        self.data['returns_5d'] = self.data['close'].pct_change(5)
        
        print("   技术指标计算完成")
    
    def generate_signals(self, index):
        """生成交易信号"""
        if index < self.lookback_period:
            return 0
        
        current_data = self.data.iloc[index]
        
        # 动量信号
        momentum_signal = 0
        if not pd.isna(current_data['macd']) and not pd.isna(current_data['macd_signal']):
            if current_data['macd'] > current_data['macd_signal']:
                momentum_signal += 0.3
            else:
                momentum_signal -= 0.3
        
        if not pd.isna(current_data['rsi']):
            if current_data['rsi'] > 70:
                momentum_signal -= 0.2
            elif current_data['rsi'] < 30:
                momentum_signal += 0.2
        
        # 均值回归信号
        mean_reversion_signal = 0
        if not pd.isna(current_data['bb_upper']) and not pd.isna(current_data['bb_lower']):
            bb_position = (current_data['close'] - current_data['bb_lower']) / (current_data['bb_upper'] - current_data['bb_lower'])
            if bb_position > 0.8:
                mean_reversion_signal -= 0.4  # 接近上轨，看跌
            elif bb_position < 0.2:
                mean_reversion_signal += 0.4  # 接近下轨，看涨
        
        # 波动率信号
        volatility_signal = 0
        if not pd.isna(current_data['volatility']):
            vol_percentile = np.percentile(self.data['volatility'].dropna(), 50)
            if current_data['volatility'] > vol_percentile * 1.5:
                volatility_signal -= 0.2  # 高波动率，谨慎
            elif current_data['volatility'] < vol_percentile * 0.5:
                volatility_signal += 0.1  # 低波动率，机会
        
        # 微观结构信号
        microstructure_signal = 0
        if not pd.isna(current_data['volume_ratio']):
            if current_data['volume_ratio'] > 1.5:
                if current_data['returns'] > 0:
                    microstructure_signal += 0.2  # 放量上涨
                else:
                    microstructure_signal -= 0.2  # 放量下跌
        
        # 综合信号
        total_signal = (momentum_signal * self.momentum_weight +
                       mean_reversion_signal * self.mean_reversion_weight +
                       volatility_signal * self.volatility_weight +
                       microstructure_signal * self.microstructure_weight)
        
        return total_signal
    
    def can_trade(self, timestamp, symbol='AAPL'):
        """检查是否可以交易"""
        current_date = timestamp.date()
        
        # 检查每日交易次数限制
        if current_date not in self.daily_trades:
            self.daily_trades[current_date] = 0
        
        if self.daily_trades[current_date] >= self.max_daily_trades:
            return False, "达到每日交易次数限制"
        
        # 检查最小交易间隔
        if symbol in self.last_trade_time:
            time_diff = (timestamp - self.last_trade_time[symbol]).total_seconds()
            if time_diff < self.min_trade_interval:
                return False, f"交易间隔不足，需等待{self.min_trade_interval - time_diff:.0f}秒"
        
        return True, "可以交易"
    
    def execute_trade(self, signal, price, timestamp, symbol='AAPL'):
        """执行交易"""
        can_trade, reason = self.can_trade(timestamp, symbol)
        if not can_trade:
            return
        
        current_position = self.positions.get(symbol, 0)
        
        # 买入信号
        if signal > self.signal_threshold and current_position <= 0:
            # 计算买入数量
            available_cash = self.portfolio_value * self.max_trade_size
            shares_to_buy = int(available_cash / price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                self.portfolio_value -= cost
                
                # 记录交易
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'cost': cost,
                    'portfolio_value': self.portfolio_value + self.positions[symbol] * price
                }
                self.trades.append(trade)
                
                # 更新交易计数和时间
                current_date = timestamp.date()
                self.daily_trades[current_date] = self.daily_trades.get(current_date, 0) + 1
                self.last_trade_time[symbol] = timestamp
        
        # 卖出信号
        elif signal < -self.signal_threshold and current_position > 0:
            shares_to_sell = min(current_position, int(current_position * self.max_trade_size))
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * price
                self.positions[symbol] -= shares_to_sell
                self.portfolio_value += revenue
                
                # 记录交易
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'revenue': revenue,
                    'portfolio_value': self.portfolio_value + self.positions[symbol] * price
                }
                self.trades.append(trade)
                
                # 更新交易计数和时间
                current_date = timestamp.date()
                self.daily_trades[current_date] = self.daily_trades.get(current_date, 0) + 1
                self.last_trade_time[symbol] = timestamp
        
        # 止损止盈检查
        if current_position > 0:
            self.check_stop_loss_take_profit(price, timestamp, symbol)
    
    def check_stop_loss_take_profit(self, current_price, timestamp, symbol='AAPL'):
        """检查止损止盈"""
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return
        
        # 找到最近的买入价格
        recent_buy_price = None
        for trade in reversed(self.trades):
            if trade['symbol'] == symbol and trade['action'] == 'BUY':
                recent_buy_price = trade['price']
                break
        
        if recent_buy_price is None:
            return
        
        current_position = self.positions[symbol]
        price_change = (current_price - recent_buy_price) / recent_buy_price
        
        # 止损
        if price_change <= -self.stop_loss:
            revenue = current_position * current_price
            self.positions[symbol] = 0
            self.portfolio_value += revenue
            
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'SELL_STOP_LOSS',
                'shares': current_position,
                'price': current_price,
                'revenue': revenue,
                'portfolio_value': self.portfolio_value
            }
            self.trades.append(trade)
            
            # 更新交易计数
            current_date = timestamp.date()
            self.daily_trades[current_date] = self.daily_trades.get(current_date, 0) + 1
        
        # 止盈
        elif price_change >= self.take_profit:
            revenue = current_position * current_price
            self.positions[symbol] = 0
            self.portfolio_value += revenue
            
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'SELL_TAKE_PROFIT',
                'shares': current_position,
                'price': current_price,
                'revenue': revenue,
                'portfolio_value': self.portfolio_value
            }
            self.trades.append(trade)
            
            # 更新交易计数
            current_date = timestamp.date()
            self.daily_trades[current_date] = self.daily_trades.get(current_date, 0) + 1
    
    def run_backtest(self):
        """运行回测"""
        print("🔄 开始回测...")
        
        for i in range(len(self.data)):
            current_data = self.data.iloc[i]
            timestamp = current_data['timestamp']
            price = current_data['close']
            
            # 生成信号
            signal = self.generate_signals(i)
            
            # 执行交易
            if abs(signal) > self.signal_threshold:
                self.execute_trade(signal, price, timestamp)
            
            # 检查现有持仓的止损止盈
            for symbol in list(self.positions.keys()):
                if self.positions[symbol] > 0:
                    self.check_stop_loss_take_profit(price, timestamp, symbol)
        
        print(f"   回测完成，总交易次数: {len(self.trades)}")
    
    def calculate_performance_metrics(self):
        """计算性能指标"""
        print("📊 计算性能指标...")
        
        if not self.trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'win_rate': 0,
                'final_portfolio_value': self.initial_capital,
                'avg_trade_return': 0
            }
        
        # 计算组合价值序列
        portfolio_values = []
        for trade in self.trades:
            portfolio_values.append(trade['portfolio_value'])
        
        # 总收益率
        final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 计算收益率序列
        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
        
        # 夏普比率
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        peak = self.initial_capital
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 胜率
        profitable_trades = 0
        for i, trade in enumerate(self.trades):
            if i > 0 and trade['portfolio_value'] > self.trades[i-1]['portfolio_value']:
                profitable_trades += 1
        
        win_rate = profitable_trades / len(self.trades) if self.trades else 0
        
        # 平均交易收益
        avg_trade_return = total_return / len(self.trades) if self.trades else 0
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'final_portfolio_value': final_value,
            'avg_trade_return': avg_trade_return
        }
        
        print(f"   总收益率: {total_return*100:.2f}%")
        print(f"   夏普比率: {sharpe_ratio:.4f}")
        print(f"   最大回撤: {max_drawdown*100:.2f}%")
        print(f"   胜率: {win_rate*100:.2f}%")
        
        return metrics
    
    def save_results(self, output_dir='competitions/citadel'):
        """保存结果"""
        print("💾 保存结果...")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存交易记录
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = os.path.join(output_dir, f'citadel_realistic_trades_{timestamp}.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"   交易记录保存到: {trades_file}")
        
        # 保存回测结果
        metrics = self.calculate_performance_metrics()
        results = {
            'strategy_name': 'citadel_realistic_hft',
            'timestamp': timestamp,
            'config': self.config,
            'performance_metrics': metrics,
            'summary': {
                'total_return_pct': metrics['total_return'] * 100,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown_pct': metrics['max_drawdown'] * 100,
                'total_trades': metrics['total_trades'],
                'win_rate_pct': metrics['win_rate'] * 100,
                'final_portfolio_value': metrics['final_portfolio_value'],
                'avg_trade_return_pct': metrics['avg_trade_return'] * 100
            }
        }
        
        results_file = os.path.join(output_dir, f'citadel_realistic_backtest_results_{timestamp}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   回测结果保存到: {results_file}")
        return results_file, trades_file if self.trades else None

def main():
    """主函数"""
    print("🎯 Citadel现实版策略回测")
    print("=" * 60)
    
    # 配置参数
    config = {
        'initial_capital': 1000000,
        'lookback_period': 20,
        'signal_threshold': 0.05,
        'position_limit': 0.3,
        'max_trade_size': 0.1,
        'stop_loss': 0.02,
        'take_profit': 0.03,
        'max_daily_trades': 10,
        'min_trade_interval': 60,  # 1分钟
        'momentum_weight': 0.25,
        'mean_reversion_weight': 0.25,
        'volatility_weight': 0.25,
        'microstructure_weight': 0.25
    }
    
    try:
        # 初始化策略
        strategy = CitadelRealisticStrategy(config)
        
        # 加载数据
        data_path = 'examples/data_cache/ohlcv_AAPL_20251005_200622.csv'
        strategy.load_data(data_path)
        
        # 计算技术指标
        strategy.calculate_technical_indicators()
        
        # 运行回测
        strategy.run_backtest()
        
        # 保存结果
        results_file, trades_file = strategy.save_results()
        
        print("\n✅ 回测完成!")
        print(f"📊 结果文件: {results_file}")
        if trades_file:
            print(f"📈 交易记录: {trades_file}")
        
    except Exception as e:
        print(f"❌ 回测失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()