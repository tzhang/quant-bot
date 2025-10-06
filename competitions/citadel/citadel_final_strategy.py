#!/usr/bin/env python3
"""
Citadel最终优化版策略
基于前面的分析和测试结果，创建最终优化版本
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CitadelFinalStrategy:
    def __init__(self, config=None):
        """初始化策略"""
        self.config = config or {}
        
        # 基本参数
        self.initial_capital = self.config.get('initial_capital', 1000000)
        self.lookback_period = self.config.get('lookback_period', 20)  # 减少到20天
        
        # 信号阈值
        self.signal_threshold = self.config.get('signal_threshold', 0.03)  # 进一步降低阈值
        
        # 仓位管理
        self.position_limit = self.config.get('position_limit', 0.30)  # 提高到30%
        self.max_trade_size = self.config.get('max_trade_size', 0.10)  # 提高到10%
        
        # 风险管理
        self.stop_loss = self.config.get('stop_loss', 0.02)  # 2%止损
        self.take_profit = self.config.get('take_profit', 0.06)  # 6%止盈，3:1风险收益比
        self.trailing_stop = self.config.get('trailing_stop', 0.015)  # 1.5%追踪止损
        
        # 交易频率控制
        self.max_daily_trades = self.config.get('max_daily_trades', 2)  # 每日最多2笔
        self.min_trade_interval = self.config.get('min_trade_interval', 0)  # 适应日线数据
        
        # 信号权重 - 重新平衡
        self.momentum_weight = self.config.get('momentum_weight', 0.40)  # 增加动量权重
        self.mean_reversion_weight = self.config.get('mean_reversion_weight', 0.30)  # 增加均值回归权重
        self.volatility_weight = self.config.get('volatility_weight', 0.20)
        self.microstructure_weight = self.config.get('microstructure_weight', 0.10)  # 降低微观结构权重
        
        # 过滤参数 - 进一步放宽
        self.min_volume_ratio = self.config.get('min_volume_ratio', 0.6)  # 进一步降低
        self.max_volatility = self.config.get('max_volatility', 1.0)  # 进一步放宽
        self.trend_confirmation = self.config.get('trend_confirmation', False)  # 关闭
        
        # 状态变量
        self.portfolio_value = self.initial_capital
        self.position = 0
        self.avg_cost = 0
        self.trades = []
        self.daily_trades = {}
        self.last_trade_time = None
        
        print(f"🚀 Citadel最终版策略初始化完成")
        print(f"   初始资金: ${self.initial_capital:,.2f}")
        print(f"   信号阈值: {self.signal_threshold}")
        print(f"   最大仓位: {self.position_limit * 100:.1f}%")
        print(f"   止损/止盈: {self.stop_loss * 100:.1f}%/{self.take_profit * 100:.1f}%")
        print(f"   追踪止损: {self.trailing_stop * 100:.1f}%")
        print(f"   每日最大交易: {self.max_daily_trades}笔")

    def load_data(self, file_path):
        """加载数据"""
        print(f"📊 加载数据: {file_path}")
        
        # 跳过注释行
        data = pd.read_csv(file_path, comment='#')
        
        # 重命名列
        column_mapping = {
            'Price': 'timestamp',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        data = data.rename(columns=column_mapping)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        print(f"   数据行数: {len(data)}")
        print(f"   时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
        
        return data.sort_values('timestamp').reset_index(drop=True)

    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        print("🔧 计算技术指标...")
        
        # 移动平均线
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # 指数移动平均线
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # ATR
        data['atr'] = self.calculate_atr(data)
        
        # 成交量指标
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # 价格变化率
        data['price_change'] = data['close'].pct_change()
        data['price_change_5'] = data['close'].pct_change(5)
        
        print("   技术指标计算完成")
        return data

    def calculate_atr(self, data, period=14):
        """计算ATR"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()

    def generate_signal(self, data, i):
        """生成交易信号"""
        if i < self.lookback_period:
            return 0
        
        # 动量信号 (权重: 40%)
        momentum_signal = 0
        
        # MACD信号
        macd = data.iloc[i]['macd']
        macd_signal = data.iloc[i]['macd_signal']
        macd_prev = data.iloc[i-1]['macd']
        macd_signal_prev = data.iloc[i-1]['macd_signal']
        
        if macd > macd_signal and macd_prev <= macd_signal_prev:
            momentum_signal += 0.3  # MACD金叉
        elif macd < macd_signal and macd_prev >= macd_signal_prev:
            momentum_signal -= 0.3  # MACD死叉
        
        # 价格动量
        price_momentum = data.iloc[i]['price_change_5']
        momentum_signal += np.tanh(price_momentum * 10) * 0.2
        
        # 移动平均线趋势
        sma_5 = data.iloc[i]['sma_5']
        sma_20 = data.iloc[i]['sma_20']
        if sma_5 > sma_20:
            momentum_signal += 0.1
        else:
            momentum_signal -= 0.1
        
        # 均值回归信号 (权重: 30%)
        mean_reversion_signal = 0
        
        # RSI信号
        rsi = data.iloc[i]['rsi']
        if rsi < 30:
            mean_reversion_signal += 0.4  # 超卖
        elif rsi > 70:
            mean_reversion_signal -= 0.4  # 超买
        else:
            mean_reversion_signal += (50 - rsi) / 50 * 0.2
        
        # 布林带信号
        bb_position = data.iloc[i]['bb_position']
        if bb_position < 0.2:
            mean_reversion_signal += 0.3  # 接近下轨
        elif bb_position > 0.8:
            mean_reversion_signal -= 0.3  # 接近上轨
        
        # 波动率信号 (权重: 20%)
        volatility_signal = 0
        
        # ATR相对波动率
        atr = data.iloc[i]['atr']
        close = data.iloc[i]['close']
        volatility_ratio = atr / close
        
        # 布林带宽度
        bb_width = data.iloc[i]['bb_width']
        
        # 低波动率时增加信号强度
        if volatility_ratio < 0.02:
            volatility_signal = 0.2
        elif volatility_ratio > 0.05:
            volatility_signal = -0.2
        
        # 微观结构信号 (权重: 10%)
        microstructure_signal = 0
        
        # 成交量信号
        volume_ratio = data.iloc[i]['volume_ratio']
        if volume_ratio > 1.5:
            microstructure_signal += 0.3
        elif volume_ratio < 0.5:
            microstructure_signal -= 0.1
        
        # 组合信号
        total_signal = (
            momentum_signal * self.momentum_weight +
            mean_reversion_signal * self.mean_reversion_weight +
            volatility_signal * self.volatility_weight +
            microstructure_signal * self.microstructure_weight
        )
        
        return np.tanh(total_signal)  # 限制在[-1, 1]范围内

    def apply_filters(self, data, i, signal):
        """应用过滤条件"""
        if abs(signal) < 0.01:  # 信号太弱直接过滤
            return 0
        
        # 成交量过滤
        volume_ratio = data.iloc[i]['volume'] / data.iloc[i-20:i]['volume'].mean() if i >= 20 else 1.0
        if volume_ratio < self.min_volume_ratio:
            signal *= 0.5  # 减弱信号而不是完全过滤
        
        # 波动率过滤
        volatility = data.iloc[i]['atr'] / data.iloc[i]['close']
        if volatility > self.max_volatility:
            signal *= 0.3  # 高波动率时减弱信号
        
        return signal

    def can_trade(self, timestamp):
        """检查是否可以交易"""
        date_str = timestamp.strftime('%Y-%m-%d')
        
        # 检查每日交易次数
        if date_str in self.daily_trades:
            if self.daily_trades[date_str] >= self.max_daily_trades:
                return False
        
        # 检查交易间隔
        if self.last_trade_time is not None:
            time_diff = (timestamp - self.last_trade_time).total_seconds()
            if time_diff < self.min_trade_interval:
                return False
        
        return True

    def execute_trade(self, timestamp, symbol, action, price, shares=None):
        """执行交易"""
        if not self.can_trade(timestamp):
            return
        
        date_str = timestamp.strftime('%Y-%m-%d')
        
        if action == 'BUY':
            # 计算买入股数
            if shares is None:
                trade_value = min(
                    self.portfolio_value * self.max_trade_size,
                    self.portfolio_value * self.position_limit - self.position * self.avg_cost
                )
                shares = int(trade_value / price)
            
            if shares > 0:
                cost = shares * price
                self.position += shares
                self.avg_cost = (self.avg_cost * (self.position - shares) + cost) / self.position
                
                trade_record = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,
                    'shares': shares,
                    'price': price,
                    'cost': cost,
                    'portfolio_value': self.portfolio_value,
                    'revenue': None
                }
                self.trades.append(trade_record)
                
                # 更新交易计数
                if date_str not in self.daily_trades:
                    self.daily_trades[date_str] = 0
                self.daily_trades[date_str] += 1
                self.last_trade_time = timestamp
        
        elif action.startswith('SELL'):
            if self.position > 0:
                if shares is None:
                    shares = self.position
                
                shares = min(shares, self.position)
                revenue = shares * price
                
                self.position -= shares
                if self.position == 0:
                    self.avg_cost = 0
                
                # 更新投资组合价值
                self.portfolio_value = self.portfolio_value - shares * self.avg_cost + revenue
                
                trade_record = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,
                    'shares': shares,
                    'price': price,
                    'cost': None,
                    'portfolio_value': self.portfolio_value,
                    'revenue': revenue
                }
                self.trades.append(trade_record)
                
                # 更新交易计数
                if date_str not in self.daily_trades:
                    self.daily_trades[date_str] = 0
                self.daily_trades[date_str] += 1
                self.last_trade_time = timestamp

    def check_risk_management(self, current_price):
        """检查风险管理"""
        if self.position == 0:
            return None
        
        current_value = self.position * current_price
        cost_basis = self.position * self.avg_cost
        
        pnl_pct = (current_value - cost_basis) / cost_basis
        
        # 止损
        if pnl_pct <= -self.stop_loss:
            return 'SELL_STOP_LOSS'
        
        # 止盈
        if pnl_pct >= self.take_profit:
            return 'SELL_TAKE_PROFIT'
        
        # 追踪止损 (简化版)
        if pnl_pct > 0.02:  # 盈利超过2%时启用追踪止损
            trailing_stop_price = current_price * (1 - self.trailing_stop)
            if current_price < trailing_stop_price:
                return 'SELL_TRAILING_STOP'
        
        return None

    def run_backtest(self, data):
        """运行回测"""
        print("🔄 开始回测...")
        
        for i in range(self.lookback_period, len(data)):
            timestamp = data.iloc[i]['timestamp']
            price = data.iloc[i]['close']
            
            # 风险管理检查
            risk_action = self.check_risk_management(price)
            if risk_action:
                self.execute_trade(timestamp, 'AAPL', risk_action, price)
                continue
            
            # 生成交易信号
            signal = self.generate_signal(data, i)
            signal = self.apply_filters(data, i, signal)
            
            # 执行交易
            if abs(signal) > self.signal_threshold and self.can_trade(timestamp):
                if signal > 0 and self.position == 0:  # 买入信号
                    self.execute_trade(timestamp, 'AAPL', 'BUY', price)
                elif signal < 0 and self.position > 0:  # 卖出信号
                    self.execute_trade(timestamp, 'AAPL', 'SELL', price)
        
        print(f"   回测完成，总交易次数: {len(self.trades)}")

    def calculate_performance_metrics(self):
        """计算性能指标"""
        if len(self.trades) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'win_rate': 0,
                'final_portfolio_value': self.initial_capital,
                'avg_trade_return': 0
            }
        
        # 计算总收益
        final_value = self.portfolio_value
        if self.position > 0:
            # 如果还有持仓，按最后价格计算
            last_price = self.trades[-1]['price'] if self.trades else 0
            final_value += self.position * last_price - self.position * self.avg_cost
        
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 计算夏普比率 (简化版)
        if len(self.trades) > 1:
            returns = []
            for i in range(1, len(self.trades)):
                if self.trades[i]['action'].startswith('SELL'):
                    prev_value = self.trades[i-1]['portfolio_value']
                    curr_value = self.trades[i]['portfolio_value']
                    returns.append((curr_value - prev_value) / prev_value)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # 计算最大回撤
        portfolio_values = [trade['portfolio_value'] for trade in self.trades]
        if portfolio_values:
            peak = portfolio_values[0]
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # 计算胜率
        profitable_trades = 0
        total_completed_trades = 0
        
        for i in range(len(self.trades)):
            trade = self.trades[i]
            if trade['action'].startswith('SELL'):
                total_completed_trades += 1
                revenue = trade.get('revenue', 0)
                if revenue is not None and revenue > trade['shares'] * self.avg_cost:
                    profitable_trades += 1
        
        win_rate = profitable_trades / total_completed_trades if total_completed_trades > 0 else 0
        
        # 平均交易收益
        if total_completed_trades > 0:
            avg_trade_return = total_return / total_completed_trades
        else:
            avg_trade_return = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'final_portfolio_value': final_value,
            'avg_trade_return': avg_trade_return
        }

    def save_results(self, strategy_name):
        """保存结果"""
        print("💾 保存结果...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存交易记录
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = f"competitions/citadel/{strategy_name}_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"   交易记录保存到: {trades_file}")
        else:
            trades_file = None
        
        # 计算性能指标
        print("📊 计算性能指标...")
        metrics = self.calculate_performance_metrics()
        
        # 保存回测结果
        results = {
            'strategy_name': strategy_name,
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
        
        results_file = f"competitions/citadel/{strategy_name}_backtest_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"   总收益率: {metrics['total_return'] * 100:.2f}%")
        print(f"   夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"   最大回撤: {metrics['max_drawdown'] * 100:.2f}%")
        print(f"   胜率: {metrics['win_rate'] * 100:.2f}%")
        print(f"   回测结果保存到: {results_file}")
        
        return results_file, trades_file

def main():
    """主函数"""
    print("🎯 Citadel最终版策略回测")
    print("=" * 60)
    
    # 策略配置
    config = {
        'initial_capital': 1000000,
        'lookback_period': 20,
        'signal_threshold': 0.03,
        'position_limit': 0.30,
        'max_trade_size': 0.10,
        'stop_loss': 0.02,
        'take_profit': 0.06,
        'trailing_stop': 0.015,
        'max_daily_trades': 2,
        'min_trade_interval': 0,
        'momentum_weight': 0.40,
        'mean_reversion_weight': 0.30,
        'volatility_weight': 0.20,
        'microstructure_weight': 0.10,
        'min_volume_ratio': 0.6,
        'max_volatility': 1.0,
        'trend_confirmation': False
    }
    
    # 初始化策略
    strategy = CitadelFinalStrategy(config)
    
    # 加载数据
    data = strategy.load_data("examples/data_cache/ohlcv_AAPL_20251005_200622.csv")
    
    # 计算技术指标
    data = strategy.calculate_technical_indicators(data)
    
    # 运行回测
    strategy.run_backtest(data)
    
    # 保存结果
    results_file, trades_file = strategy.save_results('citadel_final')
    
    print("\n✅ 回测完成!")
    print(f"📊 结果文件: {results_file}")
    if trades_file:
        print(f"📈 交易记录: {trades_file}")

if __name__ == "__main__":
    main()