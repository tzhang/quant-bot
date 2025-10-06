#!/usr/bin/env python3
"""
Citadel优化版策略 - 基于现实版分析结果的改进
改进点:
1. 提高信号质量和确认度
2. 优化止盈止损比例
3. 增加更多过滤条件
4. 动态调整参数
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CitadelOptimizedStrategy:
    def __init__(self, config=None):
        """初始化优化版策略"""
        self.config = config or {}
        
        # 基础参数
        self.initial_capital = self.config.get('initial_capital', 1000000)
        self.lookback_period = self.config.get('lookback_period', 30)  # 增加到30天
        self.signal_threshold = self.config.get('signal_threshold', 0.05)  # 降低到0.05以匹配信号强度
        self.position_limit = self.config.get('position_limit', 0.25)  # 降低到25%
        self.max_trade_size = self.config.get('max_trade_size', 0.08)  # 降低到8%
        
        # 优化的风险管理参数
        self.stop_loss = self.config.get('stop_loss', 0.015)  # 1.5%止损
        self.take_profit = self.config.get('take_profit', 0.045)  # 4.5%止盈 (3:1盈亏比)
        self.trailing_stop = self.config.get('trailing_stop', 0.01)  # 1%追踪止损
        self.max_daily_trades = self.config.get('max_daily_trades', 1)  # 日线数据每天最多1次交易
        self.min_trade_interval = self.config.get('min_trade_interval', 0)  # 日线数据无需间隔限制
        
        # 信号权重 - 重新平衡
        self.momentum_weight = self.config.get('momentum_weight', 0.35)
        self.mean_reversion_weight = self.config.get('mean_reversion_weight', 0.25)
        self.volatility_weight = self.config.get('volatility_weight', 0.20)
        self.microstructure_weight = self.config.get('microstructure_weight', 0.20)
        
        # 新增过滤参数
        self.min_volume_ratio = self.config.get('min_volume_ratio', 1.2)  # 最小成交量比率
        self.max_volatility = self.config.get('max_volatility', 0.5)  # 最大波动率
        self.trend_confirmation = self.config.get('trend_confirmation', True)  # 趋势确认
        
        # 状态变量
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.position = 0
        self.avg_cost = 0
        self.trades = []
        self.daily_trades = {}
        self.last_trade_time = None
        self.highest_value = self.initial_capital
        
    def load_data(self, file_path):
        """加载和预处理数据"""
        print(f"📊 加载数据: {file_path}")
        
        # 读取数据，跳过注释行
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
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"   数据行数: {len(data)}")
        print(f"   时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
        
        return data
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        print("🔧 计算技术指标...")
        
        # 移动平均线
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
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
        data['bb_middle'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # 随机指标
        low_14 = data['low'].rolling(14).min()
        high_14 = data['high'].rolling(14).max()
        data['stoch_k'] = 100 * (data['close'] - low_14) / (high_14 - low_14)
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()
        
        # 威廉指标
        data['williams_r'] = -100 * (high_14 - data['close']) / (high_14 - low_14)
        
        # 波动率
        data['volatility'] = data['close'].pct_change().rolling(20).std() * np.sqrt(252)
        data['atr'] = self.calculate_atr(data)
        
        # 成交量指标
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['obv'] = (data['volume'] * np.sign(data['close'].diff())).cumsum()
        
        # 价格变化和趋势
        data['returns'] = data['close'].pct_change()
        data['returns_5d'] = data['close'].pct_change(5)
        data['price_momentum'] = data['close'] / data['close'].shift(10) - 1
        
        # 趋势强度
        data['trend_strength'] = abs(data['close'] - data['sma_20']) / data['sma_20']
        
        print("   技术指标计算完成")
        return data
    
    def calculate_atr(self, data, period=14):
        """计算平均真实波幅"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def generate_signal(self, data, i):
        """生成交易信号 - 优化版"""
        if i < self.lookback_period:
            return 0
        
        current_data = data.iloc[i]
        
        # 1. 动量信号 (权重35%)
        momentum_signal = 0
        
        # MACD信号
        if not pd.isna(current_data['macd']) and not pd.isna(current_data['macd_signal']):
            macd_diff = current_data['macd'] - current_data['macd_signal']
            macd_prev_diff = data.iloc[i-1]['macd'] - data.iloc[i-1]['macd_signal']
            
            # MACD金叉死叉
            if macd_diff > 0 and macd_prev_diff <= 0:
                momentum_signal += 0.4
            elif macd_diff < 0 and macd_prev_diff >= 0:
                momentum_signal -= 0.4
            elif macd_diff > 0:
                momentum_signal += 0.2
            else:
                momentum_signal -= 0.2
        
        # RSI信号 - 更严格的阈值
        if not pd.isna(current_data['rsi']):
            if current_data['rsi'] > 75:
                momentum_signal -= 0.3
            elif current_data['rsi'] < 25:
                momentum_signal += 0.3
            elif current_data['rsi'] > 60:
                momentum_signal -= 0.1
            elif current_data['rsi'] < 40:
                momentum_signal += 0.1
        
        # 价格动量
        if not pd.isna(current_data['price_momentum']):
            if current_data['price_momentum'] > 0.02:
                momentum_signal += 0.2
            elif current_data['price_momentum'] < -0.02:
                momentum_signal -= 0.2
        
        # 2. 均值回归信号 (权重25%)
        mean_reversion_signal = 0
        
        # 布林带信号
        if not pd.isna(current_data['bb_position']):
            if current_data['bb_position'] > 0.9:
                mean_reversion_signal -= 0.5
            elif current_data['bb_position'] < 0.1:
                mean_reversion_signal += 0.5
            elif current_data['bb_position'] > 0.8:
                mean_reversion_signal -= 0.2
            elif current_data['bb_position'] < 0.2:
                mean_reversion_signal += 0.2
        
        # 随机指标
        if not pd.isna(current_data['stoch_k']) and not pd.isna(current_data['stoch_d']):
            if current_data['stoch_k'] > 80 and current_data['stoch_d'] > 80:
                mean_reversion_signal -= 0.3
            elif current_data['stoch_k'] < 20 and current_data['stoch_d'] < 20:
                mean_reversion_signal += 0.3
        
        # 3. 波动率信号 (权重20%)
        volatility_signal = 0
        
        if not pd.isna(current_data['volatility']):
            vol_percentile = np.percentile(data['volatility'].dropna(), 70)
            if current_data['volatility'] > vol_percentile * 1.5:
                volatility_signal -= 0.4  # 高波动率时谨慎
            elif current_data['volatility'] < vol_percentile * 0.3:
                volatility_signal += 0.2  # 低波动率时积极
        
        # ATR信号
        if not pd.isna(current_data['atr']):
            atr_ma = data['atr'].rolling(20).mean().iloc[i]
            if not pd.isna(atr_ma):
                if current_data['atr'] > atr_ma * 1.5:
                    volatility_signal -= 0.2
        
        # 4. 微观结构信号 (权重20%)
        microstructure_signal = 0
        
        # 成交量确认
        if not pd.isna(current_data['volume_ratio']):
            if current_data['volume_ratio'] > 1.5:
                if current_data['returns'] > 0:
                    microstructure_signal += 0.3
                else:
                    microstructure_signal -= 0.3
            elif current_data['volume_ratio'] < 0.5:
                microstructure_signal -= 0.2
        
        # OBV趋势
        if i > 5:
            obv_trend = current_data['obv'] - data.iloc[i-5]['obv']
            price_trend = current_data['close'] - data.iloc[i-5]['close']
            
            if obv_trend > 0 and price_trend > 0:
                microstructure_signal += 0.2
            elif obv_trend < 0 and price_trend < 0:
                microstructure_signal -= 0.2
            elif (obv_trend > 0 and price_trend < 0) or (obv_trend < 0 and price_trend > 0):
                microstructure_signal -= 0.1  # 背离信号
        
        # 综合信号
        total_signal = (momentum_signal * self.momentum_weight +
                       mean_reversion_signal * self.mean_reversion_weight +
                       volatility_signal * self.volatility_weight +
                       microstructure_signal * self.microstructure_weight)
        
        return total_signal
    
    def apply_filters(self, data, i, signal):
        """应用过滤条件"""
        if signal == 0:
            return 0
        
        current_data = data.iloc[i]
        
        # 成交量过滤
        if not pd.isna(current_data['volume_ratio']):
            if current_data['volume_ratio'] < self.min_volume_ratio:
                return 0
        
        # 波动率过滤
        if not pd.isna(current_data['volatility']):
            if current_data['volatility'] > self.max_volatility:
                return 0
        
        # 趋势确认过滤
        if self.trend_confirmation and i > 10:
            # 检查短期和长期趋势一致性
            short_trend = data['sma_5'].iloc[i] - data['sma_5'].iloc[i-5]
            long_trend = data['sma_20'].iloc[i] - data['sma_20'].iloc[i-10]
            
            if signal > 0 and (short_trend < 0 or long_trend < 0):
                signal *= 0.5  # 减弱信号
            elif signal < 0 and (short_trend > 0 or long_trend > 0):
                signal *= 0.5
        
        return signal
    
    def can_trade(self, timestamp):
        """检查是否可以交易"""
        # 检查交易间隔
        if self.last_trade_time:
            time_diff = (timestamp - self.last_trade_time).total_seconds()
            if time_diff < self.min_trade_interval:
                return False
        
        # 检查每日交易次数
        date_str = timestamp.strftime('%Y-%m-%d')
        daily_count = self.daily_trades.get(date_str, 0)
        if daily_count >= self.max_daily_trades:
            return False
        
        return True
    
    def execute_trade(self, timestamp, symbol, action, price, shares=None):
        """执行交易"""
        if not shares:
            if action == 'BUY':
                max_shares = int((self.cash * self.max_trade_size) / price)
                shares = min(max_shares, int((self.cash * self.position_limit) / price))
            else:
                shares = self.position
        
        if shares <= 0:
            return
        
        cost = shares * price
        
        if action == 'BUY':
            if cost > self.cash:
                return
            
            self.cash -= cost
            if self.position == 0:
                self.avg_cost = price
            else:
                total_cost = self.position * self.avg_cost + cost
                self.position += shares
                self.avg_cost = total_cost / self.position
                shares = self.position - shares  # 记录新增份额
            
            self.position += shares
        
        else:  # SELL
            if shares > self.position:
                shares = self.position
            
            revenue = shares * price
            self.cash += revenue
            self.position -= shares
            
            if self.position == 0:
                self.avg_cost = 0
        
        # 更新组合价值
        self.portfolio_value = self.cash + self.position * price
        self.highest_value = max(self.highest_value, self.portfolio_value)
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'cost': cost if action == 'BUY' else None,
            'portfolio_value': self.portfolio_value,
            'revenue': revenue if action == 'SELL' else None
        }
        
        self.trades.append(trade)
        
        # 更新交易统计
        date_str = timestamp.strftime('%Y-%m-%d')
        self.daily_trades[date_str] = self.daily_trades.get(date_str, 0) + 1
        self.last_trade_time = timestamp
    
    def check_risk_management(self, current_price):
        """检查风险管理条件"""
        if self.position == 0:
            return None
        
        # 止损检查
        if current_price <= self.avg_cost * (1 - self.stop_loss):
            return 'SELL_STOP_LOSS'
        
        # 止盈检查
        if current_price >= self.avg_cost * (1 + self.take_profit):
            return 'SELL_TAKE_PROFIT'
        
        # 追踪止损检查
        if self.portfolio_value < self.highest_value * (1 - self.trailing_stop):
            return 'SELL_TRAILING_STOP'
        
        return None
    
    def run_backtest(self, data):
        """运行回测"""
        print("🔄 开始回测...")
        
        for i in range(len(data)):
            current_data = data.iloc[i]
            timestamp = current_data['timestamp']
            price = current_data['close']
            
            # 更新组合价值
            if self.position > 0:
                self.portfolio_value = self.cash + self.position * price
                self.highest_value = max(self.highest_value, self.portfolio_value)
            
            # 检查风险管理
            risk_action = self.check_risk_management(price)
            if risk_action and self.can_trade(timestamp):
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
        print("📊 计算性能指标...")
        
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
        
        # 计算收益率
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # 计算每日收益率
        trades_df = pd.DataFrame(self.trades)
        trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
        
        daily_values = trades_df.groupby('date')['portfolio_value'].last()
        daily_returns = daily_values.pct_change().dropna()
        
        # 夏普比率
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        peak = daily_values.expanding().max()
        drawdown = (daily_values - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # 胜率计算
        profitable_trades = 0
        total_completed_trades = 0
        
        for i in range(len(self.trades)):
            trade = self.trades[i]
            if trade['action'].startswith('SELL'):
                total_completed_trades += 1
                # 检查revenue是否为None
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
            'final_portfolio_value': self.portfolio_value,
            'avg_trade_return': avg_trade_return
        }
    
    def save_results(self, strategy_name):
        """保存回测结果"""
        print("💾 保存结果...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存交易记录
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = f'competitions/citadel/{strategy_name}_trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False)
            print(f"   交易记录保存到: {trades_file}")
        
        # 计算性能指标
        metrics = self.calculate_performance_metrics()
        
        # 保存回测结果
        results = {
            'strategy_name': strategy_name,
            'timestamp': timestamp,
            'config': {
                'initial_capital': self.initial_capital,
                'lookback_period': self.lookback_period,
                'signal_threshold': self.signal_threshold,
                'position_limit': self.position_limit,
                'max_trade_size': self.max_trade_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'trailing_stop': self.trailing_stop,
                'max_daily_trades': self.max_daily_trades,
                'min_trade_interval': self.min_trade_interval,
                'momentum_weight': self.momentum_weight,
                'mean_reversion_weight': self.mean_reversion_weight,
                'volatility_weight': self.volatility_weight,
                'microstructure_weight': self.microstructure_weight,
                'min_volume_ratio': self.min_volume_ratio,
                'max_volatility': self.max_volatility,
                'trend_confirmation': self.trend_confirmation
            },
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
        
        results_file = f'competitions/citadel/{strategy_name}_backtest_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   总收益率: {metrics['total_return']*100:.2f}%")
        print(f"   夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"   最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"   胜率: {metrics['win_rate']*100:.2f}%")
        print(f"   回测结果保存到: {results_file}")
        
        return results_file, trades_file if self.trades else None

def main():
    """主函数"""
    print("🎯 Citadel优化版策略回测")
    print("=" * 60)
    
    # 策略配置
    config = {
        'initial_capital': 1000000,
        'lookback_period': 30,
        'signal_threshold': 0.05,  # 降低信号阈值
        'position_limit': 0.25,
        'max_trade_size': 0.08,
        'stop_loss': 0.015,
        'take_profit': 0.045,
        'trailing_stop': 0.01,
        'max_daily_trades': 3,
        'min_trade_interval': 0,  # 适应日线数据
        'momentum_weight': 0.35,
        'mean_reversion_weight': 0.25,
        'volatility_weight': 0.20,
        'microstructure_weight': 0.20,
        'min_volume_ratio': 0.8,  # 降低成交量要求
        'max_volatility': 0.8,  # 放宽波动率限制
        'trend_confirmation': False  # 关闭趋势确认
    }
    
    # 初始化策略
    strategy = CitadelOptimizedStrategy(config)
    
    print("🚀 Citadel优化版策略初始化完成")
    print(f"   初始资金: ${strategy.initial_capital:,.2f}")
    print(f"   信号阈值: {strategy.signal_threshold}")
    print(f"   最大仓位: {strategy.position_limit*100:.1f}%")
    print(f"   止损/止盈: {strategy.stop_loss*100:.1f}%/{strategy.take_profit*100:.1f}%")
    print(f"   追踪止损: {strategy.trailing_stop*100:.1f}%")
    print(f"   每日最大交易: {strategy.max_daily_trades}笔")
    
    # 加载数据
    data = strategy.load_data('examples/data_cache/ohlcv_AAPL_20251005_200622.csv')
    
    # 计算技术指标
    data = strategy.calculate_technical_indicators(data)
    
    # 运行回测
    strategy.run_backtest(data)
    
    # 保存结果
    results_file, trades_file = strategy.save_results('citadel_optimized')
    
    print(f"\n✅ 回测完成!")
    print(f"📊 结果文件: {results_file}")
    if trades_file:
        print(f"📈 交易记录: {trades_file}")

if __name__ == "__main__":
    main()