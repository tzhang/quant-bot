#!/usr/bin/env python3
"""
Citadel 保守优化高频交易策略
解决过度交易问题的改进版本
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.ml.terminal_ai_tools import run_terminal_ai_simulation
from src.ml.feature_engineering import AdvancedFeatureEngineer
from src.risk.risk_manager import RiskManager

class ConservativeCitadelHFTStrategy:
    """保守优化的Citadel高频交易策略"""
    
    def __init__(self, config_file: Optional[str] = None):
        # 首先设置logger
        self.logger = self._setup_logger()
        
        # 加载配置
        self.config = self._load_config(config_file)
        
        # 初始化组件
        self.feature_engineer = AdvancedFeatureEngineer()
        self.risk_manager = RiskManager()
        
        # 策略参数（保守设置）
        self.lookback_period = 20  # 增加回看期
        self.signal_threshold = 0.8  # 大幅提高信号阈值
        self.position_limit = 0.05  # 降低仓位限制
        self.max_trade_size = 5000  # 降低最大交易规模
        
        # 风险管理参数（更严格）
        self.stop_loss = 0.003  # 更严格的止损
        self.take_profit = 0.01  # 更保守的止盈
        self.max_portfolio_risk = 0.01  # 降低组合风险
        
        # 技术指标参数
        self.rsi_period = 14  # 标准RSI周期
        self.bb_period = 20  # 标准布林带周期
        self.bb_std_multiplier = 2
        self.macd_fast = 12  # 标准MACD参数
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 信号权重（更平衡）
        self.signal_weights = {
            'momentum': 0.3,
            'mean_reversion': 0.3,
            'volatility': 0.2,
            'volume': 0.1,
            'microstructure': 0.1
        }
        
        # 交易状态
        self.current_position = 0
        self.portfolio_value = 1000000  # 初始资金100万
        self.trades = []
        self.entry_price = None
        self.entry_time = None
        self.last_trade_time = None
        self.min_trade_interval = 60  # 最小交易间隔（秒）
        
        self.logger.info("🚀 保守优化Citadel高频交易策略初始化完成")
    
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict:
        """加载配置文件"""
        if config_file is None:
            config_file = "competitions/citadel/citadel_optimized_config_20251006_205957.json"
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"✅ 配置文件加载成功: {config_file}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"⚠️ 配置文件未找到: {config_file}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置（保守设置）"""
        return {
            "strategy_name": "ConservativeCitadelHFT",
            "version": "1.0",
            "description": "保守优化的Citadel高频交易策略",
            "signal_parameters": {
                "lookback_period": 20,
                "signal_threshold": 0.8,
                "position_limit": 0.05,
                "max_trade_size": 5000
            },
            "risk_management": {
                "stop_loss": 0.003,
                "take_profit": 0.01,
                "max_portfolio_risk": 0.01,
                "max_single_position": 0.05
            }
        }
    
    def load_market_data(self, data_file: str = None) -> pd.DataFrame:
        """加载市场数据"""
        if data_file is None:
            data_file = "competitions/citadel/data/sample_market_data.csv"
        
        try:
            if os.path.exists(data_file):
                data = pd.read_csv(data_file)
                self.logger.info(f"✅ 数据文件加载成功: {data_file}")
            else:
                self.logger.warning(f"⚠️ 数据文件不存在，生成模拟数据: {data_file}")
                data = self._generate_high_quality_sample_data()
            
            # 确保时间列存在并转换
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            elif 'time' in data.columns:
                data['timestamp'] = pd.to_datetime(data['time'])
            else:
                # 生成时间序列
                start_time = datetime.now() - timedelta(days=4)
                data['timestamp'] = pd.date_range(start=start_time, periods=len(data), freq='1S')
            
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            return self._generate_high_quality_sample_data()
    
    def _generate_high_quality_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """生成高质量的模拟高频数据（减少样本数量）"""
        self.logger.info(f"🔄 生成 {n_samples} 条高质量模拟数据...")
        
        np.random.seed(42)
        
        # 生成时间序列（更长的时间间隔）
        start_time = datetime.now() - timedelta(days=4)
        timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5S')  # 5秒间隔
        
        # 生成更真实的价格数据
        base_price = 100.0
        returns = np.random.normal(0, 0.0002, n_samples)  # 稍大的波动
        
        # 添加趋势和周期性
        trend = np.linspace(0, 0.01, n_samples)  # 轻微上升趋势
        cycle = 0.002 * np.sin(np.linspace(0, 2*np.pi, n_samples))  # 周期性波动
        
        returns += trend + cycle
        
        # 生成价格序列
        prices = base_price * np.exp(np.cumsum(returns))
        
        # 生成OHLC数据
        high_noise = np.random.exponential(0.001, n_samples)
        low_noise = -np.random.exponential(0.001, n_samples)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + high_noise,
            'low': prices + low_noise,
            'close': prices,
            'volume': np.random.lognormal(8, 1, n_samples).astype(int),
            'vwap': prices + np.random.normal(0, 0.0002, n_samples)
        })
        
        # 确保OHLC逻辑正确
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        return data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        data = data.copy()
        
        # 移动平均线
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['ema_5'] = data['close'].ewm(span=5).mean()
        data['ema_10'] = data['close'].ewm(span=10).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        bb_sma = data['close'].rolling(window=self.bb_period).mean()
        bb_std = data['close'].rolling(window=self.bb_period).std()
        data['bb_upper'] = bb_sma + (bb_std * self.bb_std_multiplier)
        data['bb_lower'] = bb_sma - (bb_std * self.bb_std_multiplier)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # MACD
        ema_fast = data['close'].ewm(span=self.macd_fast).mean()
        ema_slow = data['close'].ewm(span=self.macd_slow).mean()
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=self.macd_signal).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # 波动率指标
        data['volatility'] = data['close'].rolling(window=20).std()
        data['volatility_ratio'] = data['volatility'] / data['volatility'].rolling(window=50).mean()
        
        # 成交量指标
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # 价格位置指标
        data['price_position'] = (data['close'] - data['close'].rolling(window=50).min()) / \
                                (data['close'].rolling(window=50).max() - data['close'].rolling(window=50).min())
        
        return data
    
    def generate_signals(self, data: pd.DataFrame, idx: int) -> float:
        """生成交易信号（保守版本）"""
        if idx < self.lookback_period:
            return 0
        
        signals = {}
        current_row = data.iloc[idx]
        
        # 动量信号（权重: 0.3）- 更严格的条件
        momentum_score = 0
        if not pd.isna(current_row['sma_10']) and not pd.isna(current_row['sma_20']):
            if current_row['close'] > current_row['sma_10'] > current_row['sma_20']:
                momentum_score += 0.4
            elif current_row['close'] < current_row['sma_10'] < current_row['sma_20']:
                momentum_score -= 0.4
        
        if not pd.isna(current_row['macd']) and not pd.isna(current_row['macd_signal']):
            if current_row['macd'] > current_row['macd_signal'] and current_row['macd_histogram'] > 0:
                momentum_score += 0.3
            elif current_row['macd'] < current_row['macd_signal'] and current_row['macd_histogram'] < 0:
                momentum_score -= 0.3
        
        signals['momentum'] = momentum_score
        
        # 均值回归信号（权重: 0.3）- 更严格的条件
        mean_reversion_score = 0
        rsi = current_row['rsi']
        if not pd.isna(rsi):
            if rsi < 20:  # 极度超卖
                mean_reversion_score += 0.6
            elif rsi > 80:  # 极度超买
                mean_reversion_score -= 0.6
        
        bb_pos = current_row['bb_position']
        if not pd.isna(bb_pos):
            if bb_pos < 0.05:  # 极度接近下轨
                mean_reversion_score += 0.4
            elif bb_pos > 0.95:  # 极度接近上轨
                mean_reversion_score -= 0.4
        
        signals['mean_reversion'] = mean_reversion_score
        
        # 波动率信号（权重: 0.2）
        vol_ratio = current_row['volatility_ratio']
        if not pd.isna(vol_ratio):
            if vol_ratio > 3.0:  # 极高波动
                volatility_score = -0.5
            elif vol_ratio < 0.3:  # 极低波动
                volatility_score = 0.3
            else:
                volatility_score = 0
        else:
            volatility_score = 0
        signals['volatility'] = volatility_score
        
        # 成交量信号（权重: 0.1）
        vol_ratio = current_row['volume_ratio']
        if not pd.isna(vol_ratio):
            if vol_ratio > 2.0:  # 显著放量
                volume_score = 0.3
            elif vol_ratio < 0.3:  # 显著缩量
                volume_score = -0.2
            else:
                volume_score = 0
        else:
            volume_score = 0
        signals['volume'] = volume_score
        
        # 微观结构信号（权重: 0.1）
        close = current_row['close']
        if 'vwap' in current_row.index:
            vwap = current_row['vwap']
            if not pd.isna(vwap):
                if close > vwap * 1.002:  # 更严格的阈值
                    microstructure_score = 0.2
                elif close < vwap * 0.998:
                    microstructure_score = -0.2
                else:
                    microstructure_score = 0
            else:
                microstructure_score = 0
        else:
            microstructure_score = 0
        signals['microstructure'] = microstructure_score
        
        # 加权组合信号
        final_signal = sum(signals[key] * self.signal_weights[key] for key in signals)
        
        return final_signal
    
    def can_trade(self, timestamp: pd.Timestamp) -> bool:
        """检查是否可以交易（防止过度交易）"""
        if self.last_trade_time is None:
            return True
        
        time_diff = (timestamp - self.last_trade_time).total_seconds()
        return time_diff >= self.min_trade_interval
    
    def execute_trade(self, signal: float, current_price: float, timestamp: pd.Timestamp):
        """执行交易（保守版本）"""
        if abs(signal) < self.signal_threshold:
            return
        
        if not self.can_trade(timestamp):
            return
        
        # 计算交易规模（更保守）
        base_size = min(self.max_trade_size, 
                       abs(signal) * self.max_trade_size * 0.5,  # 减半
                       self.portfolio_value * self.position_limit)
        
        # 根据波动率调整交易规模
        trade_size = base_size * 0.8  # 进一步减小
        
        if signal > 0 and self.current_position <= 0:
            # 买入信号
            if self.current_position < 0:
                # 先平空仓
                self._close_position(current_price, timestamp, "平空")
            
            # 开多仓
            self.current_position = trade_size
            self.entry_price = current_price
            self.entry_time = timestamp
            self.last_trade_time = timestamp
            
            trade_record = {
                'timestamp': timestamp,
                'action': '买入',
                'price': current_price,
                'size': trade_size,
                'signal': signal,
                'portfolio_value': self.portfolio_value
            }
            self.trades.append(trade_record)
            
        elif signal < 0 and self.current_position >= 0:
            # 卖出信号
            if self.current_position > 0:
                # 先平多仓
                self._close_position(current_price, timestamp, "平多")
            
            # 开空仓
            self.current_position = -trade_size
            self.entry_price = current_price
            self.entry_time = timestamp
            self.last_trade_time = timestamp
            
            trade_record = {
                'timestamp': timestamp,
                'action': '卖出',
                'price': current_price,
                'size': trade_size,
                'signal': signal,
                'portfolio_value': self.portfolio_value
            }
            self.trades.append(trade_record)
    
    def _close_position(self, current_price: float, timestamp: pd.Timestamp, reason: str):
        """平仓"""
        if self.current_position == 0:
            return
        
        # 计算盈亏
        if self.current_position > 0:
            # 平多仓
            pnl = (current_price - self.entry_price) * abs(self.current_position)
        else:
            # 平空仓
            pnl = (self.entry_price - current_price) * abs(self.current_position)
        
        self.portfolio_value += pnl
        
        trade_record = {
            'timestamp': timestamp,
            'action': reason,
            'price': current_price,
            'size': abs(self.current_position),
            'pnl': pnl,
            'portfolio_value': self.portfolio_value,
            'entry_price': self.entry_price,
            'hold_time': (timestamp - self.entry_time).total_seconds() if self.entry_time else 0
        }
        self.trades.append(trade_record)
        
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
    
    def check_risk_management(self, current_price: float, timestamp: pd.Timestamp):
        """风险管理检查"""
        if self.current_position == 0 or self.entry_price is None:
            return
        
        # 计算当前盈亏比例
        if self.current_position > 0:
            pnl_ratio = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_ratio = (self.entry_price - current_price) / self.entry_price
        
        # 止损检查
        if pnl_ratio <= -self.stop_loss:
            self._close_position(current_price, timestamp, "止损")
            return
        
        # 止盈检查
        if pnl_ratio >= self.take_profit:
            self._close_position(current_price, timestamp, "止盈")
            return
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """运行回测"""
        self.logger.info("🚀 开始运行保守优化策略回测...")
        
        # 计算技术指标
        data = self.calculate_technical_indicators(data)
        
        # 回测循环
        for idx in range(len(data)):
            current_row = data.iloc[idx]
            current_price = current_row['close']
            timestamp = current_row['timestamp']
            
            # 风险管理检查
            self.check_risk_management(current_price, timestamp)
            
            # 生成信号
            signal = self.generate_signals(data, idx)
            
            # 执行交易
            self.execute_trade(signal, current_price, timestamp)
        
        # 最终平仓
        if self.current_position != 0:
            final_price = data.iloc[-1]['close']
            final_timestamp = data.iloc[-1]['timestamp']
            self._close_position(final_price, final_timestamp, "最终平仓")
        
        # 计算性能指标
        performance = self._calculate_performance_metrics()
        
        return performance
    
    def _calculate_performance_metrics(self) -> Dict:
        """计算性能指标"""
        if not self.trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'win_rate': 0,
                'final_portfolio_value': self.portfolio_value,
                'average_trade_return': 0
            }
        
        # 提取交易记录
        trades_df = pd.DataFrame(self.trades)
        
        # 计算收益序列
        pnl_trades = trades_df[trades_df['action'].isin(['止损', '止盈', '平多', '平空', '最终平仓'])]
        
        if len(pnl_trades) == 0:
            returns = [0]
        else:
            returns = pnl_trades['pnl'].tolist()
        
        # 总收益率
        total_return = (self.portfolio_value - 1000000) / 1000000
        
        # 夏普比率
        if len(returns) > 1:
            returns_array = np.array(returns)
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        portfolio_values = trades_df['portfolio_value'].tolist()
        if portfolio_values:
            peak = portfolio_values[0]
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0
        
        # 胜率
        winning_trades = len([r for r in returns if r > 0])
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 平均交易收益
        avg_trade_return = np.mean(returns) if returns else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_portfolio_value': self.portfolio_value,
            'average_trade_return': avg_trade_return
        }
    
    def save_results(self, performance: Dict):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存交易记录
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = f"competitions/citadel/citadel_conservative_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            self.logger.info(f"交易记录已保存到: {trades_file}")
        
        # 保存回测结果
        results = {
            'timestamp': timestamp,
            'strategy': 'ConservativeCitadelHFT',
            'config': self.config,
            'performance_metrics': performance,
            'trades_summary': {
                'total_trades': len(self.trades),
                'final_portfolio_value': self.portfolio_value
            }
        }
        
        results_file = f"competitions/citadel/citadel_conservative_backtest_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"回测结果已保存到: {results_file}")
        
        return results_file

def main():
    """主函数"""
    print("🚀 Citadel 保守优化高频交易策略")
    print("=" * 60)
    
    # 创建策略实例
    strategy = ConservativeCitadelHFTStrategy()
    
    # 加载数据
    data = strategy.load_market_data()
    print(f"数据加载完成，共 {len(data)} 条记录")
    print(f"数据时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    
    # 运行回测
    performance = strategy.run_backtest(data)
    
    # 保存结果
    strategy.save_results(performance)
    
    # 显示结果
    print(f"\n📊 回测结果:")
    print("-" * 40)
    print(f"总收益率: {performance['total_return']:.4f} ({performance['total_return']*100:.2f}%)")
    print(f"夏普比率: {performance['sharpe_ratio']:.4f}")
    print(f"最大回撤: {performance['max_drawdown']:.4f} ({performance['max_drawdown']*100:.2f}%)")
    print(f"总交易次数: {performance['total_trades']}")
    print(f"胜率: {performance['win_rate']:.4f} ({performance['win_rate']*100:.2f}%)")
    print(f"最终组合价值: ${performance['final_portfolio_value']:,.2f}")
    print(f"平均交易收益: {performance['average_trade_return']:.2f}")
    
    print(f"\n🎉 保守优化策略回测完成!")

if __name__ == "__main__":
    main()