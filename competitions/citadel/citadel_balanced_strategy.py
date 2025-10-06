#!/usr/bin/env python3
"""
Citadel 平衡优化高频交易策略
基于网格搜索结果，平衡交易频率和收益率
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

class BalancedCitadelHFTStrategy:
    """平衡优化的Citadel高频交易策略"""
    
    def __init__(self, config_path: str = None):
        """初始化策略"""
        self.logger = get_logger(__name__)
        
        # 平衡优化的策略参数
        self.config = {
            "strategy_name": "BalancedCitadelHFT",
            "version": "1.0",
            "description": "平衡优化的Citadel高频交易策略",
            "signal_parameters": {
                "lookback_period": 15,  # 增加回看期
                "signal_threshold": 0.35,  # 适中的信号阈值
                "position_limit": 0.08,  # 适中的仓位限制
                "max_trade_size": 8000  # 适中的交易规模
            },
            "risk_management": {
                "stop_loss": 0.008,  # 适中的止损
                "take_profit": 0.020,  # 适中的止盈
                "max_portfolio_risk": 0.015,
                "max_single_position": 0.08
            },
            "technical_indicators": {
                "rsi_period": 12,  # 增加RSI周期
                "bb_period": 20,  # 增加布林带周期
                "bb_std_multiplier": 2.2,
                "macd_fast": 10,
                "macd_slow": 20,
                "macd_signal": 8,
                "volatility_window": 15,
                "volume_window": 15
            },
            "signal_weights": {
                "momentum": 0.30,
                "mean_reversion": 0.25,
                "volatility": 0.20,
                "volume": 0.15,
                "microstructure": 0.10
            },
            "market_conditions": {
                "min_volume_threshold": 2000,  # 提高最小成交量要求
                "max_spread_threshold": 0.008,
                "volatility_filter": True,
                "market_hours_only": False
            },
            "optimization_settings": {
                "adaptive_thresholds": True,
                "dynamic_position_sizing": True,
                "regime_detection": True,
                "correlation_filter": True
            },
            "performance_targets": {
                "target_sharpe": 1.5,
                "max_drawdown_limit": 0.05,
                "min_win_rate": 0.55
            },
            "execution_settings": {
                "slippage": 0.0001,
                "commission": 0.0001,
                "market_impact": 0.00005
            }
        }
        
        # 如果提供了配置文件路径，加载外部配置
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    external_config = json.load(f)
                # 只更新部分参数，保持平衡策略的核心设置
                if 'signal_parameters' in external_config:
                    self.config['signal_parameters'].update(external_config['signal_parameters'])
                if 'technical_indicators' in external_config:
                    self.config['technical_indicators'].update(external_config['technical_indicators'])
                self.logger.info(f"✅ 配置文件加载成功: {config_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
        
        # 初始化风险管理器
        self.risk_manager = RiskManager()
        
        # 初始化交易状态
        self.portfolio_value = 1000000  # 初始资金100万
        self.positions = {}
        self.trades = []
        self.daily_pnl = []
        
        self.logger.info("🚀 平衡优化Citadel高频交易策略初始化完成")
    
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
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    def _generate_sample_data(self, n_points: int = 50000) -> pd.DataFrame:
        """生成模拟高频交易数据"""
        np.random.seed(42)
        
        # 生成时间序列
        start_time = datetime.now() - timedelta(days=4)
        timestamps = pd.date_range(start=start_time, periods=n_points, freq='5S')
        
        # 生成价格数据（几何布朗运动）
        initial_price = 100.0
        dt = 1/252/24/60/12  # 5秒间隔
        mu = 0.05  # 年化收益率
        sigma = 0.2  # 年化波动率
        
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
            'high': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
            'low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
            'close': prices,
            'volume': np.random.lognormal(8, 1, n_points).astype(int),
            'bid_price': np.array(prices) * (1 - np.random.uniform(0.0001, 0.001, n_points)),
            'ask_price': np.array(prices) * (1 + np.random.uniform(0.0001, 0.001, n_points)),
            'bid_size': np.random.lognormal(5, 1, n_points).astype(int),
            'ask_size': np.random.lognormal(5, 1, n_points).astype(int)
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
            # 如果没有bid/ask数据，使用估算的价差
            data['spread'] = 0.001  # 默认价差0.1%
        
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
            
            # 1. 动量信号
            momentum_score = 0
            if current_row['macd_histogram'] > 0:
                momentum_score += 0.3
            if current_row['close'] > current_row['sma_10']:
                momentum_score += 0.2
            if current_row['close'] > current_row['sma_20']:
                momentum_score += 0.2
            
            # 2. 均值回归信号
            mean_reversion_score = 0
            if current_row['rsi'] < 30:
                mean_reversion_score += 0.4  # 超卖
            elif current_row['rsi'] > 70:
                mean_reversion_score -= 0.4  # 超买
            
            if current_row['bb_position'] < 0.2:
                mean_reversion_score += 0.3  # 接近下轨
            elif current_row['bb_position'] > 0.8:
                mean_reversion_score -= 0.3  # 接近上轨
            
            # 3. 波动率信号
            volatility_score = 0
            if idx >= 20:
                vol_percentile = np.percentile(data['volatility'].iloc[idx-20:idx], 50)
                if current_row['volatility'] > vol_percentile:
                    volatility_score += 0.2
            
            # 4. 成交量信号
            volume_score = 0
            if current_row['volume_ratio'] > 1.5:
                volume_score += 0.3
            elif current_row['volume_ratio'] > 1.2:
                volume_score += 0.1
            
            # 5. 微观结构信号
            microstructure_score = 0
            if current_row['spread'] < 0.001:  # 低价差
                microstructure_score += 0.2
            
            # 价格相对于移动平均线的位置
            if 'sma_10' in current_row.index and not pd.isna(current_row['sma_10']):
                price_ma_ratio = current_row['close'] / current_row['sma_10']
                if 0.998 < price_ma_ratio < 1.002:  # 接近移动平均线
                    microstructure_score += 0.1
            
            # 综合信号计算
            weights = self.config['signal_weights']
            total_signal = (
                momentum_score * weights['momentum'] +
                mean_reversion_score * weights['mean_reversion'] +
                volatility_score * weights['volatility'] +
                volume_score * weights['volume'] +
                microstructure_score * weights['microstructure']
            )
            
            # 市场条件过滤
            if (current_row['volume'] < self.config['market_conditions']['min_volume_threshold'] or
                current_row['spread'] > self.config['market_conditions']['max_spread_threshold']):
                total_signal = 0
            
            signals.append(total_signal)
        
        # 添加前面的空值
        data['signal'] = [0] * lookback + signals
        return data
    
    def execute_trades(self, data: pd.DataFrame) -> List[Dict]:
        """执行交易"""
        trades = []
        signal_threshold = self.config['signal_parameters']['signal_threshold']
        max_trade_size = self.config['signal_parameters']['max_trade_size']
        
        for idx, row in data.iterrows():
            if abs(row['signal']) > signal_threshold:
                # 确定交易方向和规模
                if row['signal'] > signal_threshold:
                    side = 'buy'
                    size = min(max_trade_size, self.portfolio_value * 0.02)  # 限制单笔交易规模
                elif row['signal'] < -signal_threshold:
                    side = 'sell'
                    size = min(max_trade_size, self.portfolio_value * 0.02)
                else:
                    continue
                
                # 计算交易成本
                price = row['close']
                slippage = self.config['execution_settings']['slippage']
                commission = self.config['execution_settings']['commission']
                
                if side == 'buy':
                    execution_price = price * (1 + slippage)
                else:
                    execution_price = price * (1 - slippage)
                
                # 记录交易
                trade = {
                    'timestamp': row['timestamp'],
                    'side': side,
                    'size': size,
                    'price': execution_price,
                    'signal_strength': row['signal'],
                    'commission': size * commission
                }
                
                trades.append(trade)
                
                # 更新组合价值（简化计算）
                if side == 'buy':
                    self.portfolio_value -= size * execution_price + trade['commission']
                else:
                    self.portfolio_value += size * execution_price - trade['commission']
        
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
        trade_returns = []
        portfolio_values = [1000000]  # 初始值
        
        for i in range(len(trades) - 1):
            if i % 2 == 0 and i + 1 < len(trades):  # 配对交易
                entry_trade = trades[i]
                exit_trade = trades[i + 1]
                
                if entry_trade['side'] == 'buy' and exit_trade['side'] == 'sell':
                    pnl = (exit_trade['price'] - entry_trade['price']) * entry_trade['size']
                elif entry_trade['side'] == 'sell' and exit_trade['side'] == 'buy':
                    pnl = (entry_trade['price'] - exit_trade['price']) * entry_trade['size']
                else:
                    continue
                
                # 扣除手续费
                pnl -= (entry_trade['commission'] + exit_trade['commission'])
                trade_returns.append(pnl)
                portfolio_values.append(portfolio_values[-1] + pnl)
        
        if not trade_returns:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': len(trades),
                'win_rate': 0,
                'final_portfolio_value': self.portfolio_value,
                'average_trade_return': 0
            }
        
        # 计算指标
        total_return = (self.portfolio_value - 1000000) / 1000000
        
        if len(trade_returns) > 1:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) if np.std(trade_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 胜率
        winning_trades = sum(1 for ret in trade_returns if ret > 0)
        win_rate = winning_trades / len(trade_returns) if trade_returns else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'final_portfolio_value': self.portfolio_value,
            'average_trade_return': np.mean(trade_returns) if trade_returns else 0
        }
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """运行回测"""
        self.logger.info("🚀 开始运行平衡优化策略回测...")
        
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
            'strategy': 'BalancedCitadelHFT',
            'config': self.config,
            'performance_metrics': performance,
            'trades_summary': {
                'total_trades': len(trades),
                'final_portfolio_value': self.portfolio_value
            }
        }
        
        # 保存到文件
        results_path = f"competitions/citadel/citadel_balanced_backtest_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"回测结果已保存到: {results_path}")
        
        return results

def main():
    """主函数"""
    print("🚀 Citadel 平衡优化高频交易策略")
    print("=" * 60)
    
    try:
        # 初始化策略
        config_path = "competitions/citadel/citadel_optimized_config_20251006_205957.json"
        strategy = BalancedCitadelHFTStrategy(config_path)
        
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
        
        print(f"\n🎉 平衡优化策略回测完成!")
        
    except Exception as e:
        print(f"❌ 策略运行失败: {e}")
        raise

if __name__ == "__main__":
    main()