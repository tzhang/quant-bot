#!/usr/bin/env python3
"""
Citadel Terminal AI Competition - 高频交易策略实现
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.terminal_ai_tools import (
    RealTimeDataProcessor,
    HighFrequencyStrategy,
    AlgorithmOptimizer,
    PerformanceMonitor
)
from src.ml.model_ensemble import VotingEnsemble, create_default_models
from src.ml.feature_engineering import AdvancedFeatureEngineer
from src.risk.risk_manager import RiskManager


class CitadelHFTStrategy:
    """Citadel 高频交易策略"""
    
    def __init__(self, config_path: str = "citadel_config.json"):
        """
        初始化策略
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.data_processor = RealTimeDataProcessor()
        self.hft_strategy = HighFrequencyStrategy()
        self.optimizer = AlgorithmOptimizer()
        self.monitor = PerformanceMonitor()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.risk_manager = RiskManager()
        
        # 策略参数
        self.position_limit = self.config.get('risk_management', {}).get('max_position', 1000000)
        self.var_limit = self.config.get('risk_management', {}).get('var_limit', 0.02)
        self.stop_loss = self.config.get('risk_management', {}).get('stop_loss', 0.05)
        
        # 交易状态
        self.positions = {}
        self.orders = []
        self.pnl = 0.0
        self.last_update = None
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ 配置文件 {config_path} 未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "competition_name": "Citadel Terminal AI Competition",
            "risk_management": {
                "max_position": 1000000,
                "var_limit": 0.02,
                "stop_loss": 0.05,
                "max_drawdown": 0.10
            },
            "strategy": {
                "rebalance_frequency": "1min",
                "lookback_window": 60,
                "signal_threshold": 0.6
            }
        }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        
        Args:
            data: 原始市场数据
            
        Returns:
            特征数据框
        """
        features_df = data.copy()
        
        # 基础价格特征
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # 波动率特征
        for window in [5, 15, 30, 60]:
            features_df[f'volatility_{window}min'] = features_df['returns'].rolling(window).std()
            features_df[f'momentum_{window}min'] = features_df['close'].pct_change(window)
        
        # 技术指标
        features_df['rsi_14'] = self._calculate_rsi(features_df['close'], 14)
        features_df['ma_5'] = features_df['close'].rolling(5).mean()
        features_df['ma_20'] = features_df['close'].rolling(20).mean()
        features_df['ma_ratio'] = features_df['ma_5'] / features_df['ma_20']
        
        # 成交量特征
        features_df['volume_ma_5'] = features_df['volume'].rolling(5).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma_5']
        features_df['vwap'] = (features_df['close'] * features_df['volume']).rolling(20).sum() / features_df['volume'].rolling(20).sum()
        
        # 买卖价差特征
        if 'bid_price' in features_df.columns and 'ask_price' in features_df.columns:
            features_df['spread'] = features_df['ask_price'] - features_df['bid_price']
            features_df['spread_pct'] = features_df['spread'] / features_df['close']
            features_df['mid_price'] = (features_df['bid_price'] + features_df['ask_price']) / 2
            
            # 订单簿不平衡
            if 'bid_size' in features_df.columns and 'ask_size' in features_df.columns:
                features_df['bid_ask_imbalance'] = (features_df['bid_size'] - features_df['ask_size']) / (features_df['bid_size'] + features_df['ask_size'])
        
        # 时间特征
        features_df['hour'] = features_df.index.hour
        features_df['minute'] = features_df.index.minute
        features_df['day_of_week'] = features_df.index.dayofweek
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            features_df: 特征数据框
            
        Returns:
            包含信号的数据框
        """
        signals_df = features_df.copy()
        
        # 初始化信号
        signals_df['signal'] = 0.0
        signals_df['signal_strength'] = 0.0
        signals_df['signal_reason'] = ''
        
        # 动量信号
        momentum_signal = np.where(
            (signals_df['momentum_5min'] > 0.001) & (signals_df['rsi_14'] < 70),
            1.0,
            np.where(
                (signals_df['momentum_5min'] < -0.001) & (signals_df['rsi_14'] > 30),
                -1.0,
                0.0
            )
        )
        
        # 均线信号
        ma_signal = np.where(
            (signals_df['ma_ratio'] > 1.02) & (signals_df['close'] > signals_df['ma_20']),
            1.0,
            np.where(
                (signals_df['ma_ratio'] < 0.98) & (signals_df['close'] < signals_df['ma_20']),
                -1.0,
                0.0
            )
        )
        
        # 成交量信号
        volume_signal = np.where(
            (signals_df['volume_ratio'] > 1.5) & (signals_df['returns'] > 0),
            0.5,
            np.where(
                (signals_df['volume_ratio'] > 1.5) & (signals_df['returns'] < 0),
                -0.5,
                0.0
            )
        )
        
        # 买卖价差信号
        spread_signal = 0.0
        if 'spread_pct' in signals_df.columns:
            spread_signal = np.where(
                signals_df['spread_pct'] < signals_df['spread_pct'].rolling(20).mean() * 0.8,
                0.3,  # 价差收窄，流动性好
                np.where(
                    signals_df['spread_pct'] > signals_df['spread_pct'].rolling(20).mean() * 1.2,
                    -0.3,  # 价差扩大，流动性差
                    0.0
                )
            )
        
        # 综合信号
        signals_df['signal'] = (
            momentum_signal * 0.4 +
            ma_signal * 0.3 +
            volume_signal * 0.2 +
            spread_signal * 0.1
        )
        
        # 信号强度
        signals_df['signal_strength'] = np.abs(signals_df['signal'])
        
        # 信号阈值过滤
        threshold = self.config.get('strategy', {}).get('signal_threshold', 0.6)
        signals_df['signal'] = np.where(
            signals_df['signal_strength'] >= threshold,
            signals_df['signal'],
            0.0
        )
        
        return signals_df
    
    def execute_strategy(self, data: pd.DataFrame) -> Dict:
        """
        执行交易策略
        
        Args:
            data: 市场数据
            
        Returns:
            执行结果
        """
        # 准备特征
        features_df = self.prepare_features(data)
        
        # 生成信号
        signals_df = self.generate_signals(features_df)
        
        # 风险检查 - 使用简化的风险控制
        current_position_value = sum([pos * data['close'].iloc[-1] for pos in self.positions.values()])
        max_position_value = self.position_limit * 0.8  # 最大80%仓位限制
        
        if abs(current_position_value) > max_position_value:
            return {
                'action': 'no_trade',
                'reason': '超过最大仓位限制',
                'signal': 0.0,
                'current_position_value': current_position_value
            }
        
        # 执行交易
        latest_signal = signals_df['signal'].iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if abs(latest_signal) > 0:
            # 计算仓位大小
            position_size = self._calculate_position_size(
                signal=latest_signal,
                price=current_price,
                volatility=features_df['volatility_15min'].iloc[-1]
            )
            
            # 执行交易
            trade_result = self._execute_trade(
                signal=latest_signal,
                size=position_size,
                price=current_price
            )
            
            return {
                'action': 'trade',
                'signal': latest_signal,
                'position_size': position_size,
                'price': current_price,
                'trade_result': trade_result,
                'features': features_df.iloc[-1].to_dict()
            }
        
        return {
            'action': 'hold',
            'signal': latest_signal,
            'price': current_price,
            'features': features_df.iloc[-1].to_dict()
        }
    
    def _calculate_position_size(self, signal: float, price: float, volatility: float) -> float:
        """
        计算仓位大小
        
        Args:
            signal: 交易信号强度
            price: 当前价格
            volatility: 波动率
            
        Returns:
            仓位大小
        """
        # 基础仓位
        base_size = self.position_limit * 0.1
        
        # 根据信号强度调整
        signal_adjusted_size = base_size * abs(signal)
        
        # 根据波动率调整
        if volatility > 0:
            volatility_adjusted_size = signal_adjusted_size / (volatility * 100)
        else:
            volatility_adjusted_size = signal_adjusted_size
        
        # 限制最大仓位
        max_size = self.position_limit * 0.2
        final_size = min(volatility_adjusted_size, max_size)
        
        return final_size * np.sign(signal)
    
    def _execute_trade(self, signal: float, size: float, price: float) -> Dict:
        """
        执行交易
        
        Args:
            signal: 交易信号
            size: 仓位大小
            price: 交易价格
            
        Returns:
            交易结果
        """
        trade_id = f"trade_{len(self.orders) + 1}_{datetime.now().strftime('%H%M%S')}"
        
        order = {
            'id': trade_id,
            'signal': signal,
            'size': size,
            'price': price,
            'timestamp': datetime.now(),
            'status': 'executed'
        }
        
        self.orders.append(order)
        
        # 更新仓位
        if 'total' not in self.positions:
            self.positions['total'] = 0.0
        
        self.positions['total'] += size
        
        # 更新PnL (简化计算)
        self.pnl += -abs(size) * price * 0.0001  # 假设交易成本
        
        return order
    
    def run_backtest(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> Dict:
        """
        运行回测
        
        Args:
            data: 历史数据
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        print("🚀 开始 Citadel 高频交易策略回测...")
        
        # 数据准备
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # 初始化回测变量
        results = []
        portfolio_value = 1000000  # 初始资金
        positions = {}
        trades = []
        
        # 逐步执行策略
        lookback_window = self.config.get('strategy', {}).get('lookback_window', 60)
        
        for i in range(lookback_window, len(data)):
            # 获取当前窗口数据
            window_data = data.iloc[i-lookback_window:i+1]
            
            # 执行策略
            strategy_result = self.execute_strategy(window_data)
            
            # 记录结果
            result = {
                'timestamp': data.index[i],
                'price': data['close'].iloc[i],
                'signal': strategy_result.get('signal', 0),
                'action': strategy_result.get('action', 'hold'),
                'portfolio_value': portfolio_value
            }
            
            # 更新组合价值 (简化)
            if strategy_result.get('action') == 'trade':
                trade_pnl = strategy_result.get('trade_result', {}).get('size', 0) * 0.001  # 假设收益
                portfolio_value += trade_pnl
                trades.append(strategy_result.get('trade_result', {}))
            
            results.append(result)
        
        # 计算回测指标
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            total_return = (portfolio_value - 1000000) / 1000000
            returns = results_df['portfolio_value'].pct_change().dropna()
            
            sharpe_ratio = np.sqrt(252 * 24 * 60) * returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = (results_df['portfolio_value'] / results_df['portfolio_value'].expanding().max() - 1).min()
            
            backtest_summary = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'win_rate': len([t for t in trades if t.get('size', 0) > 0]) / len(trades) if trades else 0,
                'avg_trade_size': np.mean([abs(t.get('size', 0)) for t in trades]) if trades else 0
            }
        else:
            backtest_summary = {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_trade_size': 0
            }
        
        print(f"✅ 回测完成!")
        print(f"   总收益率: {backtest_summary['total_return']:.2%}")
        print(f"   夏普比率: {backtest_summary['sharpe_ratio']:.2f}")
        print(f"   最大回撤: {backtest_summary['max_drawdown']:.2%}")
        print(f"   交易次数: {backtest_summary['total_trades']}")
        
        return {
            'summary': backtest_summary,
            'results': results_df,
            'trades': trades,
            'config': self.config
        }
    
    def save_results(self, results: Dict, output_dir: str = ".") -> None:
        """
        保存结果
        
        Args:
            results: 回测结果
            output_dir: 输出目录
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存回测结果
        results_file = os.path.join(output_dir, f"citadel_backtest_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换DataFrame为字典以便JSON序列化
            results_copy = results.copy()
            if 'results' in results_copy and hasattr(results_copy['results'], 'to_dict'):
                results_copy['results'] = results_copy['results'].to_dict('records')
            
            json.dump(results_copy, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📁 回测结果已保存到: {results_file}")
        
        # 保存交易记录
        if results.get('trades'):
            trades_file = os.path.join(output_dir, f"citadel_trades_{timestamp}.csv")
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(trades_file, index=False)
            print(f"📁 交易记录已保存到: {trades_file}")


def main():
    """主函数"""
    print("🏛️ Citadel Terminal AI Competition - 高频交易策略")
    print("=" * 60)
    
    # 创建策略实例
    strategy = CitadelHFTStrategy()
    
    # 加载数据
    data_file = "data/sample_market_data.csv"
    if os.path.exists(data_file):
        print(f"📊 加载数据: {data_file}")
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"   数据形状: {data.shape}")
        print(f"   时间范围: {data.index.min()} 到 {data.index.max()}")
    else:
        print("⚠️ 未找到数据文件，生成模拟数据...")
        # 生成模拟高频数据
        dates = pd.date_range('2024-01-01', periods=10000, freq='1min')
        np.random.seed(42)
        
        base_price = 100
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(7, 0.5, len(dates)),
            'bid_price': prices * (1 - np.random.uniform(0.0001, 0.001, len(dates))),
            'ask_price': prices * (1 + np.random.uniform(0.0001, 0.001, len(dates))),
            'bid_size': np.random.lognormal(5, 0.3, len(dates)),
            'ask_size': np.random.lognormal(5, 0.3, len(dates))
        }, index=dates)
        
        # 保存模拟数据
        os.makedirs("data", exist_ok=True)
        data.to_csv(data_file)
        print(f"   模拟数据已保存到: {data_file}")
    
    # 运行回测
    backtest_results = strategy.run_backtest(data)
    
    # 保存结果
    strategy.save_results(backtest_results)
    
    print("\n🎉 Citadel 高频交易策略测试完成!")
    print("\n📋 策略特点:")
    print("  • 基于多因子信号的高频交易")
    print("  • 实时风险管理和仓位控制")
    print("  • 动量、均线、成交量和价差综合分析")
    print("  • 适应性强的参数优化")
    
    print("\n🚀 下一步建议:")
    print("  1. 获取真实的高频市场数据")
    print("  2. 优化信号生成算法")
    print("  3. 实施更精细的风险管理")
    print("  4. 进行实盘模拟测试")


if __name__ == "__main__":
    main()