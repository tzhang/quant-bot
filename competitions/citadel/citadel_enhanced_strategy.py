#!/usr/bin/env python3
"""
Citadel Terminal AI Competition - 增强版高频交易策略

基于优化分析结果改进的策略版本
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入必要的模块
from src.ml.terminal_ai_tools import (
    RealTimeDataProcessor, 
    HighFrequencyStrategy, 
    AlgorithmOptimizer,
    PerformanceMonitor
)
from src.ml.feature_engineering import AdvancedFeatureEngineer
from src.ml.model_ensemble import VotingEnsemble, create_default_models
from src.risk.risk_manager import RiskManager

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCitadelHFTStrategy:
    """增强版 Citadel 高频交易策略"""
    
    def __init__(self, config_file: str = "citadel_enhanced_config.json"):
        """
        初始化增强版策略
        
        Args:
            config_file: 配置文件路径
        """
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config = self.load_config(config_file)
        
        # 初始化组件
        self.data_processor = RealTimeDataProcessor()
        self.hft_strategy = HighFrequencyStrategy()
        self.optimizer = AlgorithmOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.risk_manager = RiskManager()
        
        # 策略参数 (基于优化建议调整)
        self.lookback_period = self.config.get('lookback_period', 30)  # 增加到30
        self.signal_threshold = self.config.get('signal_threshold', 0.7)  # 提高到0.7
        self.position_limit = self.config.get('position_limit', 0.05)  # 降低到5%
        self.stop_loss = self.config.get('stop_loss', 0.02)  # 添加2%止损
        self.take_profit = self.config.get('take_profit', 0.04)  # 添加4%止盈
        self.max_trade_size = self.config.get('max_trade_size', 3000)  # 降低交易规模
        
        # 增强的技术指标参数
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        
        # 状态变量
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        # 创建模型集成
        base_models = create_default_models()
        self.ensemble = VotingEnsemble(base_models)
        
        self.logger.info("✅ 增强版 Citadel HFT 策略初始化完成")
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'lookback_period': 30,
            'signal_threshold': 0.7,
            'position_limit': 0.05,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_trade_size': 3000,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'min_volume_threshold': 1000,
            'volatility_threshold': 0.02,
            'correlation_threshold': 0.8
        }
        
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"✅ 配置文件加载成功: {config_file}")
                return {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"⚠️ 配置文件加载失败: {e}，使用默认配置")
        else:
            self.logger.warning(f"⚠️ 配置文件 {config_file} 未找到，使用默认配置")
        
        return default_config
    
    def prepare_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备增强的特征"""
        features_df = data.copy()
        
        # 基础价格特征
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['price_momentum'] = features_df['close'] / features_df['close'].shift(self.lookback_period) - 1
        
        # 增强的移动平均线
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = features_df['close'].rolling(period).mean()
            features_df[f'ema_{period}'] = features_df['close'].ewm(span=period).mean()
            features_df[f'price_to_sma_{period}'] = features_df['close'] / features_df[f'sma_{period}'] - 1
        
        # 技术指标
        features_df['rsi'] = self.data_processor._calculate_rsi(features_df['close'], self.rsi_period)
        
        # MACD
        ema_fast = features_df['close'].ewm(span=self.macd_fast).mean()
        ema_slow = features_df['close'].ewm(span=self.macd_slow).mean()
        features_df['macd'] = ema_fast - ema_slow
        features_df['macd_signal'] = features_df['macd'].ewm(span=self.macd_signal).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # 布林带
        sma_bb = features_df['close'].rolling(self.bb_period).mean()
        std_bb = features_df['close'].rolling(self.bb_period).std()
        features_df['bb_upper'] = sma_bb + (std_bb * self.bb_std)
        features_df['bb_lower'] = sma_bb - (std_bb * self.bb_std)
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # 波动率特征
        features_df['volatility'] = features_df['returns'].rolling(self.lookback_period).std()
        features_df['volatility_ratio'] = features_df['volatility'] / features_df['volatility'].rolling(50).mean()
        
        # 成交量特征
        if 'volume' in features_df.columns:
            features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            features_df['price_volume'] = features_df['returns'] * features_df['volume_ratio']
        
        # 价差特征
        if all(col in features_df.columns for col in ['bid', 'ask']):
            features_df['spread'] = features_df['ask'] - features_df['bid']
            features_df['spread_pct'] = features_df['spread'] / features_df['close']
            features_df['mid_price'] = (features_df['bid'] + features_df['ask']) / 2
            features_df['price_to_mid'] = features_df['close'] / features_df['mid_price'] - 1
        
        # 时间特征
        if 'timestamp' in features_df.columns:
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            features_df['hour'] = features_df['timestamp'].dt.hour
            features_df['minute'] = features_df['timestamp'].dt.minute
            features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
            
            # 市场开盘时间特征
            features_df['is_market_open'] = (features_df['hour'] >= 9) & (features_df['hour'] < 16)
            features_df['time_to_close'] = 16 - features_df['hour']
        
        # 高阶特征
        features_df['price_acceleration'] = features_df['returns'].diff()
        features_df['volume_acceleration'] = features_df['volume'].pct_change().diff() if 'volume' in features_df.columns else 0
        
        # 市场微观结构特征
        features_df['price_impact'] = features_df['returns'] / (features_df['volume'] + 1e-8) if 'volume' in features_df.columns else 0
        features_df['liquidity_proxy'] = 1 / (features_df['spread_pct'] + 1e-8) if 'spread_pct' in features_df.columns else 1
        
        return features_df.fillna(0)
    
    def generate_enhanced_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """生成增强的交易信号"""
        signals_df = features_df.copy()
        
        # 多因子信号组合
        signals = []
        
        # 1. 动量信号 (权重: 25%)
        momentum_signal = 0
        if 'price_momentum' in features_df.columns:
            momentum_signal += np.tanh(features_df['price_momentum'] * 10) * 0.3
        if 'macd_histogram' in features_df.columns:
            momentum_signal += np.tanh(features_df['macd_histogram'] * 100) * 0.7
        signals.append(momentum_signal * 0.25)
        
        # 2. 均线信号 (权重: 20%)
        ma_signal = 0
        for period in [5, 10, 20]:
            if f'price_to_sma_{period}' in features_df.columns:
                ma_signal += np.tanh(features_df[f'price_to_sma_{period}'] * 20) / 3
        signals.append(ma_signal * 0.20)
        
        # 3. RSI信号 (权重: 15%)
        rsi_signal = 0
        if 'rsi' in features_df.columns:
            # RSI超买超卖信号
            rsi_normalized = (features_df['rsi'] - 50) / 50
            rsi_signal = -np.tanh(rsi_normalized * 2)  # 反转信号
        signals.append(rsi_signal * 0.15)
        
        # 4. 布林带信号 (权重: 15%)
        bb_signal = 0
        if 'bb_position' in features_df.columns:
            # 布林带位置信号
            bb_signal = np.tanh((features_df['bb_position'] - 0.5) * 4)
        signals.append(bb_signal * 0.15)
        
        # 5. 成交量信号 (权重: 10%)
        volume_signal = 0
        if 'price_volume' in features_df.columns:
            volume_signal = np.tanh(features_df['price_volume'] * 5)
        signals.append(volume_signal * 0.10)
        
        # 6. 波动率信号 (权重: 10%)
        vol_signal = 0
        if 'volatility_ratio' in features_df.columns:
            # 高波动率时减少信号强度
            vol_signal = -np.tanh((features_df['volatility_ratio'] - 1) * 2) * 0.5
        signals.append(vol_signal * 0.10)
        
        # 7. 微观结构信号 (权重: 5%)
        microstructure_signal = 0
        if 'liquidity_proxy' in features_df.columns:
            microstructure_signal = np.tanh(features_df['liquidity_proxy'] / 1000)
        signals.append(microstructure_signal * 0.05)
        
        # 合并所有信号
        combined_signal = sum(signals)
        
        # 应用信号阈值过滤
        filtered_signal = np.where(
            np.abs(combined_signal) > self.signal_threshold,
            combined_signal,
            0
        )
        
        signals_df['signal'] = filtered_signal
        signals_df['signal_strength'] = np.abs(combined_signal)
        
        return signals_df
    
    def execute_enhanced_strategy(self, data: pd.DataFrame, portfolio_value: float = 1000000) -> Dict[str, Any]:
        """执行增强版交易策略"""
        # 准备特征
        features_df = self.prepare_enhanced_features(data)
        
        # 生成信号
        signals_df = self.generate_enhanced_signals(features_df)
        
        current_price = data['close'].iloc[-1]
        current_signal = signals_df['signal'].iloc[-1]
        
        # 增强的风险检查
        current_position_value = sum([pos * current_price for pos in self.positions.values()])
        max_position_value = portfolio_value * self.position_limit
        
        # 止损止盈检查
        if self.positions:
            for symbol, position in self.positions.items():
                if position != 0:
                    # 计算当前盈亏
                    entry_price = getattr(self, f'entry_price_{symbol}', current_price)
                    pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)
                    
                    # 止损
                    if pnl_pct < -self.stop_loss:
                        return {
                            'action': 'close_position',
                            'reason': f'止损触发: {pnl_pct:.2%}',
                            'signal': 0.0,
                            'pnl': pnl_pct
                        }
                    
                    # 止盈
                    if pnl_pct > self.take_profit:
                        return {
                            'action': 'close_position',
                            'reason': f'止盈触发: {pnl_pct:.2%}',
                            'signal': 0.0,
                            'pnl': pnl_pct
                        }
        
        # 仓位限制检查
        if abs(current_position_value) > max_position_value:
            return {
                'action': 'no_trade',
                'reason': '超过最大仓位限制',
                'signal': current_signal,
                'current_position_value': current_position_value
            }
        
        # 计算交易规模
        if abs(current_signal) > 0:
            signal_strength = signals_df['signal_strength'].iloc[-1]
            base_size = min(self.max_trade_size, portfolio_value * 0.01)  # 最大1%
            position_size = int(base_size * signal_strength / current_price)
            
            if current_signal > 0:
                action = 'buy'
            else:
                action = 'sell'
                position_size = -position_size
            
            return {
                'action': action,
                'quantity': position_size,
                'price': current_price,
                'signal': current_signal,
                'signal_strength': signal_strength,
                'reason': f'信号强度: {signal_strength:.3f}'
            }
        
        return {
            'action': 'hold',
            'signal': current_signal,
            'reason': '信号强度不足'
        }
    
    def run_enhanced_backtest(self, data: pd.DataFrame, initial_capital: float = 1000000) -> Dict[str, Any]:
        """运行增强版回测"""
        self.logger.info("🚀 开始增强版回测...")
        
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {}
        trades = []
        portfolio_history = []
        
        for i in range(self.lookback_period, len(data)):
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            timestamp = current_data.index[-1] if hasattr(current_data.index[-1], 'strftime') else str(i)
            
            # 执行策略
            decision = self.execute_enhanced_strategy(current_data, portfolio_value)
            
            # 执行交易
            if decision['action'] in ['buy', 'sell']:
                quantity = decision['quantity']
                trade_value = quantity * current_price
                
                if decision['action'] == 'buy' and cash >= abs(trade_value):
                    cash -= abs(trade_value)
                    positions['stock'] = positions.get('stock', 0) + quantity
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'buy',
                        'quantity': quantity,
                        'price': current_price,
                        'value': trade_value,
                        'signal': decision['signal']
                    })
                
                elif decision['action'] == 'sell' and positions.get('stock', 0) >= abs(quantity):
                    cash += abs(trade_value)
                    positions['stock'] = positions.get('stock', 0) + quantity  # quantity is negative
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'quantity': quantity,
                        'price': current_price,
                        'value': trade_value,
                        'signal': decision['signal']
                    })
            
            # 计算组合价值
            stock_value = positions.get('stock', 0) * current_price
            portfolio_value = cash + stock_value
            
            portfolio_history.append({
                'timestamp': timestamp,
                'price': current_price,
                'signal': decision.get('signal', 0),
                'action': decision['action'],
                'portfolio_value': portfolio_value,
                'cash': cash,
                'stock_position': positions.get('stock', 0),
                'stock_value': stock_value
            })
        
        # 计算性能指标
        portfolio_df = pd.DataFrame(portfolio_history)
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].cummax() - 1).min()
        
        win_trades = [t for t in trades if (t['action'] == 'sell' and t['quantity'] < 0) or (t['action'] == 'buy' and t['quantity'] > 0)]
        win_rate = len([t for t in win_trades if t['signal'] * t['quantity'] > 0]) / len(win_trades) if win_trades else 0
        
        results = {
            'summary': {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'final_portfolio_value': portfolio_value,
                'final_cash': cash,
                'final_positions': positions
            },
            'results': portfolio_history,
            'trades': trades
        }
        
        self.logger.info("✅ 增强版回测完成!")
        return results
    
    def save_enhanced_results(self, results: Dict[str, Any], prefix: str = "citadel_enhanced") -> Tuple[str, str]:
        """保存增强版结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存回测结果
        results_file = f"{prefix}_backtest_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存交易记录
        trades_file = f"{prefix}_trades_{timestamp}.csv"
        if results.get('trades'):
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(trades_file, index=False)
        
        return results_file, trades_file

def main():
    """主函数"""
    print("🏛️ Citadel Terminal AI Competition - 增强版高频交易策略")
    print("=" * 70)
    
    # 创建增强版策略
    strategy = EnhancedCitadelHFTStrategy()
    
    # 加载数据
    data_file = "competitions/citadel/data/sample_market_data.csv"
    if not Path(data_file).exists():
        logger.error(f"❌ 数据文件未找到: {data_file}")
        return
    
    data = pd.read_csv(data_file)
    logger.info(f"📊 加载数据: {data_file}")
    logger.info(f"   数据形状: {data.shape}")
    logger.info(f"   时间范围: {data['timestamp'].iloc[0]} 到 {data['timestamp'].iloc[-1]}")
    
    # 运行回测
    results = strategy.run_enhanced_backtest(data)
    
    # 保存结果
    results_file, trades_file = strategy.save_enhanced_results(results)
    
    # 打印结果
    summary = results['summary']
    print(f"\n✅ 增强版回测完成!")
    print(f"   总收益率: {summary['total_return']:.2%}")
    print(f"   夏普比率: {summary['sharpe_ratio']:.2f}")
    print(f"   最大回撤: {summary['max_drawdown']:.2%}")
    print(f"   交易次数: {summary['total_trades']}")
    print(f"   胜率: {summary['win_rate']:.2%}")
    print(f"📁 回测结果已保存到: {results_file}")
    print(f"📁 交易记录已保存到: {trades_file}")
    
    print(f"\n🎉 增强版 Citadel 高频交易策略测试完成!")
    
    print(f"\n📋 增强版策略特点:")
    print(f"  • 提高信号阈值到 0.7，减少噪音交易")
    print(f"  • 增加历史窗口到 30 期，提高信号稳定性")
    print(f"  • 添加止损 (2%) 和止盈 (4%) 机制")
    print(f"  • 降低仓位限制到 5%，控制风险")
    print(f"  • 增强技术指标：MACD、布林带、RSI")
    print(f"  • 多因子信号加权组合")
    print(f"  • 微观结构特征集成")

if __name__ == "__main__":
    main()