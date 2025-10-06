#!/usr/bin/env python3
"""
Citadel 策略参数网格搜索优化
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import logging
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.ml.terminal_ai_tools import run_terminal_ai_simulation
from src.ml.feature_engineering import AdvancedFeatureEngineer
from src.risk.risk_manager import RiskManager

class CitadelGridSearchOptimizer:
    """Citadel策略网格搜索优化器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        
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
    
    def define_parameter_grid(self) -> Dict[str, List]:
        """定义参数网格"""
        param_grid = {
            'lookback_period': [10, 15, 20],
            'signal_threshold': [0.2, 0.3, 0.4],
            'position_limit': [0.1, 0.15, 0.2],
            'max_trade_size': [5000, 8000, 10000],
            'stop_loss': [0.005, 0.008, 0.012],
            'take_profit': [0.008, 0.012, 0.015],
            'rsi_period': [8, 10, 14],
            'bb_period': [12, 15, 20],
            'momentum_weight': [0.2, 0.25, 0.3],
            'mean_reversion_weight': [0.2, 0.25, 0.3]
        }
        return param_grid
    
    def create_config(self, params: Dict) -> Dict:
        """根据参数创建配置"""
        # 确保权重总和为1
        total_weight = params['momentum_weight'] + params['mean_reversion_weight']
        remaining_weight = 1.0 - total_weight
        
        config = {
            "strategy_name": "GridSearchCitadelHFT",
            "version": "1.0",
            "description": "网格搜索优化的Citadel高频交易策略",
            "signal_parameters": {
                "lookback_period": params['lookback_period'],
                "signal_threshold": params['signal_threshold'],
                "position_limit": params['position_limit'],
                "max_trade_size": params['max_trade_size']
            },
            "risk_management": {
                "stop_loss": params['stop_loss'],
                "take_profit": params['take_profit'],
                "max_portfolio_risk": 0.02,
                "max_single_position": params['position_limit']
            },
            "technical_indicators": {
                "rsi_period": params['rsi_period'],
                "bb_period": params['bb_period'],
                "bb_std_multiplier": 2,
                "macd_fast": 8,
                "macd_slow": 17,
                "macd_signal": 6,
                "volatility_window": 10,
                "volume_window": 10
            },
            "signal_weights": {
                "momentum": params['momentum_weight'],
                "mean_reversion": params['mean_reversion_weight'],
                "volatility": remaining_weight * 0.4,
                "volume": remaining_weight * 0.35,
                "microstructure": remaining_weight * 0.25
            },
            "market_conditions": {
                "min_volume_threshold": 1000,
                "max_spread_threshold": 0.01,
                "volatility_filter": True,
                "market_hours_only": False
            },
            "optimization_settings": {
                "adaptive_thresholds": True,
                "dynamic_position_sizing": True,
                "regime_detection": False,
                "correlation_filter": False
            }
        }
        return config
    
    def evaluate_parameters(self, params: Dict) -> Dict:
        """评估参数组合"""
        try:
            # 创建配置
            config = self.create_config(params)
            
            # 运行回测（这里简化为模拟）
            # 在实际应用中，这里应该调用完整的回测系统
            results = self._simulate_backtest(config)
            
            # 计算综合评分
            score = self._calculate_score(results)
            
            return {
                'params': params,
                'config': config,
                'results': results,
                'score': score
            }
            
        except Exception as e:
            self.logger.error(f"参数评估失败: {e}")
            return {
                'params': params,
                'config': None,
                'results': None,
                'score': -np.inf,
                'error': str(e)
            }
    
    def _simulate_backtest(self, config: Dict) -> Dict:
        """模拟回测（简化版本）"""
        # 这是一个简化的模拟，实际应用中需要完整的回测
        np.random.seed(42)
        
        # 基于参数生成模拟结果
        signal_threshold = config['signal_parameters']['signal_threshold']
        position_limit = config['signal_parameters']['position_limit']
        stop_loss = config['risk_management']['stop_loss']
        take_profit = config['risk_management']['take_profit']
        
        # 模拟交易次数（基于信号阈值）
        base_trades = 50
        trade_multiplier = 1 / signal_threshold  # 阈值越低，交易越多
        num_trades = int(base_trades * trade_multiplier)
        
        # 模拟收益率
        win_rate = 0.45 + (take_profit - stop_loss) * 10  # 止盈止损比影响胜率
        win_rate = np.clip(win_rate, 0.3, 0.7)
        
        # 生成交易结果
        trades = np.random.choice([1, -1], size=num_trades, p=[win_rate, 1-win_rate])
        
        # 计算收益
        avg_win = take_profit * 0.8  # 平均盈利
        avg_loss = -stop_loss * 0.9  # 平均亏损
        
        returns = []
        for trade in trades:
            if trade == 1:
                returns.append(avg_win)
            else:
                returns.append(avg_loss)
        
        total_return = sum(returns)
        
        # 计算其他指标
        if len(returns) > 0:
            returns_array = np.array(returns)
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
            max_drawdown = abs(min(np.cumsum(returns_array)))
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade_return': total_return / max(num_trades, 1)
        }
    
    def _calculate_score(self, results: Dict) -> float:
        """计算综合评分"""
        if results is None:
            return -np.inf
        
        # 权重设置
        weights = {
            'return': 0.3,
            'sharpe': 0.3,
            'drawdown': 0.2,
            'trades': 0.1,
            'win_rate': 0.1
        }
        
        # 标准化指标
        return_score = results['total_return'] * 100  # 收益率
        sharpe_score = results['sharpe_ratio']  # 夏普比率
        drawdown_score = -results['max_drawdown'] * 100  # 最大回撤（负值）
        trades_score = min(results['num_trades'] / 100, 1)  # 交易次数（标准化）
        win_rate_score = results['win_rate'] * 100  # 胜率
        
        # 综合评分
        score = (
            weights['return'] * return_score +
            weights['sharpe'] * sharpe_score +
            weights['drawdown'] * drawdown_score +
            weights['trades'] * trades_score +
            weights['win_rate'] * win_rate_score
        )
        
        return score
    
    def run_grid_search(self, max_combinations: int = 50) -> Dict:
        """运行网格搜索"""
        self.logger.info("🔍 开始Citadel策略参数网格搜索优化...")
        
        # 定义参数网格
        param_grid = self.define_parameter_grid()
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))
        
        # 限制组合数量
        if len(all_combinations) > max_combinations:
            self.logger.info(f"参数组合过多({len(all_combinations)})，随机选择{max_combinations}个进行测试")
            np.random.seed(42)
            selected_indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            selected_combinations = [all_combinations[i] for i in selected_indices]
        else:
            selected_combinations = all_combinations
        
        self.logger.info(f"总共测试 {len(selected_combinations)} 个参数组合")
        
        # 测试每个参数组合
        for i, combination in enumerate(selected_combinations):
            params = dict(zip(param_names, combination))
            
            self.logger.info(f"测试组合 {i+1}/{len(selected_combinations)}: {params}")
            
            result = self.evaluate_parameters(params)
            self.results.append(result)
            
            # 更新最佳参数
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_params = params
                self.logger.info(f"发现更好的参数组合，评分: {self.best_score:.4f}")
        
        # 保存结果
        self._save_results()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def _save_results(self):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = f"competitions/citadel/citadel_grid_search_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'all_results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        # 保存最佳配置
        if self.best_params:
            best_config = self.create_config(self.best_params)
            config_file = f"competitions/citadel/citadel_optimized_config_{timestamp}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"优化结果已保存到: {results_file}")
        self.logger.info(f"最佳配置已保存到: {config_file}")
    
    def generate_report(self) -> str:
        """生成优化报告"""
        if not self.results:
            return "没有优化结果可报告"
        
        # 排序结果
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        top_5 = sorted_results[:5]
        
        report = []
        report.append("🎯 Citadel策略网格搜索优化报告")
        report.append("=" * 60)
        report.append(f"测试参数组合数量: {len(self.results)}")
        report.append(f"最佳评分: {self.best_score:.4f}")
        report.append("")
        
        report.append("🏆 最佳参数组合:")
        report.append("-" * 40)
        if self.best_params:
            for key, value in self.best_params.items():
                report.append(f"{key}: {value}")
        report.append("")
        
        report.append("📊 前5名参数组合:")
        report.append("-" * 40)
        for i, result in enumerate(top_5):
            if result['results']:
                report.append(f"第{i+1}名 (评分: {result['score']:.4f}):")
                report.append(f"  收益率: {result['results']['total_return']:.4f}")
                report.append(f"  夏普比率: {result['results']['sharpe_ratio']:.4f}")
                report.append(f"  最大回撤: {result['results']['max_drawdown']:.4f}")
                report.append(f"  交易次数: {result['results']['num_trades']}")
                report.append(f"  胜率: {result['results']['win_rate']:.4f}")
                report.append("")
        
        return "\n".join(report)

def main():
    """主函数"""
    print("🔍 Citadel策略参数网格搜索优化")
    print("=" * 60)
    
    # 创建优化器
    optimizer = CitadelGridSearchOptimizer()
    
    # 运行网格搜索
    results = optimizer.run_grid_search(max_combinations=30)
    
    # 生成报告
    report = optimizer.generate_report()
    print(report)
    
    print("\n🎉 网格搜索优化完成!")

if __name__ == "__main__":
    main()