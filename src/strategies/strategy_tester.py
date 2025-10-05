import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .templates import (
    MeanReversionStrategy, MomentumStrategy, RSIStrategy, 
    BollingerBandsStrategy, MACDStrategy, VolatilityBreakoutStrategy
)
from ..backtest.engine import BacktestEngine
from ..performance.analyzer import PerformanceAnalyzer


class StrategyTester:
    """
    策略测试器：批量测试多种策略并进行性能对比分析
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        test_period_ratio: float = 0.3  # 测试期占总数据的比例
    ):
        """
        初始化策略测试器
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点成本
            test_period_ratio: 测试期比例
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.test_period_ratio = test_period_ratio
        
        # 预定义策略配置
        self.strategy_configs = {
            'MeanReversion_Conservative': {
                'class': MeanReversionStrategy,
                'params': {'lookback': 20, 'entry_z': -1.5, 'exit_z': 0.0}
            },
            'MeanReversion_Aggressive': {
                'class': MeanReversionStrategy,
                'params': {'lookback': 10, 'entry_z': -1.0, 'exit_z': 0.5}
            },
            'Momentum_Fast': {
                'class': MomentumStrategy,
                'params': {'fast': 5, 'slow': 15}
            },
            'Momentum_Standard': {
                'class': MomentumStrategy,
                'params': {'fast': 12, 'slow': 26}
            },
            'Momentum_Slow': {
                'class': MomentumStrategy,
                'params': {'fast': 20, 'slow': 50}
            },
            'RSI_Oversold': {
                'class': RSIStrategy,
                'params': {'period': 14, 'oversold': 30, 'overbought': 70}
            },
            'RSI_Extreme': {
                'class': RSIStrategy,
                'params': {'period': 14, 'oversold': 20, 'overbought': 80}
            },
            'Bollinger_MeanReversion': {
                'class': BollingerBandsStrategy,
                'params': {'period': 20, 'std_dev': 2.0, 'strategy_type': 'mean_reversion'}
            },
            'Bollinger_Breakout': {
                'class': BollingerBandsStrategy,
                'params': {'period': 20, 'std_dev': 2.0, 'strategy_type': 'breakout'}
            },
            'MACD_Standard': {
                'class': MACDStrategy,
                'params': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            'MACD_Fast': {
                'class': MACDStrategy,
                'params': {'fast': 8, 'slow': 17, 'signal': 6}
            },
            'VolBreakout_Conservative': {
                'class': VolatilityBreakoutStrategy,
                'params': {'lookback': 20, 'multiplier': 2.0}
            },
            'VolBreakout_Aggressive': {
                'class': VolatilityBreakoutStrategy,
                'params': {'lookback': 10, 'multiplier': 1.5}
            }
        }
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分割训练集和测试集
        
        Args:
            data: 完整数据集
            
        Returns:
            训练集和测试集
        """
        split_point = int(len(data) * (1 - self.test_period_ratio))
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        return train_data, test_data
    
    def test_single_strategy(
        self, 
        strategy_name: str, 
        data: pd.DataFrame,
        custom_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        测试单个策略
        
        Args:
            strategy_name: 策略名称
            data: 测试数据
            custom_params: 自定义参数
            
        Returns:
            测试结果
        """
        if strategy_name not in self.strategy_configs:
            raise ValueError(f"未知策略: {strategy_name}")
        
        config = self.strategy_configs[strategy_name].copy()
        if custom_params:
            config['params'].update(custom_params)
        
        # 创建策略实例
        strategy = config['class'](**config['params'])
        
        # 生成信号
        if hasattr(strategy, 'generate_signal'):
            signals = strategy.generate_signal(data)
        elif hasattr(strategy, 'signal'):
            signals = strategy.signal(data)
        else:
            raise AttributeError(f"策略 {strategy.__class__.__name__} 没有 signal 或 generate_signal 方法")
        
        # 创建回测引擎
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage
        )
        
        # 运行回测
        backtest_result = engine.run(data, signals)
        
        # 性能分析
        analyzer = PerformanceAnalyzer()
        performance_metrics = analyzer.metrics(
            backtest_result['returns']
        )
        
        return {
            'strategy_name': strategy_name,
            'strategy_params': config['params'],
            'backtest_result': backtest_result,
            'performance_metrics': performance_metrics,
            'signals': signals
        }
    
    def test_all_strategies(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        测试所有预定义策略
        
        Args:
            data: 测试数据
            
        Returns:
            所有策略的测试结果
        """
        results = {}
        
        print(f"开始测试 {len(self.strategy_configs)} 个策略...")
        
        for i, strategy_name in enumerate(self.strategy_configs.keys(), 1):
            try:
                print(f"[{i}/{len(self.strategy_configs)}] 测试策略: {strategy_name}")
                result = self.test_single_strategy(strategy_name, data)
                results[strategy_name] = result
                
                # 打印简要结果
                metrics = result['performance_metrics']
                print(f"  总收益: {metrics.get('total_return', 0):.2f}%")
                print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
                print()
                
            except Exception as e:
                print(f"  策略 {strategy_name} 测试失败: {e}")
                continue
        
        return results
    
    def compare_strategies(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        比较策略性能
        
        Args:
            results: 策略测试结果
            
        Returns:
            性能对比表
        """
        comparison_data = []
        
        for strategy_name, result in results.items():
            metrics = result['performance_metrics']
            backtest = result['backtest_result']
            
            comparison_data.append({
                '策略名称': strategy_name,
                '总收益率(%)': metrics.get('total_return', 0),
                '年化收益率(%)': metrics.get('annualized_return', 0),
                '年化波动率(%)': metrics.get('annualized_volatility', 0),
                '夏普比率': metrics.get('sharpe_ratio', 0),
                '索提诺比率': metrics.get('sortino_ratio', 0),
                '最大回撤(%)': metrics.get('max_drawdown', 0),
                '卡尔玛比率': metrics.get('calmar_ratio', 0),
                '胜率(%)': metrics.get('win_rate', 0),
                '盈亏比': metrics.get('profit_loss_ratio', 0),
                '换手率': backtest.get('turnover', 0),
                '交易次数': backtest.get('total_trades', 0),
                '最终价值': backtest.get('final_value', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 按夏普比率排序
        df = df.sort_values('夏普比率', ascending=False)
        
        return df
    
    def plot_strategy_comparison(
        self, 
        results: Dict[str, Dict[str, Any]], 
        save_path: str = None
    ) -> None:
        """
        绘制策略对比图表
        
        Args:
            results: 策略测试结果
            save_path: 保存路径
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('策略性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 净值曲线对比
        ax1 = axes[0, 0]
        for strategy_name, result in results.items():
            portfolio_value = result['backtest_result']['portfolio_value']
            normalized_value = portfolio_value / portfolio_value.iloc[0]
            ax1.plot(normalized_value.index, normalized_value.values, 
                    label=strategy_name, linewidth=1.5, alpha=0.8)
        
        ax1.set_title('净值曲线对比', fontweight='bold')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('净值（标准化）')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益率vs风险散点图
        ax2 = axes[0, 1]
        returns = []
        volatilities = []
        names = []
        
        for strategy_name, result in results.items():
            metrics = result['performance_metrics']
            returns.append(metrics.get('annualized_return', 0))
            volatilities.append(metrics.get('annualized_volatility', 0))
            names.append(strategy_name)
        
        scatter = ax2.scatter(volatilities, returns, s=100, alpha=0.7, c=range(len(names)), cmap='viridis')
        
        for i, name in enumerate(names):
            ax2.annotate(name, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_title('收益率 vs 风险', fontweight='bold')
        ax2.set_xlabel('年化波动率 (%)')
        ax2.set_ylabel('年化收益率 (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 夏普比率对比
        ax3 = axes[1, 0]
        sharpe_ratios = [result['performance_metrics'].get('sharpe_ratio', 0) 
                        for result in results.values()]
        strategy_names = list(results.keys())
        
        bars = ax3.bar(range(len(strategy_names)), sharpe_ratios, alpha=0.7)
        ax3.set_title('夏普比率对比', fontweight='bold')
        ax3.set_xlabel('策略')
        ax3.set_ylabel('夏普比率')
        ax3.set_xticks(range(len(strategy_names)))
        ax3.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 为正值和负值使用不同颜色
        for i, bar in enumerate(bars):
            if sharpe_ratios[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # 4. 最大回撤对比
        ax4 = axes[1, 1]
        max_drawdowns = [abs(result['performance_metrics'].get('max_drawdown', 0)) 
                        for result in results.values()]
        
        bars = ax4.bar(range(len(strategy_names)), max_drawdowns, alpha=0.7, color='red')
        ax4.set_title('最大回撤对比', fontweight='bold')
        ax4.set_xlabel('策略')
        ax4.set_ylabel('最大回撤 (%)')
        ax4.set_xticks(range(len(strategy_names)))
        ax4.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def generate_report(
        self, 
        results: Dict[str, Dict[str, Any]], 
        data: pd.DataFrame,
        save_path: str = None
    ) -> str:
        """
        生成策略测试报告
        
        Args:
            results: 策略测试结果
            data: 测试数据
            save_path: 报告保存路径
            
        Returns:
            报告内容
        """
        report = []
        report.append("=" * 80)
        report.append("量化策略测试报告")
        report.append("=" * 80)
        report.append(f"测试期间: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}")
        report.append(f"测试天数: {len(data)} 天")
        report.append(f"初始资金: {self.initial_capital:,.0f}")
        report.append(f"手续费率: {self.commission:.3f}")
        report.append(f"滑点成本: {self.slippage:.4f}")
        report.append("")
        
        # 策略性能排名
        comparison_df = self.compare_strategies(results)
        report.append("策略性能排名（按夏普比率）:")
        report.append("-" * 50)
        
        for i, row in comparison_df.head(10).iterrows():
            report.append(f"{row['策略名称']:25} | 夏普比率: {row['夏普比率']:6.3f} | "
                         f"总收益: {row['总收益率(%)']:7.2f}% | 最大回撤: {row['最大回撤(%)']:6.2f}%")
        
        report.append("")
        
        # 最佳策略详细分析
        best_strategy_name = comparison_df.iloc[0]['策略名称']
        best_result = results[best_strategy_name]
        
        report.append(f"最佳策略详细分析: {best_strategy_name}")
        report.append("-" * 50)
        report.append(f"策略参数: {best_result['strategy_params']}")
        
        metrics = best_result['performance_metrics']
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'ratio' in key.lower() or 'rate' in key.lower():
                    report.append(f"{key:20}: {value:8.3f}")
                elif '%' in str(value) or 'return' in key.lower() or 'drawdown' in key.lower():
                    report.append(f"{key:20}: {value:8.2f}%")
                else:
                    report.append(f"{key:20}: {value:8.2f}")
        
        report.append("")
        
        # 策略类型分析
        report.append("策略类型分析:")
        report.append("-" * 30)
        
        strategy_types = {}
        for strategy_name, result in results.items():
            strategy_type = strategy_name.split('_')[0]
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = []
            strategy_types[strategy_type].append(result['performance_metrics'].get('sharpe_ratio', 0))
        
        for strategy_type, sharpe_ratios in strategy_types.items():
            avg_sharpe = np.mean(sharpe_ratios)
            report.append(f"{strategy_type:15}: 平均夏普比率 {avg_sharpe:6.3f} ({len(sharpe_ratios)} 个策略)")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {save_path}")
        
        return report_text
    
    def run_comprehensive_test(
        self, 
        data: pd.DataFrame,
        save_dir: str = None
    ) -> Dict[str, Any]:
        """
        运行综合测试
        
        Args:
            data: 测试数据
            save_dir: 保存目录
            
        Returns:
            综合测试结果
        """
        print("开始综合策略测试...")
        print(f"数据期间: {data.index[0]} 至 {data.index[-1]}")
        print(f"数据点数: {len(data)}")
        print()
        
        # 分割数据
        train_data, test_data = self.split_data(data)
        print(f"训练期: {train_data.index[0]} 至 {train_data.index[-1]} ({len(train_data)} 天)")
        print(f"测试期: {test_data.index[0]} 至 {test_data.index[-1]} ({len(test_data)} 天)")
        print()
        
        # 测试所有策略
        results = self.test_all_strategies(test_data)
        
        if not results:
            print("没有成功测试的策略!")
            return {}
        
        # 生成对比表
        comparison_df = self.compare_strategies(results)
        print("策略性能对比:")
        print(comparison_df.to_string(index=False))
        print()
        
        # 生成图表
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            chart_path = os.path.join(save_dir, "strategy_comparison.png")
            self.plot_strategy_comparison(results, chart_path)
            
            # 保存对比表
            csv_path = os.path.join(save_dir, "strategy_comparison.csv")
            comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"对比表已保存到: {csv_path}")
            
            # 生成报告
            report_path = os.path.join(save_dir, "strategy_test_report.txt")
            report = self.generate_report(results, test_data, report_path)
        else:
            self.plot_strategy_comparison(results)
            report = self.generate_report(results, test_data)
        
        return {
            'results': results,
            'comparison_df': comparison_df,
            'report': report,
            'train_data': train_data,
            'test_data': test_data
        }