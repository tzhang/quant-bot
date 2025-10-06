#!/usr/bin/env python3
"""
Citadel Terminal AI Competition - 策略优化脚本

分析回测结果并优化策略参数
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CitadelOptimizer:
    """Citadel 策略优化器"""
    
    def __init__(self, results_file: str, trades_file: str):
        """
        初始化优化器
        
        Args:
            results_file: 回测结果文件路径
            trades_file: 交易记录文件路径
        """
        self.results_file = results_file
        self.trades_file = trades_file
        self.results_data = None
        self.trades_data = None
        self.analysis_results = {}
        
        self.load_data()
    
    def load_data(self):
        """加载回测数据"""
        try:
            # 加载回测结果
            with open(self.results_file, 'r') as f:
                self.results_data = json.load(f)
            
            # 加载交易记录
            if Path(self.trades_file).exists():
                self.trades_data = pd.read_csv(self.trades_file)
                if 'timestamp' in self.trades_data.columns:
                    self.trades_data['timestamp'] = pd.to_datetime(self.trades_data['timestamp'])
            
            logger.info(f"✅ 数据加载完成")
            logger.info(f"   回测记录数: {len(self.results_data.get('results', []))}")
            if self.trades_data is not None:
                logger.info(f"   交易记录数: {len(self.trades_data)}")
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析策略表现"""
        logger.info("📊 开始性能分析...")
        
        summary = self.results_data.get('summary', {})
        results = self.results_data.get('results', [])
        
        # 基础指标分析
        performance_metrics = {
            'total_return': summary.get('total_return', 0),
            'sharpe_ratio': summary.get('sharpe_ratio', 0),
            'max_drawdown': summary.get('max_drawdown', 0),
            'total_trades': summary.get('total_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'avg_trade_size': summary.get('avg_trade_size', 0)
        }
        
        # 时间序列分析
        if results:
            df = pd.DataFrame(results)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 计算收益率序列
            df['returns'] = df['portfolio_value'].pct_change().fillna(0)
            df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
            
            # 波动率分析
            daily_vol = df['returns'].std() * np.sqrt(252 * 24 * 60)  # 假设分钟级数据
            performance_metrics['volatility'] = daily_vol
            
            # 最大连续亏损
            df['is_loss'] = df['returns'] < 0
            df['loss_streak'] = df['is_loss'].groupby((df['is_loss'] != df['is_loss'].shift()).cumsum()).cumsum()
            performance_metrics['max_loss_streak'] = df['loss_streak'].max()
            
            # 信号分析
            signal_stats = {
                'avg_signal_strength': df['signal'].abs().mean(),
                'signal_frequency': (df['signal'] != 0).mean(),
                'strong_signals_ratio': (df['signal'].abs() > 0.7).mean()
            }
            performance_metrics.update(signal_stats)
        
        self.analysis_results['performance'] = performance_metrics
        return performance_metrics
    
    def analyze_trades(self) -> Dict[str, Any]:
        """分析交易模式"""
        if self.trades_data is None or self.trades_data.empty:
            logger.warning("⚠️ 无交易数据可分析")
            return {}
        
        logger.info("📈 开始交易分析...")
        
        trades_analysis = {}
        
        # 交易频率分析
        if 'timestamp' in self.trades_data.columns:
            self.trades_data['hour'] = self.trades_data['timestamp'].dt.hour
            self.trades_data['day_of_week'] = self.trades_data['timestamp'].dt.dayofweek
            
            trades_analysis['hourly_distribution'] = self.trades_data['hour'].value_counts().to_dict()
            trades_analysis['daily_distribution'] = self.trades_data['day_of_week'].value_counts().to_dict()
        
        # 交易规模分析
        if 'quantity' in self.trades_data.columns:
            trades_analysis['trade_size_stats'] = {
                'mean': self.trades_data['quantity'].mean(),
                'std': self.trades_data['quantity'].std(),
                'min': self.trades_data['quantity'].min(),
                'max': self.trades_data['quantity'].max()
            }
        
        # 盈亏分析
        if 'pnl' in self.trades_data.columns:
            profitable_trades = self.trades_data[self.trades_data['pnl'] > 0]
            losing_trades = self.trades_data[self.trades_data['pnl'] < 0]
            
            trades_analysis['pnl_stats'] = {
                'total_pnl': self.trades_data['pnl'].sum(),
                'avg_profit': profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'profit_factor': abs(profitable_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
            }
        
        self.analysis_results['trades'] = trades_analysis
        return trades_analysis
    
    def identify_optimization_opportunities(self) -> Dict[str, List[str]]:
        """识别优化机会"""
        logger.info("🔍 识别优化机会...")
        
        opportunities = {
            'critical': [],
            'important': [],
            'minor': []
        }
        
        performance = self.analysis_results.get('performance', {})
        
        # 关键问题
        if performance.get('total_return', 0) < 0:
            opportunities['critical'].append("策略整体亏损，需要重新评估信号生成逻辑")
        
        if performance.get('sharpe_ratio', 0) < 0.5:
            opportunities['critical'].append("夏普比率过低，风险调整后收益不佳")
        
        if performance.get('max_drawdown', 0) < -0.1:
            opportunities['critical'].append("最大回撤过大，需要加强风险控制")
        
        # 重要改进
        if performance.get('win_rate', 0) < 0.55:
            opportunities['important'].append("胜率偏低，需要优化信号质量")
        
        if performance.get('signal_frequency', 0) > 0.8:
            opportunities['important'].append("信号过于频繁，可能存在过度交易")
        
        if performance.get('volatility', 0) > 0.3:
            opportunities['important'].append("策略波动率过高，需要平滑处理")
        
        # 次要优化
        if performance.get('strong_signals_ratio', 0) < 0.3:
            opportunities['minor'].append("强信号比例偏低，可以提高信号阈值")
        
        if performance.get('max_loss_streak', 0) > 10:
            opportunities['minor'].append("连续亏损次数较多，考虑添加止损机制")
        
        self.analysis_results['opportunities'] = opportunities
        return opportunities
    
    def suggest_parameter_optimization(self) -> Dict[str, Any]:
        """建议参数优化方案"""
        logger.info("⚙️ 生成参数优化建议...")
        
        suggestions = {
            'signal_parameters': {},
            'risk_parameters': {},
            'execution_parameters': {}
        }
        
        performance = self.analysis_results.get('performance', {})
        
        # 信号参数建议
        if performance.get('signal_frequency', 0) > 0.8:
            suggestions['signal_parameters']['signal_threshold'] = {
                'current': 0.5,
                'suggested': 0.7,
                'reason': '降低交易频率，提高信号质量'
            }
        
        if performance.get('win_rate', 0) < 0.55:
            suggestions['signal_parameters']['lookback_period'] = {
                'current': 20,
                'suggested': 30,
                'reason': '增加历史数据窗口，提高信号稳定性'
            }
        
        # 风险参数建议
        if performance.get('max_drawdown', 0) < -0.05:
            suggestions['risk_parameters']['position_limit'] = {
                'current': 0.1,
                'suggested': 0.05,
                'reason': '降低单笔交易风险敞口'
            }
        
        if performance.get('volatility', 0) > 0.25:
            suggestions['risk_parameters']['stop_loss'] = {
                'current': None,
                'suggested': 0.02,
                'reason': '添加止损机制控制单笔损失'
            }
        
        # 执行参数建议
        if performance.get('avg_trade_size', 0) > 5000:
            suggestions['execution_parameters']['max_trade_size'] = {
                'current': 10000,
                'suggested': 3000,
                'reason': '减小交易规模，降低市场冲击'
            }
        
        self.analysis_results['suggestions'] = suggestions
        return suggestions
    
    def generate_optimization_report(self) -> str:
        """生成优化报告"""
        logger.info("📋 生成优化报告...")
        
        report = []
        report.append("🏛️ Citadel 策略优化报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 性能摘要
        performance = self.analysis_results.get('performance', {})
        report.append("📊 性能摘要")
        report.append("-" * 30)
        report.append(f"总收益率: {performance.get('total_return', 0):.4f} ({performance.get('total_return', 0)*100:.2f}%)")
        report.append(f"夏普比率: {performance.get('sharpe_ratio', 0):.4f}")
        report.append(f"最大回撤: {performance.get('max_drawdown', 0):.4f} ({performance.get('max_drawdown', 0)*100:.2f}%)")
        report.append(f"总交易次数: {performance.get('total_trades', 0)}")
        report.append(f"胜率: {performance.get('win_rate', 0):.2%}")
        report.append(f"年化波动率: {performance.get('volatility', 0):.2%}")
        report.append("")
        
        # 优化机会
        opportunities = self.analysis_results.get('opportunities', {})
        report.append("🔍 优化机会")
        report.append("-" * 30)
        
        for level, items in opportunities.items():
            if items:
                report.append(f"{level.upper()}:")
                for item in items:
                    report.append(f"  • {item}")
                report.append("")
        
        # 参数建议
        suggestions = self.analysis_results.get('suggestions', {})
        report.append("⚙️ 参数优化建议")
        report.append("-" * 30)
        
        for category, params in suggestions.items():
            if params:
                report.append(f"{category.replace('_', ' ').title()}:")
                for param, details in params.items():
                    report.append(f"  • {param}:")
                    report.append(f"    当前值: {details.get('current', 'N/A')}")
                    report.append(f"    建议值: {details.get('suggested', 'N/A')}")
                    report.append(f"    原因: {details.get('reason', 'N/A')}")
                report.append("")
        
        # 下一步行动
        report.append("🚀 下一步行动建议")
        report.append("-" * 30)
        report.append("1. 实施关键参数调整")
        report.append("2. 进行参数网格搜索优化")
        report.append("3. 添加更多技术指标")
        report.append("4. 优化风险管理机制")
        report.append("5. 测试不同市场条件下的表现")
        
        return "\n".join(report)
    
    def save_analysis_results(self, output_dir: str = "./"):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细分析结果
        results_file = f"{output_dir}/citadel_optimization_analysis_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存优化报告
        report = self.generate_optimization_report()
        report_file = f"{output_dir}/citadel_optimization_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📁 分析结果已保存到: {results_file}")
        logger.info(f"📁 优化报告已保存到: {report_file}")
        
        return results_file, report_file
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """运行完整分析"""
        logger.info("🚀 开始完整策略分析...")
        
        # 执行各项分析
        self.analyze_performance()
        self.analyze_trades()
        self.identify_optimization_opportunities()
        self.suggest_parameter_optimization()
        
        # 保存结果
        results_file, report_file = self.save_analysis_results()
        
        # 打印报告
        report = self.generate_optimization_report()
        print(report)
        
        logger.info("✅ 策略分析完成!")
        
        return {
            'analysis_results': self.analysis_results,
            'results_file': results_file,
            'report_file': report_file
        }

def main():
    """主函数"""
    print("🏛️ Citadel Terminal AI Competition - 策略优化")
    print("=" * 60)
    
    # 查找最新的回测结果文件
    results_files = list(Path(".").glob("citadel_backtest_results_*.json"))
    trades_files = list(Path(".").glob("citadel_trades_*.csv"))
    
    if not results_files:
        logger.error("❌ 未找到回测结果文件")
        return
    
    # 使用最新的文件
    latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
    latest_trades = max(trades_files, key=lambda x: x.stat().st_mtime) if trades_files else None
    
    logger.info(f"📊 使用回测结果: {latest_results}")
    if latest_trades:
        logger.info(f"📈 使用交易记录: {latest_trades}")
    
    # 创建优化器并运行分析
    optimizer = CitadelOptimizer(str(latest_results), str(latest_trades) if latest_trades else "")
    results = optimizer.run_full_analysis()
    
    print("\n🎉 策略优化分析完成!")
    print(f"📁 详细结果: {results['results_file']}")
    print(f"📋 优化报告: {results['report_file']}")

if __name__ == "__main__":
    main()