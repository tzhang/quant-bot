#!/usr/bin/env python3
"""
Citadel 策略综合分析和对比
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import logging
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class CitadelComprehensiveAnalyzer:
    """Citadel策略综合分析器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.strategies_data = {}
        self.comparison_results = {}
        
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
    
    def load_all_results(self):
        """加载所有策略的回测结果"""
        self.logger.info("📊 加载所有策略的回测结果...")
        
        # 定义策略文件模式
        strategy_patterns = {
            'original': 'competitions/citadel/citadel_hft_backtest_results_*.json',
            'enhanced': 'competitions/citadel/citadel_enhanced_backtest_results_*.json',
            'final': 'competitions/citadel/citadel_final_backtest_results_*.json'
        }
        
        for strategy_name, pattern in strategy_patterns.items():
            files = glob.glob(pattern)
            if files:
                # 选择最新的文件
                latest_file = max(files, key=os.path.getctime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.strategies_data[strategy_name] = data
                    self.logger.info(f"✅ 加载 {strategy_name} 策略结果: {latest_file}")
                except Exception as e:
                    self.logger.error(f"❌ 加载 {strategy_name} 策略失败: {e}")
            else:
                self.logger.warning(f"⚠️  未找到 {strategy_name} 策略的回测结果")
        
        # 加载网格搜索结果
        grid_search_files = glob.glob('competitions/citadel/citadel_grid_search_results_*.json')
        if grid_search_files:
            latest_grid_file = max(grid_search_files, key=os.path.getctime)
            try:
                with open(latest_grid_file, 'r', encoding='utf-8') as f:
                    grid_data = json.load(f)
                self.strategies_data['grid_search'] = grid_data
                self.logger.info(f"✅ 加载网格搜索结果: {latest_grid_file}")
            except Exception as e:
                self.logger.error(f"❌ 加载网格搜索结果失败: {e}")
    
    def extract_performance_metrics(self, strategy_name: str, data: Dict) -> Dict:
        """提取性能指标"""
        if strategy_name == 'grid_search':
            # 网格搜索结果的处理
            if 'best_params' in data and data['all_results']:
                best_result = max(data['all_results'], key=lambda x: x['score'])
                if best_result['results']:
                    return {
                        'total_return': best_result['results']['total_return'],
                        'sharpe_ratio': best_result['results']['sharpe_ratio'],
                        'max_drawdown': best_result['results']['max_drawdown'],
                        'num_trades': best_result['results']['num_trades'],
                        'win_rate': best_result['results']['win_rate'],
                        'avg_trade_return': best_result['results']['avg_trade_return'],
                        'score': best_result['score']
                    }
            return {}
        else:
            # 常规回测结果的处理
            performance = data.get('performance_metrics', {})
            return {
                'total_return': performance.get('total_return', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'num_trades': performance.get('total_trades', 0),
                'win_rate': performance.get('win_rate', 0),
                'final_portfolio_value': performance.get('final_portfolio_value', 1000000),
                'avg_trade_return': performance.get('average_trade_return', 0)
            }
    
    def compare_strategies(self):
        """对比策略性能"""
        self.logger.info("🔍 开始策略性能对比分析...")
        
        comparison_data = []
        
        for strategy_name, data in self.strategies_data.items():
            metrics = self.extract_performance_metrics(strategy_name, data)
            if metrics:
                metrics['strategy'] = strategy_name
                comparison_data.append(metrics)
        
        if not comparison_data:
            self.logger.warning("没有可对比的策略数据")
            return
        
        # 创建对比DataFrame
        df = pd.DataFrame(comparison_data)
        self.comparison_results = df
        
        # 打印对比表格
        self._print_comparison_table(df)
        
        # 生成可视化图表
        self._create_comparison_charts(df)
    
    def _print_comparison_table(self, df: pd.DataFrame):
        """打印对比表格"""
        print("\n📊 策略性能对比表")
        print("=" * 80)
        
        # 格式化显示
        display_df = df.copy()
        
        # 格式化数值
        if 'total_return' in display_df.columns:
            display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.4f}")
        if 'sharpe_ratio' in display_df.columns:
            display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.4f}")
        if 'max_drawdown' in display_df.columns:
            display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.4f}")
        if 'win_rate' in display_df.columns:
            display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")
        
        print(display_df.to_string(index=False))
    
    def _create_comparison_charts(self, df: pd.DataFrame):
        """创建对比图表"""
        if len(df) < 2:
            self.logger.warning("策略数量不足，跳过图表生成")
            return
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Citadel策略性能对比', fontsize=16, fontweight='bold')
        
        # 1. 总收益率对比
        if 'total_return' in df.columns:
            ax1 = axes[0, 0]
            bars1 = ax1.bar(df['strategy'], df['total_return'])
            ax1.set_title('总收益率对比')
            ax1.set_ylabel('收益率')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
        
        # 2. 夏普比率对比
        if 'sharpe_ratio' in df.columns:
            ax2 = axes[0, 1]
            bars2 = ax2.bar(df['strategy'], df['sharpe_ratio'], color='orange')
            ax2.set_title('夏普比率对比')
            ax2.set_ylabel('夏普比率')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
        
        # 3. 最大回撤对比
        if 'max_drawdown' in df.columns:
            ax3 = axes[0, 2]
            bars3 = ax3.bar(df['strategy'], df['max_drawdown'], color='red')
            ax3.set_title('最大回撤对比')
            ax3.set_ylabel('最大回撤')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
        
        # 4. 交易次数对比
        if 'num_trades' in df.columns:
            ax4 = axes[1, 0]
            bars4 = ax4.bar(df['strategy'], df['num_trades'], color='green')
            ax4.set_title('交易次数对比')
            ax4.set_ylabel('交易次数')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 5. 胜率对比
        if 'win_rate' in df.columns:
            ax5 = axes[1, 1]
            bars5 = ax5.bar(df['strategy'], df['win_rate'], color='purple')
            ax5.set_title('胜率对比')
            ax5.set_ylabel('胜率')
            ax5.tick_params(axis='x', rotation=45)
            
            for bar in bars5:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}', ha='center', va='bottom')
        
        # 6. 综合评分雷达图
        ax6 = axes[1, 2]
        if len(df) > 0:
            # 标准化指标用于雷达图
            metrics = ['total_return', 'sharpe_ratio', 'win_rate']
            available_metrics = [m for m in metrics if m in df.columns]
            
            if available_metrics:
                angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))
                
                for i, row in df.iterrows():
                    values = []
                    for metric in available_metrics:
                        # 简单标准化到0-1范围
                        val = row[metric]
                        if metric == 'total_return':
                            val = max(0, min(1, (val + 0.1) / 0.2))  # -0.1到0.1映射到0-1
                        elif metric == 'sharpe_ratio':
                            val = max(0, min(1, val / 3))  # 0到3映射到0-1
                        elif metric == 'win_rate':
                            val = max(0, min(1, val))  # 已经是0-1范围
                        values.append(val)
                    
                    values += [values[0]]  # 闭合图形
                    
                    ax6.plot(angles, values, 'o-', linewidth=2, label=row['strategy'])
                    ax6.fill(angles, values, alpha=0.25)
                
                ax6.set_xticks(angles[:-1])
                ax6.set_xticklabels(available_metrics)
                ax6.set_ylim(0, 1)
                ax6.set_title('综合性能雷达图')
                ax6.legend()
                ax6.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = f"competitions/citadel/citadel_strategy_comparison_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"📈 对比图表已保存到: {chart_file}")
        
        plt.show()
    
    def generate_insights(self) -> List[str]:
        """生成分析洞察"""
        insights = []
        
        if self.comparison_results.empty:
            return ["没有足够的数据生成洞察"]
        
        df = self.comparison_results
        
        # 最佳策略分析
        if 'total_return' in df.columns:
            best_return_strategy = df.loc[df['total_return'].idxmax(), 'strategy']
            best_return_value = df['total_return'].max()
            insights.append(f"🏆 最佳收益率策略: {best_return_strategy} ({best_return_value:.4f})")
        
        if 'sharpe_ratio' in df.columns:
            best_sharpe_strategy = df.loc[df['sharpe_ratio'].idxmax(), 'strategy']
            best_sharpe_value = df['sharpe_ratio'].max()
            insights.append(f"📊 最佳夏普比率策略: {best_sharpe_strategy} ({best_sharpe_value:.4f})")
        
        if 'max_drawdown' in df.columns:
            best_drawdown_strategy = df.loc[df['max_drawdown'].idxmin(), 'strategy']
            best_drawdown_value = df['max_drawdown'].min()
            insights.append(f"🛡️ 最小回撤策略: {best_drawdown_strategy} ({best_drawdown_value:.4f})")
        
        # 策略改进分析
        if 'original' in df['strategy'].values and 'final' in df['strategy'].values:
            original_return = df[df['strategy'] == 'original']['total_return'].iloc[0]
            final_return = df[df['strategy'] == 'final']['total_return'].iloc[0]
            improvement = ((final_return - original_return) / abs(original_return)) * 100
            insights.append(f"📈 最终策略相比原版收益率改进: {improvement:.2f}%")
        
        # 风险收益分析
        if 'sharpe_ratio' in df.columns and 'max_drawdown' in df.columns:
            df['risk_adjusted_return'] = df['sharpe_ratio'] / (df['max_drawdown'] + 0.001)
            best_risk_adj_strategy = df.loc[df['risk_adjusted_return'].idxmax(), 'strategy']
            insights.append(f"⚖️ 最佳风险调整收益策略: {best_risk_adj_strategy}")
        
        return insights
    
    def generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if self.comparison_results.empty:
            return ["需要更多数据来生成建议"]
        
        df = self.comparison_results
        
        # 基于网格搜索结果的建议
        if 'grid_search' in df['strategy'].values:
            recommendations.append("🔍 网格搜索发现了更优的参数组合，建议采用优化后的参数")
            recommendations.append("📊 建议进一步扩大参数搜索范围，探索更多可能性")
        
        # 基于性能指标的建议
        avg_sharpe = df['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else 0
        if avg_sharpe < 1.0:
            recommendations.append("📉 夏普比率偏低，建议优化信号质量和风险管理")
        
        avg_drawdown = df['max_drawdown'].mean() if 'max_drawdown' in df.columns else 0
        if avg_drawdown > 0.05:
            recommendations.append("⚠️ 最大回撤较大，建议加强止损机制")
        
        # 交易频率建议
        avg_trades = df['num_trades'].mean() if 'num_trades' in df.columns else 0
        if avg_trades < 10:
            recommendations.append("📈 交易频率较低，可能错失机会，建议降低信号阈值")
        elif avg_trades > 1000:
            recommendations.append("⚡ 交易频率过高，可能增加交易成本，建议提高信号阈值")
        
        # 通用建议
        recommendations.extend([
            "🔄 建议实施动态参数调整机制",
            "📊 建议增加更多市场状态识别指标",
            "🛡️ 建议完善风险管理体系",
            "📈 建议进行实盘模拟测试验证"
        ])
        
        return recommendations
    
    def save_comprehensive_report(self):
        """保存综合分析报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成洞察和建议
        insights = self.generate_insights()
        recommendations = self.generate_recommendations()
        
        # 创建报告
        report = {
            'timestamp': timestamp,
            'analysis_type': 'comprehensive_strategy_comparison',
            'strategies_analyzed': list(self.strategies_data.keys()),
            'performance_comparison': self.comparison_results.to_dict('records') if not self.comparison_results.empty else [],
            'key_insights': insights,
            'recommendations': recommendations,
            'summary': {
                'total_strategies': len(self.strategies_data),
                'best_performing_metrics': self._get_best_performing_metrics()
            }
        }
        
        # 保存报告
        report_file = f"competitions/citadel/citadel_comprehensive_analysis_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 综合分析报告已保存到: {report_file}")
        
        return report
    
    def _get_best_performing_metrics(self) -> Dict:
        """获取最佳性能指标"""
        if self.comparison_results.empty:
            return {}
        
        df = self.comparison_results
        best_metrics = {}
        
        for metric in ['total_return', 'sharpe_ratio', 'win_rate']:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_metrics[f'best_{metric}'] = {
                    'strategy': df.loc[best_idx, 'strategy'],
                    'value': df.loc[best_idx, metric]
                }
        
        if 'max_drawdown' in df.columns:
            best_idx = df['max_drawdown'].idxmin()
            best_metrics['best_max_drawdown'] = {
                'strategy': df.loc[best_idx, 'strategy'],
                'value': df.loc[best_idx, 'max_drawdown']
            }
        
        return best_metrics

def main():
    """主函数"""
    print("📊 Citadel策略综合分析和对比")
    print("=" * 60)
    
    # 创建分析器
    analyzer = CitadelComprehensiveAnalyzer()
    
    # 加载所有结果
    analyzer.load_all_results()
    
    if not analyzer.strategies_data:
        print("❌ 没有找到任何策略结果文件")
        return
    
    # 对比策略
    analyzer.compare_strategies()
    
    # 生成洞察
    insights = analyzer.generate_insights()
    print("\n🔍 关键洞察:")
    print("-" * 40)
    for insight in insights:
        print(f"  {insight}")
    
    # 生成建议
    recommendations = analyzer.generate_recommendations()
    print("\n💡 优化建议:")
    print("-" * 40)
    for rec in recommendations:
        print(f"  {rec}")
    
    # 保存报告
    report = analyzer.save_comprehensive_report()
    
    print(f"\n📋 分析完成! 共分析了 {len(analyzer.strategies_data)} 个策略版本")
    print("🎉 综合分析报告已生成!")

if __name__ == "__main__":
    main()