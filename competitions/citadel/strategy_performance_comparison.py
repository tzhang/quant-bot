#!/usr/bin/env python3
"""
🏛️ Citadel 策略性能对比报告
比较所有策略版本的表现
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StrategyPerformanceComparison:
    """策略性能对比分析"""
    
    def __init__(self):
        self.strategies_data = {}
        self.comparison_results = {}
        
    def load_strategy_results(self, results_dir="competitions/citadel"):
        """加载所有策略的回测结果"""
        results_files = []
        
        # 查找所有回测结果文件
        for file in os.listdir(results_dir):
            if 'backtest_results' in file and file.endswith('.json') and not file.startswith('._'):
                results_files.append(os.path.join(results_dir, file))
        
        print(f"找到 {len(results_files)} 个回测结果文件")
        
        for file_path in results_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # 提取策略名称
                strategy_name = data.get('config', {}).get('strategy_name', 'unknown')
                if strategy_name == 'unknown':
                    # 从文件名提取策略名称
                    filename = os.path.basename(file_path)
                    if 'ultimate' in filename:
                        strategy_name = 'ultimate'
                    elif 'conservative' in filename:
                        strategy_name = 'conservative'
                    elif 'balanced' in filename:
                        strategy_name = 'balanced'
                    elif 'robust' in filename:
                        strategy_name = 'robust'
                    elif 'final' in filename:
                        strategy_name = 'final'
                    elif 'improved' in filename:
                        strategy_name = 'improved'
                    else:
                        strategy_name = filename.split('_')[1]
                
                # 提取时间戳
                timestamp = filename.split('_')[-1].replace('.json', '')
                
                self.strategies_data[f"{strategy_name}_{timestamp}"] = {
                    'name': strategy_name,
                    'timestamp': timestamp,
                    'config': data.get('config', {}),
                    'summary': data.get('summary', {}),
                    'file_path': file_path
                }
                
                print(f"加载策略: {strategy_name} ({timestamp})")
                
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        return len(self.strategies_data)
    
    def create_performance_summary(self):
        """创建性能汇总表"""
        summary_data = []
        
        for strategy_id, data in self.strategies_data.items():
            summary = data['summary']
            config = data['config']
            
            # 提取关键指标
            row = {
                '策略名称': data['name'],
                '时间戳': data['timestamp'],
                '总收益率(%)': round(summary.get('total_return', 0) * 100, 2),
                '夏普比率': round(summary.get('sharpe_ratio', 0), 4),
                '最大回撤(%)': round(summary.get('max_drawdown', 0) * 100, 2),
                '总交易次数': summary.get('total_trades', 0),
                '胜率(%)': round(summary.get('win_rate', 0) * 100, 2),
                '最终组合价值($)': round(summary.get('final_portfolio_value', 0), 2),
                '平均交易收益(%)': round(summary.get('avg_trade_return', 0) * 100, 2),
                '信号阈值': config.get('signal_parameters', {}).get('signal_threshold', 'N/A'),
                '仓位限制(%)': round(config.get('signal_parameters', {}).get('position_limit', 0) * 100, 2),
                '最大交易规模': config.get('signal_parameters', {}).get('max_trade_size', 'N/A')
            }
            
            summary_data.append(row)
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # 按收益率排序
        self.summary_df = self.summary_df.sort_values('总收益率(%)', ascending=False)
        
        return self.summary_df
    
    def analyze_risk_return_profile(self):
        """分析风险收益特征"""
        risk_return_data = []
        
        for strategy_id, data in self.strategies_data.items():
            summary = data['summary']
            
            risk_return_data.append({
                '策略': data['name'],
                '收益率': summary.get('total_return', 0),
                '波动率': summary.get('max_drawdown', 0),  # 用最大回撤作为风险代理
                '夏普比率': summary.get('sharpe_ratio', 0),
                '交易次数': summary.get('total_trades', 0)
            })
        
        return pd.DataFrame(risk_return_data)
    
    def create_visualization(self):
        """创建可视化图表"""
        if len(self.strategies_data) == 0:
            print("没有数据可供可视化")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Citadel 策略性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 收益率对比
        strategies = [data['name'] for data in self.strategies_data.values()]
        returns = [data['summary'].get('total_return', 0) * 100 for data in self.strategies_data.values()]
        
        axes[0, 0].bar(strategies, returns, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('总收益率对比 (%)')
        axes[0, 0].set_ylabel('收益率 (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 夏普比率对比
        sharpe_ratios = [data['summary'].get('sharpe_ratio', 0) for data in self.strategies_data.values()]
        
        axes[0, 1].bar(strategies, sharpe_ratios, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('夏普比率对比')
        axes[0, 1].set_ylabel('夏普比率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 最大回撤对比
        max_drawdowns = [data['summary'].get('max_drawdown', 0) * 100 for data in self.strategies_data.values()]
        
        axes[0, 2].bar(strategies, max_drawdowns, color='salmon', alpha=0.7)
        axes[0, 2].set_title('最大回撤对比 (%)')
        axes[0, 2].set_ylabel('最大回撤 (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 交易次数对比
        trade_counts = [data['summary'].get('total_trades', 0) for data in self.strategies_data.values()]
        
        axes[1, 0].bar(strategies, trade_counts, color='orange', alpha=0.7)
        axes[1, 0].set_title('交易次数对比')
        axes[1, 0].set_ylabel('交易次数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. 胜率对比
        win_rates = [data['summary'].get('win_rate', 0) * 100 for data in self.strategies_data.values()]
        
        axes[1, 1].bar(strategies, win_rates, color='purple', alpha=0.7)
        axes[1, 1].set_title('胜率对比 (%)')
        axes[1, 1].set_ylabel('胜率 (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. 风险收益散点图
        risk_return_df = self.analyze_risk_return_profile()
        
        scatter = axes[1, 2].scatter(risk_return_df['波动率'] * 100, 
                                   risk_return_df['收益率'] * 100,
                                   s=risk_return_df['交易次数'] / 50,  # 气泡大小表示交易次数
                                   alpha=0.6,
                                   c=risk_return_df['夏普比率'],
                                   cmap='viridis')
        
        axes[1, 2].set_xlabel('风险 (最大回撤 %)')
        axes[1, 2].set_ylabel('收益率 (%)')
        axes[1, 2].set_title('风险收益散点图')
        
        # 添加策略标签
        for i, strategy in enumerate(risk_return_df['策略']):
            axes[1, 2].annotate(strategy, 
                              (risk_return_df['波动率'].iloc[i] * 100, 
                               risk_return_df['收益率'].iloc[i] * 100),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.8)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=axes[1, 2], label='夏普比率')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = f"competitions/citadel/strategy_comparison_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"对比图表已保存到: {chart_file}")
        
        return chart_file
    
    def generate_detailed_report(self):
        """生成详细的对比报告"""
        report = []
        report.append("🏛️ Citadel 策略性能对比报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"对比策略数量: {len(self.strategies_data)}")
        report.append("")
        
        # 性能汇总
        report.append("📊 性能汇总")
        report.append("-" * 40)
        
        if hasattr(self, 'summary_df') and not self.summary_df.empty:
            # 最佳策略
            best_return = self.summary_df.iloc[0]
            report.append(f"🏆 最佳收益策略: {best_return['策略名称']}")
            report.append(f"   收益率: {best_return['总收益率(%)']}%")
            report.append(f"   夏普比率: {best_return['夏普比率']}")
            report.append(f"   最大回撤: {best_return['最大回撤(%)']}%")
            report.append("")
            
            # 最佳夏普比率
            best_sharpe = self.summary_df.loc[self.summary_df['夏普比率'].idxmax()]
            report.append(f"📈 最佳夏普比率策略: {best_sharpe['策略名称']}")
            report.append(f"   夏普比率: {best_sharpe['夏普比率']}")
            report.append(f"   收益率: {best_sharpe['总收益率(%)']}%")
            report.append("")
            
            # 最低风险
            min_drawdown = self.summary_df.loc[self.summary_df['最大回撤(%)'].idxmin()]
            report.append(f"🛡️ 最低风险策略: {min_drawdown['策略名称']}")
            report.append(f"   最大回撤: {min_drawdown['最大回撤(%)']}%")
            report.append(f"   收益率: {min_drawdown['总收益率(%)']}%")
            report.append("")
        
        # 策略分析
        report.append("🔍 策略分析")
        report.append("-" * 40)
        
        for strategy_id, data in self.strategies_data.items():
            name = data['name']
            summary = data['summary']
            config = data['config']
            
            report.append(f"策略: {name}")
            report.append(f"  总收益率: {summary.get('total_return', 0)*100:.2f}%")
            report.append(f"  夏普比率: {summary.get('sharpe_ratio', 0):.4f}")
            report.append(f"  最大回撤: {summary.get('max_drawdown', 0)*100:.2f}%")
            report.append(f"  交易次数: {summary.get('total_trades', 0)}")
            report.append(f"  胜率: {summary.get('win_rate', 0)*100:.2f}%")
            report.append(f"  信号阈值: {config.get('signal_parameters', {}).get('signal_threshold', 'N/A')}")
            report.append("")
        
        # 结论和建议
        report.append("💡 结论和建议")
        report.append("-" * 40)
        
        if hasattr(self, 'summary_df') and not self.summary_df.empty:
            # 分析交易频率vs收益
            high_return_strategies = self.summary_df[self.summary_df['总收益率(%)'] > 10]
            low_risk_strategies = self.summary_df[self.summary_df['最大回撤(%)'] < 5]
            
            if not high_return_strategies.empty:
                report.append("高收益策略特征:")
                avg_trades = high_return_strategies['总交易次数'].mean()
                avg_threshold = high_return_strategies['信号阈值'].mean() if high_return_strategies['信号阈值'].dtype != 'object' else 'N/A'
                report.append(f"  平均交易次数: {avg_trades:.0f}")
                report.append(f"  平均信号阈值: {avg_threshold}")
                report.append("")
            
            if not low_risk_strategies.empty:
                report.append("低风险策略特征:")
                avg_return = low_risk_strategies['总收益率(%)'].mean()
                avg_sharpe = low_risk_strategies['夏普比率'].mean()
                report.append(f"  平均收益率: {avg_return:.2f}%")
                report.append(f"  平均夏普比率: {avg_sharpe:.4f}")
                report.append("")
        
        report.append("建议:")
        report.append("1. 平衡收益与风险，避免过度交易")
        report.append("2. 优化信号阈值，提高交易质量")
        report.append("3. 加强风险管理，控制最大回撤")
        report.append("4. 考虑市场条件，动态调整参数")
        
        return "\n".join(report)
    
    def save_comparison_report(self):
        """保存对比报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细报告
        report_content = self.generate_detailed_report()
        report_file = f"competitions/citadel/strategy_comparison_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 保存汇总表
        if hasattr(self, 'summary_df'):
            excel_file = f"competitions/citadel/strategy_comparison_summary_{timestamp}.xlsx"
            self.summary_df.to_excel(excel_file, index=False)
            print(f"汇总表已保存到: {excel_file}")
        
        print(f"对比报告已保存到: {report_file}")
        return report_file

def main():
    """主函数"""
    print("🏛️ Citadel 策略性能对比分析")
    print("=" * 60)
    
    # 初始化对比分析
    comparison = StrategyPerformanceComparison()
    
    # 加载策略结果
    num_strategies = comparison.load_strategy_results()
    
    if num_strategies == 0:
        print("未找到任何策略回测结果文件")
        return
    
    print(f"\n成功加载 {num_strategies} 个策略的回测结果")
    
    # 创建性能汇总
    summary_df = comparison.create_performance_summary()
    print("\n📊 策略性能汇总:")
    print(summary_df.to_string(index=False))
    
    # 创建可视化
    chart_file = comparison.create_visualization()
    
    # 生成并保存报告
    report_file = comparison.save_comparison_report()
    
    print(f"\n📁 对比分析完成!")
    print(f"   报告文件: {report_file}")
    print(f"   图表文件: {chart_file}")
    
    # 显示关键发现
    print("\n🔍 关键发现:")
    if not summary_df.empty:
        best_strategy = summary_df.iloc[0]
        print(f"   最佳策略: {best_strategy['策略名称']}")
        print(f"   最高收益: {best_strategy['总收益率(%)']}%")
        print(f"   最佳夏普: {summary_df.loc[summary_df['夏普比率'].idxmax(), '夏普比率']:.4f}")
        print(f"   最低风险: {summary_df['最大回撤(%)'].min()}%")

if __name__ == "__main__":
    main()