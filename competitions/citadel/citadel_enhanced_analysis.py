#!/usr/bin/env python3
"""
Citadel 增强版策略分析脚本

分析增强版策略的回测结果，对比原版策略，并提供优化建议
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CitadelEnhancedAnalyzer:
    """Citadel 增强版策略分析器"""
    
    def __init__(self):
        self.results_dir = Path("competitions/citadel")
        self.analysis_results = {}
        
    def load_results(self):
        """加载回测结果"""
        results = {}
        
        # 查找最新的结果文件
        original_files = list(self.results_dir.glob("citadel_backtest_results_*.json"))
        enhanced_files = list(self.results_dir.glob("citadel_enhanced_backtest_results_*.json"))
        
        if original_files:
            original_file = max(original_files, key=lambda x: x.stat().st_mtime)
            with open(original_file, 'r', encoding='utf-8') as f:
                results['original'] = json.load(f)
            print(f"📊 加载原版策略结果: {original_file.name}")
        
        if enhanced_files:
            enhanced_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
            with open(enhanced_file, 'r', encoding='utf-8') as f:
                results['enhanced'] = json.load(f)
            print(f"📊 加载增强版策略结果: {enhanced_file.name}")
        
        return results
    
    def analyze_performance(self, results):
        """分析策略性能"""
        analysis = {}
        
        for strategy_name, data in results.items():
            summary = data['summary']
            
            # 基础指标
            analysis[strategy_name] = {
                'total_return': summary.get('total_return', 0),
                'sharpe_ratio': summary.get('sharpe_ratio', 0),
                'max_drawdown': summary.get('max_drawdown', 0),
                'total_trades': summary.get('total_trades', 0),
                'win_rate': summary.get('win_rate', 0),
                'final_portfolio_value': summary.get('final_portfolio_value', 0),
                'avg_trade_size': summary.get('avg_trade_size', 0)
            }
            
            # 计算额外指标
            if 'results' in data and len(data['results']) > 0:
                portfolio_values = [r['portfolio_value'] for r in data['results']]
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                
                analysis[strategy_name].update({
                    'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0,
                    'max_portfolio_value': max(portfolio_values),
                    'min_portfolio_value': min(portfolio_values),
                    'portfolio_stability': np.std(portfolio_values) / np.mean(portfolio_values) if np.mean(portfolio_values) > 0 else 0
                })
        
        return analysis
    
    def compare_strategies(self, analysis):
        """对比策略性能"""
        if 'original' not in analysis or 'enhanced' not in analysis:
            print("⚠️ 缺少对比数据，无法进行策略对比")
            return {}
        
        original = analysis['original']
        enhanced = analysis['enhanced']
        
        comparison = {}
        
        # 计算改进幅度
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades', 'win_rate']:
            orig_val = original.get(metric, 0)
            enh_val = enhanced.get(metric, 0)
            
            if orig_val != 0:
                improvement = (enh_val - orig_val) / abs(orig_val) * 100
            else:
                improvement = 0 if enh_val == 0 else float('inf')
            
            comparison[metric] = {
                'original': orig_val,
                'enhanced': enh_val,
                'improvement_pct': improvement
            }
        
        return comparison
    
    def generate_insights(self, analysis, comparison):
        """生成分析洞察"""
        insights = []
        
        # 策略表现分析
        if 'enhanced' in analysis:
            enhanced = analysis['enhanced']
            
            if enhanced['total_trades'] == 0:
                insights.append("🚨 增强版策略没有执行任何交易，可能存在以下问题：")
                insights.append("   • 信号阈值设置过高 (当前0.7)")
                insights.append("   • 历史窗口过长 (当前30期)")
                insights.append("   • 风险控制过于严格")
                insights.append("   • 数据质量或特征工程问题")
            
            if enhanced['sharpe_ratio'] == 0:
                insights.append("📉 夏普比率为0，表明策略没有产生超额收益")
            
            if enhanced['max_drawdown'] == 0:
                insights.append("📊 最大回撤为0，可能是因为没有交易或风险控制过严")
        
        # 对比分析
        if comparison:
            for metric, data in comparison.items():
                if data['improvement_pct'] > 10:
                    insights.append(f"✅ {metric} 显著改善: {data['improvement_pct']:.1f}%")
                elif data['improvement_pct'] < -10:
                    insights.append(f"❌ {metric} 显著恶化: {data['improvement_pct']:.1f}%")
        
        return insights
    
    def generate_recommendations(self, analysis, insights):
        """生成优化建议"""
        recommendations = []
        
        if 'enhanced' in analysis and analysis['enhanced']['total_trades'] == 0:
            recommendations.extend([
                "🔧 参数调整建议：",
                "   • 降低信号阈值从0.7到0.3-0.5",
                "   • 减少历史窗口从30到10-20期",
                "   • 放宽仓位限制从5%到10%",
                "   • 调整止损止盈比例",
                "",
                "📊 特征工程优化：",
                "   • 检查技术指标计算逻辑",
                "   • 增加更多市场微观结构特征",
                "   • 优化多因子权重配置",
                "   • 添加市场状态识别",
                "",
                "🎯 策略逻辑改进：",
                "   • 实施动态阈值调整",
                "   • 添加市场流动性过滤",
                "   • 优化入场和出场时机",
                "   • 增强风险管理机制"
            ])
        
        recommendations.extend([
            "",
            "🚀 下一步行动计划：",
            "   1. 参数敏感性分析",
            "   2. 网格搜索优化",
            "   3. 多市场环境测试",
            "   4. 实时交易模拟",
            "   5. 风险指标监控"
        ])
        
        return recommendations
    
    def create_visualization(self, analysis, comparison):
        """创建可视化图表"""
        if not analysis:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Citadel 策略性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 性能指标对比
        if len(analysis) > 1:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            strategies = list(analysis.keys())
            
            x = np.arange(len(metrics))
            width = 0.35
            
            for i, strategy in enumerate(strategies):
                values = [analysis[strategy].get(metric, 0) for metric in metrics]
                axes[0, 0].bar(x + i*width, values, width, label=strategy.title())
            
            axes[0, 0].set_xlabel('性能指标')
            axes[0, 0].set_ylabel('数值')
            axes[0, 0].set_title('关键性能指标对比')
            axes[0, 0].set_xticks(x + width/2)
            axes[0, 0].set_xticklabels(['总收益率', '夏普比率', '最大回撤', '胜率'])
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 交易统计
        if len(analysis) > 1:
            trade_metrics = ['total_trades', 'avg_trade_size']
            for i, strategy in enumerate(strategies):
                values = [analysis[strategy].get(metric, 0) for metric in trade_metrics]
                axes[0, 1].bar(x[:len(trade_metrics)] + i*width, values, width, label=strategy.title())
            
            axes[0, 1].set_xlabel('交易指标')
            axes[0, 1].set_ylabel('数值')
            axes[0, 1].set_title('交易统计对比')
            axes[0, 1].set_xticks(x[:len(trade_metrics)] + width/2)
            axes[0, 1].set_xticklabels(['交易次数', '平均交易规模'])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 改进幅度
        if comparison:
            metrics = list(comparison.keys())
            improvements = [comparison[metric]['improvement_pct'] for metric in metrics]
            
            colors = ['green' if x > 0 else 'red' for x in improvements]
            axes[1, 0].bar(metrics, improvements, color=colors, alpha=0.7)
            axes[1, 0].set_xlabel('性能指标')
            axes[1, 0].set_ylabel('改进幅度 (%)')
            axes[1, 0].set_title('增强版策略改进幅度')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].grid(True, alpha=0.3)
            plt.setp(axes[1, 0].get_xticklabels(), rotation=45)
        
        # 4. 风险收益散点图
        if len(analysis) > 1:
            for strategy in strategies:
                ret = analysis[strategy].get('total_return', 0)
                vol = analysis[strategy].get('volatility', 0)
                axes[1, 1].scatter(vol, ret, s=100, label=strategy.title(), alpha=0.7)
                axes[1, 1].annotate(strategy.title(), (vol, ret), 
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[1, 1].set_xlabel('波动率')
            axes[1, 1].set_ylabel('总收益率')
            axes[1, 1].set_title('风险收益分布')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.results_dir / f"citadel_enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 分析图表已保存到: {chart_file}")
        
        return chart_file
    
    def run_analysis(self):
        """运行完整分析"""
        print("🏛️ Citadel 增强版策略分析")
        print("=" * 60)
        
        # 加载数据
        results = self.load_results()
        if not results:
            print("❌ 未找到回测结果文件")
            return
        
        # 性能分析
        analysis = self.analyze_performance(results)
        
        # 策略对比
        comparison = self.compare_strategies(analysis)
        
        # 生成洞察
        insights = self.generate_insights(analysis, comparison)
        
        # 生成建议
        recommendations = self.generate_recommendations(analysis, insights)
        
        # 创建可视化
        chart_file = self.create_visualization(analysis, comparison)
        
        # 输出结果
        print("\n📊 策略性能分析:")
        print("-" * 40)
        for strategy_name, metrics in analysis.items():
            print(f"\n{strategy_name.upper()} 策略:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        if comparison:
            print("\n📈 策略对比分析:")
            print("-" * 40)
            for metric, data in comparison.items():
                print(f"{metric}:")
                print(f"  原版: {data['original']:.4f}")
                print(f"  增强版: {data['enhanced']:.4f}")
                print(f"  改进: {data['improvement_pct']:.1f}%")
        
        print("\n🔍 关键洞察:")
        print("-" * 40)
        for insight in insights:
            print(insight)
        
        print("\n💡 优化建议:")
        print("-" * 40)
        for recommendation in recommendations:
            print(recommendation)
        
        # 保存分析结果
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'comparison': comparison,
            'insights': insights,
            'recommendations': recommendations,
            'chart_file': str(chart_file) if 'chart_file' in locals() else None
        }
        
        result_file = self.results_dir / f"citadel_enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 分析结果已保存到: {result_file}")
        print("\n🎉 Citadel 增强版策略分析完成!")

def main():
    """主函数"""
    analyzer = CitadelEnhancedAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()