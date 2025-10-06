#!/usr/bin/env python3
"""
🔍 改进版策略分析
验证超高收益率(6682.18%)的可靠性
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedStrategyAnalysis:
    """改进版策略分析"""
    
    def __init__(self):
        self.trades_df = None
        self.results_data = None
        self.analysis_results = {}
        
    def load_data(self):
        """加载交易数据和回测结果"""
        try:
            # 加载交易记录
            trades_file = "competitions/citadel/citadel_improved_trades_20251006_212038.csv"
            self.trades_df = pd.read_csv(trades_file)
            print(f"加载交易记录: {len(self.trades_df)} 条交易")
            
            # 加载回测结果
            results_file = "competitions/citadel/citadel_improved_backtest_results_20251006_212038.json"
            with open(results_file, 'r') as f:
                self.results_data = json.load(f)
            
            print(f"加载回测结果: {self.results_data['summary']['total_return']*100:.2f}% 收益率")
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def analyze_trade_patterns(self):
        """分析交易模式"""
        if self.trades_df is None:
            return
        
        print("\n📊 交易模式分析")
        print("=" * 50)
        
        # 基本统计
        total_trades = len(self.trades_df)
        buy_trades = len(self.trades_df[self.trades_df['action'] == 'BUY'])
        sell_trades = len(self.trades_df[self.trades_df['action'].str.contains('SELL')])
        
        print(f"总交易次数: {total_trades}")
        print(f"买入交易: {buy_trades}")
        print(f"卖出交易: {sell_trades}")
        
        # 分析卖出类型
        sell_types = self.trades_df[self.trades_df['action'].str.contains('SELL')]['action'].value_counts()
        print(f"\n卖出类型分布:")
        for action, count in sell_types.items():
            print(f"  {action}: {count} ({count/sell_trades*100:.1f}%)")
        
        # 分析交易规模
        trade_sizes = self.trades_df['shares'].unique()
        print(f"\n交易规模: {trade_sizes}")
        
        # 分析价格范围
        price_stats = self.trades_df['price'].describe()
        print(f"\n价格统计:")
        print(f"  最低价: ${price_stats['min']:.2f}")
        print(f"  最高价: ${price_stats['max']:.2f}")
        print(f"  平均价: ${price_stats['mean']:.2f}")
        print(f"  标准差: ${price_stats['std']:.2f}")
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'sell_types': sell_types.to_dict(),
            'price_stats': price_stats.to_dict()
        }
    
    def analyze_profit_loss(self):
        """分析盈亏情况"""
        if self.trades_df is None:
            return
        
        print("\n💰 盈亏分析")
        print("=" * 50)
        
        # 计算每笔交易的盈亏
        profits = []
        current_position = 0
        buy_price = 0
        
        for _, trade in self.trades_df.iterrows():
            if trade['action'] == 'BUY':
                current_position = trade['shares']
                buy_price = trade['price']
            elif 'SELL' in trade['action']:
                if current_position > 0:
                    profit = (trade['price'] - buy_price) * current_position
                    profit_pct = (trade['price'] - buy_price) / buy_price
                    profits.append({
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'action': trade['action'],
                        'buy_price': buy_price,
                        'sell_price': trade['price']
                    })
                current_position = 0
        
        if profits:
            profits_df = pd.DataFrame(profits)
            
            # 统计盈亏
            winning_trades = len(profits_df[profits_df['profit'] > 0])
            losing_trades = len(profits_df[profits_df['profit'] < 0])
            total_profit_trades = len(profits_df)
            
            print(f"盈利交易: {winning_trades} ({winning_trades/total_profit_trades*100:.1f}%)")
            print(f"亏损交易: {losing_trades} ({losing_trades/total_profit_trades*100:.1f}%)")
            
            # 平均盈亏
            avg_profit = profits_df['profit'].mean()
            avg_profit_pct = profits_df['profit_pct'].mean()
            
            print(f"平均盈亏: ${avg_profit:.2f}")
            print(f"平均盈亏率: {avg_profit_pct*100:.2f}%")
            
            # 最大盈亏
            max_profit = profits_df['profit'].max()
            max_loss = profits_df['profit'].min()
            
            print(f"最大盈利: ${max_profit:.2f}")
            print(f"最大亏损: ${max_loss:.2f}")
            
            # 分析止盈止损效果
            take_profit_trades = profits_df[profits_df['action'] == 'SELL_TAKE_PROFIT']
            stop_loss_trades = profits_df[profits_df['action'] == 'SELL_STOP_LOSS']
            
            if len(take_profit_trades) > 0:
                print(f"\n止盈交易: {len(take_profit_trades)}")
                print(f"止盈平均收益: {take_profit_trades['profit_pct'].mean()*100:.2f}%")
            
            if len(stop_loss_trades) > 0:
                print(f"止损交易: {len(stop_loss_trades)}")
                print(f"止损平均亏损: {stop_loss_trades['profit_pct'].mean()*100:.2f}%")
            
            return profits_df
        
        return None
    
    def analyze_portfolio_growth(self):
        """分析组合价值增长"""
        if self.trades_df is None:
            return
        
        print("\n📈 组合增长分析")
        print("=" * 50)
        
        # 提取组合价值序列
        portfolio_values = self.trades_df['portfolio_value'].values
        
        # 计算增长率
        initial_value = 1000000  # 初始资金
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        print(f"初始资金: ${initial_value:,.2f}")
        print(f"最终价值: ${final_value:,.2f}")
        print(f"总收益率: {total_return*100:.2f}%")
        
        # 分析增长趋势
        growth_rates = []
        for i in range(1, len(portfolio_values)):
            growth_rate = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            growth_rates.append(growth_rate)
        
        growth_rates = np.array(growth_rates)
        
        print(f"平均单笔增长率: {np.mean(growth_rates)*100:.4f}%")
        print(f"增长率标准差: {np.std(growth_rates)*100:.4f}%")
        print(f"最大单笔增长: {np.max(growth_rates)*100:.4f}%")
        print(f"最大单笔下跌: {np.min(growth_rates)*100:.4f}%")
        
        return {
            'portfolio_values': portfolio_values,
            'growth_rates': growth_rates,
            'total_return': total_return
        }
    
    def check_data_validity(self):
        """检查数据有效性"""
        print("\n🔍 数据有效性检查")
        print("=" * 50)
        
        issues = []
        
        # 检查交易数据
        if self.trades_df is not None:
            # 检查价格异常
            price_changes = self.trades_df['price'].pct_change().dropna()
            extreme_changes = price_changes[abs(price_changes) > 0.5]  # 50%以上的价格变化
            
            if len(extreme_changes) > 0:
                issues.append(f"发现 {len(extreme_changes)} 个极端价格变化 (>50%)")
            
            # 检查组合价值异常增长
            portfolio_changes = self.trades_df['portfolio_value'].pct_change().dropna()
            extreme_portfolio_changes = portfolio_changes[portfolio_changes > 0.1]  # 10%以上的组合增长
            
            if len(extreme_portfolio_changes) > 0:
                issues.append(f"发现 {len(extreme_portfolio_changes)} 个极端组合增长 (>10%)")
            
            # 检查交易频率
            timestamps = pd.to_datetime(self.trades_df['timestamp'])
            time_diffs = timestamps.diff().dropna()
            very_frequent = time_diffs[time_diffs < pd.Timedelta(seconds=1)]
            
            if len(very_frequent) > 0:
                issues.append(f"发现 {len(very_frequent)} 个高频交易 (<1秒间隔)")
        
        # 检查回测结果
        if self.results_data is not None:
            summary = self.results_data['summary']
            
            # 检查异常指标
            if summary['sharpe_ratio'] > 10:
                issues.append(f"夏普比率异常高: {summary['sharpe_ratio']:.2f}")
            
            if summary['total_return'] > 10:  # 1000%以上收益
                issues.append(f"收益率异常高: {summary['total_return']*100:.2f}%")
            
            if summary['max_drawdown'] < 0.01:  # 最大回撤小于1%
                issues.append(f"最大回撤异常小: {summary['max_drawdown']*100:.2f}%")
        
        if issues:
            print("⚠️  发现以下可疑问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("✅ 未发现明显的数据异常")
        
        return issues
    
    def create_visualization(self):
        """创建可视化图表"""
        if self.trades_df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('改进版策略详细分析', fontsize=16, fontweight='bold')
        
        # 1. 组合价值增长
        portfolio_values = self.trades_df['portfolio_value'].values
        axes[0, 0].plot(portfolio_values, color='blue', linewidth=2)
        axes[0, 0].set_title('组合价值增长')
        axes[0, 0].set_ylabel('组合价值 ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 价格走势
        prices = self.trades_df['price'].values
        axes[0, 1].plot(prices, color='green', alpha=0.7)
        axes[0, 1].set_title('价格走势')
        axes[0, 1].set_ylabel('价格 ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 交易动作分布
        action_counts = self.trades_df['action'].value_counts()
        axes[1, 0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('交易动作分布')
        
        # 4. 信号强度分布
        signals = self.trades_df['signal'].values
        axes[1, 1].hist(signals, bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title('信号强度分布')
        axes[1, 1].set_xlabel('信号强度')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = f"competitions/citadel/improved_strategy_analysis_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"\n分析图表已保存到: {chart_file}")
        
        return chart_file
    
    def generate_analysis_report(self):
        """生成分析报告"""
        report = []
        report.append("🔍 改进版策略分析报告")
        report.append("=" * 60)
        report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 基本信息
        if self.results_data:
            summary = self.results_data['summary']
            report.append("📊 基本信息")
            report.append("-" * 40)
            report.append(f"总收益率: {summary['total_return']*100:.2f}%")
            report.append(f"夏普比率: {summary['sharpe_ratio']:.4f}")
            report.append(f"最大回撤: {summary['max_drawdown']*100:.2f}%")
            report.append(f"总交易次数: {summary['total_trades']}")
            report.append(f"胜率: {summary['win_rate']*100:.2f}%")
            report.append("")
        
        # 数据有效性
        issues = self.check_data_validity()
        report.append("🔍 数据有效性评估")
        report.append("-" * 40)
        if issues:
            report.append("发现以下可疑问题:")
            for issue in issues:
                report.append(f"  • {issue}")
        else:
            report.append("✅ 数据通过基本有效性检查")
        report.append("")
        
        # 结论
        report.append("💡 分析结论")
        report.append("-" * 40)
        
        if len(issues) > 0:
            report.append("⚠️  策略结果存在以下风险:")
            report.append("1. 超高收益率可能不可持续")
            report.append("2. 数据中存在异常模式")
            report.append("3. 需要进一步验证策略逻辑")
            report.append("4. 建议在真实市场环境中谨慎测试")
        else:
            report.append("✅ 策略表现良好，但仍需注意:")
            report.append("1. 高收益伴随高风险")
            report.append("2. 历史表现不代表未来")
            report.append("3. 需要持续监控和调整")
        
        return "\n".join(report)

def main():
    """主函数"""
    print("🔍 改进版策略分析")
    print("=" * 60)
    
    # 初始化分析
    analyzer = ImprovedStrategyAnalysis()
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 分析交易模式
    trade_analysis = analyzer.analyze_trade_patterns()
    
    # 分析盈亏
    profit_analysis = analyzer.analyze_profit_loss()
    
    # 分析组合增长
    growth_analysis = analyzer.analyze_portfolio_growth()
    
    # 检查数据有效性
    validity_issues = analyzer.check_data_validity()
    
    # 创建可视化
    chart_file = analyzer.create_visualization()
    
    # 生成报告
    report_content = analyzer.generate_analysis_report()
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"competitions/citadel/improved_strategy_analysis_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📁 分析完成!")
    print(f"   报告文件: {report_file}")
    print(f"   图表文件: {chart_file}")
    
    # 显示关键结论
    print("\n🎯 关键结论:")
    if validity_issues:
        print("   ⚠️  发现数据异常，结果可信度存疑")
        print("   建议进一步验证策略逻辑")
    else:
        print("   ✅ 数据基本正常，但超高收益需谨慎对待")

if __name__ == "__main__":
    main()