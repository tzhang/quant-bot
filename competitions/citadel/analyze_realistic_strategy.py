#!/usr/bin/env python3
"""
分析现实版策略表现 - 找出问题并提出改进建议
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_realistic_strategy():
    """分析现实版策略表现"""
    print("📊 分析现实版策略表现")
    print("=" * 50)
    
    # 加载回测结果
    results_file = 'competitions/citadel/citadel_realistic_backtest_results_20251006_213754.json'
    trades_file = 'competitions/citadel/citadel_realistic_trades_20251006_213754.csv'
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    trades = pd.read_csv(trades_file)
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    
    print(f"📈 策略表现概览:")
    print(f"   总收益率: {results['summary']['total_return_pct']:.2f}%")
    print(f"   夏普比率: {results['summary']['sharpe_ratio']:.4f}")
    print(f"   最大回撤: {results['summary']['max_drawdown_pct']:.2f}%")
    print(f"   总交易次数: {results['summary']['total_trades']}")
    print(f"   胜率: {results['summary']['win_rate_pct']:.2f}%")
    print(f"   最终组合价值: ${results['summary']['final_portfolio_value']:,.2f}")
    
    # 分析交易模式
    print(f"\n🔍 交易模式分析:")
    
    buy_trades = trades[trades['action'] == 'BUY']
    sell_trades = trades[trades['action'].str.contains('SELL')]
    
    print(f"   买入交易: {len(buy_trades)} 次")
    print(f"   卖出交易: {len(sell_trades)} 次")
    
    # 分析卖出类型
    sell_types = sell_trades['action'].value_counts()
    print(f"   卖出类型分布:")
    for action, count in sell_types.items():
        percentage = (count / len(sell_trades)) * 100
        print(f"     {action}: {count} 次 ({percentage:.1f}%)")
    
    # 分析交易间隔
    if len(trades) > 1:
        trades_sorted = trades.sort_values('timestamp')
        time_diffs = trades_sorted['timestamp'].diff().dt.total_seconds()
        time_diffs = time_diffs.dropna()
        
        print(f"\n⏰ 交易时间间隔分析:")
        print(f"   平均间隔: {time_diffs.mean():.0f} 秒 ({time_diffs.mean()/3600:.1f} 小时)")
        print(f"   最短间隔: {time_diffs.min():.0f} 秒")
        print(f"   最长间隔: {time_diffs.max():.0f} 秒 ({time_diffs.max()/86400:.1f} 天)")
    
    # 分析盈亏情况
    print(f"\n💰 盈亏分析:")
    
    # 计算每笔交易的盈亏
    trade_pnl = []
    position = 0
    avg_cost = 0
    
    for _, trade in trades.iterrows():
        if trade['action'] == 'BUY':
            if position == 0:
                position = trade['shares']
                avg_cost = trade['price']
            else:
                # 加仓
                total_cost = position * avg_cost + trade['shares'] * trade['price']
                position += trade['shares']
                avg_cost = total_cost / position
        else:  # SELL
            if position > 0:
                pnl = (trade['price'] - avg_cost) * trade['shares']
                pnl_pct = (trade['price'] - avg_cost) / avg_cost
                trade_pnl.append({
                    'timestamp': trade['timestamp'],
                    'action': trade['action'],
                    'shares': trade['shares'],
                    'buy_price': avg_cost,
                    'sell_price': trade['price'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                position -= trade['shares']
    
    if trade_pnl:
        pnl_df = pd.DataFrame(trade_pnl)
        
        profitable_trades = pnl_df[pnl_df['pnl'] > 0]
        losing_trades = pnl_df[pnl_df['pnl'] < 0]
        
        print(f"   盈利交易: {len(profitable_trades)} 次")
        print(f"   亏损交易: {len(losing_trades)} 次")
        print(f"   实际胜率: {len(profitable_trades) / len(pnl_df) * 100:.2f}%")
        
        if len(profitable_trades) > 0:
            print(f"   平均盈利: ${profitable_trades['pnl'].mean():.2f} ({profitable_trades['pnl_pct'].mean()*100:.2f}%)")
            print(f"   最大盈利: ${profitable_trades['pnl'].max():.2f} ({profitable_trades['pnl_pct'].max()*100:.2f}%)")
        
        if len(losing_trades) > 0:
            print(f"   平均亏损: ${losing_trades['pnl'].mean():.2f} ({losing_trades['pnl_pct'].mean()*100:.2f}%)")
            print(f"   最大亏损: ${losing_trades['pnl'].min():.2f} ({losing_trades['pnl_pct'].min()*100:.2f}%)")
        
        # 盈亏比
        if len(losing_trades) > 0 and len(profitable_trades) > 0:
            profit_loss_ratio = abs(profitable_trades['pnl'].mean() / losing_trades['pnl'].mean())
            print(f"   盈亏比: {profit_loss_ratio:.2f}")
    
    # 分析组合价值变化
    print(f"\n📈 组合价值变化:")
    portfolio_values = trades['portfolio_value'].dropna()
    if len(portfolio_values) > 1:
        max_value = portfolio_values.max()
        min_value = portfolio_values.min()
        final_value = portfolio_values.iloc[-1]
        
        print(f"   初始价值: ${portfolio_values.iloc[0]:,.2f}")
        print(f"   最高价值: ${max_value:,.2f}")
        print(f"   最低价值: ${min_value:,.2f}")
        print(f"   最终价值: ${final_value:,.2f}")
        print(f"   最大回撤: {(max_value - min_value) / max_value * 100:.2f}%")
    
    # 问题诊断
    print(f"\n🔍 问题诊断:")
    issues = []
    
    if results['summary']['win_rate_pct'] < 30:
        issues.append("胜率过低 (<30%)")
    
    if results['summary']['total_return_pct'] < 0:
        issues.append("总收益为负")
    
    if results['summary']['sharpe_ratio'] < 0:
        issues.append("夏普比率为负")
    
    if 'SELL_STOP_LOSS' in sell_types and sell_types['SELL_STOP_LOSS'] > sell_types.get('SELL_TAKE_PROFIT', 0):
        issues.append("止损次数多于止盈次数")
    
    if len(issues) > 0:
        print("   发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")
    else:
        print("   ✅ 未发现明显问题")
    
    # 改进建议
    print(f"\n💡 改进建议:")
    suggestions = []
    
    if results['summary']['win_rate_pct'] < 30:
        suggestions.append("调整信号阈值，提高信号质量")
        suggestions.append("优化技术指标参数")
        suggestions.append("增加更多过滤条件")
    
    if 'SELL_STOP_LOSS' in sell_types and sell_types['SELL_STOP_LOSS'] > sell_types.get('SELL_TAKE_PROFIT', 0):
        suggestions.append("调整止盈止损比例")
        suggestions.append("使用动态止损策略")
    
    if results['summary']['sharpe_ratio'] < 0:
        suggestions.append("降低交易频率")
        suggestions.append("提高信号确认度")
        suggestions.append("加强风险管理")
    
    suggestions.extend([
        "考虑使用机器学习优化信号",
        "增加市场状态识别",
        "实施动态参数调整",
        "添加更多数据源"
    ])
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    # 创建可视化
    print(f"\n📊 生成分析图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('现实版策略表现分析', fontsize=16, fontweight='bold')
    
    # 1. 组合价值变化
    if len(portfolio_values) > 1:
        axes[0, 0].plot(range(len(portfolio_values)), portfolio_values, 'b-', linewidth=2)
        axes[0, 0].set_title('组合价值变化')
        axes[0, 0].set_xlabel('交易序号')
        axes[0, 0].set_ylabel('组合价值 ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].ticklabel_format(style='plain', axis='y')
    
    # 2. 交易类型分布
    if len(sell_types) > 0:
        axes[0, 1].pie(sell_types.values, labels=sell_types.index, autopct='%1.1f%%')
        axes[0, 1].set_title('卖出类型分布')
    
    # 3. 盈亏分布
    if trade_pnl:
        pnl_values = [t['pnl'] for t in trade_pnl]
        axes[1, 0].hist(pnl_values, bins=10, alpha=0.7, color='green' if np.mean(pnl_values) > 0 else 'red')
        axes[1, 0].set_title('单笔交易盈亏分布')
        axes[1, 0].set_xlabel('盈亏 ($)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 交易时间分布
    if len(trades) > 1:
        trade_hours = trades['timestamp'].dt.hour
        axes[1, 1].hist(trade_hours, bins=24, alpha=0.7, color='blue')
        axes[1, 1].set_title('交易时间分布')
        axes[1, 1].set_xlabel('小时')
        axes[1, 1].set_ylabel('交易次数')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = f'competitions/citadel/realistic_strategy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"   图表保存到: {chart_file}")
    
    # 保存分析报告
    report_file = f'competitions/citadel/realistic_strategy_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("现实版策略表现分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("策略表现概览:\n")
        f.write(f"  总收益率: {results['summary']['total_return_pct']:.2f}%\n")
        f.write(f"  夏普比率: {results['summary']['sharpe_ratio']:.4f}\n")
        f.write(f"  最大回撤: {results['summary']['max_drawdown_pct']:.2f}%\n")
        f.write(f"  总交易次数: {results['summary']['total_trades']}\n")
        f.write(f"  胜率: {results['summary']['win_rate_pct']:.2f}%\n")
        f.write(f"  最终组合价值: ${results['summary']['final_portfolio_value']:,.2f}\n\n")
        
        f.write("发现的问题:\n")
        if issues:
            for i, issue in enumerate(issues, 1):
                f.write(f"  {i}. {issue}\n")
        else:
            f.write("  未发现明显问题\n")
        f.write("\n")
        
        f.write("改进建议:\n")
        for i, suggestion in enumerate(suggestions, 1):
            f.write(f"  {i}. {suggestion}\n")
    
    print(f"   报告保存到: {report_file}")
    
    print(f"\n✅ 分析完成!")
    return results, trades, trade_pnl

if __name__ == "__main__":
    analyze_realistic_strategy()