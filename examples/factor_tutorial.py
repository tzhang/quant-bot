#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算和评估教程

本教程演示如何使用量化交易系统进行因子分析：
1. 因子计算基础
2. 因子评估方法
3. 图表解读技巧
4. 实战案例分析
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.factors.engine import FactorEngine

def tutorial_1_factor_basics():
    """
    教程1: 因子计算基础
    介绍什么是因子以及如何计算基础因子
    """
    print("=" * 60)
    print("📊 教程1: 因子计算基础")
    print("=" * 60)
    
    print("💡 什么是因子？")
    print("   因子是用来解释股票收益率的变量，常见类型包括:")
    print("   📈 价值因子: P/E, P/B, EV/EBITDA")
    print("   📊 成长因子: 收入增长率, 利润增长率")
    print("   💰 盈利因子: ROE, ROA, 毛利率")
    print("   📉 技术因子: 动量, 反转, 波动率")
    print("   💸 质量因子: 债务比率, 现金流稳定性")
    
    # 初始化因子引擎
    print("\n🔧 初始化因子引擎...")
    engine = FactorEngine()
    
    # 获取数据
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    print(f"📥 获取股票数据: {symbols}")
    
    try:
        data = engine.get_data(symbols, period='3m')
        print(f"✅ 成功获取数据，形状: {data.shape}")
        
        # 计算基础因子
        print("\n🧮 计算基础技术因子:")
        
        # 为每只股票计算因子
        factor_data = []
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) < 20:  # 确保有足够数据
                continue
                
            # 计算各种因子
            factors = calculate_basic_factors(symbol_data, symbol)
            factor_data.append(factors)
            
            print(f"   {symbol}: 计算完成")
        
        # 合并因子数据
        if factor_data:
            factors_df = pd.DataFrame(factor_data)
            print(f"\n📊 因子计算结果:")
            print(factors_df.round(4))
            
            return factors_df
        else:
            print("❌ 无法计算因子，数据不足")
            return None
            
    except Exception as e:
        print(f"❌ 因子计算失败: {str(e)}")
        return None

def calculate_basic_factors(data, symbol):
    """
    计算基础因子
    
    Args:
        data: 股票价格数据
        symbol: 股票代码
        
    Returns:
        dict: 因子值字典
    """
    # 确保数据按时间排序
    data = data.sort_index()
    
    # 计算收益率
    data['returns'] = data['close'].pct_change()
    
    # 1. 动量因子 (过去20天收益率)
    momentum_20d = (data['close'].iloc[-1] / data['close'].iloc[-21] - 1) if len(data) >= 21 else np.nan
    
    # 2. 反转因子 (过去5天收益率的负值)
    reversal_5d = -(data['close'].iloc[-1] / data['close'].iloc[-6] - 1) if len(data) >= 6 else np.nan
    
    # 3. 波动率因子 (过去20天收益率标准差)
    volatility_20d = data['returns'].tail(20).std() * np.sqrt(252) if len(data) >= 20 else np.nan
    
    # 4. 成交量因子 (相对成交量)
    avg_volume = data['volume'].tail(20).mean()
    recent_volume = data['volume'].tail(5).mean()
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else np.nan
    
    # 5. 价格位置因子 (当前价格在过去20天高低点中的位置)
    high_20d = data['high'].tail(20).max()
    low_20d = data['low'].tail(20).min()
    current_price = data['close'].iloc[-1]
    price_position = (current_price - low_20d) / (high_20d - low_20d) if (high_20d - low_20d) > 0 else np.nan
    
    # 6. RSI因子
    rsi = calculate_rsi(data['close'], 14)
    
    return {
        'symbol': symbol,
        'momentum_20d': momentum_20d,
        'reversal_5d': reversal_5d,
        'volatility_20d': volatility_20d,
        'volume_ratio': volume_ratio,
        'price_position': price_position,
        'rsi': rsi,
        'current_price': current_price
    }

def calculate_rsi(prices, window=14):
    """
    计算RSI指标
    
    Args:
        prices: 价格序列
        window: 计算窗口
        
    Returns:
        float: RSI值
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else np.nan

def tutorial_2_factor_evaluation():
    """
    教程2: 因子评估方法
    演示如何评估因子的有效性
    """
    print("\n" + "=" * 60)
    print("📊 教程2: 因子评估方法")
    print("=" * 60)
    
    print("🎯 因子评估的核心指标:")
    print("   📈 IC (Information Coefficient): 因子与未来收益的相关性")
    print("   📊 IC_IR: IC的信息比率，衡量IC的稳定性")
    print("   🏆 分层测试: 按因子值分组，比较各组收益")
    print("   🔄 换手率: 因子选股的稳定性")
    print("   📉 最大回撤: 因子策略的风险控制")
    
    # 获取因子数据
    factors_df = tutorial_1_factor_basics()
    if factors_df is None:
        print("❌ 无法进行因子评估，因子计算失败")
        return None
    
    # 使用FactorEngine进行完整评估
    print("\n🔍 使用FactorEngine进行完整因子评估...")
    engine = FactorEngine()
    
    try:
        # 选择一个因子进行详细评估
        factor_name = 'momentum_20d'
        print(f"📊 评估因子: {factor_name}")
        
        # 准备因子数据
        symbols = factors_df['symbol'].tolist()
        factor_values = factors_df.set_index('symbol')[factor_name].to_dict()
        
        print(f"   因子值范围: {min(factor_values.values()):.4f} 到 {max(factor_values.values()):.4f}")
        
        # 进行因子评估
        results = engine.evaluate_factor(
            factor_values=factor_values,
            symbols=symbols,
            period='2m',
            forward_days=5
        )
        
        if results:
            print("✅ 因子评估完成!")
            print_evaluation_results(results)
            return results
        else:
            print("❌ 因子评估失败")
            return None
            
    except Exception as e:
        print(f"❌ 因子评估出错: {str(e)}")
        return None

def print_evaluation_results(results):
    """
    打印因子评估结果
    
    Args:
        results: 评估结果字典
    """
    print("\n📊 因子评估结果:")
    print("-" * 40)
    
    # IC分析
    if 'ic_mean' in results:
        print(f"📈 IC均值: {results['ic_mean']:.4f}")
        print(f"📊 IC标准差: {results['ic_std']:.4f}")
        print(f"🎯 IC信息比率: {results['ic_ir']:.4f}")
        print(f"📋 IC胜率: {results['ic_win_rate']:.2%}")
    
    # 分层测试结果
    if 'layer_returns' in results:
        print(f"\n🏆 分层测试结果:")
        layer_returns = results['layer_returns']
        for i, ret in enumerate(layer_returns):
            print(f"   第{i+1}层收益: {ret:.2%}")
        
        # 多空收益
        if len(layer_returns) >= 2:
            long_short = layer_returns[-1] - layer_returns[0]
            print(f"   📈 多空收益: {long_short:.2%}")
    
    # 其他指标
    if 'turnover' in results:
        print(f"\n🔄 平均换手率: {results['turnover']:.2%}")
    
    if 'max_drawdown' in results:
        print(f"📉 最大回撤: {results['max_drawdown']:.2%}")

def tutorial_3_factor_visualization():
    """
    教程3: 因子可视化分析
    演示如何创建因子分析图表
    """
    print("\n" + "=" * 60)
    print("📊 教程3: 因子可视化分析")
    print("=" * 60)
    
    # 获取因子数据
    factors_df = tutorial_1_factor_basics()
    if factors_df is None:
        return
    
    # 创建因子分析图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('因子分析可视化图表', fontsize=16, fontweight='bold')
    
    # 1. 因子分布图
    ax1 = axes[0, 0]
    factor_cols = ['momentum_20d', 'reversal_5d', 'volatility_20d']
    for col in factor_cols:
        if col in factors_df.columns:
            ax1.hist(factors_df[col].dropna(), alpha=0.6, label=col, bins=10)
    ax1.set_title('因子分布图')
    ax1.set_xlabel('因子值')
    ax1.set_ylabel('频次')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 因子相关性热力图
    ax2 = axes[0, 1]
    numeric_cols = factors_df.select_dtypes(include=[np.number]).columns
    corr_matrix = factors_df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('因子相关性矩阵')
    
    # 3. 因子散点图
    ax3 = axes[0, 2]
    if 'momentum_20d' in factors_df.columns and 'volatility_20d' in factors_df.columns:
        scatter = ax3.scatter(factors_df['momentum_20d'], factors_df['volatility_20d'], 
                             c=factors_df['current_price'], cmap='viridis', alpha=0.7)
        ax3.set_xlabel('动量因子')
        ax3.set_ylabel('波动率因子')
        ax3.set_title('因子关系散点图')
        plt.colorbar(scatter, ax=ax3, label='当前价格')
    
    # 4. 因子排名图
    ax4 = axes[1, 0]
    if 'momentum_20d' in factors_df.columns:
        sorted_data = factors_df.sort_values('momentum_20d')
        ax4.bar(range(len(sorted_data)), sorted_data['momentum_20d'])
        ax4.set_title('动量因子排名')
        ax4.set_xlabel('股票排名')
        ax4.set_ylabel('因子值')
        ax4.grid(True, alpha=0.3)
    
    # 5. RSI分布图
    ax5 = axes[1, 1]
    if 'rsi' in factors_df.columns:
        rsi_data = factors_df['rsi'].dropna()
        ax5.hist(rsi_data, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(30, color='green', linestyle='--', label='超卖线')
        ax5.axvline(70, color='red', linestyle='--', label='超买线')
        ax5.set_title('RSI分布图')
        ax5.set_xlabel('RSI值')
        ax5.set_ylabel('频次')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. 价格位置分析
    ax6 = axes[1, 2]
    if 'price_position' in factors_df.columns and 'symbol' in factors_df.columns:
        ax6.bar(factors_df['symbol'], factors_df['price_position'])
        ax6.set_title('价格位置因子')
        ax6.set_xlabel('股票代码')
        ax6.set_ylabel('价格位置 (0-1)')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path('examples/factor_tutorial_charts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 因子分析图表已保存到: {output_path}")
    
    plt.show()

def tutorial_4_practical_case():
    """
    教程4: 实战案例分析
    演示完整的因子分析流程
    """
    print("\n" + "=" * 60)
    print("🎯 教程4: 实战案例分析")
    print("=" * 60)
    
    print("📋 实战案例: 构建动量选股策略")
    print("   目标: 使用动量因子选择表现最好的股票")
    print("   方法: 计算20日动量，选择前20%的股票")
    
    # 获取更多股票数据
    engine = FactorEngine()
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE']
    
    try:
        print(f"📥 获取{len(symbols)}只股票数据...")
        data = engine.get_data(symbols, period='3m')
        
        # 计算所有股票的动量因子
        print("🧮 计算动量因子...")
        factor_data = []
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) >= 21:
                momentum = (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[-21] - 1)
                factor_data.append({
                    'symbol': symbol,
                    'momentum_20d': momentum,
                    'current_price': symbol_data['close'].iloc[-1]
                })
        
        if not factor_data:
            print("❌ 无法计算因子数据")
            return
        
        # 创建因子DataFrame
        factors_df = pd.DataFrame(factor_data)
        factors_df = factors_df.sort_values('momentum_20d', ascending=False)
        
        print(f"\n📊 动量因子排名:")
        print(factors_df.round(4))
        
        # 选股策略
        top_n = max(1, len(factors_df) // 5)  # 选择前20%
        selected_stocks = factors_df.head(top_n)
        
        print(f"\n🎯 选股结果 (前{top_n}只):")
        for _, stock in selected_stocks.iterrows():
            print(f"   {stock['symbol']}: 动量={stock['momentum_20d']:.2%}, 价格=${stock['current_price']:.2f}")
        
        # 策略回测模拟
        print(f"\n📈 策略表现分析:")
        avg_momentum = selected_stocks['momentum_20d'].mean()
        print(f"   选中股票平均动量: {avg_momentum:.2%}")
        
        all_avg_momentum = factors_df['momentum_20d'].mean()
        print(f"   全市场平均动量: {all_avg_momentum:.2%}")
        
        excess_return = avg_momentum - all_avg_momentum
        print(f"   策略超额收益: {excess_return:.2%}")
        
        if excess_return > 0:
            print("   ✅ 策略表现优于市场平均水平")
        else:
            print("   ⚠️  策略表现低于市场平均水平")
        
        return selected_stocks
        
    except Exception as e:
        print(f"❌ 实战案例分析失败: {str(e)}")
        return None

def tutorial_5_advanced_tips():
    """
    教程5: 进阶技巧和最佳实践
    """
    print("\n" + "=" * 60)
    print("🚀 教程5: 进阶技巧和最佳实践")
    print("=" * 60)
    
    print("💡 因子分析最佳实践:")
    print("\n1. 📊 因子构建技巧:")
    print("   • 标准化处理: 使用Z-score或排名标准化")
    print("   • 去极值: 处理异常值，通常使用3倍标准差")
    print("   • 中性化: 去除行业、市值等风格因子影响")
    print("   • 时间衰减: 考虑因子的时效性")
    
    print("\n2. 🎯 因子评估要点:")
    print("   • IC分析: 关注IC均值、稳定性和显著性")
    print("   • 分层测试: 确保单调性和显著的多空收益")
    print("   • 换手率控制: 平衡收益和交易成本")
    print("   • 风险调整: 考虑最大回撤和夏普比率")
    
    print("\n3. 🔧 实战应用建议:")
    print("   • 多因子组合: 避免单一因子风险")
    print("   • 动态调整: 根据市场环境调整因子权重")
    print("   • 风险控制: 设置止损和仓位管理")
    print("   • 定期回测: 验证因子的持续有效性")
    
    print("\n4. ⚠️  常见陷阱:")
    print("   • 过度拟合: 避免在历史数据上过度优化")
    print("   • 幸存者偏差: 考虑退市股票的影响")
    print("   • 前视偏差: 确保使用当时可获得的信息")
    print("   • 数据挖掘: 避免无理论基础的因子挖掘")
    
    print("\n5. 📚 学习资源推荐:")
    print("   • 《量化投资策略与技术》- 丁鹏")
    print("   • 《因子投资：方法与实践》- 石川等")
    print("   • 《Active Portfolio Management》- Grinold & Kahn")
    print("   • WorldQuant研究论文系列")

def main():
    """
    主函数：运行所有因子分析教程
    """
    print("🎓 量化交易系统因子分析教程")
    print("本教程将带您掌握因子计算、评估和应用的完整流程")
    
    try:
        # 运行所有教程
        tutorial_1_factor_basics()
        tutorial_2_factor_evaluation()
        tutorial_3_factor_visualization()
        tutorial_4_practical_case()
        tutorial_5_advanced_tips()
        
        print("\n" + "=" * 60)
        print("🎉 恭喜！您已完成因子分析教程")
        print("=" * 60)
        print("📚 您已掌握:")
        print("   ✅ 因子计算的基本方法")
        print("   ✅ 因子评估的核心指标")
        print("   ✅ 因子可视化分析技巧")
        print("   ✅ 实战选股策略构建")
        print("   ✅ 进阶技巧和最佳实践")
        
        print("\n📖 继续学习:")
        print("   1. 完整因子评估: python examples/factor_evaluation.py")
        print("   2. 策略回测教程: python examples/backtest_tutorial.py")
        print("   3. 风险管理教程: python examples/risk_tutorial.py")
        print("   4. 阅读完整文档: docs/BEGINNER_GUIDE.md")
        
    except KeyboardInterrupt:
        print("\n⏹️  教程被用户中断")
    except Exception as e:
        print(f"\n❌ 教程执行出错: {str(e)}")
        print("请检查环境配置或联系技术支持")

if __name__ == "__main__":
    main()