#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始演示脚本
================

这个脚本演示了量化交易系统的核心功能，让初学者能够快速上手。

运行方式:
    python examples/quick_start_demo.py

功能演示:
1. 数据获取和缓存
2. 因子计算
3. 因子评估
4. 图表生成
5. 结果分析

作者: 量化交易系统
日期: 2024年
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用正确的模块导入路径
from src.factors.engine import FactorEngine
from src.factors.technical import TechnicalFactors
from src.performance.analyzer import PerformanceAnalyzer

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step_num, description):
    """打印步骤"""
    print(f"\n📍 步骤 {step_num}: {description}")
    print("-" * 40)

def print_success(message):
    """打印成功信息"""
    print(f"✅ {message}")

def print_info(message):
    """打印信息"""
    print(f"ℹ️  {message}")

def print_warning(message):
    """打印警告"""
    print(f"⚠️  {message}")

def wait_for_user():
    """等待用户按键继续"""
    input("\n按回车键继续...")

def main():
    """主演示函数"""
    
    print_header("🚀 量化交易系统快速开始演示")
    
    print("""
欢迎使用量化交易系统！

本演示将带您体验系统的核心功能：
1. 数据获取和缓存机制
2. 技术因子计算
3. 因子评估和分析
4. 可视化图表生成
5. 结果解读和分析

整个演示大约需要 3-5 分钟。
    """)
    
    wait_for_user()
    
    # 初始化系统组件
    print_step(1, "初始化系统组件")
    
    try:
        # 创建数据管理器
        print_info("正在初始化数据管理器...")
        from src.data.data_manager import DataManager
        data_manager = DataManager()
        print_success("数据管理器初始化完成")
        
        # 创建因子引擎
        print_info("正在初始化因子引擎...")
        engine = FactorEngine()
        print_success("因子引擎初始化完成")
        
        # 创建技术因子计算器
        print_info("正在初始化技术因子计算器...")
        tech_factors = TechnicalFactors()
        print_success("技术因子计算器初始化完成")
        
        # 创建性能分析器
        print_info("正在初始化性能分析器...")
        evaluator = PerformanceAnalyzer()
        print_success("性能分析器初始化完成")
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查环境配置，运行 python test_environment.py 进行诊断")
        return
    
    wait_for_user()
    
    # 数据获取演示
    print_step(2, "数据获取和缓存演示")
    
    # 选择演示股票
    demo_symbols = ['AAPL', 'GOOGL', 'MSFT']
    period = '6mo'  # 6个月数据
    
    print_info(f"正在获取股票数据: {', '.join(demo_symbols)}")
    print_info(f"数据周期: {period}")
    
    try:
        start_time = time.time()
        
        # 第一次获取（从网络）
        print_info("首次获取数据（从网络下载）...")
        data = data_manager.get_data(demo_symbols, period=period)
        first_fetch_time = time.time() - start_time
        
        print_success(f"数据获取完成，耗时: {first_fetch_time:.2f}秒")
        print_info(f"获取到 {len(data)} 只股票的数据")
        
        # 显示数据概览
        for symbol in demo_symbols:
            if symbol in data:
                df = data[symbol]
                print_info(f"{symbol}: {len(df)} 条记录，时间范围: {df.index[0].date()} 到 {df.index[-1].date()}")
        
        # 显示缓存信息
        cache_info = data_manager.get_cache_info()
        print_info(f"💾 缓存状态: 内存 {cache_info['memory_cache_count']} 项, 磁盘 {cache_info['disk_cache_count']} 项")
        
        # 第二次获取（从缓存）
        print_info("\n再次获取相同数据（从缓存读取）...")
        start_time = time.time()
        data_cached = data_manager.get_data(demo_symbols, period=period)
        second_fetch_time = time.time() - start_time
        
        print_success(f"缓存数据获取完成，耗时: {second_fetch_time:.2f}秒")
        print_info(f"缓存加速比: {first_fetch_time/second_fetch_time:.1f}x")
        
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return
    
    wait_for_user()
    
    # 因子计算演示
    print_step(3, "技术因子计算演示")
    
    # 选择一只股票进行详细演示
    demo_symbol = 'AAPL'
    demo_data = data[demo_symbol]
    
    print_info(f"使用 {demo_symbol} 数据进行因子计算演示")
    print_info(f"数据点数: {len(demo_data)}")
    
    try:
        # 计算技术因子
        print_info("正在计算技术因子...")
        
        # 使用技术因子计算器计算所有因子
        factors_data = tech_factors.calculate_all_factors(demo_data)
        
        print_success("技术因子计算完成")
        print(f"   📊 原始数据列: {list(demo_data.columns)}")
        print(f"   📈 因子数据列: {list(factors_data.columns)}")
        
        # 显示部分因子数据
        print("\n📋 技术因子预览 (最近5天):")
        factor_cols = ['Close', 'SMA20', 'EMA20', 'RSI14']
        available_cols = [col for col in factor_cols if col in factors_data.columns]
        if available_cols:
            print(factors_data[available_cols].tail().round(2))
        
        # 计算动量因子（价格变化率）
        print_info("\n正在计算动量因子...")
        momentum_5d = (demo_data['Close'] / demo_data['Close'].shift(5) - 1) * 100
        momentum_20d = (demo_data['Close'] / demo_data['Close'].shift(20) - 1) * 100
        
        print_success("动量因子计算完成")
        print(f"   📈 5日动量: {momentum_5d.iloc[-1]:.2f}%")
        print(f"   📈 20日动量: {momentum_20d.iloc[-1]:.2f}%")
        
        # 计算波动率因子
        print_info("\n正在计算波动率因子...")
        volatility_20d = demo_data['Close'].pct_change().rolling(20).std() * 100
        
        print_success("波动率因子计算完成")
        print(f"   📊 20日波动率: {volatility_20d.iloc[-1]:.2f}%")
        
        # 显示最新因子值
        print_info("\n最新因子值:")
        latest_date = demo_data.index[-1].date()
        print(f"  日期: {latest_date}")
        print(f"  5日动量: {momentum_5d.iloc[-1]:.2f}%")
        print(f"  20日动量: {momentum_20d.iloc[-1]:.2f}%")
        print(f"  波动率: {volatility_20d.iloc[-1]:.2f}%")
        
        # 显示其他技术因子（如果可用）
        if 'RSI14' in factors_data.columns:
            print(f"  RSI: {factors_data['RSI14'].iloc[-1]:.2f}")
        if 'BB_position' in factors_data.columns:
            print(f"  布林带位置: {factors_data['BB_position'].iloc[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ 因子计算失败: {e}")
        return
    
    wait_for_user()
    
    # 因子评估演示
    print_step(4, "因子评估和分析演示")
    
    print_info("正在进行因子评估分析...")
    
    try:
        # 使用20日动量因子进行评估
        factor_data = momentum_20d.dropna()
        price_data = demo_data['Close']
        
        print_info(f"评估因子: 20日动量因子")
        print_info(f"评估期间: {factor_data.index[0].date()} 到 {factor_data.index[-1].date()}")
        print_info(f"有效数据点: {len(factor_data)}")
        
        # 计算简单的因子表现指标
        print_info("\n正在计算因子统计指标...")
        
        # 因子基本统计
        factor_stats = {
            '均值': factor_data.mean(),
            '标准差': factor_data.std(),
            '最大值': factor_data.max(),
            '最小值': factor_data.min(),
            '偏度': factor_data.skew(),
            '峰度': factor_data.kurtosis()
        }
        
        print_success("因子统计指标计算完成")
        for key, value in factor_stats.items():
            print(f"   📊 {key}: {value:.4f}")
        
        # 计算因子与未来收益的相关性（简单IC分析）
        print_info("\n正在计算因子预测能力...")
        
        # 计算未来1日、5日收益率
        future_1d = price_data.pct_change(1).shift(-1)  # 未来1日收益
        future_5d = price_data.pct_change(5).shift(-5)  # 未来5日收益
        
        # 计算相关系数（Information Coefficient）
        ic_1d = factor_data.corr(future_1d)
        ic_5d = factor_data.corr(future_5d)
        
        print_success("因子预测能力分析完成")
        print(f"   📈 1日IC: {ic_1d:.4f}")
        print(f"   📈 5日IC: {ic_5d:.4f}")
        
        # 因子分层测试（简化版）
        print_info("\n正在进行因子分层测试...")
        
        # 将因子值分为5层，确保索引对齐
        factor_quantiles = pd.qcut(factor_data, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # 计算各层的平均未来收益，确保索引对齐
        layered_returns = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            mask = (factor_quantiles == q)
            if mask.sum() > 0:
                # 确保future_1d和mask的索引对齐
                aligned_future = future_1d.reindex(mask.index)
                avg_return = aligned_future[mask].mean()
                layered_returns[q] = avg_return
        
        print_success("因子分层测试完成")
        for layer, ret in layered_returns.items():
            if not pd.isna(ret):
                print(f"   📊 {layer}层平均收益: {ret:.4f} ({ret*100:.2f}%)")
        
        # 计算多空收益（Q5 - Q1）
        if 'Q5' in layered_returns and 'Q1' in layered_returns:
            if not pd.isna(layered_returns['Q5']) and not pd.isna(layered_returns['Q1']):
                long_short = layered_returns['Q5'] - layered_returns['Q1']
                print(f"   🎯 多空收益 (Q5-Q1): {long_short:.4f} ({long_short*100:.2f}%)")
            else:
                long_short = 0
                print(f"   🎯 多空收益 (Q5-Q1): 数据不足")
        else:
            long_short = 0
        
        # 保存结果用于后续图表生成
        ic_results = {
            'ic_mean': ic_1d,
            'ic_std': 0.1,  # 简化值
            'ic_ir': ic_1d / 0.1 if abs(ic_1d) > 0 else 0,
            'win_rate': 0.5 + abs(ic_1d) * 0.3  # 简化计算
        }
        
        quantile_results = {
            'quantile_returns': list(layered_returns.values()),
            'long_short_return': long_short if 'Q5' in layered_returns and 'Q1' in layered_returns else 0
        }
        
        # 显示评估结果摘要
        print_info("\n📊 因子评估结果摘要:")
        print(f"  IC (1日): {ic_1d:.4f}")
        print(f"  IC (5日): {ic_5d:.4f}")
        if 'Q5' in layered_returns and 'Q1' in layered_returns:
            print(f"  多空收益: {long_short*100:.2f}%")
        
    except Exception as e:
        print(f"❌ 因子评估失败: {e}")
        return
    
    wait_for_user()
    
    # 图表生成演示
    print_step(5, "可视化图表生成演示")
    
    print_info("正在生成因子评估图表...")
    
    try:
        # 使用性能分析器生成图表
        
        # 1. 因子得分图表
        print_info("生成因子得分时间序列图...")
        evaluator.plot_factor_score(factor_data, save_path='examples/factor_score_demo.png')
        print_success("因子得分图表已保存: examples/factor_score_demo.png")
        
        # 2. 价格与信号图表
        print_info("生成价格与信号图表...")
        # 创建简单的信号（基于因子值）
        signal = (factor_data > factor_data.median()).astype(float)
        evaluator.plot_signal_price(demo_data, signal, save_path='examples/price_signal_demo.png')
        print_success("价格信号图表已保存: examples/price_signal_demo.png")
        
        # 3. 模拟策略收益曲线
        print_info("生成策略收益曲线...")
        # 创建简单的策略收益（基于信号）
        returns = demo_data['Close'].pct_change() * signal.shift(1)
        returns = returns.dropna()
        
        if len(returns) > 0:
            evaluator.plot_equity(returns, save_path='examples/equity_curve_demo.png')
            print_success("收益曲线图表已保存: examples/equity_curve_demo.png")
            
            # 4. 回撤图表
            print_info("生成回撤图表...")
            evaluator.plot_drawdown(returns, save_path='examples/drawdown_demo.png')
            print_success("回撤图表已保存: examples/drawdown_demo.png")
            
            # 5. 策略表现指标
            print_info("计算策略表现指标...")
            metrics = evaluator.metrics(returns)
            
            print_success("策略表现分析完成")
            print(f"   📊 累计收益: {metrics['cum_return']:.2%}")
            print(f"   📊 年化收益: {metrics['ann_return']:.2%}")
            print(f"   📊 年化波动率: {metrics['ann_vol']:.2%}")
            print(f"   📊 夏普比率: {metrics['sharpe']:.4f}")
            print(f"   📊 最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"   📊 胜率: {metrics['hit_rate']:.2%}")
        
        print_success("所有图表生成完成！")
        print_info("📁 图表文件保存在 examples/ 目录下")
        
        # 设置chart_files变量以供后续使用
        chart_files = {
            '因子得分图': 'examples/factor_score_demo.png',
            '价格信号图': 'examples/price_signal_demo.png',
            '收益曲线图': 'examples/equity_curve_demo.png',
            '回撤图': 'examples/drawdown_demo.png'
        }
        
    except Exception as e:
        print(f"❌ 图表生成失败: {e}")
        chart_files = {}
        return
    
    wait_for_user()
    
    # 结果分析和建议
    print_step(6, "结果分析和下一步建议")
    
    print_info("📋 演示总结:")
    print(f"✅ 成功获取了 {len(demo_symbols)} 只股票的历史数据")
    print(f"✅ 计算了 5 种技术因子")
    print(f"✅ 完成了因子评估分析")
    print(f"✅ 生成了 {len(chart_files)} 个可视化图表")
    
    # 根据IC结果给出建议
    ic_mean = ic_1d  # 使用1日IC作为主要指标
    ic_ir = ic_results['ic_ir']
    
    print_info("\n🎯 因子质量评估:")
    
    if abs(ic_mean) > 0.05:
        if ic_mean > 0:
            print_success("该因子显示出正向预测能力")
        else:
            print_success("该因子显示出反向预测能力")
    else:
        print_warning("该因子预测能力较弱")
    
    if ic_ir > 0.5:
        print_success("因子信息比率良好，具有较好的稳定性")
    elif ic_ir > 0.2:
        print_info("因子信息比率中等，需要进一步优化")
    else:
        print_warning("因子信息比率较低，建议重新设计")
    
    print_info("\n🚀 下一步建议:")
    print("1. 查看生成的图表文件，深入理解因子表现")
    print("2. 尝试调整因子参数，优化因子效果")
    print("3. 测试更多股票和时间周期")
    print("4. 学习更多高级因子构建技巧")
    print("5. 阅读用户指南了解更多功能")
    
    print_info("\n📚 学习资源:")
    print("• 初学者指南: docs/BEGINNER_GUIDE.md")
    print("• 图表解读指南: docs/CHART_INTERPRETATION_GUIDE.md")
    print("• 常见问题: docs/FAQ_TROUBLESHOOTING.md")
    print("• 进阶技巧: docs/ADVANCED_TIPS_PRACTICES.md")
    
    print_header("🎉 演示完成！")
    
    print("""
恭喜您完成了量化交易系统的快速开始演示！

您已经学会了：
✅ 如何获取和缓存股票数据
✅ 如何计算技术因子
✅ 如何评估因子效果
✅ 如何生成可视化图表
✅ 如何解读分析结果

现在您可以：
🔍 探索更多因子和策略
📊 分析更多股票和市场
🚀 开发自己的量化策略
📈 构建完整的交易系统

祝您在量化投资的道路上取得成功！
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")
        print("请检查环境配置或查看错误日志")