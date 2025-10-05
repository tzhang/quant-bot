#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序化交易系统
基于投资策略推荐系统的输出，实现自动化交易执行

功能模块：
1. 信号生成器 - 基于策略推荐生成交易信号
2. 风险管理器 - 控制仓位和风险
3. 订单执行器 - 模拟交易执行
4. 监控系统 - 实时监控和报告
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 导入投资策略推荐系统
from examples.investment_strategy_recommendation import InvestmentStrategyRecommendation

class TradingSignalGenerator:
    """交易信号生成器"""
    
    def __init__(self):
        """初始化信号生成器"""
        self.strategy_system = InvestmentStrategyRecommendation()
        self.signals = []
        
    def generate_signals_from_strategy(self, strategy_type='balanced'):
        """
        基于策略推荐生成交易信号
        
        Args:
            strategy_type: 策略类型 ('conservative', 'balanced', 'aggressive')
        
        Returns:
            list: 交易信号列表
        """
        print(f"📊 生成 {strategy_type} 策略的交易信号...")
        
        # 运行策略分析
        self.strategy_system.run_analysis()
        recommendations = self.strategy_system.analysis_results.get('recommendations', {})
        
        if strategy_type not in recommendations:
            print(f"❌ 未找到 {strategy_type} 策略推荐")
            return []
        
        strategy_rec = recommendations[strategy_type]
        recommended_stocks = strategy_rec.get('recommended_stocks', [])
        
        signals = []
        current_time = datetime.now()
        
        for stock in recommended_stocks:
            signal = {
                'timestamp': current_time,
                'symbol': stock['symbol'],
                'name': stock['name'],
                'action': 'BUY',  # 基于推荐，生成买入信号
                'target_weight': stock['weight'] / 100,  # 转换为小数
                'score': stock['score'],
                'sector': stock['sector'],
                'risk_level': stock['risk_level'],
                'confidence': self._calculate_confidence(stock['score']),
                'strategy_type': strategy_type
            }
            signals.append(signal)
        
        self.signals = signals
        print(f"✅ 生成了 {len(signals)} 个交易信号")
        return signals
    
    def _calculate_confidence(self, score):
        """
        根据评分计算信号置信度
        
        Args:
            score: 股票评分
        
        Returns:
            str: 置信度等级
        """
        if score >= 80:
            return 'HIGH'
        elif score >= 60:
            return 'MEDIUM'
        else:
            return 'LOW'

class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position_size=0.15, max_sector_exposure=0.3, max_total_risk=0.8):
        """
        初始化风险管理器
        
        Args:
            max_position_size: 单个股票最大仓位比例
            max_sector_exposure: 单个板块最大暴露比例
            max_total_risk: 最大总风险暴露
        """
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_total_risk = max_total_risk
        
    def validate_signals(self, signals, current_portfolio=None):
        """
        验证交易信号的风险合规性
        
        Args:
            signals: 交易信号列表
            current_portfolio: 当前投资组合
        
        Returns:
            list: 经过风险调整的信号列表
        """
        print("🛡️ 进行风险管理验证...")
        
        if current_portfolio is None:
            current_portfolio = {}
        
        validated_signals = []
        sector_exposure = {}
        total_exposure = 0
        
        for signal in signals:
            # 检查单个仓位大小
            adjusted_weight = min(signal['target_weight'], self.max_position_size)
            
            # 检查板块暴露
            sector = signal['sector']
            current_sector_exposure = sector_exposure.get(sector, 0)
            
            if current_sector_exposure + adjusted_weight > self.max_sector_exposure:
                adjusted_weight = max(0, self.max_sector_exposure - current_sector_exposure)
            
            # 检查总暴露
            if total_exposure + adjusted_weight > self.max_total_risk:
                adjusted_weight = max(0, self.max_total_risk - total_exposure)
            
            if adjusted_weight > 0.01:  # 最小仓位阈值
                signal_copy = signal.copy()
                signal_copy['adjusted_weight'] = adjusted_weight
                signal_copy['risk_adjusted'] = adjusted_weight != signal['target_weight']
                
                validated_signals.append(signal_copy)
                sector_exposure[sector] = current_sector_exposure + adjusted_weight
                total_exposure += adjusted_weight
        
        print(f"✅ 风险验证完成，保留 {len(validated_signals)} 个信号")
        return validated_signals

class OrderExecutor:
    """订单执行器（模拟）"""
    
    def __init__(self, initial_capital=100000):
        """
        初始化订单执行器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio = {}
        self.transaction_history = []
        self.execution_log = []
        
    def execute_signals(self, signals, market_data=None):
        """
        执行交易信号
        
        Args:
            signals: 交易信号列表
            market_data: 市场数据（用于获取当前价格）
        
        Returns:
            dict: 执行结果
        """
        print("💼 执行交易信号...")
        
        execution_results = {
            'successful_orders': 0,
            'failed_orders': 0,
            'total_invested': 0,
            'orders': []
        }
        
        for signal in signals:
            try:
                # 模拟获取当前价格（实际应用中需要连接实时数据源）
                current_price = self._get_mock_price(signal['symbol'])
                
                # 计算投资金额
                target_amount = self.current_capital * signal['adjusted_weight']
                shares = int(target_amount / current_price)
                actual_amount = shares * current_price
                
                if shares > 0 and actual_amount <= self.current_capital:
                    # 执行买入
                    order = {
                        'timestamp': datetime.now(),
                        'symbol': signal['symbol'],
                        'action': signal['action'],
                        'shares': shares,
                        'price': current_price,
                        'amount': actual_amount,
                        'status': 'FILLED'
                    }
                    
                    # 更新投资组合
                    if signal['symbol'] in self.portfolio:
                        self.portfolio[signal['symbol']]['shares'] += shares
                        self.portfolio[signal['symbol']]['total_cost'] += actual_amount
                    else:
                        self.portfolio[signal['symbol']] = {
                            'shares': shares,
                            'avg_price': current_price,
                            'total_cost': actual_amount,
                            'sector': signal['sector']
                        }
                    
                    # 更新资金
                    self.current_capital -= actual_amount
                    
                    # 记录交易
                    self.transaction_history.append(order)
                    execution_results['orders'].append(order)
                    execution_results['successful_orders'] += 1
                    execution_results['total_invested'] += actual_amount
                    
                    print(f"✅ {signal['symbol']}: 买入 {shares} 股，价格 ${current_price:.2f}")
                    
                else:
                    print(f"❌ {signal['symbol']}: 资金不足或股数为0")
                    execution_results['failed_orders'] += 1
                    
            except Exception as e:
                print(f"❌ {signal['symbol']}: 执行失败 - {str(e)}")
                execution_results['failed_orders'] += 1
        
        print(f"📊 执行完成: {execution_results['successful_orders']} 成功, {execution_results['failed_orders']} 失败")
        return execution_results
    
    def _get_mock_price(self, symbol):
        """
        获取模拟价格（实际应用中应连接实时数据源）
        
        Args:
            symbol: 股票代码
        
        Returns:
            float: 模拟价格
        """
        # 模拟价格，实际应用中需要连接实时数据API
        mock_prices = {
            'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 'AMZN': 3200.0,
            'TSLA': 800.0, 'META': 250.0, 'NVDA': 400.0, 'NFLX': 400.0,
            'JPM': 140.0, 'BAC': 35.0, 'WMT': 140.0, 'PG': 150.0,
            'JNJ': 160.0, 'UNH': 450.0, 'HD': 320.0, 'V': 220.0
        }
        return mock_prices.get(symbol, 100.0)  # 默认价格
    
    def get_portfolio_summary(self):
        """
        获取投资组合摘要
        
        Returns:
            dict: 投资组合摘要
        """
        total_value = self.current_capital
        portfolio_details = []
        
        for symbol, position in self.portfolio.items():
            current_price = self._get_mock_price(symbol)
            market_value = position['shares'] * current_price
            total_value += market_value
            
            pnl = market_value - position['total_cost']
            pnl_pct = (pnl / position['total_cost']) * 100 if position['total_cost'] > 0 else 0
            
            portfolio_details.append({
                'symbol': symbol,
                'shares': position['shares'],
                'avg_price': position['avg_price'],
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': position['total_cost'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'weight': (market_value / total_value) * 100,
                'sector': position['sector']
            })
        
        return {
            'total_value': total_value,
            'cash': self.current_capital,
            'invested_value': total_value - self.current_capital,
            'total_return': total_value - self.initial_capital,
            'total_return_pct': ((total_value - self.initial_capital) / self.initial_capital) * 100,
            'positions': portfolio_details
        }

class TradingMonitor:
    """交易监控系统"""
    
    def __init__(self):
        """初始化监控系统"""
        self.monitoring_data = []
        
    def log_execution(self, signals, execution_results, portfolio_summary):
        """
        记录执行日志
        
        Args:
            signals: 交易信号
            execution_results: 执行结果
            portfolio_summary: 投资组合摘要
        """
        log_entry = {
            'timestamp': datetime.now(),
            'signals_count': len(signals),
            'successful_orders': execution_results['successful_orders'],
            'failed_orders': execution_results['failed_orders'],
            'total_invested': execution_results['total_invested'],
            'portfolio_value': portfolio_summary['total_value'],
            'cash_remaining': portfolio_summary['cash'],
            'total_return_pct': portfolio_summary['total_return_pct']
        }
        
        self.monitoring_data.append(log_entry)
        
    def generate_report(self, portfolio_summary):
        """
        生成监控报告
        
        Args:
            portfolio_summary: 投资组合摘要
        """
        print("\n" + "="*60)
        print("📊 程序化交易系统监控报告")
        print("="*60)
        
        print(f"💰 投资组合总价值: ${portfolio_summary['total_value']:,.2f}")
        print(f"💵 现金余额: ${portfolio_summary['cash']:,.2f}")
        print(f"📈 总收益: ${portfolio_summary['total_return']:,.2f} ({portfolio_summary['total_return_pct']:.2f}%)")
        
        print(f"\n📋 持仓明细 ({len(portfolio_summary['positions'])} 只股票):")
        print("-" * 80)
        print(f"{'股票':<8} {'股数':<8} {'成本':<10} {'市值':<12} {'收益率':<10} {'权重':<8}")
        print("-" * 80)
        
        for pos in portfolio_summary['positions']:
            print(f"{pos['symbol']:<8} {pos['shares']:<8} ${pos['cost_basis']:<9.0f} "
                  f"${pos['market_value']:<11.0f} {pos['pnl_pct']:<9.1f}% {pos['weight']:<7.1f}%")
        
        print("\n" + "="*60)

class AutomatedTradingSystem:
    """程序化交易系统主类"""
    
    def __init__(self, initial_capital=100000, strategy_type='balanced'):
        """
        初始化程序化交易系统
        
        Args:
            initial_capital: 初始资金
            strategy_type: 策略类型
        """
        self.signal_generator = TradingSignalGenerator()
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor(initial_capital)
        self.monitor = TradingMonitor()
        self.strategy_type = strategy_type
        
    def run_trading_cycle(self):
        """
        运行一个完整的交易周期
        
        Returns:
            dict: 交易周期结果
        """
        print("🚀 启动程序化交易系统...")
        print(f"📊 策略类型: {self.strategy_type}")
        print(f"💰 初始资金: ${self.order_executor.initial_capital:,.2f}")
        
        try:
            # 1. 生成交易信号
            signals = self.signal_generator.generate_signals_from_strategy(self.strategy_type)
            
            if not signals:
                print("❌ 未生成任何交易信号")
                return None
            
            # 2. 风险管理验证
            validated_signals = self.risk_manager.validate_signals(signals)
            
            if not validated_signals:
                print("❌ 所有信号都被风险管理系统拒绝")
                return None
            
            # 3. 执行交易
            execution_results = self.order_executor.execute_signals(validated_signals)
            
            # 4. 获取投资组合摘要
            portfolio_summary = self.order_executor.get_portfolio_summary()
            
            # 5. 记录监控日志
            self.monitor.log_execution(validated_signals, execution_results, portfolio_summary)
            
            # 6. 生成报告
            self.monitor.generate_report(portfolio_summary)
            
            return {
                'signals': validated_signals,
                'execution_results': execution_results,
                'portfolio_summary': portfolio_summary
            }
            
        except Exception as e:
            print(f"❌ 交易周期执行失败: {str(e)}")
            return None
    
    def save_results(self, results, filename=None):
        """
        保存交易结果
        
        Args:
            results: 交易结果
            filename: 保存文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_results_{timestamp}.json"
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        # 转换datetime对象为字符串以便JSON序列化
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=convert_datetime)
        
        print(f"💾 交易结果已保存到: {filepath}")

def main():
    """主函数"""
    print("🤖 程序化交易系统启动")
    
    # 创建交易系统实例
    trading_system = AutomatedTradingSystem(
        initial_capital=100000,
        strategy_type='balanced'  # 可选: 'conservative', 'balanced', 'aggressive'
    )
    
    # 运行交易周期
    results = trading_system.run_trading_cycle()
    
    if results:
        # 保存结果
        trading_system.save_results(results)
        print("\n✅ 程序化交易系统运行完成")
    else:
        print("\n❌ 程序化交易系统运行失败")

if __name__ == "__main__":
    main()