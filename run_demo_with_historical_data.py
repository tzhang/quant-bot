#!/usr/bin/env python3
"""
使用Qlib历史数据的演示脚本
由于实时数据源存在限制，使用2020年的历史数据进行演示
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_adapter import DataAdapter
from src.strategies.templates import MomentumStrategy
from src.backtest.engine import BacktestEngine
from src.performance.analyzer import PerformanceAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_historical_demo():
    """运行历史数据演示"""
    print("🚀 启动量化交易系统历史数据演示")
    print("=" * 60)
    
    # 使用Qlib可用的历史数据范围
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    start_date = '2020-01-02'  # Qlib可用数据开始日期
    end_date = '2020-01-31'    # 使用一个月的数据进行演示
    
    print(f"📊 交易标的: {', '.join(symbols)}")
    print(f"📅 数据范围: {start_date} 至 {end_date}")
    print(f"🔄 策略类型: 动量策略")
    print()
    
    try:
        # 1. 初始化数据适配器
        print("1️⃣ 初始化数据适配器...")
        data_adapter = DataAdapter()
        
        # 2. 获取历史数据
        print("2️⃣ 获取历史数据...")
        all_data = {}
        
        for symbol in symbols:
            print(f"   获取 {symbol} 数据...")
            try:
                data = data_adapter.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if data is not None and not data.empty:
                    all_data[symbol] = data
                    print(f"   ✅ {symbol}: {len(data)} 条记录")
                else:
                    print(f"   ❌ {symbol}: 无数据")
            except Exception as e:
                print(f"   ❌ {symbol}: 获取失败 - {str(e)}")
        
        if not all_data:
            print("❌ 未能获取任何股票数据，演示终止")
            return
        
        print(f"\n✅ 成功获取 {len(all_data)} 只股票的数据")
        
        # 3. 初始化策略
        print("\n3️⃣ 初始化动量策略...")
        strategy = MomentumStrategy(
            fast=12,  # 快速EMA周期
            slow=26   # 慢速EMA周期
        )
        
        # 4. 运行回测
        print("4️⃣ 运行策略回测...")
        backtest_engine = BacktestEngine(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
            min_periods=10  # 降低最小数据点要求以适应历史数据
        )
        
        # 运行回测 - 为每个股票单独回测
        print("4️⃣ 运行策略回测...")
        all_results = {}
        
        for symbol, data in all_data.items():
            print(f"   回测 {symbol}...")
            try:
                # 标准化列名（将小写转换为大写）
                data_normalized = data.copy()
                data_normalized.columns = [col.capitalize() for col in data_normalized.columns]
                
                # 生成交易信号
                signals = strategy.generate_signal(data_normalized)
                
                # 运行回测
                result = backtest_engine.run(data_normalized, signals)
                all_results[symbol] = result
                
                # 计算夏普比率
                returns = result['returns']
                if len(returns) > 0 and returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
                
                print(f"   ✅ {symbol}: 总收益 {result['total_return']:.2f}%, 夏普比率 {sharpe_ratio:.2f}")
            except Exception as e:
                print(f"   ❌ {symbol} 回测失败: {e}")
                continue
        
        # 5. 分析结果
        print("\n5️⃣ 分析回测结果...")
        
        if all_results:
            total_return = 0
            total_trades = 0
            successful_trades = 0
            
            print("\n📈 回测结果汇总:")
            print("-" * 50)
            
            for symbol, result in all_results.items():
                if isinstance(result, dict):
                    if result and 'portfolio_value' in result:
                        portfolio_values = result['portfolio_value']
                        if len(portfolio_values) > 1:
                            symbol_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
                            total_return += symbol_return
                            
                            trades = result.get('trades', pd.DataFrame())
                            if isinstance(trades, pd.DataFrame) and not trades.empty:
                                total_trades += len(trades)
                                # 假设有pnl列，否则跳过
                                if 'pnl' in trades.columns:
                                    successful_trades += len(trades[trades['pnl'] > 0])
                            elif isinstance(trades, list):
                                total_trades += len(trades)
                                successful_trades += len([t for t in trades if t.get('pnl', 0) > 0])
                            
                            print(f"{symbol:>6}: 收益率 {symbol_return:>6.2f}%, 交易次数 {len(trades) if isinstance(trades, list) else len(trades) if isinstance(trades, pd.DataFrame) else 0:>3}")
            
            print("-" * 50)
            avg_return = total_return / len(all_results) if all_results else 0
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            print(f"平均收益率: {avg_return:>6.2f}%")
            print(f"总交易次数: {total_trades:>6}")
            print(f"胜率: {win_rate:>6.2f}%")
        else:
            print("\n❌ 没有成功的回测结果")
        
        # 6. 生成模拟实时数据用于监控面板
        print("\n6️⃣ 生成模拟实时数据...")
        generate_mock_realtime_data(all_data)
        
        print("\n✅ 历史数据演示完成!")
        print("\n💡 提示:")
        print("   - 监控面板现在应该显示基于历史数据的模拟实时数据")
        print("   - 可以访问 http://localhost:8080 查看监控面板")
        print("   - 实际生产环境中需要配置实时数据源")
        
    except Exception as e:
        logger.error(f"演示运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_mock_realtime_data(historical_data):
    """基于历史数据生成模拟实时数据"""
    print("   生成模拟市场数据...")
    
    # 创建模拟数据文件供监控面板使用
    mock_data = {
        'timestamp': datetime.now().isoformat(),
        'market_data': [],
        'trading_metrics': {
            'total_trades': 15,
            'successful_trades': 9,
            'win_rate': 60.0,
            'total_pnl': 2500.0,
            'portfolio_value': 102500.0,
            'active_positions': 3
        },
        'positions': []
    }
    
    # 基于历史数据生成当前价格
    for symbol, data in historical_data.items():
        if not data.empty:
            last_price = float(data['close'].iloc[-1])
            # 添加小幅随机波动模拟实时价格变化
            import random
            current_price = last_price * (1 + random.uniform(-0.02, 0.02))
            change = current_price - last_price
            change_percent = (change / last_price) * 100
            
            mock_data['market_data'].append({
                'symbol': symbol,
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'volume': random.randint(1000000, 5000000),
                'timestamp': datetime.now().isoformat()
            })
            
            # 添加模拟持仓
            if random.random() > 0.5:  # 50%概率持有该股票
                quantity = random.randint(10, 100)
                avg_cost = last_price * random.uniform(0.95, 1.05)
                market_value = current_price * quantity
                unrealized_pnl = (current_price - avg_cost) * quantity
                
                mock_data['positions'].append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_cost': round(avg_cost, 2),
                    'current_price': round(current_price, 2),
                    'market_value': round(market_value, 2),
                    'unrealized_pnl': round(unrealized_pnl, 2),
                    'unrealized_pnl_percent': round((unrealized_pnl / (avg_cost * quantity)) * 100, 2),
                    'timestamp': datetime.now().isoformat()
                })
    
    # 保存模拟数据到文件
    import json
    with open('mock_realtime_data.json', 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"   ✅ 已生成 {len(mock_data['market_data'])} 只股票的模拟实时数据")
    print(f"   ✅ 已生成 {len(mock_data['positions'])} 个模拟持仓")

if __name__ == "__main__":
    run_historical_demo()