#!/usr/bin/env python3
"""
自动化交易系统启动脚本
提供简单的命令行界面来启动和管理交易系统
"""

import sys
import os
import time
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.trading.automated_trading_main import AutomatedTradingSystem

def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    自动化交易系统 v1.0                        ║
║                  Automated Trading System                    ║
╠══════════════════════════════════════════════════════════════╣
║  功能特性:                                                    ║
║  • 多策略信号聚合                                             ║
║  • 智能风险管理                                               ║
║  • 实时监控和报告                                             ║
║  • 模拟/实盘交易支持                                          ║
║  • 灵活的配置管理                                             ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def interactive_setup():
    """交互式设置"""
    print("\n=== 交互式配置 ===")
    
    # 交易模式
    print("\n1. 选择交易模式:")
    print("   [1] 模拟交易 (Paper Trading)")
    print("   [2] 实盘交易 (Live Trading)")
    print("   [3] 回测模式 (Backtesting)")
    
    while True:
        choice = input("请选择 (1-3): ").strip()
        if choice == '1':
            mode = 'paper'
            break
        elif choice == '2':
            mode = 'live'
            print("⚠️  警告: 您选择了实盘交易模式，请确保已充分测试策略！")
            confirm = input("确认使用实盘交易? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                break
            else:
                continue
        elif choice == '3':
            mode = 'backtest'
            break
        else:
            print("无效选择，请重新输入")
    
    # 初始资金
    while True:
        try:
            capital = input(f"\n2. 初始资金 (默认: $100,000): ").strip()
            if not capital:
                capital = 100000.0
            else:
                capital = float(capital)
            break
        except ValueError:
            print("请输入有效的数字")
    
    # 交易标的
    print(f"\n3. 交易标的 (默认: AAPL, GOOGL, MSFT, TSLA, AMZN)")
    symbols_input = input("输入股票代码 (用空格分隔，回车使用默认): ").strip()
    if symbols_input:
        symbols = symbols_input.upper().split()
    else:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    return {
        'mode': mode,
        'capital': capital,
        'symbols': symbols
    }

def quick_start():
    """快速启动"""
    print("\n=== 快速启动模式 ===")
    print("使用配置文件中的设置启动交易...")
    
    return {
        # 不覆盖配置文件中的模式设置
        'capital': 100000.0,
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    }

def show_status(system):
    """显示系统状态"""
    try:
        portfolio = system.trading_engine.get_portfolio_summary()
        
        print("\n" + "="*60)
        print("                    系统状态")
        print("="*60)
        print(f"运行状态: {'运行中' if system.running else '已停止'}")
        print(f"交易模式: {system.config.get('trading.mode', 'unknown').upper()}")
        
        # 显示IB连接状态
        ib_status = _get_ib_connection_status(system)
        print(f"IB连接状态: {ib_status}")
        
        print(f"总权益: ${portfolio['total_equity']:,.2f}")
        print(f"可用资金: ${portfolio['available_cash']:,.2f}")
        print(f"总盈亏: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:.2%})")
        print(f"持仓数量: {portfolio['positions_count']}")
        print(f"交易次数: {system.trade_count}")
        print(f"最大回撤: {portfolio['max_drawdown']:.2%}")
        print("="*60)
        
        # 显示持仓详情
        if portfolio['positions_count'] > 0:
            print("\n持仓详情:")
            for symbol, position in system.trading_engine.positions.items():
                pnl_color = "📈" if position.unrealized_pnl >= 0 else "📉"
                print(f"  {symbol}: {position.quantity} 股 @ ${position.avg_cost:.2f} "
                      f"{pnl_color} ${position.unrealized_pnl:.2f}")
        
    except Exception as e:
        print(f"获取状态失败: {e}")

def _get_ib_connection_status(system):
    """获取IB连接状态信息"""
    try:
        # 检查是否使用IB数据源
        config = system.config
        primary_source = config.get('data_sources.primary', 'yahoo')
        backup_source = config.get('data_sources.backup', 'alpha_vantage')
        
        # 检查IB是否在数据源配置中
        using_ib = primary_source == 'ib' or backup_source == 'ib'
        
        if not using_ib:
            return f"⏭️ 未使用 (当前数据源: {primary_source})"
        
        # 尝试获取IB连接状态
        ib_status = "❓ 未知"
        ib_host = "N/A"
        ib_port = "N/A"
        
        # 从数据提供者获取IB状态
        if hasattr(system, 'data_provider'):
            try:
                # 尝试创建IB提供者来检查连接状态
                from src.data.ib_data_provider import create_ib_provider
                ib_provider = create_ib_provider()
                
                if ib_provider and ib_provider.is_available:
                    # 获取连接信息
                    data_info = ib_provider.get_data_info()
                    connection_status = data_info.get('connection_status', False)
                    
                    if connection_status:
                        ib_status = "✅ 已连接"
                    else:
                        ib_status = "❌ 未连接"
                    
                    # 获取连接配置
                    ib_config = config.get('data_sources.api_keys.ib', {})
                    ib_host = ib_config.get('host', '127.0.0.1')
                    ib_port = ib_config.get('port', '4001')
                else:
                    ib_status = "❌ 不可用"
                    
            except Exception as e:
                ib_status = f"❌ 错误: {str(e)[:30]}..."
        
        # 检查是否为主要数据源
        source_type = "主要" if primary_source == 'ib' else "备用"
        return f"{ib_status} ({source_type}数据源, {ib_host}:{ib_port})"
        
    except Exception as e:
        return f"❌ 获取状态失败: {str(e)[:30]}..."

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动化交易系统启动器')
    parser.add_argument('--config', '-c', default='trading_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', '-m', choices=['paper', 'live', 'backtest'],
                       help='交易模式')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='交互式配置')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='快速启动')
    parser.add_argument('--symbols', '-s', nargs='+',
                       help='交易标的列表')
    parser.add_argument('--capital', '-k', type=float,
                       help='初始资金')
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    try:
        # 配置参数
        config_params = {}
        
        if args.interactive:
            config_params = interactive_setup()
        elif args.quick:
            config_params = quick_start()
        else:
            # 使用命令行参数
            if args.mode:
                config_params['mode'] = args.mode
            if args.symbols:
                config_params['symbols'] = args.symbols
            if args.capital:
                config_params['capital'] = args.capital
        
        # 创建交易系统
        print(f"\n正在初始化交易系统...")
        system = AutomatedTradingSystem(args.config)
        
        # 应用配置参数
        if 'mode' in config_params:
            system.config.set('trading.mode', config_params['mode'])
        if 'symbols' in config_params:
            system.config.set('trading.symbols', config_params['symbols'])
        if 'capital' in config_params:
            system.config.set('trading.initial_capital', config_params['capital'])
        
        # 显示配置信息
        print(f"\n配置信息:")
        print(f"  交易模式: {system.config.get('trading.mode', 'paper').upper()}")
        print(f"  初始资金: ${system.config.get('trading.initial_capital', 100000):,.2f}")
        print(f"  交易标的: {', '.join(system.config.get('trading.symbols', []))}")
        print(f"  配置文件: {args.config}")
        
        # 启动确认
        if not args.quick:
            input("\n按回车键启动交易系统...")
        
        # 启动系统
        print("\n🚀 启动交易系统...")
        system.start()
        
        print("\n✅ 交易系统启动成功！")
        print("\n可用命令:")
        print("  's' - 显示系统状态")
        print("  'r' - 生成性能报告")
        print("  'q' - 退出系统")
        print("  'h' - 显示帮助")
        
        # 主循环
        try:
            while system.running:
                try:
                    command = input("\n> ").strip().lower()
                    
                    if command == 'q' or command == 'quit':
                        break
                    elif command == 's' or command == 'status':
                        show_status(system)
                    elif command == 'r' or command == 'report':
                        system._generate_performance_report()
                        print("性能报告已生成")
                    elif command == 'h' or command == 'help':
                        print("\n可用命令:")
                        print("  's' - 显示系统状态")
                        print("  'r' - 生成性能报告")
                        print("  'q' - 退出系统")
                        print("  'h' - 显示帮助")
                    elif command == '':
                        continue
                    else:
                        print(f"未知命令: {command}")
                        
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n收到中断信号...")
        
        print("\n🛑 正在停止交易系统...")
        system.stop()
        print("✅ 交易系统已安全停止")
        
    except Exception as e:
        print(f"\n❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())