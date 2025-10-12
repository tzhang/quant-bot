#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易系统 - 主入口程序

统一的主流程，整合所有核心功能模块
提供简单易用的接口和清晰的使用流程
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入核心模块
from src import (
    DataManager,
    FactorEngine,
    BacktestEngine,
    RiskManager,
    PerformanceAnalyzer,
    MomentumStrategy,
    MeanReversionStrategy,
    get_version,
    SYSTEM_INFO
)

# 导入工具模块
from src.utils.logger import LoggerManager
from src.core.utils import ConfigManager

# 导入实时交易相关模块
try:
    from src.trading import IBTradingManager, TradeOrder, TradingSignal
    from src.strategies.live_strategy import LiveTradingStrategy, StrategyConfig, StrategyManager
    IB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ IB交易模块导入失败: {e}")
    IB_AVAILABLE = False

class QuantTradingSystem:
    """量化交易系统主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化系统"""
        self.config = ConfigManager(config_path) if config_path else ConfigManager()
        
        # 初始化日志系统
        self.logger_manager = LoggerManager()
        self.logger_manager.load_config()
        self.logger = self.logger_manager.get_logger('main')
        
        # 核心组件
        self.data_manager = None
        self.factor_engine = None
        self.backtest_engine = None
        self.risk_manager = None
        self.performance_analyzer = None
        
        # 实时交易组件
        self.ib_trading_manager = None
        self.trading_enabled = False
        
        # 系统状态
        self.initialized = False
        
        self.logger.info(f"量化交易系统初始化 - 版本: {get_version()}")
    
    def initialize(self, cache_dir: str = "data_cache", initial_capital: float = 100000.0, 
                  enable_trading: bool = False, trading_config: Optional[Dict] = None):
        """初始化所有核心组件"""
        try:
            self.logger.info("正在初始化核心组件...")
            
            # 初始化数据管理器
            self.data_manager = DataManager(cache_dir=cache_dir)
            self.logger.info("✓ 数据管理器初始化完成")
            
            # 初始化因子引擎
            self.factor_engine = FactorEngine()
            self.logger.info("✓ 因子引擎初始化完成")
            
            # 初始化回测引擎
            self.backtest_engine = BacktestEngine(initial_capital=initial_capital)
            self.logger.info("✓ 回测引擎初始化完成")
            
            # 初始化风险管理器
            self.risk_manager = RiskManager()
            self.logger.info("✓ 风险管理器初始化完成")
            
            # 初始化性能分析器
            self.performance_analyzer = PerformanceAnalyzer()
            self.logger.info("✓ 性能分析器初始化完成")
            
            # 初始化IB实时交易组件（主要交易系统）
            if enable_trading and IB_AVAILABLE and trading_config:
                try:
                    self.ib_trading_manager = IBTradingManager(trading_config)
                    self.trading_enabled = True
                    
                    # 初始化策略管理器
                    self.strategy_manager = StrategyManager()
                    
                    self.logger.info("✅ Interactive Brokers (IB) 交易系统初始化完成")
                except Exception as e:
                    self.logger.error(f"❌ IB交易系统初始化失败: {e}")
                    self.logger.error("💡 请确保IB TWS或Gateway已启动并配置正确")
                    self.ib_trading_manager = None
                    self.trading_enabled = False
                    self.strategy_manager = None
            else:
                self.ib_trading_manager = None
                self.trading_enabled = False
                self.strategy_manager = None
                if enable_trading and not IB_AVAILABLE:
                    self.logger.warning("⚠️ 实时交易功能不可用，IB模块未正确安装")
                    self.logger.warning("💡 请安装ib_insync: pip install ib_insync")
            
            self.initialized = True
            self.logger.info("🎉 所有核心组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise
    
    def get_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """获取股票数据"""
        if not self.initialized:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        self.logger.info(f"获取数据: {symbols}, {start_date} - {end_date}")
        
        data = {}
        for symbol in symbols:
            try:
                stock_data = self.data_manager.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                data[symbol] = stock_data
                self.logger.info(f"✓ {symbol} 数据获取成功: {len(stock_data)} 条记录")
            except Exception as e:
                self.logger.error(f"✗ {symbol} 数据获取失败: {e}")
        
        return data
    
    def calculate_factors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """计算技术因子"""
        if not self.initialized:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        self.logger.info("开始计算技术因子...")
        
        factors = {}
        for symbol, stock_data in data.items():
            try:
                symbol_factors = self.factor_engine.compute_technical(stock_data)
                factors[symbol] = symbol_factors
                self.logger.info(f"✓ {symbol} 因子计算完成")
            except Exception as e:
                self.logger.error(f"✗ {symbol} 因子计算失败: {e}")
        
        return factors
    
    def run_backtest(self, strategy_name: str, symbols: List[str], 
                    start_date: str, end_date: str, **kwargs) -> Dict[str, Any]:
        """运行回测"""
        if not self.initialized:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        self.logger.info(f"开始回测: {strategy_name}")
        
        # 获取数据
        data = self.get_data(symbols, start_date, end_date)
        
        # 创建策略实例（使用正确的参数）
        if strategy_name == "momentum":
            strategy = MomentumStrategy(fast=kwargs.get('fast', 12), slow=kwargs.get('slow', 26))
        elif strategy_name == "mean_reversion":
            strategy = MeanReversionStrategy(**kwargs)
        else:
            raise ValueError(f"未知策略: {strategy_name}")
        
        # 生成交易信号
        signals = {}
        for symbol in symbols:
            if symbol in data:
                signal_series = strategy.signal(data[symbol])
                signals[symbol] = signal_series
        
        # 运行回测
        results = {}
        for symbol in symbols:
            if symbol in signals:
                result = self.backtest_engine.run(data[symbol], signals[symbol])
                results[symbol] = result
        
        self.logger.info("✓ 回测完成")
        return results
    
    def analyze_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析回测性能"""
        if not self.initialized:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        # 分析性能
        self.logger.info("开始性能分析...")
        performance_metrics = self.performance_analyzer.metrics(backtest_results)
        
        self.logger.info("✓ 性能分析完成")
        return performance_metrics
    
    def assess_risk(self, portfolio_returns: Any) -> Dict[str, Any]:
        """评估风险"""
        if not self.initialized:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        self.logger.info("开始风险评估...")
        
        risk_metrics = self.risk_manager.calculate_risk_metrics(portfolio_returns)
        
        self.logger.info("✓ 风险评估完成")
        return risk_metrics
    
    def quick_start_demo(self):
        """快速开始演示"""
        self.logger.info("🚀 开始快速演示...")
        
        # 演示参数
        symbols = ["AAPL", "GOOGL", "MSFT"]
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        try:
            # 1. 获取数据
            self.logger.info("1️⃣ 获取股票数据...")
            data = self.get_data(symbols, start_date, end_date)
            
            # 2. 计算因子
            self.logger.info("2️⃣ 计算技术因子...")
            factors = self.calculate_factors(data)
            
            # 3. 运行回测
            self.logger.info("3️⃣ 运行动量策略回测...")
            backtest_results = self.run_backtest(
                strategy_name="momentum",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                lookback_period=20
            )
            
            # 4. 性能分析
            self.logger.info("4️⃣ 分析策略性能...")
            performance = self.analyze_performance(backtest_results)
            
            # 5. 风险评估
            self.logger.info("5️⃣ 评估投资风险...")
            if 'returns' in backtest_results:
                risk_metrics = self.assess_risk(backtest_results['returns'])
            
            self.logger.info("🎉 快速演示完成！")
            
            # 输出结果摘要
            self.print_summary(performance if 'performance' in locals() else {})
            
        except Exception as e:
            self.logger.error(f"演示过程中出现错误: {e}")
            raise
    
    def print_summary(self, performance: Dict[str, Any]):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("📊 量化交易系统 - 结果摘要")
        print("="*60)
        print(f"系统版本: {get_version()}")
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if performance:
            print("\n📈 性能指标:")
            for key, value in performance.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n✅ 系统运行正常")
        print("="*60)
    
    def print_system_info(self):
        """打印系统信息"""
        print("\n" + "="*60)
        print("🔧 系统信息")
        print("="*60)
        for key, value in SYSTEM_INFO.items():
            if key == "modules":
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="量化交易系统")
    parser.add_argument("--demo", action="store_true", help="运行快速演示")
    parser.add_argument("--info", action="store_true", help="显示系统信息")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="数据缓存目录")
    parser.add_argument("--capital", type=float, default=100000.0, help="初始资金")
    
    # 实时交易参数
    parser.add_argument("--trading", action="store_true", help="启用实时交易功能")
    parser.add_argument("--paper", action="store_true", help="使用模拟交易模式")
    parser.add_argument("--ib-host", type=str, default="127.0.0.1", help="IB TWS主机地址")
    parser.add_argument("--ib-port", type=int, help="IB TWS端口 (模拟:7497, 实盘:7496)")
    parser.add_argument("--client-id", type=int, default=1, help="IB客户端ID")
    
    # 策略参数
    parser.add_argument("--strategy", type=str, choices=["momentum", "mean_reversion"], 
                       help="选择交易策略")
    parser.add_argument("--symbols", type=str, nargs="+", default=["AAPL", "GOOGL", "MSFT"],
                       help="交易标的列表")
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = QuantTradingSystem(config_path=args.config)
    
    try:
        # 配置实时交易参数
        trading_config = None
        if args.trading:
            # 确定端口
            if args.ib_port:
                port = args.ib_port
            else:
                port = 7497 if args.paper else 7496
            
            trading_config = {
                'host': args.ib_host,
                'paper_port': 7497,
                'live_port': 7496,
                'client_id': args.client_id,
                'paper_trading': args.paper,
                'risk_limits': {
                    'max_position_value': 50000.0,
                    'max_daily_loss': 5000.0,
                    'max_symbol_exposure': 20000.0,
                    'max_daily_trades': 100,
                    'stop_loss_pct': 0.05,
                    'take_profit_pct': 0.10
                }
            }
        
        # 初始化系统
        system.initialize(
            cache_dir=args.cache_dir,
            initial_capital=args.capital,
            enable_trading=args.trading,
            trading_config=trading_config
        )
        
        # 处理命令行参数
        if args.info:
            system.print_system_info()
            if args.trading and system.trading_enabled:
                print("\n🔗 实时交易功能:")
                print(f"  模式: {'模拟交易' if args.paper else '实盘交易'}")
                print(f"  主机: {args.ib_host}")
                print(f"  端口: {trading_config.get('paper_port' if args.paper else 'live_port')}")
                print(f"  客户端ID: {args.client_id}")
                
        elif args.demo:
            system.quick_start_demo()
            
        elif args.trading and args.strategy:
            # 启动实时交易
            start_live_trading(system, args)
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
        if system.trading_enabled and system.ib_trading_manager:
            print("正在断开IB连接...")
            system.ib_trading_manager.disconnect()
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        if system.trading_enabled and system.ib_trading_manager:
            system.ib_trading_manager.disconnect()
        raise


def start_live_trading(system: QuantTradingSystem, args):
    """启动实时交易"""
    if not system.trading_enabled:
        print("❌ 实时交易功能未启用")
        return
    
    print("\n🚀 启动实时交易系统...")
    print(f"策略: {args.strategy}")
    print(f"标的: {', '.join(args.symbols)}")
    print(f"模式: {'模拟交易' if args.paper else '实盘交易'}")
    
    try:
        # 连接到IB
        if not system.ib_trading_manager.connect():
            print("❌ 无法连接到IB TWS")
            return
        
        # 初始化策略
        strategy_config = StrategyConfig(
            strategy_type=args.strategy,
            lookback_period=20,
            signal_threshold=0.02,
            position_size=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            max_positions=5
        )
        
        strategy = LiveTradingStrategy(strategy_config)
        system.strategy_manager.add_strategy(f"{args.strategy}_strategy", strategy)
        
        # 订阅市场数据
        system.ib_trading_manager.subscribe_market_data(args.symbols)
        
        # 设置回调函数
        def on_price_update(symbol, price, timestamp):
            """价格更新回调"""
            # 更新策略价格数据
            system.strategy_manager.update_prices({symbol: price})
            
            # 生成交易信号
            signals = system.strategy_manager.generate_signals([symbol])
            
            if signals:
                # 获取账户信息
                account_summary = system.ib_trading_manager.get_account_summary()
                account_value = account_summary.get('NetLiquidation', 100000.0)
                
                # 获取当前持仓
                positions = system.ib_trading_manager.get_positions()
                current_positions = {pos.symbol: pos.quantity for pos in positions}
                
                # 创建订单
                orders = system.strategy_manager.create_orders(signals, account_value, current_positions)
                
                # 执行订单
                for order in orders:
                    print(f"📋 执行订单: {order.action} {order.quantity} {order.symbol} @ {order.price}")
                    order_id = system.ib_trading_manager.place_order(order)
                    if order_id:
                        print(f"✅ 订单已提交: ID={order_id}")
        
        def on_order_filled(order_id, status, filled_qty, avg_price):
            print(f"📋 订单更新: ID={order_id}, 状态={status}, 成交量={filled_qty}, 均价={avg_price}")
        
        def on_position_update(symbol, position):
            print(f"📊 持仓更新: {symbol} = {position.quantity}@{position.avg_cost}")
        
        def on_risk_alert(alert_type, data):
            print(f"⚠️ 风险警报: {alert_type} - {data}")
        
        # 注册回调
        system.ib_trading_manager.add_callback('price_update', on_price_update)
        system.ib_trading_manager.add_callback('order_filled', on_order_filled)
        system.ib_trading_manager.add_callback('position_update', on_position_update)
        system.ib_trading_manager.add_callback('risk_alert', on_risk_alert)
        
        print("✅ 实时交易系统已启动")
        print("按 Ctrl+C 停止交易...")
        
        # 交易循环
        import time
        while True:
            try:
                # 获取交易状态
                status = system.ib_trading_manager.get_trading_status()
                print(f"\n📈 交易状态: 连接={status['connected']}, 活跃订单={status['active_orders']}, 持仓={status['positions']}")
                
                # 显示策略状态
                strategy_status = system.strategy_manager.get_all_status()
                for name, status in strategy_status.items():
                    print(f"📊 策略 {name}: 跟踪标的={len(status['tracked_symbols'])}, 最近信号={status['last_signals']}")
                
                time.sleep(30)  # 每30秒检查一次
                
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        print(f"❌ 实时交易出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("正在断开IB连接...")
        system.ib_trading_manager.disconnect()
        print("✅ 实时交易系统已停止")


if __name__ == "__main__":
    main()