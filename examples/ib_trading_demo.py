#!/usr/bin/env python3
"""
Interactive Brokers 交易系统完整演示
集成风险管理、订单管理和交易执行的完整演示程序
"""

import time
import threading
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json

# 导入自定义模块
from enhanced_ib_trading_system import EnhancedIBTradingSystem, TradingMode, TradingConfig
from ib_risk_manager import IBRiskManager, RiskLimit, RiskLevel, RiskAction
from ib_order_manager import IBOrderManager, OrderRequest, OrderType, TimeInForce, OrderInfo, ExecutionInfo

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ib_trading_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IBTradingDemo:
    """IB交易系统演示"""
    
    def __init__(self):
        # 交易配置
        self.config = TradingConfig(
            host="127.0.0.1",
            port=7497,  # TWS Paper Trading端口
            client_id=2,
            mode=TradingMode.PAPER,  # 模拟交易模式
            enable_risk_management=True,
            max_position_value=50000.0,
            max_daily_loss=2000.0
        )
        
        # 风险限制配置
        self.risk_limits = RiskLimit(
            max_position_value=50000.0,
            max_symbol_exposure=15000.0,
            max_sector_exposure=25000.0,
            max_leverage=2.0,
            max_daily_trades=50,
            max_order_size=500,
            min_order_value=100.0,
            max_daily_loss=2000.0,
            max_drawdown=0.10,
            stop_loss_pct=0.05,
            trading_start_time="09:30",
            trading_end_time="16:00",
            max_volatility=0.3,
            volatility_window=20
        )
        
        # 系统组件
        self.trading_system = None
        self.risk_manager = None
        self.order_manager = None
        
        # 演示数据
        self.demo_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        self.demo_strategies = ["momentum", "mean_reversion", "breakout"]
        
        # 运行状态
        self.running = False
        self.demo_thread = None
        
        # 统计数据
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'total_pnl': 0.0,
            'risk_alerts': 0,
            'start_time': None
        }
    
    def initialize(self) -> bool:
        """初始化系统"""
        try:
            logger.info("初始化IB交易系统演示...")
            
            # 创建风险管理器
            self.risk_manager = IBRiskManager(self.risk_limits)
            
            # 创建订单管理器
            self.order_manager = IBOrderManager(
                host=self.config.host,
                port=self.config.port,
                client_id=self.config.client_id
            )
            
            # 设置风险管理器
            self.order_manager.set_risk_manager(self.risk_manager)
            
            # 创建交易系统
            self.trading_system = EnhancedIBTradingSystem(self.config)
            
            # 连接到IB
            if not self.order_manager.connect_to_ib():
                logger.error("无法连接到IB")
                return False
            
            if not self.trading_system.connect():
                logger.error("交易系统连接失败")
                return False
            
            # 启动风险监控
            self.risk_manager.start_monitoring()
            
            # 设置回调函数
            self._setup_callbacks()
            
            logger.info("系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    def start_demo(self):
        """启动演示"""
        if not self.initialize():
            logger.error("系统初始化失败，无法启动演示")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info("=" * 60)
        logger.info("IB交易系统演示开始")
        logger.info(f"模式: {self.config.mode.value}")
        logger.info(f"连接: {self.config.host}:{self.config.port}")
        logger.info("=" * 60)
        
        # 启动演示线程
        self.demo_thread = threading.Thread(target=self._demo_loop, daemon=True)
        self.demo_thread.start()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # 主循环
            while self.running:
                self._print_status()
                time.sleep(30)  # 每30秒打印一次状态
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止演示...")
        finally:
            self.stop_demo()
    
    def stop_demo(self):
        """停止演示"""
        logger.info("正在停止演示...")
        
        self.running = False
        
        # 取消所有订单
        if self.order_manager:
            cancelled_count = self.order_manager.cancel_all_orders()
            logger.info(f"已取消 {cancelled_count} 个订单")
        
        # 停止风险监控
        if self.risk_manager:
            self.risk_manager.stop_monitoring()
        
        # 断开连接
        if self.order_manager:
            self.order_manager.disconnect_from_ib()
        
        if self.trading_system:
            self.trading_system.disconnect()
        
        # 打印最终统计
        self._print_final_stats()
        
        logger.info("演示已停止")
    
    def _demo_loop(self):
        """演示主循环"""
        demo_scenarios = [
            self._demo_basic_orders,
            self._demo_bracket_orders,
            self._demo_strategy_orders,
            self._demo_risk_management,
            self._demo_market_data
        ]
        
        scenario_index = 0
        
        while self.running:
            try:
                # 执行演示场景
                scenario = demo_scenarios[scenario_index % len(demo_scenarios)]
                logger.info(f"执行演示场景: {scenario.__name__}")
                
                scenario()
                
                scenario_index += 1
                
                # 等待一段时间
                time.sleep(60)  # 每分钟执行一个场景
                
            except Exception as e:
                logger.error(f"演示场景执行错误: {e}")
                time.sleep(10)
    
    def _demo_basic_orders(self):
        """演示基本订单"""
        logger.info("--- 基本订单演示 ---")
        
        # 市价买入订单
        market_order = OrderRequest(
            symbol="AAPL",
            action="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
            strategy_id="demo_basic",
            notes="演示市价订单"
        )
        
        order_id = self.order_manager.submit_order(market_order)
        if order_id:
            self.stats['orders_submitted'] += 1
            logger.info(f"市价订单已提交: {order_id}")
        
        time.sleep(5)
        
        # 限价卖出订单
        limit_order = OrderRequest(
            symbol="MSFT",
            action="SELL",
            quantity=5,
            order_type=OrderType.LIMIT,
            price=350.0,
            time_in_force=TimeInForce.DAY,
            strategy_id="demo_basic",
            notes="演示限价订单"
        )
        
        order_id2 = self.order_manager.submit_order(limit_order)
        if order_id2:
            self.stats['orders_submitted'] += 1
            logger.info(f"限价订单已提交: {order_id2}")
    
    def _demo_bracket_orders(self):
        """演示括号订单（带止损止盈）"""
        logger.info("--- 括号订单演示 ---")
        
        bracket_order = OrderRequest(
            symbol="GOOGL",
            action="BUY",
            quantity=2,
            order_type=OrderType.LIMIT,
            price=2800.0,
            stop_loss_price=2750.0,  # 止损价
            take_profit_price=2900.0,  # 止盈价
            strategy_id="demo_bracket",
            notes="演示括号订单"
        )
        
        order_id = self.order_manager.submit_order(bracket_order)
        if order_id:
            self.stats['orders_submitted'] += 1
            logger.info(f"括号订单已提交: {order_id}")
            
            # 等待一段时间查看子订单
            time.sleep(10)
            order_info = self.order_manager.get_order_info(order_id)
            if order_info and order_info.child_orders:
                logger.info(f"子订单: {order_info.child_orders}")
    
    def _demo_strategy_orders(self):
        """演示策略订单"""
        logger.info("--- 策略订单演示 ---")
        
        strategy_id = "momentum_strategy"
        
        # 提交多个策略订单
        for i, symbol in enumerate(self.demo_symbols[:3]):
            order = OrderRequest(
                symbol=symbol,
                action="BUY" if i % 2 == 0 else "SELL",
                quantity=5 + i * 2,
                order_type=OrderType.LIMIT,
                price=100.0 + i * 10,  # 模拟价格
                strategy_id=strategy_id,
                notes=f"动量策略订单 {i+1}"
            )
            
            order_id = self.order_manager.submit_order(order)
            if order_id:
                self.stats['orders_submitted'] += 1
                logger.info(f"策略订单已提交: {order_id} - {symbol}")
        
        time.sleep(15)
        
        # 获取策略订单
        strategy_orders = self.order_manager.get_strategy_orders(strategy_id)
        logger.info(f"策略 {strategy_id} 共有 {len(strategy_orders)} 个订单")
        
        # 取消部分策略订单
        cancelled_count = self.order_manager.cancel_strategy_orders(strategy_id)
        self.stats['orders_cancelled'] += cancelled_count
        logger.info(f"已取消策略订单: {cancelled_count}")
    
    def _demo_risk_management(self):
        """演示风险管理"""
        logger.info("--- 风险管理演示 ---")
        
        # 尝试提交超限订单
        large_order = OrderRequest(
            symbol="TSLA",
            action="BUY",
            quantity=1000,  # 超过限制
            order_type=OrderType.MARKET,
            strategy_id="demo_risk",
            notes="测试风险限制"
        )
        
        # 检查订单风险
        allow, alerts = self.risk_manager.check_order_risk(
            large_order.symbol, large_order.action, large_order.quantity, 800.0
        )
        
        logger.info(f"大额订单风险检查: {'允许' if allow else '拒绝'}")
        for alert in alerts:
            logger.warning(f"风险警报: {alert.message}")
            self.stats['risk_alerts'] += 1
        
        if not allow:
            logger.info("订单被风险管理器拒绝")
        else:
            order_id = self.order_manager.submit_order(large_order)
            if order_id:
                self.stats['orders_submitted'] += 1
        
        # 获取风险摘要
        risk_summary = self.risk_manager.get_risk_summary()
        metrics = risk_summary['metrics']
        logger.info(f"当前风险等级: {metrics.overall_risk_level.value}")
        logger.info(f"总仓位价值: ${metrics.total_position_value:.2f}")
        logger.info(f"日盈亏: ${metrics.daily_pnl:.2f}")
    
    def _demo_market_data(self):
        """演示市场数据"""
        logger.info("--- 市场数据演示 ---")
        
        # 订阅市场数据
        for symbol in self.demo_symbols[:2]:
            success = self.trading_system.subscribe_market_data(symbol)
            if success:
                logger.info(f"已订阅 {symbol} 市场数据")
            
            time.sleep(2)
        
        # 等待数据更新
        time.sleep(10)
        
        # 获取市场数据
        for symbol in self.demo_symbols[:2]:
            data = self.trading_system.get_market_data(symbol)
            if data:
                logger.info(f"{symbol} 市场数据: 最新价 ${data.last:.2f}, 买价 ${data.bid:.2f}, 卖价 ${data.ask:.2f}")
                
                # 更新风险管理器的市场数据
                self.risk_manager.update_market_data(symbol, {
                    'last': data.last,
                    'bid': data.bid,
                    'ask': data.ask
                })
    
    def _setup_callbacks(self):
        """设置回调函数"""
        # 订单状态回调
        def on_order_filled(order_info: OrderInfo):
            self.stats['orders_filled'] += 1
            logger.info(f"订单成交: {order_info.order_id} - {order_info.symbol} {order_info.action} {order_info.filled_quantity}")
        
        def on_order_cancelled(order_info: OrderInfo):
            self.stats['orders_cancelled'] += 1
            logger.info(f"订单取消: {order_info.order_id} - {order_info.symbol}")
        
        def on_order_rejected(order_info: OrderInfo):
            logger.error(f"订单拒绝: {order_info.order_id} - {order_info.error_message}")
        
        self.order_manager.add_order_callback("status_change", on_order_filled)
        self.order_manager.add_order_callback("cancelled", on_order_cancelled)
        self.order_manager.add_order_callback("rejected", on_order_rejected)
        
        # 执行回调
        def on_execution(exec_info: ExecutionInfo):
            logger.info(f"执行: {exec_info.symbol} {exec_info.side} {exec_info.quantity}@${exec_info.price:.2f}")
            self.stats['total_pnl'] += exec_info.realized_pnl
        
        self.order_manager.add_execution_callback(on_execution)
    
    def _print_status(self):
        """打印状态信息"""
        logger.info("=" * 50)
        logger.info("系统状态")
        logger.info("=" * 50)
        
        # 运行时间
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            logger.info(f"运行时间: {runtime}")
        
        # 订单统计
        order_stats = self.order_manager.get_order_statistics()
        logger.info(f"订单统计:")
        logger.info(f"  总订单: {order_stats['total_orders']}")
        logger.info(f"  已成交: {order_stats['filled_orders']}")
        logger.info(f"  已取消: {order_stats['cancelled_orders']}")
        logger.info(f"  活跃订单: {order_stats['active_orders']}")
        logger.info(f"  成交率: {order_stats['fill_rate']:.2%}")
        
        # 风险指标
        risk_summary = self.risk_manager.get_risk_summary()
        metrics = risk_summary['metrics']
        logger.info(f"风险指标:")
        logger.info(f"  风险等级: {metrics.overall_risk_level.value}")
        logger.info(f"  总仓位: ${metrics.total_position_value:.2f}")
        logger.info(f"  杠杆: {metrics.leverage:.2f}x")
        logger.info(f"  日盈亏: ${metrics.daily_pnl:.2f}")
        
        # 演示统计
        logger.info(f"演示统计:")
        logger.info(f"  提交订单: {self.stats['orders_submitted']}")
        logger.info(f"  成交订单: {self.stats['orders_filled']}")
        logger.info(f"  取消订单: {self.stats['orders_cancelled']}")
        logger.info(f"  风险警报: {self.stats['risk_alerts']}")
        logger.info(f"  总盈亏: ${self.stats['total_pnl']:.2f}")
        
        logger.info("=" * 50)
    
    def _print_final_stats(self):
        """打印最终统计"""
        logger.info("=" * 60)
        logger.info("演示最终统计")
        logger.info("=" * 60)
        
        if self.stats['start_time']:
            total_runtime = datetime.now() - self.stats['start_time']
            logger.info(f"总运行时间: {total_runtime}")
        
        # 订单统计
        order_stats = self.order_manager.get_order_statistics()
        logger.info(f"订单统计:")
        logger.info(f"  总订单数: {order_stats['total_orders']}")
        logger.info(f"  成交订单: {order_stats['filled_orders']}")
        logger.info(f"  取消订单: {order_stats['cancelled_orders']}")
        logger.info(f"  拒绝订单: {order_stats['rejected_orders']}")
        logger.info(f"  成交率: {order_stats['fill_rate']:.2%}")
        logger.info(f"  总交易量: {order_stats['total_volume']}")
        logger.info(f"  总佣金: ${order_stats['total_commission']:.2f}")
        
        # 风险统计
        logger.info(f"风险管理:")
        logger.info(f"  风险警报数: {self.stats['risk_alerts']}")
        logger.info(f"  最大风险等级: {self.risk_manager.get_risk_summary()['metrics'].overall_risk_level.value}")
        
        # 盈亏统计
        logger.info(f"盈亏统计:")
        logger.info(f"  总盈亏: ${self.stats['total_pnl']:.2f}")
        
        logger.info("=" * 60)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，正在停止演示...")
        self.running = False


def main():
    """主函数"""
    print("Interactive Brokers 交易系统演示")
    print("=" * 60)
    print("注意: 请确保TWS或IB Gateway已启动并配置为Paper Trading模式")
    print("默认连接: 127.0.0.1:7497")
    print("=" * 60)
    
    # 等待用户确认
    input("按Enter键开始演示...")
    
    # 创建并启动演示
    demo = IBTradingDemo()
    
    try:
        demo.start_demo()
    except Exception as e:
        logger.error(f"演示运行错误: {e}")
    finally:
        demo.stop_demo()


if __name__ == "__main__":
    main()