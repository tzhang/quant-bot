#!/usr/bin/env python3
"""
完整的Citadel高频交易系统示例
整合IB API、策略执行、风险管理、监控日志等所有模块
"""

import sys
import os
import time
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入自定义模块
try:
    # 核心模块导入（不依赖GUI）
    from examples.ib_adapter import IBAdapter
    from examples.ib_market_data_stream import IBMarketDataStream
    from examples.citadel_ib_integration import CitadelIBIntegration
    from examples.advanced_risk_manager import AdvancedRiskManager
    from examples.hft_monitoring_logger import HFTMonitoringLogger
    from competitions.citadel.citadel_hft_strategy import CitadelHFTStrategy
    
    # 尝试导入GUI模块，如果失败则跳过
    try:
        from examples.hft_monitor_system import HFTMonitorSystem
        GUI_AVAILABLE = True
    except ImportError as gui_error:
        print(f"GUI模块不可用: {gui_error}")
        GUI_AVAILABLE = False
        
except ImportError as e:
    print(f"导入模块错误: {e}")
    print("请确保所有必要的模块文件都存在")
    sys.exit(1)

class CompleteHFTSystem:
    """完整的高频交易系统"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化完整的HFT系统
        
        Args:
            config: 系统配置参数
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # 系统状态
        self.running = False
        self.start_time = None
        
        # 核心组件
        self.ib_adapter = None
        self.market_data_stream = None
        self.strategy = None
        self.risk_manager = None
        self.monitoring_logger = None
        self.monitor_system = None
        self.integration_engine = None
        
        # 数据存储
        self.market_data = {}
        self.positions = {}
        self.orders = {}
        self.performance_stats = {
            'total_pnl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # 线程管理
        self.threads = []
        self.shutdown_event = threading.Event()
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        log_dir = Path(self.config.get('log_dir', './logs'))
        log_dir.mkdir(exist_ok=True)
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 主日志文件
        log_file = log_dir / f"hft_system_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # 配置logger
        logger = logging.getLogger('HFTSystem')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"接收到信号 {signum}，开始关闭系统...")
        self.shutdown()
    
    def initialize(self) -> bool:
        """初始化系统组件"""
        try:
            self.logger.info("开始初始化HFT系统组件...")
            
            # 1. 初始化IB适配器
            self.logger.info("初始化IB适配器...")
            ib_config = self.config.get('ib_config', {})
            self.ib_adapter = IBAdapter(
                host=ib_config.get('host', '127.0.0.1'),
                port=ib_config.get('port', 7497),
                client_id=ib_config.get('client_id', 1)
            )
            
            if not self.ib_adapter.connect():
                self.logger.error("IB连接失败")
                return False
            
            # 2. 初始化市场数据流
            self.logger.info("初始化市场数据流...")
            self.market_data_stream = IBMarketDataStream(self.ib_adapter)
            
            # 3. 初始化策略
            self.logger.info("初始化交易策略...")
            strategy_config = self.config.get('strategy_config', {})
            self.strategy = CitadelHFTStrategy(
                symbols=strategy_config.get('symbols', ['AAPL', 'MSFT', 'GOOGL']),
                **strategy_config.get('parameters', {})
            )
            
            # 4. 初始化风险管理器
            self.logger.info("初始化风险管理器...")
            risk_config = self.config.get('risk_config', {})
            self.risk_manager = AdvancedRiskManager(risk_config)
            
            # 5. 初始化监控日志系统
            self.logger.info("初始化监控日志系统...")
            monitoring_config = self.config.get('monitoring_config', {})
            self.monitoring_logger = HFTMonitoringLogger(monitoring_config)
            
            # 6. 初始化集成引擎
            self.logger.info("初始化集成引擎...")
            self.integration_engine = CitadelIBIntegration(
                ib_adapter=self.ib_adapter,
                strategy=self.strategy,
                risk_manager=self.risk_manager,
                monitoring_logger=self.monitoring_logger
            )
            
            # 7. 初始化监控系统（可选）
            if self.config.get('enable_gui_monitor', False):
                self.logger.info("初始化GUI监控系统...")
                self.monitor_system = HFTMonitorSystem()
            
            self.logger.info("所有组件初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    def start(self) -> bool:
        """启动系统"""
        try:
            if self.running:
                self.logger.warning("系统已在运行中")
                return True
            
            self.logger.info("启动HFT系统...")
            self.start_time = datetime.now()
            self.running = True
            
            # 启动监控日志系统
            self.monitoring_logger.start()
            
            # 启动市场数据流
            symbols = self.config.get('strategy_config', {}).get('symbols', ['AAPL'])
            for symbol in symbols:
                self.market_data_stream.subscribe_market_data(symbol)
                self.market_data_stream.subscribe_realtime_bars(symbol)
            
            # 设置数据回调
            self.market_data_stream.set_tick_callback(self._on_tick_data)
            self.market_data_stream.set_bar_callback(self._on_bar_data)
            
            # 启动集成引擎
            self.integration_engine.start()
            
            # 启动GUI监控系统（如果启用）
            if self.monitor_system:
                monitor_thread = threading.Thread(
                    target=self.monitor_system.run,
                    daemon=True
                )
                monitor_thread.start()
                self.threads.append(monitor_thread)
            
            # 启动主循环线程
            main_thread = threading.Thread(target=self._main_loop, daemon=True)
            main_thread.start()
            self.threads.append(main_thread)
            
            # 启动性能统计线程
            stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
            stats_thread.start()
            self.threads.append(stats_thread)
            
            self.logger.info("HFT系统启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            return False
    
    def _main_loop(self):
        """主循环"""
        self.logger.info("主循环开始运行...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # 检查系统状态
                self._check_system_health()
                
                # 更新性能统计
                self._update_performance_stats()
                
                # 检查风险状况
                self._check_risk_status()
                
                # 等待一段时间
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"主循环错误: {e}")
                time.sleep(5)
        
        self.logger.info("主循环结束")
    
    def _stats_loop(self):
        """性能统计循环"""
        self.logger.info("性能统计循环开始运行...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # 每分钟输出一次统计信息
                self._log_performance_summary()
                
                # 等待60秒
                for _ in range(60):
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"统计循环错误: {e}")
                time.sleep(10)
        
        self.logger.info("性能统计循环结束")
    
    def _on_tick_data(self, symbol: str, tick_data: Dict[str, Any]):
        """处理tick数据"""
        try:
            # 更新市场数据
            self.market_data[symbol] = tick_data
            
            # 记录市场事件
            if 'bid' in tick_data and 'ask' in tick_data:
                self.monitoring_logger.log_market_event(
                    symbol=symbol,
                    bid=tick_data['bid'],
                    ask=tick_data['ask'],
                    last=tick_data.get('last', tick_data['bid']),
                    volume=tick_data.get('volume', 0)
                )
            
            # 更新监控系统
            if self.monitor_system:
                self.monitor_system.update_market_data(symbol, tick_data)
            
        except Exception as e:
            self.logger.error(f"处理tick数据错误: {e}")
    
    def _on_bar_data(self, symbol: str, bar_data: Dict[str, Any]):
        """处理K线数据"""
        try:
            # 更新策略数据
            if hasattr(self.integration_engine, 'update_market_data'):
                self.integration_engine.update_market_data(symbol, bar_data)
            
            # 更新监控系统
            if self.monitor_system:
                self.monitor_system.update_bar_data(symbol, bar_data)
            
        except Exception as e:
            self.logger.error(f"处理K线数据错误: {e}")
    
    def _check_system_health(self):
        """检查系统健康状态"""
        try:
            # 检查IB连接
            if not self.ib_adapter.is_connected():
                self.logger.warning("IB连接断开，尝试重连...")
                if not self.ib_adapter.connect():
                    self.logger.error("IB重连失败")
                    self.monitoring_logger.log_risk_event(
                        risk_type="CONNECTION_LOST",
                        severity="CRITICAL",
                        symbol="SYSTEM",
                        message="IB连接断开且重连失败",
                        current_value=0,
                        threshold=1,
                        action_taken="尝试重连"
                    )
            
            # 检查数据流
            current_time = datetime.now()
            for symbol in self.market_data:
                last_update = self.market_data[symbol].get('timestamp')
                if last_update and (current_time - last_update).total_seconds() > 30:
                    self.logger.warning(f"{symbol} 数据流可能中断")
            
        except Exception as e:
            self.logger.error(f"系统健康检查错误: {e}")
    
    def _update_performance_stats(self):
        """更新性能统计"""
        try:
            # 从集成引擎获取统计数据
            if hasattr(self.integration_engine, 'get_performance_summary'):
                stats = self.integration_engine.get_performance_summary()
                self.performance_stats.update(stats)
            
            # 从监控日志获取实时统计
            real_time_stats = self.monitoring_logger.get_real_time_stats()
            self.performance_stats.update(real_time_stats)
            
        except Exception as e:
            self.logger.error(f"性能统计更新错误: {e}")
    
    def _check_risk_status(self):
        """检查风险状况"""
        try:
            # 获取风险摘要
            risk_summary = self.risk_manager.get_risk_summary()
            
            # 检查整体风险等级
            overall_risk = self.risk_manager.get_overall_risk_level()
            
            if overall_risk in ['HIGH', 'CRITICAL']:
                self.logger.warning(f"系统风险等级: {overall_risk}")
                
                # 如果风险过高，考虑暂停交易
                if overall_risk == 'CRITICAL':
                    self.logger.error("风险等级达到CRITICAL，暂停交易")
                    if hasattr(self.integration_engine, 'pause_trading'):
                        self.integration_engine.pause_trading()
            
        except Exception as e:
            self.logger.error(f"风险状况检查错误: {e}")
    
    def _log_performance_summary(self):
        """记录性能摘要"""
        try:
            runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            summary = f"""
            ========== HFT系统性能摘要 ==========
            运行时间: {runtime}
            总PnL: {self.performance_stats.get('total_pnl', 0):.2f}
            总交易数: {self.performance_stats.get('total_trades', 0)}
            胜率: {self.performance_stats.get('win_rate', 0):.2%}
            最大回撤: {self.performance_stats.get('max_drawdown', 0):.2%}
            夏普比率: {self.performance_stats.get('sharpe_ratio', 0):.2f}
            平均延迟: {self.performance_stats.get('avg_latency_ms', 0):.2f}ms
            每秒订单数: {self.performance_stats.get('orders_per_second', 0):.2f}
            =====================================
            """
            
            self.logger.info(summary)
            
        except Exception as e:
            self.logger.error(f"性能摘要记录错误: {e}")
    
    def shutdown(self):
        """关闭系统"""
        try:
            if not self.running:
                return
            
            self.logger.info("开始关闭HFT系统...")
            self.running = False
            self.shutdown_event.set()
            
            # 停止集成引擎
            if self.integration_engine:
                self.integration_engine.stop()
            
            # 停止监控日志系统
            if self.monitoring_logger:
                self.monitoring_logger.stop()
            
            # 停止市场数据流
            if self.market_data_stream:
                symbols = self.config.get('strategy_config', {}).get('symbols', [])
                for symbol in symbols:
                    self.market_data_stream.unsubscribe_market_data(symbol)
            
            # 断开IB连接
            if self.ib_adapter:
                self.ib_adapter.disconnect()
            
            # 等待线程结束
            for thread in self.threads:
                thread.join(timeout=5)
            
            # 导出最终报告
            if self.monitoring_logger:
                report_file = self.monitoring_logger.export_daily_report()
                self.logger.info(f"最终报告已导出: {report_file}")
            
            self.logger.info("HFT系统已完全关闭")
            
        except Exception as e:
            self.logger.error(f"系统关闭错误: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'runtime': str(datetime.now() - self.start_time) if self.start_time else None,
            'ib_connected': self.ib_adapter.is_connected() if self.ib_adapter else False,
            'performance_stats': self.performance_stats.copy(),
            'active_symbols': list(self.market_data.keys()),
            'positions_count': len(self.positions),
            'orders_count': len(self.orders)
        }

def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        # IB配置
        'ib_config': {
            'host': '127.0.0.1',
            'port': 7497,  # TWS Demo: 7497, Live: 7496, Gateway Demo: 4002, Live: 4001
            'client_id': 1
        },
        
        # 策略配置
        'strategy_config': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'parameters': {
                'max_position_size': 1000,
                'risk_per_trade': 0.02,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'max_daily_trades': 100,
                'trading_hours': {
                    'start': '09:30',
                    'end': '16:00'
                }
            }
        },
        
        # 风险管理配置
        'risk_config': {
            'max_portfolio_risk': 0.05,
            'max_position_size': 10000,
            'max_daily_loss': 5000,
            'max_drawdown': 0.1,
            'position_limits': {
                'AAPL': 2000,
                'MSFT': 2000,
                'GOOGL': 1000
            },
            'sector_limits': {
                'Technology': 0.6
            }
        },
        
        # 监控配置
        'monitoring_config': {
            'log_dir': './logs',
            'max_log_size_mb': 100,
            'max_log_files': 10,
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email@gmail.com',
                'password': 'your_password',
                'from_email': 'your_email@gmail.com',
                'to_emails': ['alert@company.com'],
                'min_severity': 'HIGH'
            },
            'alert_thresholds': {
                'max_drawdown': 0.05,
                'min_win_rate': 0.4,
                'max_latency_ms': 100
            }
        },
        
        # 系统配置
        'log_dir': './logs',
        'enable_gui_monitor': False,  # 设置为True启用GUI监控
        'auto_start_trading': True,
        'paper_trading': True  # 设置为False进行实盘交易
    }

def main():
    """主函数"""
    print("=" * 60)
    print("Citadel高频交易系统 - 完整示例")
    print("=" * 60)
    
    # 创建配置
    config = create_default_config()
    
    # 检查是否为演示模式
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        print("运行演示模式...")
        config['paper_trading'] = True
        config['enable_gui_monitor'] = GUI_AVAILABLE  # 只有在GUI可用时才启用
    else:
        config['enable_gui_monitor'] = False  # 默认不启用GUI
    
    # 创建系统
    hft_system = CompleteHFTSystem(config)
    
    try:
        # 初始化系统
        print("初始化系统...")
        if not hft_system.initialize():
            print("系统初始化失败")
            return 1
        
        # 启动系统
        print("启动系统...")
        if not hft_system.start():
            print("系统启动失败")
            return 1
        
        print("系统运行中... (按Ctrl+C停止)")
        
        # 主循环
        try:
            while hft_system.running:
                # 每10秒输出一次状态
                status = hft_system.get_status()
                print(f"运行时间: {status['runtime']}, "
                      f"PnL: {status['performance_stats'].get('total_pnl', 0):.2f}, "
                      f"交易数: {status['performance_stats'].get('total_trades', 0)}")
                
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n接收到中断信号...")
        
    except Exception as e:
        print(f"系统运行错误: {e}")
        return 1
    
    finally:
        # 关闭系统
        print("关闭系统...")
        hft_system.shutdown()
        print("系统已关闭")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())