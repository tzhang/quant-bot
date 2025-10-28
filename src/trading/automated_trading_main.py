#!/usr/bin/env python3
"""
自动化交易主程序
整合策略管理、交易执行、风险管理和监控功能
支持配置文件驱动的灵活交易系统
"""

import sys
import os
import time
import signal
import threading
import json
import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import schedule
from dataclasses import asdict

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.trading.enhanced_trading_engine import (
    EnhancedTradingEngine, TradingMode, RiskLimits, 
    TradingSignal, OrderAction, Position
)
from src.trading.strategy_manager import (
    StrategyManager, MarketData, create_default_strategies
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingConfig:
    """交易配置类"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "trading_config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'trading': {
                'mode': 'paper',  # paper, live, backtest - 默认使用模拟交易
                'initial_capital': 100000.0,
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                'trading_hours': {
                    'start': '09:30',
                    'end': '16:00',
                    'timezone': 'US/Eastern'
                },
                'data_update_interval': 60,  # 秒
                'signal_generation_interval': 300,  # 秒
            },
            'risk_management': {
                'max_position_value': 50000.0,
                'max_daily_loss': 5000.0,
                'max_symbol_exposure': 20000.0,
                'max_daily_trades': 100,
                'max_portfolio_risk': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'max_leverage': 2.0
            },
            'strategies': {
                'momentum': {
                    'enabled': True,
                    'weight': 1.0,
                    'parameters': {
                        'base_position_size': 100,
                        'max_position_size': 500
                    }
                },
                'mean_reversion': {
                    'enabled': True,
                    'weight': 0.8,
                    'parameters': {
                        'base_position_size': 100,
                        'max_position_size': 300
                    }
                },
                'breakout': {
                    'enabled': True,
                    'weight': 1.2,
                    'parameters': {
                        'base_position_size': 150,
                        'max_position_size': 600
                    }
                },
                'ml_prediction': {
                    'enabled': False,  # 默认关闭ML策略
                    'weight': 0.9,
                    'parameters': {
                        'base_position_size': 100,
                        'max_position_size': 400
                    }
                }
            },
            'signal_aggregation': {
                'method': 'weighted_average',  # weighted_average, majority_vote, ensemble
                'min_signal_strength': 0.3,
                'min_confidence_level': 'MEDIUM'
            },
            'data_sources': {
                'primary': 'yahoo',  # yahoo, alpha_vantage, ib
                'backup': 'alpha_vantage',
                'api_keys': {
                    'alpha_vantage': 'YOUR_API_KEY',
                    'ib': {
                        'host': '127.0.0.1',
                        'port': 4001,
                        'client_id': 2
                    }
                }
            },
            'monitoring': {
                'enable_email_alerts': False,
                'enable_slack_alerts': False,
                'performance_report_interval': 3600,  # 秒
                'save_trades_to_db': True,
                'backup_interval': 1800  # 秒
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                        loaded_config = yaml.safe_load(f)
                    else:
                        loaded_config = json.load(f)
                
                # 递归合并配置
                self._merge_config(default_config, loaded_config)
                logger.info(f"配置文件加载成功: {self.config_file}")
                
            except Exception as e:
                logger.error(f"配置文件加载失败: {e}")
                logger.info("使用默认配置")
        else:
            logger.info("配置文件不存在，使用默认配置")
            # 创建默认配置文件
            self._save_config(default_config)
        
        return default_config
    
    def _merge_config(self, default: Dict, loaded: Dict):
        """递归合并配置"""
        for key, value in loaded.items():
            if key in default:
                if isinstance(default[key], dict) and isinstance(value, dict):
                    self._merge_config(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"配置文件保存成功: {self.config_file}")
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
    
    def get(self, key_path: str, default=None):
        """获取配置值"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """设置配置值"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self._save_config(self.config)

class DataProvider:
    """数据提供者"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.symbols = config.get('trading.symbols', [])
        self.primary_source = config.get('data_sources.primary', 'yahoo')
        self.backup_source = config.get('data_sources.backup', 'alpha_vantage')
        
        # 数据缓存
        self.data_cache: Dict[str, MarketData] = {}
        self.price_history: Dict[str, List[Dict]] = {}
        
        logger.info(f"数据提供者初始化: 主要源={self.primary_source}, 备用源={self.backup_source}")
    
    def get_market_data(self, symbols: List[str] = None) -> Dict[str, MarketData]:
        """获取市场数据"""
        if symbols is None:
            symbols = self.symbols
        
        market_data = {}
        
        for symbol in symbols:
            try:
                data = self._fetch_data(symbol)
                if data:
                    market_data[symbol] = data
                    self.data_cache[symbol] = data
                    self._update_price_history(symbol, data)
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
                # 使用缓存数据
                if symbol in self.data_cache:
                    market_data[symbol] = self.data_cache[symbol]
        
        return market_data
    
    def _fetch_data(self, symbol: str) -> Optional[MarketData]:
        """获取单个标的数据"""
        logger.info(f"尝试从主要数据源 {self.primary_source} 获取 {symbol} 数据")
        
        # 尝试主要数据源
        data = None
        if self.primary_source == 'ib':
            data = self._fetch_ib_data(symbol)
        elif self.primary_source == 'yahoo':
            data = self._fetch_yahoo_data(symbol)
        elif self.primary_source == 'alpha_vantage':
            data = self._fetch_alpha_vantage_data(symbol)
        
        # 如果主要数据源失败，尝试备用数据源
        if data is None:
            logger.warning(f"主要数据源 {self.primary_source} 获取 {symbol} 数据失败，尝试备用数据源 {self.backup_source}")
            if self.backup_source == 'ib' and self.primary_source != 'ib':
                data = self._fetch_ib_data(symbol)
            elif self.backup_source == 'yahoo' and self.primary_source != 'yahoo':
                data = self._fetch_yahoo_data(symbol)
            elif self.backup_source == 'alpha_vantage' and self.primary_source != 'alpha_vantage':
                data = self._fetch_alpha_vantage_data(symbol)
        
        # 如果所有数据源都失败，生成模拟数据
        if data is None:
            logger.warning(f"所有数据源都无法获取 {symbol} 数据，使用模拟数据")
            data = self._generate_mock_data(symbol)
        
        return data
    
    def _fetch_yahoo_data(self, symbol: str) -> Optional[MarketData]:
        """从Yahoo Finance获取数据"""
        try:
            # import yfinance as yf  # 已移除，不再使用yfinance
            logger.warning("yfinance已移除，无法获取股票价格数据")
            return None  # 返回None而不是空DataFrame
            # ticker = yf.Ticker(symbol)
            # hist = ticker.history(period="1d", interval="1m")
            # 
            # if hist.empty:
            #     return None
            # 
            # latest = hist.iloc[-1]
            # 
            # return MarketData(
            #     symbol=symbol,
            #     timestamp=datetime.now(),
            #     open=float(latest['Open']),
            #     high=float(latest['High']),
            #     low=float(latest['Low']),
            #     close=float(latest['Close']),
            #     volume=int(latest['Volume'])
            # )
            
        except Exception as e:
            logger.error(f"Yahoo Finance数据获取失败: {e}")
            return self._generate_mock_data(symbol)
    
    def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[MarketData]:
        """从Alpha Vantage获取数据"""
        try:
            # 导入Alpha Vantage数据源
            from src.data.alternative_sources import AlphaVantageSource
            
            # 创建Alpha Vantage数据源
            av_source = AlphaVantageSource()
            
            if not av_source.is_available():
                logger.error("Alpha Vantage API密钥未配置，请检查ALPHA_VANTAGE_API_KEY环境变量")
                return None
            
            # 获取最近的数据（最近5天）
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            
            # 获取股票数据
            df = av_source.fetch_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.error(f"无法从Alpha Vantage获取{symbol}数据")
                return None
            
            # 获取最新数据
            latest_data = df.iloc[-1]
            
            # 转换为MarketData格式
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=float(latest_data['open']),
                high=float(latest_data['high']),
                low=float(latest_data['low']),
                close=float(latest_data['close']),
                volume=int(latest_data['volume'])
            )
            
            logger.info(f"成功从Alpha Vantage获取{symbol}数据: {market_data.close}")
            return market_data
            
        except Exception as e:
            logger.error(f"从Alpha Vantage获取{symbol}数据失败: {e}")
            return None
    
    def _fetch_ib_data(self, symbol: str) -> Optional[MarketData]:
        """从Interactive Brokers获取数据"""
        try:
            # 导入IB数据提供者
            from src.data.ib_data_provider import create_ib_provider
            
            # 创建IB数据提供者
            ib_provider = create_ib_provider()
            
            if not ib_provider.is_available:
                logger.error("IB数据提供者不可用，请检查IB API安装和TWS连接")
                return None
            
            # 获取最近的数据（最近5天）
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            
            # 获取股票数据
            df = ib_provider.get_stock_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.error(f"无法从IB获取{symbol}数据")
                return None
            
            # 获取最新数据
            latest_data = df.iloc[-1]
            
            # 转换为MarketData格式
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=float(latest_data['open']),
                high=float(latest_data['high']),
                low=float(latest_data['low']),
                close=float(latest_data['close']),
                volume=int(latest_data['volume']),
                price=float(latest_data['close'])  # 使用收盘价作为当前价格
            )
            
            logger.info(f"成功从IB获取{symbol}真实市场数据: 价格=${market_data.price:.2f}")
            return market_data
            
        except ImportError as e:
            logger.error(f"无法导入IB数据提供者: {e}")
            return None
        except Exception as e:
            logger.error(f"从IB获取{symbol}数据失败: {e}")
            return None
    
    def _generate_mock_data(self, symbol: str) -> MarketData:
        """生成模拟数据"""
        # 基础价格
        base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0,
            'TSLA': 800.0,
            'AMZN': 3200.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # 添加随机波动
        price_change = np.random.normal(0, 0.02)
        current_price = base_price * (1 + price_change)
        
        # 生成OHLC数据
        open_price = current_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.normal(1000000, 200000))
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=open_price,
            high=high_price,
            low=low_price,
            close=current_price,
            volume=max(volume, 1000)
        )
    
    def _update_price_history(self, symbol: str, data: MarketData):
        """更新价格历史"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'timestamp': data.timestamp,
            'close': data.close,
            'volume': data.volume
        })
        
        # 保持历史数据长度
        max_history = 1000
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]

class AutomatedTradingSystem:
    """自动化交易系统主类"""
    
    def __init__(self, config_file: str = None):
        self.config = TradingConfig(config_file)
        
        # 初始化组件
        self._init_trading_engine()
        self._init_strategy_manager()
        self._init_data_provider()
        
        # 运行状态
        self.running = False
        self.threads: List[threading.Thread] = []
        
        # 性能统计
        self.start_time = None
        self.trade_count = 0
        self.last_report_time = datetime.now()
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("自动化交易系统初始化完成")
    
    def _init_trading_engine(self):
        """初始化交易引擎"""
        # 交易模式
        mode_str = self.config.get('trading.mode', 'paper')
        trading_mode = TradingMode.PAPER
        if mode_str == 'live':
            trading_mode = TradingMode.LIVE
        elif mode_str == 'backtest':
            trading_mode = TradingMode.BACKTEST
        
        # 风险限制
        risk_config = self.config.get('risk_management', {})
        risk_limits = RiskLimits(
            max_position_value=risk_config.get('max_position_value', 50000.0),
            max_daily_loss=risk_config.get('max_daily_loss', 5000.0),
            max_symbol_exposure=risk_config.get('max_symbol_exposure', 20000.0),
            max_daily_trades=risk_config.get('max_daily_trades', 100),
            max_portfolio_risk=risk_config.get('max_portfolio_risk', 0.02),
            stop_loss_pct=risk_config.get('stop_loss_pct', 0.05),
            take_profit_pct=risk_config.get('take_profit_pct', 0.10),
            max_leverage=risk_config.get('max_leverage', 2.0)
        )
        
        # 创建交易引擎
        self.trading_engine = EnhancedTradingEngine(
            initial_capital=self.config.get('trading.initial_capital', 100000.0),
            trading_mode=trading_mode,
            risk_limits=risk_limits
        )
        
        # 添加回调函数
        self.trading_engine.add_callback('on_order_filled', self._on_order_filled)
        self.trading_engine.add_callback('on_position_update', self._on_position_update)
        self.trading_engine.add_callback('on_risk_alert', self._on_risk_alert)
    
    def _init_strategy_manager(self):
        """初始化策略管理器"""
        self.strategy_manager = StrategyManager()
        
        # 设置信号聚合参数
        aggregation_config = self.config.get('signal_aggregation', {})
        self.strategy_manager.signal_aggregation_method = aggregation_config.get('method', 'weighted_average')
        self.strategy_manager.min_signal_strength = aggregation_config.get('min_signal_strength', 0.3)
        self.strategy_manager.min_confidence_level = aggregation_config.get('min_confidence_level', 'MEDIUM')
        
        # 添加策略
        strategies = create_default_strategies()
        strategy_configs = self.config.get('strategies', {})
        
        for strategy in strategies:
            strategy_name = strategy.strategy_type.value
            if strategy_name in strategy_configs:
                config = strategy_configs[strategy_name]
                
                # 更新策略参数
                strategy.enabled = config.get('enabled', True)
                strategy.parameters.update(config.get('parameters', {}))
                
                # 添加到管理器
                weight = config.get('weight', 1.0)
                self.strategy_manager.add_strategy(strategy, weight)
    
    def _init_data_provider(self):
        """初始化数据提供者"""
        self.data_provider = DataProvider(self.config)
    
    def start(self):
        """启动自动化交易系统"""
        if self.running:
            logger.warning("交易系统已在运行中")
            return
        
        self.running = True
        self.start_time = datetime.now()
        
        # 启动交易引擎
        self.trading_engine.start()
        
        # 启动数据更新线程
        data_thread = threading.Thread(target=self._data_update_loop, daemon=True)
        data_thread.start()
        self.threads.append(data_thread)
        
        # 启动信号生成线程
        signal_thread = threading.Thread(target=self._signal_generation_loop, daemon=True)
        signal_thread.start()
        self.threads.append(signal_thread)
        
        # 启动性能监控线程
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        self.threads.append(monitor_thread)
        
        # 启动定时任务
        self._schedule_tasks()
        schedule_thread = threading.Thread(target=self._schedule_loop, daemon=True)
        schedule_thread.start()
        self.threads.append(schedule_thread)
        
        logger.info("自动化交易系统启动成功")
    
    def stop(self):
        """停止自动化交易系统"""
        logger.info("正在停止自动化交易系统...")
        
        self.running = False
        
        # 停止交易引擎
        self.trading_engine.stop()
        
        # 等待所有线程结束
        for thread in self.threads:
            thread.join(timeout=5)
        
        self.threads.clear()
        
        # 保存最终报告
        self._generate_performance_report()
        
        logger.info("自动化交易系统已停止")
    
    def _data_update_loop(self):
        """数据更新循环"""
        update_interval = self.config.get('trading.data_update_interval', 60)
        
        while self.running:
            try:
                # 获取市场数据
                market_data = self.data_provider.get_market_data()
                
                # 更新策略管理器
                for symbol, data in market_data.items():
                    self.strategy_manager.update_market_data(symbol, data)
                    
                    # 更新交易引擎的市场数据
                    self.trading_engine.update_market_data(symbol, data.close, data.volume)
                
                logger.debug(f"市场数据更新完成: {len(market_data)} 个标的")
                
            except Exception as e:
                logger.error(f"数据更新错误: {e}")
            
            time.sleep(update_interval)
    
    def _signal_generation_loop(self):
        """信号生成循环"""
        generation_interval = self.config.get('trading.signal_generation_interval', 300)
        
        while self.running:
            try:
                # 检查交易时间
                if not self._is_trading_hours():
                    time.sleep(60)  # 非交易时间，等待1分钟
                    continue
                
                # 生成交易信号
                symbols = self.config.get('trading.symbols', [])
                signals = self.strategy_manager.generate_aggregated_signals(symbols)
                
                # 执行信号
                for signal in signals:
                    self.trading_engine.add_signal(signal)
                    logger.info(f"添加交易信号: {signal.symbol} {signal.action.value} "
                              f"{signal.quantity} (强度: {signal.signal_strength:.2f})")
                
                if signals:
                    logger.info(f"本轮生成 {len(signals)} 个交易信号")
                
            except Exception as e:
                logger.error(f"信号生成错误: {e}")
            
            time.sleep(generation_interval)
    
    def _monitoring_loop(self):
        """监控循环"""
        report_interval = self.config.get('monitoring.performance_report_interval', 3600)
        last_report = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # 定期生成性能报告
                if current_time - last_report >= report_interval:
                    self._generate_performance_report()
                    last_report = current_time
                
                # 检查系统健康状态
                self._check_system_health()
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
            
            time.sleep(60)  # 每分钟检查一次
    
    def _schedule_loop(self):
        """定时任务循环"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"定时任务错误: {e}")
    
    def _schedule_tasks(self):
        """设置定时任务"""
        # 每日开盘前重置统计
        schedule.every().day.at("09:00").do(self._daily_reset)
        
        # 每日收盘后生成报告
        schedule.every().day.at("16:30").do(self._daily_report)
        
        # 每小时备份数据
        schedule.every().hour.do(self._backup_data)
    
    def _is_trading_hours(self) -> bool:
        """检查是否在交易时间内"""
        trading_hours = self.config.get('trading.trading_hours', {})
        start_time = trading_hours.get('start', '09:30')
        end_time = trading_hours.get('end', '16:00')
        
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        
        # 简单的时间检查（忽略时区和节假日）
        return start_time <= current_time <= end_time and now.weekday() < 5
    
    def _on_order_filled(self, order):
        """订单成交回调"""
        self.trade_count += 1
        logger.info(f"订单成交: {order.symbol} {order.action.value} "
                   f"{order.filled_quantity}@${order.avg_fill_price:.2f}")
    
    def _on_position_update(self, position):
        """持仓更新回调"""
        logger.debug(f"持仓更新: {position.symbol} {position.quantity} 股, "
                    f"未实现盈亏: ${position.unrealized_pnl:.2f}")
    
    def _on_risk_alert(self, alert):
        """风险警报回调"""
        logger.warning(f"风险警报: {alert['type']} - {alert['message']}")
        
        # 这里可以添加邮件或Slack通知
        if self.config.get('monitoring.enable_email_alerts', False):
            self._send_email_alert(alert)
        
        if self.config.get('monitoring.enable_slack_alerts', False):
            self._send_slack_alert(alert)
    
    def _generate_performance_report(self):
        """生成性能报告"""
        try:
            # 获取投资组合摘要
            portfolio = self.trading_engine.get_portfolio_summary()
            
            # 获取策略性能
            strategy_performance = self.strategy_manager.get_strategy_performance()
            
            # 计算运行时间
            runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'runtime_hours': runtime.total_seconds() / 3600,
                'portfolio': portfolio,
                'strategy_performance': strategy_performance,
                'trade_count': self.trade_count,
                'system_status': 'running' if self.running else 'stopped'
            }
            
            # 保存报告
            report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # 打印摘要
            logger.info("=== 性能报告 ===")
            logger.info(f"运行时间: {runtime}")
            logger.info(f"总权益: ${portfolio['total_equity']:,.2f}")
            logger.info(f"总盈亏: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:.2%})")
            logger.info(f"持仓数量: {portfolio['positions_count']}")
            logger.info(f"交易次数: {self.trade_count}")
            logger.info(f"最大回撤: {portfolio['max_drawdown']:.2%}")
            
        except Exception as e:
            logger.error(f"生成性能报告错误: {e}")
    
    def _check_system_health(self):
        """检查系统健康状态"""
        # 检查内存使用
        import psutil
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage > 90:
            logger.warning(f"内存使用率过高: {memory_usage:.1f}%")
        
        # 检查线程状态
        active_threads = sum(1 for t in self.threads if t.is_alive())
        expected_threads = len(self.threads)
        
        if active_threads < expected_threads:
            logger.warning(f"线程异常: {active_threads}/{expected_threads} 个线程运行中")
    
    def _daily_reset(self):
        """每日重置"""
        logger.info("执行每日重置")
        # 重置日交易统计
        self.trading_engine.daily_trades = 0
        self.trading_engine.daily_pnl = 0.0
    
    def _daily_report(self):
        """每日报告"""
        logger.info("生成每日报告")
        self._generate_performance_report()
    
    def _backup_data(self):
        """备份数据"""
        try:
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 备份交易历史
            if self.trading_engine.trade_history:
                trades_df = pd.DataFrame([asdict(trade) for trade in self.trading_engine.trade_history])
                trades_df.to_csv(backup_dir / f"trades_{timestamp}.csv", index=False)
            
            # 备份持仓信息
            if self.trading_engine.positions:
                positions_df = pd.DataFrame([asdict(pos) for pos in self.trading_engine.positions.values()])
                positions_df.to_csv(backup_dir / f"positions_{timestamp}.csv", index=False)
            
            logger.debug(f"数据备份完成: {timestamp}")
            
        except Exception as e:
            logger.error(f"数据备份错误: {e}")
    
    def _send_email_alert(self, alert):
        """发送邮件警报"""
        # 这里需要实现邮件发送功能
        logger.info(f"邮件警报: {alert['message']}")
    
    def _send_slack_alert(self, alert):
        """发送Slack警报"""
        # 这里需要实现Slack通知功能
        logger.info(f"Slack警报: {alert['message']}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，正在优雅关闭...")
        self.stop()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动化交易系统')
    parser.add_argument('--config', '-c', default='trading_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', '-m', choices=['paper', 'live', 'backtest'],
                       help='交易模式')
    parser.add_argument('--symbols', '-s', nargs='+',
                       help='交易标的列表')
    parser.add_argument('--capital', '-k', type=float,
                       help='初始资金')
    
    args = parser.parse_args()
    
    try:
        # 创建交易系统
        system = AutomatedTradingSystem(args.config)
        
        # 覆盖配置参数
        if args.mode:
            system.config.set('trading.mode', args.mode)
        if args.symbols:
            system.config.set('trading.symbols', args.symbols)
        if args.capital:
            system.config.set('trading.initial_capital', args.capital)
        
        # 启动系统
        system.start()
        
        # 保持运行
        try:
            while system.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        return 1
    
    finally:
        if 'system' in locals():
            system.stop()
    
    return 0

if __name__ == "__main__":
    exit(main())