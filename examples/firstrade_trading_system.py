#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firstrade自动交易系统
基于非官方Firstrade API实现自动化交易

注意：这是一个非官方API，使用风险自负
需要安装：pip install firstrade
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
import secrets
from dataclasses import dataclass
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 导入投资策略分析模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入可选的分析模块
try:
    from data_fetcher import DataFetcher
    from sentiment_analyzer import SentimentAnalyzer
    from portfolio_optimizer import PortfolioOptimizer
    import pandas as pd
    HAS_ANALYSIS_MODULES = True
except ImportError:
    print("警告: 分析模块未找到，将使用简化版本")
    HAS_ANALYSIS_MODULES = False
    # 创建简化版本的类
    class DataFetcher:
        def get_stock_data(self, symbol, period="1y"):
            return {"symbol": symbol, "data": []}
    
    class SentimentAnalyzer:
        def analyze_sentiment(self, symbol):
            return {"sentiment": "neutral", "score": 0.0}
    
    class PortfolioOptimizer:
        def optimize_portfolio(self, symbols, data):
            return {"weights": {symbol: 1.0/len(symbols) for symbol in symbols}}

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from firstrade.account import FTSession
    HAS_FIRSTRADE_API = True
except ImportError:
    print("警告: Firstrade API未安装，将使用模拟模式")
    HAS_FIRSTRADE_API = False
    # 创建模拟的FTSession类
    class FTSession:
        def __init__(self, username, password, pin=None):
            self.username = username
            self.password = password
            self.pin = pin
            self.logged_in = False
        
        def is_logged_in(self):
            return self.logged_in
        
        def get_account(self):
            self.logged_in = True
            return {"account_value": 100000, "buying_power": 50000}
        
        def get_positions(self):
            return []
        
        def get_quote(self, symbol):
            return {"symbol": symbol, "price": 100.0, "change": 0.0}

# 尝试导入src模块
try:
    from src.data_fetcher import DataFetcher
    from src.sentiment_analyzer import SentimentAnalyzer
    from src.portfolio_optimizer import PortfolioOptimizer
except ImportError:
    pass  # 如果src模块不存在，继续执行

class RetryConfig:
    """重试配置类"""
    def __init__(self):
        self.max_retries = 3
        self.backoff_factor = 1.0
        self.status_forcelist = [500, 502, 503, 504, 429]
        self.retry_on_timeout = True
        self.retry_on_connection_error = True

class ErrorHandler:
    """错误处理和异常恢复类"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.retry_config = RetryConfig()
        self.error_counts = {}
        self.last_errors = {}
        
    def setup_requests_session(self) -> requests.Session:
        """
        设置带重试机制的requests会话
        
        Returns:
            requests.Session: 配置好的会话对象
        """
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.retry_config.max_retries,
            status_forcelist=self.retry_config.status_forcelist,
            backoff_factor=self.retry_config.backoff_factor,
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def handle_api_error(self, error: Exception, operation: str, max_retries: int = 3) -> bool:
        """
        处理API错误
        
        Args:
            error: 异常对象
            operation: 操作名称
            max_retries: 最大重试次数
            
        Returns:
            bool: 是否应该重试
        """
        error_key = f"{operation}_{type(error).__name__}"
        
        # 记录错误次数
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        self.error_counts[error_key] += 1
        
        # 记录最后一次错误
        self.last_errors[error_key] = {
            'error': str(error),
            'timestamp': datetime.now().isoformat(),
            'count': self.error_counts[error_key]
        }
        
        self.logger.error(f"{operation} 发生错误 (第{self.error_counts[error_key]}次): {str(error)}")
        
        # 判断是否应该重试
        if self.error_counts[error_key] >= max_retries:
            self.logger.error(f"{operation} 达到最大重试次数 ({max_retries})，停止重试")
            return False
        
        # 特定错误类型的处理
        if isinstance(error, requests.exceptions.Timeout):
            self.logger.info(f"{operation} 超时，将在 {self.retry_config.backoff_factor * self.error_counts[error_key]} 秒后重试")
            time.sleep(self.retry_config.backoff_factor * self.error_counts[error_key])
            return True
        
        elif isinstance(error, requests.exceptions.ConnectionError):
            self.logger.info(f"{operation} 连接错误，将在 {self.retry_config.backoff_factor * self.error_counts[error_key]} 秒后重试")
            time.sleep(self.retry_config.backoff_factor * self.error_counts[error_key])
            return True
        
        elif isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response.status_code in self.retry_config.status_forcelist:
                self.logger.info(f"{operation} HTTP错误 {error.response.status_code}，将重试")
                time.sleep(self.retry_config.backoff_factor * self.error_counts[error_key])
                return True
            else:
                self.logger.error(f"{operation} HTTP错误 {error.response.status_code if hasattr(error, 'response') else 'Unknown'}，不重试")
                return False
        
        elif "rate limit" in str(error).lower() or "429" in str(error):
            # API限制错误
            wait_time = min(60, self.retry_config.backoff_factor * (2 ** self.error_counts[error_key]))
            self.logger.info(f"{operation} API限制，将在 {wait_time} 秒后重试")
            time.sleep(wait_time)
            return True
        
        elif "authentication" in str(error).lower() or "unauthorized" in str(error).lower():
            # 认证错误，通常不应该重试
            self.logger.error(f"{operation} 认证错误，不重试")
            return False
        
        else:
            # 其他错误，短暂等待后重试
            self.logger.info(f"{operation} 未知错误，将在 {self.retry_config.backoff_factor} 秒后重试")
            time.sleep(self.retry_config.backoff_factor)
            return True
    
    def execute_with_retry(self, func, operation: str, *args, **kwargs):
        """
        带重试机制执行函数
        
        Args:
            func: 要执行的函数
            operation: 操作名称
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # 重置错误计数（成功执行）
                error_key = f"{operation}_success"
                if error_key in self.error_counts:
                    del self.error_counts[error_key]
                
                return result
                
            except Exception as e:
                last_error = e
                
                if attempt < self.retry_config.max_retries:
                    should_retry = self.handle_api_error(e, operation, self.retry_config.max_retries)
                    if not should_retry:
                        break
                else:
                    self.logger.error(f"{operation} 最终失败: {str(e)}")
                    break
        
        # 如果所有重试都失败，抛出最后一个错误
        raise last_error
    
    def get_error_summary(self) -> Dict:
        """
        获取错误摘要
        
        Returns:
            Dict: 错误摘要信息
        """
        return {
            'error_counts': self.error_counts.copy(),
            'last_errors': self.last_errors.copy(),
            'total_errors': sum(self.error_counts.values())
        }
    
    def reset_error_counts(self):
        """重置错误计数"""
        self.error_counts.clear()
        self.last_errors.clear()
        self.logger.info("错误计数已重置")

class CircuitBreaker:
    """熔断器类 - 防止系统在持续错误时继续执行"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, logger=None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.logger = logger or logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """
        通过熔断器调用函数
        
        Args:
            func: 要调用的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.logger.info("熔断器进入半开状态，尝试恢复")
            else:
                raise Exception(f"熔断器开启中，拒绝执行。将在 {self.recovery_timeout} 秒后尝试恢复")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """判断是否应该尝试重置熔断器"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """成功执行时的处理"""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.logger.info("熔断器已关闭，系统恢复正常")
    
    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(f"熔断器开启，失败次数: {self.failure_count}")
    
    def get_state(self) -> Dict:
        """获取熔断器状态"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'recovery_timeout': self.recovery_timeout
        }

class FirstradeConnector:
    """
    Firstrade API连接器
    处理登录、认证和基础API调用
    """
    
    def __init__(self, username: str, password: str, pin: str = None):
        """
        初始化Firstrade连接器
        
        Args:
            username: Firstrade用户名
            password: Firstrade密码
            pin: PIN码（如果需要）
        """
        self.username = username
        self.password = password
        self.pin = pin
        self.ft = None
        self.is_logged_in = False
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化错误处理器和熔断器
        self.error_handler = ErrorHandler(self.logger)
        self.circuit_breaker = CircuitBreaker(logger=self.logger)
    
    def login(self) -> bool:
        """
        登录Firstrade账户
        
        Returns:
            bool: 登录是否成功
        """
        def _login():
            self.ft = FTSession(
                username=self.username, 
                password=self.password, 
                pin=self.pin
            )
            
            # 检查登录状态
            if hasattr(self.ft, 'is_logged_in') and self.ft.is_logged_in():
                self.is_logged_in = True
                self.logger.info("成功登录Firstrade账户")
                return True
            else:
                # 尝试获取账户信息来验证登录
                try:
                    account_info = self.ft.get_account()
                    if account_info:
                        self.is_logged_in = True
                        self.logger.info(f"成功登录Firstrade账户: {account_info.get('account_number', 'Unknown')}")
                        return True
                except:
                    pass
                
                self.logger.error("登录失败：无法获取账户信息")
                return False
        
        try:
            return self.circuit_breaker.call(_login)
        except Exception as e:
            self.logger.error(f"登录Firstrade时发生错误: {str(e)}")
            self.is_logged_in = False
            return False
    
    def get_account_info(self) -> Dict:
        """
        获取账户信息
        
        Returns:
            Dict: 账户信息
        """
        if not self.is_logged_in:
            raise Exception("请先登录")
        
        try:
            return self.ft.get_account()
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {str(e)}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """
        获取当前持仓
        
        Returns:
            List[Dict]: 持仓列表
        """
        if not self.is_logged_in:
            raise Exception("请先登录")
        
        try:
            return self.ft.get_positions()
        except Exception as e:
            self.logger.error(f"获取持仓信息失败: {str(e)}")
            return []
    
    def get_quote(self, symbol: str) -> Dict:
        """
        获取股票报价
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 报价信息
        """
        if not self.is_logged_in:
            raise Exception("请先登录")
        
        try:
            return self.ft.get_quote(symbol)
        except Exception as e:
            self.logger.error(f"获取{symbol}报价失败: {str(e)}")
            return {}
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        批量获取市场数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            Dict[str, Dict]: 股票代码到市场数据的映射
        """
        if not self.is_logged_in:
            raise Exception("请先登录")
        
        market_data = {}
        for symbol in symbols:
            try:
                quote = self.ft.get_quote(symbol)
                if quote:
                    market_data[symbol] = {
                        'symbol': symbol,
                        'last_price': quote.get('last_price', 0),
                        'bid': quote.get('bid', 0),
                        'ask': quote.get('ask', 0),
                        'volume': quote.get('volume', 0),
                        'change': quote.get('change', 0),
                        'change_percent': quote.get('change_percent', 0),
                        'high': quote.get('high', 0),
                        'low': quote.get('low', 0),
                        'open': quote.get('open', 0),
                        'previous_close': quote.get('previous_close', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.logger.info(f"获取{symbol}市场数据成功: ${quote.get('last_price', 0):.2f}")
                else:
                    self.logger.warning(f"无法获取{symbol}的市场数据")
                    
            except Exception as e:
                self.logger.error(f"获取{symbol}市场数据失败: {str(e)}")
                
            # 避免请求过于频繁
            time.sleep(0.1)
        
        return market_data
    
    def get_watchlist_data(self) -> List[Dict]:
        """
        获取关注列表的市场数据
        
        Returns:
            List[Dict]: 关注列表股票的市场数据
        """
        if not self.is_logged_in:
            raise Exception("请先登录")
        
        try:
            watchlist = self.ft.get_watchlist()
            if not watchlist:
                self.logger.info("关注列表为空")
                return []
            
            symbols = [item.get('symbol') for item in watchlist if item.get('symbol')]
            market_data = self.get_market_data(symbols)
            
            # 将市场数据转换为列表格式
            watchlist_data = []
            for symbol, data in market_data.items():
                watchlist_data.append(data)
            
            return watchlist_data
            
        except Exception as e:
            self.logger.error(f"获取关注列表数据失败: {str(e)}")
            return []
    
    def get_real_time_price(self, symbol: str) -> float:
        """
        获取实时股价
        
        Args:
            symbol: 股票代码
            
        Returns:
            float: 当前股价
        """
        quote = self.get_quote(symbol)
        return quote.get('last_price', 0.0) if quote else 0.0
    
    def monitor_price_changes(self, symbols: List[str], threshold: float = 0.05, duration: int = 300) -> Dict:
        """
        监控股价变化
        
        Args:
            symbols: 要监控的股票代码列表
            threshold: 价格变化阈值（百分比）
            duration: 监控持续时间（秒）
            
        Returns:
            Dict: 监控结果
        """
        if not self.is_logged_in:
            raise Exception("请先登录")
        
        self.logger.info(f"开始监控{len(symbols)}只股票的价格变化，阈值: {threshold*100:.1f}%")
        
        # 获取初始价格
        initial_prices = {}
        for symbol in symbols:
            price = self.get_real_time_price(symbol)
            if price > 0:
                initial_prices[symbol] = price
        
        alerts = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            for symbol in symbols:
                if symbol not in initial_prices:
                    continue
                
                current_price = self.get_real_time_price(symbol)
                if current_price <= 0:
                    continue
                
                initial_price = initial_prices[symbol]
                change_percent = (current_price - initial_price) / initial_price
                
                if abs(change_percent) >= threshold:
                    alert = {
                        'symbol': symbol,
                        'initial_price': initial_price,
                        'current_price': current_price,
                        'change_percent': change_percent,
                        'timestamp': datetime.now().isoformat(),
                        'alert_type': 'price_change'
                    }
                    alerts.append(alert)
                    self.logger.warning(f"价格变化警报: {symbol} {change_percent*100:+.2f}% (${initial_price:.2f} -> ${current_price:.2f})")
                    
                    # 更新初始价格，避免重复警报
                    initial_prices[symbol] = current_price
            
            # 等待一段时间再检查
            time.sleep(10)
        
        return {
            'alerts': alerts,
            'monitored_symbols': symbols,
            'duration': duration,
            'threshold': threshold
        }

class FirstradeOrderExecutor:
    """
    Firstrade订单执行器
    处理买入、卖出等交易操作
    """
    
    def __init__(self, connector: FirstradeConnector):
        """
        初始化订单执行器
        
        Args:
            connector: Firstrade连接器实例
        """
        self.connector = connector
        self.logger = logging.getLogger(__name__)
    
    def place_market_order(self, symbol: str, quantity: int, side: str, dry_run: bool = True) -> Dict:
        """
        下市价单
        
        Args:
            symbol: 股票代码
            quantity: 数量
            side: 'buy' 或 'sell'
            dry_run: 是否为模拟交易
            
        Returns:
            Dict: 订单结果
        """
        if not self.connector.is_logged_in:
            raise Exception("请先登录")
        
        try:
            if dry_run:
                # 模拟交易
                quote = self.connector.get_quote(symbol)
                price = quote.get('last_price', 0)
                
                order_result = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': side,
                    'order_type': 'market',
                    'price': price,
                    'total_value': price * quantity,
                    'status': 'simulated',
                    'timestamp': datetime.now().isoformat(),
                    'dry_run': True
                }
                
                self.logger.info(f"模拟{side}订单: {symbol} x{quantity} @ ${price:.2f}")
                return order_result
            else:
                # 真实交易
                if side.lower() == 'buy':
                    result = self.connector.ft.place_order(
                        symbol=symbol,
                        quantity=quantity,
                        side='buy',
                        order_type='market'
                    )
                elif side.lower() == 'sell':
                    result = self.connector.ft.place_order(
                        symbol=symbol,
                        quantity=quantity,
                        side='sell',
                        order_type='market'
                    )
                else:
                    raise ValueError("side必须是'buy'或'sell'")
                
                self.logger.info(f"真实{side}订单已提交: {symbol} x{quantity}")
                return result
                
        except Exception as e:
            self.logger.error(f"下单失败: {str(e)}")
            return {'error': str(e)}
    
    def place_limit_order(self, symbol: str, quantity: int, side: str, price: float, dry_run: bool = True) -> Dict:
        """
        下限价单
        
        Args:
            symbol: 股票代码
            quantity: 数量
            side: 'buy' 或 'sell'
            price: 限价
            dry_run: 是否为模拟交易
            
        Returns:
            Dict: 订单结果
        """
        if not self.connector.is_logged_in:
            raise Exception("请先登录")
        
        try:
            if dry_run:
                # 模拟交易
                order_result = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': side,
                    'order_type': 'limit',
                    'price': price,
                    'total_value': price * quantity,
                    'status': 'simulated',
                    'timestamp': datetime.now().isoformat(),
                    'dry_run': True
                }
                
                self.logger.info(f"模拟{side}限价单: {symbol} x{quantity} @ ${price:.2f}")
                return order_result
            else:
                # 真实交易
                if side.lower() == 'buy':
                    result = self.connector.ft.place_order(
                        symbol=symbol,
                        quantity=quantity,
                        side='buy',
                        order_type='limit',
                        price=price
                    )
                elif side.lower() == 'sell':
                    result = self.connector.ft.place_order(
                        symbol=symbol,
                        quantity=quantity,
                        side='sell',
                        order_type='limit',
                        price=price
                    )
                else:
                    raise ValueError("side必须是'buy'或'sell'")
                
                self.logger.info(f"真实{side}限价单已提交: {symbol} x{quantity} @ ${price:.2f}")
                return result
                
        except Exception as e:
            self.logger.error(f"下限价单失败: {str(e)}")
            return {'error': str(e)}
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        查询订单状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            Dict: 订单状态信息
        """
        if not self.connector.is_logged_in:
            raise Exception("请先登录")
        
        try:
            order_status = self.connector.ft.get_order(order_id)
            self.logger.info(f"查询订单状态: {order_id}")
            return order_status
        except Exception as e:
            self.logger.error(f"查询订单状态失败: {str(e)}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str, dry_run: bool = True) -> Dict:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            dry_run: 是否为模拟交易
            
        Returns:
            Dict: 取消结果
        """
        if not self.connector.is_logged_in:
            raise Exception("请先登录")
        
        try:
            if dry_run:
                cancel_result = {
                    'order_id': order_id,
                    'action': 'cancel',
                    'status': 'simulated',
                    'timestamp': datetime.now().isoformat(),
                    'dry_run': True
                }
                self.logger.info(f"模拟取消订单: {order_id}")
                return cancel_result
            else:
                result = self.connector.ft.cancel_order(order_id)
                self.logger.info(f"真实取消订单: {order_id}")
                return result
        except Exception as e:
            self.logger.error(f"取消订单失败: {str(e)}")
            return {'error': str(e)}
    
    def get_open_orders(self) -> List[Dict]:
        """
        获取未完成订单列表
        
        Returns:
            List[Dict]: 未完成订单列表
        """
        if not self.connector.is_logged_in:
            raise Exception("请先登录")
        
        try:
            open_orders = self.connector.ft.get_orders()
            self.logger.info(f"获取到{len(open_orders)}个未完成订单")
            return open_orders
        except Exception as e:
            self.logger.error(f"获取未完成订单失败: {str(e)}")
            return []
    
    def place_stop_loss_order(self, symbol: str, quantity: int, stop_price: float, dry_run: bool = True) -> Dict:
        """
        下止损单
        
        Args:
            symbol: 股票代码
            quantity: 数量
            stop_price: 止损价格
            dry_run: 是否为模拟交易
            
        Returns:
            Dict: 订单结果
        """
        if not self.connector.is_logged_in:
            raise Exception("请先登录")
        
        try:
            if dry_run:
                order_result = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': 'sell',
                    'order_type': 'stop_loss',
                    'stop_price': stop_price,
                    'status': 'simulated',
                    'timestamp': datetime.now().isoformat(),
                    'dry_run': True
                }
                self.logger.info(f"模拟止损单: {symbol} x{quantity} @ ${stop_price:.2f}")
                return order_result
            else:
                result = self.connector.ft.place_order(
                    symbol=symbol,
                    quantity=quantity,
                    side='sell',
                    order_type='stop',
                    stop_price=stop_price
                )
                self.logger.info(f"真实止损单已提交: {symbol} x{quantity} @ ${stop_price:.2f}")
                return result
        except Exception as e:
            self.logger.error(f"下止损单失败: {str(e)}")
            return {'error': str(e)}
    
    def execute_batch_orders(self, orders: List[Dict], dry_run: bool = True) -> List[Dict]:
        """
        批量执行订单
        
        Args:
            orders: 订单列表，每个订单包含symbol, quantity, side, order_type等信息
            dry_run: 是否为模拟交易
            
        Returns:
            List[Dict]: 执行结果列表
        """
        if not self.connector.is_logged_in:
            raise Exception("请先登录")
        
        results = []
        self.logger.info(f"开始批量执行{len(orders)}个订单")
        
        for i, order in enumerate(orders):
            try:
                symbol = order.get('symbol')
                quantity = order.get('quantity')
                side = order.get('side')
                order_type = order.get('order_type', 'market')
                price = order.get('price')
                
                if order_type == 'market':
                    result = self.place_market_order(symbol, quantity, side, dry_run)
                elif order_type == 'limit':
                    if not price:
                        raise ValueError(f"限价单必须指定价格: {symbol}")
                    result = self.place_limit_order(symbol, quantity, side, price, dry_run)
                else:
                    raise ValueError(f"不支持的订单类型: {order_type}")
                
                result['batch_index'] = i
                results.append(result)
                
                # 避免请求过于频繁
                if not dry_run:
                    time.sleep(1)
                    
            except Exception as e:
                error_result = {
                    'batch_index': i,
                    'symbol': order.get('symbol', 'Unknown'),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
                self.logger.error(f"批量订单执行失败 [{i}]: {str(e)}")
        
        success_count = len([r for r in results if 'error' not in r])
        self.logger.info(f"批量订单执行完成: {success_count}/{len(orders)} 成功")
        
        return results
    
    def calculate_position_size(self, symbol: str, target_weight: float, total_portfolio_value: float) -> int:
        """
        计算仓位大小
        
        Args:
            symbol: 股票代码
            target_weight: 目标权重
            total_portfolio_value: 投资组合总价值
            
        Returns:
            int: 应购买的股数
        """
        try:
            current_price = self.connector.get_real_time_price(symbol)
            if current_price <= 0:
                self.logger.warning(f"无法获取{symbol}的当前价格")
                return 0
            
            target_value = total_portfolio_value * target_weight
            shares = int(target_value / current_price)
            
            self.logger.info(f"{symbol} 目标权重: {target_weight:.2%}, 目标价值: ${target_value:.2f}, 股数: {shares}")
            return shares
            
        except Exception as e:
            self.logger.error(f"计算{symbol}仓位大小失败: {str(e)}")
            return 0

class FirstradeTradingSystem:
    """
    Firstrade自动交易系统主类
    整合投资策略分析和真实交易执行
    """
    
    def __init__(self, username: str, password: str, pin: str = None, dry_run: bool = True):
        """
        初始化交易系统
        
        Args:
            username: Firstrade用户名
            password: Firstrade密码
            pin: PIN码（如果需要）
            dry_run: 是否为模拟交易模式
        """
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
        
        # 初始化Firstrade连接器
        self.connector = FirstradeConnector(username, password, pin)
        self.executor = FirstradeOrderExecutor(self.connector)
        
        # 初始化分析模块
        self.data_fetcher = DataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # 交易记录
        self.trade_history = []
    
    def connect(self) -> bool:
        """
        连接到Firstrade
        
        Returns:
            bool: 连接是否成功
        """
        return self.connector.login()
    
    def get_portfolio_status(self) -> Dict:
        """
        获取投资组合状态
        
        Returns:
            Dict: 投资组合信息
        """
        try:
            account_info = self.connector.get_account_info()
            positions = self.connector.get_positions()
            
            return {
                'account_info': account_info,
                'positions': positions,
                'total_value': account_info.get('total_value', 0),
                'cash_balance': account_info.get('cash_balance', 0),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"获取投资组合状态失败: {str(e)}")
            return {}
    
    def get_account_balance(self) -> Dict:
        """
        获取详细的账户余额信息
        
        Returns:
            Dict: 账户余额详情
        """
        try:
            account_info = self.connector.get_account_info()
            
            balance_info = {
                'account_number': account_info.get('account_number', ''),
                'total_value': account_info.get('total_value', 0),
                'cash_balance': account_info.get('cash_balance', 0),
                'buying_power': account_info.get('buying_power', 0),
                'market_value': account_info.get('market_value', 0),
                'day_change': account_info.get('day_change', 0),
                'day_change_percent': account_info.get('day_change_percent', 0),
                'total_gain_loss': account_info.get('total_gain_loss', 0),
                'total_gain_loss_percent': account_info.get('total_gain_loss_percent', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"账户总价值: ${balance_info['total_value']:,.2f}")
            self.logger.info(f"现金余额: ${balance_info['cash_balance']:,.2f}")
            self.logger.info(f"购买力: ${balance_info['buying_power']:,.2f}")
            
            return balance_info
            
        except Exception as e:
            self.logger.error(f"获取账户余额失败: {str(e)}")
            return {}
    
    def get_detailed_positions(self) -> List[Dict]:
        """
        获取详细的持仓信息
        
        Returns:
            List[Dict]: 详细持仓列表
        """
        try:
            positions = self.connector.get_positions()
            detailed_positions = []
            
            for position in positions:
                symbol = position.get('symbol', '')
                if not symbol:
                    continue
                
                # 获取实时报价
                quote = self.connector.get_quote(symbol)
                current_price = quote.get('last_price', 0) if quote else 0
                
                # 计算详细信息
                quantity = position.get('quantity', 0)
                avg_cost = position.get('average_cost', 0)
                market_value = current_price * quantity
                cost_basis = avg_cost * quantity
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
                
                detailed_position = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'average_cost': avg_cost,
                    'current_price': current_price,
                    'market_value': market_value,
                    'cost_basis': cost_basis,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_percent': unrealized_pnl_percent,
                    'day_change': quote.get('change', 0) if quote else 0,
                    'day_change_percent': quote.get('change_percent', 0) if quote else 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                detailed_positions.append(detailed_position)
                
                self.logger.info(f"{symbol}: {quantity}股 @ ${current_price:.2f} "
                               f"(成本: ${avg_cost:.2f}, 盈亏: {unrealized_pnl_percent:+.2f}%)")
            
            return detailed_positions
            
        except Exception as e:
            self.logger.error(f"获取详细持仓失败: {str(e)}")
            return []
    
    def get_portfolio_allocation(self) -> Dict:
        """
        获取投资组合配置分析
        
        Returns:
            Dict: 投资组合配置信息
        """
        try:
            positions = self.get_detailed_positions()
            account_balance = self.get_account_balance()
            
            total_market_value = sum(pos['market_value'] for pos in positions)
            cash_balance = account_balance.get('cash_balance', 0)
            total_portfolio_value = total_market_value + cash_balance
            
            # 计算各股票权重
            allocations = []
            for position in positions:
                weight = (position['market_value'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                allocations.append({
                    'symbol': position['symbol'],
                    'market_value': position['market_value'],
                    'weight': weight,
                    'unrealized_pnl': position['unrealized_pnl'],
                    'unrealized_pnl_percent': position['unrealized_pnl_percent']
                })
            
            # 按权重排序
            allocations.sort(key=lambda x: x['weight'], reverse=True)
            
            cash_weight = (cash_balance / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            
            portfolio_summary = {
                'total_portfolio_value': total_portfolio_value,
                'total_market_value': total_market_value,
                'cash_balance': cash_balance,
                'cash_weight': cash_weight,
                'stock_allocations': allocations,
                'number_of_positions': len(positions),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"投资组合总价值: ${total_portfolio_value:,.2f}")
            self.logger.info(f"股票市值: ${total_market_value:,.2f} ({100-cash_weight:.1f}%)")
            self.logger.info(f"现金余额: ${cash_balance:,.2f} ({cash_weight:.1f}%)")
            
            return portfolio_summary
            
        except Exception as e:
            self.logger.error(f"获取投资组合配置失败: {str(e)}")
            return {}
    
    def get_trading_history(self, days: int = 30) -> List[Dict]:
        """
        获取交易历史记录
        
        Args:
            days: 查询天数
            
        Returns:
            List[Dict]: 交易历史列表
        """
        try:
            # 计算查询日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 获取交易历史（这里假设API支持日期范围查询）
            trading_history = self.connector.ft.get_transactions(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            processed_history = []
            for transaction in trading_history:
                processed_transaction = {
                    'transaction_id': transaction.get('id', ''),
                    'symbol': transaction.get('symbol', ''),
                    'action': transaction.get('action', ''),
                    'quantity': transaction.get('quantity', 0),
                    'price': transaction.get('price', 0),
                    'total_amount': transaction.get('total_amount', 0),
                    'fees': transaction.get('fees', 0),
                    'transaction_date': transaction.get('date', ''),
                    'status': transaction.get('status', ''),
                    'order_type': transaction.get('order_type', '')
                }
                processed_history.append(processed_transaction)
            
            self.logger.info(f"获取到{len(processed_history)}条交易记录（过去{days}天）")
            return processed_history
            
        except Exception as e:
            self.logger.error(f"获取交易历史失败: {str(e)}")
            return []
    
    def calculate_portfolio_performance(self, days: int = 30) -> Dict:
        """
        计算投资组合表现
        
        Args:
            days: 计算天数
            
        Returns:
            Dict: 投资组合表现数据
        """
        try:
            # 获取历史交易数据
            trading_history = self.get_trading_history(days)
            
            # 计算总收益
            total_pnl = sum([trade.get('pnl', 0) for trade in trading_history])
            
            # 获取当前投资组合价值
            portfolio_status = self.get_portfolio_status()
            current_value = portfolio_status.get('total_value', 0)
            
            # 计算收益率
            initial_value = current_value - total_pnl
            return_rate = (total_pnl / initial_value * 100) if initial_value > 0 else 0
            
            # 计算胜率
            winning_trades = len([trade for trade in trading_history if trade.get('pnl', 0) > 0])
            total_trades = len(trading_history)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_pnl': total_pnl,
                'return_rate': return_rate,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'current_value': current_value,
                'period_days': days
            }
            
        except Exception as e:
            self.logger.error(f"计算投资组合表现失败: {str(e)}")
            return {
                'total_pnl': 0,
                'return_rate': 0,
                'win_rate': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'current_value': 0,
                'period_days': days
            }
    
    def run_investment_analysis(self, symbols: List[str] = None) -> Dict:
        """
        运行投资策略分析
        
        Args:
            symbols: 股票代码列表，如果为None则使用默认列表
            
        Returns:
            Dict: 分析结果
        """
        try:
            self.logger.info("开始运行投资策略分析...")
            
            # 使用默认股票列表
            if symbols is None:
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'JNJ', 'JPM', 'V']
            
            # 获取市场数据
            self.logger.info("获取市场数据...")
            market_data = self.data_fetcher.fetch_stock_data(symbols)
            
            # 进行情感分析
            self.logger.info("进行市场情感分析...")
            sentiment_results = {}
            for symbol in symbols:
                sentiment = self.sentiment_analyzer.analyze_stock_sentiment(symbol)
                sentiment_results[symbol] = sentiment
            
            # 进行投资组合优化
            self.logger.info("进行投资组合优化...")
            optimization_result = self.portfolio_optimizer.optimize_portfolio(symbols)
            
            # 生成交易信号
            trading_signals = []
            for symbol in symbols:
                if symbol in optimization_result.get('weights', {}):
                    weight = optimization_result['weights'][symbol]
                    sentiment = sentiment_results.get(symbol, {})
                    
                    # 根据权重和情感分析生成交易信号
                    if weight > 0.05 and sentiment.get('score', 0) > 0.1:
                        signal = {
                            'symbol': symbol,
                            'action': 'BUY',
                            'target_weight': weight,
                            'confidence': sentiment.get('confidence', 0.5),
                            'sentiment_score': sentiment.get('score', 0),
                            'reason': f"优化权重: {weight:.3f}, 情感得分: {sentiment.get('score', 0):.3f}"
                        }
                        trading_signals.append(signal)
            
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': symbols,
                'market_data': market_data,
                'sentiment_analysis': sentiment_results,
                'portfolio_optimization': optimization_result,
                'trading_signals': trading_signals,
                'signal_count': len(trading_signals)
            }
            
            self.logger.info(f"投资策略分析完成，生成 {len(trading_signals)} 个交易信号")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"投资策略分析失败: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'trading_signals': [],
                'signal_count': 0
            }
    
    def execute_trading_signals(self, trading_signals: List[Dict], dry_run: bool = True) -> Dict:
        """
        执行交易信号
        
        Args:
            trading_signals: 交易信号列表
            dry_run: 是否为模拟交易
            
        Returns:
            Dict: 执行结果
        """
        try:
            self.logger.info(f"开始执行 {len(trading_signals)} 个交易信号...")
            
            # 获取当前投资组合状态
            portfolio_status = self.get_portfolio_status()
            total_value = portfolio_status.get('total_value', 100000)  # 默认10万美元
            
            executed_orders = []
            failed_orders = []
            
            for signal in trading_signals:
                try:
                    symbol = signal['symbol']
                    action = signal['action']
                    target_weight = signal['target_weight']
                    
                    # 计算目标仓位
                    target_value = total_value * target_weight
                    
                    # 获取当前股价
                    quote = self.connector.get_quote(symbol)
                    current_price = quote.get('price', 0)
                    
                    if current_price <= 0:
                        self.logger.warning(f"无法获取 {symbol} 的有效价格")
                        continue
                    
                    # 计算需要购买的股数
                    quantity = int(target_value / current_price)
                    
                    if quantity <= 0:
                        self.logger.warning(f"{symbol} 计算出的股数为0，跳过")
                        continue
                    
                    # 执行订单
                    if action.upper() == 'BUY':
                        order_result = self.order_executor.place_market_order(
                            symbol=symbol,
                            quantity=quantity,
                            side='buy',
                            dry_run=dry_run
                        )
                    elif action.upper() == 'SELL':
                        order_result = self.order_executor.place_market_order(
                            symbol=symbol,
                            quantity=quantity,
                            side='sell',
                            dry_run=dry_run
                        )
                    else:
                        self.logger.warning(f"未知的交易动作: {action}")
                        continue
                    
                    if order_result.get('success', False):
                        executed_orders.append({
                            'signal': signal,
                            'order_result': order_result,
                            'quantity': quantity,
                            'estimated_value': quantity * current_price
                        })
                        self.logger.info(f"成功执行 {symbol} {action} 订单，数量: {quantity}")
                    else:
                        failed_orders.append({
                            'signal': signal,
                            'error': order_result.get('error', '未知错误'),
                            'quantity': quantity
                        })
                        self.logger.error(f"执行 {symbol} {action} 订单失败: {order_result.get('error', '未知错误')}")
                        
                except Exception as e:
                    self.logger.error(f"处理交易信号失败 {signal}: {str(e)}")
                    failed_orders.append({
                        'signal': signal,
                        'error': str(e)
                    })
            
            execution_result = {
                'timestamp': datetime.now().isoformat(),
                'total_signals': len(trading_signals),
                'executed_orders': executed_orders,
                'failed_orders': failed_orders,
                'success_count': len(executed_orders),
                'failure_count': len(failed_orders),
                'total_investment': sum([order['estimated_value'] for order in executed_orders]),
                'dry_run': dry_run
            }
            
            self.logger.info(f"交易信号执行完成: 成功 {len(executed_orders)}, 失败 {len(failed_orders)}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"执行交易信号失败: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'executed_orders': [],
                'failed_orders': trading_signals,
                'success_count': 0,
                'failure_count': len(trading_signals),
                'total_investment': 0,
                'dry_run': dry_run
            }
    
    def run_automated_trading(self, symbols: List[str] = None, dry_run: bool = True) -> Dict:
        """
        运行完整的自动化交易流程
        
        Args:
            symbols: 股票代码列表
            dry_run: 是否为模拟交易
            
        Returns:
            Dict: 完整的交易结果
        """
        try:
            self.logger.info("开始运行自动化交易系统...")
            
            # 1. 运行投资策略分析
            analysis_result = self.run_investment_analysis(symbols)
            
            if analysis_result.get('signal_count', 0) == 0:
                self.logger.warning("没有生成交易信号，停止交易")
                return {
                    'timestamp': datetime.now().isoformat(),
                    'analysis_result': analysis_result,
                    'execution_result': None,
                    'status': 'no_signals'
                }
            
            # 2. 执行交易信号
            trading_signals = analysis_result.get('trading_signals', [])
            execution_result = self.execute_trading_signals(trading_signals, dry_run)
            
            # 3. 保存交易结果
            complete_result = {
                'timestamp': datetime.now().isoformat(),
                'analysis_result': analysis_result,
                'execution_result': execution_result,
                'status': 'completed',
                'summary': {
                    'total_signals': len(trading_signals),
                    'successful_trades': execution_result.get('success_count', 0),
                    'failed_trades': execution_result.get('failure_count', 0),
                    'total_investment': execution_result.get('total_investment', 0),
                    'dry_run': dry_run
                }
            }
            
            # 保存结果到文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"results/automated_trading_result_{timestamp}.json"
            
            os.makedirs("results", exist_ok=True)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(complete_result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"自动化交易完成，结果已保存到: {result_file}")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"自动化交易失败: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }

def main():
    """
    主函数 - 演示Firstrade自动化交易系统
    """
    print("=== Firstrade 自动化交易系统 ===")
    print()
    
    # 配置参数
    USERNAME = os.getenv('FIRSTRADE_USERNAME', 'your_username')
    PASSWORD = os.getenv('FIRSTRADE_PASSWORD', 'your_password')
    PIN = os.getenv('FIRSTRADE_PIN', 'your_pin')
    
    # 创建交易系统实例（默认为模拟交易模式）
    trading_system = FirstradeTradingSystem(
        username=USERNAME,
        password=PASSWORD,
        pin=PIN,
        dry_run=True  # 设置为False进行真实交易
    )
    
    try:
        # 1. 连接到Firstrade
        print("1. 连接到Firstrade...")
        if not trading_system.connect():
            print("❌ 连接失败，请检查账户信息")
            return
        print("✅ 连接成功")
        
        # 2. 获取账户信息
        print("\n2. 获取账户信息...")
        account_balance = trading_system.get_account_balance()
        print(f"账户余额: ${account_balance.get('total_value', 0):,.2f}")
        print(f"可用现金: ${account_balance.get('cash_balance', 0):,.2f}")
        
        # 3. 获取当前持仓
        print("\n3. 当前持仓:")
        positions = trading_system.get_detailed_positions()
        if positions:
            for pos in positions[:5]:  # 显示前5个持仓
                print(f"  {pos['symbol']}: {pos['quantity']} 股, 价值: ${pos['market_value']:,.2f}")
        else:
            print("  暂无持仓")
        
        # 4. 运行投资策略分析
        print("\n4. 运行投资策略分析...")
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'JNJ', 'JPM', 'V']
        analysis_result = trading_system.run_investment_analysis(symbols)
        
        signal_count = analysis_result.get('signal_count', 0)
        print(f"✅ 分析完成，生成 {signal_count} 个交易信号")
        
        if signal_count > 0:
            print("\n交易信号:")
            for i, signal in enumerate(analysis_result.get('trading_signals', [])[:5], 1):
                print(f"  {i}. {signal['symbol']} - {signal['action']} - 权重: {signal['target_weight']:.3f} - 置信度: {signal['confidence']:.3f}")
        
        # 5. 执行自动化交易
        print(f"\n5. 执行自动化交易 ({'模拟交易' if trading_system.dry_run else '真实交易'})...")
        
        if signal_count > 0:
            # 询问用户是否继续
            if trading_system.dry_run:
                proceed = input("是否继续执行模拟交易? (y/n): ").lower().strip()
            else:
                proceed = input("⚠️  警告：这将执行真实交易！是否继续? (y/n): ").lower().strip()
            
            if proceed == 'y':
                # 执行完整的自动化交易流程
                complete_result = trading_system.run_automated_trading(symbols, trading_system.dry_run)
                
                # 显示执行结果
                summary = complete_result.get('summary', {})
                print(f"\n✅ 自动化交易完成:")
                print(f"  总信号数: {summary.get('total_signals', 0)}")
                print(f"  成功交易: {summary.get('successful_trades', 0)}")
                print(f"  失败交易: {summary.get('failed_trades', 0)}")
                print(f"  总投资额: ${summary.get('total_investment', 0):,.2f}")
                
                # 显示执行的订单
                execution_result = complete_result.get('execution_result', {})
                executed_orders = execution_result.get('executed_orders', [])
                
                if executed_orders:
                    print(f"\n执行的订单:")
                    for order in executed_orders:
                        signal = order['signal']
                        print(f"  {signal['symbol']}: {signal['action']} {order['quantity']} 股, 价值: ${order['estimated_value']:,.2f}")
                
                # 保存结果文件路径
                if 'analysis_result' in complete_result:
                    timestamp = complete_result['timestamp'][:8] + '_' + complete_result['timestamp'][11:19].replace(':', '')
                    result_file = f"results/automated_trading_result_{timestamp}.json"
                    print(f"\n📄 详细结果已保存到: {result_file}")
            else:
                print("❌ 用户取消交易")
        else:
            print("❌ 没有生成交易信号，无需执行交易")
        
        # 6. 显示投资组合表现
        print("\n6. 投资组合表现 (过去30天):")
        performance = trading_system.calculate_portfolio_performance(30)
        print(f"  总收益: ${performance.get('total_pnl', 0):,.2f}")
        print(f"  收益率: {performance.get('return_rate', 0):.2f}%")
        print(f"  胜率: {performance.get('win_rate', 0):.2f}%")
        print(f"  交易次数: {performance.get('total_trades', 0)}")
        
    except Exception as e:
        print(f"❌ 系统运行出错: {str(e)}")
        logging.error(f"系统运行出错: {str(e)}")
    
    print(f"\n注意：当前为{'模拟交易' if trading_system.dry_run else '真实交易'}模式")
    print("=== 系统运行完成 ===")

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class SecurityConfig:
    """安全配置类"""
    max_daily_trades: int = 10  # 每日最大交易次数
    max_position_size: float = 0.15  # 单个股票最大仓位比例
    max_daily_loss: float = 0.05  # 每日最大亏损比例
    max_total_exposure: float = 0.8  # 最大总仓位比例
    min_cash_reserve: float = 0.1  # 最小现金储备比例
    stop_loss_threshold: float = 0.08  # 止损阈值
    take_profit_threshold: float = 0.20  # 止盈阈值
    max_order_value: float = 50000  # 单笔订单最大金额
    allowed_symbols: List[str] = None  # 允许交易的股票列表
    forbidden_symbols: List[str] = None  # 禁止交易的股票列表
    trading_hours_only: bool = True  # 仅在交易时间内交易
    require_confirmation: bool = False  # 是否需要确认
    enable_circuit_breaker: bool = True  # 启用熔断机制

class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: SecurityConfig):
        """
        初始化风险管理器
        
        Args:
            config: 安全配置
        """
        self.config = config
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.circuit_breaker_triggered = False
        self.logger = logging.getLogger(__name__)
        
        # 初始化允许和禁止的股票列表
        if self.config.allowed_symbols is None:
            self.config.allowed_symbols = []
        if self.config.forbidden_symbols is None:
            self.config.forbidden_symbols = []
    
    def reset_daily_counters(self):
        """重置每日计数器"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trades_count = 0
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            self.circuit_breaker_triggered = False
            self.logger.info("每日风险计数器已重置")
    
    def check_trading_hours(self) -> bool:
        """
        检查是否在交易时间内
        
        Returns:
            bool: 是否在交易时间内
        """
        if not self.config.trading_hours_only:
            return True
        
        now = datetime.now()
        # 美股交易时间：周一到周五 9:30-16:00 EST
        # 这里简化处理，实际应该考虑时区和节假日
        if now.weekday() >= 5:  # 周末
            return False
        
        # 简化的交易时间检查（需要根据实际时区调整）
        trading_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        trading_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return trading_start <= now <= trading_end
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        验证股票代码是否允许交易
        
        Args:
            symbol: 股票代码
            
        Returns:
            Tuple[bool, str]: (是否允许, 错误信息)
        """
        # 检查禁止列表
        if symbol in self.config.forbidden_symbols:
            return False, f"股票 {symbol} 在禁止交易列表中"
        
        # 检查允许列表（如果设置了）
        if self.config.allowed_symbols and symbol not in self.config.allowed_symbols:
            return False, f"股票 {symbol} 不在允许交易列表中"
        
        return True, ""
    
    def validate_order_size(self, symbol: str, quantity: int, price: float, 
                          portfolio_value: float) -> Tuple[bool, str]:
        """
        验证订单大小是否符合风险控制要求
        
        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格
            portfolio_value: 投资组合总价值
            
        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        order_value = quantity * price
        
        # 检查单笔订单最大金额
        if order_value > self.config.max_order_value:
            return False, f"订单金额 ${order_value:,.2f} 超过最大限制 ${self.config.max_order_value:,.2f}"
        
        # 检查单个股票最大仓位比例
        max_position_value = portfolio_value * self.config.max_position_size
        if order_value > max_position_value:
            return False, f"订单金额超过单个股票最大仓位限制 ({self.config.max_position_size*100:.1f}%)"
        
        return True, ""
    
    def validate_portfolio_exposure(self, current_positions: List[Dict], 
                                  new_order_value: float, 
                                  portfolio_value: float) -> Tuple[bool, str]:
        """
        验证投资组合总仓位是否超限
        
        Args:
            current_positions: 当前持仓
            new_order_value: 新订单价值
            portfolio_value: 投资组合总价值
            
        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        # 计算当前总仓位价值
        current_exposure = sum(pos.get('market_value', 0) for pos in current_positions)
        
        # 计算新订单后的总仓位
        new_total_exposure = current_exposure + new_order_value
        exposure_ratio = new_total_exposure / portfolio_value
        
        if exposure_ratio > self.config.max_total_exposure:
            return False, f"总仓位比例 {exposure_ratio*100:.1f}% 超过最大限制 {self.config.max_total_exposure*100:.1f}%"
        
        return True, ""
    
    def check_daily_limits(self, order_value: float = 0) -> Tuple[bool, str]:
        """
        检查每日交易限制
        
        Args:
            order_value: 订单价值（用于计算潜在亏损）
            
        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        self.reset_daily_counters()
        
        # 检查每日交易次数
        if self.daily_trades_count >= self.config.max_daily_trades:
            return False, f"已达到每日最大交易次数限制 ({self.config.max_daily_trades})"
        
        # 检查熔断机制
        if self.circuit_breaker_triggered:
            return False, "熔断机制已触发，今日停止交易"
        
        return True, ""
    
    def check_stop_loss(self, symbol: str, current_price: float, 
                       avg_cost: float) -> Tuple[bool, str]:
        """
        检查是否触发止损
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            avg_cost: 平均成本
            
        Returns:
            Tuple[bool, str]: (是否触发止损, 信息)
        """
        if avg_cost <= 0:
            return False, ""
        
        loss_ratio = (avg_cost - current_price) / avg_cost
        
        if loss_ratio >= self.config.stop_loss_threshold:
            return True, f"{symbol} 触发止损：当前亏损 {loss_ratio*100:.2f}%"
        
        return False, ""
    
    def check_take_profit(self, symbol: str, current_price: float, 
                         avg_cost: float) -> Tuple[bool, str]:
        """
        检查是否触发止盈
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            avg_cost: 平均成本
            
        Returns:
            Tuple[bool, str]: (是否触发止盈, 信息)
        """
        if avg_cost <= 0:
            return False, ""
        
        profit_ratio = (current_price - avg_cost) / avg_cost
        
        if profit_ratio >= self.config.take_profit_threshold:
            return True, f"{symbol} 触发止盈：当前收益 {profit_ratio*100:.2f}%"
        
        return False, ""
    
    def update_daily_pnl(self, pnl: float, portfolio_value: float):
        """
        更新每日盈亏并检查熔断
        
        Args:
            pnl: 盈亏金额
            portfolio_value: 投资组合总价值
        """
        self.daily_pnl += pnl
        daily_loss_ratio = abs(self.daily_pnl) / portfolio_value
        
        # 检查是否触发熔断
        if self.daily_pnl < 0 and daily_loss_ratio >= self.config.max_daily_loss:
            self.circuit_breaker_triggered = True
            self.logger.warning(f"触发熔断机制：每日亏损 {daily_loss_ratio*100:.2f}% "
                              f"超过限制 {self.config.max_daily_loss*100:.2f}%")
    
    def increment_trade_count(self):
        """增加交易计数"""
        self.daily_trades_count += 1
    
    def get_risk_assessment(self, symbol: str, quantity: int, price: float,
                          current_positions: List[Dict], 
                          portfolio_value: float) -> Dict:
        """
        综合风险评估
        
        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格
            current_positions: 当前持仓
            portfolio_value: 投资组合总价值
            
        Returns:
            Dict: 风险评估结果
        """
        assessment = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'order_value': quantity * price,
            'risk_level': RiskLevel.LOW,
            'warnings': [],
            'errors': [],
            'approved': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # 检查交易时间
        if not self.check_trading_hours():
            assessment['errors'].append("当前不在交易时间内")
            assessment['approved'] = False
        
        # 验证股票代码
        symbol_valid, symbol_error = self.validate_symbol(symbol)
        if not symbol_valid:
            assessment['errors'].append(symbol_error)
            assessment['approved'] = False
        
        # 验证订单大小
        size_valid, size_error = self.validate_order_size(symbol, quantity, price, portfolio_value)
        if not size_valid:
            assessment['errors'].append(size_error)
            assessment['approved'] = False
        
        # 验证投资组合仓位
        exposure_valid, exposure_error = self.validate_portfolio_exposure(
            current_positions, quantity * price, portfolio_value
        )
        if not exposure_valid:
            assessment['errors'].append(exposure_error)
            assessment['approved'] = False
        
        # 检查每日限制
        daily_valid, daily_error = self.check_daily_limits(quantity * price)
        if not daily_valid:
            assessment['errors'].append(daily_error)
            assessment['approved'] = False
        
        # 评估风险等级
        order_ratio = (quantity * price) / portfolio_value
        if order_ratio > 0.1:
            assessment['risk_level'] = RiskLevel.HIGH
            assessment['warnings'].append(f"大额订单：占投资组合 {order_ratio*100:.1f}%")
        elif order_ratio > 0.05:
            assessment['risk_level'] = RiskLevel.MEDIUM
            assessment['warnings'].append(f"中等订单：占投资组合 {order_ratio*100:.1f}%")
        
        return assessment

class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        """初始化安全管理器"""
        self.logger = logging.getLogger(__name__)
        self.session_token = None
        self.token_expiry = None
        
    def generate_session_token(self) -> str:
        """
        生成会话令牌
        
        Returns:
            str: 会话令牌
        """
        token = secrets.token_urlsafe(32)
        self.session_token = token
        self.token_expiry = datetime.now() + timedelta(hours=8)  # 8小时有效期
        return token
    
    def validate_session_token(self, token: str) -> bool:
        """
        验证会话令牌
        
        Args:
            token: 令牌
            
        Returns:
            bool: 是否有效
        """
        if not self.session_token or not self.token_expiry:
            return False
        
        if datetime.now() > self.token_expiry:
            self.logger.warning("会话令牌已过期")
            return False
        
        return token == self.session_token
    
    def hash_credentials(self, username: str, password: str) -> str:
        """
        对凭据进行哈希处理
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            str: 哈希值
        """
        combined = f"{username}:{password}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def secure_store_credentials(self, username: str, password: str, 
                               filepath: str = "credentials.enc") -> bool:
        """
        安全存储凭据（简化版，实际应使用更强的加密）
        
        Args:
            username: 用户名
            password: 密码
            filepath: 存储文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 这里应该使用更强的加密方法，如 cryptography 库
            hashed = self.hash_credentials(username, password)
            
            with open(filepath, 'w') as f:
                f.write(hashed)
            
            # 设置文件权限（仅所有者可读写）
            os.chmod(filepath, 0o600)
            
            self.logger.info("凭据已安全存储")
            return True
            
        except Exception as e:
            self.logger.error(f"存储凭据失败: {str(e)}")
            return False
    
    def load_secure_credentials(self, filepath: str = "credentials.enc") -> Optional[str]:
        """
        加载安全存储的凭据
        
        Args:
            filepath: 凭据文件路径
            
        Returns:
            Optional[str]: 凭据哈希值
        """
        try:
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r') as f:
                return f.read().strip()
                
        except Exception as e:
            self.logger.error(f"加载凭据失败: {str(e)}")
            return None

if __name__ == "__main__":
    main()