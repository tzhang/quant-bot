"""
错误处理和异常管理模块
提供统一的错误处理、异常捕获、错误恢复和通知机制
"""

import sys
import traceback
import functools
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Type, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .logger import get_logger


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别枚举"""
    NETWORK = "network"
    DATABASE = "database"
    TRADING = "trading"
    DATA = "data"
    SYSTEM = "system"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """错误信息数据类"""
    error_id: str
    timestamp: str
    error_type: str
    error_message: str
    error_traceback: str
    severity: ErrorSeverity
    category: ErrorCategory
    function_name: str
    module_name: str
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[str] = None
    resolution_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['category'] = self.category.value
        return data


class RetryConfig:
    """重试配置类"""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 delay: float = 1.0,
                 backoff_factor: float = 2.0,
                 max_delay: float = 60.0,
                 exceptions: tuple = (Exception,)):
        """
        初始化重试配置
        
        Args:
            max_attempts: 最大重试次数
            delay: 初始延迟时间（秒）
            backoff_factor: 退避因子
            max_delay: 最大延迟时间（秒）
            exceptions: 需要重试的异常类型
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.exceptions = exceptions


class CircuitBreakerState(Enum):
    """熔断器状态枚举"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


class CircuitBreaker:
    """熔断器类"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        """
        初始化熔断器
        
        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时时间（秒）
            expected_exception: 预期异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """
        通过熔断器调用函数
        
        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            Exception: 熔断器开启时抛出异常
        """
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("熔断器开启，拒绝请求")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置熔断器"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """成功时的处理"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class ErrorNotifier:
    """错误通知器"""
    
    def __init__(self):
        """初始化错误通知器"""
        self.logger = get_logger('error.notifier')
        
        # 邮件配置
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'from_email': '',
            'to_emails': []
        }
        
        # Webhook配置
        self.webhook_urls = []
        
        # 通知频率限制
        self.notification_cache = defaultdict(lambda: deque(maxlen=10))
        self.rate_limit_window = 300  # 5分钟
        self.max_notifications_per_window = 3
    
    def configure_email(self, smtp_server: str, smtp_port: int, 
                       username: str, password: str, 
                       from_email: str, to_emails: List[str]):
        """
        配置邮件通知
        
        Args:
            smtp_server: SMTP服务器
            smtp_port: SMTP端口
            username: 用户名
            password: 密码
            from_email: 发送邮箱
            to_emails: 接收邮箱列表
        """
        self.email_config.update({
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_email': from_email,
            'to_emails': to_emails
        })
    
    def configure_webhook(self, webhook_urls: List[str]):
        """
        配置Webhook通知
        
        Args:
            webhook_urls: Webhook URL列表
        """
        self.webhook_urls = webhook_urls
    
    def should_notify(self, error_info: ErrorInfo) -> bool:
        """
        检查是否应该发送通知
        
        Args:
            error_info: 错误信息
            
        Returns:
            bool: 是否应该通知
        """
        # 只通知中等及以上严重程度的错误
        if error_info.severity in [ErrorSeverity.LOW]:
            return False
        
        # 检查频率限制
        cache_key = f"{error_info.error_type}:{error_info.function_name}"
        now = time.time()
        
        # 清理过期的通知记录
        self.notification_cache[cache_key] = deque([
            timestamp for timestamp in self.notification_cache[cache_key]
            if now - timestamp < self.rate_limit_window
        ], maxlen=10)
        
        # 检查是否超过频率限制
        if len(self.notification_cache[cache_key]) >= self.max_notifications_per_window:
            return False
        
        # 记录本次通知
        self.notification_cache[cache_key].append(now)
        
        return True
    
    def send_email_notification(self, error_info: ErrorInfo):
        """
        发送邮件通知
        
        Args:
            error_info: 错误信息
        """
        if not self.email_config['username'] or not self.email_config['to_emails']:
            return
        
        try:
            # 创建邮件内容
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"[{error_info.severity.value.upper()}] 系统错误通知 - {error_info.error_type}"
            
            # 邮件正文
            body = f"""
系统发生错误，详细信息如下：

错误ID: {error_info.error_id}
时间: {error_info.timestamp}
严重程度: {error_info.severity.value}
类别: {error_info.category.value}
错误类型: {error_info.error_type}
错误消息: {error_info.error_message}
函数: {error_info.function_name}
模块: {error_info.module_name}

错误堆栈:
{error_info.error_traceback}

上下文信息:
{error_info.context}

请及时处理此错误。
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_emails'], text)
            server.quit()
            
            self.logger.info(f"错误通知邮件已发送: {error_info.error_id}")
            
        except Exception as e:
            self.logger.error(f"发送邮件通知失败: {e}")
    
    def send_webhook_notification(self, error_info: ErrorInfo):
        """
        发送Webhook通知
        
        Args:
            error_info: 错误信息
        """
        if not self.webhook_urls:
            return
        
        payload = {
            'error_id': error_info.error_id,
            'timestamp': error_info.timestamp,
            'severity': error_info.severity.value,
            'category': error_info.category.value,
            'error_type': error_info.error_type,
            'error_message': error_info.error_message,
            'function_name': error_info.function_name,
            'module_name': error_info.module_name,
            'context': error_info.context
        }
        
        for webhook_url in self.webhook_urls:
            try:
                response = requests.post(
                    webhook_url,
                    json=payload,
                    timeout=10,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    self.logger.info(f"Webhook通知已发送: {error_info.error_id}")
                else:
                    self.logger.warning(f"Webhook通知发送失败，状态码: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"发送Webhook通知失败: {e}")
    
    def notify(self, error_info: ErrorInfo):
        """
        发送错误通知
        
        Args:
            error_info: 错误信息
        """
        if not self.should_notify(error_info):
            return
        
        # 发送邮件通知
        self.send_email_notification(error_info)
        
        # 发送Webhook通知
        self.send_webhook_notification(error_info)


class ErrorHandler:
    """错误处理器主类"""
    
    def __init__(self):
        """初始化错误处理器"""
        self.logger = get_logger('error')
        
        # 错误存储
        self.errors: List[ErrorInfo] = []
        self.error_stats = defaultdict(int)
        
        # 错误通知器
        self.notifier = ErrorNotifier()
        
        # 熔断器存储
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 错误分类规则
        self.classification_rules = {
            'requests.exceptions.ConnectionError': ErrorCategory.NETWORK,
            'requests.exceptions.Timeout': ErrorCategory.TIMEOUT,
            'psycopg2.OperationalError': ErrorCategory.DATABASE,
            'sqlalchemy.exc.OperationalError': ErrorCategory.DATABASE,
            'ValueError': ErrorCategory.VALIDATION,
            'TypeError': ErrorCategory.VALIDATION,
            'PermissionError': ErrorCategory.PERMISSION,
            'FileNotFoundError': ErrorCategory.SYSTEM,
        }
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """
        分类错误
        
        Args:
            error: 异常对象
            
        Returns:
            ErrorCategory: 错误类别
        """
        error_type = f"{error.__class__.__module__}.{error.__class__.__name__}"
        
        # 精确匹配
        if error_type in self.classification_rules:
            return self.classification_rules[error_type]
        
        # 类名匹配
        class_name = error.__class__.__name__
        for rule_type, category in self.classification_rules.items():
            if class_name in rule_type:
                return category
        
        # 关键词匹配
        error_message = str(error).lower()
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_message for keyword in ['database', 'sql', 'query']):
            return ErrorCategory.DATABASE
        elif any(keyword in error_message for keyword in ['permission', 'access', 'forbidden']):
            return ErrorCategory.PERMISSION
        elif any(keyword in error_message for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION
        
        return ErrorCategory.UNKNOWN
    
    def determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """
        确定错误严重程度
        
        Args:
            error: 异常对象
            category: 错误类别
            
        Returns:
            ErrorSeverity: 错误严重程度
        """
        # 根据异常类型确定严重程度
        critical_exceptions = [
            'SystemExit', 'KeyboardInterrupt', 'MemoryError',
            'RecursionError', 'SystemError'
        ]
        
        high_exceptions = [
            'ConnectionError', 'OperationalError', 'DatabaseError',
            'AuthenticationError', 'PermissionError'
        ]
        
        medium_exceptions = [
            'ValueError', 'TypeError', 'AttributeError',
            'KeyError', 'IndexError'
        ]
        
        exception_name = error.__class__.__name__
        
        if exception_name in critical_exceptions:
            return ErrorSeverity.CRITICAL
        elif exception_name in high_exceptions:
            return ErrorSeverity.HIGH
        elif exception_name in medium_exceptions:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def handle_error(self, error: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    severity: Optional[ErrorSeverity] = None,
                    category: Optional[ErrorCategory] = None) -> ErrorInfo:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 上下文信息
            severity: 错误严重程度（可选）
            category: 错误类别（可选）
            
        Returns:
            ErrorInfo: 错误信息对象
        """
        # 获取调用栈信息
        frame = sys._getframe(1)
        function_name = frame.f_code.co_name
        module_name = frame.f_globals.get('__name__', 'unknown')
        
        # 自动分类和确定严重程度
        if category is None:
            category = self.classify_error(error)
        
        if severity is None:
            severity = self.determine_severity(error, category)
        
        # 创建错误信息
        error_info = ErrorInfo(
            error_id=f"{int(time.time() * 1000)}_{id(error)}",
            timestamp=datetime.now().isoformat(),
            error_type=error.__class__.__name__,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            severity=severity,
            category=category,
            function_name=function_name,
            module_name=module_name,
            context=context or {}
        )
        
        # 存储错误信息
        with self.lock:
            self.errors.append(error_info)
            self.error_stats[error_info.error_type] += 1
            
            # 限制错误存储数量
            if len(self.errors) > 1000:
                self.errors = self.errors[-1000:]
        
        # 记录日志
        log_message = f"错误处理 - {error_info.error_type}: {error_info.error_message}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # 发送通知
        self.notifier.notify(error_info)
        
        return error_info
    
    def retry_with_backoff(self, retry_config: RetryConfig):
        """
        重试装饰器
        
        Args:
            retry_config: 重试配置
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(retry_config.max_attempts):
                    try:
                        return func(*args, **kwargs)
                        
                    except retry_config.exceptions as e:
                        last_exception = e
                        
                        if attempt < retry_config.max_attempts - 1:
                            # 计算延迟时间
                            delay = min(
                                retry_config.delay * (retry_config.backoff_factor ** attempt),
                                retry_config.max_delay
                            )
                            
                            self.logger.warning(
                                f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败，"
                                f"{delay:.2f}秒后重试: {e}"
                            )
                            
                            time.sleep(delay)
                        else:
                            # 最后一次尝试失败，处理错误
                            self.handle_error(e, {
                                'function': func.__name__,
                                'attempts': retry_config.max_attempts,
                                'args': str(args)[:200],
                                'kwargs': str(kwargs)[:200]
                            })
                
                # 重试次数用完，抛出最后一个异常
                raise last_exception
                
            return wrapper
        return decorator
    
    def with_circuit_breaker(self, name: str, 
                           failure_threshold: int = 5,
                           recovery_timeout: float = 60.0):
        """
        熔断器装饰器
        
        Args:
            name: 熔断器名称
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时时间
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            # 创建或获取熔断器
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout
                )
            
            circuit_breaker = self.circuit_breakers[name]
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return circuit_breaker.call(func, *args, **kwargs)
                except Exception as e:
                    self.handle_error(e, {
                        'circuit_breaker': name,
                        'state': circuit_breaker.state.value,
                        'failure_count': circuit_breaker.failure_count
                    })
                    raise
                    
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, *args, 
                    default_return: Any = None,
                    log_errors: bool = True,
                    **kwargs) -> Any:
        """
        安全执行函数
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            default_return: 默认返回值
            log_errors: 是否记录错误
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果或默认值
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                self.handle_error(e, {
                    'function': func.__name__ if hasattr(func, '__name__') else str(func),
                    'safe_execute': True
                })
            return default_return
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        获取错误统计信息
        
        Returns:
            Dict[str, Any]: 错误统计信息
        """
        with self.lock:
            total_errors = len(self.errors)
            
            if total_errors == 0:
                return {'total_errors': 0}
            
            # 按严重程度统计
            severity_stats = defaultdict(int)
            category_stats = defaultdict(int)
            recent_errors = []
            
            # 最近24小时的错误
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for error in self.errors:
                error_time = datetime.fromisoformat(error.timestamp)
                
                severity_stats[error.severity.value] += 1
                category_stats[error.category.value] += 1
                
                if error_time > cutoff_time:
                    recent_errors.append(error)
            
            return {
                'total_errors': total_errors,
                'recent_errors_24h': len(recent_errors),
                'severity_distribution': dict(severity_stats),
                'category_distribution': dict(category_stats),
                'error_type_distribution': dict(self.error_stats),
                'most_common_errors': sorted(
                    self.error_stats.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
    
    def get_recent_errors(self, hours: int = 24, 
                         severity: Optional[ErrorSeverity] = None) -> List[ErrorInfo]:
        """
        获取最近的错误
        
        Args:
            hours: 时间范围（小时）
            severity: 错误严重程度过滤
            
        Returns:
            List[ErrorInfo]: 错误信息列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = []
        for error in self.errors:
            error_time = datetime.fromisoformat(error.timestamp)
            
            if error_time > cutoff_time:
                if severity is None or error.severity == severity:
                    recent_errors.append(error)
        
        return sorted(recent_errors, key=lambda x: x.timestamp, reverse=True)
    
    def mark_error_resolved(self, error_id: str, resolution_method: str):
        """
        标记错误已解决
        
        Args:
            error_id: 错误ID
            resolution_method: 解决方法
        """
        with self.lock:
            for error in self.errors:
                if error.error_id == error_id:
                    error.resolved = True
                    error.resolution_time = datetime.now().isoformat()
                    error.resolution_method = resolution_method
                    
                    self.logger.info(f"错误 {error_id} 已标记为已解决: {resolution_method}")
                    break
    
    def clear_old_errors(self, days: int = 30):
        """
        清理旧错误记录
        
        Args:
            days: 保留天数
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with self.lock:
            original_count = len(self.errors)
            
            self.errors = [
                error for error in self.errors
                if datetime.fromisoformat(error.timestamp) > cutoff_time
            ]
            
            cleared_count = original_count - len(self.errors)
            
            if cleared_count > 0:
                self.logger.info(f"已清理 {cleared_count} 条旧错误记录")


# 全局错误处理器实例
error_handler = ErrorHandler()


# 便捷装饰器
def handle_errors(severity: Optional[ErrorSeverity] = None,
                 category: Optional[ErrorCategory] = None,
                 reraise: bool = True):
    """
    错误处理装饰器
    
    Args:
        severity: 错误严重程度
        category: 错误类别
        reraise: 是否重新抛出异常
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(
                    e,
                    context={
                        'function': func.__name__,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200]
                    },
                    severity=severity,
                    category=category
                )
                
                if reraise:
                    raise
                    
        return wrapper
    return decorator


def retry_on_error(max_attempts: int = 3,
                  delay: float = 1.0,
                  backoff_factor: float = 2.0,
                  exceptions: tuple = (Exception,)):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型
        
    Returns:
        装饰器函数
    """
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        delay=delay,
        backoff_factor=backoff_factor,
        exceptions=exceptions
    )
    
    return error_handler.retry_with_backoff(retry_config)


def circuit_breaker(name: str,
                   failure_threshold: int = 5,
                   recovery_timeout: float = 60.0):
    """
    熔断器装饰器
    
    Args:
        name: 熔断器名称
        failure_threshold: 失败阈值
        recovery_timeout: 恢复超时时间
        
    Returns:
        装饰器函数
    """
    return error_handler.with_circuit_breaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )


# 使用示例
if __name__ == "__main__":
    # 配置错误通知
    error_handler.notifier.configure_email(
        smtp_server='smtp.gmail.com',
        smtp_port=587,
        username='your_email@gmail.com',
        password='your_password',
        from_email='your_email@gmail.com',
        to_emails=['admin@example.com']
    )
    
    # 使用装饰器
    @handle_errors(severity=ErrorSeverity.HIGH, category=ErrorCategory.TRADING)
    @retry_on_error(max_attempts=3, delay=1.0)
    @circuit_breaker('test_function', failure_threshold=3)
    def test_function():
        # 模拟可能出错的函数
        import random
        if random.random() < 0.5:
            raise ValueError("测试错误")
        return "成功"
    
    # 测试错误处理
    for i in range(10):
        try:
            result = test_function()
            print(f"第 {i+1} 次调用成功: {result}")
        except Exception as e:
            print(f"第 {i+1} 次调用失败: {e}")
    
    # 查看错误统计
    stats = error_handler.get_error_statistics()
    print(f"错误统计: {stats}")