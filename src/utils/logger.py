"""
日志管理工具类
提供统一的日志记录接口和配置管理
"""

import os
import sys
import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import traceback


class LoggerManager:
    """日志管理器类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化日志管理器"""
        if not self._initialized:
            self.config_path = None
            self.log_dir = None
            self.loggers = {}
            self._setup_default_config()
            LoggerManager._initialized = True
    
    def _setup_default_config(self):
        """设置默认日志配置"""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'default',
                    'stream': 'ext://sys.stdout'
                }
            },
            'loggers': {
                '': {
                    'level': 'INFO',
                    'handlers': ['console']
                }
            }
        }
        return config
    
    def load_config(self, config_path: str = None):
        """
        加载日志配置文件
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = "config/logging.yaml"
        
        self.config_path = config_path
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                
                # 确保日志目录存在
                self._ensure_log_directories(self.config)
                
                # 应用配置
                logging.config.dictConfig(self.config)
                
                print(f"日志配置已从 {config_path} 加载")
            else:
                print(f"配置文件 {config_path} 不存在，使用默认配置")
                self.config = self._setup_default_config()
                
        except Exception as e:
            print(f"加载日志配置失败: {e}")
            print("使用默认配置")
            self.config = self._setup_default_config()
    
    def _ensure_log_directories(self, config: Dict[str, Any]):
        """确保日志目录存在"""
        handlers = config.get('handlers', {})
        
        for handler_name, handler_config in handlers.items():
            if 'filename' in handler_config:
                log_file = Path(handler_config['filename'])
                log_file.parent.mkdir(parents=True, exist_ok=True)
                # 设置log_dir为第一个文件处理器的目录
                if not hasattr(self, 'log_dir') or self.log_dir is None:
                    self.log_dir = log_file.parent
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取指定名称的日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            logging.Logger: 日志器实例
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        
        return self.loggers[name]
    
    def log_exception(self, logger: logging.Logger, message: str = "发生异常"):
        """
        记录异常信息
        
        Args:
            logger: 日志器实例
            message: 异常消息
        """
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            logger.error(f"{message}: {traceback.format_exc()}")
        else:
            logger.error(f"{message}: 无异常信息")
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        记录性能信息
        
        Args:
            operation: 操作名称
            duration: 执行时间（秒）
            **kwargs: 其他性能指标
        """
        perf_logger = self.get_logger('performance')
        
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            **kwargs
        }
        
        perf_logger.info(json.dumps(perf_data, ensure_ascii=False))
    
    def log_audit(self, action: str, user: str = None, details: Dict[str, Any] = None):
        """
        记录审计信息
        
        Args:
            action: 操作动作
            user: 用户标识
            details: 详细信息
        """
        audit_logger = self.get_logger('audit')
        
        audit_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user': user or 'system',
            'details': details or {}
        }
        
        audit_logger.info(json.dumps(audit_data, ensure_ascii=False))
    
    def log_trading_event(self, event_type: str, symbol: str = None, **kwargs):
        """
        记录交易事件
        
        Args:
            event_type: 事件类型
            symbol: 交易标的
            **kwargs: 其他事件数据
        """
        trading_logger = self.get_logger('trading')
        
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'symbol': symbol,
            **kwargs
        }
        
        trading_logger.info(json.dumps(event_data, ensure_ascii=False))
    
    def set_level(self, logger_name: str, level: str):
        """
        设置日志器级别
        
        Args:
            logger_name: 日志器名称
            level: 日志级别
        """
        logger = self.get_logger(logger_name)
        numeric_level = getattr(logging, level.upper(), None)
        
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
        else:
            raise ValueError(f'无效的日志级别: {level}')
    
    def get_log_files(self) -> Dict[str, str]:
        """
        获取所有日志文件路径
        
        Returns:
            Dict[str, str]: 日志文件映射
        """
        log_files = {}
        
        if self.log_dir and self.log_dir.exists():
            for log_file in self.log_dir.glob("*.log"):
                log_files[log_file.stem] = str(log_file)
        
        return log_files
    
    def cleanup_old_logs(self, days: int = 30):
        """
        清理旧日志文件
        
        Args:
            days: 保留天数
            
        Returns:
            int: 清理的文件数量
        """
        if not self.log_dir or not self.log_dir.exists():
            return 0
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        cleaned_count = 0
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    print(f"删除日志文件 {log_file} 失败: {e}")
        
        if cleaned_count > 0:
            print(f"已清理 {cleaned_count} 个旧日志文件")
            
        return cleaned_count


# 全局日志管理器实例
logger_manager = LoggerManager()


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器的便捷函数
    
    Args:
        name: 日志器名称
        
    Returns:
        logging.Logger: 日志器实例
    """
    return logger_manager.get_logger(name)


def setup_logging(config_path: str = None):
    """
    设置日志系统的便捷函数
    
    Args:
        config_path: 配置文件路径
    """
    logger_manager.load_config(config_path)


class PerformanceLogger:
    """性能日志记录器上下文管理器"""
    
    def __init__(self, operation: str, logger: logging.Logger = None, **kwargs):
        """
        初始化性能日志记录器
        
        Args:
            operation: 操作名称
            logger: 日志器实例
            **kwargs: 其他性能指标
        """
        self.operation = operation
        self.logger = logger or get_logger('performance')
        self.kwargs = kwargs
        self.start_time = None
        self.details = {}
    
    def start(self):
        """手动开始计时"""
        self.start_time = datetime.now()
        return self
    
    def stop(self):
        """手动停止计时并记录"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            # 合并详细信息
            all_kwargs = {**self.kwargs, **self.details}
            
            # 记录性能信息
            logger_manager.log_performance(
                self.operation,
                duration,
                **all_kwargs
            )
            
            self.start_time = None
            return duration
        return 0
    
    def add_detail(self, key: str, value: Any):
        """添加详细信息"""
        self.details[key] = value
    
    def __enter__(self):
        """进入上下文"""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            # 合并详细信息
            all_kwargs = {**self.kwargs, **self.details}
            
            # 记录性能信息
            logger_manager.log_performance(
                self.operation,
                duration,
                **all_kwargs
            )
            
            # 如果有异常，也记录异常信息
            if exc_type is not None:
                self.logger.error(f"操作 {self.operation} 执行失败: {exc_val}")


# 使用示例
if __name__ == "__main__":
    # 设置日志系统
    setup_logging()
    
    # 获取不同类型的日志器
    app_logger = get_logger('app')
    trading_logger = get_logger('trading')
    
    # 记录普通日志
    app_logger.info("应用启动")
    
    # 记录交易事件
    logger_manager.log_trading_event(
        'order_placed',
        symbol='AAPL',
        quantity=100,
        price=150.0
    )
    
    # 使用性能日志记录器
    with PerformanceLogger('数据处理'):
        import time
        time.sleep(1)  # 模拟耗时操作
    
    # 记录审计事件
    logger_manager.log_audit(
        'user_login',
        user='admin',
        details={'ip': '127.0.0.1', 'success': True}
    )