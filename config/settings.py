"""量化交易系统配置文件

该模块包含系统的所有配置参数，支持从环境变量和配置文件加载设置。
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent.parent


class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        self.url = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/quant_db')
        self.host = os.getenv('DATABASE_HOST', 'localhost')
        self.port = int(os.getenv('DATABASE_PORT', '5432'))
        self.name = os.getenv('DATABASE_NAME', 'quant_db')
        self.user = os.getenv('DATABASE_USER', 'postgres')
        self.password = os.getenv('DATABASE_PASSWORD', '')
        
        # 连接池配置
        self.pool_size = int(os.getenv('DATABASE_POOL_SIZE', '10'))
        self.max_overflow = int(os.getenv('DATABASE_MAX_OVERFLOW', '20'))
        self.pool_timeout = int(os.getenv('DATABASE_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('DATABASE_POOL_RECYCLE', '3600'))


class RedisConfig:
    """Redis配置类"""
    
    def __init__(self):
        self.host = os.getenv('REDIS_HOST', 'localhost')
        self.port = int(os.getenv('REDIS_PORT', '6379'))
        self.password = os.getenv('REDIS_PASSWORD', None)
        self.db = int(os.getenv('REDIS_DB', '0'))
        self.decode_responses = True
        self.socket_timeout = 5
        self.socket_connect_timeout = 5
        self.retry_on_timeout = True


class APIConfig:
    """API服务配置类"""
    
    def __init__(self):
        self.host = os.getenv('API_HOST', '0.0.0.0')
        self.port = int(os.getenv('API_PORT', '8000'))
        self.debug = os.getenv('API_DEBUG', 'False').lower() == 'true'
        self.reload = self.debug
        self.workers = int(os.getenv('API_WORKERS', '1'))
        
        # CORS配置
        self.cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
        self.cors_methods = ['GET', 'POST', 'PUT', 'DELETE']
        self.cors_headers = ['*']


class LoggingConfig:
    """日志配置类"""
    
    def __init__(self):
        self.level = os.getenv('LOG_LEVEL', 'INFO')
        self.file = os.getenv('LOG_FILE', str(BASE_DIR / 'logs' / 'quant_system.log'))
        self.max_size = '10 MB'
        self.backup_count = 5
        self.format = '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}'
        
        # 确保日志目录存在
        log_dir = Path(self.file).parent
        log_dir.mkdir(parents=True, exist_ok=True)


class DataSourceConfig:
    """数据源配置类"""
    
    def __init__(self):
        # Yahoo Finance
        self.yahoo_enabled = os.getenv('YAHOO_FINANCE_ENABLED', 'True').lower() == 'true'
        
        # Alpha Vantage
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        self.alpha_vantage_enabled = bool(self.alpha_vantage_key)
        
        # Quandl
        self.quandl_key = os.getenv('QUANDL_API_KEY', '')
        self.quandl_enabled = bool(self.quandl_key)
        
        # 数据更新配置
        self.data_update_interval = int(os.getenv('DATA_UPDATE_INTERVAL', '60'))  # 分钟
        self.factor_update_interval = int(os.getenv('FACTOR_UPDATE_INTERVAL', '1440'))  # 分钟
        
        # 请求限制
        self.request_delay = 0.1  # 秒
        self.max_retries = 3
        self.timeout = 30


class TradingConfig:
    """交易配置类"""
    
    def __init__(self):
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '1000000'))
        self.commission_rate = float(os.getenv('COMMISSION_RATE', '0.001'))
        self.slippage_rate = float(os.getenv('SLIPPAGE_RATE', '0.0005'))
        
        # 风险管理
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
        self.max_drawdown = float(os.getenv('MAX_DRAWDOWN', '0.2'))
        self.stop_loss = float(os.getenv('STOP_LOSS', '0.05'))
        
        # 交易时间
        self.market_open = '09:30'
        self.market_close = '16:00'
        self.timezone = 'US/Eastern'


class BacktestConfig:
    """回测配置类"""
    
    def __init__(self):
        self.start_date = os.getenv('BACKTEST_START_DATE', '2020-01-01')
        self.end_date = os.getenv('BACKTEST_END_DATE', '2024-01-01')
        self.benchmark_symbol = os.getenv('BENCHMARK_SYMBOL', 'SPY')
        
        # 回测参数
        self.frequency = 'daily'  # daily, hourly, minute
        self.lookback_window = 252  # 交易日
        self.min_periods = 20
        
        # 性能优化
        self.use_multiprocessing = True
        self.max_workers = int(os.getenv('MAX_WORKERS', '4'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '100'))


class CacheConfig:
    """缓存配置类"""
    
    def __init__(self):
        self.enabled = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
        self.ttl = int(os.getenv('CACHE_TTL', '3600'))  # 秒
        self.max_size = 1000  # 最大缓存条目数
        
        # 磁盘缓存
        self.disk_cache_dir = BASE_DIR / 'cache'
        self.disk_cache_size_limit = 1024 * 1024 * 1024  # 1GB


class SecurityConfig:
    """安全配置类"""
    
    def __init__(self):
        self.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
        self.jwt_secret_key = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key')
        self.jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
        self.jwt_expiration_hours = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
        
        # 密码加密
        self.password_hash_rounds = 12


class MonitoringConfig:
    """监控配置类"""
    
    def __init__(self):
        self.enabled = os.getenv('MONITORING_ENABLED', 'True').lower() == 'true'
        self.alert_email = os.getenv('ALERT_EMAIL', '')
        
        # 性能监控
        self.performance_threshold = 0.95  # 95%分位数
        self.memory_threshold = 0.8  # 80%内存使用率
        self.disk_threshold = 0.9  # 90%磁盘使用率


class Settings:
    """主配置类，整合所有配置"""
    
    def __init__(self):
        # 环境配置
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = os.getenv('DEBUG', 'True').lower() == 'true'
        self.testing = os.getenv('TESTING', 'False').lower() == 'true'
        
        # 各模块配置
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.data_source = DataSourceConfig()
        self.trading = TradingConfig()
        self.backtest = BacktestConfig()
        self.cache = CacheConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        
        # 项目信息
        self.project_name = 'Quant Trading System'
        self.version = '1.0.0'
        self.description = '量化交易系统'
        
    def is_production(self) -> bool:
        """判断是否为生产环境"""
        return self.environment.lower() == 'production'
    
    def is_development(self) -> bool:
        """判断是否为开发环境"""
        return self.environment.lower() == 'development'
    
    def is_testing(self) -> bool:
        """判断是否为测试环境"""
        return self.testing or self.environment.lower() == 'testing'


# 全局配置实例
settings = Settings()


# 导出常用配置
__all__ = [
    'settings',
    'DatabaseConfig',
    'RedisConfig', 
    'APIConfig',
    'LoggingConfig',
    'DataSourceConfig',
    'TradingConfig',
    'BacktestConfig',
    'CacheConfig',
    'SecurityConfig',
    'MonitoringConfig',
    'Settings',
    'BASE_DIR'
]