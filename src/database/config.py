"""
数据库配置管理模块

提供数据库连接配置的统一管理和验证
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PostgreSQLConfig:
    """PostgreSQL数据库配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "quant_trading"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    @property
    def connection_url(self) -> str:
        """获取数据库连接URL"""
        # 如果database字段包含sqlite://，直接返回
        if self.database.startswith('sqlite://'):
            return self.database
        
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        required_fields = ['host', 'database', 'username']
        for field in required_fields:
            if not getattr(self, field):
                return False
        return True


@dataclass
class RedisConfig:
    """Redis缓存配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    connect_timeout: int = 5
    max_connections: int = 50
    
    @property
    def connection_kwargs(self) -> Dict[str, Any]:
        """获取Redis连接参数"""
        kwargs = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'socket_timeout': self.socket_timeout,
            'socket_connect_timeout': self.connect_timeout,
            'decode_responses': True
        }
        
        if self.password:
            kwargs['password'] = self.password
            
        return kwargs
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        return bool(self.host and self.port > 0)


@dataclass
class CacheConfig:
    """缓存配置"""
    ttl: int = 3600  # 默认缓存时间（秒）
    prefix: str = "quant_trading"  # 缓存键前缀
    
    def get_cache_key(self, key: str) -> str:
        """生成缓存键"""
        return f"{self.prefix}:{key}"


class DatabaseConfig:
    """数据库配置管理器"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            env_file: 环境变量文件路径
        """
        self._load_env_file(env_file)
        self.postgresql = self._load_postgresql_config()
        self.redis = self._load_redis_config()
        self.cache = self._load_cache_config()
    
    def _load_env_file(self, env_file: Optional[str] = None):
        """
        加载环境变量文件
        
        Args:
            env_file: 环境变量文件路径
        """
        if env_file is None:
            # 查找项目根目录下的.env文件
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / '.env'
        
        if Path(env_file).exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
            except ImportError:
                pass  # python-dotenv未安装时忽略
    
    def _load_postgresql_config(self) -> PostgreSQLConfig:
        """加载PostgreSQL配置"""
        return PostgreSQLConfig(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'quant_trading'),
            username=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', ''),
            pool_size=int(os.getenv('POSTGRES_POOL_SIZE', 10)),
            max_overflow=int(os.getenv('POSTGRES_MAX_OVERFLOW', 20)),
            pool_timeout=int(os.getenv('POSTGRES_POOL_TIMEOUT', 30)),
            pool_recycle=int(os.getenv('POSTGRES_POOL_RECYCLE', 3600)),
            echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """加载Redis配置"""
        password = os.getenv('REDIS_PASSWORD')
        if password == '':
            password = None
            
        return RedisConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            password=password,
            socket_timeout=int(os.getenv('REDIS_SOCKET_TIMEOUT', 5)),
            connect_timeout=int(os.getenv('REDIS_CONNECT_TIMEOUT', 5)),
            max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', 50))
        )
    
    def _load_cache_config(self) -> CacheConfig:
        """加载缓存配置"""
        return CacheConfig(
            ttl=int(os.getenv('CACHE_TTL', 3600)),
            prefix=os.getenv('CACHE_PREFIX', 'quant_trading')
        )
    
    def validate(self) -> Dict[str, bool]:
        """
        验证所有配置
        
        Returns:
            dict: 各配置模块的验证结果
        """
        return {
            'postgresql': self.postgresql.validate(),
            'redis': self.redis.validate(),
            'cache': True  # 缓存配置总是有效的
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要信息
        
        Returns:
            dict: 配置摘要
        """
        return {
            'postgresql': {
                'host': self.postgresql.host,
                'port': self.postgresql.port,
                'database': self.postgresql.database,
                'username': self.postgresql.username,
                'pool_size': self.postgresql.pool_size
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'max_connections': self.redis.max_connections
            },
            'cache': {
                'ttl': self.cache.ttl,
                'prefix': self.cache.prefix
            }
        }


# 全局配置实例
config = DatabaseConfig()


def get_config() -> DatabaseConfig:
    """
    获取全局配置实例
    
    Returns:
        DatabaseConfig: 配置实例
    """
    return config


def reload_config(env_file: Optional[str] = None) -> DatabaseConfig:
    """
    重新加载配置
    
    Args:
        env_file: 环境变量文件路径
        
    Returns:
        DatabaseConfig: 新的配置实例
    """
    global config
    config = DatabaseConfig(env_file)
    return config