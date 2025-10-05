"""
数据库连接管理模块

提供PostgreSQL和Redis的连接管理功能
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import redis
from redis import Redis

from .models import Base
from .config import get_config, DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库连接管理器"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        初始化数据库管理器
        
        Args:
            config: 数据库配置实例，如果为None则使用全局配置
        """
        self._postgresql_engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._redis_client: Optional[Redis] = None
        self._config = config or get_config()
    
    def get_postgresql_engine(self) -> Engine:
        """
        获取PostgreSQL数据库引擎
        
        Returns:
            Engine: SQLAlchemy引擎实例
        """
        if self._postgresql_engine is None:
            pg_config = self._config.postgresql
            
            self._postgresql_engine = create_engine(
                pg_config.connection_url,
                poolclass=QueuePool,
                pool_size=pg_config.pool_size,
                max_overflow=pg_config.max_overflow,
                pool_timeout=pg_config.pool_timeout,
                pool_recycle=pg_config.pool_recycle,
                echo=pg_config.echo
            )
            
            logger.info(f"PostgreSQL引擎已创建: {pg_config.host}:{pg_config.port}/{pg_config.database}")
        
        return self._postgresql_engine
    
    def get_session_factory(self) -> sessionmaker:
        """
        获取会话工厂
        
        Returns:
            sessionmaker: SQLAlchemy会话工厂
        """
        if self._session_factory is None:
            engine = self.get_postgresql_engine()
            self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)
            logger.info("数据库会话工厂已创建")
        
        return self._session_factory
    
    @contextmanager
    def get_session(self):
        """
        获取数据库会话上下文管理器
        
        Yields:
            Session: SQLAlchemy会话实例
        """
        session_factory = self.get_session_factory()
        session = session_factory()
        
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库会话错误: {e}")
            raise
        finally:
            session.close()
    
    def get_redis_client(self) -> Redis:
        """
        获取Redis客户端
        
        Returns:
            Redis: Redis客户端实例
        """
        if self._redis_client is None:
            redis_config = self._config.redis
            
            # 创建连接池
            connection_pool = redis.ConnectionPool(
                max_connections=redis_config.max_connections,
                **redis_config.connection_kwargs
            )
            
            self._redis_client = Redis(connection_pool=connection_pool)
            
            logger.info(f"Redis客户端已创建: {redis_config.host}:{redis_config.port}/{redis_config.db}")
        
        return self._redis_client
    
    def create_tables(self):
        """创建所有数据库表"""
        try:
            engine = self.get_postgresql_engine()
            Base.metadata.create_all(engine)
            logger.info("数据库表创建成功")
        except Exception as e:
            logger.error(f"数据库表创建失败: {e}")
            raise
    
    def drop_tables(self):
        """删除所有数据库表"""
        try:
            engine = self.get_postgresql_engine()
            Base.metadata.drop_all(engine)
            logger.info("数据库表删除成功")
        except Exception as e:
            logger.error(f"数据库表删除失败: {e}")
            raise
    
    def test_connections(self) -> Dict[str, bool]:
        """
        测试数据库连接
        
        Returns:
            dict: 连接测试结果
        """
        results = {}
        
        # 测试PostgreSQL连接
        try:
            engine = self.get_postgresql_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            results['postgresql'] = True
            logger.info("PostgreSQL连接测试成功")
        except Exception as e:
            results['postgresql'] = False
            logger.error(f"PostgreSQL连接测试失败: {e}")
        
        # 测试Redis连接
        try:
            redis_client = self.get_redis_client()
            redis_client.ping()
            results['redis'] = True
            logger.info("Redis连接测试成功")
        except Exception as e:
            results['redis'] = False
            logger.error(f"Redis连接测试失败: {e}")
        
        return results
    
    def close_connections(self):
        """关闭所有数据库连接"""
        if self._postgresql_engine:
            self._postgresql_engine.dispose()
            self._postgresql_engine = None
            logger.info("PostgreSQL连接已关闭")
        
        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None
            logger.info("Redis连接已关闭")
        
        self._session_factory = None
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要
        
        Returns:
            dict: 配置摘要信息
        """
        return self._config.get_summary()


# 全局数据库管理器实例
db_manager = DatabaseManager()

# 便捷函数
def get_db_session():
    """获取数据库会话"""
    return db_manager.get_session()

def get_redis():
    """获取Redis客户端"""
    return db_manager.get_redis_client()

def get_engine():
    """获取数据库引擎"""
    return db_manager.get_postgresql_engine()

def init_database():
    """初始化数据库"""
    db_manager.create_tables()

__all__ = ['DatabaseManager', 'db_manager', 'get_db_session', 'get_redis', 'get_engine', 'init_database']