import os
import sys
from pathlib import Path
import pytest


def pytest_sessionstart(session):
    # 确保项目根目录在 sys.path，保证 `import src.*` 可用
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # 使用SQLite文件数据库，避免依赖PostgreSQL；覆盖环境变量
    os.environ.setdefault("POSTGRES_DB", "sqlite:///./test_unit.db")
    os.environ.setdefault("SQL_ECHO", "false")
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")

    # 重新加载配置，并替换全局数据库管理器为新的SQLite配置
    from src.database.config import reload_config
    cfg = reload_config()

    from src.database.connection import DatabaseManager
    from src.database import connection as db_conn
    db_conn.db_manager = DatabaseManager(cfg)

    # 创建SQLite中的所有表
    from src.database.models import Base
    engine = db_conn.db_manager.get_postgresql_engine()
    Base.metadata.create_all(engine)

    # 提供一个简单的 Dummy Redis 客户端，避免真实连接
    class DummyRedis:
        def __init__(self):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def setex(self, key, ttl, value):
            self.store[key] = value

        def keys(self, pattern):
            # 简化实现：返回所有键，不进行通配匹配
            return list(self.store.keys())

        def delete(self, *keys):
            for k in keys:
                self.store.pop(k, None)
            return True

        def ping(self):
            return True

        def close(self):
            return True

    # 将 DummyRedis 注入全局管理器
    db_conn.db_manager._redis_client = DummyRedis()


@pytest.fixture(scope="session")
def sqlite_db_url():
    return os.getenv("POSTGRES_DB", "sqlite:///./test_unit.db")