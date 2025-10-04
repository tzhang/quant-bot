"""
数据库重置脚本

执行删除所有表并重新创建，用于开发环境快速重置。
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import DatabaseManager
from src.database.init_db import load_env_file, test_database_connections, create_database_tables


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def reset_database() -> bool:
    """
    重置数据库：删除所有表并重新创建
    """
    logger.info("🚨 开始重置数据库（删除并重建所有表）...")

    # 1. 加载环境变量
    load_env_file()

    # 2. 创建数据库管理器
    db_manager = DatabaseManager()

    try:
        # 3. 测试连接
        connection_results = test_database_connections(db_manager)
        if not connection_results.get('postgresql', False):
            logger.error("PostgreSQL连接失败，无法重置数据库")
            return False

        # 4. 删除所有表
        logger.info("🔧 删除所有数据库表...")
        db_manager.drop_tables()

        # 5. 重新创建所有表
        logger.info("🧱 重新创建所有数据库表...")
        create_database_tables(db_manager)

        logger.info("🎉 数据库重置完成！")
        return True

    except Exception as e:
        logger.error(f"❌ 数据库重置失败: {e}")
        return False
    finally:
        db_manager.close_connections()


def main():
    # 非交互式直接执行重置
    success = reset_database()
    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)