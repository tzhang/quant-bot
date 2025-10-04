"""
数据库初始化脚本

用于创建数据库表、初始化数据和测试连接
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import DatabaseManager
from src.database.models import Base
from sqlalchemy import text

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_env_file(env_file: str = '.env'):
    """
    加载环境变量文件
    
    Args:
        env_file: 环境变量文件路径
    """
    env_path = project_root / env_file
    
    if not env_path.exists():
        logger.warning(f"环境变量文件不存在: {env_path}")
        logger.info("请复制 .env.database 文件为 .env 并配置数据库连接信息")
        return
    
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        logger.info(f"已加载环境变量文件: {env_path}")
    except ImportError:
        logger.warning("python-dotenv 未安装，请手动设置环境变量")


def test_database_connections(db_manager: DatabaseManager) -> Dict[str, bool]:
    """
    测试数据库连接
    
    Args:
        db_manager: 数据库管理器实例
        
    Returns:
        dict: 连接测试结果
    """
    logger.info("开始测试数据库连接...")
    
    results = db_manager.test_connections()
    
    for db_type, success in results.items():
        if success:
            logger.info(f"✅ {db_type.upper()} 连接成功")
        else:
            logger.error(f"❌ {db_type.upper()} 连接失败")
    
    return results


def create_database_tables(db_manager: DatabaseManager):
    """
    创建数据库表
    
    Args:
        db_manager: 数据库管理器实例
    """
    logger.info("开始创建数据库表...")
    
    try:
        db_manager.create_tables()
        logger.info("✅ 数据库表创建成功")
        
        # 验证表是否创建成功
        engine = db_manager.get_postgresql_engine()
        with engine.connect() as conn:
            # 根据数据库类型选择不同的查询语句
            if 'sqlite' in str(engine.url):
                # SQLite查询语句
                result = conn.execute(text("""
                    SELECT name as table_name 
                    FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """))
            else:
                # PostgreSQL查询语句
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """))
            
            tables = [row[0] for row in result]
            logger.info(f"已创建的表: {', '.join(tables)}")
            
    except Exception as e:
        logger.error(f"❌ 数据库表创建失败: {e}")
        raise


def initialize_sample_data(db_manager: DatabaseManager):
    """
    初始化示例数据（可选）
    
    Args:
        db_manager: 数据库管理器实例
    """
    logger.info("初始化示例数据...")
    
    try:
        from datetime import datetime, timedelta
        from src.database.dao import stock_data_dao
        
        # 创建一些示例股票数据
        sample_data = [
            {
                'symbol': 'AAPL',
                'date': datetime.now() - timedelta(days=2),
                'open': 150.0,
                'high': 155.0,
                'low': 148.0,
                'close': 152.0,
                'volume': 1000000
            },
            {
                'symbol': 'AAPL',
                'date': datetime.now() - timedelta(days=1),
                'open': 152.0,
                'high': 158.0,
                'low': 151.0,
                'close': 156.0,
                'volume': 1200000
            }
        ]
        
        stock_data_dao.batch_create(sample_data)
        logger.info("✅ 示例数据初始化成功")
        
    except Exception as e:
        logger.warning(f"示例数据初始化失败: {e}")


def main():
    """
    主函数：执行数据库初始化流程
    """
    logger.info("🚀 开始数据库初始化...")
    
    # 1. 加载环境变量
    load_env_file()
    
    # 2. 创建数据库管理器
    db_manager = DatabaseManager()
    
    try:
        # 3. 测试数据库连接
        connection_results = test_database_connections(db_manager)
        
        # 4. 如果PostgreSQL连接成功，创建表
        if connection_results.get('postgresql', False):
            create_database_tables(db_manager)
            
            # 5. 可选：初始化示例数据
            if '--sample-data' in sys.argv:
                initialize_sample_data(db_manager)
        else:
            logger.error("PostgreSQL连接失败，无法创建表")
            return False
        
        # 6. 显示配置信息
        logger.info("📊 数据库配置信息:")
        logger.info(f"  PostgreSQL: {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', 5432)}")
        logger.info(f"  数据库名: {os.getenv('POSTGRES_DB', 'quant_trading')}")
        logger.info(f"  Redis: {os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}")
        
        logger.info("🎉 数据库初始化完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        return False
    
    finally:
        # 7. 关闭连接
        db_manager.close_connections()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)