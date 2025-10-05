"""
数据库迁移管理器

实现数据库版本控制、自动迁移、回滚等功能
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import importlib.util
import inspect

from sqlalchemy import text, MetaData, Table, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .connection import get_db_session, get_engine
from .models import Base

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationType(Enum):
    """迁移类型枚举"""
    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    MODIFY_COLUMN = "modify_column"
    CREATE_INDEX = "create_index"
    DROP_INDEX = "drop_index"
    INSERT_DATA = "insert_data"
    UPDATE_DATA = "update_data"
    DELETE_DATA = "delete_data"
    CUSTOM_SQL = "custom_sql"


@dataclass
class MigrationStep:
    """迁移步骤"""
    step_type: MigrationType
    description: str
    sql: str
    rollback_sql: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    
    def execute(self, session: Session) -> bool:
        """
        执行迁移步骤
        
        Args:
            session: 数据库会话
            
        Returns:
            bool: 是否执行成功
        """
        try:
            if self.parameters:
                session.execute(text(self.sql), self.parameters)
            else:
                session.execute(text(self.sql))
            return True
        except Exception as e:
            logger.error(f"执行迁移步骤失败: {e}")
            return False
    
    def rollback(self, session: Session) -> bool:
        """
        回滚迁移步骤
        
        Args:
            session: 数据库会话
            
        Returns:
            bool: 是否回滚成功
        """
        if not self.rollback_sql:
            logger.warning("没有定义回滚SQL")
            return False
        
        try:
            if self.parameters:
                session.execute(text(self.rollback_sql), self.parameters)
            else:
                session.execute(text(self.rollback_sql))
            return True
        except Exception as e:
            logger.error(f"回滚迁移步骤失败: {e}")
            return False


@dataclass
class Migration:
    """数据库迁移"""
    version: str
    name: str
    description: str
    steps: List[MigrationStep]
    dependencies: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
    
    def get_checksum(self) -> str:
        """
        获取迁移校验和
        
        Returns:
            str: 校验和
        """
        content = f"{self.version}:{self.name}:{self.description}"
        for step in self.steps:
            content += f":{step.sql}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def execute(self, session: Session) -> bool:
        """
        执行迁移
        
        Args:
            session: 数据库会话
            
        Returns:
            bool: 是否执行成功
        """
        logger.info(f"开始执行迁移: {self.version} - {self.name}")
        
        executed_steps = []
        
        try:
            for i, step in enumerate(self.steps):
                logger.info(f"执行步骤 {i+1}/{len(self.steps)}: {step.description}")
                
                if step.execute(session):
                    executed_steps.append(step)
                else:
                    # 执行失败，回滚已执行的步骤
                    logger.error(f"步骤执行失败，开始回滚")
                    self._rollback_steps(session, executed_steps)
                    return False
            
            logger.info(f"迁移执行成功: {self.version}")
            return True
            
        except Exception as e:
            logger.error(f"迁移执行异常: {e}")
            self._rollback_steps(session, executed_steps)
            return False
    
    def rollback(self, session: Session) -> bool:
        """
        回滚迁移
        
        Args:
            session: 数据库会话
            
        Returns:
            bool: 是否回滚成功
        """
        logger.info(f"开始回滚迁移: {self.version} - {self.name}")
        
        # 反向执行回滚
        return self._rollback_steps(session, list(reversed(self.steps)))
    
    def _rollback_steps(self, session: Session, steps: List[MigrationStep]) -> bool:
        """
        回滚步骤列表
        
        Args:
            session: 数据库会话
            steps: 要回滚的步骤列表
            
        Returns:
            bool: 是否回滚成功
        """
        success = True
        
        for step in reversed(steps):
            if not step.rollback(session):
                success = False
                logger.error(f"回滚步骤失败: {step.description}")
        
        return success


class MigrationManager:
    """
    数据库迁移管理器
    
    实现数据库版本控制、自动迁移、回滚等功能
    """
    
    def __init__(self, engine: Engine = None, migrations_dir: str = "migrations"):
        """
        初始化迁移管理器
        
        Args:
            engine: 数据库引擎
            migrations_dir: 迁移文件目录
        """
        self.engine = engine or get_engine()
        self.migrations_dir = migrations_dir
        self.logger = logger
        
        # 确保迁移目录存在
        os.makedirs(migrations_dir, exist_ok=True)
        
        # 初始化迁移历史表
        self._init_migration_table()
        
        # 加载迁移文件
        self.migrations: Dict[str, Migration] = {}
        self._load_migrations()
        
        self.logger.info(f"迁移管理器已初始化，迁移目录: {migrations_dir}")
    
    def _init_migration_table(self):
        """初始化迁移历史表"""
        try:
            with get_db_session() as session:
                # 创建迁移历史表
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS migration_history (
                    id INTEGER PRIMARY KEY,
                    version VARCHAR(50) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    checksum VARCHAR(32) NOT NULL,
                    executed_at TIMESTAMP NOT NULL,
                    execution_time FLOAT,
                    success BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
                
                session.execute(text(create_table_sql))
                session.commit()
                
                self.logger.info("迁移历史表已初始化")
                
        except Exception as e:
            self.logger.error(f"初始化迁移历史表失败: {e}")
            raise
    
    def _load_migrations(self):
        """加载迁移文件"""
        try:
            # 扫描迁移目录
            for filename in os.listdir(self.migrations_dir):
                if filename.endswith('.py') and not filename.startswith('__'):
                    migration_path = os.path.join(self.migrations_dir, filename)
                    migration = self._load_migration_file(migration_path)
                    
                    if migration:
                        self.migrations[migration.version] = migration
                        self.logger.debug(f"加载迁移: {migration.version} - {migration.name}")
            
            self.logger.info(f"已加载 {len(self.migrations)} 个迁移文件")
            
        except Exception as e:
            self.logger.error(f"加载迁移文件失败: {e}")
    
    def _load_migration_file(self, file_path: str) -> Optional[Migration]:
        """
        加载单个迁移文件
        
        Args:
            file_path: 迁移文件路径
            
        Returns:
            Optional[Migration]: 迁移对象
        """
        try:
            # 动态导入迁移模块
            spec = importlib.util.spec_from_file_location("migration", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找迁移函数
            if hasattr(module, 'get_migration'):
                return module.get_migration()
            else:
                self.logger.warning(f"迁移文件缺少get_migration函数: {file_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"加载迁移文件失败 {file_path}: {e}")
            return None
    
    def create_migration(self, name: str, description: str = "") -> str:
        """
        创建新的迁移文件
        
        Args:
            name: 迁移名称
            description: 迁移描述
            
        Returns:
            str: 迁移版本号
        """
        # 生成版本号（时间戳格式）
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成文件名
        filename = f"{version}_{name.lower().replace(' ', '_')}.py"
        file_path = os.path.join(self.migrations_dir, filename)
        
        # 生成迁移文件模板
        template = f'''"""
迁移: {name}
版本: {version}
描述: {description}
创建时间: {datetime.now().isoformat()}
"""

from datetime import datetime
from src.database.migration_manager import Migration, MigrationStep, MigrationType


def get_migration() -> Migration:
    """
    获取迁移定义
    
    Returns:
        Migration: 迁移对象
    """
    return Migration(
        version="{version}",
        name="{name}",
        description="{description}",
        steps=[
            # 在这里添加迁移步骤
            # 示例:
            # MigrationStep(
            #     step_type=MigrationType.CREATE_TABLE,
            #     description="创建示例表",
            #     sql="""
            #     CREATE TABLE example_table (
            #         id INTEGER PRIMARY KEY,
            #         name VARCHAR(100) NOT NULL,
            #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            #     )
            #     """,
            #     rollback_sql="DROP TABLE IF EXISTS example_table"
            # )
        ]
    )


def upgrade():
    """
    升级函数（可选）
    
    可以在这里编写自定义的升级逻辑
    """
    pass


def downgrade():
    """
    降级函数（可选）
    
    可以在这里编写自定义的降级逻辑
    """
    pass
'''
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(template)
        
        self.logger.info(f"已创建迁移文件: {file_path}")
        return version
    
    def get_pending_migrations(self) -> List[Migration]:
        """
        获取待执行的迁移
        
        Returns:
            List[Migration]: 待执行的迁移列表
        """
        try:
            with get_db_session() as session:
                # 获取已执行的迁移版本
                executed_versions = set()
                result = session.execute(text("SELECT version FROM migration_history WHERE success = TRUE"))
                
                for row in result:
                    executed_versions.add(row[0])
                
                # 筛选待执行的迁移
                pending = []
                for version in sorted(self.migrations.keys()):
                    if version not in executed_versions:
                        pending.append(self.migrations[version])
                
                return pending
                
        except Exception as e:
            self.logger.error(f"获取待执行迁移失败: {e}")
            return []
    
    def migrate(self, target_version: Optional[str] = None) -> bool:
        """
        执行迁移
        
        Args:
            target_version: 目标版本（可选，默认迁移到最新版本）
            
        Returns:
            bool: 是否迁移成功
        """
        pending_migrations = self.get_pending_migrations()
        
        if not pending_migrations:
            self.logger.info("没有待执行的迁移")
            return True
        
        # 如果指定了目标版本，筛选迁移
        if target_version:
            pending_migrations = [
                m for m in pending_migrations 
                if m.version <= target_version
            ]
        
        self.logger.info(f"开始执行 {len(pending_migrations)} 个迁移")
        
        success_count = 0
        
        for migration in pending_migrations:
            if self._execute_migration(migration):
                success_count += 1
            else:
                self.logger.error(f"迁移失败，停止执行: {migration.version}")
                break
        
        self.logger.info(f"迁移完成，成功执行 {success_count}/{len(pending_migrations)} 个迁移")
        return success_count == len(pending_migrations)
    
    def _execute_migration(self, migration: Migration) -> bool:
        """
        执行单个迁移
        
        Args:
            migration: 迁移对象
            
        Returns:
            bool: 是否执行成功
        """
        start_time = datetime.now()
        
        try:
            with get_db_session() as session:
                # 检查依赖
                if not self._check_dependencies(migration, session):
                    self.logger.error(f"迁移依赖检查失败: {migration.version}")
                    return False
                
                # 执行迁移
                if migration.execute(session):
                    # 记录迁移历史
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    self._record_migration_history(
                        session, migration, True, execution_time
                    )
                    
                    session.commit()
                    return True
                else:
                    # 记录失败的迁移
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    self._record_migration_history(
                        session, migration, False, execution_time
                    )
                    
                    session.rollback()
                    return False
                    
        except Exception as e:
            self.logger.error(f"执行迁移异常: {e}")
            return False
    
    def _check_dependencies(self, migration: Migration, session: Session) -> bool:
        """
        检查迁移依赖
        
        Args:
            migration: 迁移对象
            session: 数据库会话
            
        Returns:
            bool: 依赖是否满足
        """
        if not migration.dependencies:
            return True
        
        try:
            # 检查依赖的迁移是否已执行
            for dep_version in migration.dependencies:
                result = session.execute(
                    text("SELECT COUNT(*) FROM migration_history WHERE version = :version AND success = TRUE"),
                    {"version": dep_version}
                ).fetchone()
                
                if result[0] == 0:
                    self.logger.error(f"依赖迁移未执行: {dep_version}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查迁移依赖失败: {e}")
            return False
    
    def _record_migration_history(self, session: Session, migration: Migration, 
                                success: bool, execution_time: float):
        """
        记录迁移历史
        
        Args:
            session: 数据库会话
            migration: 迁移对象
            success: 是否成功
            execution_time: 执行时间
        """
        try:
            insert_sql = """
            INSERT INTO migration_history 
            (version, name, description, checksum, executed_at, execution_time, success)
            VALUES (:version, :name, :description, :checksum, :executed_at, :execution_time, :success)
            """
            
            session.execute(text(insert_sql), {
                "version": migration.version,
                "name": migration.name,
                "description": migration.description,
                "checksum": migration.get_checksum(),
                "executed_at": datetime.now(),
                "execution_time": execution_time,
                "success": success
            })
            
        except Exception as e:
            self.logger.error(f"记录迁移历史失败: {e}")
    
    def rollback(self, target_version: str) -> bool:
        """
        回滚到指定版本
        
        Args:
            target_version: 目标版本
            
        Returns:
            bool: 是否回滚成功
        """
        try:
            with get_db_session() as session:
                # 获取需要回滚的迁移
                result = session.execute(
                    text("""
                    SELECT version FROM migration_history 
                    WHERE version > :target_version AND success = TRUE
                    ORDER BY version DESC
                    """),
                    {"target_version": target_version}
                )
                
                versions_to_rollback = [row[0] for row in result]
                
                if not versions_to_rollback:
                    self.logger.info("没有需要回滚的迁移")
                    return True
                
                self.logger.info(f"开始回滚 {len(versions_to_rollback)} 个迁移")
                
                success_count = 0
                
                for version in versions_to_rollback:
                    if version in self.migrations:
                        migration = self.migrations[version]
                        
                        if migration.rollback(session):
                            # 更新迁移历史
                            session.execute(
                                text("UPDATE migration_history SET success = FALSE WHERE version = :version"),
                                {"version": version}
                            )
                            success_count += 1
                        else:
                            self.logger.error(f"回滚失败，停止执行: {version}")
                            break
                    else:
                        self.logger.warning(f"找不到迁移定义: {version}")
                
                if success_count == len(versions_to_rollback):
                    session.commit()
                    self.logger.info("回滚完成")
                    return True
                else:
                    session.rollback()
                    self.logger.error("回滚失败")
                    return False
                    
        except Exception as e:
            self.logger.error(f"回滚异常: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        获取迁移状态
        
        Returns:
            Dict[str, Any]: 迁移状态信息
        """
        try:
            with get_db_session() as session:
                # 获取迁移历史
                result = session.execute(
                    text("""
                    SELECT version, name, executed_at, execution_time, success
                    FROM migration_history
                    ORDER BY version
                    """)
                )
                
                executed_migrations = []
                for row in result:
                    executed_migrations.append({
                        'version': row[0],
                        'name': row[1],
                        'executed_at': row[2].isoformat() if row[2] else None,
                        'execution_time': row[3],
                        'success': row[4]
                    })
                
                # 获取待执行的迁移
                pending_migrations = self.get_pending_migrations()
                
                return {
                    'total_migrations': len(self.migrations),
                    'executed_migrations': len(executed_migrations),
                    'pending_migrations': len(pending_migrations),
                    'executed_list': executed_migrations,
                    'pending_list': [
                        {
                            'version': m.version,
                            'name': m.name,
                            'description': m.description
                        }
                        for m in pending_migrations
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"获取迁移状态失败: {e}")
            return {}
    
    def validate_migrations(self) -> List[Dict[str, Any]]:
        """
        验证迁移完整性
        
        Returns:
            List[Dict[str, Any]]: 验证结果
        """
        issues = []
        
        try:
            with get_db_session() as session:
                # 检查已执行迁移的校验和
                result = session.execute(
                    text("SELECT version, checksum FROM migration_history WHERE success = TRUE")
                )
                
                for row in result:
                    version, stored_checksum = row[0], row[1]
                    
                    if version in self.migrations:
                        current_checksum = self.migrations[version].get_checksum()
                        
                        if stored_checksum != current_checksum:
                            issues.append({
                                'type': 'CHECKSUM_MISMATCH',
                                'version': version,
                                'message': f'迁移 {version} 的校验和不匹配',
                                'severity': 'ERROR'
                            })
                    else:
                        issues.append({
                            'type': 'MISSING_MIGRATION',
                            'version': version,
                            'message': f'找不到已执行迁移 {version} 的定义文件',
                            'severity': 'WARNING'
                        })
                
                # 检查依赖关系
                for migration in self.migrations.values():
                    for dep_version in migration.dependencies:
                        if dep_version not in self.migrations:
                            issues.append({
                                'type': 'MISSING_DEPENDENCY',
                                'version': migration.version,
                                'message': f'迁移 {migration.version} 依赖的迁移 {dep_version} 不存在',
                                'severity': 'ERROR'
                            })
                
        except Exception as e:
            issues.append({
                'type': 'VALIDATION_ERROR',
                'message': f'验证过程中发生异常: {e}',
                'severity': 'ERROR'
            })
        
        return issues


# 全局迁移管理器实例
_migration_manager = None


def get_migration_manager() -> MigrationManager:
    """获取全局迁移管理器实例"""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = MigrationManager()
    return _migration_manager


if __name__ == "__main__":
    # 测试迁移管理器
    manager = MigrationManager()
    
    # 创建测试迁移
    version = manager.create_migration("test_migration", "测试迁移")
    print(f"创建迁移: {version}")
    
    # 获取迁移状态
    status = manager.get_migration_status()
    print(f"迁移状态: {status}")
    
    # 验证迁移
    issues = manager.validate_migrations()
    print(f"验证结果: {issues}")