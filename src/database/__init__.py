"""数据库模块

提供数据库连接、模型定义和数据访问功能"""

from .connection import get_db_session, get_engine, init_database
from .models import (
    Base, StockData, StrategyPerformance, FactorData,
    CompanyInfo, FinancialStatement, FinancialRatios,
    MacroEconomicData, SectorData, SentimentData, AlternativeData
)
from .optimized_models import (
    PartitionedStockData, CompressedFinancialData, OptimizedFactorData,
    CachedMarketData, DataQualityMetrics, MaterializedViewConfig
)
from .cache_manager import SmartCacheManager
from .query_optimizer import QueryOptimizer
from .migration_manager import MigrationManager, get_migration_manager
from .performance_monitor import PerformanceMonitor

__all__ = [
    # 基础组件
    'get_db_session',
    'get_engine', 
    'init_database',
    
    # 基础模型
    'Base',
    'StockData',
    'StrategyPerformance',
    'FactorData',
    'CompanyInfo',
    'FinancialStatement',
    'FinancialRatios',
    'MacroEconomicData',
    'SectorData',
    'SentimentData',
    'AlternativeData',
    
    # 优化模型
    'PartitionedStockData',
    'CompressedFinancialData',
    'OptimizedFactorData',
    'CachedMarketData',
    'DataQualityMetrics',
    'MaterializedViewConfig',
    
    # 优化组件
    'SmartCacheManager',
    'QueryOptimizer',
    'PerformanceMonitor',
    'MigrationManager',
    'get_migration_manager'
]