"""
优化的数据库模型

实现分区表设计、数据压缩、查询性能优化等功能
"""

from sqlalchemy import (
    Column, String, Float, DateTime, Integer, Text, Boolean, 
    Index, ForeignKey, UniqueConstraint, CheckConstraint,
    text, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import expression
from datetime import datetime
import uuid

Base = declarative_base()


class PartitionedStockData(Base):
    """
    分区股票数据表 - 按日期分区以提高查询性能
    
    使用PostgreSQL的声明式分区功能，按月分区存储股票数据
    """
    __tablename__ = 'partitioned_stock_data'
    
    # 主键字段
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # OHLCV数据
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # 调整后价格
    adj_close = Column(Float)
    
    # 元数据
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 复合索引优化查询性能
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'),
        Index('idx_date_symbol', 'date', 'symbol'),
        Index('idx_symbol_date_close', 'symbol', 'date', 'close'),
        UniqueConstraint('symbol', 'date', name='uq_symbol_date'),
        CheckConstraint('high >= low', name='check_high_low'),
        CheckConstraint('volume >= 0', name='check_volume_positive'),
        # PostgreSQL分区配置
        {
            'postgresql_partition_by': 'RANGE (date)',
            'postgresql_inherits': 'stock_data_template'
        }
    )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adj_close': self.adj_close,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class CompressedFinancialData(Base):
    """
    压缩财务数据表 - 使用JSONB存储财务指标以减少存储空间
    """
    __tablename__ = 'compressed_financial_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    report_date = Column(DateTime, nullable=False, index=True)
    report_type = Column(String(20), nullable=False)  # 'quarterly', 'annual'
    
    # 使用JSONB存储所有财务指标，支持高效查询和压缩
    financial_metrics = Column(JSONB, nullable=False)
    
    # 常用指标提取到单独列以优化查询
    revenue = Column(Float, index=True)
    net_income = Column(Float, index=True)
    total_assets = Column(Float)
    shareholders_equity = Column(Float)
    
    # 元数据
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_report_date', 'symbol', 'report_date'),
        Index('idx_report_date_type', 'report_date', 'report_type'),
        # JSONB字段的GIN索引，支持高效的JSON查询
        Index('idx_financial_metrics_gin', 'financial_metrics', postgresql_using='gin'),
        UniqueConstraint('symbol', 'report_date', 'report_type', name='uq_symbol_report'),
    )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'report_date': self.report_date.isoformat() if self.report_date else None,
            'report_type': self.report_type,
            'financial_metrics': self.financial_metrics,
            'revenue': self.revenue,
            'net_income': self.net_income,
            'total_assets': self.total_assets,
            'shareholders_equity': self.shareholders_equity,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class OptimizedFactorData(Base):
    """
    优化的因子数据表 - 使用列式存储和压缩
    """
    __tablename__ = 'optimized_factor_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # 技术因子
    rsi_14 = Column(Float)
    macd_signal = Column(Float)
    bollinger_upper = Column(Float)
    bollinger_lower = Column(Float)
    
    # 基本面因子
    pe_ratio = Column(Float, index=True)
    pb_ratio = Column(Float, index=True)
    roe = Column(Float, index=True)
    revenue_growth = Column(Float)
    
    # 风险因子
    volatility_20d = Column(Float)
    beta = Column(Float)
    var_95 = Column(Float)
    
    # 综合评分
    factor_score = Column(Float, index=True)
    
    # 元数据
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_date_score', 'symbol', 'date', 'factor_score'),
        Index('idx_date_pe_pb', 'date', 'pe_ratio', 'pb_ratio'),
        Index('idx_symbol_score', 'symbol', 'factor_score'),
        # 分区配置
        {
            'postgresql_partition_by': 'RANGE (date)',
        }
    )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'rsi_14': self.rsi_14,
            'macd_signal': self.macd_signal,
            'bollinger_upper': self.bollinger_upper,
            'bollinger_lower': self.bollinger_lower,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'roe': self.roe,
            'revenue_growth': self.revenue_growth,
            'volatility_20d': self.volatility_20d,
            'beta': self.beta,
            'var_95': self.var_95,
            'factor_score': self.factor_score,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class CachedMarketData(Base):
    """
    缓存市场数据表 - 用于存储计算结果和减少重复计算
    """
    __tablename__ = 'cached_market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    data_type = Column(String(50), nullable=False, index=True)
    
    # 缓存数据（使用JSONB存储）
    cached_data = Column(JSONB, nullable=False)
    
    # 缓存元数据
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, index=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # 数据大小（字节）
    data_size = Column(Integer)
    
    __table_args__ = (
        Index('idx_data_type_created', 'data_type', 'created_at'),
        Index('idx_expires_at', 'expires_at'),
        Index('idx_cached_data_gin', 'cached_data', postgresql_using='gin'),
    )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'cache_key': self.cache_key,
            'data_type': self.data_type,
            'cached_data': self.cached_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'data_size': self.data_size
        }


class DataQualityMetrics(Base):
    """
    数据质量指标表 - 监控数据质量和完整性
    """
    __tablename__ = 'data_quality_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(100), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    
    # 质量检查详情
    check_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    status = Column(String(20), nullable=False, index=True)  # 'pass', 'warning', 'fail'
    details = Column(JSONB)
    
    # 阈值配置
    warning_threshold = Column(Float)
    error_threshold = Column(Float)
    
    __table_args__ = (
        Index('idx_table_metric_date', 'table_name', 'metric_name', 'check_date'),
        Index('idx_status_date', 'status', 'check_date'),
    )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'table_name': self.table_name,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'check_date': self.check_date.isoformat() if self.check_date else None,
            'status': self.status,
            'details': self.details,
            'warning_threshold': self.warning_threshold,
            'error_threshold': self.error_threshold
        }


class MaterializedViewConfig(Base):
    """
    物化视图配置表 - 管理预计算的聚合数据
    """
    __tablename__ = 'materialized_view_config'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    view_name = Column(String(100), nullable=False, unique=True, index=True)
    base_tables = Column(JSONB, nullable=False)  # 依赖的基础表
    
    # 刷新配置
    refresh_schedule = Column(String(100))  # cron表达式
    last_refresh = Column(DateTime, index=True)
    next_refresh = Column(DateTime, index=True)
    
    # 视图定义
    sql_definition = Column(Text, nullable=False)
    
    # 状态信息
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_next_refresh_active', 'next_refresh', 'is_active'),
    )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'view_name': self.view_name,
            'base_tables': self.base_tables,
            'refresh_schedule': self.refresh_schedule,
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'next_refresh': self.next_refresh.isoformat() if self.next_refresh else None,
            'sql_definition': self.sql_definition,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# 导出所有优化模型
__all__ = [
    'Base',
    'PartitionedStockData',
    'CompressedFinancialData', 
    'OptimizedFactorData',
    'CachedMarketData',
    'DataQualityMetrics',
    'MaterializedViewConfig'
]


# 数据库优化SQL脚本
OPTIMIZATION_SCRIPTS = {
    'create_partitions': """
    -- 创建股票数据分区表（按月分区）
    CREATE TABLE IF NOT EXISTS stock_data_template (
        LIKE partitioned_stock_data INCLUDING ALL
    );
    
    -- 创建分区函数
    CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
    RETURNS void AS $$
    DECLARE
        partition_name text;
        end_date date;
    BEGIN
        partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
        end_date := start_date + interval '1 month';
        
        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I 
                       FOR VALUES FROM (%L) TO (%L)',
                       partition_name, table_name, start_date, end_date);
    END;
    $$ LANGUAGE plpgsql;
    """,
    
    'create_indexes': """
    -- 创建高性能索引
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_data_symbol_date_btree 
    ON partitioned_stock_data USING btree (symbol, date DESC);
    
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_financial_data_jsonb_gin 
    ON compressed_financial_data USING gin (financial_metrics);
    
    -- 创建部分索引（只对活跃股票）
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_stocks_recent 
    ON partitioned_stock_data (symbol, date, close) 
    WHERE date >= CURRENT_DATE - INTERVAL '1 year';
    """,
    
    'enable_compression': """
    -- 启用表压缩（PostgreSQL 14+）
    ALTER TABLE compressed_financial_data SET (toast_tuple_target = 128);
    ALTER TABLE cached_market_data SET (toast_tuple_target = 128);
    
    -- 设置表的存储参数
    ALTER TABLE partitioned_stock_data SET (fillfactor = 90);
    ALTER TABLE optimized_factor_data SET (fillfactor = 85);
    """,
    
    'create_materialized_views': """
    -- 创建常用的物化视图
    CREATE MATERIALIZED VIEW IF NOT EXISTS daily_market_summary AS
    SELECT 
        date,
        COUNT(*) as stock_count,
        AVG(close) as avg_price,
        SUM(volume) as total_volume,
        STDDEV(close) as price_volatility
    FROM partitioned_stock_data 
    WHERE date >= CURRENT_DATE - INTERVAL '1 year'
    GROUP BY date
    ORDER BY date DESC;
    
    CREATE UNIQUE INDEX ON daily_market_summary (date);
    
    -- 创建因子排名视图
    CREATE MATERIALIZED VIEW IF NOT EXISTS factor_rankings AS
    SELECT 
        symbol,
        date,
        pe_ratio,
        pb_ratio,
        roe,
        factor_score,
        ROW_NUMBER() OVER (PARTITION BY date ORDER BY factor_score DESC) as rank
    FROM optimized_factor_data 
    WHERE date >= CURRENT_DATE - INTERVAL '3 months'
    AND factor_score IS NOT NULL;
    
    CREATE UNIQUE INDEX ON factor_rankings (date, rank);
    """
}