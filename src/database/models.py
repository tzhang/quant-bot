"""
数据库模型定义

定义PostgreSQL数据库表结构，使用SQLAlchemy ORM
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Numeric, BigInteger, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class StockData(Base):
    """
    股票数据表模型
    
    存储股票的OHLCV数据
    """
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, comment='股票代码')
    date = Column(DateTime, nullable=False, comment='交易日期')
    open = Column(Numeric(10, 4), comment='开盘价')
    high = Column(Numeric(10, 4), comment='最高价')
    low = Column(Numeric(10, 4), comment='最低价')
    close = Column(Numeric(10, 4), comment='收盘价')
    volume = Column(BigInteger, comment='成交量')
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'),
        Index('idx_date', 'date'),
    )
    
    def __repr__(self):
        return f"<StockData(symbol='{self.symbol}', date='{self.date}', close={self.close})>"
    
    def to_dict(self) -> dict:
        """
        将模型转换为字典格式
        
        Returns:
            dict: 包含所有字段的字典
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'open': float(self.open) if self.open else None,
            'high': float(self.high) if self.high else None,
            'low': float(self.low) if self.low else None,
            'close': float(self.close) if self.close else None,
            'volume': self.volume,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class StrategyPerformance(Base):
    """
    策略绩效表模型
    
    存储策略的历史表现数据
    """
    __tablename__ = 'strategy_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False, comment='策略名称')
    date = Column(DateTime, nullable=False, comment='日期')
    returns = Column(Numeric(10, 6), comment='当日收益率')
    cumulative_returns = Column(Numeric(10, 6), comment='累计收益率')
    drawdown = Column(Numeric(10, 6), comment='回撤')
    positions = Column(Text, comment='持仓信息(JSON格式)')
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_strategy_date', 'strategy_name', 'date'),
        Index('idx_strategy_name', 'strategy_name'),
    )
    
    def __repr__(self):
        return f"<StrategyPerformance(strategy='{self.strategy_name}', date='{self.date}', returns={self.returns})>"
    
    def to_dict(self) -> dict:
        """
        将模型转换为字典格式
        
        Returns:
            dict: 包含所有字段的字典
        """
        return {
            'id': self.id,
            'strategy_name': self.strategy_name,
            'date': self.date.isoformat() if self.date else None,
            'returns': float(self.returns) if self.returns else None,
            'cumulative_returns': float(self.cumulative_returns) if self.cumulative_returns else None,
            'drawdown': float(self.drawdown) if self.drawdown else None,
            'positions': self.positions,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FactorData(Base):
    """
    因子数据表模型
    
    存储各种因子的计算结果
    """
    __tablename__ = 'factor_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, comment='股票代码')
    date = Column(DateTime, nullable=False, comment='日期')
    factor_name = Column(String(50), nullable=False, comment='因子名称')
    factor_value = Column(Numeric(15, 8), comment='因子值')
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_symbol_date_factor', 'symbol', 'date', 'factor_name'),
        Index('idx_factor_date', 'factor_name', 'date'),
    )
    
    def __repr__(self):
        return f"<FactorData(symbol='{self.symbol}', factor='{self.factor_name}', value={self.factor_value})>"
    
    def to_dict(self) -> dict:
        """
        将模型转换为字典格式
        
        Returns:
            dict: 包含所有字段的字典
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'factor_name': self.factor_name,
            'factor_value': float(self.factor_value) if self.factor_value else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# 导出所有模型
__all__ = ['Base', 'StockData', 'StrategyPerformance', 'FactorData']