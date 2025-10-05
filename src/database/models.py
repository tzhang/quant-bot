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


class CompanyInfo(Base):
    """
    公司基本信息表模型
    
    存储公司的基本信息和概览数据
    """
    __tablename__ = 'company_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True, comment='股票代码')
    company_name = Column(String(200), comment='公司名称')
    sector = Column(String(100), comment='行业')
    industry = Column(String(100), comment='子行业')
    market_cap = Column(BigInteger, comment='市值')
    employees = Column(Integer, comment='员工数量')
    description = Column(Text, comment='公司描述')
    website = Column(String(200), comment='公司网站')
    headquarters = Column(String(200), comment='总部地址')
    founded_year = Column(Integer, comment='成立年份')
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp(), comment='更新时间')
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建索引
    __table_args__ = (
        Index('idx_company_symbol', 'symbol'),
        Index('idx_company_sector', 'sector'),
        Index('idx_company_industry', 'industry'),
    )
    
    def __repr__(self):
        return f"<CompanyInfo(symbol='{self.symbol}', name='{self.company_name}')>"
    
    def to_dict(self) -> dict:
        """将模型转换为字典格式"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'employees': self.employees,
            'description': self.description,
            'website': self.website,
            'headquarters': self.headquarters,
            'founded_year': self.founded_year,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FinancialStatement(Base):
    """
    财务报表数据表模型
    
    存储公司的财务报表数据（损益表、资产负债表、现金流量表）
    """
    __tablename__ = 'financial_statements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, comment='股票代码')
    report_date = Column(DateTime, nullable=False, comment='报告期')
    report_type = Column(String(20), nullable=False, comment='报表类型: income, balance, cashflow')
    period_type = Column(String(10), nullable=False, comment='期间类型: annual, quarterly')
    
    # 损益表字段
    revenue = Column(Numeric(20, 2), comment='营业收入')
    gross_profit = Column(Numeric(20, 2), comment='毛利润')
    operating_income = Column(Numeric(20, 2), comment='营业利润')
    net_income = Column(Numeric(20, 2), comment='净利润')
    ebitda = Column(Numeric(20, 2), comment='EBITDA')
    eps = Column(Numeric(10, 4), comment='每股收益')
    
    # 资产负债表字段
    total_assets = Column(Numeric(20, 2), comment='总资产')
    total_liabilities = Column(Numeric(20, 2), comment='总负债')
    shareholders_equity = Column(Numeric(20, 2), comment='股东权益')
    current_assets = Column(Numeric(20, 2), comment='流动资产')
    current_liabilities = Column(Numeric(20, 2), comment='流动负债')
    cash_and_equivalents = Column(Numeric(20, 2), comment='现金及现金等价物')
    
    # 现金流量表字段
    operating_cash_flow = Column(Numeric(20, 2), comment='经营活动现金流')
    investing_cash_flow = Column(Numeric(20, 2), comment='投资活动现金流')
    financing_cash_flow = Column(Numeric(20, 2), comment='筹资活动现金流')
    free_cash_flow = Column(Numeric(20, 2), comment='自由现金流')
    
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_financial_symbol_date', 'symbol', 'report_date'),
        Index('idx_financial_symbol_type', 'symbol', 'report_type', 'period_type'),
        Index('idx_financial_date', 'report_date'),
    )
    
    def __repr__(self):
        return f"<FinancialStatement(symbol='{self.symbol}', date='{self.report_date}', type='{self.report_type}')>"
    
    def to_dict(self) -> dict:
        """将模型转换为字典格式"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'report_date': self.report_date.isoformat() if self.report_date else None,
            'report_type': self.report_type,
            'period_type': self.period_type,
            'revenue': float(self.revenue) if self.revenue else None,
            'gross_profit': float(self.gross_profit) if self.gross_profit else None,
            'operating_income': float(self.operating_income) if self.operating_income else None,
            'net_income': float(self.net_income) if self.net_income else None,
            'ebitda': float(self.ebitda) if self.ebitda else None,
            'eps': float(self.eps) if self.eps else None,
            'total_assets': float(self.total_assets) if self.total_assets else None,
            'total_liabilities': float(self.total_liabilities) if self.total_liabilities else None,
            'shareholders_equity': float(self.shareholders_equity) if self.shareholders_equity else None,
            'current_assets': float(self.current_assets) if self.current_assets else None,
            'current_liabilities': float(self.current_liabilities) if self.current_liabilities else None,
            'cash_and_equivalents': float(self.cash_and_equivalents) if self.cash_and_equivalents else None,
            'operating_cash_flow': float(self.operating_cash_flow) if self.operating_cash_flow else None,
            'investing_cash_flow': float(self.investing_cash_flow) if self.investing_cash_flow else None,
            'financing_cash_flow': float(self.financing_cash_flow) if self.financing_cash_flow else None,
            'free_cash_flow': float(self.free_cash_flow) if self.free_cash_flow else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FinancialRatios(Base):
    """
    财务比率数据表模型
    
    存储计算得出的财务比率和估值指标
    """
    __tablename__ = 'financial_ratios'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, comment='股票代码')
    date = Column(DateTime, nullable=False, comment='计算日期')
    
    # 估值比率
    pe_ratio = Column(Numeric(10, 4), comment='市盈率')
    pb_ratio = Column(Numeric(10, 4), comment='市净率')
    ps_ratio = Column(Numeric(10, 4), comment='市销率')
    peg_ratio = Column(Numeric(10, 4), comment='PEG比率')
    ev_ebitda = Column(Numeric(10, 4), comment='企业价值倍数')
    
    # 盈利能力比率
    roe = Column(Numeric(10, 4), comment='净资产收益率')
    roa = Column(Numeric(10, 4), comment='总资产收益率')
    gross_margin = Column(Numeric(10, 4), comment='毛利率')
    operating_margin = Column(Numeric(10, 4), comment='营业利润率')
    net_margin = Column(Numeric(10, 4), comment='净利率')
    
    # 偿债能力比率
    current_ratio = Column(Numeric(10, 4), comment='流动比率')
    quick_ratio = Column(Numeric(10, 4), comment='速动比率')
    debt_to_equity = Column(Numeric(10, 4), comment='负债权益比')
    debt_ratio = Column(Numeric(10, 4), comment='资产负债率')
    interest_coverage = Column(Numeric(10, 4), comment='利息保障倍数')
    
    # 营运能力比率
    asset_turnover = Column(Numeric(10, 4), comment='总资产周转率')
    inventory_turnover = Column(Numeric(10, 4), comment='存货周转率')
    receivables_turnover = Column(Numeric(10, 4), comment='应收账款周转率')
    
    # 成长性指标
    revenue_growth = Column(Numeric(10, 4), comment='营收增长率')
    earnings_growth = Column(Numeric(10, 4), comment='盈利增长率')
    book_value_growth = Column(Numeric(10, 4), comment='净资产增长率')
    
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_ratios_symbol_date', 'symbol', 'date'),
        Index('idx_ratios_date', 'date'),
    )
    
    def __repr__(self):
        return f"<FinancialRatios(symbol='{self.symbol}', date='{self.date}', pe={self.pe_ratio})>"
    
    def to_dict(self) -> dict:
        """将模型转换为字典格式"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'pe_ratio': float(self.pe_ratio) if self.pe_ratio else None,
            'pb_ratio': float(self.pb_ratio) if self.pb_ratio else None,
            'ps_ratio': float(self.ps_ratio) if self.ps_ratio else None,
            'peg_ratio': float(self.peg_ratio) if self.peg_ratio else None,
            'ev_ebitda': float(self.ev_ebitda) if self.ev_ebitda else None,
            'roe': float(self.roe) if self.roe else None,
            'roa': float(self.roa) if self.roa else None,
            'gross_margin': float(self.gross_margin) if self.gross_margin else None,
            'operating_margin': float(self.operating_margin) if self.operating_margin else None,
            'net_margin': float(self.net_margin) if self.net_margin else None,
            'current_ratio': float(self.current_ratio) if self.current_ratio else None,
            'quick_ratio': float(self.quick_ratio) if self.quick_ratio else None,
            'debt_to_equity': float(self.debt_to_equity) if self.debt_to_equity else None,
            'debt_ratio': float(self.debt_ratio) if self.debt_ratio else None,
            'interest_coverage': float(self.interest_coverage) if self.interest_coverage else None,
            'asset_turnover': float(self.asset_turnover) if self.asset_turnover else None,
            'inventory_turnover': float(self.inventory_turnover) if self.inventory_turnover else None,
            'receivables_turnover': float(self.receivables_turnover) if self.receivables_turnover else None,
            'revenue_growth': float(self.revenue_growth) if self.revenue_growth else None,
            'earnings_growth': float(self.earnings_growth) if self.earnings_growth else None,
            'book_value_growth': float(self.book_value_growth) if self.book_value_growth else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class MacroEconomicData(Base):
    """
    宏观经济数据表模型
    
    存储宏观经济指标数据
    """
    __tablename__ = 'macro_economic_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator_name = Column(String(100), nullable=False, comment='指标名称')
    date = Column(DateTime, nullable=False, comment='数据日期')
    value = Column(Numeric(15, 6), comment='指标值')
    unit = Column(String(50), comment='单位')
    frequency = Column(String(20), comment='频率: daily, weekly, monthly, quarterly, annual')
    source = Column(String(50), comment='数据源')
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_macro_indicator_date', 'indicator_name', 'date'),
        Index('idx_macro_date', 'date'),
    )
    
    def __repr__(self):
        return f"<MacroEconomicData(indicator='{self.indicator_name}', date='{self.date}', value={self.value})>"
    
    def to_dict(self) -> dict:
        """将模型转换为字典格式"""
        return {
            'id': self.id,
            'indicator_name': self.indicator_name,
            'date': self.date.isoformat() if self.date else None,
            'value': float(self.value) if self.value else None,
            'unit': self.unit,
            'frequency': self.frequency,
            'source': self.source,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SectorData(Base):
    """
    行业板块数据表模型
    
    存储行业板块的表现和轮动数据
    """
    __tablename__ = 'sector_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sector_name = Column(String(100), nullable=False, comment='板块名称')
    date = Column(DateTime, nullable=False, comment='日期')
    price = Column(Numeric(10, 4), comment='板块价格/指数')
    returns_1d = Column(Numeric(10, 6), comment='1日收益率')
    returns_1w = Column(Numeric(10, 6), comment='1周收益率')
    returns_1m = Column(Numeric(10, 6), comment='1月收益率')
    returns_3m = Column(Numeric(10, 6), comment='3月收益率')
    returns_ytd = Column(Numeric(10, 6), comment='年初至今收益率')
    relative_strength = Column(Numeric(10, 4), comment='相对强弱指标')
    momentum_score = Column(Numeric(10, 4), comment='动量得分')
    volume = Column(BigInteger, comment='成交量')
    market_cap = Column(BigInteger, comment='板块市值')
    pe_ratio = Column(Numeric(10, 4), comment='板块市盈率')
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_sector_name_date', 'sector_name', 'date'),
        Index('idx_sector_date', 'date'),
    )
    
    def __repr__(self):
        return f"<SectorData(sector='{self.sector_name}', date='{self.date}', returns_1d={self.returns_1d})>"
    
    def to_dict(self) -> dict:
        """将模型转换为字典格式"""
        return {
            'id': self.id,
            'sector_name': self.sector_name,
            'date': self.date.isoformat() if self.date else None,
            'price': float(self.price) if self.price else None,
            'returns_1d': float(self.returns_1d) if self.returns_1d else None,
            'returns_1w': float(self.returns_1w) if self.returns_1w else None,
            'returns_1m': float(self.returns_1m) if self.returns_1m else None,
            'returns_3m': float(self.returns_3m) if self.returns_3m else None,
            'returns_ytd': float(self.returns_ytd) if self.returns_ytd else None,
            'relative_strength': float(self.relative_strength) if self.relative_strength else None,
            'momentum_score': float(self.momentum_score) if self.momentum_score else None,
            'volume': self.volume,
            'market_cap': self.market_cap,
            'pe_ratio': float(self.pe_ratio) if self.pe_ratio else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SentimentData(Base):
    """
    情感数据表模型
    
    存储市场情感和情绪指标数据
    """
    __tablename__ = 'sentiment_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), comment='股票代码，为空表示市场整体情感')
    date = Column(DateTime, nullable=False, comment='日期')
    sentiment_type = Column(String(50), nullable=False, comment='情感类型: news, social, analyst, institutional')
    
    # 情感得分
    sentiment_score = Column(Numeric(10, 4), comment='情感得分 (-1到1)')
    confidence_score = Column(Numeric(10, 4), comment='置信度得分 (0到1)')
    
    # 新闻情感
    news_sentiment = Column(Numeric(10, 4), comment='新闻情感得分')
    news_volume = Column(Integer, comment='新闻数量')
    
    # 社交媒体情感
    social_sentiment = Column(Numeric(10, 4), comment='社交媒体情感得分')
    social_volume = Column(Integer, comment='社交媒体提及量')
    
    # 分析师情感
    analyst_rating = Column(Numeric(10, 4), comment='分析师评级 (1-5)')
    target_price = Column(Numeric(10, 4), comment='目标价')
    price_change_expectation = Column(Numeric(10, 4), comment='价格变化预期')
    
    # 机构情感
    institutional_flow = Column(Numeric(20, 2), comment='机构资金流向')
    institutional_sentiment = Column(Numeric(10, 4), comment='机构情感得分')
    
    source = Column(String(100), comment='数据源')
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_sentiment_symbol_date', 'symbol', 'date'),
        Index('idx_sentiment_type_date', 'sentiment_type', 'date'),
        Index('idx_sentiment_date', 'date'),
    )
    
    def __repr__(self):
        return f"<SentimentData(symbol='{self.symbol}', type='{self.sentiment_type}', score={self.sentiment_score})>"
    
    def to_dict(self) -> dict:
        """将模型转换为字典格式"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'sentiment_type': self.sentiment_type,
            'sentiment_score': float(self.sentiment_score) if self.sentiment_score else None,
            'confidence_score': float(self.confidence_score) if self.confidence_score else None,
            'news_sentiment': float(self.news_sentiment) if self.news_sentiment else None,
            'news_volume': self.news_volume,
            'social_sentiment': float(self.social_sentiment) if self.social_sentiment else None,
            'social_volume': self.social_volume,
            'analyst_rating': float(self.analyst_rating) if self.analyst_rating else None,
            'target_price': float(self.target_price) if self.target_price else None,
            'price_change_expectation': float(self.price_change_expectation) if self.price_change_expectation else None,
            'institutional_flow': float(self.institutional_flow) if self.institutional_flow else None,
            'institutional_sentiment': float(self.institutional_sentiment) if self.institutional_sentiment else None,
            'source': self.source,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AlternativeData(Base):
    """
    另类数据表模型
    
    存储期权、债券、商品、加密货币等另类数据
    """
    __tablename__ = 'alternative_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    data_type = Column(String(50), nullable=False, comment='数据类型: options, bonds, commodities, crypto, forex')
    symbol = Column(String(50), nullable=False, comment='标的代码')
    date = Column(DateTime, nullable=False, comment='日期')
    
    # 通用价格字段
    price = Column(Numeric(15, 8), comment='价格')
    volume = Column(BigInteger, comment='成交量')
    
    # 期权特定字段
    strike_price = Column(Numeric(10, 4), comment='行权价')
    expiration_date = Column(DateTime, comment='到期日')
    option_type = Column(String(10), comment='期权类型: call, put')
    implied_volatility = Column(Numeric(10, 6), comment='隐含波动率')
    delta = Column(Numeric(10, 6), comment='Delta')
    gamma = Column(Numeric(10, 6), comment='Gamma')
    theta = Column(Numeric(10, 6), comment='Theta')
    vega = Column(Numeric(10, 6), comment='Vega')
    
    # 债券特定字段
    yield_rate = Column(Numeric(10, 6), comment='收益率')
    duration = Column(Numeric(10, 4), comment='久期')
    maturity_date = Column(DateTime, comment='到期日')
    credit_rating = Column(String(10), comment='信用评级')
    
    # 商品特定字段
    commodity_type = Column(String(50), comment='商品类型')
    contract_month = Column(String(10), comment='合约月份')
    
    # 加密货币特定字段
    market_cap = Column(BigInteger, comment='市值')
    circulating_supply = Column(BigInteger, comment='流通量')
    
    # 外汇特定字段
    base_currency = Column(String(10), comment='基础货币')
    quote_currency = Column(String(10), comment='报价货币')
    
    source = Column(String(100), comment='数据源')
    created_at = Column(DateTime, default=func.current_timestamp(), comment='创建时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_alternative_type_symbol_date', 'data_type', 'symbol', 'date'),
        Index('idx_alternative_symbol_date', 'symbol', 'date'),
        Index('idx_alternative_date', 'date'),
    )
    
    def __repr__(self):
        return f"<AlternativeData(type='{self.data_type}', symbol='{self.symbol}', price={self.price})>"
    
    def to_dict(self) -> dict:
        """将模型转换为字典格式"""
        return {
            'id': self.id,
            'data_type': self.data_type,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'price': float(self.price) if self.price else None,
            'volume': self.volume,
            'strike_price': float(self.strike_price) if self.strike_price else None,
            'expiration_date': self.expiration_date.isoformat() if self.expiration_date else None,
            'option_type': self.option_type,
            'implied_volatility': float(self.implied_volatility) if self.implied_volatility else None,
            'delta': float(self.delta) if self.delta else None,
            'gamma': float(self.gamma) if self.gamma else None,
            'theta': float(self.theta) if self.theta else None,
            'vega': float(self.vega) if self.vega else None,
            'yield_rate': float(self.yield_rate) if self.yield_rate else None,
            'duration': float(self.duration) if self.duration else None,
            'maturity_date': self.maturity_date.isoformat() if self.maturity_date else None,
            'credit_rating': self.credit_rating,
            'commodity_type': self.commodity_type,
            'contract_month': self.contract_month,
            'market_cap': self.market_cap,
            'circulating_supply': self.circulating_supply,
            'base_currency': self.base_currency,
            'quote_currency': self.quote_currency,
            'source': self.source,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# 导出所有模型
__all__ = [
    'Base', 'StockData', 'StrategyPerformance', 'FactorData',
    'CompanyInfo', 'FinancialStatement', 'FinancialRatios',
    'MacroEconomicData', 'SectorData', 'SentimentData', 'AlternativeData'
]