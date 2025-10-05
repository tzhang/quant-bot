#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易系统 - 主模块初始化文件

集成所有核心功能模块，提供统一的API接口
"""

# 数据管理模块
from .data import (
    DataManager,
    FundamentalDataManager,
    DataQualityManager,
    MacroDataManager,
    SectorDataManager,
    SentimentDataManager,
    AlternativeDataManager
)

# 因子计算模块
from .factors import (
    TechnicalFactors,
    RiskFactors,
    FundamentalFactorCalculator,
    FactorEngine,
    MultiFactorModel,
    FactorConfig,
    ModelConfig,
    FactorProcessor,
    RiskModel,
    FactorExposure,
    ModelOutput,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationResult,
    BaseOptimizer,
    MeanVarianceOptimizer,
    BlackLittermanOptimizer,
    RiskParityOptimizer,
    CVXPYOptimizer,
    PortfolioOptimizer
)

# 回测引擎模块
from .backtest import (
    BacktestEngine
)

from .backtesting import (
    Event,
    EventType,
    OrderType,
    OrderSide,
    OrderStatus,
    Order,
    Fill,
    Position,
    MarketDataHandler,
    SignalHandler,
    OrderHandler,
    FillHandler,
    BaseStrategy,
    SimpleMovingAverageStrategy,
    Portfolio,
    EnhancedBacktestEngine,
    MultiStrategyBacktestManager
)

# 性能分析模块
from .performance import (
    PerformanceAnalyzer
)

# 数据库优化模块
from .database import (
    PartitionedStockData,
    CompressedFinancialData,
    OptimizedFactorData,
    CachedMarketData,
    DataQualityMetrics,
    MaterializedViewConfig,
    SmartCacheManager,
    QueryOptimizer,
    PerformanceMonitor,
    MigrationManager
)

# 风险管理模块
from .risk import (
    RiskMetric,
    VaRMethod,
    RiskMeasure,
    StressTestScenario,
    StressTestResult,
    BaseRiskModel,
    HistoricalVaRModel,
    ParametricVaRModel,
    MonteCarloVaRModel,
    StressTestEngine,
    RiskAttributionAnalyzer,
    RiskManager
)

# 策略模块
from .strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    MACDStrategy,
    PairsTradingStrategy,
    VolatilityBreakoutStrategy,
    StrategyTester
)

# 工具模块 - 暂时为空
# from .utils import (
#     TechnicalIndicators,
#     StatisticalTests,
#     DataValidator,
#     PerformanceMetrics,
#     RiskMetrics,
#     Visualizer
# )

# 版本信息
__version__ = "1.0.0"
__author__ = "Quant Team"
__email__ = "quant@example.com"

# 导出所有公共接口
__all__ = [
    # 数据管理
    "DataManager",
    "FundamentalDataManager", 
    "DataQualityManager",
    "MacroDataManager",
    "SectorDataManager",
    "SentimentDataManager",
    "AlternativeDataManager",
    
    # 因子计算
    "TechnicalFactors",
    "RiskFactors", 
    "FundamentalFactorCalculator",
    "FactorEngine",
    "MultiFactorModel",
    "FactorConfig",
    "ModelConfig",
    "FactorProcessor",
    "RiskModel",
    "FactorExposure",
    "ModelOutput",
    
    # 投资组合优化
    "OptimizationConstraint",
    "OptimizationObjective", 
    "OptimizationResult",
    "BaseOptimizer",
    "MeanVarianceOptimizer",
    "BlackLittermanOptimizer",
    "RiskParityOptimizer",
    "CVXPYOptimizer",
    "PortfolioOptimizer",
    
    # 回测引擎
    "BacktestEngine",
    "PerformanceAnalyzer",
    "Event",
    "EventType",
    "OrderType", 
    "OrderSide",
    "OrderStatus",
    "Order",
    "Fill",
    "Position",
    "MarketDataHandler",
    "SignalHandler",
    "OrderHandler",
    "FillHandler",
    "BaseStrategy",
    "SimpleMovingAverageStrategy",
    "Portfolio",
    "EnhancedBacktestEngine",
    "MultiStrategyBacktestManager",
    
    # 数据库优化
    "PartitionedStockData",
    "CompressedFinancialData",
    "OptimizedFactorData", 
    "CachedMarketData",
    "DataQualityMetrics",
    "MaterializedViewConfig",
    "SmartCacheManager",
    "QueryOptimizer",
    "PerformanceMonitor",
    "MigrationManager",
    
    # 风险管理
    "RiskMetric",
    "VaRMethod",
    "RiskMeasure",
    "StressTestScenario",
    "StressTestResult",
    "BaseRiskModel",
    "HistoricalVaRModel",
    "ParametricVaRModel", 
    "MonteCarloVaRModel",
    "StressTestEngine",
    "RiskAttributionAnalyzer",
    "RiskManager",
    
    # 策略模块
    "MeanReversionStrategy",
    "MomentumStrategy", 
    "RSIStrategy",
    "BollingerBandsStrategy",
    "MACDStrategy",
    "PairsTradingStrategy",
    "VolatilityBreakoutStrategy",
    "StrategyTester",
    
    # 工具模块 - 暂时为空
    # "TechnicalIndicators",
    # "StatisticalTests",
    # "DataValidator",
    # "PerformanceMetrics",
    # "RiskMetrics",
    # "Visualizer",
]

# 模块级别的便捷函数
def create_data_manager(cache_dir: str = "data_cache") -> DataManager:
    """
    创建数据管理器实例
    
    Args:
        cache_dir: 缓存目录路径
        
    Returns:
        DataManager实例
    """
    return DataManager(cache_dir=cache_dir)

def create_factor_engine() -> FactorEngine:
    """
    创建因子引擎实例
    
    Returns:
        FactorEngine实例
    """
    return FactorEngine()

def create_backtest_engine(initial_capital: float = 100000.0) -> BacktestEngine:
    """
    创建回测引擎实例
    
    Args:
        initial_capital: 初始资金
        
    Returns:
        BacktestEngine实例
    """
    return BacktestEngine(initial_capital=initial_capital)

def create_risk_manager() -> RiskManager:
    """
    创建风险管理器实例
    
    Returns:
        RiskManager实例
    """
    return RiskManager()

def get_version() -> str:
    """
    获取系统版本信息
    
    Returns:
        版本字符串
    """
    return __version__

# 系统信息
SYSTEM_INFO = {
    "name": "量化交易系统",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": "集成数据管理、因子计算、回测引擎、风险管理的量化交易系统",
    "modules": [
        "data",
        "factors", 
        "backtesting",
        "database",
        "risk",
        "strategies",
        "utils"
    ]
}