"""
风险管理模块

提供VaR计算、压力测试、风险归因分析等功能
"""

from .risk_manager import (
    # 枚举类型
    RiskMetric,
    VaRMethod,
    
    # 数据类
    RiskMeasure,
    StressTestScenario,
    StressTestResult,
    
    # 风险模型
    BaseRiskModel,
    HistoricalVaRModel,
    ParametricVaRModel,
    MonteCarloVaRModel,
    
    # 分析引擎
    StressTestEngine,
    RiskAttributionAnalyzer,
    
    # 主要管理器
    RiskManager
)

__all__ = [
    # 枚举类型
    'RiskMetric',
    'VaRMethod',
    
    # 数据类
    'RiskMeasure',
    'StressTestScenario',
    'StressTestResult',
    
    # 风险模型
    'BaseRiskModel',
    'HistoricalVaRModel',
    'ParametricVaRModel',
    'MonteCarloVaRModel',
    
    # 分析引擎
    'StressTestEngine',
    'RiskAttributionAnalyzer',
    
    # 主要管理器
    'RiskManager'
]