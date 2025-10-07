"""核心量化交易模块

从Citadel高频交易竞赛中提炼的通用量化交易能力和模块
包含信号生成、风险管理、参数优化、机器学习、监控诊断等核心功能
"""

# 信号生成与处理系统
from .signal_generator import (
    SignalGenerator,
    SignalFusion,
    SignalOptimizer
)

# 自适应风险管理系统
from .risk_manager import (
    AdaptiveRiskManager,
    MarketRegimeDetector,
    VolatilityPredictor
)

# 多目标参数优化框架
from .optimizer import (
    BayesianOptimizer,
    GeneticOptimizer,
    MultiObjectiveOptimizer
)

# ML增强交易系统
from .ml_engine import (
    MLFeatureAnalyzer,
    ModelEnsemble,
    TimeSeriesValidator
)

# 实时监控与预警系统
from .monitor import (
    PerformanceMonitor,
    RiskMonitor,
    SystemHealthMonitor
)

# 系统化调试与诊断框架
from .diagnostics import (
    StrategyDiagnostics,
    PerformanceProfiler,
    ErrorAnalyzer
)

# 工具函数
from .utils import (
    DataValidator,
    TimeSeriesUtils,
    PerformanceUtils,
    RiskUtils,
    OptimizationUtils,
    SignalUtils,
    ConfigManager,
    global_config,
    DEFAULT_CONFIG
)

__all__ = [
    # 信号生成与处理
    'SignalGenerator',
    'SignalFusion',
    'SignalOptimizer',
    
    # 风险管理
    'AdaptiveRiskManager',
    'MarketRegimeDetector',
    'VolatilityPredictor',
    
    # 参数优化
    'BayesianOptimizer',
    'GeneticOptimizer',
    'MultiObjectiveOptimizer',
    
    # 机器学习
    'MLFeatureAnalyzer',
    'ModelEnsemble',
    'TimeSeriesValidator',
    
    # 监控系统
    'PerformanceMonitor',
    'RiskMonitor',
    'SystemHealthMonitor',
    
    # 诊断系统
    'StrategyDiagnostics',
    'PerformanceProfiler',
    'ErrorAnalyzer',
    
    # 工具函数
    'DataValidator',
    'TimeSeriesUtils',
    'PerformanceUtils',
    'RiskUtils',
    'OptimizationUtils',
    'SignalUtils',
    'ConfigManager',
    'global_config',
    'DEFAULT_CONFIG'
]