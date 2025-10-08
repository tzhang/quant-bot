"""
自适应执行策略模块

根据任务规模和复杂度自动选择最优的回测执行方式。
"""

import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """执行策略枚举"""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    PARALLEL = "parallel"
    ASYNC = "async"


@dataclass
class TaskMetrics:
    """任务指标"""
    num_symbols: int
    data_length: int
    num_strategies: int
    complexity_score: float
    estimated_memory_mb: float


@dataclass
class PerformanceThreshold:
    """性能阈值配置"""
    small_task_limit: int = 1000
    medium_task_limit: int = 5000
    large_task_limit: int = 20000
    memory_limit_mb: float = 1000.0
    cpu_cores: int = 4


class AdaptiveExecutionStrategy:
    """自适应执行策略选择器"""
    
    def __init__(self, performance_threshold: PerformanceThreshold = None):
        """
        初始化自适应执行策略
        
        Args:
            performance_threshold: 性能阈值配置
        """
        self.threshold = performance_threshold or PerformanceThreshold()
        self.performance_history: Dict[str, List[float]] = {}
        
    def calculate_task_complexity(self, 
                                num_symbols: int,
                                data_length: int, 
                                num_strategies: int = 1,
                                strategy_complexity: float = 1.0) -> TaskMetrics:
        """
        计算任务复杂度指标
        
        Args:
            num_symbols: 股票数量
            data_length: 数据长度
            num_strategies: 策略数量
            strategy_complexity: 策略复杂度系数
            
        Returns:
            TaskMetrics: 任务指标
        """
        # 基础复杂度计算
        base_complexity = num_symbols * data_length * num_strategies
        
        # 考虑策略复杂度
        complexity_score = base_complexity * strategy_complexity
        
        # 估算内存使用（MB）
        # 假设每个数据点需要约100字节（包括OHLCV和计算中间结果）
        estimated_memory_mb = (num_symbols * data_length * 100) / (1024 * 1024)
        
        return TaskMetrics(
            num_symbols=num_symbols,
            data_length=data_length,
            num_strategies=num_strategies,
            complexity_score=complexity_score,
            estimated_memory_mb=estimated_memory_mb
        )
    
    def choose_execution_strategy(self, task_metrics: TaskMetrics) -> ExecutionStrategy:
        """
        根据任务指标选择最优执行策略
        
        Args:
            task_metrics: 任务指标
            
        Returns:
            ExecutionStrategy: 推荐的执行策略
        """
        complexity = task_metrics.complexity_score
        memory_usage = task_metrics.estimated_memory_mb
        
        logger.info(f"任务复杂度: {complexity:.0f}, 预估内存: {memory_usage:.1f}MB")
        
        # 内存限制检查
        if memory_usage > self.threshold.memory_limit_mb:
            logger.warning(f"内存使用超限 ({memory_usage:.1f}MB > {self.threshold.memory_limit_mb}MB)")
            return ExecutionStrategy.ASYNC  # 异步处理内存效率更高
        
        # 基于复杂度选择策略
        if complexity < self.threshold.small_task_limit:
            # 小任务：顺序执行开销最小
            strategy = ExecutionStrategy.SEQUENTIAL
            reason = "小任务，顺序执行开销最小"
            
        elif complexity < self.threshold.medium_task_limit:
            # 中等任务：线程池效果最佳
            strategy = ExecutionStrategy.THREADED
            reason = "中等任务，线程池效果最佳"
            
        elif complexity < self.threshold.large_task_limit:
            # 大任务：异步处理
            strategy = ExecutionStrategy.ASYNC
            reason = "大任务，异步处理效率高"
            
        else:
            # 超大任务：并行处理（如果有足够CPU核心）
            if task_metrics.num_symbols >= self.threshold.cpu_cores:
                strategy = ExecutionStrategy.PARALLEL
                reason = "超大任务且有足够CPU核心，使用并行处理"
            else:
                strategy = ExecutionStrategy.ASYNC
                reason = "超大任务但CPU核心不足，使用异步处理"
        
        logger.info(f"选择执行策略: {strategy.value} - {reason}")
        return strategy
    
    def record_performance(self, 
                          strategy: ExecutionStrategy, 
                          execution_time: float,
                          task_metrics: TaskMetrics):
        """
        记录执行性能
        
        Args:
            strategy: 执行策略
            execution_time: 执行时间
            task_metrics: 任务指标
        """
        key = f"{strategy.value}_{task_metrics.complexity_score:.0f}"
        
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append(execution_time)
        
        # 保持历史记录在合理范围内
        if len(self.performance_history[key]) > 10:
            self.performance_history[key] = self.performance_history[key][-10:]
    
    def get_performance_recommendation(self, task_metrics: TaskMetrics) -> Dict[str, Any]:
        """
        基于历史性能数据提供推荐
        
        Args:
            task_metrics: 任务指标
            
        Returns:
            Dict: 性能推荐信息
        """
        recommendations = {}
        
        # 查找相似复杂度的历史记录
        complexity = task_metrics.complexity_score
        tolerance = complexity * 0.2  # 20%容差
        
        for key, times in self.performance_history.items():
            strategy_name, recorded_complexity = key.split('_')
            recorded_complexity = float(recorded_complexity)
            
            if abs(recorded_complexity - complexity) <= tolerance:
                avg_time = sum(times) / len(times)
                recommendations[strategy_name] = {
                    'avg_time': avg_time,
                    'sample_count': len(times),
                    'complexity_diff': abs(recorded_complexity - complexity)
                }
        
        return recommendations
    
    def optimize_thresholds(self):
        """
        基于历史性能数据优化阈值
        """
        if not self.performance_history:
            logger.info("没有足够的历史数据进行阈值优化")
            return
        
        # 分析不同策略在不同复杂度下的表现
        strategy_performance = {}
        
        for key, times in self.performance_history.items():
            strategy_name, complexity = key.split('_')
            complexity = float(complexity)
            avg_time = sum(times) / len(times)
            
            if strategy_name not in strategy_performance:
                strategy_performance[strategy_name] = []
            
            strategy_performance[strategy_name].append((complexity, avg_time))
        
        # 找到策略切换的最优点
        # 这里可以实现更复杂的优化算法
        logger.info("性能数据分析完成，可考虑调整阈值")
        
        for strategy, data in strategy_performance.items():
            if len(data) >= 3:
                complexities = [d[0] for d in data]
                times = [d[1] for d in data]
                logger.info(f"{strategy}: 复杂度范围 {min(complexities):.0f}-{max(complexities):.0f}, "
                           f"平均时间 {sum(times)/len(times):.3f}s")


def create_adaptive_strategy(cpu_cores: int = 4, 
                           memory_limit_gb: float = 1.0) -> AdaptiveExecutionStrategy:
    """
    创建自适应执行策略实例
    
    Args:
        cpu_cores: CPU核心数
        memory_limit_gb: 内存限制（GB）
        
    Returns:
        AdaptiveExecutionStrategy: 配置好的策略实例
    """
    threshold = PerformanceThreshold(
        cpu_cores=cpu_cores,
        memory_limit_mb=memory_limit_gb * 1024
    )
    
    return AdaptiveExecutionStrategy(threshold)


# 使用示例
if __name__ == "__main__":
    # 创建自适应策略
    adaptive_strategy = create_adaptive_strategy(cpu_cores=8, memory_limit_gb=2.0)
    
    # 测试不同规模的任务
    test_cases = [
        (2, 252, 1),    # 小任务
        (4, 252, 1),    # 中等任务
        (8, 252, 1),    # 大任务
        (20, 504, 2),   # 超大任务
    ]
    
    for num_symbols, data_length, num_strategies in test_cases:
        metrics = adaptive_strategy.calculate_task_complexity(
            num_symbols, data_length, num_strategies
        )
        
        strategy = adaptive_strategy.choose_execution_strategy(metrics)
        
        print(f"任务 ({num_symbols}股票, {data_length}天, {num_strategies}策略): "
              f"复杂度={metrics.complexity_score:.0f}, "
              f"推荐策略={strategy.value}")