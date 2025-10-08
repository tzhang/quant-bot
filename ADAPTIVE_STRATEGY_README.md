# 自适应执行策略 (Adaptive Execution Strategy)

## 概述

自适应执行策略是一个智能的回测执行优化系统，能够根据任务的复杂度和系统资源自动选择最优的执行策略，显著提升大规模量化回测的性能。

## 核心特性

### 🧠 智能策略选择
- **自动分析**: 根据股票数量、数据长度、策略复杂度自动计算任务复杂度
- **动态选择**: 基于任务特征智能选择最优执行策略
- **性能学习**: 记录历史执行性能，持续优化策略选择

### 🚀 多种执行策略
1. **Sequential (顺序执行)**: 适用于小规模任务，内存占用最低
2. **Threaded (线程并行)**: 适用于中等规模任务，平衡性能和资源
3. **Async (异步执行)**: 适用于大规模任务，高效处理I/O密集型操作
4. **Parallel (进程并行)**: 适用于超大规模任务，充分利用多核CPU

### 📊 性能监控
- **实时监控**: 跟踪执行时间、内存使用、CPU利用率
- **性能分析**: 提供详细的性能报告和优化建议
- **基准测试**: 建立不同规模任务的性能基准

## 使用方法

### 基本使用

```python
from backtesting.enhanced_backtest_engine import ParallelBacktestEngine
from backtesting.adaptive_execution_strategy import AdaptiveExecutionStrategy

# 创建并行回测引擎（自动集成自适应策略）
engine = ParallelBacktestEngine()

# 添加回测引擎和策略
for symbol, data in market_data.items():
    backtest_engine = EnhancedBacktestEngine()
    backtest_engine.set_market_data(symbol, data)
    backtest_engine.add_strategy(your_strategy)
    engine.add_engine(symbol, backtest_engine)

# 执行回测（自动选择最优策略）
results = engine.run_parallel_backtest(start_date, end_date)
```

### 高级配置

```python
from backtesting.adaptive_execution_strategy import (
    AdaptiveExecutionStrategy, 
    PerformanceThreshold
)

# 自定义性能阈值
thresholds = PerformanceThreshold(
    small_task_limit=500,      # 小任务阈值
    medium_task_limit=2000,    # 中等任务阈值
    large_task_limit=10000,    # 大任务阈值
    memory_limit_mb=2000.0,    # 内存限制
    cpu_cores=8                # CPU核心数
)

# 创建自适应策略
adaptive_strategy = AdaptiveExecutionStrategy(thresholds)

# 手动计算任务复杂度
task_metrics = adaptive_strategy.calculate_task_complexity(
    num_symbols=50,
    data_length=252,
    num_strategies=2
)

# 获取推荐策略
recommended_strategy = adaptive_strategy.choose_execution_strategy(task_metrics)
print(f"推荐策略: {recommended_strategy.value}")
```

## 性能基准

基于测试结果，以下是不同规模任务的性能表现：

### 执行时间对比

| 任务规模 | 股票数量 | 数据天数 | 复杂度 | 推荐策略 | 执行时间 | 成功率 |
|---------|---------|---------|--------|----------|----------|--------|
| 小规模   | 5       | 100     | 500    | sequential | 0.04s   | 100%   |
| 中规模   | 10      | 252     | 2,520  | threaded   | 0.23s   | 100%   |
| 大规模   | 20      | 252     | 5,040  | async      | 0.46s   | 100%   |
| 超大规模 | 50      | 252     | 12,600 | async      | 1.14s   | 100%   |

### 内存使用优化

| 任务规模 | 股票数量 | 峰值内存 | 内存增长 | 内存效率 |
|---------|---------|----------|----------|----------|
| 基准测试 | 10      | 105.5 MB | 9.9 MB   | 0.09 股票/MB |
| 中等规模 | 25      | 128.8 MB | 22.4 MB  | 0.19 股票/MB |
| 大规模   | 50      | 148.2 MB | 24.3 MB  | 0.34 股票/MB |
| 超大规模 | 100     | 178.7 MB | 38.4 MB  | 0.56 股票/MB |

## 策略选择逻辑

### 复杂度计算公式

```
复杂度 = 股票数量 × 数据长度 × 策略复杂度系数
内存估算 = (股票数量 × 数据长度 × 8字节) / (1024 × 1024) MB
```

### 策略选择规则

1. **Sequential**: 复杂度 < 1,000 或 内存 < 100MB
2. **Threaded**: 1,000 ≤ 复杂度 < 5,000 且 内存 < 500MB
3. **Async**: 5,000 ≤ 复杂度 < 20,000 且 内存 < 1000MB
4. **Parallel**: 复杂度 ≥ 20,000 或 内存 ≥ 1000MB

## 内存优化建议

### 数据类型优化
- 使用 `float32` 替代 `float64` 可节省约50%内存
- 使用 `int32` 替代 `int64` 适用于大多数场景
- 合理选择数据精度，避免过度精确

### 处理策略优化
- **分批处理**: 大数据集分块处理，避免内存峰值
- **及时清理**: 使用 `gc.collect()` 及时释放不需要的对象
- **流式处理**: 考虑使用数据流处理而不是一次性加载

### 系统配置优化
- **内存限制**: 根据系统内存合理设置 `memory_limit_mb`
- **CPU核心**: 根据实际CPU核心数设置 `cpu_cores`
- **并发控制**: 避免过度并发导致系统资源竞争

## 测试和验证

### 运行基本测试

```bash
# 基本功能测试
python test_adaptive_strategy.py

# 大规模性能测试
python test_simple_large_scale.py

# 内存优化测试
python memory_optimization_test.py
```

### 自定义测试

```python
# 创建自定义测试场景
def custom_test():
    engine = ParallelBacktestEngine()
    
    # 添加你的数据和策略
    # ...
    
    # 执行回测
    results = engine.run_parallel_backtest(start_date, end_date)
    
    # 分析性能
    adaptive_strategy = engine.adaptive_strategy
    performance_history = adaptive_strategy.performance_history
    
    return results, performance_history
```

## 故障排除

### 常见问题

1. **内存不足错误**
   - 减少批处理大小
   - 使用更节省内存的数据类型
   - 增加系统内存或使用分布式处理

2. **执行时间过长**
   - 检查任务复杂度计算是否合理
   - 验证选择的执行策略是否最优
   - 考虑优化策略算法本身

3. **策略选择不当**
   - 调整性能阈值配置
   - 检查历史性能数据
   - 手动指定执行策略进行对比

### 性能调优

1. **阈值调整**: 根据实际硬件性能调整 `PerformanceThreshold`
2. **策略优化**: 基于历史性能数据优化策略选择逻辑
3. **资源监控**: 使用系统监控工具观察资源使用情况

## 未来发展

### 计划功能
- **机器学习优化**: 使用ML算法优化策略选择
- **分布式支持**: 支持跨机器的分布式回测
- **实时监控**: 提供Web界面的实时性能监控
- **自动调优**: 基于历史数据自动调整参数

### 贡献指南
欢迎提交Issue和Pull Request来改进自适应执行策略系统。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。