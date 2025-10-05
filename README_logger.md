# 日志系统说明

## 概述

本项目实现了一个完整的日志管理系统，专为量化交易应用设计，提供了多种日志类型、性能监控、自动轮转和清理等功能。

## 主要特性

- ✅ **多类型日志支持**: 应用日志、错误日志、交易日志、审计日志、性能日志
- ✅ **单例模式**: 确保全局唯一的日志管理器实例
- ✅ **灵活配置**: 基于YAML的配置文件，支持动态调整
- ✅ **自动轮转**: 支持按文件大小自动轮转日志文件
- ✅ **性能监控**: 内置性能监控器，支持上下文管理器和手动控制
- ✅ **异常处理**: 专门的异常日志记录功能
- ✅ **审计跟踪**: 用户操作和系统事件的审计日志
- ✅ **自动清理**: 定期清理旧日志文件
- ✅ **完整测试**: 21个测试用例，覆盖所有核心功能

## 快速开始

```python
from src.utils.logger import LoggerManager, PerformanceLogger

# 初始化日志管理器
logger_manager = LoggerManager()
logger_manager.load_config('config/logging.yaml')

# 获取日志器
app_logger = logger_manager.get_logger('app')
app_logger.info("系统启动成功")

# 性能监控
with PerformanceLogger("数据处理") as perf:
    # 执行业务逻辑
    process_data()
    perf.add_details({"records": 1000})

# 记录交易事件
logger_manager.log_trading_event(
    event_type="order_placed",
    symbol="AAPL",
    details={"quantity": 100, "price": 150.25}
)
```

## 文件结构

```
src/utils/
├── logger.py              # 核心日志管理器
└── __init__.py

tests/
├── test_logger.py         # 完整测试套件
└── __init__.py

config/
└── logging.yaml           # 日志配置文件

docs/
└── logger_system_guide.md # 详细使用指南

logs/                      # 日志文件目录
├── app.log
├── error.log
├── trading.log
├── audit.log
└── performance.log
```

## 核心组件

### LoggerManager
- 单例模式的日志管理器
- 支持多种日志类型配置
- 提供异常、审计、交易事件记录功能
- 自动日志文件清理

### PerformanceLogger
- 性能监控专用日志器
- 支持上下文管理器和手动控制
- 可添加自定义性能指标
- 自动计算执行时间

## 配置示例

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers:
  app_file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: standard
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  app:
    level: INFO
    handlers: [app_file]
    propagate: false
```

## 测试覆盖

系统包含21个测试用例，覆盖以下功能：

- ✅ 单例模式验证
- ✅ 配置文件加载
- ✅ 日志目录创建
- ✅ 多种日志器获取
- ✅ 异常日志记录
- ✅ 性能监控
- ✅ 审计日志
- ✅ 交易事件记录
- ✅ 日志级别设置
- ✅ 日志文件路径获取
- ✅ 旧日志清理
- ✅ 日志轮转
- ✅ 完整工作流程

运行测试：
```bash
python -m pytest tests/test_logger.py -v
```

## 使用场景

1. **量化交易系统**: 记录交易信号、订单执行、风险控制等
2. **数据处理**: 监控数据获取、清洗、分析性能
3. **系统监控**: 跟踪系统状态、用户操作、异常情况
4. **审计合规**: 记录关键操作，满足监管要求

## 最佳实践

- 根据环境调整日志级别（开发用DEBUG，生产用INFO）
- 定期清理旧日志文件，避免磁盘空间不足
- 敏感信息不要记录到日志中
- 使用结构化日志格式，便于后续分析
- 关键业务流程添加性能监控

## 详细文档

完整的使用指南请参考：[docs/logger_system_guide.md](docs/logger_system_guide.md)

## 技术栈

- Python 3.8+
- logging 模块
- PyYAML
- unittest (测试)
- pathlib (路径处理)

---

该日志系统已经过完整测试验证，可以直接在生产环境中使用。