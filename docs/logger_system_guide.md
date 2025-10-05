# 量化交易系统日志管理指南

## 概述

本文档详细介绍了量化交易系统中的日志管理功能，包括日志配置、使用方法、性能监控和最佳实践。

## 系统架构

### 核心组件

1. **LoggerManager**: 日志管理器主类，采用单例模式
2. **PerformanceLogger**: 性能监控日志器
3. **配置系统**: 基于YAML的灵活配置

### 日志类型

- **应用日志** (app.log): 记录应用程序的一般信息
- **错误日志** (error.log): 记录错误和异常信息
- **交易日志** (trading.log): 记录交易相关事件
- **审计日志** (audit.log): 记录用户操作和系统审计信息
- **性能日志** (performance.log): 记录性能监控数据

## 快速开始

### 1. 基本使用

```python
from src.utils.logger import LoggerManager

# 获取日志管理器实例
logger_manager = LoggerManager()

# 加载配置文件
logger_manager.load_config('config/logging.yaml')

# 获取不同类型的日志器
app_logger = logger_manager.get_logger('app')
error_logger = logger_manager.get_logger('error')
trading_logger = logger_manager.get_logger('trading')
```

### 2. 记录不同类型的日志

```python
# 记录应用信息
app_logger.info("系统启动成功")

# 记录错误信息
try:
    # 业务逻辑
    pass
except Exception as e:
    logger_manager.log_exception(error_logger, e, "交易执行失败")

# 记录交易事件
logger_manager.log_trading_event(
    event_type="order_placed",
    symbol="AAPL",
    details={
        "quantity": 100,
        "price": 150.25,
        "order_id": "ORD001"
    }
)

# 记录审计信息
logger_manager.log_audit(
    action="用户登录",
    user="trader_001",
    details={
        "ip": "192.168.1.100",
        "timestamp": "2024-01-01 10:00:00"
    }
)
```

### 3. 性能监控

```python
from src.utils.logger import PerformanceLogger

# 使用上下文管理器
with PerformanceLogger("数据处理") as perf:
    # 执行需要监控的代码
    process_market_data()
    perf.add_details({"records": 1000, "source": "yahoo"})

# 手动控制
perf = PerformanceLogger("算法执行")
perf.start()
run_trading_algorithm()
perf.stop()
perf.log()
```

## 配置说明

### 配置文件结构 (logging.yaml)

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'

handlers:
  app_file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: standard
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    encoding: utf8

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/error.log
    maxBytes: 10485760
    backupCount: 5
    encoding: utf8

  trading_file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: standard
    filename: logs/trading.log
    maxBytes: 10485760
    backupCount: 3
    encoding: utf8

loggers:
  app:
    level: INFO
    handlers: [app_file]
    propagate: false

  error:
    level: ERROR
    handlers: [error_file]
    propagate: false

  trading:
    level: INFO
    handlers: [trading_file]
    propagate: false
```

### 配置参数说明

- **maxBytes**: 单个日志文件的最大大小（字节）
- **backupCount**: 保留的备份文件数量
- **level**: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- **formatter**: 日志格式化器
- **encoding**: 文件编码格式

## 高级功能

### 1. 动态日志级别调整

```python
# 设置特定日志器的级别
logger_manager.set_level('app', 'DEBUG')

# 设置所有日志器的级别
logger_manager.set_level('all', 'WARNING')
```

### 2. 日志文件清理

```python
# 清理30天前的日志文件
cleaned_count = logger_manager.cleanup_old_logs(days=30)
print(f"清理了 {cleaned_count} 个旧日志文件")
```

### 3. 获取日志文件路径

```python
# 获取特定日志器的文件路径
app_log_path = logger_manager.get_log_file_path('app')
if app_log_path:
    print(f"应用日志文件: {app_log_path}")
```

### 4. 性能监控详细信息

```python
# 添加自定义性能指标
with PerformanceLogger("数据库查询") as perf:
    result = execute_query()
    perf.add_details({
        "query_type": "SELECT",
        "table": "market_data",
        "rows_returned": len(result),
        "cache_hit": True
    })
```

## 最佳实践

### 1. 日志级别使用指南

- **DEBUG**: 详细的调试信息，仅在开发环境使用
- **INFO**: 一般信息，记录程序正常运行状态
- **WARNING**: 警告信息，程序仍能正常运行但需要注意
- **ERROR**: 错误信息，程序出现错误但仍能继续运行
- **CRITICAL**: 严重错误，程序可能无法继续运行

### 2. 日志内容规范

```python
# 好的日志记录
logger.info("用户登录成功", extra={
    "user_id": "user123",
    "ip": "192.168.1.100",
    "timestamp": datetime.now().isoformat()
})

# 避免的日志记录
logger.info("用户登录")  # 缺少关键信息
```

### 3. 异常处理

```python
# 推荐的异常处理方式
try:
    result = risky_operation()
except SpecificException as e:
    logger_manager.log_exception(error_logger, e, "具体操作失败", {
        "operation": "risky_operation",
        "parameters": {"param1": "value1"}
    })
    # 处理异常
except Exception as e:
    logger_manager.log_exception(error_logger, e, "未知错误")
    raise  # 重新抛出未知异常
```

### 4. 性能监控策略

```python
# 监控关键业务流程
with PerformanceLogger("交易执行流程") as perf:
    # 数据获取
    market_data = get_market_data()
    perf.add_details({"data_points": len(market_data)})
    
    # 信号生成
    signals = generate_signals(market_data)
    perf.add_details({"signals_generated": len(signals)})
    
    # 订单执行
    orders = execute_orders(signals)
    perf.add_details({"orders_placed": len(orders)})
```

### 5. 日志文件管理

- 定期清理旧日志文件，避免磁盘空间不足
- 根据业务需求调整日志轮转策略
- 在生产环境中避免使用DEBUG级别
- 敏感信息（如密码、API密钥）不应记录到日志中

## 故障排查

### 常见问题

1. **日志文件无法创建**
   - 检查日志目录权限
   - 确保目录存在或程序有创建目录的权限

2. **日志轮转不工作**
   - 检查maxBytes和backupCount配置
   - 确保有足够的磁盘空间

3. **性能日志缺失**
   - 确保调用了PerformanceLogger的log()方法
   - 检查性能日志器的配置

4. **日志格式异常**
   - 检查formatter配置
   - 确保日志消息格式正确

### 调试技巧

```python
# 启用详细日志输出
logger_manager.set_level('all', 'DEBUG')

# 检查日志器配置
logger = logger_manager.get_logger('app')
print(f"日志器级别: {logger.level}")
print(f"处理器数量: {len(logger.handlers)}")

# 测试日志输出
logger.debug("这是调试信息")
logger.info("这是信息")
logger.warning("这是警告")
logger.error("这是错误")
```

## 扩展开发

### 自定义日志格式化器

```python
import logging

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # 添加自定义字段
        record.trader_id = getattr(record, 'trader_id', 'unknown')
        return super().format(record)
```

### 自定义日志处理器

```python
class DatabaseHandler(logging.Handler):
    def emit(self, record):
        # 将日志记录到数据库
        log_entry = self.format(record)
        save_to_database(log_entry)
```

## 总结

本日志系统提供了完整的日志管理功能，支持多种日志类型、性能监控、自动轮转和清理等特性。通过合理配置和使用，可以有效提升量化交易系统的可观测性和可维护性。

更多详细信息请参考源代码中的注释和测试用例。