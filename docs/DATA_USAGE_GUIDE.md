# 数据使用指南

## 概述

本量化因子分析系统提供了完整的数据管理和分析功能。本指南将帮助您正确使用数据获取和分析功能。

## 数据管理架构

### 核心组件

1. **DataManager** (`src/data/data_manager.py`)
   - 主要的数据获取和管理类
   - 支持多数据源（yfinance等）
   - 内置缓存机制
   - 数据质量检查

2. **FactorEngine** (`src/factors/engine.py`)
   - 因子计算引擎
   - 技术指标计算
   - 风险指标分析
   - 基本面分析

## 正确的数据获取方法

### 使用DataManager获取数据

```python
from src.data.data_manager import DataManager

# 初始化数据管理器
data_manager = DataManager()

# 获取单只股票数据
data = data_manager.get_data(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    data_type='ohlcv'
)

# 获取多只股票数据
symbols = ['AAPL', 'GOOGL', 'MSFT']
for symbol in symbols:
    data = data_manager.get_data(
        symbol=symbol,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    print(f"{symbol} 数据形状: {data.shape}")
```

### 使用缓存数据

系统会自动缓存获取的数据，避免重复请求：

```python
# 第一次请求会从网络获取并缓存
data1 = data_manager.get_data('AAPL', '2023-01-01', '2024-01-01')

# 第二次请求会直接使用缓存
data2 = data_manager.get_data('AAPL', '2023-01-01', '2024-01-01')
```

## 数据源优先级

系统按以下优先级获取数据：

1. **IB TWS API** - 实时数据，高质量
2. **Qlib** - 学术研究数据
3. **OpenBB** - 开源数据平台

### 智能回退机制

当主要数据源不可用时，系统会自动切换到下一个数据源：

```
IB TWS API (失败) → Qlib → OpenBB
```

## 避免API限制的最佳实践

### 1. 使用数据缓存
```python
# 启用缓存减少API调用
config = {
    'ENABLE_CACHE': True,
    'CACHE_EXPIRY_HOURS': 24
}
```

### 2. 合理设置请求频率
```python
# 控制请求频率
import time
time.sleep(1)  # 每次请求间隔1秒
```

### 3. 批量获取数据
```python
# 一次获取多个股票数据
symbols = ['AAPL', 'GOOGL', 'MSFT']
data = adapter.get_batch_data(symbols)
```

### 4. 错误处理和重试机制

```python
def safe_get_data(data_manager, symbol, start_date, end_date, max_retries=3):
    """安全的数据获取函数，包含重试机制"""
    for attempt in range(max_retries):
        try:
            data = data_manager.get_data(symbol, start_date, end_date)
            return data
        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"❌ {symbol} 数据获取最终失败")
                return None
```

## 因子分析示例

### 使用FactorEngine进行因子计算

```python
from src.factors.engine import FactorEngine

# 初始化因子引擎
factor_engine = FactorEngine()

# 获取数据（使用DataManager）
data_manager = DataManager()
data = data_manager.get_data('AAPL', '2023-01-01', '2024-01-01')

# 计算技术因子
technical_factors = factor_engine.compute_technical(data)

# 计算风险因子
risk_factors = factor_engine.compute_risk(data)

# 计算所有因子
all_factors = factor_engine.compute_all(data)
```

## 数据质量检查

```python
def check_data_quality(data):
    """检查数据质量"""
    print(f"数据形状: {data.shape}")
    print(f"缺失值: {data.isnull().sum().sum()}")
    print(f"时间范围: {data.index[0]} 到 {data.index[-1]}")
    print(f"数据类型: {data.dtypes}")
    
    # 检查异常值
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in data.columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = data[(data[col] < q1 - 1.5*iqr) | (data[col] > q3 + 1.5*iqr)]
            print(f"{col} 异常值数量: {len(outliers)}")
```

## 常见问题解决

### 1. yfinance API限制

**问题**: `YFRateLimitError: Too many requests`

**解决方案**:
- 使用缓存数据
- 添加请求间隔
- 减少并发请求
- 使用代理服务器

### 2. 数据库连接问题

**问题**: `FATAL: database "quant_trading" does not exist`

**解决方案**:
- 检查数据库配置
- 创建必要的数据库
- 使用文件缓存作为替代

### 3. 数据格式问题

**问题**: 缓存数据格式不匹配

**解决方案**:
- 检查CSV文件结构
- 正确处理多级列索引
- 验证日期格式

## 示例脚本

系统提供了以下示例脚本：

1. **`examples/data_tutorial.py`** - 基础数据获取教程
2. **`examples/cached_data_demo.py`** - 缓存数据使用演示
3. **`examples/data_fetch_demo.py`** - 数据获取演示

## 最佳实践总结

1. ✅ 优先使用缓存数据
2. ✅ 添加适当的请求延迟
3. ✅ 实现错误处理和重试机制
4. ✅ 定期检查数据质量
5. ✅ 使用DataManager而不是直接调用FactorEngine.get_data
6. ✅ 监控API使用限制
7. ✅ 保持数据缓存的整洁和更新

通过遵循这些指南，您可以有效地使用量化因子分析系统进行数据分析，避免常见的API限制和数据质量问题。