# NASDAQ-100 数据抓取使用指南

## 概述

本系统提供了自动化的NASDAQ-100股票数据抓取功能，支持批量获取历史日频OHLCV数据并入库到PostgreSQL数据库。

## 功能特性

- **自动成分股获取**: 支持网络获取NASDAQ-100成分股列表，失败时自动回退到本地缓存
- **智能增量更新**: 自动检测数据库中已有数据，仅抓取缺失的日期范围
- **多种缓存格式**: 支持CSV和Parquet格式的本地数据缓存
- **灵活配置**: 通过环境变量控制抓取参数
- **错误处理**: 完善的错误处理和日志记录机制

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MAX_TICKERS` | 100 | 最大抓取股票数量 |
| `DATA_START_DATE` | 5年前 | 数据开始日期 (YYYY-MM-DD) |
| `DATA_END_DATE` | 今天 | 数据结束日期 (YYYY-MM-DD) |
| `BATCH_SIZE` | 1000 | 每批入库记录数 |
| `THROTTLE_SEC` | 0.5 | 每个股票间的暂停秒数 |
| `DATA_CACHE_DIR` | data_cache/nasdaq | 数据缓存目录 |
| `USE_PARQUET` | false | 是否使用Parquet格式缓存 |

## 使用方法

### 1. 基础使用

```bash
# 使用默认配置抓取所有NASDAQ-100股票
make data-fetch-nasdaq
```

### 2. 自定义配置

```bash
# 抓取前10只股票，从2024年开始，增加延迟
MAX_TICKERS=10 DATA_START_DATE=2024-01-01 THROTTLE_SEC=2.0 make data-fetch-nasdaq

# 使用Parquet格式缓存，自定义缓存目录
USE_PARQUET=true DATA_CACHE_DIR=my_data make data-fetch-nasdaq

# 小样本测试（推荐用于首次运行）
MAX_TICKERS=3 DATA_START_DATE=2024-12-01 THROTTLE_SEC=2.0 make data-fetch-nasdaq
```

### 3. 环境文件配置

创建 `.env` 文件：

```bash
# 数据抓取配置
MAX_TICKERS=50
DATA_START_DATE=2023-01-01
DATA_END_DATE=2024-12-31
BATCH_SIZE=500
THROTTLE_SEC=1.0
USE_PARQUET=true
DATA_CACHE_DIR=data_cache/nasdaq_custom
```

然后运行：

```bash
make data-fetch-nasdaq
```

## 最佳实践建议

### 🚀 首次使用建议
1. **小样本测试**：首次运行建议使用小样本验证系统
   ```bash
   MAX_TICKERS=3 DATA_START_DATE=2024-12-01 THROTTLE_SEC=2.0 make data-fetch-nasdaq
   ```

2. **检查网络环境**：确保网络连接稳定，避免在网络高峰期运行

3. **逐步扩大规模**：验证成功后再逐步增加 `MAX_TICKERS` 数量

### ⚡ 性能优化建议
1. **合理设置延迟**：
   - 网络良好时：`THROTTLE_SEC=0.5`（默认）
   - 频繁失败时：`THROTTLE_SEC=2.0` 或更高
   - 系统会自动检测失败率并动态调整延迟

2. **分批处理**：对于大量数据，建议分批运行
   ```bash
   # 第一批：前50只
   MAX_TICKERS=50 make data-fetch-nasdaq
   
   # 等待一段时间后运行第二批
   # 注意：需要修改脚本支持跳过前N只股票
   ```

3. **使用Parquet格式**：大数据量时推荐使用Parquet
   ```bash
   USE_PARQUET=true make data-fetch-nasdaq
   ```

### 🛠️ 故障排除
1. **频率限制错误**：
   - 增加 `THROTTLE_SEC` 值（如2.0或更高）
   - 等待一段时间后重新运行
   - 系统已内置指数退避重试机制

2. **网络连接问题**：
   - 检查网络连接
   - 考虑使用VPN或代理
   - 在网络较好的时段运行

3. **数据库连接问题**：
   - 确保PostgreSQL服务正在运行
   - 检查数据库配置和权限

### 📊 监控和验证
1. **查看处理进度**：脚本会显示实时进度和统计信息
2. **验证数据入库**：
   ```bash
   # 检查数据库记录数
   python -c "
   import sys
   sys.path.append('.')
   from src.database.dao import stock_data_dao
   from sqlalchemy import text
   
   with stock_data_dao.get_session() as session:
       result = session.execute(text('SELECT COUNT(*) FROM stock_data')).scalar()
       print(f'总记录数: {result}')
       
       symbols = session.execute(text('SELECT DISTINCT symbol FROM stock_data ORDER BY symbol')).fetchall()
       print(f'股票数量: {len(symbols)}')
       print('股票列表:', [s[0] for s in symbols[:10]])  # 显示前10个
   "
   ```

### 🔧 高级配置
1. **自定义日期范围**：
   ```bash
   DATA_START_DATE=2020-01-01 DATA_END_DATE=2023-12-31 make data-fetch-nasdaq
   ```

2. **增量更新**：系统自动检测已有数据，只下载新增部分

3. **批量大小调整**：
   ```bash
   BATCH_SIZE=500 make data-fetch-nasdaq  # 减少内存使用
   ```

## 前置条件

### 1. 安装依赖

```bash
# 安装Python依赖
make install

# 或手动安装
pip install yfinance pandas requests
```

### 2. 数据库准备

```bash
# 创建数据库并建表
make db-setup

# 或分步执行
make db-init
```

### 3. 环境检查

```bash
# 检查Python环境
python test_environment.py

# 检查数据库连接
python test_database_system.py
```

## 输出说明

### 1. 控制台输出

```
配置: MAX_TICKERS=10, 范围=2024-01-01~2024-12-31, BATCH_SIZE=1000, THROTTLE_SEC=0.5, CACHE_DIR=data_cache/nasdaq, PARQUET=false
开始处理 10 个符号
AAPL: 下载 2024-01-01 至 2024-12-31 的日线数据
AAPL: 入库完成，共 252 条
MSFT: 数据已最新，无需更新。
...
处理完成: 10 个符号，共入库 2520 条记录
```

### 2. 缓存文件

```
data_cache/nasdaq/
├── AAPL.csv
├── MSFT.csv
├── GOOGL.csv
└── ...
```

### 3. 数据库记录

数据将存储在 `stock_data` 表中，包含以下字段：
- `symbol`: 股票代码
- `date`: 交易日期
- `open`, `high`, `low`, `close`: OHLC价格
- `volume`: 成交量

## 常见问题

### Q: 如何验证数据是否正确入库？

```bash
# 查看数据库中的记录数
psql -d quant_trading -c "SELECT symbol, COUNT(*) FROM stock_data GROUP BY symbol ORDER BY symbol;"

# 查看特定股票的数据范围
psql -d quant_trading -c "SELECT symbol, MIN(date), MAX(date), COUNT(*) FROM stock_data WHERE symbol='AAPL' GROUP BY symbol;"
```

### Q: 抓取失败怎么办？

1. 检查网络连接
2. 检查数据库连接：`make db-setup`
3. 查看日志输出中的错误信息
4. 尝试减少 `MAX_TICKERS` 进行小批量测试

### Q: 如何重新抓取某个股票的数据？

```bash
# 删除数据库中的记录
psql -d quant_trading -c "DELETE FROM stock_data WHERE symbol='AAPL';"

# 删除缓存文件
rm data_cache/nasdaq/AAPL.csv

# 重新抓取
make data-fetch-nasdaq
```

### Q: 如何加速抓取过程？

1. 减少 `THROTTLE_SEC` 值（注意不要过于频繁请求）
2. 增加 `BATCH_SIZE` 值
3. 使用 `USE_PARQUET=true` 提高缓存性能
4. 分批处理：先抓取部分股票，再抓取其余

## 性能建议

- **首次抓取**: 建议使用较小的 `MAX_TICKERS` 值进行测试
- **增量更新**: 系统会自动跳过已有数据，适合定期运行
- **网络优化**: 适当调整 `THROTTLE_SEC` 平衡速度和稳定性
- **存储优化**: 大量数据建议启用 `USE_PARQUET=true`

## 监控和维护

### 定期更新

```bash
# 每日更新脚本示例
#!/bin/bash
cd /path/to/my-quant
source venv/bin/activate
make data-fetch-nasdaq
```

### 数据质量检查

```bash
# 检查数据完整性
python -c "
from src.database.dao import stock_data_dao
symbols = stock_data_dao.get_symbols()
print(f'数据库中共有 {len(symbols)} 只股票')
for symbol in symbols[:5]:
    count = len(stock_data_dao.get_by_symbol_and_date_range(symbol, None, None))
    print(f'{symbol}: {count} 条记录')
"
```