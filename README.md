# 量化交易系统 (Quant Trading System)

一个专为两人初创团队设计的完整量化交易系统，包含数据获取、因子计算、策略回测、绩效分析等核心功能。

## 🚀 项目特色

- **模块化设计**: 清晰的代码结构，易于维护和扩展
- **完整工具链**: 从数据获取到策略部署的全流程支持
- **高性能**: 支持大规模数据处理和并行计算
- **可视化**: 丰富的图表和报告生成功能
- **易于部署**: Docker支持，一键部署
- **测试覆盖**: 完整的单元测试和集成测试

## 📋 系统架构

```
量化交易系统
├── 数据获取层 (Data Layer)
│   ├── 多数据源支持 (Yahoo Finance, Alpha Vantage等)
│   ├── 数据缓存机制
│   └── 数据质量检查
├── 策略开发层 (Strategy Layer)
│   ├── 因子计算引擎
│   ├── 策略模板库
│   └── 信号生成器
├── 回测引擎 (Backtest Engine)
│   ├── 历史数据回测
│   ├── 交易成本模拟
│   └── 风险控制
└── 分析报告层 (Analytics Layer)
    ├── 绩效指标计算
    ├── 风险分析
    └── 可视化报告
```

## 🛠️ 技术栈

- **后端**: Python 3.9+, FastAPI, SQLAlchemy
- **数据库**: PostgreSQL, Redis
- **数据处理**: Pandas, NumPy, SciPy
- **机器学习**: Scikit-learn, Statsmodels
- **可视化**: Matplotlib, Seaborn, Plotly
- **前端**: Streamlit
- **测试**: Pytest, Coverage
- **部署**: Docker, Docker Compose

## 📁 项目结构

```
my-quant/
├── src/                    # 源代码目录
│   ├── data/              # 数据获取和管理
│   ├── factors/           # 因子计算
│   ├── backtest/          # 回测引擎
│   ├── performance/       # 绩效分析
│   ├── strategies/        # 策略模板
│   └── utils/             # 工具函数
├── config/                # 配置文件
├── tests/                 # 测试文件
├── docs/                  # 文档
├── notebooks/             # Jupyter笔记本
├── logs/                  # 日志文件
├── requirements.txt       # 依赖包
├── pyproject.toml        # 项目配置
├── Makefile              # 开发工具
└── README.md             # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

确保你的系统已安装:
- Python 3.9+
- Git
- PostgreSQL (可选，用于生产环境)
- Redis (可选，用于缓存)

### 2. 克隆项目

```bash
git clone <your-repo-url>
cd my-quant
```

### 3. 一键设置开发环境

```bash
make quickstart
```

这个命令会:
- 创建Python虚拟环境
- 安装所有依赖包
- 创建必要的目录
- 复制配置文件模板

### 4. 配置环境变量

编辑 `.env` 文件，根据你的环境修改配置:

```bash
cp .env.example .env
vim .env  # 或使用你喜欢的编辑器
```

### 5. 初始化数据库（可选）

如果使用PostgreSQL:

```bash
make db-init
```

### 6. 运行测试

```bash
make test
```

### 7. 启动服务

启动API服务:
```bash
make run-api
```

启动Web界面:
```bash
make run-streamlit
```

## 📖 开发指南

### 常用命令

```bash
# 查看所有可用命令
make help

# 代码格式化
make format

# 代码检查
make lint

# 类型检查
make type-check

# 运行快速测试
make test-fast

# 生成测试覆盖率报告
make test-cov

# 运行所有预提交检查
make pre-commit
```

### 开发工作流

1. **创建新分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **开发功能**
   - 编写代码
   - 添加测试
   - 更新文档

3. **代码检查**
   ```bash
   make pre-commit
   ```

4. **提交代码**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **推送并创建PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### 代码规范

- 使用 **Black** 进行代码格式化
- 使用 **isort** 进行导入排序
- 使用 **flake8** 进行代码检查
- 使用 **mypy** 进行类型检查
- 使用 **pytest** 进行测试
- 测试覆盖率要求 > 80%

## 🧪 测试

### 测试分类

- **单元测试**: 测试单个函数或类
- **集成测试**: 测试模块间的交互
- **性能测试**: 测试系统性能
- **API测试**: 测试API接口

### 运行测试

```bash
# 运行所有测试
make test

# 运行快速测试（排除慢速测试）
make test-fast

# 运行特定标记的测试
pytest -m "unit"          # 只运行单元测试
pytest -m "integration"   # 只运行集成测试
pytest -m "not slow"      # 排除慢速测试

# 运行特定文件的测试
pytest tests/test_data.py

# 运行特定测试函数
pytest tests/test_data.py::test_data_manager
```

## 📊 使用示例

### 基本数据获取

```python
from src.data.manager import DataManager

# 创建数据管理器
dm = DataManager()

# 获取股票数据
data = dm.get_stock_data('AAPL', '2023-01-01', '2023-12-31')
print(data.head())
```

### 策略回测

```python
from src.backtest.engine import BacktestEngine
from src.strategies.momentum import MomentumStrategy

# 创建策略
strategy = MomentumStrategy(lookback=20)

# 创建回测引擎
engine = BacktestEngine(initial_capital=1000000)

# 运行回测
results = engine.run_backtest(strategy, start_date='2023-01-01', end_date='2023-12-31')

# 查看结果
print(f"总收益率: {results.total_return:.2%}")
print(f"夏普比率: {results.sharpe_ratio:.2f}")
print(f"最大回撤: {results.max_drawdown:.2%}")
```

### 因子计算

```python
from src.factors.technical import TechnicalFactors

# 创建技术因子计算器
tf = TechnicalFactors()

# 计算技术指标
data_with_factors = tf.calculate_all_factors(data)
print(data_with_factors.columns.tolist())
```

## 🐳 Docker部署

### 构建镜像

```bash
make docker-build
```

### 运行容器

```bash
make docker-run
```

### 使用Docker Compose

```bash
# 启动所有服务
make docker-compose-up

# 停止所有服务
make docker-compose-down
```

## 📈 性能优化

- **数据缓存**: 使用Redis缓存频繁访问的数据
- **并行计算**: 使用多进程处理大量数据
- **数据库优化**: 合理设计索引和查询
- **内存管理**: 及时释放不需要的数据

## 🔒 安全考虑

- **环境变量**: 敏感信息存储在环境变量中
- **API认证**: 实现JWT认证机制
- **数据加密**: 敏感数据加密存储
- **访问控制**: 实现基于角色的访问控制

## 📚 文档

- [API文档](docs/api.md)
- [策略开发指南](docs/strategy_guide.md)
- [部署指南](docs/deployment.md)
- [FAQ](docs/faq.md)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系我们

- 项目主页: [GitHub Repository](https://github.com/your-org/quant-trading-system)
- 问题反馈: [Issues](https://github.com/your-org/quant-trading-system/issues)
- 邮箱: team@quantsystem.com

## 🙏 致谢

感谢以下开源项目的支持:
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)

---

**Happy Trading! 📈**