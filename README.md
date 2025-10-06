# 🚀 量化交易系统 (Quantitative Trading System)

一个功能完整、易于使用的Python量化交易系统，专为量化投资研究和策略开发而设计。

## ✨ 主要特性

### 🎯 策略开发与回测 (v1.4.0 新增)
- **完整策略框架**: 6种内置量化策略（均值回归、动量、RSI、布林带、MACD、波动率突破）
- **增强回测引擎**: 支持风险管理、仓位管理、走势前进分析
- **批量策略测试**: 多策略并行测试、性能对比、参数优化
- **交互式仪表板**: 净值曲线、性能雷达图、风险收益散点图、回撤分析

### 🧠 多因子量化模型 (v1.4.0 新增)
- **多因子计算**: 动量、均值回归、波动率、成交量、RSI、MACD、布林带
- **投资组合构建**: 等权重、因子加权、风险平价
- **模型训练**: 线性回归、随机森林
- **策略性能评估**: 全面的策略性能分析和可视化

### 📊 数据管理 (v1.5.0 新增三数据源集成)
- **三级数据源回退**: Qlib → OpenBB → yfinance 智能回退机制
- **OpenBB Platform集成**: 支持开源金融数据平台，数据源更丰富
- **智能缓存系统**: 支持磁盘缓存和TTL过期机制
- **多数据源支持**: 集成Qlib、OpenBB、Yahoo Finance等主流数据源
- **数据可用性检查**: 自动检测各数据源状态，推荐最佳数据源
- **灵活数据源选择**: 支持强制使用特定数据源或自动回退
- **自动数据清理**: 定期清理过期缓存，优化存储空间
- **高效数据获取**: 缓存机制可提升数据获取速度10-50倍

### 🧮 因子计算
- **技术因子库**: 内置20+种常用技术因子
- **多时间框架**: 支持不同周期的因子计算
- **自定义因子**: 灵活的因子开发框架
- **批量计算**: 支持多股票、多因子并行计算

### 📈 因子评估
- **IC分析**: 信息系数计算和统计检验
- **分层测试**: 多分位数组合收益分析
- **换手率分析**: 因子稳定性评估
- **风险调整收益**: Sharpe比率、最大回撤等指标

### 📋 可视化分析
- **交互式图表**: 基于Plotly的高质量图表
- **多维度展示**: IC时序、分层收益、累计收益等
- **自动报告生成**: 一键生成完整的因子评估报告
- **移动端适配**: 图表支持移动设备查看

### 🛠️ 开发工具
- **环境检测**: 自动检测和诊断开发环境
- **调试工具**: 性能分析、内存监控、错误追踪
- **实时监控**: 策略运行状态实时监控
- **最佳实践**: 完整的开发规范和代码示例

## 🎯 适用场景

- **量化研究**: 因子挖掘、策略回测、风险分析
- **策略开发**: 多策略开发、参数优化、性能评估
- **投资决策**: 股票筛选、组合优化、风险控制
- **教学培训**: 量化投资教学、实践演示
- **个人投资**: 个人投资者的量化工具

## 🚀 快速开始

### 环境要求

- **Python 3.12** (强制要求，必须使用此版本)
- 操作系统: Windows/macOS/Linux
- 内存: 建议4GB以上
- 磁盘空间: 建议1GB以上

> ⚠️ **重要提醒**: 本项目强制要求使用 Python 3.12 版本。其他版本可能导致依赖库兼容性问题。

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/quant-bot.git
   cd quant-bot
   ```

2. **创建Python 3.12虚拟环境**
   ```bash
   # 确保使用Python 3.12
   python3.12 --version  # 应显示 Python 3.12.x
   
   # 创建虚拟环境
   python3.12 -m venv venv
   
   # 激活虚拟环境
   source venv/bin/activate  # Linux/macOS
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **环境检测**
   ```bash
   python test_environment.py
   ```

5. **快速体验**
   ```bash
   python examples/quick_start_demo.py
   ```

6. **策略测试演示** (v1.4.0 新增)
   ```bash
   python examples/strategy_testing_demo.py
   ```

### 5分钟快速体验

运行快速开始演示，体验系统核心功能：

```bash
python examples/quick_start_demo.py
```

演示内容包括：
- 📊 数据获取和缓存 (体验10-50倍速度提升)
- 🧮 技术因子计算 (5种常用因子)
- 📈 因子评估分析 (IC分析、分层测试)
- 📋 可视化图表 (5种交互式图表)
- 🎯 结果解读 (个性化分析建议)

### 策略开发与回测演示 (v1.4.0 新增)

运行策略测试演示，体验完整的策略开发流程：

```bash
python examples/strategy_testing_demo.py
```

演示内容包括：
- 🎯 6种量化策略测试 (均值回归、动量、RSI、布林带、MACD、波动率突破)
- 📊 批量策略性能对比 (夏普比率、最大回撤、年化收益)
- 📈 交互式策略仪表板 (净值曲线、性能雷达图、风险收益散点图)
- 📋 策略回撤分析 (最大回撤、回撤持续时间、回撤恢复)
- 🔥 月度收益热力图 (直观展示策略月度表现)
- 📁 自动生成报告 (HTML交互式报告 + PNG图片导出)

## 📚 文档指南

### 🎓 初学者资源
- **[初学者指南](docs/BEGINNER_GUIDE.md)** - 从零开始的完整教程
- **[环境配置](test_environment.py)** - 环境检测和配置指南
- **[快速演示](examples/quick_start_demo.py)** - 5分钟快速体验

### 📖 使用教程
- **[数据获取教程](examples/data_tutorial.py)** - 数据获取和缓存使用
- **[因子计算教程](examples/factor_tutorial.py)** - 因子计算和评估实战
- **[策略测试教程](examples/strategy_testing_demo.py)** - 策略开发与回测实战 (v1.4.0 新增)
- **[图表解读指南](docs/CHART_INTERPRETATION_GUIDE.md)** - 图表分析和解读

### 🔧 进阶资源
- **[进阶技巧](docs/ADVANCED_TIPS_PRACTICES.md)** - 高级功能和最佳实践
- **[常见问题](docs/FAQ_TROUBLESHOOTING.md)** - 问题排查和解决方案
- **[API文档](docs/API_REFERENCE.md)** - 详细的API参考

## 💡 使用示例

### 基础数据获取 (v1.5.0 三数据源集成)

```python
from src.data.data_adapter import create_data_adapter

# 创建数据适配器（支持三级回退）
adapter = create_data_adapter(
    prefer_qlib=True,          # 优先使用Qlib
    enable_openbb=True,        # 启用OpenBB
    fallback_to_yfinance=True  # 回退到yfinance
)

# 获取股票数据（自动选择最佳数据源）
symbols = ['AAPL', 'GOOGL', 'MSFT']
data = adapter.get_multiple_stocks_data(symbols, start_date='2024-01-01')

print(f"获取到 {len(data)} 只股票的数据")

# 检查数据可用性
availability = adapter.check_data_availability('AAPL')
print(f"推荐数据源: {availability['recommended_source']}")

# 强制使用特定数据源
openbb_data = adapter.get_stock_data('AAPL', force_source='openbb')
```

### 因子计算和评估

```python
from src.technical_factors import TechnicalFactors
from src.factor_evaluation import FactorEvaluator

# 计算技术因子
tech_factors = TechnicalFactors()
momentum = tech_factors.momentum(data['AAPL'], period=20)

# 因子评估
evaluator = FactorEvaluator()
ic_results = evaluator.calculate_ic(momentum, data['AAPL']['close'])

print(f"IC均值: {ic_results['ic_mean']:.4f}")
print(f"IC信息比率: {ic_results['ic_ir']:.4f}")
```

### 策略开发与回测 (v1.4.0 新增)

```python
from src.strategies.templates import MACDStrategy
from src.strategies.strategy_tester import StrategyTester
from src.visualization.strategy_dashboard import StrategyDashboard

# 创建MACD策略
strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)

# 策略回测
tester = StrategyTester(initial_capital=100000)
result = tester.backtest_strategy(strategy, data['AAPL'])

print(f"策略收益: {result['metrics']['cum_return']:.2%}")
print(f"夏普比率: {result['metrics']['sharpe']:.4f}")
print(f"最大回撤: {result['metrics']['max_drawdown']:.2%}")

# 生成策略仪表板
dashboard = StrategyDashboard()
dashboard.create_comprehensive_dashboard(
    [result], 
    ['MACD策略'], 
    output_file='strategy_dashboard.html'
)
```

### 多因子量化模型 (v1.4.0 新增)

```python
from src.strategies.multi_factor import MultiFactorModel

# 创建多因子模型
model = MultiFactorModel()

# 计算多因子
factors = model.calculate_factors(data)
print(f"计算了 {len(factors.columns)} 个因子")

# 构建投资组合
portfolio = model.build_portfolio(factors, method='factor_weighted')
print(f"投资组合包含 {len(portfolio)} 只股票")

# 模型训练
model.train_model(factors, returns, model_type='linear')
predictions = model.predict(factors)
```

### 生成评估图表

```python
# 生成完整的因子评估图表
charts = evaluator.create_factor_charts(
    momentum, 
    data['AAPL']['close'],
    factor_name="20日动量",
    symbol="AAPL"
)

print("生成的图表:")
for chart_type, filepath in charts.items():
    print(f"  {chart_type}: {filepath}")
```

## 📊 系统架构

```
量化交易系统 (v1.5.0)
├── 数据层 (Data Layer) - 三数据源集成
│   ├── Qlib数据源 (本地量化数据，优先级1)
│   ├── OpenBB Platform (开源金融平台，优先级2)
│   ├── Yahoo Finance (备用数据源，优先级3)
│   ├── 智能回退机制 (自动选择最佳数据源)
│   ├── 缓存管理 (磁盘缓存 + TTL)
│   └── 数据清理 (自动清理机制)
├── 计算层 (Computation Layer)
│   ├── 技术因子 (20+ 技术指标)
│   ├── 基本面因子 (财务指标)
│   └── 自定义因子 (用户扩展)
├── 分析层 (Analysis Layer)
│   ├── 因子评估 (IC分析、分层测试)
│   ├── 风险分析 (回撤、波动率)
│   └── 绩效归因 (收益分解)
├── 策略层 (Strategy Layer) - v1.4.0 新增
│   ├── 策略框架 (6种内置策略)
│   ├── 回测引擎 (风险管理、仓位管理)
│   ├── 多因子模型 (投资组合构建)
│   └── 性能评估 (全面分析)
└── 展示层 (Presentation Layer)
    ├── 交互式图表 (Plotly)
    ├── 策略仪表板 (实时监控)
    ├── 报告生成 (HTML/PDF)
    └── 移动端适配 (响应式设计)
```

## 🎨 功能展示

### 数据缓存效果 (三数据源对比)
```
Qlib本地数据: 0.05秒 (本地读取) ⚡⚡⚡
OpenBB平台: 1.23秒 (API调用) ⚡⚡
yfinance: 2.34秒 (网络下载) ⚡
缓存获取: 0.19秒 (磁盘缓存)
最大加速比: 46.8x (Qlib vs 网络)
```

### 三数据源可用性统计
```
📊 数据源覆盖率统计:
  Qlib: 中国A股 + 部分美股 (本地数据)
  OpenBB: 全球股票 + 多资产类别 ✅
  yfinance: 全球股票 (Yahoo Finance) ✅
  
📈 数据质量对比:
  Qlib: 高质量量化数据 ⭐⭐⭐⭐⭐
  OpenBB: 专业金融数据 ⭐⭐⭐⭐
  yfinance: 免费公开数据 ⭐⭐⭐
  
🚀 推荐使用场景:
  量化研究: Qlib (本地高速)
  全球投资: OpenBB (数据丰富)
  个人学习: yfinance (免费易用)
```

### 因子评估结果
```
📊 20日动量因子评估结果:
  IC均值: 0.0234 ✅
  IC标准差: 0.156
  IC信息比率: 0.456 ✅
  胜率: 54.2% ✅
  
📈 分层测试 (年化收益):
  第1层(最低): -2.3%
  第2层: 4.1%
  第3层: 8.7%
  第4层: 12.4%
  第5层(最高): 18.9% ✅
  
  多空收益: 21.2% 🚀
```

### 生成的图表类型
- 📈 **IC时序图**: 因子预测能力的时间变化
- 📊 **分层收益图**: 不同分位数组合的收益对比
- 📋 **累计收益图**: 因子策略的长期表现
- 🎯 **换手率分析**: 因子稳定性评估
- 📉 **回撤分析**: 风险控制效果展示

## 🔄 版本历史

### v1.5.0 (当前版本) - 三数据源集成
- ✅ **OpenBB Platform集成**: 新增开源金融数据平台支持
- ✅ **三级数据源回退**: Qlib → OpenBB → yfinance 智能回退机制
- ✅ **数据可用性检查**: 自动检测各数据源状态和推荐
- ✅ **灵活数据源选择**: 支持强制使用特定数据源
- ✅ **多股票批量获取**: 优化多股票数据获取性能
- ✅ **完整集成测试**: 三数据源集成测试和验证

### v1.4.0 - 策略开发与回测
- ✅ **完整策略框架**: 6种内置量化策略
- ✅ **增强回测引擎**: 风险管理、仓位管理
- ✅ **多因子量化模型**: 投资组合构建和模型训练
- ✅ **交互式仪表板**: 策略性能可视化

### v1.2.0 - 券商集成
- ✅ **7个券商API支持**: 统一交易接口
- ✅ **实时监控系统**: 系统监控和告警
- ✅ **完整的数据管理持久化层**
- ✅ **因子评估报告系统**
- ✅ **5种交互式图表生成**
- ✅ **智能缓存和自动清理**
- ✅ **完整的初学者指南**

### v1.1.0
- ✅ 基础因子计算框架
- ✅ 数据获取和缓存
- ✅ 基本的可视化功能

### v1.0.0
- ✅ 项目初始化
- ✅ 核心架构设计

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献类型
- 🐛 Bug修复
- ✨ 新功能开发
- 📚 文档改进
- 🎨 代码优化
- 🧪 测试用例
- 💡 功能建议

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系我们

- **项目主页**: https://github.com/your-username/quant-bot
- **问题反馈**: https://github.com/your-username/quant-bot/issues
- **讨论交流**: https://github.com/your-username/quant-bot/discussions

## 🙏 致谢

感谢以下开源项目的支持：
- [pandas](https://pandas.pydata.org/) - 数据处理
- [numpy](https://numpy.org/) - 数值计算
- [plotly](https://plotly.com/) - 数据可视化
- [yfinance](https://github.com/ranaroussi/yfinance) - 金融数据获取
- [OpenBB Platform](https://openbb.co/) - 开源金融数据平台 (v1.5.0 新增)
- [Qlib](https://github.com/microsoft/qlib) - 微软量化投资平台

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！

🚀 开始您的量化投资之旅：`python examples/quick_start_demo.py`