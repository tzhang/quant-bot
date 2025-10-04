# 🚀 量化交易系统 (Quantitative Trading System)

一个功能完整、易于使用的Python量化交易系统，专为量化投资研究和策略开发而设计。

## ✨ 主要特性

### 📊 数据管理
- **智能缓存系统**: 支持磁盘缓存和TTL过期机制
- **多数据源支持**: 集成Yahoo Finance等主流数据源
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
- **投资决策**: 股票筛选、组合优化、风险控制
- **教学培训**: 量化投资教学、实践演示
- **个人投资**: 个人投资者的量化工具

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 操作系统: Windows/macOS/Linux
- 内存: 建议4GB以上
- 磁盘空间: 建议1GB以上

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/quant-bot.git
   cd quant-bot
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **环境检测**
   ```bash
   python test_environment.py
   ```

4. **快速体验**
   ```bash
   python examples/quick_start_demo.py
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

## 📚 文档指南

### 🎓 初学者资源
- **[初学者指南](docs/BEGINNER_GUIDE.md)** - 从零开始的完整教程
- **[环境配置](test_environment.py)** - 环境检测和配置指南
- **[快速演示](examples/quick_start_demo.py)** - 5分钟快速体验

### 📖 使用教程
- **[数据获取教程](examples/data_tutorial.py)** - 数据获取和缓存使用
- **[因子计算教程](examples/factor_tutorial.py)** - 因子计算和评估实战
- **[图表解读指南](docs/CHART_INTERPRETATION_GUIDE.md)** - 图表分析和解读

### 🔧 进阶资源
- **[进阶技巧](docs/ADVANCED_TIPS_PRACTICES.md)** - 高级功能和最佳实践
- **[常见问题](docs/FAQ_TROUBLESHOOTING.md)** - 问题排查和解决方案
- **[API文档](docs/API_REFERENCE.md)** - 详细的API参考

## 💡 使用示例

### 基础数据获取

```python
from src.data_engine import DataEngine

# 初始化数据引擎
engine = DataEngine()

# 获取股票数据
symbols = ['AAPL', 'GOOGL', 'MSFT']
data = engine.get_data(symbols, period='1y')

print(f"获取到 {len(data)} 只股票的数据")
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
量化交易系统
├── 数据层 (Data Layer)
│   ├── 数据获取 (Yahoo Finance API)
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
└── 展示层 (Presentation Layer)
    ├── 交互式图表 (Plotly)
    ├── 报告生成 (HTML/PDF)
    └── 实时监控 (Dashboard)
```

## 🎨 功能展示

### 数据缓存效果
```
首次获取: 2.34秒 (网络下载)
缓存获取: 0.19秒 (本地读取)
加速比: 12.3x ⚡
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

### v1.2.0 (当前版本)
- ✅ 完整的数据管理持久化层
- ✅ 因子评估报告系统
- ✅ 5种交互式图表生成
- ✅ 智能缓存和自动清理
- ✅ 完整的初学者指南

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

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！

🚀 开始您的量化投资之旅：`python examples/quick_start_demo.py`