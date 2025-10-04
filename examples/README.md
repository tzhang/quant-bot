# 量化交易系统示例集合

本目录包含了量化交易系统的各种示例和演示脚本，帮助用户快速了解和使用系统功能。

## 📁 文件结构

### 🚀 快速开始
- **`quick_start_demo.py`** - 系统快速入门演示
  - 数据获取和缓存机制
  - 技术因子计算
  - 因子评估和分析
  - 策略表现可视化

### 📊 图表画廊
- **`chart_gallery.py`** - 完整的图表演示集合
  - 技术分析图表 (`technical_analysis_gallery.png`)
  - 因子分析图表 (`factor_analysis_gallery.png`)
  - 策略表现图表 (`strategy_performance_gallery.png`)
  - 市场分析图表 (`market_analysis_gallery.png`)

### 📚 教程系列
- **`data_tutorial.py`** - 数据管理教程
- **`factor_tutorial.py`** - 因子计算教程
- **`factor_evaluation.py`** - 因子评估教程

### 🎯 MVP演示
- **`mvp_demo.py`** - 最小可行产品演示

## 🖼️ 生成的图表文件

### 技术分析图表
- `technical_analysis_gallery.png` - 包含价格走势、技术指标、布林带等
- `price_signal_demo.png` - 价格与交易信号图
- `factor_score_demo.png` - 因子得分图

### 快速演示图表
- `equity_curve_demo.png` - 收益曲线图
- `drawdown_demo.png` - 回撤分析图
- `price_signal_demo.png` - 价格信号图

### 策略表现图表
- `strategy_performance_gallery.png` - 策略收益、回撤、风险指标
- `equity_curve_demo.png` - 权益曲线图
- `drawdown_demo.png` - 回撤分析图

### 因子分析图表
- `factor_analysis_gallery.png` - 因子分布、相关性、IC分析
- `factor_eval_*.png` - 各种因子评估图表

### 市场分析图表
- `market_analysis_gallery.png` - 多股票对比、相关性分析
- `mvp_*.png` - MVP演示相关图表

## 🏃‍♂️ 快速运行

### 1. 快速入门演示
```bash
python examples/quick_start_demo.py
```

### 2. 生成完整图表画廊
```bash
python examples/chart_gallery.py
```

### 3. 运行特定教程
```bash
python examples/data_tutorial.py
python examples/factor_tutorial.py
python examples/factor_evaluation.py
```

## 📈 图表说明

### 技术分析图表集合
- **价格走势图**: 显示股票价格的历史走势
- **移动平均线**: SMA和EMA的对比分析
- **布林带**: 价格波动区间分析
- **RSI指标**: 相对强弱指数分析
- **MACD指标**: 移动平均收敛发散分析
- **成交量分析**: 价格与成交量关系

### 因子分析图表集合
- **因子分布图**: 因子值的统计分布
- **因子时序图**: 因子随时间的变化
- **IC分析图**: 因子预测能力分析
- **因子分层回测**: 不同因子值的收益表现
- **因子相关性**: 多因子间的相关性分析

### 策略表现图表集合
- **累计收益对比**: 策略与基准的收益对比
- **回撤分析**: 最大回撤和回撤持续时间
- **收益分布**: 日收益率的统计分布
- **风险指标**: 夏普比率、波动率等指标
- **月度收益热力图**: 按月份显示收益表现

### 市场分析图表集合
- **股票价格对比**: 多只股票的标准化价格走势
- **相关性矩阵**: 股票间收益率相关性
- **波动率对比**: 各股票的年化波动率
- **滚动相关性**: 股票间相关性的时间变化
- **收益率分布**: 各股票收益率的概率分布
- **风险收益散点图**: 风险与收益的关系分析

## 🎯 使用建议

1. **新手用户**: 先运行 `quick_start_demo.py` 了解基本功能
2. **进阶用户**: 运行 `chart_gallery.py` 查看完整的分析图表
3. **开发者**: 使用 `test_all_examples.py` 验证系统功能
4. **学习者**: 按顺序运行各个教程脚本

## ⚠️ 重要提醒

- 所有示例都使用模拟数据，避免了对外部API的依赖
- 图表生成可能需要几秒钟时间，请耐心等待
- 如果遇到字体警告，不影响图表生成，可以忽略
- 建议在虚拟环境中运行示例

## 🔧 故障排除

如果遇到问题，请：

1. 确保已安装所有依赖包
2. 检查Python版本（建议3.8+）
3. 运行 `test_all_examples.py` 进行诊断
4. 查看生成的错误日志

## 📚 更多资源

- 查看 `../docs/visual_guide.md` 获取详细的图表说明
- 参考主项目README了解系统架构
- 查看源代码了解实现细节

---

🎉 **祝您使用愉快！如有问题，欢迎反馈。**