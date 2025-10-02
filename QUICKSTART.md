# 量化交易系统快速启动指南

## 🎉 环境已就绪！

恭喜！您的 Python 量化交易开发环境已经成功配置完成。所有核心库都已安装并测试通过。

## 📋 环境概览

### ✅ 已安装的核心库
- **数据处理**: pandas 2.3.3, numpy 2.2.6
- **数据获取**: yfinance 0.2.66 (已测试，可获取实时股票数据)
- **机器学习**: scikit-learn 1.7.2, statsmodels 0.14.5
- **技术分析**: pandas-ta (已测试，SMA、RSI等指标正常)
- **可视化**: matplotlib, seaborn 0.13.2, plotly
- **Web框架**: FastAPI 0.118.0, Streamlit 1.50.0
- **数据库**: SQLAlchemy 2.0.43, Redis 6.4.0
- **开发工具**: pytest 8.4.2, black, flake8, jupyter

## 🚀 立即开始

### 1. 激活虚拟环境
```bash
source venv/bin/activate
```

### 2. 验证环境
```bash
python test_environment.py
```

### 3. 启动 Jupyter Notebook
```bash
jupyter notebook
```

### 4. 或者启动 JupyterLab
```bash
jupyter lab
```

## 📊 快速示例

### 获取股票数据
```python
import yfinance as yf
import pandas as pd

# 获取苹果股票数据
aapl = yf.Ticker("AAPL")
data = aapl.history(period="1mo")
print(data.head())
```

### 计算技术指标
```python
import pandas_ta as ta

# 计算移动平均线
data['SMA_20'] = ta.sma(data['Close'], length=20)
data['SMA_50'] = ta.sma(data['Close'], length=50)

# 计算 RSI
data['RSI'] = ta.rsi(data['Close'], length=14)

# 计算 MACD
macd = ta.macd(data['Close'])
data = data.join(macd)
```

### 可视化数据
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
sns.set_style("whitegrid")

# 绘制价格和移动平均线
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='收盘价', linewidth=2)
plt.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
plt.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
plt.title('AAPL 股价和移动平均线')
plt.legend()
plt.show()
```

### 简单策略回测
```python
import numpy as np

# 简单的移动平均交叉策略
data['Signal'] = 0
data['Signal'][data['SMA_20'] > data['SMA_50']] = 1
data['Signal'][data['SMA_20'] <= data['SMA_50']] = -1

# 计算策略收益
data['Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']

# 计算累积收益
data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Cumulative_Returns'], label='买入持有')
plt.plot(data.index, data['Cumulative_Strategy'], label='移动平均策略')
plt.title('策略回测结果')
plt.legend()
plt.show()
```

## 🛠️ 开发工具

### 代码格式化
```bash
# 格式化代码
black .

# 排序导入
isort .

# 代码检查
flake8 .
```

### 运行测试
```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=src
```

### 启动 Web 应用
```bash
# 启动 Streamlit 应用
streamlit run app.py

# 启动 FastAPI 应用
uvicorn main:app --reload
```

## 📁 项目结构

```
my-quant/
├── src/                    # 源代码
│   ├── data/              # 数据获取模块
│   ├── factors/           # 因子计算模块
│   ├── backtest/          # 回测引擎
│   ├── performance/       # 绩效分析
│   ├── strategies/        # 交易策略
│   └── utils/             # 工具函数
├── config/                # 配置文件
├── tests/                 # 测试文件
├── notebooks/             # Jupyter 笔记本
├── docs/                  # 文档
├── logs/                  # 日志文件
└── requirements.txt       # 依赖包
```

## 🔧 常用命令

```bash
# 安装新的包
pip install package_name

# 更新 requirements.txt
pip freeze > requirements.txt

# 退出虚拟环境
deactivate
```

## 📚 下一步

1. **阅读文档**: 查看 `06-量化交易系统开发需求详细说明书.md`
2. **开始编码**: 在 `src/` 目录下开始开发
3. **编写测试**: 在 `tests/` 目录下编写单元测试
4. **使用笔记本**: 在 `notebooks/` 目录下进行数据探索
5. **查看兼容性**: 阅读 `PYTHON_313_COMPATIBILITY.md` 了解 Python 3.13 兼容性

## ⚠️ 注意事项

- 某些量化库（如 TA-Lib、QuantLib）暂时不支持 Python 3.13
- 使用 `pandas-ta` 作为技术分析的替代方案
- 定期检查库的更新以获得 Python 3.13 支持

## 🆘 获取帮助

- 运行 `python test_environment.py` 检查环境状态
- 查看 `README.md` 获取详细信息
- 检查 `PYTHON_313_COMPATIBILITY.md` 了解兼容性问题

---

**开发环境配置完成！祝您量化交易开发愉快！** 🚀📈