# Python 3.13 兼容性说明

## 概述

本项目目前使用 Python 3.13，部分量化金融库尚未完全支持此版本。以下是兼容性状态和解决方案。

## 已成功安装的核心库

✅ **数据处理**
- pandas 2.3.3
- numpy 2.2.6
- scipy 1.16.2

✅ **数据获取**
- yfinance 0.2.66
- beautifulsoup4
- lxml

✅ **机器学习**
- scikit-learn 1.7.2
- statsmodels 0.14.5

✅ **技术分析**
- pandas-ta 0.4.71b0 (beta版本，但可用)

✅ **可视化**
- matplotlib 3.10.6
- seaborn 0.13.2
- plotly 6.3.0

✅ **Web框架**
- FastAPI
- uvicorn
- streamlit 1.50.0

✅ **开发工具**
- pytest 8.4.2
- black
- flake8
- jupyter
- ipython

## 暂时不兼容的库

❌ **技术分析**
- `TA-Lib` - 需要 C 编译器，在 Python 3.13 上编译困难
  - **替代方案**: 使用 `pandas-ta` 或自实现技术指标

❌ **数值计算优化**
- `numba` - 可能不完全支持 Python 3.13
  - **替代方案**: 使用 numpy 的向量化操作

❌ **量化金融专用库**
- `quantlib-python` - 等待官方 Python 3.13 支持
- `zipline-reloaded` - 等待官方 Python 3.13 支持
- `empyrical` - 等待官方 Python 3.13 支持
- `pyfolio` - 等待官方 Python 3.13 支持
- `riskfolio-lib` - 等待官方 Python 3.13 支持

## 解决方案和建议

### 1. 技术指标计算

使用 `pandas-ta` 替代 `TA-Lib`：

```python
import pandas_ta as ta

# 移动平均线
df['SMA_20'] = ta.sma(df['Close'], length=20)

# RSI
df['RSI'] = ta.rsi(df['Close'], length=14)

# MACD
macd = ta.macd(df['Close'])
df = df.join(macd)
```

### 2. 性能优化

使用 numpy 向量化操作替代 numba：

```python
import numpy as np

# 向量化计算而不是循环
returns = np.diff(np.log(prices))
rolling_std = pd.Series(returns).rolling(window=20).std()
```

### 3. 量化分析

自实现常用的量化指标：

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """计算夏普比率"""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(prices):
    """计算最大回撤"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()
```

## 升级路径

当这些库支持 Python 3.13 时，可以通过以下步骤升级：

1. 取消 `requirements.txt` 中相关库的注释
2. 运行 `pip install -r requirements.txt`
3. 更新相关代码以使用原生库

## 监控更新

定期检查以下库的 Python 3.13 支持状态：
- [TA-Lib PyPI](https://pypi.org/project/TA-Lib/)
- [QuantLib PyPI](https://pypi.org/project/QuantLib/)
- [Zipline-Reloaded PyPI](https://pypi.org/project/zipline-reloaded/)

## 当前状态

✅ **开发环境已就绪**
- 所有核心功能库已安装
- 可以开始数据获取、处理、分析和可视化
- Web 应用框架已就绪
- 测试和代码质量工具已配置

**最后更新**: 2024年12月