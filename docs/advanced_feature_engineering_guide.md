# 高级特征工程指南

## 概述

本指南介绍量化交易系统中的高级特征工程功能，帮助您构建更强大的预测模型和交易策略。

## 目录

1. [高阶统计特征](#高阶统计特征)
2. [频域分析特征](#频域分析特征)
3. [图论网络特征](#图论网络特征)
4. [宏观经济特征](#宏观经济特征)
5. [非线性特征](#非线性特征)
6. [特征选择](#特征选择)
7. [特征验证](#特征验证)
8. [完整工作流程](#完整工作流程)
9. [最佳实践](#最佳实践)

## 高阶统计特征

### 功能概述

高阶统计特征提供比均值和方差更深层的数据分布信息，包括：

- **矩特征**: 偏度、峰度等高阶矩
- **分布检验**: 正态性检验、分布拟合度
- **熵特征**: 信息熵、相对熵

### 使用示例

```python
from src.ml.advanced_feature_engineering import HighOrderStatisticalFeatures

# 初始化
stat_features = HighOrderStatisticalFeatures()

# 添加矩特征
df = stat_features.add_moment_features(
    df, 
    features=['returns', 'volume'], 
    windows=[5, 10, 20]
)

# 添加分布检验特征
df = stat_features.add_distribution_test_features(
    df, 
    features=['returns'], 
    windows=[20, 60]
)

# 添加熵特征
df = stat_features.add_entropy_features(
    df, 
    features=['returns', 'volume'], 
    windows=[10, 20]
)
```

### 应用场景

- **风险管理**: 通过峰度检测极端风险
- **市场状态识别**: 通过偏度判断市场偏向
- **异常检测**: 通过熵变化识别市场异常

## 频域分析特征

### 功能概述

频域分析将时间序列转换到频率域，捕捉周期性和频率特性：

- **FFT特征**: 快速傅里叶变换提取主要频率成分
- **小波变换**: 时频分析，同时保留时间和频率信息

### 使用示例

```python
from src.ml.advanced_feature_engineering import FrequencyDomainFeatures

# 初始化
freq_features = FrequencyDomainFeatures()

# 添加FFT特征
df = freq_features.add_fft_features(
    df, 
    features=['returns', 'volume'], 
    window=60
)

# 添加小波变换特征
df = freq_features.add_wavelet_features(
    df, 
    features=['returns'], 
    window=60
)
```

### 应用场景

- **周期性检测**: 识别市场的周期性模式
- **噪声过滤**: 分离信号和噪声
- **趋势分析**: 多尺度趋势分解

## 图论网络特征

### 功能概述

分析资产间的关联性和网络结构：

- **相关性网络**: 基于相关性构建网络图
- **网络指标**: 度中心性、介数中心性等
- **社区检测**: 识别资产群组

### 使用示例

```python
from src.ml.advanced_feature_engineering import GraphNetworkFeatures

# 初始化
graph_features = GraphNetworkFeatures()

# 添加相关性网络特征
df = graph_features.add_correlation_network_features(
    df, 
    asset_columns=['asset_1', 'asset_2', 'asset_3'], 
    window=60
)
```

### 应用场景

- **投资组合构建**: 基于网络结构优化配置
- **风险传染分析**: 识别系统性风险路径
- **配对交易**: 发现高相关性资产对

## 宏观经济特征

### 功能概述

整合宏观经济环境信息：

- **经济指标**: VIX、美元指数、国债收益率等
- **大宗商品**: 黄金、原油价格
- **衍生指标**: 变化率、移动平均、相对位置

### 使用示例

```python
from src.ml.advanced_feature_engineering import MacroEconomicFeatures

# 初始化
macro_features = MacroEconomicFeatures()

# 添加宏观经济特征
df = macro_features.add_macro_features(
    df, 
    lookback_days=252
)
```

### 应用场景

- **宏观择时**: 基于宏观环境调整策略
- **风险管理**: 宏观风险因子对冲
- **多资产配置**: 跨资产类别配置优化

## 非线性特征

### 功能概述

通过非线性方法发现隐藏模式：

- **流形学习**: PCA、ICA降维
- **聚类特征**: K-means聚类标签
- **非线性变换**: 核方法、神经网络嵌入

### 使用示例

```python
from src.ml.advanced_feature_engineering import NonlinearFeatures

# 初始化
nonlinear_features = NonlinearFeatures()

# 流形学习特征
X_manifold = nonlinear_features.add_manifold_features(
    X, 
    n_components=3
)

# 聚类特征
X_cluster = nonlinear_features.add_clustering_features(
    X, 
    n_clusters=5
)
```

### 应用场景

- **降维可视化**: 高维数据可视化
- **模式识别**: 发现非线性模式
- **状态识别**: 市场状态聚类

## 特征选择

### 功能概述

自动筛选最有价值的特征：

- **递归特征消除**: 基于模型重要性递归选择
- **稳定性选择**: 多次采样的稳定特征
- **信息增益**: 基于信息论的特征选择
- **相关性分析**: 去除冗余特征

### 使用示例

```python
from src.ml.feature_selection import FeatureSelectionPipeline

# 初始化
selector = FeatureSelectionPipeline()

# 执行特征选择
selected_features = selector.select_features(
    X, y, 
    n_features=20
)

# 获取选择摘要
summary = selector.get_selection_summary()
```

### 选择策略

1. **重要性排序**: 基于模型重要性
2. **稳定性筛选**: 多次验证的稳定特征
3. **冗余去除**: 相关性过高的特征去重
4. **共识选择**: 多种方法的共识结果

## 特征验证

### 功能概述

确保特征质量和防止信息泄露：

- **稳定性测试**: 时间稳定性验证
- **信息泄露检测**: 未来信息泄露检测
- **重要性分析**: 特征重要性评估
- **质量评分**: 综合质量评分

### 使用示例

```python
from src.ml.feature_validation import FeatureValidationPipeline

# 初始化
validator = FeatureValidationPipeline()

# 执行验证
results = validator.validate_features(df, features)

# 获取推荐
recommendations = validator.get_feature_recommendations()
```

### 验证指标

1. **稳定性评分**: 时间序列稳定性
2. **泄露风险**: 信息泄露风险评估
3. **预测能力**: 特征预测价值
4. **综合质量**: 整体质量评分

## 完整工作流程

### 1. 数据准备

```python
import pandas as pd
from datetime import datetime

# 加载数据
df = pd.read_csv('stock_data.csv')
df['date'] = pd.to_datetime(df['date'])
```

### 2. 基础特征工程

```python
from src.ml.feature_engineering import AdvancedFeatureEngineer

# 基础特征
engineer = AdvancedFeatureEngineer()
df = engineer.create_lag_features(df, ['close', 'volume'], lags=[1, 2, 3])
df = engineer.create_rolling_features(df, ['close'], windows=[5, 10, 20])
```

### 3. 高级特征工程

```python
from src.ml.advanced_feature_engineering import AdvancedFeatureEngineeringPipeline

# 高级特征管道
pipeline = AdvancedFeatureEngineeringPipeline()

config = {
    'high_order_stats': {
        'features': ['returns', 'volume'],
        'windows': [5, 10, 20]
    },
    'frequency_domain': {
        'features': ['returns'],
        'window': 60
    },
    'nonlinear': {
        'features': ['returns', 'volume', 'volatility'],
        'manifold_components': 3,
        'n_clusters': 5
    }
}

df_enhanced = pipeline.transform(df, config)
```

### 4. 特征选择

```python
from src.ml.feature_selection import FeatureSelectionPipeline

# 准备数据
feature_columns = [col for col in df_enhanced.columns if col not in ['date', 'target']]
X = df_enhanced[feature_columns].fillna(method='ffill')
y = df_enhanced['target'].fillna(method='ffill')

# 特征选择
selector = FeatureSelectionPipeline()
selected_features = selector.select_features(X, y, n_features=50)
```

### 5. 特征验证

```python
from src.ml.feature_validation import FeatureValidationPipeline

# 特征验证
validator = FeatureValidationPipeline()
validation_results = validator.validate_features(df_enhanced, selected_features)

# 获取高质量特征
recommendations = validator.get_feature_recommendations()
final_features = recommendations['accept'] + recommendations['conditional']
```

### 6. 模型训练

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# 准备最终数据
X_final = df_enhanced[final_features].fillna(method='ffill')
y_final = df_enhanced['target'].fillna(method='ffill')

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X_final):
    X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train, y_val = y_final.iloc[train_idx], y_final.iloc[val_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"验证集R²: {score:.4f}")
```

## 最佳实践

### 1. 特征工程原则

- **领域知识**: 结合金融领域知识设计特征
- **时间序列**: 注意时间序列的特殊性质
- **前瞻偏差**: 避免使用未来信息
- **稳定性**: 确保特征在不同时期的稳定性

### 2. 计算效率

- **批量处理**: 批量计算相似特征
- **缓存机制**: 缓存中间计算结果
- **并行计算**: 利用多核并行处理
- **内存管理**: 及时释放不需要的数据

### 3. 特征质量控制

- **缺失值处理**: 合理处理缺失数据
- **异常值检测**: 识别和处理异常值
- **标准化**: 统一特征量纲
- **去噪**: 过滤噪声特征

### 4. 模型适配

- **特征选择**: 根据模型类型选择特征
- **特征变换**: 适应模型假设的变换
- **交叉验证**: 时间序列交叉验证
- **过拟合防护**: 防止特征过拟合

### 5. 生产部署

- **实时计算**: 支持实时特征计算
- **版本管理**: 特征版本控制
- **监控告警**: 特征质量监控
- **回测验证**: 定期回测验证

## 性能优化建议

### 1. 数据存储

```python
# 使用高效的数据格式
df.to_parquet('features.parquet')  # 推荐
df.to_hdf('features.h5', key='data')  # 大数据量
```

### 2. 并行计算

```python
from multiprocessing import Pool
import numpy as np

def parallel_feature_calculation(data_chunk):
    # 特征计算逻辑
    return processed_chunk

# 并行处理
with Pool() as pool:
    results = pool.map(parallel_feature_calculation, data_chunks)
```

### 3. 内存优化

```python
# 使用适当的数据类型
df['feature'] = df['feature'].astype('float32')  # 减少内存使用

# 及时删除不需要的列
df.drop(['temp_column'], axis=1, inplace=True)
```

## 常见问题解答

### Q1: 如何选择合适的窗口大小？

**A**: 窗口大小应该根据数据频率和预测目标来选择：
- 日频数据：5-20天（短期），20-60天（中期），60-252天（长期）
- 分钟数据：根据交易时间调整，如60分钟、240分钟等
- 考虑市场周期性，如周（5天）、月（20天）、季（60天）

### Q2: 如何处理特征之间的多重共线性？

**A**: 多种方法可以处理：
1. 相关性分析：移除高相关特征（>0.9）
2. 主成分分析：降维去相关
3. 正则化：使用L1/L2正则化
4. 方差膨胀因子：VIF>10的特征需要处理

### Q3: 如何验证特征的有效性？

**A**: 建议的验证流程：
1. 单变量分析：IC、IR指标
2. 时间稳定性：滚动窗口验证
3. 信息泄露检测：未来信息检查
4. 模型验证：交叉验证性能

### Q4: 特征过多时如何处理？

**A**: 特征降维策略：
1. 统计筛选：方差、相关性筛选
2. 模型筛选：基于重要性筛选
3. 稳定性筛选：时间稳定性筛选
4. 业务筛选：结合业务逻辑筛选

## 参考资料

1. [量化交易特征工程实战](docs/feature_engineering_practice.md)
2. [时间序列特征工程指南](docs/time_series_features.md)
3. [机器学习特征选择方法](docs/feature_selection_methods.md)
4. [特征工程最佳实践](docs/feature_engineering_best_practices.md)

## 更新日志

- **v2.0.0** (2024-01-20): 新增高级特征工程模块
- **v1.5.0** (2023-12-15): 增强特征验证功能
- **v1.4.0** (2023-11-20): 添加特征选择管道
- **v1.3.0** (2023-10-15): 优化计算性能