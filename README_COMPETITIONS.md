# 量化比赛参与指南

## 🏆 支持的顶级比赛

本项目已完全准备好参加以下四个顶级量化比赛：

### 1. Jane Street Market Prediction
- **类型**: 市场预测挑战
- **数据**: 匿名化金融特征（130+维度）
- **目标**: 预测二分类结果（action = 0/1）
- **评估**: 加权对数损失（Weighted Log Loss）
- **特点**: 强时间依赖性，需要防止数据泄露

### 2. Optiver - Trading at the Close
- **类型**: 收盘价交易优化
- **数据**: 订单簿数据（bid/ask价格和数量）
- **目标**: 预测收盘前60秒的价格变动
- **评估**: 均方根误差（RMSE）
- **特点**: 实时性要求高，需要快速预测

### 3. Citadel Terminal AI Competition
- **类型**: AI交易竞赛
- **环境**: 模拟交易终端
- **目标**: 最大化交易收益
- **评估**: 夏普比率、总收益、最大回撤
- **特点**: 实时决策，多策略组合

### 4. CME Group Crypto Classic
- **类型**: 加密货币交易竞赛
- **标的**: 比特币和以太坊期货及期权
- **资金**: 10万美元虚拟账户
- **目标**: 最大化账户余额
- **特点**: 期货期权交易，风险管理重要

## 🚀 快速开始

### 1. 环境配置

```bash
# 设置所有比赛环境
python scripts/competition_setup.py all

# 或者设置特定比赛
python scripts/competition_setup.py setup --competition jane_street
python scripts/competition_setup.py setup --competition optiver
python scripts/competition_setup.py setup --competition citadel
python scripts/competition_setup.py setup --competition cme_crypto
```

### 2. 查看可用比赛

```bash
python scripts/competition_setup.py list
```

### 3. 使用主控制脚本

```bash
# 创建主控制脚本
python scripts/competition_setup.py master

# 使用主控制脚本
python scripts/competition_manager.py setup --competition jane_street
python scripts/competition_manager.py train --competition jane_street
python scripts/competition_manager.py predict --competition jane_street
```

## 📁 项目结构

设置完成后，项目将包含以下结构：

```
quant-bot/
├── competitions/                    # 比赛专用目录
│   ├── jane_street/                # Jane Street比赛
│   │   ├── data/                   # 数据目录
│   │   ├── models/                 # 模型目录
│   │   ├── features/               # 特征目录
│   │   ├── submissions/            # 提交目录
│   │   ├── logs/                   # 日志目录
│   │   ├── train.py               # 训练脚本
│   │   ├── predict.py             # 预测脚本
│   │   └── .env                   # 环境配置
│   ├── optiver/                   # Optiver比赛
│   ├── citadel/                   # Citadel比赛
│   └── cme_crypto/                # CME比赛
├── configs/competitions/           # 比赛配置文件
├── docs/                          # 文档目录
│   ├── COMPETITION_STRATEGIES.md  # 参赛策略
│   ├── PREMIUM_COMPETITIONS_ANALYSIS.md  # 比赛分析
│   └── KAGGLE_COMPETITION_ANALYSIS.md    # Kaggle分析
├── src/ml/                        # 机器学习模块
│   ├── crypto_tools.py           # 加密货币工具
│   ├── terminal_ai_tools.py      # Terminal AI工具
│   ├── advanced_feature_engineering.py
│   ├── model_ensemble.py
│   ├── model_validation.py
│   └── kaggle_tools.py
└── examples/                      # 使用示例
    ├── premium_competitions_example.py
    └── kaggle_competition_example.py
```

## 🛠️ 核心功能

### 1. 数据处理管道
- **多格式支持**: 表格数据、订单簿数据、时间序列数据
- **实时处理**: 支持实时数据流处理
- **数据清洗**: 缺失值处理、异常值检测
- **特征缓存**: 提高处理效率

### 2. 高级特征工程
```python
from src.ml.advanced_feature_engineering import AdvancedFeatureEngineer

# 创建特征工程器
feature_engineer = AdvancedFeatureEngineer()

# 时间序列特征
features = feature_engineer.create_time_series_features(data, 
                                                       lag_periods=[1,2,3,5,10],
                                                       rolling_windows=[5,10,20])

# 交互特征
interaction_features = feature_engineer.create_interaction_features(data, 
                                                                   max_combinations=2)

# 目标编码
target_encoded = feature_engineer.create_target_encoding(data, 
                                                        categorical_cols=['cat1', 'cat2'],
                                                        target_col='target')
```

### 3. 模型集成
```python
from src.ml.model_ensemble import ModelEnsemble

# 创建集成模型
ensemble = ModelEnsemble()

# 添加基础模型
ensemble.add_model('lightgbm', lgb_params)
ensemble.add_model('xgboost', xgb_params)
ensemble.add_model('catboost', cat_params)

# 训练集成模型
ensemble.fit(X_train, y_train, X_val, y_val)

# 预测
predictions = ensemble.predict(X_test)
```

### 4. 时间序列验证
```python
from src.ml.model_validation import TimeSeriesValidator

# 创建验证器
validator = TimeSeriesValidator(n_splits=5, test_size=0.2)

# 交叉验证
cv_scores = validator.cross_validate(model, X, y, 
                                   scoring='neg_log_loss',
                                   time_column='date_id')

# 验证报告
report = validator.generate_validation_report(cv_scores)
```

### 5. 比赛专用工具

#### Jane Street工具
```python
from src.ml.kaggle_tools import KaggleCompetitionTools

tools = KaggleCompetitionTools()

# 创建提交文件
submission = tools.create_submission(predictions, sample_submission)

# 特征重要性分析
importance = tools.analyze_feature_importance(model, feature_names)
```

#### Optiver工具
```python
# 订单簿特征工程
features = create_orderbook_features(data)

# 实时预测优化
optimized_model = optimize_for_inference(model)
```

#### Citadel工具
```python
from src.ml.terminal_ai_tools import create_terminal_ai_system

# 创建AI交易系统
system = create_terminal_ai_system(
    strategies=['momentum', 'mean_reversion', 'arbitrage'],
    risk_params={'max_position': 0.1, 'stop_loss': 0.05}
)

# 运行模拟
results = run_terminal_ai_simulation(system, data)
```

#### CME工具
```python
from src.ml.crypto_tools import create_crypto_pipeline

# 创建加密货币交易管道
pipeline = create_crypto_pipeline(
    strategies=['basis_trading', 'volatility_trading'],
    instruments=['BTC_futures', 'ETH_futures']
)

# 模拟交易
results = simulate_crypto_trading(pipeline, data, initial_capital=100000)
```

## 📊 使用示例

### 完整比赛流程示例

```python
import pandas as pd
from src.data.data_loader import DataLoader
from src.ml.advanced_feature_engineering import AdvancedFeatureEngineer
from src.ml.model_ensemble import ModelEnsemble
from src.ml.model_validation import TimeSeriesValidator
from src.ml.kaggle_tools import KaggleCompetitionTools

def jane_street_pipeline():
    """Jane Street比赛完整流程"""
    
    # 1. 数据加载
    data_loader = DataLoader()
    train_data = data_loader.load_csv("competitions/jane_street/data/train.csv")
    
    # 2. 特征工程
    feature_engineer = AdvancedFeatureEngineer()
    
    # 基础特征
    features = feature_engineer.create_time_series_features(
        train_data, 
        lag_periods=[1, 2, 3, 5, 10],
        rolling_windows=[5, 10, 20, 50]
    )
    
    # 交互特征
    interaction_features = feature_engineer.create_interaction_features(
        features, max_combinations=2
    )
    
    # 目标编码
    target_encoded = feature_engineer.create_target_encoding(
        interaction_features, 
        categorical_cols=['feature_0', 'feature_1'],
        target_col='action'
    )
    
    # 3. 数据分割
    X = target_encoded.drop(['action', 'date_id', 'ts_id'], axis=1)
    y = target_encoded['action']
    
    # 4. 模型训练
    ensemble = ModelEnsemble()
    ensemble.add_model('lightgbm', {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    })
    
    ensemble.add_model('xgboost', {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8
    })
    
    # 5. 时间序列验证
    validator = TimeSeriesValidator(n_splits=5, test_size=0.2)
    cv_scores = validator.cross_validate(
        ensemble, X, y,
        scoring='neg_log_loss',
        time_column='date_id'
    )
    
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 6. 最终训练
    ensemble.fit(X, y)
    
    # 7. 预测和提交
    test_data = data_loader.load_csv("competitions/jane_street/data/test.csv")
    test_features = feature_engineer.transform(test_data)
    predictions = ensemble.predict(test_features)
    
    # 创建提交文件
    tools = KaggleCompetitionTools()
    submission = tools.create_submission(predictions, test_data[['row_id']])
    submission.to_csv("competitions/jane_street/submissions/submission.csv", index=False)
    
    return ensemble, cv_scores

if __name__ == "__main__":
    model, scores = jane_street_pipeline()
    print("✅ Jane Street pipeline completed!")
```

## 📈 性能优化

### 1. 内存优化
```python
# 使用数据类型优化
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# 分块处理大数据
def process_in_chunks(data, chunk_size=10000):
    for chunk in pd.read_csv(data, chunksize=chunk_size):
        yield process_chunk(chunk)
```

### 2. 计算优化
```python
# 并行特征工程
from joblib import Parallel, delayed

def parallel_feature_engineering(data, n_jobs=-1):
    features = Parallel(n_jobs=n_jobs)(
        delayed(create_feature)(data, col) 
        for col in data.columns
    )
    return pd.concat(features, axis=1)

# GPU加速（如果可用）
import cudf  # RAPIDS

def gpu_feature_engineering(data):
    gpu_data = cudf.from_pandas(data)
    # GPU加速的特征工程
    return gpu_data.to_pandas()
```

### 3. 实时优化
```python
# 模型推理优化
import onnx
import onnxruntime as ort

def optimize_model_for_inference(model):
    # 转换为ONNX格式
    onnx_model = convert_to_onnx(model)
    
    # 创建推理会话
    session = ort.InferenceSession(onnx_model)
    
    return session

# 特征缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_feature_calculation(data_hash):
    return expensive_feature_calculation(data_hash)
```

## 🔧 配置管理

### 比赛配置文件示例

```json
{
  "name": "Jane Street Market Prediction",
  "type": "classification",
  "data_format": "tabular",
  "target_column": "action",
  "evaluation_metric": "weighted_log_loss",
  "time_series": true,
  "features": {
    "numerical_features": 130,
    "categorical_features": 0,
    "time_features": ["date_id", "ts_id"]
  },
  "model_config": {
    "primary_models": ["lightgbm", "xgboost", "catboost"],
    "ensemble_method": "stacking",
    "cv_strategy": "time_series_split",
    "cv_folds": 5
  },
  "feature_engineering": {
    "lag_features": [1, 2, 3, 5, 10],
    "rolling_windows": [5, 10, 20, 50],
    "interaction_features": true,
    "target_encoding": true
  },
  "optimization": {
    "objective": "minimize_log_loss",
    "early_stopping": 100,
    "max_iterations": 10000
  }
}
```

## 📋 检查清单

### 比赛前准备
- [ ] 环境配置完成
- [ ] 数据下载和预处理
- [ ] 基础特征工程测试
- [ ] 模型训练管道验证
- [ ] 交叉验证框架测试
- [ ] 提交格式确认

### 比赛期间
- [ ] 数据探索和分析
- [ ] 特征工程优化
- [ ] 模型调优和集成
- [ ] 验证策略执行
- [ ] 性能监控
- [ ] 定期提交

### 比赛后
- [ ] 结果分析
- [ ] 代码整理
- [ ] 经验总结
- [ ] 模型保存

## 🆘 常见问题

### Q: 如何处理内存不足问题？
A: 使用数据类型优化、分块处理、特征选择等方法。参考性能优化章节。

### Q: 如何避免过拟合？
A: 使用时间序列交叉验证、正则化、早停等技术。

### Q: 如何提高模型性能？
A: 重点关注特征工程、模型集成、超参数优化。

### Q: 如何处理实时性要求？
A: 使用模型压缩、特征缓存、推理优化等技术。

## 📚 参考资源

- [详细比赛分析](docs/PREMIUM_COMPETITIONS_ANALYSIS.md)
- [参赛策略指南](docs/COMPETITION_STRATEGIES.md)
- [Kaggle比赛分析](docs/KAGGLE_COMPETITION_ANALYSIS.md)
- [完整使用示例](examples/premium_competitions_example.py)

## 🎯 下一步

1. **选择目标比赛**: 根据兴趣和专长选择1-2个比赛重点参与
2. **环境配置**: 运行配置脚本设置比赛环境
3. **数据准备**: 下载比赛数据并进行初步分析
4. **基线模型**: 建立简单的基线模型
5. **迭代优化**: 持续改进特征工程和模型性能
6. **提交参赛**: 定期提交结果并监控排名

---

🚀 **现在就开始你的量化比赛之旅吧！** 

项目已经为你准备好了所有必要的工具和框架，只需要根据具体比赛调整策略和参数即可。祝你在比赛中取得优异成绩！