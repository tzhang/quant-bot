# ❓ 常见问题与故障排除

本文档收集了使用量化交易系统时的常见问题及解决方案，帮助您快速解决遇到的困难。

## 📋 目录

1. [环境配置问题](#环境配置问题)
2. [数据获取问题](#数据获取问题)
3. [因子计算问题](#因子计算问题)
4. [图表生成问题](#图表生成问题)
5. [性能优化问题](#性能优化问题)
6. [系统错误处理](#系统错误处理)

---

## 🔧 环境配置问题

### Q1: 安装依赖包时出现错误

**问题描述**: 运行 `pip install -r requirements.txt` 时出现安装失败

**常见错误**:
```bash
ERROR: Could not find a version that satisfies the requirement
ERROR: No matching distribution found
```

**解决方案**:
```bash
# 方案1: 升级pip
pip install --upgrade pip

# 方案2: 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 方案3: 逐个安装核心包
pip install pandas numpy matplotlib yfinance requests

# 方案4: 使用conda安装
conda install pandas numpy matplotlib
pip install yfinance requests
```

**预防措施**:
- 使用Python 3.12版本（强制要求）
- 定期更新pip和setuptools
- 在虚拟环境中安装依赖

### Q2: 导入模块时出现ModuleNotFoundError

**问题描述**: 
```python
ModuleNotFoundError: No module named 'src.factors.engine'
```

**解决方案**:
```python
# 方案1: 添加项目路径
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 方案2: 设置PYTHONPATH环境变量
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"

# 方案3: 在项目根目录运行
cd /path/to/my-quant
python examples/data_tutorial.py
```

### Q3: 环境测试脚本报错

**问题描述**: 运行 `python test_environment.py` 时出现各种错误

**解决步骤**:
```bash
# 1. 检查Python版本
python --version  # 应该是3.8+

# 2. 检查当前目录
pwd  # 应该在项目根目录

# 3. 检查文件结构
ls -la  # 应该看到src/, examples/, docs/等目录

# 4. 重新安装依赖
pip install -r requirements.txt --force-reinstall

# 5. 清理缓存
pip cache purge
```

---

## 📊 数据获取问题

### Q4: yfinance获取数据失败

**问题描述**: 
```python
Exception: No data found, symbol may be delisted
```

**解决方案**:
```python
# 方案1: 检查股票代码格式
# 正确格式
symbols = ['AAPL', 'GOOGL', 'MSFT']  # 美股
symbols = ['000001.SZ', '000002.SZ']  # A股需要后缀

# 方案2: 添加重试机制
import time
import yfinance as yf

def get_data_with_retry(symbol, max_retries=3):
    for i in range(max_retries):
        try:
            data = yf.download(symbol, period='1y')
            if not data.empty:
                return data
        except Exception as e:
            print(f"尝试 {i+1}/{max_retries} 失败: {e}")
            time.sleep(2)
    return None

# 方案3: 使用更短的时间周期
data = yf.download('AAPL', period='1m')  # 改为1个月
```

### Q5: 网络连接超时

**问题描述**: 
```python
requests.exceptions.ConnectTimeout: HTTPSConnectionPool
```

**解决方案**:
```python
# 方案1: 增加超时时间
import yfinance as yf
yf.pdr_override()
data = yf.download('AAPL', period='1y', timeout=30)

# 方案2: 使用代理
import os
os.environ['HTTP_PROXY'] = 'http://your-proxy:port'
os.environ['HTTPS_PROXY'] = 'https://your-proxy:port'

# 方案3: 分批获取数据
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
batch_size = 2
for i in range(0, len(symbols), batch_size):
    batch = symbols[i:i+batch_size]
    data = yf.download(batch, period='1y')
    time.sleep(1)  # 避免请求过于频繁
```

### Q6: 数据缓存问题

**问题描述**: 缓存文件损坏或无法读取

**解决方案**:
```bash
# 方案1: 清理缓存目录
rm -rf data_cache/
mkdir data_cache

# 方案2: 检查磁盘空间
df -h

# 方案3: 修复权限问题
chmod -R 755 data_cache/
```

```python
# 方案4: 在代码中处理缓存错误
try:
    data = engine.get_data(symbols, period='1y')
except Exception as e:
    print(f"缓存读取失败: {e}")
    # 清理缓存并重新获取
    import shutil
    shutil.rmtree('data_cache', ignore_errors=True)
    data = engine.get_data(symbols, period='1y')
```

---

## 🧮 因子计算问题

### Q7: 因子计算结果为NaN

**问题描述**: 计算的因子值全部为NaN或空值

**常见原因及解决方案**:
```python
# 原因1: 数据不足
def check_data_length(data, min_length=20):
    if len(data) < min_length:
        print(f"数据长度不足: {len(data)} < {min_length}")
        return False
    return True

# 原因2: 数据包含NaN值
def clean_data(data):
    # 检查NaN值
    nan_count = data.isnull().sum().sum()
    if nan_count > 0:
        print(f"发现 {nan_count} 个NaN值")
        # 前向填充
        data = data.fillna(method='ffill')
        # 后向填充
        data = data.fillna(method='bfill')
    return data

# 原因3: 除零错误
def safe_divide(numerator, denominator):
    return np.where(denominator != 0, numerator / denominator, np.nan)

# 使用示例
momentum = safe_divide(
    data['close'].iloc[-1] - data['close'].iloc[-21],
    data['close'].iloc[-21]
)
```

### Q8: 因子评估IC值异常

**问题描述**: IC值过高(>0.5)或计算失败

**解决方案**:
```python
# 检查数据质量
def validate_factor_data(factor_values, returns):
    # 1. 检查数据长度
    if len(factor_values) != len(returns):
        print("因子值和收益率长度不匹配")
        return False
    
    # 2. 检查NaN值
    factor_nan = pd.isna(factor_values).sum()
    returns_nan = pd.isna(returns).sum()
    if factor_nan > 0 or returns_nan > 0:
        print(f"因子NaN: {factor_nan}, 收益NaN: {returns_nan}")
        return False
    
    # 3. 检查数据范围
    if factor_values.std() == 0:
        print("因子值无变化")
        return False
    
    return True

# 计算IC时的错误处理
def calculate_ic_safe(factor_values, returns):
    try:
        # 移除NaN值
        valid_mask = ~(pd.isna(factor_values) | pd.isna(returns))
        factor_clean = factor_values[valid_mask]
        returns_clean = returns[valid_mask]
        
        if len(factor_clean) < 10:  # 至少需要10个有效样本
            return np.nan
        
        ic = factor_clean.corr(returns_clean)
        
        # 检查IC值合理性
        if abs(ic) > 0.8:
            print(f"警告: IC值异常高 {ic:.4f}")
        
        return ic
    except Exception as e:
        print(f"IC计算错误: {e}")
        return np.nan
```

### Q9: 分层测试结果不合理

**问题描述**: 分层测试没有显示单调性或收益差异很小

**解决方案**:
```python
def analyze_layered_test(factor_values, returns, n_layers=5):
    # 1. 检查样本数量
    if len(factor_values) < n_layers * 2:
        print(f"样本数量不足进行{n_layers}层测试")
        return None
    
    # 2. 因子标准化
    factor_standardized = (factor_values - factor_values.mean()) / factor_values.std()
    
    # 3. 分层
    df = pd.DataFrame({
        'factor': factor_standardized,
        'returns': returns
    }).dropna()
    
    df['layer'] = pd.qcut(df['factor'], n_layers, labels=False)
    
    # 4. 计算各层收益
    layer_returns = df.groupby('layer')['returns'].mean()
    
    # 5. 检查单调性
    monotonic = all(layer_returns.iloc[i] <= layer_returns.iloc[i+1] 
                   for i in range(len(layer_returns)-1))
    
    print(f"单调性检查: {'通过' if monotonic else '未通过'}")
    print("各层收益:")
    for i, ret in enumerate(layer_returns):
        print(f"  第{i+1}层: {ret:.4f}")
    
    return layer_returns
```

---

## 📊 图表生成问题

### Q10: matplotlib中文显示乱码

**问题描述**: 图表中的中文显示为方框或乱码

**解决方案**:
```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# 方案1: 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 方案2: 使用系统字体
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/System/Library/Fonts/Arial Unicode.ttf')
plt.title('中文标题', fontproperties=font)

# 方案3: 安装中文字体包
# pip install matplotlib -U
# 下载并安装SimHei字体

# 方案4: 检查可用字体
from matplotlib.font_manager import fontManager
fonts = [f.name for f in fontManager.ttflist if 'Chinese' in f.name or 'SimHei' in f.name]
print("可用中文字体:", fonts)
```

### Q11: 图表保存失败

**问题描述**: 
```python
PermissionError: [Errno 13] Permission denied: 'examples/chart.png'
```

**解决方案**:
```python
import os
from pathlib import Path

# 方案1: 检查目录权限
output_dir = Path('examples')
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

# 方案2: 使用绝对路径
import tempfile
temp_dir = tempfile.gettempdir()
output_path = os.path.join(temp_dir, 'chart.png')

# 方案3: 错误处理
try:
    plt.savefig('examples/chart.png', dpi=300, bbox_inches='tight')
    print("图表保存成功")
except PermissionError:
    # 保存到用户目录
    home_dir = Path.home()
    backup_path = home_dir / 'chart.png'
    plt.savefig(backup_path, dpi=300, bbox_inches='tight')
    print(f"图表保存到: {backup_path}")
```

### Q12: 图表显示不完整

**问题描述**: 图表标签被截断或重叠

**解决方案**:
```python
# 方案1: 调整图表布局
plt.figure(figsize=(12, 8))
plt.tight_layout()

# 方案2: 手动调整边距
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# 方案3: 旋转标签
plt.xticks(rotation=45)
plt.ylabel('收益率', rotation=90)

# 方案4: 使用bbox_inches
plt.savefig('chart.png', bbox_inches='tight', pad_inches=0.2)

# 方案5: 设置字体大小
plt.rcParams.update({'font.size': 10})
```

---

## ⚡ 性能优化问题

### Q13: 程序运行速度慢

**问题描述**: 数据处理或因子计算耗时过长

**优化方案**:
```python
# 方案1: 使用向量化操作
# 慢速循环方式
results = []
for i in range(len(data)):
    result = data.iloc[i]['close'] / data.iloc[i]['open'] - 1
    results.append(result)

# 快速向量化方式
results = data['close'] / data['open'] - 1

# 方案2: 使用numba加速
from numba import jit

@jit(nopython=True)
def fast_calculation(prices):
    returns = np.zeros(len(prices)-1)
    for i in range(1, len(prices)):
        returns[i-1] = prices[i] / prices[i-1] - 1
    return returns

# 方案3: 并行处理
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def process_symbol(symbol):
    # 处理单个股票的逻辑
    return result

# 并行处理多个股票
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    results = list(executor.map(process_symbol, symbols))
```

### Q14: 内存使用过多

**问题描述**: 程序占用内存过大，可能导致系统卡顿

**解决方案**:
```python
# 方案1: 分批处理
def process_in_batches(symbols, batch_size=10):
    results = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        batch_result = process_batch(batch)
        results.extend(batch_result)
        
        # 清理内存
        import gc
        gc.collect()
    
    return results

# 方案2: 使用生成器
def data_generator(symbols):
    for symbol in symbols:
        data = get_data(symbol)
        yield symbol, data

# 方案3: 优化数据类型
# 使用更小的数据类型
data['close'] = data['close'].astype('float32')  # 而不是float64
data['volume'] = data['volume'].astype('int32')   # 而不是int64

# 方案4: 及时删除不需要的变量
del large_dataframe
import gc
gc.collect()
```

### Q15: 缓存文件过大

**问题描述**: data_cache目录占用磁盘空间过多

**解决方案**:
```python
# 方案1: 定期清理缓存
import os
import time
from pathlib import Path

def clean_old_cache(cache_dir='data_cache', days=7):
    """清理超过指定天数的缓存文件"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return
    
    cutoff_time = time.time() - (days * 24 * 3600)
    
    for file_path in cache_path.glob('*.meta'):
        if file_path.stat().st_mtime < cutoff_time:
            file_path.unlink()
            print(f"删除过期缓存: {file_path}")

# 方案2: 压缩缓存文件
import pickle
import gzip

def save_compressed_cache(data, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_compressed_cache(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

# 方案3: 设置缓存大小限制
def manage_cache_size(cache_dir='data_cache', max_size_mb=1000):
    """限制缓存目录大小"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return
    
    # 计算总大小
    total_size = sum(f.stat().st_size for f in cache_path.glob('*'))
    total_size_mb = total_size / (1024 * 1024)
    
    if total_size_mb > max_size_mb:
        # 删除最旧的文件
        files = sorted(cache_path.glob('*'), key=lambda x: x.stat().st_mtime)
        for file_path in files:
            file_path.unlink()
            total_size -= file_path.stat().st_size
            if total_size / (1024 * 1024) <= max_size_mb * 0.8:
                break
```

---

## 🚨 系统错误处理

### Q16: 程序意外崩溃

**问题描述**: 程序运行过程中突然停止，没有明确错误信息

**调试方案**:
```python
import logging
import traceback

# 方案1: 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# 方案2: 全局异常处理
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logging.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# 方案3: 使用try-except包装主要逻辑
def safe_main():
    try:
        # 主要程序逻辑
        main_logic()
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        logging.error(traceback.format_exc())
        return False
    return True

if __name__ == "__main__":
    success = safe_main()
    if not success:
        print("程序执行失败，请查看debug.log获取详细信息")
```

### Q17: 数据类型错误

**问题描述**: 
```python
TypeError: unsupported operand type(s) for /: 'str' and 'int'
```

**解决方案**:
```python
# 方案1: 数据类型检查和转换
def ensure_numeric(data, column):
    """确保列为数值类型"""
    if data[column].dtype == 'object':
        # 尝试转换为数值
        data[column] = pd.to_numeric(data[column], errors='coerce')
    return data

# 方案2: 安全的数学运算
def safe_math_operation(a, b, operation='divide'):
    """安全的数学运算"""
    try:
        a = float(a) if not pd.isna(a) else np.nan
        b = float(b) if not pd.isna(b) else np.nan
        
        if operation == 'divide':
            return a / b if b != 0 else np.nan
        elif operation == 'multiply':
            return a * b
        # 其他操作...
    except (ValueError, TypeError):
        return np.nan

# 方案3: 数据验证函数
def validate_dataframe(df, required_columns):
    """验证DataFrame的结构和数据类型"""
    errors = []
    
    # 检查必需列
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"缺少列: {missing_cols}")
    
    # 检查数据类型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"列 {col} 不是数值类型")
    
    if errors:
        raise ValueError("数据验证失败: " + "; ".join(errors))
    
    return True
```

### Q18: 文件路径问题

**问题描述**: 
```python
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**:
```python
import os
from pathlib import Path

# 方案1: 使用绝对路径
def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent

def get_safe_path(*path_parts):
    """构建安全的文件路径"""
    root = get_project_root()
    return root / Path(*path_parts)

# 使用示例
config_path = get_safe_path('config', 'settings.json')
data_path = get_safe_path('data', 'stock_data.csv')

# 方案2: 检查文件存在性
def safe_file_operation(file_path, operation='read'):
    """安全的文件操作"""
    path = Path(file_path)
    
    if operation == 'read':
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"路径不是文件: {path}")
    
    elif operation == 'write':
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path

# 方案3: 跨平台路径处理
def normalize_path(path_str):
    """标准化路径，处理不同操作系统的差异"""
    return str(Path(path_str).resolve())
```

---

## 🆘 获取帮助

### 联系方式
- **GitHub Issues**: 在项目仓库提交问题
- **邮件支持**: support@example.com
- **技术文档**: 查看 `docs/` 目录下的详细文档

### 提交问题时请包含
1. **错误信息**: 完整的错误堆栈信息
2. **环境信息**: Python版本、操作系统、依赖包版本
3. **重现步骤**: 详细的操作步骤
4. **相关代码**: 出错的代码片段
5. **数据样本**: 如果涉及数据问题，提供样本数据

### 自助排查清单
- [ ] 检查Python版本是否为3.8+
- [ ] 确认所有依赖包已正确安装
- [ ] 验证项目目录结构完整
- [ ] 检查网络连接是否正常
- [ ] 查看是否有足够的磁盘空间
- [ ] 确认文件权限设置正确
- [ ] 查看日志文件获取详细错误信息

---

## 📚 相关文档

- [初学者使用指南](BEGINNER_GUIDE.md)
- [图表解读指南](CHART_INTERPRETATION_GUIDE.md)
- [API参考文档](API_REFERENCE.md)
- [开发者指南](DEVELOPER_GUIDE.md)

---

**记住**: 遇到问题时不要慌张，大多数问题都有解决方案。仔细阅读错误信息，按照本文档的指导逐步排查，通常能够快速解决问题。如果问题依然存在，请不要犹豫寻求帮助！