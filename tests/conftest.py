import sys
from pathlib import Path

# 将项目根目录加入sys.path，确保可以导入顶层的 `src` 包
ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT_DIR)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)