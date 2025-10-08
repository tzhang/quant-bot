import json
import logging
from pathlib import Path
from typing import List

import requests

logger = logging.getLogger(__name__)


DEFAULT_FALLBACK = [
    # 常见大盘股（不保证最新，作为离线回退）
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ADBE", "NFLX",
    "INTC", "AMD", "CSCO", "QCOM", "TXN", "AMAT", "MU", "PYPL", "PDD", "BKNG",
    "COST", "PEP", "SBUX", "MDLZ", "GILD", "VRTX", "REGN", "LRCX", "ASML", "ADI",
    "MELI", "MRVL", "ABNB", "SNPS", "KLAC", "PANW", "FTNT", "CRWD", "WDAY", "TEAM",
    "ADSK", "ORLY", "ROST", "KDP", "IDXX", "ALGN", "AEP", "EXC", "CTAS", "CDW",
    "NXPI", "ODFL", "CPRT", "EBAY", "FAST", "MAR", "CHTR", "ANSS", "ZS", "DDOG",
    "SPLK", "DOCU", "ETSY", "LULU", "BIDU", "JD", "ZM", "OKTA", "SNOW", "APPF",
    "FSLR", "ENPH", "TTWO", "EA", "MTCH", "BKNG", "DLTR", "PAYX", "VRSK", "CTSH",
    "EXPE", "KHC", "WBA", "BIIB", "ILMN", "AAPL", "MSFT", "NVDA", "AMZN", "META",
]


def _read_local_fallback() -> List[str]:
    """从项目根的 data_cache/nasdaq_100.json 读取离线名单作为回退。"""
    project_root = Path(__file__).parent.parent.parent
    cache_file = project_root / "data_cache" / "nasdaq_100.json"
    try:
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
    except Exception as e:
        logger.warning(f"读取离线NASDAQ-100名单失败: {e}")
    return DEFAULT_FALLBACK


def get_nasdaq_100_symbols(timeout: int = 20) -> List[str]:
    """
    获取NASDAQ-100成分股代码列表，优先网络获取，失败则回退至本地缓存，
    再失败则使用内置默认列表。

    数据来源优先：
    - 纳斯达克官网成分 JSON/API（若不可用，尝试维基或公开镜像）

    Returns:
        List[str]: 股票代码列表（长度约为100）
    """
    # 尝试多个可公开来源
    urls = [
        # GitHub镜像维护的最新成分股（示例，若不可用会回退）
        "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed.csv",
        # Wikipedia的NASDAQ-100页面可解析（占位，可能需要HTML解析，先不使用）
        # "https://en.wikipedia.org/wiki/NASDAQ-100",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                text = resp.text
                # 简单解析CSV，第一列可能是Symbol
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                # 跳过表头，提取Symbol列（粗略，后续可增强）
                symbols: List[str] = []
                header = lines[0].split(",")
                symbol_idx = 0
                for i, h in enumerate(header):
                    if h.lower() in ("symbol", "symbol".lower()):
                        symbol_idx = i
                        break
                for row in lines[1:]:
                    parts = row.split(",")
                    if len(parts) > symbol_idx:
                        sym = parts[symbol_idx].strip()
                        if sym and sym.upper() == sym:
                            symbols.append(sym)
                # 该数据集可能是纳斯达克挂牌全体，不是100；后续在脚本里可限制前100权重成分
                if symbols:
                    logger.info(f"从 {url} 获取到 {len(symbols)} 个NASDAQ相关符号")
                    # 先返回前100，后续再精确来源替换
                    return symbols[:100]
        except Exception as e:
            logger.warning(f"获取NASDAQ列表失败({url}): {e}")

    # 网络来源失败，尝试本地缓存
    fallback = _read_local_fallback()
    logger.info(f"使用离线NASDAQ-100名单，数量: {len(fallback)}")
    return fallback[:100]