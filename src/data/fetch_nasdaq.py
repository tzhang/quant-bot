#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取 NASDAQ-100 股票的日频 OHLCV，并入库到 Postgres。

功能：
- 通过 `get_nasdaq_100_symbols` 获取成分股（带网络失败回退）。
- 读取环境变量控制抓取范围与缓存：
  - `MAX_TICKERS`（默认 100）
  - `DATA_START_DATE`（默认 5 年前）
  - `DATA_END_DATE`（默认今天）
  - `BATCH_SIZE`（每批入库记录数，默认 1000）
  - `THROTTLE_SEC`（每个符号间的暂停秒数，默认 0.5）
  - `DATA_CACHE_DIR`（默认 `data_cache/nasdaq`）
  - `USE_PARQUET`（true/false，默认 false）
- 自动避免重复：若数据库已有该符号最新日期，则仅拉取后续日期。
- 将每个符号的数据缓存为 CSV（可选 Parquet）。
"""

import os
import time
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import random

import pandas as pd
import yfinance as yf
import requests
from requests.exceptions import RequestException
from urllib.parse import urljoin
import hashlib
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.nasdaq import get_nasdaq_100_symbols
from src.database.dao import stock_data_dao
from src.data.alternative_sources import alternative_data_manager


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


class AdvancedRateLimitHandler:
    """
    高级反频率限制处理器
    
    功能：
    - 动态User-Agent轮换
    - 会话管理和连接池优化
    - 智能延迟策略
    - 数据缓存机制
    """
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
        ]
        self.current_ua_index = 0
        self.session = requests.Session()
        self.failure_count = 0
        self.success_count = 0
        self.cache_dir = Path("data_cache/yfinance_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置会话
        self._setup_session()
    
    def _setup_session(self):
        """配置会话参数和连接池优化"""
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        # 连接池配置
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 统计变量
        self.cache_hits = 0
        self.ua_rotations = 0
    
    def get_next_user_agent(self) -> str:
        """获取下一个User-Agent"""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua
    
    def update_user_agent(self):
        """更新会话的User-Agent"""
        self.session.headers['User-Agent'] = self.get_next_user_agent()
        self.ua_rotations += 1
    
    def get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        key_str = f"{symbol}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cached_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取缓存数据"""
        cache_key = self.get_cache_key(symbol, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if cache_file.exists():
            try:
                # 检查缓存是否过期（1小时）
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < 3600:  # 1小时内的缓存有效
                    logger.info(f"使用缓存数据: {symbol}")
                    return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        return pd.DataFrame()
    
    def cache_data(self, data: pd.DataFrame, symbol: str, start_date: str, end_date: str):
        """缓存数据"""
        if data.empty:
            return
            
        cache_key = self.get_cache_key(symbol, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        try:
            data.to_parquet(cache_file)
            logger.debug(f"数据已缓存: {symbol}")
        except Exception as e:
            logger.warning(f"缓存数据失败: {e}")
    
    def calculate_dynamic_delay(self) -> float:
        """计算动态延迟时间"""
        total_requests = self.success_count + self.failure_count
        if total_requests == 0:
            return float(os.getenv("THROTTLE_SEC", "1.0"))
        
        failure_rate = self.failure_count / total_requests
        base_delay = float(os.getenv("THROTTLE_SEC", "1.0"))
        
        if failure_rate > 0.5:  # 失败率超过50%
            return base_delay * 4.0
        elif failure_rate > 0.3:  # 失败率超过30%
            return base_delay * 2.5
        elif failure_rate > 0.1:  # 失败率超过10%
            return base_delay * 1.5
        else:
            return base_delay
    
    def get_adaptive_delay(self) -> float:
        """获取自适应延迟时间"""
        return self.calculate_dynamic_delay()
    
    def get_failure_rate(self) -> float:
        """获取失败率"""
        total_requests = self.success_count + self.failure_count
        if total_requests == 0:
            return 0.0
        return self.failure_count / total_requests
    
    def record_success(self):
        """记录成功请求"""
        self.success_count += 1
    
    def record_failure(self):
        """记录失败请求"""
        self.failure_count += 1


# 全局实例
rate_limit_handler = AdvancedRateLimitHandler()


def _get_env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).lower() in {"1", "true", "yes", "y"}


def _parse_date(name: str, default: datetime) -> datetime:
    """
    解析环境变量中的日期字符串。
    
    Args:
        name: 环境变量名
        default: 默认datetime对象
        
    Returns:
        解析后的datetime对象（时区感知）
    """
    v = os.getenv(name)
    if not v:
        return default
    try:
        parsed = datetime.strptime(v, "%Y-%m-%d")
        # 确保返回时区感知的datetime
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        logger.warning(f"环境变量 {name} 格式错误，使用默认值 {default.date()}")
        return default


def fetch_and_store_symbol(symbol: str, start_date: datetime, end_date: datetime,
                           cache_dir: Path, use_parquet: bool,
                           batch_size: int, max_retries: int = 3) -> int:
    """
    拉取单个符号的数据并入库与缓存，返回成功入库的记录数。
    
    Args:
        symbol: 股票符号
        start_date: 开始日期
        end_date: 结束日期
        cache_dir: 缓存目录
        use_parquet: 是否使用Parquet格式
        batch_size: 批量大小
        max_retries: 最大重试次数
        
    Returns:
        成功入库的记录数
    """
    # 避免重复：查询数据库最新日期
    latest = stock_data_dao.get_latest_by_symbol(symbol)
    eff_start = start_date
    if latest and latest.date:
        eff_start = max(eff_start, latest.date + timedelta(days=1))

    if eff_start > end_date:
        logger.info(f"{symbol}: 数据已最新，无需更新。")
        return 0

    # 检查缓存
    start_str = eff_start.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    cached_data = rate_limit_handler.get_cached_data(symbol, start_str, end_str)
    
    if not cached_data.empty:
        logger.info(f"{symbol}: 使用缓存数据，跳过下载")
        df = cached_data
    else:
        logger.info(f"{symbol}: 下载 {eff_start.date()} 至 {end_date.date()} 的日线数据")
        
        # 实现指数退避重试机制
        df = None
        for attempt in range(max_retries):
            try:
                # 更新User-Agent
                rate_limit_handler.update_user_agent()
                
                # 使用yfinance下载数据
                df = yf.download(
                    symbol,
                    start=eff_start.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=False,
                    prepost=False,
                    threads=False,  # 禁用多线程避免频率限制
                    progress=False
                )
                
                if df is not None and not df.empty:
                    rate_limit_handler.record_success()
                    # 缓存成功下载的数据
                    rate_limit_handler.cache_data(df, symbol, start_str, end_str)
                    break
                else:
                    logger.warning(f"{symbol}: 第 {attempt + 1} 次尝试返回空数据")
                    rate_limit_handler.record_failure()
                    
            except Exception as e:
                rate_limit_handler.record_failure()
                if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"{symbol}: 频率限制，第 {attempt + 1} 次重试，等待 {wait_time:.1f} 秒")
                    time.sleep(wait_time)
                else:
                    logger.error(f"{symbol}: 下载失败 (第 {attempt + 1} 次): {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
        
        if df is None or df.empty:
            # 尝试使用备用数据源
            logger.info(f"{symbol}: yfinance失败，尝试备用数据源")
            df = alternative_data_manager.fetch_data_with_fallback(
                symbol, start_str, end_str
            )
            
            if df is not None and not df.empty:
                logger.info(f"{symbol}: 备用数据源成功获取数据")
                # 缓存备用数据源的数据
                rate_limit_handler.cache_data(df, symbol, start_str, end_str)
            else:
                logger.error(f"{symbol}: 所有数据源均失败，跳过")
                return 0

    if df is None or df.empty:
        logger.warning(f"{symbol}: 无数据返回")
        return 0

    # 统一列名与类型
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    # 去除成交量为0的行
    df = df[df["volume"] > 0]

    if df.empty:
        logger.warning(f"{symbol}: 清洗后无有效数据")
        return 0

    # 缓存到文件
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out.insert(0, "date", pd.to_datetime(df_out.index))
    df_out.reset_index(drop=True, inplace=True)
    csv_path = cache_dir / f"{symbol}.csv"
    df_out.to_csv(csv_path, index=False)
    if use_parquet:
        parquet_path = cache_dir / f"{symbol}.parquet"
        try:
            df_out.to_parquet(parquet_path, index=False)
        except Exception as e:
            logger.warning(f"{symbol}: 写入Parquet失败: {e}")

    # 构建批量入库数据
    records: List[Dict] = []
    for dt, row in df.iterrows():
        try:
            records.append({
                "symbol": symbol,
                "date": pd.to_datetime(dt).to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })
        except Exception:
            continue

    if not records:
        logger.warning(f"{symbol}: 无可入库记录")
        return 0

    # 分批入库，避免一次提交过大
    total = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i:i+batch_size]
        try:
            stock_data_dao.batch_create(chunk)
            total += len(chunk)
        except Exception as e:
            logger.error(f"{symbol}: 批量入库失败 ({len(chunk)} 条): {e}")

    logger.info(f"{symbol}: 入库完成，共 {total} 条")
    return total


def main():
    """
    主函数：批量抓取NASDAQ-100数据并入库。
    
    支持的环境变量：
    - MAX_TICKERS: 最大处理股票数量（默认100）
    - DATA_START_DATE: 开始日期（默认5年前）
    - DATA_END_DATE: 结束日期（默认今天）
    - BATCH_SIZE: 批量入库大小（默认1000）
    - THROTTLE_SEC: 请求间隔秒数（默认0.5）
    - DATA_CACHE_DIR: 缓存目录（默认data_cache/nasdaq）
    - USE_PARQUET: 是否使用Parquet格式（默认false）
    """
    max_tickers = int(os.getenv("MAX_TICKERS", "100"))
    batch_size = int(os.getenv("BATCH_SIZE", "1000"))
    throttle_sec = float(os.getenv("THROTTLE_SEC", "0.5"))
    use_parquet = _get_env_bool("USE_PARQUET", False)

    today = datetime.now(timezone.utc).date()
    default_start = datetime(today.year - 5, today.month, today.day, tzinfo=timezone.utc)
    start_date = _parse_date("DATA_START_DATE", default_start)
    end_date = _parse_date("DATA_END_DATE", datetime.now(timezone.utc))

    cache_dir_env = os.getenv("DATA_CACHE_DIR", "data_cache/nasdaq")
    cache_dir = Path(cache_dir_env)

    logger.info(
        f"配置: MAX_TICKERS={max_tickers}, 范围={start_date.date()}~{end_date.date()}, "
        f"BATCH_SIZE={batch_size}, THROTTLE_SEC={throttle_sec}, CACHE_DIR={cache_dir}, PARQUET={use_parquet}"
    )

    symbols = get_nasdaq_100_symbols()
    if not symbols:
        logger.error("无法获取NASDAQ-100成分股列表")
        return

    symbols = symbols[:max_tickers]
    logger.info(f"开始处理 {len(symbols)} 个符号")

    total_inserted = 0
    processed = 0
    failed = 0
    
    for i, sym in enumerate(symbols, 1):
        try:
            logger.info(f"处理进度: {i}/{len(symbols)} - {sym}")
            
            # 使用智能延迟策略
            delay = rate_limit_handler.get_adaptive_delay()
            if delay > 0:
                logger.info(f"智能延迟: {delay:.1f} 秒 (基于失败率: {rate_limit_handler.get_failure_rate():.1%})")
                time.sleep(delay)
            
            inserted = fetch_and_store_symbol(
                sym, start_date, end_date, cache_dir, use_parquet, batch_size
            )
            total_inserted += inserted
            processed += 1
            if inserted > 0:
                logger.info(f"{sym}: 成功入库 {inserted} 条记录")
        except KeyboardInterrupt:
            logger.info("用户中断，正在退出...")
            break
        except Exception as e:
            logger.error(f"{sym}: 处理失败: {e}")
            failed += 1
        finally:
            # 基础延迟
            time.sleep(throttle_sec)

    # 显示统计信息
    success_rate = processed / (processed + failed) if (processed + failed) > 0 else 0
    logger.info(f"处理完成: {processed} 个符号成功，{failed} 个失败，共入库 {total_inserted} 条记录")
    logger.info(f"成功率: {success_rate:.1%}, 缓存命中: {rate_limit_handler.cache_hits}, User-Agent轮换: {rate_limit_handler.ua_rotations}")


if __name__ == "__main__":
    main()