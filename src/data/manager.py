
# ==========================================
# 迁移说明 - 2025-10-10 23:06:36
# ==========================================
# 本文件已从yfinance迁移到IB TWS API
# 原始文件备份在: backup_before_ib_migration/src/data/manager.py
# 
# 主要变更:
# # - 替换yfinance导入为IB导入
# - 检测到yf.download()调用，需要手动调整
# 
# 注意事项:
# 1. 需要启动IB TWS或Gateway
# 2. 确保API设置已正确配置
# 3. 某些yfinance特有功能可能需要手动调整
# ==========================================

import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from src.data.ib_data_provider import IBDataProvider, IBConfig
from diskcache import Cache
import requests

# 尝试导入yfinance作为备用数据源
try:
    # import yfinance as yf  # 已移除，不再使用yfinance
    YF_AVAILABLE = False
except ImportError:
    YF_AVAILABLE = False
    print("Warning: yfinance not available, using alternative data sources")


class DataManager:
    """Unified data access and caching manager with memory + disk persistence."""

    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: str = ".cache",
        default_interval: str = "1d",
        disk_cache_dir: str = "data_cache",
        default_ttl: int = 6 * 3600,
        use_parquet: bool = False,
    ) -> None:
        self.default_interval = default_interval
        self.cache: Optional[Cache] = Cache(cache_dir) if use_cache else None
        # Disk cache configuration
        self.disk_cache_dir = Path(disk_cache_dir)
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        self.use_parquet = use_parquet

    def _safe_key(self, key: str) -> str:
        return key.replace(":", "_").replace("/", "_").replace("\\", "_")

    def _get_disk_cache_path(self, key: str) -> Path:
        safe = self._safe_key(key)
        # Prefer parquet if configured, but we may fallback to csv
        ext = ".parquet" if self.use_parquet else ".csv"
        return self.disk_cache_dir / f"{safe}{ext}"

    def _get_disk_meta_path(self, key: str) -> Path:
        safe = self._safe_key(key)
        return self.disk_cache_dir / f"{safe}.meta"

    def _find_existing_data_path(self, safe_stem: str) -> Optional[Path]:
        # Try parquet first, then csv
        pq = self.disk_cache_dir / f"{safe_stem}.parquet"
        csv = self.disk_cache_dir / f"{safe_stem}.csv"
        if pq.exists():
            return pq
        if csv.exists():
            return csv
        return None

    def _is_disk_cache_valid(self, key: str) -> bool:
        meta_path = self._get_disk_meta_path(key)
        data_path = self._get_disk_cache_path(key)
        # If preferred file is missing, try alternate extension
        if not data_path.exists():
            alt = self._find_existing_data_path(self._safe_key(key))
            if alt is None:
                return False
            data_path = alt
        if not meta_path.exists():
            return False
        try:
            with open(meta_path, "r") as f:
                ts = f.read().strip()
            cached_time = dt.datetime.fromisoformat(ts)
            return (dt.datetime.now() - cached_time).total_seconds() < self.default_ttl
        except Exception:
            return False

    def _save_to_disk_cache(self, key: str, data: pd.DataFrame) -> None:
        data_path = self._get_disk_cache_path(key)
        meta_path = self._get_disk_meta_path(key)
        try:
            if self.use_parquet:
                try:
                    data.to_parquet(data_path)
                except Exception:
                    # Fallback to CSV if parquet engine not available
                    data_path = self.disk_cache_dir / f"{self._safe_key(key)}.csv"
                    data.to_csv(data_path, index=True)
            else:
                data.to_csv(data_path, index=True)
            with open(meta_path, "w") as f:
                f.write(dt.datetime.now().isoformat())
        except Exception:
            # Non-fatal if disk write fails
            pass

    def _load_from_disk_cache(self, key: str) -> Optional[pd.DataFrame]:
        if not self._is_disk_cache_valid(key):
            return None
        # Prefer parquet if exists, else csv
        safe = self._safe_key(key)
        path = self._find_existing_data_path(safe)
        if path is None:
            return None
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            return None

    def _cleanup_expired_cache(self) -> None:
        for meta in self.disk_cache_dir.glob("*.meta"):
            try:
                with open(meta, "r") as f:
                    ts = f.read().strip()
                cached_time = dt.datetime.fromisoformat(ts)
            except Exception:
                cached_time = dt.datetime.min
            expired = (dt.datetime.now() - cached_time).total_seconds() >= self.default_ttl
            if expired:
                stem = meta.stem
                data_path = self._find_existing_data_path(stem)
                try:
                    meta.unlink(missing_ok=True)
                    if data_path is not None:
                        data_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def cache_data(self, key: str, data: Union[pd.DataFrame, pd.Series, Dict], expiry: int = 3600) -> None:
        if self.cache is not None:
            self.cache.set(key, data, expire=expiry)

    def _get_cached(self, key: str) -> Optional[Union[pd.DataFrame, pd.Series, Dict]]:
        if self.cache is None:
            return None
        return self.cache.get(key)

    def get_sp500_symbols(self) -> List[str]:
        key = "sp500_symbols"
        cached = self._get_cached(key)
        if cached is not None:
            return cached  # type: ignore
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df["Symbol"].astype(str).tolist()
        symbols = [s.replace(".", "-") for s in symbols]
        self.cache_data(key, symbols, expiry=24 * 3600)
        return symbols

    def _standardize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Adj Close": "AdjClose", "Volume": "Volume"}
        out = df.rename(columns=cols)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in out.columns and c in df.columns:
                out[c] = df[c]
        return out[[c for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"] if c in out.columns]].sort_index()

    def get_stock_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, dt.date, dt.datetime],
        end_date: Union[str, dt.date, dt.datetime],
        interval: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        interval = interval or self.default_interval
        if isinstance(symbols, str):
            symbols = [symbols]
        # Periodically cleanup expired disk cache
        self._cleanup_expired_cache()
        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            key = f"ohlcv:{sym}:{start_date}:{end_date}:{interval}"
            cached = self._get_cached(key)
            if isinstance(cached, pd.DataFrame):
                result[sym] = cached
                continue
            # Try disk cache
            disk_cached = self._load_from_disk_cache(key)
            if isinstance(disk_cached, pd.DataFrame):
                # Warm memory cache
                self.cache_data(key, disk_cached, expiry=self.default_ttl)
                result[sym] = disk_cached
                continue
            # 第一优先级：尝试使用IB TWS API获取数据
            try:
                ib_provider = IBDataProvider(IBConfig())
                df = ib_provider.get_stock_data(sym, start_date, end_date)
                if df is not None and len(df) > 0:
                    std = self._standardize_ohlcv(df)
                    self.cache_data(key, std, expiry=self.default_ttl)
                    self._save_to_disk_cache(key, std)
                    result[sym] = std
                    continue
            except Exception as e:
                print(f"IB TWS API获取数据失败 {sym}: {e}")
            
            # 第二优先级：尝试使用yfinance获取数据
            if YF_AVAILABLE:
                try:
                    df = yf.download(sym, start=start_date, end=end_date, interval=interval, progress=False)
                    if df is None or len(df) == 0:
                        print(f"Warning: No data found for {sym} using yfinance")
                        result[sym] = pd.DataFrame()
                        continue
                    std = self._standardize_ohlcv(df)
                    self.cache_data(key, std, expiry=self.default_ttl)
                    self._save_to_disk_cache(key, std)
                    result[sym] = std
                    continue
                except Exception as e:
                    print(f"Error fetching data for {sym} using yfinance: {e}")
                    result[sym] = pd.DataFrame()
                    continue
            else:
                print(f"Warning: yfinance not available, skipping {sym}")
                result[sym] = pd.DataFrame()
                continue
        return result

    def get_market_data(
        self,
        start_date: Union[str, dt.date, dt.datetime],
        end_date: Union[str, dt.date, dt.datetime],
        symbols: Optional[List[str]] = None,
        interval: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        tickers = symbols or ["^GSPC", "^VIX", "SPY"]
        return self.get_stock_data(tickers, start_date, end_date, interval)