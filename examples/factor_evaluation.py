import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import DataManager
from src.factors import FactorEngine
from src.performance import PerformanceAnalyzer


def get_symbols(dm: DataManager, max_symbols: int = 50) -> List[str]:
    """Fetch S&P500 symbols with a safe fallback list."""
    try:
        syms = dm.get_sp500_symbols()
    except Exception:
        syms = [
            # 科技股
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE", "CRM",
            # 金融股
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "V", "MA",
            # 消费股
            "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST",
            # 医疗股
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "LLY",
            # 工业股
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "DE", "EMR",
            "BRK-B", "DIS", "PYPL", "VZ"
        ]
    # Deduplicate and take a slice
    syms = [s for s in syms if isinstance(s, str)]
    return syms[:max_symbols]


def load_panel_data(dm: DataManager, symbols: List[str], start: dt.date, end: dt.date) -> Dict[str, pd.DataFrame]:
    data = dm.get_stock_data(symbols, start, end)
    # Filter out empty
    return {s: df for s, df in data.items() if isinstance(df, pd.DataFrame) and not df.empty}


def compute_factor_scores_per_symbol(
    fe: FactorEngine,
    panel: Dict[str, pd.DataFrame],
    benchmark_returns: pd.Series | None,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for sym, df in panel.items():
        try:
            # 检查数据完整性
            if df.empty or len(df) < 20:  # 至少需要20个交易日
                print(f"跳过 {sym}: 数据不足")
                continue
                
            # 处理多级索引列名
            if isinstance(df.columns, pd.MultiIndex):
                # 展平多级索引，只保留第一级
                df_flat = df.copy()
                df_flat.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                df = df_flat
            
            # 检查必需列
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"跳过 {sym}: 缺少必需列")
                continue
            
            # 计算因子分数
            factors = fe.compute_factor_score(df, benchmark_returns=benchmark_returns)
            
            if "FACTOR_SCORE" in factors.columns:
                factor_score = pd.to_numeric(factors["FACTOR_SCORE"], errors="coerce")
                # 确保有足够的有效数据
                if factor_score.notna().sum() >= 10:
                    out[sym] = factor_score
                else:
                    print(f"跳过 {sym}: 因子分数有效值不足")
            else:
                print(f"跳过 {sym}: 未生成FACTOR_SCORE")
                
        except Exception as e:
            print(f"计算 {sym} 因子分数时出错: {e}")
            continue
    
    print(f"成功计算因子分数的股票数量: {len(out)}")
    return out


def align_cross_section(scores: Dict[str, pd.Series], next_rets: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build DataFrame: index=dates, columns=symbols
    s_df = pd.DataFrame({s: v for s, v in scores.items()})
    r_df = pd.DataFrame({s: v for s, v in next_rets.items()})
    # Align by intersection of dates
    common_index = s_df.index.intersection(r_df.index)
    s_df = s_df.loc[common_index].sort_index()
    r_df = r_df.loc[common_index].sort_index()
    return s_df, r_df


def compute_ic_series(scores_cs: pd.DataFrame, fwd_returns_cs: pd.DataFrame, use_rank: bool = True) -> pd.Series:
    """计算IC时间序列"""
    ics = []
    idx = []
    
    for dt_idx in scores_cs.index:
        s = scores_cs.loc[dt_idx]
        r = fwd_returns_cs.loc[dt_idx]
        
        # 过滤有效数据
        mask = s.notna() & r.notna()
        s_clean = s[mask]
        r_clean = r[mask]
        
        if len(s_clean) < 3:  # 至少需要3个数据点
            ics.append(np.nan)
        else:
            try:
                if use_rank:
                    s_rank = s_clean.rank(method="average")
                    r_rank = r_clean.rank(method="average")
                    # 检查是否有变化
                    if s_rank.std() > 1e-8 and r_rank.std() > 1e-8:
                        corr = np.corrcoef(s_rank.values, r_rank.values)[0, 1]
                    else:
                        corr = np.nan
                else:
                    # 检查是否有变化
                    if s_clean.std() > 1e-8 and r_clean.std() > 1e-8:
                        corr = np.corrcoef(s_clean.values, r_clean.values)[0, 1]
                    else:
                        corr = np.nan
                
                ics.append(corr if not np.isnan(corr) else np.nan)
            except Exception:
                ics.append(np.nan)
        
        idx.append(dt_idx)
    
    return pd.Series(ics, index=pd.Index(idx))


def compute_layer_returns(
    scores_cs: pd.DataFrame,
    fwd_returns_cs: pd.DataFrame,
    n_quantiles: int = 5,
    trading_cost_bps: float = 10.0,
) -> Tuple[Dict[int, pd.Series], pd.Series, Dict[int, pd.Series]]:
    """Return (quantile_returns, long_short_returns, quantile_turnover).

    - Equal-weight portfolios, daily rebalanced by quantile membership.
    - Turnover approximated by membership change fraction.
    - Cost applied as turnover * bps/10000.
    """
    q_rets: Dict[int, pd.Series] = {}
    q_turn: Dict[int, pd.Series] = {}
    prev_bins: pd.Series | None = None
    ls_ret_vals = []
    ls_idx = []

    # Prepare containers per quantile
    for q in range(n_quantiles):
        q_rets[q] = pd.Series(dtype=float)
        q_turn[q] = pd.Series(dtype=float)

    bps = trading_cost_bps / 10000.0
    for dt_idx in scores_cs.index:
        s = scores_cs.loc[dt_idx]
        r = fwd_returns_cs.loc[dt_idx]
        mask = s.notna() & r.notna()
        s = s[mask]
        r = r[mask]
        if len(s) < n_quantiles:
            # Not enough symbols
            for q in range(n_quantiles):
                q_rets[q].loc[dt_idx] = np.nan
                q_turn[q].loc[dt_idx] = np.nan
            ls_ret_vals.append(np.nan)
            ls_idx.append(dt_idx)
            prev_bins = None
            continue

        # Assign quantiles [0..n_quantiles-1]
        try:
            bins = pd.qcut(s, q=n_quantiles, labels=False, duplicates="drop")
        except Exception:
            # Fallback: rank then cut by integer division
            ranks = s.rank(method="first")
            bins = (ranks / (len(s) + 1e-12) * n_quantiles).clip(0, n_quantiles - 1).astype(int)

        # Compute turnover per quantile
        for q in range(n_quantiles):
            members = bins.index[bins == q]
            if prev_bins is not None:
                prev_members = prev_bins.index[prev_bins == q]
                inter = len(set(members).intersection(set(prev_members)))
                size_prev = max(len(prev_members), 1)
                # Fraction changed from previous
                turn = 1.0 - (inter / size_prev)
            else:
                turn = 1.0  # First day considered full rebalance
            q_turn[q].loc[dt_idx] = turn
            # Equal-weight mean of next-day returns
            raw_ret = float(r.loc[members].mean()) if len(members) > 0 else np.nan
            net_ret = raw_ret - (turn * bps)
            q_rets[q].loc[dt_idx] = net_ret

        # Long-short top minus bottom
        top = q_rets[n_quantiles - 1].loc[dt_idx]
        bot = q_rets[0].loc[dt_idx]
        ls_ret_vals.append(top - bot if pd.notna(top) and pd.notna(bot) else np.nan)
        ls_idx.append(dt_idx)
        prev_bins = bins

    ls_ret = pd.Series(ls_ret_vals, index=pd.Index(ls_idx))
    return q_rets, ls_ret, q_turn


def plot_ic_series(ic_rank: pd.Series, ic_pearson: pd.Series, save_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ic_rank.index, ic_rank.values, label="Rank IC", color="blue")
    ax.plot(ic_pearson.index, ic_pearson.values, label="Pearson IC", color="orange", linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_title("Information Coefficient over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"examples/{save_prefix}_ic_ts.png", dpi=150)
    plt.close(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ic_rank.dropna().values, bins=30, alpha=0.6, label="Rank IC")
    ax.hist(ic_pearson.dropna().values, bins=30, alpha=0.6, label="Pearson IC")
    ax.set_title("IC distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"examples/{save_prefix}_ic_hist.png", dpi=150)
    plt.close(fig)


def plot_quantiles(q_rets: Dict[int, pd.Series], ls_ret: pd.Series, save_prefix: str) -> None:
    # Cumulative returns of each quantile
    fig, ax = plt.subplots(figsize=(10, 5))
    for q, s in sorted(q_rets.items()):
        eq = (1 + s.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=f"Q{q+1}")
    ax.set_title("Quantile portfolios cumulative returns (net of costs)")
    ax.set_ylabel("Equity")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(f"examples/{save_prefix}_quantiles.png", dpi=150)
    plt.close(fig)

    # Long-short equity
    fig, ax = plt.subplots(figsize=(10, 4))
    ls_eq = (1 + ls_ret.fillna(0)).cumprod()
    ax.plot(ls_eq.index, ls_eq.values, label="Long-Short (Top-Bottom)", color="purple")
    ax.set_title("Long-Short portfolio cumulative returns")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"examples/{save_prefix}_longshort_equity.png", dpi=150)
    plt.close(fig)


def plot_turnover(q_turn: Dict[int, pd.Series], save_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for q, s in sorted(q_turn.items()):
        ax.plot(s.index, s.values, label=f"Q{q+1}")
    ax.set_title("Quantile portfolios daily turnover")
    ax.set_ylabel("Turnover")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(f"examples/{save_prefix}_turnover.png", dpi=150)
    plt.close(fig)


def main() -> None:
    dm = DataManager(use_cache=True, disk_cache_dir="data_cache", default_ttl=6 * 3600)
    start = dt.date.today().replace(year=dt.date.today().year - 1)
    end = dt.date.today()
    symbols = get_symbols(dm, max_symbols=25)

    # Load benchmark for beta (optional)
    bench_df = dm.get_stock_data("^GSPC", start, end)["^GSPC"]
    bench_ret = bench_df["Close"].pct_change().fillna(0.0)

    # Load panel data
    panel = load_panel_data(dm, symbols, start, end)
    # Compute factor scores per symbol
    fe = FactorEngine()
    scores = compute_factor_scores_per_symbol(fe, panel, benchmark_returns=bench_ret)

    # Compute next-day returns per symbol
    next_rets: Dict[str, pd.Series] = {}
    for sym, df in panel.items():
        try:
            # 处理多级索引列名
            if isinstance(df.columns, pd.MultiIndex):
                close_col = None
                for col in df.columns:
                    if isinstance(col, tuple) and col[0] == 'Close':
                        close_col = col
                        break
                if close_col is None:
                    continue
                close_series = df[close_col]
            else:
                close_series = df.get("Close")
                
            if not isinstance(close_series, pd.Series):
                continue
                
            # 计算收益率并向前移动一期
            returns = close_series.pct_change()
            nxt = returns.shift(-1)
            
            # 确保Series类型和正确的索引
            nxt = pd.Series(nxt.values, index=df.index, name=sym)
            next_rets[sym] = nxt
            
        except Exception:
            print(f"计算 {sym} 未来收益时出错")
            continue

    # Align cross-section
    s_df, r_df = align_cross_section(scores, next_rets)

    # IC series
    ic_rank = compute_ic_series(s_df, r_df, use_rank=True)
    ic_pear = compute_ic_series(s_df, r_df, use_rank=False)

    # Layer test with turnover & costs
    q_rets, ls_ret, q_turn = compute_layer_returns(s_df, r_df, n_quantiles=5, trading_cost_bps=10.0)

    # Plots
    prefix = "factor_eval"
    plot_ic_series(ic_rank, ic_pear, prefix)
    plot_quantiles(q_rets, ls_ret, prefix)
    plot_turnover(q_turn, prefix)

    # Summary metrics
    perf = PerformanceAnalyzer()
    ls_metrics = perf.metrics(ls_ret.fillna(0))
    rank_ic_mean = float(ic_rank.mean()) if ic_rank.notna().sum() > 0 else np.nan
    rank_ic_ir = float(ic_rank.mean() / (ic_rank.std(ddof=1) + 1e-12)) if ic_rank.notna().sum() > 1 else np.nan
    pear_ic_mean = float(ic_pear.mean()) if ic_pear.notna().sum() > 0 else np.nan
    pear_ic_ir = float(ic_pear.mean() / (ic_pear.std(ddof=1) + 1e-12)) if ic_pear.notna().sum() > 1 else np.nan

    avg_turn = {f"Q{q+1}": float(s.mean()) for q, s in q_turn.items()}

    print("Factor Evaluation Summary:")
    print({
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_ir": rank_ic_ir,
        "pearson_ic_mean": pear_ic_mean,
        "pearson_ic_ir": pear_ic_ir,
        "long_short_metrics": ls_metrics,
        "avg_turnover": avg_turn,
    })


if __name__ == "__main__":
    main()