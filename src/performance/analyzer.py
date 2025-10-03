from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PerformanceAnalyzer:
    """Compute basic performance metrics from strategy returns."""

    @staticmethod
    def _ensure_series(returns) -> pd.Series:
        """Ensure input returns are a 1D numeric Series.

        - If DataFrame with one column: use that column.
        - If DataFrame with multiple columns: use row-wise mean.
        - If array-like: convert to Series.
        """
        if isinstance(returns, pd.Series):
            s = returns
        elif isinstance(returns, pd.DataFrame):
            if returns.shape[1] == 1:
                s = returns.iloc[:, 0]
            else:
                s = returns.mean(axis=1)
        else:
            s = pd.Series(returns)
        s = pd.to_numeric(s, errors="coerce")
        return s

    def metrics(self, returns: pd.Series, rf_annual: float = 0.0) -> Dict[str, float]:
        ret = self._ensure_series(returns).fillna(0.0)
        n = len(ret)
        total_return = (1 + ret).prod() - 1
        ann_return = ((1 + ret).prod()) ** (252 / n) - 1 if n > 0 else 0.0
        vol = float(ret.std(ddof=1))
        ann_vol = vol * np.sqrt(252)
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        excess = ret - rf_daily
        sharpe = (excess.mean() / (vol + 1e-12)) * np.sqrt(252) if vol > 0 else 0.0

        # Sortino ratio: downside deviation uses negative returns only
        downside = ret.clip(upper=0)
        downside_dev = float(downside.std(ddof=1))
        sortino = (excess.mean() / (downside_dev + 1e-12)) * np.sqrt(252) if downside_dev > 0 else 0.0

        # Max drawdown & Calmar (annual return divided by abs max drawdown)
        max_dd = self.max_drawdown(ret)
        calmar = (ann_return / (abs(max_dd) + 1e-12)) if max_dd != 0 else 0.0

        # Hit rate: fraction of positive returns
        hit_rate = float((ret > 0).sum() / n) if n > 0 else 0.0

        return {
            "cum_return": float(total_return),
            "ann_return": float(ann_return),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "hit_rate": float(hit_rate),
        }

    def equity_curve(self, returns: pd.Series) -> pd.Series:
        ret = self._ensure_series(returns).fillna(0.0)
        return (1 + ret).cumprod()

    def max_drawdown(self, returns: pd.Series) -> float:
        equity = self.equity_curve(returns)
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return float(dd.min())

    def plot_equity(self, returns: pd.Series, save_path: Optional[str] = None):
        equity = self.equity_curve(returns)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity.index, equity.values, label="Equity")
        ax.set_title("Strategy Equity Curve")
        ax.set_ylabel("Equity (normalized)")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return None
        return fig

    def plot_drawdown(self, returns: pd.Series, save_path: Optional[str] = None):
        equity = self.equity_curve(returns)
        peak = equity.cummax()
        dd = (equity - peak) / peak
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return None
        return fig

    def plot_equity_vs_benchmark(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[str] = None,
    ):
        eq = self.equity_curve(returns)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eq.index, eq.values, label="Strategy", color="blue")
        if benchmark_returns is not None:
            br = self._ensure_series(benchmark_returns).reindex(eq.index).fillna(0.0)
            beq = (1 + br).cumprod()
            ax.plot(beq.index, beq.values, label="Benchmark", color="orange", linestyle="--")
        ax.set_title("Equity Curve vs Benchmark")
        ax.set_ylabel("Equity (normalized)")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return None
        return fig

    def plot_signal_price(
        self,
        df: pd.DataFrame,
        signal: pd.Series,
        price_col: str = "Close",
        save_path: Optional[str] = None,
    ):
        sig = self._ensure_series(signal).reindex(df.index).fillna(0.0).clip(0.0, 1.0)
        # Robustly coerce price column to 1D numeric Series
        price = self._ensure_series(df[price_col])
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(price.index, price.values, color="black", label="Price")
        ax1.set_ylabel("Price")
        ax1.set_title("Price and Position Signal")
        ax2 = ax1.twinx()
        ax2.plot(sig.index, sig.values, color="green", label="Signal", alpha=0.6)
        ax2.set_ylabel("Signal [0,1]")
        # Handle legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return None
        return fig

    def plot_factor_score(
        self,
        score: pd.Series,
        window: int = 60,
        save_path: Optional[str] = None,
    ):
        s = self._ensure_series(score).fillna(0.0)
        roll_min = s.rolling(window).min()
        roll_max = s.rolling(window).max()
        norm = ((s - roll_min) / (roll_max - roll_min + 1e-12)).clip(0.0, 1.0)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s.index, s.values, label="FACTOR_SCORE", color="purple")
        ax.plot(norm.index, norm.values, label=f"MinMax({window})", color="teal", linestyle="--")
        ax.set_title("Factor Score and Rolling Normalized Signal")
        ax.set_ylabel("Score / Normalized")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return None
        return fig

    def plot_rolling_beta(
        self,
        beta: pd.Series,
        save_path: Optional[str] = None,
    ):
        b = self._ensure_series(beta)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(b.index, b.values, label="BETA", color="brown")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Beta=1")
        ax.set_title("Rolling Beta")
        ax.set_ylabel("Beta")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return None
        return fig