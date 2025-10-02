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
        max_dd = self.max_drawdown(ret)
        return {
            "cum_return": float(total_return),
            "ann_return": float(ann_return),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
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