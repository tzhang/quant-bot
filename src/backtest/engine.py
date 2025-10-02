from typing import Dict, Optional

import pandas as pd


class BacktestEngine:
    """Simple vectorized backtest engine using signal series.

    Assumes long-only positions sized by signal in [0,1].
    """

    def __init__(self, trading_cost_bps: float = 10.0) -> None:
        self.trading_cost_bps = trading_cost_bps

    def run(self, df: pd.DataFrame, signal: pd.Series, price_col: str = "Close") -> Dict[str, pd.Series]:
        signal = signal.reindex(df.index).fillna(0.0).clip(0.0, 1.0)
        returns = df[price_col].pct_change().fillna(0.0)
        pos = signal.copy()
        # Turnover and trading cost
        turnover = pos.diff().abs().fillna(pos.abs())
        cost = turnover * (self.trading_cost_bps / 10000.0)
        strat_ret = pos.shift(1).fillna(0.0) * returns - cost
        equity = (1 + strat_ret).cumprod()
        return {
            "returns": strat_ret,
            "turnover": turnover,
            "equity": equity,
            "position": pos,
        }