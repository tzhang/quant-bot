from typing import Dict, List

import pandas as pd

from .technical import TechnicalFactors
from .risk import RiskFactors
from .fundamental_factors import FundamentalFactorCalculator


class FactorEngine:
    """Orchestrates factor calculations across technical, risk, and fundamental factors."""

    def __init__(self) -> None:
        self.tech = TechnicalFactors()
        self.risk = RiskFactors()
        self.fundamental = FundamentalFactorCalculator()

    def compute_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术因子
        
        Args:
            df: 包含OHLCV数据的DataFrame，列名应为大写格式
            
        Returns:
            技术因子DataFrame
        """
        # 检查必要的列
        required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 转换为TechnicalFactors期望的格式（小写列名的字典）
        data = {
            'close': df['Close'],
            'high': df['High'], 
            'low': df['Low'],
            'open': df['Open'],
            'volume': df['Volume'],
            'returns': df['Close'].pct_change().fillna(0)
        }
        
        return self.tech.calculate_all_factors(data)

    def compute_risk(self, df: pd.DataFrame, benchmark_returns: pd.Series = None) -> pd.DataFrame:
        """Compute risk factors; optionally include Beta if benchmark provided."""
        risk_factors = self.risk.calculate_all_factors(df)
        if benchmark_returns is not None:
            beta = self.risk.calculate_beta(df, benchmark_returns, window=60)
            risk_factors["BETA60"] = beta
        return risk_factors

    def compute_fundamental(self, symbol: str, price_data: pd.DataFrame, 
                          financial_data: pd.DataFrame, market_cap: float = None) -> Dict[str, float]:
        """
        计算基本面因子
        
        Args:
            symbol: 股票代码
            price_data: 价格数据
            financial_data: 财务数据
            market_cap: 市值（可选）
            
        Returns:
            Dict[str, float]: 基本面因子字典
        """
        all_factors = self.fundamental.calculate_all_factors(symbol, price_data, financial_data, market_cap)
        
        # 将嵌套字典展平为单层字典
        flat_factors = {}
        for category, factors in all_factors.items():
            for factor_name, value in factors.items():
                flat_factors[f"{category}_{factor_name}"] = value
        
        return flat_factors

    def compute_all(self, df: pd.DataFrame, symbol: str = None, 
                   financial_data: pd.DataFrame = None, market_cap: float = None) -> pd.DataFrame:
        """
        计算所有因子（技术、风险、基本面）
        
        Args:
            df: 价格数据
            symbol: 股票代码（用于基本面因子）
            financial_data: 财务数据（用于基本面因子）
            market_cap: 市值（用于基本面因子）
            
        Returns:
            pd.DataFrame: 包含所有因子的数据框
        """
        tech_factors = self.compute_technical(df)
        risk_factors = self.compute_risk(df)
        
        # 处理技术因子结果
        if isinstance(tech_factors, dict):
            # 将字典转换为DataFrame
            tech_df = pd.DataFrame(tech_factors, index=df.index)
        else:
            tech_df = tech_factors
            
        # 处理风险因子结果
        if isinstance(risk_factors, dict):
            # 将字典转换为DataFrame
            risk_df = pd.DataFrame(risk_factors, index=df.index)
        else:
            risk_df = risk_factors
        
        # 合并因子
        result = tech_df.join(risk_df, how="left")
        
        # 如果提供了基本面数据，计算基本面因子
        if symbol and financial_data is not None:
            fundamental_factors = self.compute_fundamental(symbol, df, financial_data, market_cap)
            # 将基本面因子添加到结果中（作为常数列）
            for factor_name, value in fundamental_factors.items():
                result[factor_name] = value
        
        return result

    def normalize_factors(self, factors: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """Normalize factors using z-score standardization."""
        if method == "zscore":
            return (factors - factors.mean()) / factors.std()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def winsorize_factors(self, factors: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
        """Apply winsorization to remove extreme values."""
        return factors.clip(
            lower=factors.quantile(lower, axis=0),
            upper=factors.quantile(upper, axis=0),
            axis=1,
        )

    def compute_factor_score(
        self,
        df: pd.DataFrame,
        weights: Dict[str, float] | None = None,
        normalize: bool = True,
        winsorize: bool = True,
        benchmark_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Compute all factors and synthesize into FACTOR_SCORE."""
        print(f"FactorEngine: 开始计算因子分数，数据长度={len(df)}")
        print(f"FactorEngine: 数据列={list(df.columns)}")
        print(f"FactorEngine: 数据前5行价格={df['Close'].head().tolist()}")
        
        # Get all factors
        tech_factors = self.compute_technical(df)
        risk_factors = self.compute_risk(df, benchmark_returns)
        
        print(f"FactorEngine: 技术因子类型={type(tech_factors)}, 长度={len(tech_factors) if isinstance(tech_factors, dict) else 'N/A'}")
        print(f"FactorEngine: 风险因子类型={type(risk_factors)}, 长度={len(risk_factors) if isinstance(risk_factors, dict) else 'N/A'}")
        
        if isinstance(tech_factors, dict) and tech_factors:
            first_tech_key = list(tech_factors.keys())[0]
            print(f"FactorEngine: 第一个技术因子 {first_tech_key} 前5个值={tech_factors[first_tech_key].head().tolist()}")
        
        if isinstance(risk_factors, dict) and risk_factors:
            first_risk_key = list(risk_factors.keys())[0]
            print(f"FactorEngine: 第一个风险因子 {first_risk_key} 前5个值={risk_factors[first_risk_key].head().tolist()}")
        
        # 处理技术因子结果
        if isinstance(tech_factors, dict):
            tech_df = pd.DataFrame(tech_factors, index=df.index)
        else:
            tech_df = tech_factors
            
        # 处理风险因子结果
        if isinstance(risk_factors, dict):
            risk_df = pd.DataFrame(risk_factors, index=df.index)
        else:
            risk_df = risk_factors
        
        all_factors = tech_df.join(risk_df, how="left")

        # Define default weights if not provided
        if weights is None:
            weights = {
                "RSI14": 0.15,
                "MACD_12_26_9": 0.20,
                "VOL20_ANN": -0.10,  # Lower volatility is better
                "VAR95_ANN": -0.15,  # Lower VaR is better
                "RET_DAILY": 0.25,
                "SMA20": 0.10,
                "EMA20": 0.15,
            }
            # Add BETA if available
            if "BETA60" in all_factors.columns:
                weights["BETA60"] = 0.05

        # Select only factors present in weights
        factor_cols = [col for col in weights.keys() if col in all_factors.columns]
        selected_factors = all_factors[factor_cols].copy()

        # Apply preprocessing
        if winsorize and not selected_factors.empty:
            selected_factors = self.winsorize_factors(selected_factors)
        if normalize and not selected_factors.empty:
            selected_factors = self.normalize_factors(selected_factors)

        # Compute weighted score
        factor_score = pd.Series(0.0, index=selected_factors.index)
        for col in factor_cols:
            factor_score = factor_score.add(selected_factors[col].fillna(0) * weights[col], fill_value=0)

        print(f"FactorEngine: 因子分数前5个值={factor_score.head().tolist()}")
        print(f"FactorEngine: 因子分数非零数量={(factor_score != 0).sum()}")
        print(f"FactorEngine: 因子分数范围={factor_score.min():.6f} - {factor_score.max():.6f}")

        # Return combined result
        result = all_factors.copy()
        result["FACTOR_SCORE"] = factor_score
        return result