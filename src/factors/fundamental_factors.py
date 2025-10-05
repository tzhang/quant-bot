"""
基本面因子计算模块

实现各种基本面因子的计算，包括估值、盈利能力、成长性、质量等因子
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
from decimal import Decimal

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalFactorCalculator:
    """
    基本面因子计算器
    
    提供各种基本面因子的计算功能，包括：
    - 估值因子：PE、PB、PS、EV/EBITDA等
    - 盈利能力因子：ROE、ROA、毛利率、净利率等
    - 成长性因子：营收增长率、盈利增长率、净资产增长率等
    - 质量因子：负债率、流动比率、资产周转率等
    """
    
    def __init__(self):
        """初始化基本面因子计算器"""
        self.logger = logger
        
    def calculate_valuation_factors(self, 
                                  price_data: pd.DataFrame,
                                  financial_data: pd.DataFrame,
                                  market_cap: Optional[float] = None) -> Dict[str, float]:
        """
        计算估值因子
        
        Args:
            price_data: 价格数据，包含close列
            financial_data: 财务数据，包含各种财务指标
            market_cap: 市值（可选）
            
        Returns:
            Dict[str, float]: 估值因子字典
        """
        try:
            factors = {}
            
            if price_data.empty or financial_data.empty:
                self.logger.warning("价格数据或财务数据为空")
                return factors
            
            current_price = price_data['close'].iloc[-1] if not price_data.empty else None
            
            # PE比率 (市盈率)
            if 'eps' in financial_data.columns and current_price:
                latest_eps = financial_data['eps'].dropna().iloc[-1] if not financial_data['eps'].dropna().empty else None
                if latest_eps and latest_eps > 0:
                    factors['pe_ratio'] = float(current_price / latest_eps)
            
            # PB比率 (市净率)
            if 'book_value_per_share' in financial_data.columns and current_price:
                latest_bvps = financial_data['book_value_per_share'].dropna().iloc[-1] if not financial_data['book_value_per_share'].dropna().empty else None
                if latest_bvps and latest_bvps > 0:
                    factors['pb_ratio'] = float(current_price / latest_bvps)
            elif 'shareholders_equity' in financial_data.columns and 'shares_outstanding' in financial_data.columns and current_price:
                # 如果没有每股净资产，用股东权益/流通股本计算
                latest_equity = financial_data['shareholders_equity'].dropna().iloc[-1] if not financial_data['shareholders_equity'].dropna().empty else None
                latest_shares = financial_data['shares_outstanding'].dropna().iloc[-1] if not financial_data['shares_outstanding'].dropna().empty else None
                if latest_equity and latest_shares and latest_shares > 0:
                    book_value_per_share = latest_equity / latest_shares
                    if book_value_per_share > 0:
                        factors['pb_ratio'] = float(current_price / book_value_per_share)
            
            # PS比率 (市销率)
            if 'revenue_per_share' in financial_data.columns and current_price:
                latest_rps = financial_data['revenue_per_share'].dropna().iloc[-1] if not financial_data['revenue_per_share'].dropna().empty else None
                if latest_rps and latest_rps > 0:
                    factors['ps_ratio'] = float(current_price / latest_rps)
            elif 'revenue' in financial_data.columns and 'shares_outstanding' in financial_data.columns and current_price:
                # 如果没有每股营收，用营收/流通股本计算
                latest_revenue = financial_data['revenue'].dropna().iloc[-1] if not financial_data['revenue'].dropna().empty else None
                latest_shares = financial_data['shares_outstanding'].dropna().iloc[-1] if not financial_data['shares_outstanding'].dropna().empty else None
                if latest_revenue and latest_shares and latest_shares > 0:
                    revenue_per_share = latest_revenue / latest_shares
                    if revenue_per_share > 0:
                        factors['ps_ratio'] = float(current_price / revenue_per_share)
            
            # EV/EBITDA比率
            if market_cap and 'ebitda' in financial_data.columns:
                latest_ebitda = financial_data['ebitda'].dropna().iloc[-1] if not financial_data['ebitda'].dropna().empty else None
                if latest_ebitda and latest_ebitda > 0:
                    # 简化计算，假设企业价值约等于市值（忽略净债务）
                    factors['ev_ebitda'] = float(market_cap / latest_ebitda)
            
            # PEG比率 (PE相对盈利增长率)
            if 'pe_ratio' in factors and 'earnings_growth' in financial_data.columns:
                latest_growth = financial_data['earnings_growth'].dropna().iloc[-1] if not financial_data['earnings_growth'].dropna().empty else None
                if latest_growth and latest_growth > 0:
                    factors['peg_ratio'] = float(factors['pe_ratio'] / (latest_growth * 100))
            
            self.logger.info(f"计算得到 {len(factors)} 个估值因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算估值因子时出错: {e}")
            return {}
    
    def calculate_profitability_factors(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算盈利能力因子
        
        Args:
            financial_data: 财务数据
            
        Returns:
            Dict[str, float]: 盈利能力因子字典
        """
        try:
            factors = {}
            
            if financial_data.empty:
                self.logger.warning("财务数据为空")
                return factors
            
            # ROE (净资产收益率)
            if 'net_income' in financial_data.columns and 'shareholders_equity' in financial_data.columns:
                latest_ni = financial_data['net_income'].dropna().iloc[-1] if not financial_data['net_income'].dropna().empty else None
                latest_equity = financial_data['shareholders_equity'].dropna().iloc[-1] if not financial_data['shareholders_equity'].dropna().empty else None
                if latest_ni and latest_equity and latest_equity > 0:
                    factors['roe'] = float(latest_ni / latest_equity)
            
            # ROA (总资产收益率)
            if 'net_income' in financial_data.columns and 'total_assets' in financial_data.columns:
                latest_ni = financial_data['net_income'].dropna().iloc[-1] if not financial_data['net_income'].dropna().empty else None
                latest_assets = financial_data['total_assets'].dropna().iloc[-1] if not financial_data['total_assets'].dropna().empty else None
                if latest_ni and latest_assets and latest_assets > 0:
                    factors['roa'] = float(latest_ni / latest_assets)
            
            # 毛利率
            if 'gross_profit' in financial_data.columns and 'revenue' in financial_data.columns:
                latest_gp = financial_data['gross_profit'].dropna().iloc[-1] if not financial_data['gross_profit'].dropna().empty else None
                latest_revenue = financial_data['revenue'].dropna().iloc[-1] if not financial_data['revenue'].dropna().empty else None
                if latest_gp and latest_revenue and latest_revenue > 0:
                    factors['gross_margin'] = float(latest_gp / latest_revenue)
            
            # 营业利润率
            if 'operating_income' in financial_data.columns and 'revenue' in financial_data.columns:
                latest_oi = financial_data['operating_income'].dropna().iloc[-1] if not financial_data['operating_income'].dropna().empty else None
                latest_revenue = financial_data['revenue'].dropna().iloc[-1] if not financial_data['revenue'].dropna().empty else None
                if latest_oi and latest_revenue and latest_revenue > 0:
                    factors['operating_margin'] = float(latest_oi / latest_revenue)
            
            # 净利率
            if 'net_income' in financial_data.columns and 'revenue' in financial_data.columns:
                latest_ni = financial_data['net_income'].dropna().iloc[-1] if not financial_data['net_income'].dropna().empty else None
                latest_revenue = financial_data['revenue'].dropna().iloc[-1] if not financial_data['revenue'].dropna().empty else None
                if latest_ni and latest_revenue and latest_revenue > 0:
                    factors['net_margin'] = float(latest_ni / latest_revenue)
            
            # EBITDA利润率
            if 'ebitda' in financial_data.columns and 'revenue' in financial_data.columns:
                latest_ebitda = financial_data['ebitda'].dropna().iloc[-1] if not financial_data['ebitda'].dropna().empty else None
                latest_revenue = financial_data['revenue'].dropna().iloc[-1] if not financial_data['revenue'].dropna().empty else None
                if latest_ebitda and latest_revenue and latest_revenue > 0:
                    factors['ebitda_margin'] = float(latest_ebitda / latest_revenue)
            
            self.logger.info(f"计算得到 {len(factors)} 个盈利能力因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算盈利能力因子时出错: {e}")
            return {}
    
    def calculate_growth_factors(self, financial_data: pd.DataFrame, periods: int = 4) -> Dict[str, float]:
        """
        计算成长性因子
        
        Args:
            financial_data: 财务数据（按时间排序）
            periods: 计算增长率的期间数（默认4个季度）
            
        Returns:
            Dict[str, float]: 成长性因子字典
        """
        try:
            factors = {}
            
            if financial_data.empty or len(financial_data) < 2:
                self.logger.warning("财务数据不足，无法计算增长率")
                return factors
            
            # 营收增长率
            if 'revenue' in financial_data.columns:
                revenue_series = financial_data['revenue'].dropna()
                if len(revenue_series) >= 2:
                    current_revenue = revenue_series.iloc[-1]
                    past_revenue = revenue_series.iloc[-min(periods+1, len(revenue_series))]
                    if past_revenue and past_revenue > 0:
                        factors['revenue_growth'] = float((current_revenue - past_revenue) / past_revenue)
            
            # 盈利增长率
            if 'net_income' in financial_data.columns:
                ni_series = financial_data['net_income'].dropna()
                if len(ni_series) >= 2:
                    current_ni = ni_series.iloc[-1]
                    past_ni = ni_series.iloc[-min(periods+1, len(ni_series))]
                    if past_ni and past_ni > 0:
                        factors['earnings_growth'] = float((current_ni - past_ni) / past_ni)
            
            # 净资产增长率
            if 'shareholders_equity' in financial_data.columns:
                equity_series = financial_data['shareholders_equity'].dropna()
                if len(equity_series) >= 2:
                    current_equity = equity_series.iloc[-1]
                    past_equity = equity_series.iloc[-min(periods+1, len(equity_series))]
                    if past_equity and past_equity > 0:
                        factors['book_value_growth'] = float((current_equity - past_equity) / past_equity)
            
            # 总资产增长率
            if 'total_assets' in financial_data.columns:
                assets_series = financial_data['total_assets'].dropna()
                if len(assets_series) >= 2:
                    current_assets = assets_series.iloc[-1]
                    past_assets = assets_series.iloc[-min(periods+1, len(assets_series))]
                    if past_assets and past_assets > 0:
                        factors['asset_growth'] = float((current_assets - past_assets) / past_assets)
            
            # EPS增长率
            if 'eps' in financial_data.columns:
                eps_series = financial_data['eps'].dropna()
                if len(eps_series) >= 2:
                    current_eps = eps_series.iloc[-1]
                    past_eps = eps_series.iloc[-min(periods+1, len(eps_series))]
                    if past_eps and past_eps > 0:
                        factors['eps_growth'] = float((current_eps - past_eps) / past_eps)
            
            self.logger.info(f"计算得到 {len(factors)} 个成长性因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算成长性因子时出错: {e}")
            return {}
    
    def calculate_quality_factors(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算质量因子
        
        Args:
            financial_data: 财务数据
            
        Returns:
            Dict[str, float]: 质量因子字典
        """
        try:
            factors = {}
            
            if financial_data.empty:
                self.logger.warning("财务数据为空")
                return factors
            
            # 流动比率
            if 'current_assets' in financial_data.columns and 'current_liabilities' in financial_data.columns:
                latest_ca = financial_data['current_assets'].dropna().iloc[-1] if not financial_data['current_assets'].dropna().empty else None
                latest_cl = financial_data['current_liabilities'].dropna().iloc[-1] if not financial_data['current_liabilities'].dropna().empty else None
                if latest_ca and latest_cl and latest_cl > 0:
                    factors['current_ratio'] = float(latest_ca / latest_cl)
            
            # 速动比率（假设速动资产 = 流动资产 - 存货）
            if 'current_assets' in financial_data.columns and 'current_liabilities' in financial_data.columns and 'inventory' in financial_data.columns:
                latest_ca = financial_data['current_assets'].dropna().iloc[-1] if not financial_data['current_assets'].dropna().empty else None
                latest_cl = financial_data['current_liabilities'].dropna().iloc[-1] if not financial_data['current_liabilities'].dropna().empty else None
                latest_inventory = financial_data['inventory'].dropna().iloc[-1] if not financial_data['inventory'].dropna().empty else None
                if latest_ca and latest_cl and latest_cl > 0:
                    quick_assets = latest_ca - (latest_inventory or 0)
                    factors['quick_ratio'] = float(quick_assets / latest_cl)
            
            # 负债权益比
            if 'total_liabilities' in financial_data.columns and 'shareholders_equity' in financial_data.columns:
                latest_liab = financial_data['total_liabilities'].dropna().iloc[-1] if not financial_data['total_liabilities'].dropna().empty else None
                latest_equity = financial_data['shareholders_equity'].dropna().iloc[-1] if not financial_data['shareholders_equity'].dropna().empty else None
                if latest_liab and latest_equity and latest_equity > 0:
                    factors['debt_to_equity'] = float(latest_liab / latest_equity)
            
            # 资产负债率
            if 'total_liabilities' in financial_data.columns and 'total_assets' in financial_data.columns:
                latest_liab = financial_data['total_liabilities'].dropna().iloc[-1] if not financial_data['total_liabilities'].dropna().empty else None
                latest_assets = financial_data['total_assets'].dropna().iloc[-1] if not financial_data['total_assets'].dropna().empty else None
                if latest_liab and latest_assets and latest_assets > 0:
                    factors['debt_ratio'] = float(latest_liab / latest_assets)
            
            # 总资产周转率
            if 'revenue' in financial_data.columns and 'total_assets' in financial_data.columns:
                latest_revenue = financial_data['revenue'].dropna().iloc[-1] if not financial_data['revenue'].dropna().empty else None
                latest_assets = financial_data['total_assets'].dropna().iloc[-1] if not financial_data['total_assets'].dropna().empty else None
                if latest_revenue and latest_assets and latest_assets > 0:
                    factors['asset_turnover'] = float(latest_revenue / latest_assets)
            
            # 存货周转率
            if 'cost_of_goods_sold' in financial_data.columns and 'inventory' in financial_data.columns:
                latest_cogs = financial_data['cost_of_goods_sold'].dropna().iloc[-1] if not financial_data['cost_of_goods_sold'].dropna().empty else None
                latest_inventory = financial_data['inventory'].dropna().iloc[-1] if not financial_data['inventory'].dropna().empty else None
                if latest_cogs and latest_inventory and latest_inventory > 0:
                    factors['inventory_turnover'] = float(latest_cogs / latest_inventory)
            
            # 应收账款周转率
            if 'revenue' in financial_data.columns and 'accounts_receivable' in financial_data.columns:
                latest_revenue = financial_data['revenue'].dropna().iloc[-1] if not financial_data['revenue'].dropna().empty else None
                latest_ar = financial_data['accounts_receivable'].dropna().iloc[-1] if not financial_data['accounts_receivable'].dropna().empty else None
                if latest_revenue and latest_ar and latest_ar > 0:
                    factors['receivables_turnover'] = float(latest_revenue / latest_ar)
            
            # 利息保障倍数
            if 'operating_income' in financial_data.columns and 'interest_expense' in financial_data.columns:
                latest_oi = financial_data['operating_income'].dropna().iloc[-1] if not financial_data['operating_income'].dropna().empty else None
                latest_interest = financial_data['interest_expense'].dropna().iloc[-1] if not financial_data['interest_expense'].dropna().empty else None
                if latest_oi and latest_interest and latest_interest > 0:
                    factors['interest_coverage'] = float(latest_oi / latest_interest)
            
            self.logger.info(f"计算得到 {len(factors)} 个质量因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算质量因子时出错: {e}")
            return {}
    
    def calculate_all_factors(self, 
                            symbol: str,
                            price_data: pd.DataFrame,
                            financial_data: pd.DataFrame,
                            market_cap: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """
        计算所有基本面因子
        
        Args:
            symbol: 股票代码
            price_data: 价格数据
            financial_data: 财务数据
            market_cap: 市值（可选）
            
        Returns:
            Dict[str, Dict[str, float]]: 所有因子分类字典
        """
        try:
            self.logger.info(f"开始计算 {symbol} 的基本面因子")
            
            all_factors = {
                'valuation': self.calculate_valuation_factors(price_data, financial_data, market_cap),
                'profitability': self.calculate_profitability_factors(financial_data),
                'growth': self.calculate_growth_factors(financial_data),
                'quality': self.calculate_quality_factors(financial_data)
            }
            
            # 统计总因子数
            total_factors = sum(len(factors) for factors in all_factors.values())
            self.logger.info(f"为 {symbol} 计算得到总共 {total_factors} 个基本面因子")
            
            return all_factors
            
        except Exception as e:
            self.logger.error(f"计算 {symbol} 的基本面因子时出错: {e}")
            return {}
    
    def calculate_factor_scores(self, factors: Dict[str, float], 
                              benchmarks: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        计算因子得分（相对于基准的标准化得分）
        
        Args:
            factors: 因子值字典
            benchmarks: 基准值字典（可选）
            
        Returns:
            Dict[str, float]: 因子得分字典
        """
        try:
            scores = {}
            
            # 如果没有提供基准，使用简单的标准化
            if not benchmarks:
                factor_values = list(factors.values())
                if factor_values:
                    mean_val = np.mean(factor_values)
                    std_val = np.std(factor_values)
                    if std_val > 0:
                        for name, value in factors.items():
                            scores[f"{name}_score"] = float((value - mean_val) / std_val)
            else:
                # 使用基准计算相对得分
                for name, value in factors.items():
                    if name in benchmarks and benchmarks[name] != 0:
                        scores[f"{name}_score"] = float(value / benchmarks[name] - 1)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"计算因子得分时出错: {e}")
            return {}
    
    def generate_factor_report(self, symbol: str, all_factors: Dict[str, Dict[str, float]]) -> str:
        """
        生成基本面因子分析报告
        
        Args:
            symbol: 股票代码
            all_factors: 所有因子数据
            
        Returns:
            str: 分析报告
        """
        try:
            report = f"\n=== {symbol} 基本面因子分析报告 ===\n"
            report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for category, factors in all_factors.items():
                if factors:
                    report += f"【{category.upper()} 因子】\n"
                    for factor_name, value in factors.items():
                        report += f"  {factor_name}: {value:.4f}\n"
                    report += "\n"
            
            # 添加简单的评估
            report += "【综合评估】\n"
            
            # 估值评估
            valuation = all_factors.get('valuation', {})
            if 'pe_ratio' in valuation:
                pe = valuation['pe_ratio']
                if pe < 15:
                    report += f"  估值水平: 低估 (PE: {pe:.2f})\n"
                elif pe > 25:
                    report += f"  估值水平: 高估 (PE: {pe:.2f})\n"
                else:
                    report += f"  估值水平: 合理 (PE: {pe:.2f})\n"
            
            # 盈利能力评估
            profitability = all_factors.get('profitability', {})
            if 'roe' in profitability:
                roe = profitability['roe']
                if roe > 0.15:
                    report += f"  盈利能力: 优秀 (ROE: {roe:.2%})\n"
                elif roe > 0.10:
                    report += f"  盈利能力: 良好 (ROE: {roe:.2%})\n"
                else:
                    report += f"  盈利能力: 一般 (ROE: {roe:.2%})\n"
            
            # 成长性评估
            growth = all_factors.get('growth', {})
            if 'revenue_growth' in growth:
                rg = growth['revenue_growth']
                if rg > 0.20:
                    report += f"  成长性: 高成长 (营收增长: {rg:.2%})\n"
                elif rg > 0.10:
                    report += f"  成长性: 稳定成长 (营收增长: {rg:.2%})\n"
                else:
                    report += f"  成长性: 低成长 (营收增长: {rg:.2%})\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成因子报告时出错: {e}")
            return f"生成 {symbol} 因子报告时出错: {e}"


def main():
    """测试基本面因子计算功能"""
    calculator = FundamentalFactorCalculator()
    
    # 创建测试数据
    price_data = pd.DataFrame({
        'close': [100, 102, 105, 103, 108]
    })
    
    financial_data = pd.DataFrame({
        'revenue': [1000000, 1100000, 1200000, 1300000],
        'net_income': [100000, 110000, 120000, 130000],
        'total_assets': [2000000, 2100000, 2200000, 2300000],
        'shareholders_equity': [800000, 850000, 900000, 950000],
        'eps': [2.5, 2.75, 3.0, 3.25],
        'current_assets': [500000, 520000, 540000, 560000],
        'current_liabilities': [200000, 210000, 220000, 230000],
        'total_liabilities': [1200000, 1250000, 1300000, 1350000]
    })
    
    # 计算所有因子
    all_factors = calculator.calculate_all_factors('TEST', price_data, financial_data, 5000000)
    
    # 生成报告
    report = calculator.generate_factor_report('TEST', all_factors)
    print(report)


if __name__ == "__main__":
    main()