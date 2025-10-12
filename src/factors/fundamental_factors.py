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
from scipy import stats

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
            
            # 获取最新价格
            current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
            
            # 市盈率 (P/E Ratio)
            if 'eps' in financial_data.columns:
                latest_eps = financial_data['eps'].iloc[-1]
                if latest_eps > 0:
                    factors['pe_ratio'] = current_price / latest_eps
                else:
                    factors['pe_ratio'] = np.nan
            
            # 市净率 (P/B Ratio)
            if 'book_value_per_share' in financial_data.columns:
                latest_bvps = financial_data['book_value_per_share'].iloc[-1]
                if latest_bvps > 0:
                    factors['pb_ratio'] = current_price / latest_bvps
                else:
                    factors['pb_ratio'] = np.nan
            
            # 市销率 (P/S Ratio)
            if 'revenue_per_share' in financial_data.columns:
                latest_rps = financial_data['revenue_per_share'].iloc[-1]
                if latest_rps > 0:
                    factors['ps_ratio'] = current_price / latest_rps
                else:
                    factors['ps_ratio'] = np.nan
            
            # 市现率 (P/CF Ratio)
            if 'cash_flow_per_share' in financial_data.columns:
                latest_cfps = financial_data['cash_flow_per_share'].iloc[-1]
                if latest_cfps > 0:
                    factors['pcf_ratio'] = current_price / latest_cfps
                else:
                    factors['pcf_ratio'] = np.nan
            
            # EV/EBITDA
            if market_cap and 'ebitda' in financial_data.columns and 'net_debt' in financial_data.columns:
                latest_ebitda = financial_data['ebitda'].iloc[-1]
                latest_net_debt = financial_data['net_debt'].iloc[-1]
                enterprise_value = market_cap + latest_net_debt
                if latest_ebitda > 0:
                    factors['ev_ebitda'] = enterprise_value / latest_ebitda
                else:
                    factors['ev_ebitda'] = np.nan
            
            # EV/Sales
            if market_cap and 'total_revenue' in financial_data.columns and 'net_debt' in financial_data.columns:
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                latest_net_debt = financial_data['net_debt'].iloc[-1]
                enterprise_value = market_cap + latest_net_debt
                if latest_revenue > 0:
                    factors['ev_sales'] = enterprise_value / latest_revenue
                else:
                    factors['ev_sales'] = np.nan
            
            # PEG比率 (PE/Growth)
            if 'pe_ratio' in factors and 'earnings_growth_rate' in financial_data.columns:
                growth_rate = financial_data['earnings_growth_rate'].iloc[-1]
                if growth_rate > 0:
                    factors['peg_ratio'] = factors['pe_ratio'] / (growth_rate * 100)
                else:
                    factors['peg_ratio'] = np.nan
            
            # 股息收益率
            if 'dividend_per_share' in financial_data.columns:
                latest_dps = financial_data['dividend_per_share'].iloc[-1]
                if current_price > 0:
                    factors['dividend_yield'] = (latest_dps / current_price) * 100
                else:
                    factors['dividend_yield'] = 0
            
            # 自由现金流收益率
            if 'free_cash_flow_per_share' in financial_data.columns:
                latest_fcfps = financial_data['free_cash_flow_per_share'].iloc[-1]
                if current_price > 0:
                    factors['fcf_yield'] = (latest_fcfps / current_price) * 100
                else:
                    factors['fcf_yield'] = 0
            
            # 盈利收益率 (Earnings Yield)
            if 'pe_ratio' in factors and factors['pe_ratio'] > 0:
                factors['earnings_yield'] = (1 / factors['pe_ratio']) * 100
            else:
                factors['earnings_yield'] = 0
            
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
            
            # ROE (净资产收益率)
            if 'net_income' in financial_data.columns and 'shareholders_equity' in financial_data.columns:
                latest_ni = financial_data['net_income'].iloc[-1]
                latest_equity = financial_data['shareholders_equity'].iloc[-1]
                if latest_equity > 0:
                    factors['roe'] = (latest_ni / latest_equity) * 100
                else:
                    factors['roe'] = 0
            
            # ROA (总资产收益率)
            if 'net_income' in financial_data.columns and 'total_assets' in financial_data.columns:
                latest_ni = financial_data['net_income'].iloc[-1]
                latest_assets = financial_data['total_assets'].iloc[-1]
                if latest_assets > 0:
                    factors['roa'] = (latest_ni / latest_assets) * 100
                else:
                    factors['roa'] = 0
            
            # ROIC (投入资本收益率)
            if 'operating_income' in financial_data.columns and 'invested_capital' in financial_data.columns:
                latest_oi = financial_data['operating_income'].iloc[-1]
                latest_ic = financial_data['invested_capital'].iloc[-1]
                if latest_ic > 0:
                    factors['roic'] = (latest_oi / latest_ic) * 100
                else:
                    factors['roic'] = 0
            
            # 毛利率
            if 'gross_profit' in financial_data.columns and 'total_revenue' in financial_data.columns:
                latest_gp = financial_data['gross_profit'].iloc[-1]
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                if latest_revenue > 0:
                    factors['gross_margin'] = (latest_gp / latest_revenue) * 100
                else:
                    factors['gross_margin'] = 0
            
            # 营业利润率
            if 'operating_income' in financial_data.columns and 'total_revenue' in financial_data.columns:
                latest_oi = financial_data['operating_income'].iloc[-1]
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                if latest_revenue > 0:
                    factors['operating_margin'] = (latest_oi / latest_revenue) * 100
                else:
                    factors['operating_margin'] = 0
            
            # 净利润率
            if 'net_income' in financial_data.columns and 'total_revenue' in financial_data.columns:
                latest_ni = financial_data['net_income'].iloc[-1]
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                if latest_revenue > 0:
                    factors['net_margin'] = (latest_ni / latest_revenue) * 100
                else:
                    factors['net_margin'] = 0
            
            # EBITDA利润率
            if 'ebitda' in financial_data.columns and 'total_revenue' in financial_data.columns:
                latest_ebitda = financial_data['ebitda'].iloc[-1]
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                if latest_revenue > 0:
                    factors['ebitda_margin'] = (latest_ebitda / latest_revenue) * 100
                else:
                    factors['ebitda_margin'] = 0
            
            # 税前利润率
            if 'pretax_income' in financial_data.columns and 'total_revenue' in financial_data.columns:
                latest_pti = financial_data['pretax_income'].iloc[-1]
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                if latest_revenue > 0:
                    factors['pretax_margin'] = (latest_pti / latest_revenue) * 100
                else:
                    factors['pretax_margin'] = 0
            
            # 资产周转率
            if 'total_revenue' in financial_data.columns and 'total_assets' in financial_data.columns:
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                latest_assets = financial_data['total_assets'].iloc[-1]
                if latest_assets > 0:
                    factors['asset_turnover'] = latest_revenue / latest_assets
                else:
                    factors['asset_turnover'] = 0
            
            # 权益乘数
            if 'total_assets' in financial_data.columns and 'shareholders_equity' in financial_data.columns:
                latest_assets = financial_data['total_assets'].iloc[-1]
                latest_equity = financial_data['shareholders_equity'].iloc[-1]
                if latest_equity > 0:
                    factors['equity_multiplier'] = latest_assets / latest_equity
                else:
                    factors['equity_multiplier'] = 0
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算盈利能力因子时出错: {e}")
            return {}

    def calculate_growth_factors(self, financial_data: pd.DataFrame, periods: int = 4) -> Dict[str, float]:
        """
        计算成长性因子
        
        Args:
            financial_data: 财务数据
            periods: 计算增长率的期数（默认4个季度）
            
        Returns:
            Dict[str, float]: 成长性因子字典
        """
        try:
            factors = {}
            
            # 营收增长率
            if 'total_revenue' in financial_data.columns and len(financial_data) >= periods + 1:
                current_revenue = financial_data['total_revenue'].iloc[-1]
                past_revenue = financial_data['total_revenue'].iloc[-(periods + 1)]
                if past_revenue > 0:
                    factors['revenue_growth'] = ((current_revenue / past_revenue) ** (1/periods) - 1) * 100
                else:
                    factors['revenue_growth'] = 0
            
            # 净利润增长率
            if 'net_income' in financial_data.columns and len(financial_data) >= periods + 1:
                current_ni = financial_data['net_income'].iloc[-1]
                past_ni = financial_data['net_income'].iloc[-(periods + 1)]
                if past_ni > 0:
                    factors['earnings_growth'] = ((current_ni / past_ni) ** (1/periods) - 1) * 100
                else:
                    factors['earnings_growth'] = 0
            
            # EPS增长率
            if 'eps' in financial_data.columns and len(financial_data) >= periods + 1:
                current_eps = financial_data['eps'].iloc[-1]
                past_eps = financial_data['eps'].iloc[-(periods + 1)]
                if past_eps > 0:
                    factors['eps_growth'] = ((current_eps / past_eps) ** (1/periods) - 1) * 100
                else:
                    factors['eps_growth'] = 0
            
            # 总资产增长率
            if 'total_assets' in financial_data.columns and len(financial_data) >= periods + 1:
                current_assets = financial_data['total_assets'].iloc[-1]
                past_assets = financial_data['total_assets'].iloc[-(periods + 1)]
                if past_assets > 0:
                    factors['assets_growth'] = ((current_assets / past_assets) ** (1/periods) - 1) * 100
                else:
                    factors['assets_growth'] = 0
            
            # 净资产增长率
            if 'shareholders_equity' in financial_data.columns and len(financial_data) >= periods + 1:
                current_equity = financial_data['shareholders_equity'].iloc[-1]
                past_equity = financial_data['shareholders_equity'].iloc[-(periods + 1)]
                if past_equity > 0:
                    factors['equity_growth'] = ((current_equity / past_equity) ** (1/periods) - 1) * 100
                else:
                    factors['equity_growth'] = 0
            
            # EBITDA增长率
            if 'ebitda' in financial_data.columns and len(financial_data) >= periods + 1:
                current_ebitda = financial_data['ebitda'].iloc[-1]
                past_ebitda = financial_data['ebitda'].iloc[-(periods + 1)]
                if past_ebitda > 0:
                    factors['ebitda_growth'] = ((current_ebitda / past_ebitda) ** (1/periods) - 1) * 100
                else:
                    factors['ebitda_growth'] = 0
            
            # 自由现金流增长率
            if 'free_cash_flow' in financial_data.columns and len(financial_data) >= periods + 1:
                current_fcf = financial_data['free_cash_flow'].iloc[-1]
                past_fcf = financial_data['free_cash_flow'].iloc[-(periods + 1)]
                if past_fcf > 0:
                    factors['fcf_growth'] = ((current_fcf / past_fcf) ** (1/periods) - 1) * 100
                else:
                    factors['fcf_growth'] = 0
            
            # 营业现金流增长率
            if 'operating_cash_flow' in financial_data.columns and len(financial_data) >= periods + 1:
                current_ocf = financial_data['operating_cash_flow'].iloc[-1]
                past_ocf = financial_data['operating_cash_flow'].iloc[-(periods + 1)]
                if past_ocf > 0:
                    factors['ocf_growth'] = ((current_ocf / past_ocf) ** (1/periods) - 1) * 100
                else:
                    factors['ocf_growth'] = 0
            
            # 股息增长率
            if 'dividend_per_share' in financial_data.columns and len(financial_data) >= periods + 1:
                current_dps = financial_data['dividend_per_share'].iloc[-1]
                past_dps = financial_data['dividend_per_share'].iloc[-(periods + 1)]
                if past_dps > 0:
                    factors['dividend_growth'] = ((current_dps / past_dps) ** (1/periods) - 1) * 100
                else:
                    factors['dividend_growth'] = 0
            
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
            
            # 资产负债率
            if 'total_liabilities' in financial_data.columns and 'total_assets' in financial_data.columns:
                latest_liabilities = financial_data['total_liabilities'].iloc[-1]
                latest_assets = financial_data['total_assets'].iloc[-1]
                if latest_assets > 0:
                    factors['debt_to_assets'] = (latest_liabilities / latest_assets) * 100
                else:
                    factors['debt_to_assets'] = 0
            
            # 负债权益比
            if 'total_debt' in financial_data.columns and 'shareholders_equity' in financial_data.columns:
                latest_debt = financial_data['total_debt'].iloc[-1]
                latest_equity = financial_data['shareholders_equity'].iloc[-1]
                if latest_equity > 0:
                    factors['debt_to_equity'] = (latest_debt / latest_equity) * 100
                else:
                    factors['debt_to_equity'] = 0
            
            # 流动比率
            if 'current_assets' in financial_data.columns and 'current_liabilities' in financial_data.columns:
                latest_ca = financial_data['current_assets'].iloc[-1]
                latest_cl = financial_data['current_liabilities'].iloc[-1]
                if latest_cl > 0:
                    factors['current_ratio'] = latest_ca / latest_cl
                else:
                    factors['current_ratio'] = 0
            
            # 速动比率
            if all(col in financial_data.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
                latest_ca = financial_data['current_assets'].iloc[-1]
                latest_inventory = financial_data['inventory'].iloc[-1]
                latest_cl = financial_data['current_liabilities'].iloc[-1]
                if latest_cl > 0:
                    factors['quick_ratio'] = (latest_ca - latest_inventory) / latest_cl
                else:
                    factors['quick_ratio'] = 0
            
            # 现金比率
            if 'cash_and_equivalents' in financial_data.columns and 'current_liabilities' in financial_data.columns:
                latest_cash = financial_data['cash_and_equivalents'].iloc[-1]
                latest_cl = financial_data['current_liabilities'].iloc[-1]
                if latest_cl > 0:
                    factors['cash_ratio'] = latest_cash / latest_cl
                else:
                    factors['cash_ratio'] = 0
            
            # 利息覆盖倍数
            if 'operating_income' in financial_data.columns and 'interest_expense' in financial_data.columns:
                latest_oi = financial_data['operating_income'].iloc[-1]
                latest_ie = financial_data['interest_expense'].iloc[-1]
                if latest_ie > 0:
                    factors['interest_coverage'] = latest_oi / latest_ie
                else:
                    factors['interest_coverage'] = 0
            
            # 存货周转率
            if 'cost_of_goods_sold' in financial_data.columns and 'inventory' in financial_data.columns:
                latest_cogs = financial_data['cost_of_goods_sold'].iloc[-1]
                latest_inventory = financial_data['inventory'].iloc[-1]
                if latest_inventory > 0:
                    factors['inventory_turnover'] = latest_cogs / latest_inventory
                else:
                    factors['inventory_turnover'] = 0
            
            # 应收账款周转率
            if 'total_revenue' in financial_data.columns and 'accounts_receivable' in financial_data.columns:
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                latest_ar = financial_data['accounts_receivable'].iloc[-1]
                if latest_ar > 0:
                    factors['receivables_turnover'] = latest_revenue / latest_ar
                else:
                    factors['receivables_turnover'] = 0
            
            # 应付账款周转率
            if 'cost_of_goods_sold' in financial_data.columns and 'accounts_payable' in financial_data.columns:
                latest_cogs = financial_data['cost_of_goods_sold'].iloc[-1]
                latest_ap = financial_data['accounts_payable'].iloc[-1]
                if latest_ap > 0:
                    factors['payables_turnover'] = latest_cogs / latest_ap
                else:
                    factors['payables_turnover'] = 0
            
            # 现金转换周期
            if all(factor in factors for factor in ['inventory_turnover', 'receivables_turnover', 'payables_turnover']):
                if factors['inventory_turnover'] > 0 and factors['receivables_turnover'] > 0 and factors['payables_turnover'] > 0:
                    days_inventory = 365 / factors['inventory_turnover']
                    days_receivables = 365 / factors['receivables_turnover']
                    days_payables = 365 / factors['payables_turnover']
                    factors['cash_conversion_cycle'] = days_inventory + days_receivables - days_payables
                else:
                    factors['cash_conversion_cycle'] = 0
            
            # 现金流质量
            if 'operating_cash_flow' in financial_data.columns and 'net_income' in financial_data.columns:
                latest_ocf = financial_data['operating_cash_flow'].iloc[-1]
                latest_ni = financial_data['net_income'].iloc[-1]
                if latest_ni > 0:
                    factors['cash_flow_quality'] = latest_ocf / latest_ni
                else:
                    factors['cash_flow_quality'] = 0
            
            # 应计项目比率
            if 'net_income' in financial_data.columns and 'operating_cash_flow' in financial_data.columns and 'total_assets' in financial_data.columns:
                latest_ni = financial_data['net_income'].iloc[-1]
                latest_ocf = financial_data['operating_cash_flow'].iloc[-1]
                latest_assets = financial_data['total_assets'].iloc[-1]
                if latest_assets > 0:
                    factors['accruals_ratio'] = ((latest_ni - latest_ocf) / latest_assets) * 100
                else:
                    factors['accruals_ratio'] = 0
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算质量因子时出错: {e}")
            return {}

    def calculate_leverage_factors(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算杠杆因子
        
        Args:
            financial_data: 财务数据
            
        Returns:
            Dict[str, float]: 杠杆因子字典
        """
        try:
            factors = {}
            
            # 财务杠杆
            if 'total_assets' in financial_data.columns and 'shareholders_equity' in financial_data.columns:
                latest_assets = financial_data['total_assets'].iloc[-1]
                latest_equity = financial_data['shareholders_equity'].iloc[-1]
                if latest_equity > 0:
                    factors['financial_leverage'] = latest_assets / latest_equity
                else:
                    factors['financial_leverage'] = 0
            
            # 长期债务比率
            if 'long_term_debt' in financial_data.columns and 'total_assets' in financial_data.columns:
                latest_ltd = financial_data['long_term_debt'].iloc[-1]
                latest_assets = financial_data['total_assets'].iloc[-1]
                if latest_assets > 0:
                    factors['long_term_debt_ratio'] = (latest_ltd / latest_assets) * 100
                else:
                    factors['long_term_debt_ratio'] = 0
            
            # 净债务比率
            if all(col in financial_data.columns for col in ['total_debt', 'cash_and_equivalents', 'shareholders_equity']):
                latest_debt = financial_data['total_debt'].iloc[-1]
                latest_cash = financial_data['cash_and_equivalents'].iloc[-1]
                latest_equity = financial_data['shareholders_equity'].iloc[-1]
                net_debt = latest_debt - latest_cash
                if latest_equity > 0:
                    factors['net_debt_to_equity'] = (net_debt / latest_equity) * 100
                else:
                    factors['net_debt_to_equity'] = 0
            
            # EBITDA债务比率
            if 'total_debt' in financial_data.columns and 'ebitda' in financial_data.columns:
                latest_debt = financial_data['total_debt'].iloc[-1]
                latest_ebitda = financial_data['ebitda'].iloc[-1]
                if latest_ebitda > 0:
                    factors['debt_to_ebitda'] = latest_debt / latest_ebitda
                else:
                    factors['debt_to_ebitda'] = 0
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算杠杆因子时出错: {e}")
            return {}

    def calculate_efficiency_factors(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算效率因子
        
        Args:
            financial_data: 财务数据
            
        Returns:
            Dict[str, float]: 效率因子字典
        """
        try:
            factors = {}
            
            # 固定资产周转率
            if 'total_revenue' in financial_data.columns and 'fixed_assets' in financial_data.columns:
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                latest_fa = financial_data['fixed_assets'].iloc[-1]
                if latest_fa > 0:
                    factors['fixed_asset_turnover'] = latest_revenue / latest_fa
                else:
                    factors['fixed_asset_turnover'] = 0
            
            # 营运资本周转率
            if all(col in financial_data.columns for col in ['total_revenue', 'current_assets', 'current_liabilities']):
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                latest_ca = financial_data['current_assets'].iloc[-1]
                latest_cl = financial_data['current_liabilities'].iloc[-1]
                working_capital = latest_ca - latest_cl
                if working_capital > 0:
                    factors['working_capital_turnover'] = latest_revenue / working_capital
                else:
                    factors['working_capital_turnover'] = 0
            
            # 总资本周转率
            if 'total_revenue' in financial_data.columns and 'total_capital' in financial_data.columns:
                latest_revenue = financial_data['total_revenue'].iloc[-1]
                latest_tc = financial_data['total_capital'].iloc[-1]
                if latest_tc > 0:
                    factors['total_capital_turnover'] = latest_revenue / latest_tc
                else:
                    factors['total_capital_turnover'] = 0
            
            return factors
            
        except Exception as e:
            self.logger.error(f"计算效率因子时出错: {e}")
            return {}

    def calculate_piotroski_score(self, financial_data: pd.DataFrame) -> float:
        """
        计算增强版Piotroski F-Score
        
        Args:
            financial_data: 财务数据
            
        Returns:
            float: Piotroski F-Score (0-9)
        """
        try:
            score = 0
            
            # 盈利能力指标 (4分)
            # 1. 正净利润
            if 'net_income' in financial_data.columns:
                if financial_data['net_income'].iloc[-1] > 0:
                    score += 1
            
            # 2. 正经营现金流
            if 'operating_cash_flow' in financial_data.columns:
                if financial_data['operating_cash_flow'].iloc[-1] > 0:
                    score += 1
            
            # 3. ROA改善
            if 'roa' in financial_data.columns and len(financial_data) >= 2:
                if financial_data['roa'].iloc[-1] > financial_data['roa'].iloc[-2]:
                    score += 1
            
            # 4. 经营现金流 > 净利润
            if 'operating_cash_flow' in financial_data.columns and 'net_income' in financial_data.columns:
                if financial_data['operating_cash_flow'].iloc[-1] > financial_data['net_income'].iloc[-1]:
                    score += 1
            
            # 杠杆、流动性和资金来源指标 (3分)
            # 5. 长期债务减少
            if 'long_term_debt' in financial_data.columns and len(financial_data) >= 2:
                if financial_data['long_term_debt'].iloc[-1] < financial_data['long_term_debt'].iloc[-2]:
                    score += 1
            
            # 6. 流动比率改善
            if 'current_ratio' in financial_data.columns and len(financial_data) >= 2:
                if financial_data['current_ratio'].iloc[-1] > financial_data['current_ratio'].iloc[-2]:
                    score += 1
            
            # 7. 无新股发行
            if 'shares_outstanding' in financial_data.columns and len(financial_data) >= 2:
                if financial_data['shares_outstanding'].iloc[-1] <= financial_data['shares_outstanding'].iloc[-2]:
                    score += 1
            
            # 运营效率指标 (2分)
            # 8. 毛利率改善
            if 'gross_margin' in financial_data.columns and len(financial_data) >= 2:
                if financial_data['gross_margin'].iloc[-1] > financial_data['gross_margin'].iloc[-2]:
                    score += 1
            
            # 9. 资产周转率改善
            if 'asset_turnover' in financial_data.columns and len(financial_data) >= 2:
                if financial_data['asset_turnover'].iloc[-1] > financial_data['asset_turnover'].iloc[-2]:
                    score += 1
            
            return float(score)
            
        except Exception as e:
            self.logger.error(f"计算Piotroski F-Score时出错: {e}")
            return 0.0

    def calculate_altman_z_score(self, financial_data: pd.DataFrame) -> float:
        """
        计算增强版Altman Z-Score
        
        Args:
            financial_data: 财务数据
            
        Returns:
            float: Altman Z-Score
        """
        try:
            z_score = 0.0
            
            # Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
            
            # A: 营运资本/总资产
            if all(col in financial_data.columns for col in ['current_assets', 'current_liabilities', 'total_assets']):
                working_capital = financial_data['current_assets'].iloc[-1] - financial_data['current_liabilities'].iloc[-1]
                total_assets = financial_data['total_assets'].iloc[-1]
                if total_assets > 0:
                    z_score += 1.2 * (working_capital / total_assets)
            
            # B: 留存收益/总资产
            if 'retained_earnings' in financial_data.columns and 'total_assets' in financial_data.columns:
                retained_earnings = financial_data['retained_earnings'].iloc[-1]
                total_assets = financial_data['total_assets'].iloc[-1]
                if total_assets > 0:
                    z_score += 1.4 * (retained_earnings / total_assets)
            
            # C: 息税前利润/总资产
            if 'operating_income' in financial_data.columns and 'total_assets' in financial_data.columns:
                operating_income = financial_data['operating_income'].iloc[-1]
                total_assets = financial_data['total_assets'].iloc[-1]
                if total_assets > 0:
                    z_score += 3.3 * (operating_income / total_assets)
            
            # D: 股权市值/总负债
            if 'market_cap' in financial_data.columns and 'total_liabilities' in financial_data.columns:
                market_cap = financial_data['market_cap'].iloc[-1]
                total_liabilities = financial_data['total_liabilities'].iloc[-1]
                if total_liabilities > 0:
                    z_score += 0.6 * (market_cap / total_liabilities)
            
            # E: 销售收入/总资产
            if 'total_revenue' in financial_data.columns and 'total_assets' in financial_data.columns:
                total_revenue = financial_data['total_revenue'].iloc[-1]
                total_assets = financial_data['total_assets'].iloc[-1]
                if total_assets > 0:
                    z_score += 1.0 * (total_revenue / total_assets)
            
            return z_score
            
        except Exception as e:
            self.logger.error(f"计算Altman Z-Score时出错: {e}")
            return 0.0

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
            market_cap: 市值
            
        Returns:
            Dict[str, Dict[str, float]]: 所有因子的字典
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


    def calculate_advanced_valuation_factors(self, 
                                           price_data: pd.DataFrame,
                                           financial_data: pd.DataFrame,
                                           market_cap: Optional[float] = None) -> Dict[str, float]:
        """
        计算高级估值因子
        """
        factors = {}
        
        try:
            current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
            
            # PEG比率 (Price/Earnings to Growth)
            if 'eps' in financial_data.columns and len(financial_data) >= 4:
                latest_eps = financial_data['eps'].iloc[-1]
                eps_growth = self._calculate_growth_rate(financial_data['eps'], periods=4)
                
                if latest_eps > 0 and eps_growth > 0:
                    pe_ratio = current_price / latest_eps
                    factors['peg_ratio'] = pe_ratio / (eps_growth * 100)
                else:
                    factors['peg_ratio'] = np.nan
            
            # 动态市盈率 (Forward P/E)
            if 'forward_eps' in financial_data.columns:
                forward_eps = financial_data['forward_eps'].iloc[-1]
                if forward_eps > 0:
                    factors['forward_pe'] = current_price / forward_eps
                else:
                    factors['forward_pe'] = np.nan
            
            # 相对估值指标
            factors.update(self._calculate_relative_valuation(financial_data, current_price))
            
            # 内在价值相关指标
            factors.update(self._calculate_intrinsic_value_factors(financial_data, current_price))
            
        except Exception as e:
            self.logger.error(f"计算高级估值因子时出错: {e}")
        
        return factors
    
    def _calculate_relative_valuation(self, financial_data: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """计算相对估值指标"""
        factors = {}
        
        # 相对于历史均值的估值水平
        if 'eps' in financial_data.columns and len(financial_data) >= 8:
            eps_series = financial_data['eps'].dropna()
            if len(eps_series) > 0:
                pe_series = current_price / eps_series
                factors['pe_percentile'] = (pe_series.iloc[-1] <= pe_series).mean()
        
        # 相对于行业的估值水平（需要行业数据）
        if 'industry_pe' in financial_data.columns:
            industry_pe = financial_data['industry_pe'].iloc[-1]
            if 'eps' in financial_data.columns:
                latest_eps = financial_data['eps'].iloc[-1]
                if latest_eps > 0 and industry_pe > 0:
                    current_pe = current_price / latest_eps
                    factors['relative_pe'] = current_pe / industry_pe
        
        return factors
    
    def _calculate_intrinsic_value_factors(self, financial_data: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """计算内在价值相关因子"""
        factors = {}
        
        # 股息贴现模型相关
        if 'dividend_per_share' in financial_data.columns:
            dividend_series = financial_data['dividend_per_share'].dropna()
            if len(dividend_series) >= 3:
                dividend_growth = self._calculate_growth_rate(dividend_series, periods=3)
                current_dividend = dividend_series.iloc[-1]
                
                # 假设贴现率为10%
                discount_rate = 0.10
                if dividend_growth > 0 and dividend_growth < discount_rate:
                    gordon_value = current_dividend * (1 + dividend_growth) / (discount_rate - dividend_growth)
                    factors['gordon_model_ratio'] = current_price / gordon_value
        
        # 净现值相关指标
        if 'free_cash_flow' in financial_data.columns:
            fcf_series = financial_data['free_cash_flow'].dropna()
            if len(fcf_series) >= 3:
                fcf_growth = self._calculate_growth_rate(fcf_series, periods=3)
                current_fcf = fcf_series.iloc[-1]
                
                # DCF估值
                if fcf_growth > 0:
                    terminal_growth = 0.03  # 假设永续增长率3%
                    discount_rate = 0.10
                    
                    # 简化DCF计算
                    if fcf_growth < discount_rate:
                        dcf_value = current_fcf * (1 + fcf_growth) / (discount_rate - terminal_growth)
                        factors['dcf_ratio'] = current_price / dcf_value if dcf_value > 0 else np.nan
        
        return factors
    
    def calculate_earnings_quality_factors(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算盈利质量因子
        """
        factors = {}
        
        try:
            # 应计项目质量
            if all(col in financial_data.columns for col in ['net_income', 'operating_cash_flow']):
                net_income = financial_data['net_income'].iloc[-1]
                operating_cf = financial_data['operating_cash_flow'].iloc[-1]
                
                if net_income != 0:
                    factors['accruals_ratio'] = (net_income - operating_cf) / abs(net_income)
                else:
                    factors['accruals_ratio'] = np.nan
            
            # 盈利持续性
            if 'net_income' in financial_data.columns and len(financial_data) >= 8:
                ni_series = financial_data['net_income'].dropna()
                if len(ni_series) >= 4:
                    # 计算盈利的变异系数
                    factors['earnings_stability'] = ni_series.std() / abs(ni_series.mean()) if ni_series.mean() != 0 else np.nan
                    
                    # 盈利增长的一致性
                    growth_rates = ni_series.pct_change().dropna()
                    if len(growth_rates) > 0:
                        factors['earnings_growth_consistency'] = 1 - (growth_rates.std() / abs(growth_rates.mean())) if growth_rates.mean() != 0 else np.nan
            
            # 盈利预测准确性（如果有预测数据）
            if all(col in financial_data.columns for col in ['actual_eps', 'forecast_eps']):
                actual_eps = financial_data['actual_eps'].iloc[-1]
                forecast_eps = financial_data['forecast_eps'].iloc[-1]
                
                if forecast_eps != 0:
                    factors['earnings_surprise'] = (actual_eps - forecast_eps) / abs(forecast_eps)
                else:
                    factors['earnings_surprise'] = np.nan
            
            # 收入质量
            factors.update(self._calculate_revenue_quality(financial_data))
            
        except Exception as e:
            self.logger.error(f"计算盈利质量因子时出错: {e}")
        
        return factors
    
    def _calculate_revenue_quality(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """计算收入质量指标"""
        factors = {}
        
        # 收入增长的可持续性
        if 'total_revenue' in financial_data.columns and len(financial_data) >= 8:
            revenue_series = financial_data['total_revenue'].dropna()
            if len(revenue_series) >= 4:
                revenue_growth = revenue_series.pct_change().dropna()
                if len(revenue_growth) > 0:
                    factors['revenue_growth_stability'] = 1 - (revenue_growth.std() / abs(revenue_growth.mean())) if revenue_growth.mean() != 0 else np.nan
        
        # 应收账款质量
        if all(col in financial_data.columns for col in ['accounts_receivable', 'total_revenue']):
            ar_current = financial_data['accounts_receivable'].iloc[-1]
            revenue_current = financial_data['total_revenue'].iloc[-1]
            
            if len(financial_data) >= 2:
                ar_previous = financial_data['accounts_receivable'].iloc[-2]
                revenue_previous = financial_data['total_revenue'].iloc[-2]
                
                if revenue_current != 0 and revenue_previous != 0:
                    ar_turnover_current = revenue_current / ar_current if ar_current != 0 else np.inf
                    ar_turnover_previous = revenue_previous / ar_previous if ar_previous != 0 else np.inf
                    
                    factors['ar_turnover_change'] = (ar_turnover_current - ar_turnover_previous) / ar_turnover_previous if ar_turnover_previous != 0 else np.nan
        
        return factors
    
    def _calculate_growth_rate(self, series: pd.Series, periods: int = 4) -> float:
        """计算复合增长率"""
        if len(series) < periods + 1:
            return np.nan
        
        start_value = series.iloc[-periods-1]
        end_value = series.iloc[-1]
        
        if start_value <= 0:
            return np.nan
        
        growth_rate = (end_value / start_value) ** (1/periods) - 1
        return growth_rate


    def calculate_financial_strength_factors(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算财务实力因子
        """
        factors = {}
        
        try:
            # 资本结构稳定性
            factors.update(self._calculate_capital_structure_stability(financial_data))
            
            # 现金流稳定性
            factors.update(self._calculate_cash_flow_stability(financial_data))
            
            # 利息覆盖能力趋势
            factors.update(self._calculate_interest_coverage_trends(financial_data))
            
            # 营运资本管理效率
            factors.update(self._calculate_working_capital_efficiency(financial_data))
            
        except Exception as e:
            self.logger.error(f"计算财务实力因子时出错: {e}")
        
        return factors
    
    def _calculate_capital_structure_stability(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """计算资本结构稳定性指标"""
        factors = {}
        
        # 债务权益比稳定性
        if all(col in financial_data.columns for col in ['total_debt', 'total_equity']) and len(financial_data) >= 8:
            debt_equity_series = financial_data['total_debt'] / financial_data['total_equity']
            debt_equity_series = debt_equity_series.dropna()
            
            if len(debt_equity_series) >= 4:
                factors['debt_equity_stability'] = 1 - (debt_equity_series.std() / abs(debt_equity_series.mean())) if debt_equity_series.mean() != 0 else np.nan
        
        # 资产负债率趋势
        if all(col in financial_data.columns for col in ['total_liabilities', 'total_assets']) and len(financial_data) >= 4:
            debt_ratio_series = financial_data['total_liabilities'] / financial_data['total_assets']
            debt_ratio_series = debt_ratio_series.dropna()
            
            if len(debt_ratio_series) >= 2:
                # 计算趋势斜率
                x = np.arange(len(debt_ratio_series))
                if len(x) > 1:
                    slope, _, _, _, _ = stats.linregress(x, debt_ratio_series)
                    factors['debt_ratio_trend'] = slope
        
        return factors
    
    def _calculate_cash_flow_stability(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """计算现金流稳定性指标"""
        factors = {}
        
        # 经营现金流稳定性
        if 'operating_cash_flow' in financial_data.columns and len(financial_data) >= 8:
            ocf_series = financial_data['operating_cash_flow'].dropna()
            if len(ocf_series) >= 4:
                factors['ocf_stability'] = 1 - (ocf_series.std() / abs(ocf_series.mean())) if ocf_series.mean() != 0 else np.nan
                
                # 经营现金流增长趋势
                ocf_growth = ocf_series.pct_change().dropna()
                if len(ocf_growth) > 0:
                    factors['ocf_growth_trend'] = ocf_growth.mean()
        
        # 自由现金流稳定性
        if 'free_cash_flow' in financial_data.columns and len(financial_data) >= 8:
            fcf_series = financial_data['free_cash_flow'].dropna()
            if len(fcf_series) >= 4:
                factors['fcf_stability'] = 1 - (fcf_series.std() / abs(fcf_series.mean())) if fcf_series.mean() != 0 else np.nan
        
        # 现金流覆盖比率
        if all(col in financial_data.columns for col in ['operating_cash_flow', 'capital_expenditure', 'dividends_paid']):
            ocf = financial_data['operating_cash_flow'].iloc[-1]
            capex = financial_data['capital_expenditure'].iloc[-1]
            dividends = financial_data['dividends_paid'].iloc[-1]
            
            cash_obligations = abs(capex) + abs(dividends)
            if cash_obligations > 0:
                factors['cash_coverage_ratio'] = ocf / cash_obligations
        
        return factors
    
    def _calculate_interest_coverage_trends(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """计算利息覆盖能力趋势"""
        factors = {}
        
        # 利息覆盖倍数趋势
        if all(col in financial_data.columns for col in ['ebit', 'interest_expense']) and len(financial_data) >= 4:
            interest_coverage = financial_data['ebit'] / financial_data['interest_expense'].replace(0, np.nan)
            interest_coverage = interest_coverage.dropna()
            
            if len(interest_coverage) >= 2:
                # 计算趋势
                x = np.arange(len(interest_coverage))
                if len(x) > 1:
                    slope, _, _, _, _ = stats.linregress(x, interest_coverage)
                    factors['interest_coverage_trend'] = slope
                
                # 当前利息覆盖倍数
                factors['current_interest_coverage'] = interest_coverage.iloc[-1]
        
        # 现金流利息覆盖倍数
        if all(col in financial_data.columns for col in ['operating_cash_flow', 'interest_expense']):
            ocf = financial_data['operating_cash_flow'].iloc[-1]
            interest_exp = financial_data['interest_expense'].iloc[-1]
            
            if interest_exp > 0:
                factors['cash_interest_coverage'] = ocf / interest_exp
        
        return factors
    
    def _calculate_working_capital_efficiency(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """计算营运资本管理效率"""
        factors = {}
        
        # 营运资本周转率
        if all(col in financial_data.columns for col in ['total_revenue', 'current_assets', 'current_liabilities']):
            revenue = financial_data['total_revenue'].iloc[-1]
            current_assets = financial_data['current_assets'].iloc[-1]
            current_liabilities = financial_data['current_liabilities'].iloc[-1]
            
            working_capital = current_assets - current_liabilities
            if working_capital != 0:
                factors['working_capital_turnover'] = revenue / working_capital
        
        # 应收账款周转天数趋势
        if all(col in financial_data.columns for col in ['accounts_receivable', 'total_revenue']) and len(financial_data) >= 4:
            ar_turnover_days = (financial_data['accounts_receivable'] / financial_data['total_revenue']) * 365
            ar_turnover_days = ar_turnover_days.dropna()
            
            if len(ar_turnover_days) >= 2:
                x = np.arange(len(ar_turnover_days))
                if len(x) > 1:
                    slope, _, _, _, _ = stats.linregress(x, ar_turnover_days)
                    factors['ar_days_trend'] = slope  # 负值表示改善
        
        # 存货周转天数趋势
        if all(col in financial_data.columns for col in ['inventory', 'cost_of_goods_sold']) and len(financial_data) >= 4:
            inventory_days = (financial_data['inventory'] / financial_data['cost_of_goods_sold']) * 365
            inventory_days = inventory_days.dropna()
            
            if len(inventory_days) >= 2:
                x = np.arange(len(inventory_days))
                if len(x) > 1:
                    slope, _, _, _, _ = stats.linregress(x, inventory_days)
                    factors['inventory_days_trend'] = slope  # 负值表示改善
        
        # 应付账款周转天数趋势
        if all(col in financial_data.columns for col in ['accounts_payable', 'cost_of_goods_sold']) and len(financial_data) >= 4:
            ap_days = (financial_data['accounts_payable'] / financial_data['cost_of_goods_sold']) * 365
            ap_days = ap_days.dropna()
            
            if len(ap_days) >= 2:
                x = np.arange(len(ap_days))
                if len(x) > 1:
                    slope, _, _, _, _ = stats.linregress(x, ap_days)
                    factors['ap_days_trend'] = slope  # 正值表示改善
        
        # 现金转换周期
        if all(col in financial_data.columns for col in ['accounts_receivable', 'inventory', 'accounts_payable', 'total_revenue', 'cost_of_goods_sold']):
            ar_days = (financial_data['accounts_receivable'].iloc[-1] / financial_data['total_revenue'].iloc[-1]) * 365
            inv_days = (financial_data['inventory'].iloc[-1] / financial_data['cost_of_goods_sold'].iloc[-1]) * 365
            ap_days = (financial_data['accounts_payable'].iloc[-1] / financial_data['cost_of_goods_sold'].iloc[-1]) * 365
            
            factors['cash_conversion_cycle'] = ar_days + inv_days - ap_days
        
        return factors


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