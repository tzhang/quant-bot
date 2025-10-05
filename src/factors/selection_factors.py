"""
择股因子计算模块

实现各种择股相关的因子，包括：
1. 多因子选股模型
2. 因子评价体系
3. 因子合成与权重分配
4. 行业中性化处理
5. 因子有效性检验
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectionFactorCalculator:
    """
    择股因子计算器
    
    提供多因子选股模型的构建和评估功能
    """
    
    def __init__(self):
        """初始化择股因子计算器"""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
    
    def calculate_quality_factors(self, financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算质量因子
        
        Args:
            financial_data: 财务数据，包含各种财务指标
            
        Returns:
            Dict[str, pd.Series]: 质量因子字典
        """
        factors = {}
        
        try:
            # 1. 盈利质量因子
            factors.update(self._calculate_profitability_quality(financial_data))
            
            # 2. 财务稳健性因子
            factors.update(self._calculate_financial_stability(financial_data))
            
            # 3. 经营效率因子
            factors.update(self._calculate_operational_efficiency(financial_data))
            
            # 4. 现金流质量因子
            factors.update(self._calculate_cashflow_quality(financial_data))
            
            self.logger.info(f"成功计算 {len(factors)} 个质量因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算质量因子时出错: {str(e)}")
            return {}
    
    def calculate_value_factors(self, price_data: pd.DataFrame, 
                              financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算价值因子
        
        Args:
            price_data: 价格数据
            financial_data: 财务数据
            
        Returns:
            Dict[str, pd.Series]: 价值因子字典
        """
        factors = {}
        
        try:
            # 1. 传统价值因子
            factors.update(self._calculate_traditional_value(price_data, financial_data))
            
            # 2. 现金流价值因子
            factors.update(self._calculate_cashflow_value(price_data, financial_data))
            
            # 3. 资产价值因子
            factors.update(self._calculate_asset_value(price_data, financial_data))
            
            # 4. 相对价值因子
            factors.update(self._calculate_relative_value(price_data, financial_data))
            
            self.logger.info(f"成功计算 {len(factors)} 个价值因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算价值因子时出错: {str(e)}")
            return {}
    
    def calculate_growth_factors(self, financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算成长因子
        
        Args:
            financial_data: 财务数据
            
        Returns:
            Dict[str, pd.Series]: 成长因子字典
        """
        factors = {}
        
        try:
            # 1. 收入成长因子
            factors.update(self._calculate_revenue_growth(financial_data))
            
            # 2. 盈利成长因子
            factors.update(self._calculate_earnings_growth(financial_data))
            
            # 3. 资产成长因子
            factors.update(self._calculate_asset_growth(financial_data))
            
            # 4. 成长稳定性因子
            factors.update(self._calculate_growth_stability(financial_data))
            
            self.logger.info(f"成功计算 {len(factors)} 个成长因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算成长因子时出错: {str(e)}")
            return {}
    
    def calculate_momentum_factors(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算动量因子
        
        Args:
            price_data: 价格数据
            
        Returns:
            Dict[str, pd.Series]: 动量因子字典
        """
        factors = {}
        
        try:
            # 1. 价格动量因子
            factors.update(self._calculate_price_momentum(price_data))
            
            # 2. 盈利动量因子
            factors.update(self._calculate_earnings_momentum(price_data))
            
            # 3. 分析师预期动量
            factors.update(self._calculate_analyst_momentum(price_data))
            
            # 4. 技术动量因子
            factors.update(self._calculate_technical_momentum(price_data))
            
            self.logger.info(f"成功计算 {len(factors)} 个动量因子")
            return factors
            
        except Exception as e:
            self.logger.error(f"计算动量因子时出错: {str(e)}")
            return {}
    
    def build_multifactor_model(self, factor_data: Dict[str, pd.DataFrame],
                               returns_data: pd.DataFrame,
                               factor_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        构建多因子选股模型
        
        Args:
            factor_data: 各类因子数据字典
            returns_data: 股票收益率数据
            factor_weights: 因子权重（可选）
            
        Returns:
            Dict[str, Any]: 多因子模型结果
        """
        try:
            # 1. 因子预处理
            processed_factors = self._preprocess_factors(factor_data)
            
            # 2. 因子有效性检验
            factor_validity = self._validate_factors(processed_factors, returns_data)
            
            # 3. 因子权重计算
            if factor_weights is None:
                factor_weights = self._calculate_factor_weights(processed_factors, returns_data)
            
            # 4. 因子合成
            composite_factor = self._composite_factors(processed_factors, factor_weights)
            
            # 5. 模型评估
            model_performance = self._evaluate_model_performance(composite_factor, returns_data)
            
            # 6. 选股结果
            stock_selection = self._generate_stock_selection(composite_factor)
            
            result = {
                'processed_factors': processed_factors,
                'factor_validity': factor_validity,
                'factor_weights': factor_weights,
                'composite_factor': composite_factor,
                'model_performance': model_performance,
                'stock_selection': stock_selection
            }
            
            self.logger.info("多因子选股模型构建完成")
            return result
            
        except Exception as e:
            self.logger.error(f"构建多因子模型时出错: {str(e)}")
            return {}
    
    def evaluate_factor_effectiveness(self, factor_values: pd.Series,
                                    returns: pd.Series,
                                    periods: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """
        评估单个因子的有效性
        
        Args:
            factor_values: 因子值
            returns: 股票收益率
            periods: 评估周期列表
            
        Returns:
            Dict[str, Any]: 因子评估结果
        """
        try:
            results = {}
            
            for period in periods:
                # 计算前瞻收益
                forward_returns = returns.shift(-period)
                
                # 去除缺失值
                valid_data = pd.concat([factor_values, forward_returns], axis=1).dropna()
                if len(valid_data) < 30:  # 数据量太少
                    continue
                
                factor_vals = valid_data.iloc[:, 0]
                future_rets = valid_data.iloc[:, 1]
                
                # 1. IC分析
                ic_pearson = factor_vals.corr(future_rets)
                ic_spearman = factor_vals.corr(future_rets, method='spearman')
                
                # 2. 分层测试
                quantile_analysis = self._quantile_analysis(factor_vals, future_rets)
                
                # 3. 信息比率
                ic_series = self._rolling_ic(factor_vals, future_rets)
                ic_ir = ic_series.mean() / (ic_series.std() + 1e-8)
                
                # 4. 胜率分析
                win_rate = (ic_series > 0).mean()
                
                results[f'period_{period}'] = {
                    'ic_pearson': ic_pearson,
                    'ic_spearman': ic_spearman,
                    'ic_ir': ic_ir,
                    'win_rate': win_rate,
                    'quantile_analysis': quantile_analysis,
                    'ic_series': ic_series
                }
            
            # 综合评分
            overall_score = self._calculate_factor_score(results)
            results['overall_score'] = overall_score
            
            return results
            
        except Exception as e:
            self.logger.error(f"评估因子有效性时出错: {str(e)}")
            return {}
    
    def industry_neutralization(self, factor_data: pd.DataFrame,
                              industry_data: pd.DataFrame) -> pd.DataFrame:
        """
        行业中性化处理
        
        Args:
            factor_data: 因子数据
            industry_data: 行业分类数据
            
        Returns:
            pd.DataFrame: 中性化后的因子数据
        """
        try:
            neutralized_factors = factor_data.copy()
            
            for factor_name in factor_data.columns:
                # 按行业分组进行中性化
                for industry in industry_data['industry'].unique():
                    industry_mask = industry_data['industry'] == industry
                    industry_stocks = industry_data[industry_mask].index
                    
                    # 获取该行业的因子值
                    industry_factors = factor_data.loc[industry_stocks, factor_name]
                    
                    # 行业内标准化
                    if len(industry_factors) > 1:
                        industry_mean = industry_factors.mean()
                        industry_std = industry_factors.std()
                        
                        if industry_std > 0:
                            neutralized_factors.loc[industry_stocks, factor_name] = (
                                industry_factors - industry_mean
                            ) / industry_std
            
            self.logger.info("行业中性化处理完成")
            return neutralized_factors
            
        except Exception as e:
            self.logger.error(f"行业中性化处理时出错: {str(e)}")
            return factor_data
    
    # ==================== 私有方法：具体因子计算实现 ====================
    
    def _calculate_profitability_quality(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算盈利质量因子"""
        factors = {}
        
        # ROE稳定性
        if 'roe' in data.columns:
            factors['roe_stability'] = -data['roe'].rolling(window=4).std()
        
        # 毛利率趋势
        if 'gross_margin' in data.columns:
            factors['gross_margin_trend'] = data['gross_margin'].diff(4)
        
        # 净利率质量
        if 'net_margin' in data.columns:
            factors['net_margin_quality'] = data['net_margin'].rolling(window=4).mean()
        
        return factors
    
    def _calculate_financial_stability(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算财务稳健性因子"""
        factors = {}
        
        # 负债率稳定性
        if 'debt_ratio' in data.columns:
            factors['debt_stability'] = -data['debt_ratio'].rolling(window=4).std()
        
        # 流动比率
        if 'current_ratio' in data.columns:
            factors['liquidity_quality'] = data['current_ratio']
        
        # 利息保障倍数
        if 'interest_coverage' in data.columns:
            factors['interest_coverage_quality'] = np.log1p(data['interest_coverage'].clip(lower=0))
        
        return factors
    
    def _calculate_operational_efficiency(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算经营效率因子"""
        factors = {}
        
        # 资产周转率
        if 'asset_turnover' in data.columns:
            factors['asset_efficiency'] = data['asset_turnover']
        
        # 存货周转率
        if 'inventory_turnover' in data.columns:
            factors['inventory_efficiency'] = data['inventory_turnover']
        
        # 应收账款周转率
        if 'receivables_turnover' in data.columns:
            factors['receivables_efficiency'] = data['receivables_turnover']
        
        return factors
    
    def _calculate_cashflow_quality(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算现金流质量因子"""
        factors = {}
        
        # 经营现金流与净利润比
        if 'operating_cashflow' in data.columns and 'net_income' in data.columns:
            factors['cashflow_quality'] = data['operating_cashflow'] / (data['net_income'] + 1e-8)
        
        # 自由现金流稳定性
        if 'free_cashflow' in data.columns:
            factors['fcf_stability'] = -data['free_cashflow'].rolling(window=4).std()
        
        return factors
    
    def _calculate_traditional_value(self, price_data: pd.DataFrame, 
                                   financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算传统价值因子"""
        factors = {}
        
        # PE倒数
        if 'pe_ratio' in financial_data.columns:
            factors['earnings_yield'] = 1 / (financial_data['pe_ratio'] + 1e-8)
        
        # PB倒数
        if 'pb_ratio' in financial_data.columns:
            factors['book_to_market'] = 1 / (financial_data['pb_ratio'] + 1e-8)
        
        # PS倒数
        if 'ps_ratio' in financial_data.columns:
            factors['sales_to_price'] = 1 / (financial_data['ps_ratio'] + 1e-8)
        
        return factors
    
    def _calculate_cashflow_value(self, price_data: pd.DataFrame,
                                financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算现金流价值因子"""
        factors = {}
        
        # PCF倒数
        if 'pcf_ratio' in financial_data.columns:
            factors['cashflow_yield'] = 1 / (financial_data['pcf_ratio'] + 1e-8)
        
        # 自由现金流收益率
        if 'free_cashflow_yield' in financial_data.columns:
            factors['fcf_yield'] = financial_data['free_cashflow_yield']
        
        return factors
    
    def _calculate_asset_value(self, price_data: pd.DataFrame,
                             financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算资产价值因子"""
        factors = {}
        
        # 净资产收益率调整的PB
        if 'pb_ratio' in financial_data.columns and 'roe' in financial_data.columns:
            factors['roe_adjusted_pb'] = financial_data['roe'] / (financial_data['pb_ratio'] + 1e-8)
        
        return factors
    
    def _calculate_relative_value(self, price_data: pd.DataFrame,
                                financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算相对价值因子"""
        factors = {}
        
        # 相对PE
        if 'pe_ratio' in financial_data.columns:
            market_pe = financial_data['pe_ratio'].median()
            factors['relative_pe'] = market_pe / (financial_data['pe_ratio'] + 1e-8)
        
        return factors
    
    def _calculate_revenue_growth(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算收入成长因子"""
        factors = {}
        
        # 营收增长率
        if 'revenue' in data.columns:
            factors['revenue_growth_1y'] = data['revenue'].pct_change(4)  # 年度增长
            factors['revenue_growth_3y'] = data['revenue'].pct_change(12)  # 三年增长
        
        return factors
    
    def _calculate_earnings_growth(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算盈利成长因子"""
        factors = {}
        
        # 净利润增长率
        if 'net_income' in data.columns:
            factors['earnings_growth_1y'] = data['net_income'].pct_change(4)
            factors['earnings_growth_3y'] = data['net_income'].pct_change(12)
        
        # EPS增长率
        if 'eps' in data.columns:
            factors['eps_growth_1y'] = data['eps'].pct_change(4)
        
        return factors
    
    def _calculate_asset_growth(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算资产成长因子"""
        factors = {}
        
        # 总资产增长率
        if 'total_assets' in data.columns:
            factors['asset_growth_1y'] = data['total_assets'].pct_change(4)
        
        # 净资产增长率
        if 'shareholders_equity' in data.columns:
            factors['equity_growth_1y'] = data['shareholders_equity'].pct_change(4)
        
        return factors
    
    def _calculate_growth_stability(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算成长稳定性因子"""
        factors = {}
        
        # 收入增长稳定性
        if 'revenue' in data.columns:
            revenue_growth = data['revenue'].pct_change(4)
            factors['revenue_growth_stability'] = -revenue_growth.rolling(window=4).std()
        
        return factors
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算价格动量因子"""
        factors = {}
        
        if 'close' in data.columns:
            # 多期动量
            factors['momentum_1m'] = data['close'].pct_change(20)
            factors['momentum_3m'] = data['close'].pct_change(60)
            factors['momentum_6m'] = data['close'].pct_change(120)
            factors['momentum_12m'] = data['close'].pct_change(240)
            
            # 动量强度
            factors['momentum_strength'] = (
                0.25 * factors['momentum_1m'] +
                0.25 * factors['momentum_3m'] +
                0.25 * factors['momentum_6m'] +
                0.25 * factors['momentum_12m']
            )
        
        return factors
    
    def _calculate_earnings_momentum(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算盈利动量因子"""
        factors = {}
        
        # 盈利预期修正
        if 'earnings_revision' in data.columns:
            factors['earnings_revision_momentum'] = data['earnings_revision'].rolling(window=20).mean()
        
        return factors
    
    def _calculate_analyst_momentum(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算分析师预期动量"""
        factors = {}
        
        # 分析师评级变化
        if 'analyst_rating' in data.columns:
            factors['rating_momentum'] = data['analyst_rating'].diff(20)
        
        return factors
    
    def _calculate_technical_momentum(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算技术动量因子"""
        factors = {}
        
        if 'close' in data.columns:
            # RSI动量
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / (avg_loss + 1e-8)
            factors['rsi_momentum'] = 100 - (100 / (1 + rs))
            
            # MACD动量
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            factors['macd_momentum'] = ema_12 - ema_26
        
        return factors
    
    def _preprocess_factors(self, factor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """因子预处理"""
        # 合并所有因子
        all_factors = pd.concat(factor_data.values(), axis=1)
        
        # 去极值（3倍标准差）
        for col in all_factors.columns:
            mean_val = all_factors[col].mean()
            std_val = all_factors[col].std()
            upper_bound = mean_val + 3 * std_val
            lower_bound = mean_val - 3 * std_val
            all_factors[col] = all_factors[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 标准化
        all_factors = pd.DataFrame(
            self.scaler.fit_transform(all_factors),
            index=all_factors.index,
            columns=all_factors.columns
        )
        
        return all_factors
    
    def _validate_factors(self, factors: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, float]:
        """验证因子有效性"""
        validity_scores = {}
        
        for factor_name in factors.columns:
            # 计算IC
            ic_scores = []
            for i in range(1, min(21, len(returns.columns))):  # 最多20期前瞻
                future_returns = returns.iloc[:, i] if i < len(returns.columns) else returns.iloc[:, -1]
                ic = factors[factor_name].corr(future_returns)
                if not np.isnan(ic):
                    ic_scores.append(abs(ic))
            
            validity_scores[factor_name] = np.mean(ic_scores) if ic_scores else 0
        
        return validity_scores
    
    def _calculate_factor_weights(self, factors: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, float]:
        """计算因子权重"""
        validity_scores = self._validate_factors(factors, returns)
        
        # 基于有效性分配权重
        total_score = sum(validity_scores.values())
        if total_score > 0:
            weights = {k: v / total_score for k, v in validity_scores.items()}
        else:
            # 等权重
            weights = {k: 1.0 / len(validity_scores) for k in validity_scores.keys()}
        
        return weights
    
    def _composite_factors(self, factors: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """合成因子"""
        composite = pd.Series(0, index=factors.index)
        
        for factor_name, weight in weights.items():
            if factor_name in factors.columns:
                composite += weight * factors[factor_name]
        
        return composite
    
    def _evaluate_model_performance(self, composite_factor: pd.Series, 
                                  returns: pd.DataFrame) -> Dict[str, float]:
        """评估模型性能"""
        performance = {}
        
        # IC分析
        if len(returns.columns) > 0:
            future_returns = returns.iloc[:, 0]  # 使用第一期收益
            ic = composite_factor.corr(future_returns)
            performance['ic'] = ic if not np.isnan(ic) else 0
        
        # 分层测试
        quantiles = pd.qcut(composite_factor.rank(), q=5, labels=False)
        quantile_returns = []
        
        for q in range(5):
            mask = quantiles == q
            if mask.sum() > 0 and len(returns.columns) > 0:
                q_return = returns.iloc[mask, 0].mean()
                quantile_returns.append(q_return)
        
        if len(quantile_returns) == 5:
            performance['long_short_return'] = quantile_returns[-1] - quantile_returns[0]
        else:
            performance['long_short_return'] = 0
        
        return performance
    
    def _generate_stock_selection(self, composite_factor: pd.Series, 
                                top_pct: float = 0.2) -> Dict[str, List]:
        """生成选股结果"""
        # 按因子值排序
        sorted_stocks = composite_factor.sort_values(ascending=False)
        
        # 选择前20%
        top_n = int(len(sorted_stocks) * top_pct)
        selected_stocks = sorted_stocks.head(top_n)
        
        return {
            'selected_stocks': selected_stocks.index.tolist(),
            'factor_scores': selected_stocks.values.tolist(),
            'selection_ratio': top_pct
        }
    
    def _quantile_analysis(self, factor_values: pd.Series, returns: pd.Series, 
                          n_quantiles: int = 5) -> Dict[str, float]:
        """分位数分析"""
        # 按因子值分组
        quantiles = pd.qcut(factor_values.rank(), q=n_quantiles, labels=False)
        
        quantile_returns = {}
        for q in range(n_quantiles):
            mask = quantiles == q
            if mask.sum() > 0:
                q_return = returns[mask].mean()
                quantile_returns[f'Q{q+1}'] = q_return
        
        # 多空收益
        if len(quantile_returns) == n_quantiles:
            quantile_returns['long_short'] = quantile_returns['Q5'] - quantile_returns['Q1']
        
        return quantile_returns
    
    def _rolling_ic(self, factor_values: pd.Series, returns: pd.Series, 
                   window: int = 20) -> pd.Series:
        """滚动IC计算"""
        ic_series = []
        
        for i in range(window, len(factor_values)):
            factor_window = factor_values.iloc[i-window:i]
            return_window = returns.iloc[i-window:i]
            ic = factor_window.corr(return_window)
            ic_series.append(ic if not np.isnan(ic) else 0)
        
        return pd.Series(ic_series)
    
    def _calculate_factor_score(self, results: Dict[str, Any]) -> float:
        """计算因子综合评分"""
        scores = []
        
        for period_result in results.values():
            if isinstance(period_result, dict):
                # IC绝对值
                ic_score = abs(period_result.get('ic_pearson', 0)) * 0.3
                
                # IC信息比率
                ir_score = abs(period_result.get('ic_ir', 0)) * 0.3
                
                # 胜率
                win_rate_score = period_result.get('win_rate', 0.5) * 0.2
                
                # 多空收益
                quantile_analysis = period_result.get('quantile_analysis', {})
                long_short = abs(quantile_analysis.get('long_short', 0)) * 0.2
                
                period_score = ic_score + ir_score + win_rate_score + long_short
                scores.append(period_score)
        
        return np.mean(scores) if scores else 0