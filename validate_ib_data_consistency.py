#!/usr/bin/env python3
"""
IB数据一致性验证脚本

该脚本用于验证Interactive Brokers数据与其他数据源（yfinance、qlib等）的一致性。
比较价格数据、成交量等关键指标的差异。

作者: AI Assistant
日期: 2024
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.ib_data_provider import IBDataProvider, create_ib_provider
from src.data.qlib_data_provider import QlibDataProvider
import yfinance as yf

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataConsistencyValidator:
    """数据一致性验证器"""
    
    def __init__(self):
        """初始化数据提供者"""
        self.ib_provider = create_ib_provider()
        self.qlib_provider = QlibDataProvider()
        
        # 测试股票列表
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        # 数据一致性阈值
        self.price_tolerance = 0.05  # 5% 价格差异容忍度
        self.volume_tolerance = 0.10  # 10% 成交量差异容忍度
        
    def get_data_from_all_sources(self, symbol: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """从所有数据源获取数据"""
        data_sources = {}
        
        # IB数据
        try:
            if self.ib_provider.is_available:
                ib_data = self.ib_provider.get_stock_data(symbol, start_date, end_date)
                if not ib_data.empty:
                    data_sources['IB'] = ib_data
                    logger.info(f"✅ IB数据获取成功: {symbol}, {len(ib_data)} 条记录")
                else:
                    logger.warning(f"⚠️ IB数据为空: {symbol}")
            else:
                logger.warning("⚠️ IB数据源不可用")
        except Exception as e:
            logger.error(f"❌ IB数据获取失败: {symbol}, 错误: {e}")
        
        # yfinance数据
        try:
            ticker = yf.Ticker(symbol)
            # 使用period参数替代start/end日期以避免日期范围问题
            yf_data = ticker.history(period='1mo')
            
            if not yf_data.empty:
                # 标准化列名为小写
                yf_data.columns = [col.lower() for col in yf_data.columns]
                data_sources['yfinance'] = yf_data
                logger.info(f"✅ yfinance数据获取成功: {symbol}, {len(yf_data)} 条记录")
            else:
                logger.warning(f"⚠️ yfinance数据为空: {symbol}")
        except Exception as e:
            logger.error(f"❌ yfinance数据获取失败: {symbol}, 错误: {e}")
        
        # qlib数据
        try:
            qlib_data = self.qlib_provider.get_stock_data(symbol, start_date, end_date)
            if not qlib_data.empty:
                data_sources['qlib'] = qlib_data
                logger.info(f"✅ qlib数据获取成功: {symbol}, {len(qlib_data)} 条记录")
            else:
                logger.warning(f"⚠️ qlib数据为空: {symbol}")
        except Exception as e:
            logger.error(f"❌ qlib数据获取失败: {symbol}, 错误: {e}")
        
        return data_sources
    
    def compare_price_data(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                          source1: str, source2: str) -> Dict[str, float]:
        """比较两个数据源的价格数据"""
        # 标准化索引为日期（移除时区信息）
        data1_normalized = data1.copy()
        data2_normalized = data2.copy()
        
        # 标准化索引
        if hasattr(data1_normalized.index, 'tz_localize'):
            data1_normalized.index = data1_normalized.index.tz_localize(None)
        if hasattr(data2_normalized.index, 'tz_localize'):
            data2_normalized.index = data2_normalized.index.tz_localize(None)
        
        # 转换为日期索引
        data1_normalized.index = pd.to_datetime(data1_normalized.index.date)
        data2_normalized.index = pd.to_datetime(data2_normalized.index.date)
        
        # 找到共同的日期
        common_dates = data1_normalized.index.intersection(data2_normalized.index)
        
        if len(common_dates) == 0:
            return {
                'common_dates': 0,
                'close_correlation': 0.0,
                'close_mean_diff_pct': 100.0,
                'volume_correlation': 0.0,
                'volume_mean_diff_pct': 100.0
            }
        
        # 获取共同日期的数据
        df1_common = data1_normalized.loc[common_dates]
        df2_common = data2_normalized.loc[common_dates]
        
        # 价格比较
        close_corr = df1_common['close'].corr(df2_common['close'])
        close_diff_pct = abs((df1_common['close'] - df2_common['close']) / df1_common['close'] * 100).mean()
        
        # 成交量比较
        volume_corr = 0.0
        volume_diff_pct = 100.0
        
        if 'volume' in df1_common.columns and 'volume' in df2_common.columns:
            # 过滤掉零成交量的数据
            valid_volume = (df1_common['volume'] > 0) & (df2_common['volume'] > 0)
            if valid_volume.sum() > 0:
                volume_corr = df1_common.loc[valid_volume, 'volume'].corr(
                    df2_common.loc[valid_volume, 'volume']
                )
                volume_diff_pct = abs(
                    (df1_common.loc[valid_volume, 'volume'] - df2_common.loc[valid_volume, 'volume']) / 
                    df1_common.loc[valid_volume, 'volume'] * 100
                ).mean()
        
        return {
            'common_dates': len(common_dates),
            'close_correlation': close_corr if not np.isnan(close_corr) else 0.0,
            'close_mean_diff_pct': close_diff_pct if not np.isnan(close_diff_pct) else 100.0,
            'volume_correlation': volume_corr if not np.isnan(volume_corr) else 0.0,
            'volume_mean_diff_pct': volume_diff_pct if not np.isnan(volume_diff_pct) else 100.0
        }
    
    def validate_symbol_consistency(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """验证单个股票的数据一致性"""
        logger.info(f"开始验证 {symbol} 的数据一致性...")
        
        # 获取所有数据源的数据
        data_sources = self.get_data_from_all_sources(symbol, start_date, end_date)
        
        if len(data_sources) < 2:
            logger.warning(f"⚠️ {symbol} 可用数据源不足2个，无法进行一致性验证")
            return {
                'symbol': symbol,
                'available_sources': list(data_sources.keys()),
                'comparisons': {},
                'overall_consistency': 'insufficient_data'
            }
        
        # 进行两两比较
        comparisons = {}
        source_names = list(data_sources.keys())
        
        for i in range(len(source_names)):
            for j in range(i + 1, len(source_names)):
                source1, source2 = source_names[i], source_names[j]
                comparison_key = f"{source1}_vs_{source2}"
                
                comparison_result = self.compare_price_data(
                    data_sources[source1], 
                    data_sources[source2],
                    source1, 
                    source2
                )
                
                comparisons[comparison_key] = comparison_result
                
                logger.info(f"📊 {comparison_key}: "
                          f"共同日期={comparison_result['common_dates']}, "
                          f"价格相关性={comparison_result['close_correlation']:.3f}, "
                          f"价格差异={comparison_result['close_mean_diff_pct']:.2f}%")
        
        # 评估整体一致性
        overall_consistency = self._evaluate_overall_consistency(comparisons)
        
        return {
            'symbol': symbol,
            'available_sources': source_names,
            'comparisons': comparisons,
            'overall_consistency': overall_consistency
        }
    
    def _evaluate_overall_consistency(self, comparisons: Dict) -> str:
        """评估整体一致性"""
        if not comparisons:
            return 'no_comparisons'
        
        # 计算平均指标
        avg_correlation = np.mean([comp['close_correlation'] for comp in comparisons.values()])
        avg_price_diff = np.mean([comp['close_mean_diff_pct'] for comp in comparisons.values()])
        
        # 一致性评级
        if avg_correlation > 0.95 and avg_price_diff < self.price_tolerance * 100:
            return 'excellent'
        elif avg_correlation > 0.90 and avg_price_diff < self.price_tolerance * 200:
            return 'good'
        elif avg_correlation > 0.80 and avg_price_diff < self.price_tolerance * 400:
            return 'fair'
        else:
            return 'poor'
    
    def run_full_validation(self, start_date: str = None, end_date: str = None) -> Dict:
        """运行完整的数据一致性验证"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info("=" * 80)
        logger.info("🔍 开始IB数据一致性验证")
        logger.info("=" * 80)
        logger.info(f"📅 验证时间范围: {start_date} 到 {end_date}")
        logger.info(f"📈 测试股票: {', '.join(self.test_symbols)}")
        
        validation_results = {}
        consistency_summary = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0,
            'insufficient_data': 0
        }
        
        for symbol in self.test_symbols:
            try:
                result = self.validate_symbol_consistency(symbol, start_date, end_date)
                validation_results[symbol] = result
                consistency_summary[result['overall_consistency']] += 1
                
            except Exception as e:
                logger.error(f"❌ {symbol} 验证失败: {e}")
                validation_results[symbol] = {
                    'symbol': symbol,
                    'error': str(e),
                    'overall_consistency': 'error'
                }
        
        # 生成报告
        self._generate_consistency_report(validation_results, consistency_summary)
        
        return {
            'validation_results': validation_results,
            'consistency_summary': consistency_summary,
            'test_period': f"{start_date} to {end_date}"
        }
    
    def _generate_consistency_report(self, results: Dict, summary: Dict):
        """生成一致性验证报告"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 IB数据一致性验证报告")
        logger.info("=" * 80)
        
        # 总体统计
        total_stocks = len(results)
        logger.info(f"📈 测试股票总数: {total_stocks}")
        
        for level, count in summary.items():
            if count > 0:
                percentage = count / total_stocks * 100
                logger.info(f"   {level.upper()}: {count} 只股票 ({percentage:.1f}%)")
        
        # 详细结果
        logger.info("\n📋 详细验证结果:")
        for symbol, result in results.items():
            if 'error' in result:
                logger.info(f"   ❌ {symbol}: 验证失败 - {result['error']}")
                continue
                
            consistency = result['overall_consistency']
            sources = ', '.join(result['available_sources'])
            
            if consistency == 'excellent':
                emoji = "🟢"
            elif consistency == 'good':
                emoji = "🟡"
            elif consistency == 'fair':
                emoji = "🟠"
            else:
                emoji = "🔴"
            
            logger.info(f"   {emoji} {symbol}: {consistency.upper()} (数据源: {sources})")
            
            # 显示比较详情
            for comp_name, comp_data in result.get('comparisons', {}).items():
                logger.info(f"      └─ {comp_name}: "
                          f"相关性={comp_data['close_correlation']:.3f}, "
                          f"价格差异={comp_data['close_mean_diff_pct']:.2f}%")
        
        logger.info("=" * 80)
        
        # 给出建议
        if summary['excellent'] + summary['good'] >= total_stocks * 0.8:
            logger.info("✅ IB数据质量优秀，可以作为主要数据源使用")
        elif summary['excellent'] + summary['good'] + summary['fair'] >= total_stocks * 0.6:
            logger.info("⚠️ IB数据质量良好，建议与其他数据源结合使用")
        else:
            logger.info("❌ IB数据质量存在问题，建议检查配置或使用其他数据源")


def main():
    """主函数"""
    try:
        validator = DataConsistencyValidator()
        
        # 运行验证
        results = validator.run_full_validation()
        
        logger.info("✅ IB数据一致性验证完成")
        
    except Exception as e:
        logger.error(f"❌ 验证过程中发生错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())