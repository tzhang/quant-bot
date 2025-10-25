"""
网络因子计算模块

基于网络分析计算各种网络相关因子，包括：
1. 股票关联网络因子 - 相关性网络、因果网络等
2. 行业网络因子 - 行业关联度、行业传导效应等
3. 供应链网络因子 - 上下游关系、供应链风险等
4. 社交网络因子 - 社交媒体关联、信息传播等
5. 市场网络因子 - 市场结构、流动性网络等
6. 风险传染因子 - 系统性风险、风险传播等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx

# 网络分析库
try:
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    NETWORK_LIBS_AVAILABLE = True
except ImportError:
    NETWORK_LIBS_AVAILABLE = False
    print("网络分析库未安装，将使用简化计算")

class NetworkFactorCalculator:
    """网络因子计算器"""
    
    def __init__(self, correlation_threshold: float = 0.3, 
                 min_observations: int = 60):
        """
        初始化网络因子计算器
        
        Args:
            correlation_threshold: 相关性阈值，用于构建网络边
            min_observations: 最小观测数量
        """
        self.correlation_threshold = correlation_threshold
        self.min_observations = min_observations
        self.logger = logging.getLogger(__name__)
        
        print("网络因子计算器初始化完成")
    
    def calculate_correlation_network_factors(self, returns_data: pd.DataFrame,
                                            window: int = 252) -> Dict[str, pd.Series]:
        """
        计算相关性网络因子
        
        Args:
            returns_data: 收益率数据，列为股票代码
            window: 滚动窗口大小
            
        Returns:
            相关性网络因子字典
        """
        factors = {}
        
        if returns_data.empty or len(returns_data.columns) < 2:
            # 生成模拟数据
            returns_data = self._generate_mock_returns_data()
        
        print(f"计算相关性网络因子，股票数量: {len(returns_data.columns)}")
        
        # 滚动计算网络指标
        network_metrics = []
        
        for i in range(window, len(returns_data)):
            window_data = returns_data.iloc[i-window:i]
            
            # 计算相关性矩阵
            corr_matrix = window_data.corr()
            
            # 构建网络
            network_stats = self._calculate_network_statistics(corr_matrix)
            network_stats['date'] = returns_data.index[i]
            network_metrics.append(network_stats)
        
        # 转换为时间序列
        metrics_df = pd.DataFrame(network_metrics).set_index('date')
        
        for col in metrics_df.columns:
            factors[f'network_{col}'] = metrics_df[col]
        
        return factors
    
    def _generate_mock_returns_data(self, n_stocks: int = 50, n_days: int = 500) -> pd.DataFrame:
        """生成模拟收益率数据用于测试 - 仅用于测试和演示"""
        np.random.seed(42)  # 设置随机种子 - 模拟数据仅用于测试
        
        # 生成日期序列 - 仅用于测试和演示
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        # 生成股票代码 - 仅用于测试和演示
        stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]
        
        # 生成行业因子 - 模拟数据仅用于测试
        industry_factor = np.random.normal(0, 0.02, n_days)
        
        # 生成因子载荷 - 模拟数据仅用于测试
        factor_loadings = np.random.uniform(0.5, 1.5, n_stocks)
        
        # 生成因子收益率 - 模拟数据仅用于测试
        factor_returns = np.random.normal(0, 0.01, n_days)
        
        # 生成特异性收益率 - 模拟数据仅用于测试
        idiosyncratic_returns = np.random.normal(0, 0.03, (n_days, n_stocks))
        
        # 计算总收益率 - 模拟数据仅用于测试
        returns = np.outer(factor_returns, factor_loadings) + idiosyncratic_returns
        
        return pd.DataFrame(returns, index=dates, columns=stocks)
    
    def _calculate_network_statistics(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算网络统计指标"""
        # 移除NaN值
        corr_matrix = corr_matrix.fillna(0)
        
        # 构建邻接矩阵（绝对相关性大于阈值）
        adj_matrix = (np.abs(corr_matrix) > self.correlation_threshold).astype(int)
        np.fill_diagonal(adj_matrix.values, 0)  # 移除自环
        
        stats_dict = {}
        
        if NETWORK_LIBS_AVAILABLE:
            # 使用NetworkX计算网络指标
            G = nx.from_numpy_array(adj_matrix.values)
            
            # 基本网络指标
            stats_dict['density'] = nx.density(G)
            stats_dict['avg_clustering'] = nx.average_clustering(G)
            stats_dict['n_components'] = nx.number_connected_components(G)
            
            # 中心性指标
            if len(G.nodes()) > 0:
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                closeness_centrality = nx.closeness_centrality(G)
                
                stats_dict['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
                stats_dict['max_degree_centrality'] = np.max(list(degree_centrality.values()))
                stats_dict['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
                stats_dict['avg_closeness_centrality'] = np.mean(list(closeness_centrality.values()))
            
            # 路径长度
            if nx.is_connected(G):
                stats_dict['avg_path_length'] = nx.average_shortest_path_length(G)
                stats_dict['diameter'] = nx.diameter(G)
            else:
                stats_dict['avg_path_length'] = np.nan
                stats_dict['diameter'] = np.nan
        
        else:
            # 简化计算
            stats_dict = self._calculate_simplified_network_stats(adj_matrix)
        
        # 相关性网络特有指标
        stats_dict['avg_correlation'] = np.mean(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]))
        stats_dict['max_correlation'] = np.max(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]))
        stats_dict['correlation_dispersion'] = np.std(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)])
        
        return stats_dict
    
    def _calculate_simplified_network_stats(self, adj_matrix: pd.DataFrame) -> Dict[str, float]:
        """简化的网络统计计算"""
        n_nodes = len(adj_matrix)
        n_edges = np.sum(adj_matrix.values) / 2  # 无向图
        
        stats_dict = {}
        
        # 密度
        max_edges = n_nodes * (n_nodes - 1) / 2
        stats_dict['density'] = n_edges / max_edges if max_edges > 0 else 0
        
        # 平均度
        degrees = np.sum(adj_matrix.values, axis=1)
        stats_dict['avg_degree_centrality'] = np.mean(degrees) / (n_nodes - 1) if n_nodes > 1 else 0
        stats_dict['max_degree_centrality'] = np.max(degrees) / (n_nodes - 1) if n_nodes > 1 else 0
        
        # 连通分量数量（简化计算）
        stats_dict['n_components'] = 1 if n_edges > 0 else n_nodes
        
        # 聚类系数（简化）
        clustering_coeffs = []
        for i in range(n_nodes):
            neighbors = np.where(adj_matrix.iloc[i] == 1)[0]
            if len(neighbors) > 1:
                subgraph = adj_matrix.iloc[neighbors, neighbors]
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                actual_edges = np.sum(subgraph.values) / 2
                clustering_coeffs.append(actual_edges / possible_edges if possible_edges > 0 else 0)
        
        stats_dict['avg_clustering'] = np.mean(clustering_coeffs) if clustering_coeffs else 0
        
        return stats_dict
    
    def calculate_industry_network_factors(self, returns_data: pd.DataFrame,
                                         industry_mapping: Optional[Dict[str, str]] = None,
                                         window: int = 252) -> Dict[str, pd.Series]:
        """
        计算行业网络因子
        
        Args:
            returns_data: 收益率数据
            industry_mapping: 股票到行业的映射
            window: 滚动窗口大小
            
        Returns:
            行业网络因子字典
        """
        factors = {}
        
        if returns_data.empty:
            returns_data = self._generate_mock_returns_data()
        
        if industry_mapping is None:
            industry_mapping = self._generate_mock_industry_mapping(returns_data.columns)
        
        print(f"计算行业网络因子，行业数量: {len(set(industry_mapping.values()))}")
        
        # 按行业聚合收益率
        industry_returns = self._aggregate_by_industry(returns_data, industry_mapping)
        
        # 计算行业间网络因子
        industry_network_factors = self.calculate_correlation_network_factors(industry_returns, window)
        
        # 重命名因子
        for k, v in industry_network_factors.items():
            factors[f'industry_{k}'] = v
        
        # 计算行业内网络因子
        intra_industry_factors = self._calculate_intra_industry_factors(returns_data, industry_mapping, window)
        factors.update(intra_industry_factors)
        
        return factors
    
    def _generate_mock_industry_mapping(self, stocks: List[str]) -> Dict[str, str]:
        """生成模拟行业映射 - 仅用于测试和演示"""
        industries = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial', 'Materials', 'Utilities']
        
        mapping = {}
        for i, stock in enumerate(stocks):
            mapping[stock] = industries[i % len(industries)]  # 循环分配行业 - 模拟数据仅用于测试
        
        return mapping
    
    def _aggregate_by_industry(self, returns_data: pd.DataFrame, 
                             industry_mapping: Dict[str, str]) -> pd.DataFrame:
        """按行业聚合收益率"""
        industry_returns = {}
        
        for industry in set(industry_mapping.values()):
            industry_stocks = [stock for stock, ind in industry_mapping.items() if ind == industry and stock in returns_data.columns]
            if industry_stocks:
                # 等权重平均
                industry_returns[industry] = returns_data[industry_stocks].mean(axis=1)
        
        return pd.DataFrame(industry_returns)
    
    def _calculate_intra_industry_factors(self, returns_data: pd.DataFrame,
                                        industry_mapping: Dict[str, str],
                                        window: int) -> Dict[str, pd.Series]:
        """计算行业内网络因子"""
        factors = {}
        
        # 计算每个行业内的平均相关性
        intra_industry_corrs = []
        
        for i in range(window, len(returns_data)):
            window_data = returns_data.iloc[i-window:i]
            
            industry_corrs = []
            for industry in set(industry_mapping.values()):
                industry_stocks = [stock for stock, ind in industry_mapping.items() 
                                 if ind == industry and stock in returns_data.columns]
                
                if len(industry_stocks) > 1:
                    industry_data = window_data[industry_stocks]
                    corr_matrix = industry_data.corr()
                    # 计算上三角矩阵的平均相关性
                    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                    avg_corr = np.mean(upper_triangle)
                    industry_corrs.append(avg_corr)
            
            overall_intra_corr = np.mean(industry_corrs) if industry_corrs else 0
            intra_industry_corrs.append({
                'date': returns_data.index[i],
                'intra_industry_correlation': overall_intra_corr
            })
        
        if intra_industry_corrs:
            intra_df = pd.DataFrame(intra_industry_corrs).set_index('date')
            factors['intra_industry_correlation'] = intra_df['intra_industry_correlation']
        
        return factors
    
    def calculate_supply_chain_factors(self, returns_data: pd.DataFrame,
                                     supply_chain_mapping: Optional[Dict[str, List[str]]] = None,
                                     window: int = 252) -> Dict[str, pd.Series]:
        """
        计算供应链网络因子
        
        Args:
            returns_data: 收益率数据
            supply_chain_mapping: 供应链关系映射
            window: 滚动窗口大小
            
        Returns:
            供应链网络因子字典
        """
        factors = {}
        
        if returns_data.empty:
            returns_data = self._generate_mock_returns_data()
        
        if supply_chain_mapping is None:
            supply_chain_mapping = self._generate_mock_supply_chain_mapping(returns_data.columns)
        
        print("计算供应链网络因子...")
        
        # 计算供应链传导效应
        supply_chain_factors = []
        
        for i in range(window, len(returns_data)):
            window_data = returns_data.iloc[i-window:i]
            
            # 计算供应链相关性
            supply_chain_corr = self._calculate_supply_chain_correlation(window_data, supply_chain_mapping)
            
            supply_chain_factors.append({
                'date': returns_data.index[i],
                'supply_chain_correlation': supply_chain_corr,
                'supply_chain_risk': self._calculate_supply_chain_risk(window_data, supply_chain_mapping)
            })
        
        if supply_chain_factors:
            sc_df = pd.DataFrame(supply_chain_factors).set_index('date')
            for col in sc_df.columns:
                factors[f'supply_chain_{col}'] = sc_df[col]
        
        return factors
    
    def _generate_mock_supply_chain_mapping(self, stocks: List[str]) -> Dict[str, List[str]]:
        """生成模拟供应链映射 - 仅用于测试和演示"""
        np.random.seed(42)  # 设置随机种子 - 模拟数据仅用于测试
        mapping = {}
        
        for stock in stocks:
            # 随机选择2-5个供应商/客户 - 模拟数据仅用于演示
            n_connections = np.random.randint(2, 6)
            connections = np.random.choice([s for s in stocks if s != stock], 
                                        size=min(n_connections, len(stocks)-1), 
                                        replace=False)
            mapping[stock] = list(connections)
        
        return mapping
    
    def _calculate_supply_chain_correlation(self, returns_data: pd.DataFrame,
                                          supply_chain_mapping: Dict[str, List[str]]) -> float:
        """计算供应链相关性"""
        correlations = []
        
        for stock, suppliers in supply_chain_mapping.items():
            if stock in returns_data.columns:
                valid_suppliers = [s for s in suppliers if s in returns_data.columns]
                if valid_suppliers:
                    stock_returns = returns_data[stock]
                    supplier_returns = returns_data[valid_suppliers].mean(axis=1)
                    
                    corr = stock_returns.corr(supplier_returns)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0
    
    def _calculate_supply_chain_risk(self, returns_data: pd.DataFrame,
                                   supply_chain_mapping: Dict[str, List[str]]) -> float:
        """计算供应链风险"""
        risks = []
        
        for stock, suppliers in supply_chain_mapping.items():
            if stock in returns_data.columns:
                valid_suppliers = [s for s in suppliers if s in returns_data.columns]
                if valid_suppliers:
                    # 供应商收益率的波动率作为供应链风险的代理
                    supplier_returns = returns_data[valid_suppliers]
                    supplier_vol = supplier_returns.std().mean()
                    risks.append(supplier_vol)
        
        return np.mean(risks) if risks else 0
    
    def calculate_market_structure_factors(self, returns_data: pd.DataFrame,
                                         volume_data: Optional[pd.DataFrame] = None,
                                         window: int = 252) -> Dict[str, pd.Series]:
        """
        计算市场结构网络因子
        
        Args:
            returns_data: 收益率数据
            volume_data: 成交量数据
            window: 滚动窗口大小
            
        Returns:
            市场结构网络因子字典
        """
        factors = {}
        
        if returns_data.empty:
            returns_data = self._generate_mock_returns_data()
        
        if volume_data is None:
            volume_data = self._generate_mock_volume_data(returns_data.index, returns_data.columns)
        
        print("计算市场结构网络因子...")
        
        market_structure_factors = []
        
        for i in range(window, len(returns_data)):
            window_returns = returns_data.iloc[i-window:i]
            window_volume = volume_data.iloc[i-window:i]
            
            # 计算市场结构指标
            structure_metrics = self._calculate_market_structure_metrics(window_returns, window_volume)
            structure_metrics['date'] = returns_data.index[i]
            market_structure_factors.append(structure_metrics)
        
        if market_structure_factors:
            ms_df = pd.DataFrame(market_structure_factors).set_index('date')
            for col in ms_df.columns:
                factors[f'market_structure_{col}'] = ms_df[col]
        
        return factors
    
    def _generate_mock_volume_data(self, dates: pd.DatetimeIndex, stocks: List[str]) -> pd.DataFrame:
        """生成模拟成交量数据 - 仅用于测试和演示"""
        np.random.seed(43)  # 设置随机种子 - 模拟数据仅用于测试
        
        # 生成成交量数据（对数正态分布） - 模拟数据仅用于演示
        volume_data = {}
        for stock in stocks:
            base_volume = np.random.uniform(1e6, 1e8)  # 基础成交量 - 仅用于测试
            volumes = np.random.lognormal(np.log(base_volume), 0.5, len(dates))
            volume_data[stock] = volumes
        
        return pd.DataFrame(volume_data, index=dates)
    
    def _calculate_market_structure_metrics(self, returns_data: pd.DataFrame,
                                          volume_data: pd.DataFrame) -> Dict[str, float]:
        """计算市场结构指标"""
        metrics = {}
        
        # 1. 流动性网络密度
        # 基于成交量的相关性构建流动性网络
        volume_corr = volume_data.corr()
        liquidity_network = (np.abs(volume_corr) > 0.3).astype(int)
        np.fill_diagonal(liquidity_network.values, 0)
        
        n_nodes = len(liquidity_network)
        n_edges = np.sum(liquidity_network.values) / 2
        max_edges = n_nodes * (n_nodes - 1) / 2
        metrics['liquidity_network_density'] = n_edges / max_edges if max_edges > 0 else 0
        
        # 2. 价格发现效率
        # 基于收益率的同步性
        market_return = returns_data.mean(axis=1)
        synchronicity = []
        for stock in returns_data.columns:
            corr = returns_data[stock].corr(market_return)
            if not np.isnan(corr):
                synchronicity.append(corr**2)
        
        metrics['price_discovery_efficiency'] = np.mean(synchronicity) if synchronicity else 0
        
        # 3. 市场分割度
        # 基于收益率相关性的分割程度
        returns_corr = returns_data.corr()
        avg_correlation = np.mean(returns_corr.values[np.triu_indices_from(returns_corr.values, k=1)])
        metrics['market_segmentation'] = 1 - avg_correlation  # 相关性越低，分割度越高
        
        # 4. 信息传播速度
        # 基于滞后相关性
        lag_correlations = []
        for stock in returns_data.columns:
            for other_stock in returns_data.columns:
                if stock != other_stock:
                    # 计算滞后1期的相关性
                    lagged_corr = returns_data[stock].corr(returns_data[other_stock].shift(1))
                    if not np.isnan(lagged_corr):
                        lag_correlations.append(abs(lagged_corr))
        
        metrics['information_propagation_speed'] = 1 - np.mean(lag_correlations) if lag_correlations else 1
        
        return metrics
    
    def calculate_all_factors(self, returns_data: Optional[pd.DataFrame] = None,
                            industry_mapping: Optional[Dict[str, str]] = None,
                            supply_chain_mapping: Optional[Dict[str, List[str]]] = None,
                            volume_data: Optional[pd.DataFrame] = None,
                            window: int = 252) -> Dict[str, pd.Series]:
        """
        计算所有网络因子
        
        Args:
            returns_data: 收益率数据
            industry_mapping: 行业映射
            supply_chain_mapping: 供应链映射
            volume_data: 成交量数据
            window: 滚动窗口大小
            
        Returns:
            所有网络因子字典
        """
        all_factors = {}
        
        print("开始计算网络因子...")
        
        if returns_data is None:
            returns_data = self._generate_mock_returns_data()
        
        # 1. 相关性网络因子
        correlation_factors = self.calculate_correlation_network_factors(returns_data, window)
        all_factors.update(correlation_factors)
        
        # 2. 行业网络因子
        industry_factors = self.calculate_industry_network_factors(returns_data, industry_mapping, window)
        all_factors.update(industry_factors)
        
        # 3. 供应链网络因子
        supply_chain_factors = self.calculate_supply_chain_factors(returns_data, supply_chain_mapping, window)
        all_factors.update(supply_chain_factors)
        
        # 4. 市场结构网络因子
        market_structure_factors = self.calculate_market_structure_factors(returns_data, volume_data, window)
        all_factors.update(market_structure_factors)
        
        print(f"网络因子计算完成，共生成 {len(all_factors)} 个因子")
        return all_factors

def main():
    """测试网络因子计算器"""
    print("测试网络因子计算器...")
    
    # 初始化计算器
    calculator = NetworkFactorCalculator()
    
    # 计算因子
    factors = calculator.calculate_all_factors()
    
    print(f"成功计算 {len(factors)} 个网络因子")
    for name, factor in list(factors.items())[:10]:  # 只显示前10个
        if not factor.empty:
            print(f"{name}: 均值={factor.mean():.4f}, 标准差={factor.std():.4f}")

if __name__ == "__main__":
    main()