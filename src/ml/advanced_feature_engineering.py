
# ==========================================
# 迁移说明 - 2025-10-10 23:06:36
# ==========================================
# 本文件已从yfinance迁移到IB TWS API
# 原始文件备份在: backup_before_ib_migration/src/ml/advanced_feature_engineering.py
# 
# 主要变更:
# # - 替换yfinance导入为IB导入
# - 检测到yf.download()调用，需要手动调整
# 
# 注意事项:
# 1. 需要启动IB TWS或Gateway
# 2. 确保API设置已正确配置
# 3. 某些yfinance特有功能可能需要手动调整
# ==========================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征工程扩展模块

提供更多高级特征工程技术，包括：
- 高阶统计特征
- 频域分析特征  
- 图论网络特征
- 宏观经济特征
- 非线性特征
- 时频分析特征
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class HighOrderStatisticalFeatures:
    """高阶统计特征工程"""
    
    def __init__(self):
        pass
    
    def add_moment_features(self, 
                           df: pd.DataFrame, 
                           columns: List[str], 
                           windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """添加高阶矩特征"""
        result_df = df.copy()
        
        for col in columns:
            for window in windows:
                # 三阶矩（偏度）
                result_df[f'{col}_skewness_{window}'] = df[col].rolling(window).skew()
                
                # 四阶矩（峰度）
                result_df[f'{col}_kurtosis_{window}'] = df[col].rolling(window).kurt()
                
                # 五阶矩
                result_df[f'{col}_moment5_{window}'] = df[col].rolling(window).apply(
                    lambda x: stats.moment(x, moment=5, nan_policy='omit')
                )
                
                # 六阶矩
                result_df[f'{col}_moment6_{window}'] = df[col].rolling(window).apply(
                    lambda x: stats.moment(x, moment=6, nan_policy='omit')
                )
                
        return result_df
    
    def add_distribution_tests(self, 
                              df: pd.DataFrame, 
                              columns: List[str], 
                              windows: List[int] = [20, 50]) -> pd.DataFrame:
        """添加分布检验特征"""
        result_df = df.copy()
        
        for col in columns:
            for window in windows:
                # Jarque-Bera正态性检验
                result_df[f'{col}_jb_stat_{window}'] = df[col].rolling(window).apply(
                    lambda x: stats.jarque_bera(x.dropna())[0] if len(x.dropna()) > 8 else np.nan
                )
                
                # Anderson-Darling检验
                result_df[f'{col}_ad_stat_{window}'] = df[col].rolling(window).apply(
                    lambda x: stats.anderson(x.dropna())[0] if len(x.dropna()) > 8 else np.nan
                )
                
                # Shapiro-Wilk检验
                result_df[f'{col}_sw_stat_{window}'] = df[col].rolling(window).apply(
                    lambda x: stats.shapiro(x.dropna())[0] if 3 <= len(x.dropna()) <= 5000 else np.nan
                )
                
        return result_df
    
    def add_entropy_features(self, 
                            df: pd.DataFrame, 
                            columns: List[str], 
                            windows: List[int] = [20, 50]) -> pd.DataFrame:
        """添加熵特征"""
        result_df = df.copy()
        
        for col in columns:
            for window in windows:
                # Shannon熵
                result_df[f'{col}_shannon_entropy_{window}'] = df[col].rolling(window).apply(
                    self._calculate_shannon_entropy
                )
                
                # 近似熵
                result_df[f'{col}_approx_entropy_{window}'] = df[col].rolling(window).apply(
                    lambda x: self._calculate_approximate_entropy(x.values, m=2, r=0.2)
                )
                
        return result_df
    
    def _calculate_shannon_entropy(self, x):
        """计算Shannon熵"""
        try:
            x = x.dropna()
            if len(x) < 2:
                return np.nan
            
            # 离散化
            bins = min(10, len(x) // 2)
            counts, _ = np.histogram(x, bins=bins)
            counts = counts[counts > 0]
            
            # 计算概率
            probs = counts / counts.sum()
            
            # 计算熵
            entropy = -np.sum(probs * np.log2(probs))
            return entropy
        except:
            return np.nan
    
    def _calculate_approximate_entropy(self, data, m, r):
        """计算近似熵"""
        try:
            data = data[~np.isnan(data)]
            N = len(data)
            
            if N < m + 1:
                return np.nan
            
            def _maxdist(xi, xj, N, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], N, m) <= r:
                            C[i] += 1.0
                
                phi = (N - m + 1.0) ** (-1) * sum(np.log(C / (N - m + 1.0)))
                return phi
            
            return _phi(m) - _phi(m + 1)
        except:
            return np.nan


class FrequencyDomainFeatures:
    """频域分析特征"""
    
    def __init__(self):
        pass
    
    def add_fft_features(self, 
                        df: pd.DataFrame, 
                        columns: List[str], 
                        windows: List[int] = [50, 100]) -> pd.DataFrame:
        """添加FFT频域特征"""
        result_df = df.copy()
        
        for col in columns:
            for window in windows:
                # 主频率
                result_df[f'{col}_dominant_freq_{window}'] = df[col].rolling(window).apply(
                    self._get_dominant_frequency
                )
                
                # 频谱能量
                result_df[f'{col}_spectral_energy_{window}'] = df[col].rolling(window).apply(
                    self._get_spectral_energy
                )
                
                # 频谱重心
                result_df[f'{col}_spectral_centroid_{window}'] = df[col].rolling(window).apply(
                    self._get_spectral_centroid
                )
                
                # 频谱带宽
                result_df[f'{col}_spectral_bandwidth_{window}'] = df[col].rolling(window).apply(
                    self._get_spectral_bandwidth
                )
                
        return result_df
    
    def add_wavelet_features(self, 
                            df: pd.DataFrame, 
                            columns: List[str], 
                            windows: List[int] = [50, 100]) -> pd.DataFrame:
        """添加小波变换特征"""
        result_df = df.copy()
        
        try:
            import pywt
            
            for col in columns:
                for window in windows:
                    # 小波能量
                    result_df[f'{col}_wavelet_energy_{window}'] = df[col].rolling(window).apply(
                        lambda x: self._get_wavelet_energy(x.values)
                    )
                    
                    # 小波熵
                    result_df[f'{col}_wavelet_entropy_{window}'] = df[col].rolling(window).apply(
                        lambda x: self._get_wavelet_entropy(x.values)
                    )
                    
        except ImportError:
            print("PyWavelets not installed, skipping wavelet features")
            
        return result_df
    
    def _get_dominant_frequency(self, x):
        """获取主频率"""
        try:
            x = x.dropna().values
            if len(x) < 8:
                return np.nan
            
            # FFT
            fft_vals = fft(x)
            freqs = fftfreq(len(x))
            
            # 找到最大幅值对应的频率
            dominant_freq = freqs[np.argmax(np.abs(fft_vals[1:len(x)//2])) + 1]
            return abs(dominant_freq)
        except:
            return np.nan
    
    def _get_spectral_energy(self, x):
        """获取频谱能量"""
        try:
            x = x.dropna().values
            if len(x) < 8:
                return np.nan
            
            fft_vals = fft(x)
            energy = np.sum(np.abs(fft_vals) ** 2)
            return energy
        except:
            return np.nan
    
    def _get_spectral_centroid(self, x):
        """获取频谱重心"""
        try:
            x = x.dropna().values
            if len(x) < 8:
                return np.nan
            
            fft_vals = fft(x)
            freqs = fftfreq(len(x))
            magnitude = np.abs(fft_vals)
            
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return abs(centroid)
        except:
            return np.nan
    
    def _get_spectral_bandwidth(self, x):
        """获取频谱带宽"""
        try:
            x = x.dropna().values
            if len(x) < 8:
                return np.nan
            
            fft_vals = fft(x)
            freqs = fftfreq(len(x))
            magnitude = np.abs(fft_vals)
            
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
            return bandwidth
        except:
            return np.nan
    
    def _get_wavelet_energy(self, x):
        """获取小波能量"""
        try:
            import pywt
            if len(x) < 8:
                return np.nan
            
            coeffs = pywt.wavedec(x, 'db4', level=3)
            energy = sum([np.sum(c ** 2) for c in coeffs])
            return energy
        except:
            return np.nan
    
    def _get_wavelet_entropy(self, x):
        """获取小波熵"""
        try:
            import pywt
            if len(x) < 8:
                return np.nan
            
            coeffs = pywt.wavedec(x, 'db4', level=3)
            energies = [np.sum(c ** 2) for c in coeffs]
            total_energy = sum(energies)
            
            if total_energy == 0:
                return 0
            
            probs = [e / total_energy for e in energies if e > 0]
            entropy = -sum([p * np.log2(p) for p in probs])
            return entropy
        except:
            return np.nan


class NetworkGraphFeatures:
    """图论网络特征"""
    
    def __init__(self):
        pass
    
    def create_correlation_network_features(self, 
                                          df: pd.DataFrame, 
                                          columns: List[str], 
                                          windows: List[int] = [50, 100],
                                          threshold: float = 0.5) -> pd.DataFrame:
        """基于相关性网络的特征"""
        result_df = df.copy()
        
        for window in windows:
            # 滚动计算网络特征
            network_features = df[columns].rolling(window).apply(
                lambda x: self._calculate_network_features(x, threshold),
                raw=False
            )
            
            # 添加网络特征
            for i, feature_name in enumerate(['density', 'clustering', 'centrality_mean', 'path_length']):
                result_df[f'network_{feature_name}_{window}'] = network_features.iloc[:, 0] if len(network_features.columns) > i else np.nan
                
        return result_df
    
    def _calculate_network_features(self, corr_matrix, threshold):
        """计算网络特征"""
        try:
            if len(corr_matrix) < 10:
                return pd.Series([np.nan, np.nan, np.nan, np.nan])
            
            # 计算相关性矩阵
            corr = corr_matrix.corr()
            
            # 创建图
            G = nx.Graph()
            n = len(corr)
            
            # 添加边（相关性超过阈值）
            for i in range(n):
                for j in range(i+1, n):
                    if abs(corr.iloc[i, j]) > threshold:
                        G.add_edge(i, j, weight=abs(corr.iloc[i, j]))
            
            if len(G.edges()) == 0:
                return pd.Series([0, 0, 0, np.inf])
            
            # 计算网络特征
            density = nx.density(G)
            clustering = nx.average_clustering(G)
            
            # 中心性
            centrality = nx.degree_centrality(G)
            centrality_mean = np.mean(list(centrality.values()))
            
            # 平均路径长度
            if nx.is_connected(G):
                path_length = nx.average_shortest_path_length(G)
            else:
                # 对于非连通图，计算最大连通分量的平均路径长度
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                path_length = nx.average_shortest_path_length(subgraph) if len(subgraph) > 1 else 0
            
            return pd.Series([density, clustering, centrality_mean, path_length])
            
        except Exception as e:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])


class MacroEconomicFeatures:
    """宏观经济特征"""
    
    def __init__(self):
        self.macro_indicators = {
            'vix': 'VIX',  # 恐慌指数
            'dxy': 'DXY',  # 美元指数
            'tnx': '^TNX', # 10年期国债收益率
            'gold': 'GC=F', # 黄金期货
            'oil': 'CL=F',  # 原油期货
        }
    
    def add_macro_features(self, 
                          df: pd.DataFrame, 
                          date_column: str = 'date') -> pd.DataFrame:
        """添加宏观经济特征"""
        result_df = df.copy()
        
        try:
            from src.data.ib_data_provider import IBDataProvider, IBConfig
            
            # 获取日期范围
            start_date = df[date_column].min()
            end_date = df[date_column].max()
            
            for indicator_name, symbol in self.macro_indicators.items():
                try:
                    # 第一优先级：尝试使用IB TWS API获取宏观数据
                    try:
                        ib_provider = IBDataProvider(IBConfig())
                        macro_data = ib_provider.get_stock_data(symbol, start_date, end_date)
                    except Exception as e:
                        print(f"IB TWS API获取{symbol}数据失败: {e}")
                        # 第二优先级：回退到yfinance
                        macro_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    
                    if not macro_data.empty:
                        # 计算宏观指标特征
                        macro_features = self._calculate_macro_features(macro_data, indicator_name)
                        
                        # 合并到主数据框
                        macro_features.index = pd.to_datetime(macro_features.index)
                        result_df = result_df.set_index(date_column)
                        result_df = result_df.join(macro_features, how='left')
                        result_df = result_df.reset_index()
                        
                except Exception as e:
                    print(f"Failed to get {indicator_name} data: {e}")
                    continue
                    
        except ImportError:
            print("yfinance not installed, skipping macro features")
            
        return result_df
    
    def _calculate_macro_features(self, macro_data, indicator_name):
        """计算宏观指标特征"""
        features = pd.DataFrame(index=macro_data.index)
        
        close = macro_data['Close']
        
        # 价格水平
        features[f'{indicator_name}_level'] = close
        
        # 变化率
        features[f'{indicator_name}_pct_1d'] = close.pct_change(1)
        features[f'{indicator_name}_pct_5d'] = close.pct_change(5)
        features[f'{indicator_name}_pct_20d'] = close.pct_change(20)
        
        # 移动平均
        features[f'{indicator_name}_ma_20'] = close.rolling(20).mean()
        features[f'{indicator_name}_ma_50'] = close.rolling(50).mean()
        
        # 相对位置
        features[f'{indicator_name}_position_20'] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min())
        
        # 波动率
        features[f'{indicator_name}_vol_20'] = close.pct_change().rolling(20).std()
        
        return features


class NonLinearFeatures:
    """非线性特征"""
    
    def __init__(self):
        pass
    
    def add_manifold_features(self, 
                             df: pd.DataFrame, 
                             columns: List[str], 
                             windows: List[int] = [50, 100]) -> pd.DataFrame:
        """添加流形学习特征"""
        result_df = df.copy()
        
        for window in windows:
            # PCA特征
            pca_features = self._rolling_pca(df[columns], window)
            for i in range(min(3, len(columns))):
                result_df[f'pca_{i+1}_{window}'] = pca_features[:, i] if pca_features.shape[1] > i else np.nan
            
            # ICA特征
            ica_features = self._rolling_ica(df[columns], window)
            for i in range(min(3, len(columns))):
                result_df[f'ica_{i+1}_{window}'] = ica_features[:, i] if ica_features.shape[1] > i else np.nan
                
        return result_df
    
    def add_clustering_features(self, 
                               df: pd.DataFrame, 
                               columns: List[str], 
                               windows: List[int] = [50, 100],
                               n_clusters: int = 3) -> pd.DataFrame:
        """添加聚类特征"""
        result_df = df.copy()
        
        for window in windows:
            # K-means聚类标签
            cluster_labels = self._rolling_kmeans(df[columns], window, n_clusters)
            result_df[f'cluster_label_{window}'] = cluster_labels
            
            # 到聚类中心的距离
            cluster_distances = self._rolling_cluster_distance(df[columns], window, n_clusters)
            result_df[f'cluster_distance_{window}'] = cluster_distances
            
        return result_df
    
    def _rolling_pca(self, df, window):
        """滚动PCA"""
        n_components = min(3, df.shape[1])
        pca = PCA(n_components=n_components)
        
        result = np.full((len(df), n_components), np.nan)
        
        for i in range(window, len(df)):
            try:
                data = df.iloc[i-window:i].dropna()
                if len(data) >= window // 2 and data.shape[1] >= n_components:
                    transformed = pca.fit_transform(data)
                    result[i] = transformed[-1]  # 取最后一个点的变换结果
            except:
                continue
                
        return result
    
    def _rolling_ica(self, df, window):
        """滚动ICA"""
        n_components = min(3, df.shape[1])
        ica = FastICA(n_components=n_components, random_state=42)
        
        result = np.full((len(df), n_components), np.nan)
        
        for i in range(window, len(df)):
            try:
                data = df.iloc[i-window:i].dropna()
                if len(data) >= window // 2 and data.shape[1] >= n_components:
                    transformed = ica.fit_transform(data)
                    result[i] = transformed[-1]
            except:
                continue
                
        return result
    
    def _rolling_kmeans(self, df, window, n_clusters):
        """滚动K-means聚类"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        result = np.full(len(df), np.nan)
        
        for i in range(window, len(df)):
            try:
                data = df.iloc[i-window:i].dropna()
                if len(data) >= n_clusters * 2:
                    labels = kmeans.fit_predict(data)
                    result[i] = labels[-1]  # 取最后一个点的标签
            except:
                continue
                
        return result
    
    def _rolling_cluster_distance(self, df, window, n_clusters):
        """滚动聚类距离"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        result = np.full(len(df), np.nan)
        
        for i in range(window, len(df)):
            try:
                data = df.iloc[i-window:i].dropna()
                if len(data) >= n_clusters * 2:
                    kmeans.fit(data)
                    distances = kmeans.transform(data)
                    result[i] = np.min(distances[-1])  # 到最近聚类中心的距离
            except:
                continue
                
        return result


class AdvancedFeatureEngineeringPipeline:
    """高级特征工程管道"""
    
    def __init__(self):
        self.high_order_stats = HighOrderStatisticalFeatures()
        self.frequency_features = FrequencyDomainFeatures()
        self.network_features = NetworkGraphFeatures()
        self.macro_features = MacroEconomicFeatures()
        self.nonlinear_features = NonLinearFeatures()
    
    def transform(self, 
                  df: pd.DataFrame, 
                  feature_types: List[str] = ['all'],
                  columns: Optional[List[str]] = None,
                  date_column: str = 'date') -> pd.DataFrame:
        """
        应用高级特征工程
        
        Args:
            df: 输入数据框
            feature_types: 特征类型列表 ['stats', 'frequency', 'network', 'macro', 'nonlinear', 'all']
            columns: 要处理的列名列表
            date_column: 日期列名
            
        Returns:
            增强后的数据框
        """
        result_df = df.copy()
        
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if date_column in numeric_columns:
                numeric_columns.remove(date_column)
            columns = numeric_columns
        
        if 'all' in feature_types:
            feature_types = ['stats', 'frequency', 'network', 'nonlinear', 'macro']
        
        print(f"开始高级特征工程，处理 {len(columns)} 个特征...")
        
        # 高阶统计特征
        if 'stats' in feature_types:
            print("添加高阶统计特征...")
            result_df = self.high_order_stats.add_moment_features(result_df, columns)
            result_df = self.high_order_stats.add_distribution_tests(result_df, columns)
            result_df = self.high_order_stats.add_entropy_features(result_df, columns)
        
        # 频域特征
        if 'frequency' in feature_types:
            print("添加频域分析特征...")
            result_df = self.frequency_features.add_fft_features(result_df, columns)
            result_df = self.frequency_features.add_wavelet_features(result_df, columns)
        
        # 网络特征
        if 'network' in feature_types and len(columns) >= 3:
            print("添加图论网络特征...")
            result_df = self.network_features.create_correlation_network_features(result_df, columns)
        
        # 非线性特征
        if 'nonlinear' in feature_types and len(columns) >= 2:
            print("添加非线性特征...")
            result_df = self.nonlinear_features.add_manifold_features(result_df, columns)
            result_df = self.nonlinear_features.add_clustering_features(result_df, columns)
        
        # 宏观经济特征
        if 'macro' in feature_types and date_column in df.columns:
            print("添加宏观经济特征...")
            result_df = self.macro_features.add_macro_features(result_df, date_column)
        
        print(f"特征工程完成！从 {df.shape[1]} 列扩展到 {result_df.shape[1]} 列")
        
        return result_df


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据进行测试 - 仅用于测试和演示
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # 创建模拟市场数据 - 模拟数据仅用于演示
    data = {
        'date': dates,
        'price': np.cumsum(np.random.randn(200) * 0.02) + 100,  # 模拟价格仅用于测试
        'volume': np.random.lognormal(10, 0.5, 200),  # 模拟成交量仅用于测试
        'feature1': np.random.randn(200),  # 模拟特征仅用于测试
        'feature2': np.random.randn(200),  # 模拟特征仅用于测试
        'feature3': np.random.randn(200),  # 模拟特征仅用于测试
    }
    
    df = pd.DataFrame(data)
    
    # 测试高级特征工程管道
    pipeline = AdvancedFeatureEngineeringPipeline()
    enhanced_df = pipeline.transform(
        df, 
        feature_types=['stats', 'frequency', 'nonlinear'],
        columns=['price', 'volume', 'feature1', 'feature2', 'feature3']
    )
    
    print(f"原始特征数: {df.shape[1]}")
    print(f"增强后特征数: {enhanced_df.shape[1]}")
    print(f"新增特征数: {enhanced_df.shape[1] - df.shape[1]}")