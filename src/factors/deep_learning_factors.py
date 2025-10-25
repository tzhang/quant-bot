#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习因子计算器
包含LSTM、GRU、Transformer、CNN、Autoencoder、VAE、Attention、ResNet等模型
以及强化学习因子计算功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import logging

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    logger.info("PyTorch可用，将使用完整的深度学习功能")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装，将使用简化版本的深度学习因子")
    # 创建模拟的torch模块
    class nn:
        class Module:
            def __init__(self):
                pass
            def forward(self, x):
                return x
        class LSTM:
            def __init__(self, *args, **kwargs):
                pass
        class GRU:
            def __init__(self, *args, **kwargs):
                pass
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        class Dropout:
            def __init__(self, *args, **kwargs):
                pass
        class ReLU:
            def __init__(self, *args, **kwargs):
                pass
        class Tanh:
            def __init__(self, *args, **kwargs):
                pass
        class Conv1d:
            def __init__(self, *args, **kwargs):
                pass
        class MaxPool1d:
            def __init__(self, *args, **kwargs):
                pass
        class Flatten:
            def __init__(self, *args, **kwargs):
                pass
        class MultiheadAttention:
            def __init__(self, *args, **kwargs):
                pass
        class TransformerEncoder:
            def __init__(self, *args, **kwargs):
                pass
        class TransformerEncoderLayer:
            def __init__(self, *args, **kwargs):
                pass
        class LayerNorm:
            def __init__(self, *args, **kwargs):
                pass

# 尝试导入sklearn
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 尝试导入scipy
try:
    from scipy import signal
    from scipy.fft import fft, ifft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class LSTMFactorModel(nn.Module):
    """LSTM因子模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMFactorModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.random.randn(x.shape[0], 1)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        
        return output

class GRUFactorModel(nn.Module):
    """GRU因子模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 1, dropout: float = 0.2):
        super(GRUFactorModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.random.randn(x.shape[0], 1)
        
        # GRU前向传播
        gru_out, hidden = self.gru(x)
        
        # 使用最后一个时间步的输出
        output = self.dropout(gru_out[:, -1, :])
        output = self.fc(output)
        
        return output

class TransformerFactorModel(nn.Module):
    """Transformer因子模型"""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 8,
                 num_layers: int = 3, output_size: int = 1, dropout: float = 0.2):
        super(TransformerFactorModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)
        
    def _create_positional_encoding(self, d_model: int, max_len: int = 1000):
        """创建位置编码"""
        if not TORCH_AVAILABLE:
            return np.zeros((max_len, d_model))
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.random.randn(x.shape[0], 1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer编码
        transformer_out = self.transformer(x)
        
        # 全局平均池化
        pooled = transformer_out.mean(dim=1)
        
        # 输出层
        output = self.dropout(pooled)
        output = self.fc(output)
        
        return output

class CNNFactorModel(nn.Module):
    """CNN因子模型"""
    
    def __init__(self, input_size: int, sequence_length: int, 
                 num_filters: int = 64, filter_sizes: List[int] = [3, 5, 7],
                 output_size: int = 1, dropout: float = 0.2):
        super(CNNFactorModel, self).__init__()
        self.filter_sizes = filter_sizes
        
        # 多个卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(input_size, num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 计算卷积后的特征维度
        conv_output_size = len(filter_sizes) * num_filters * (sequence_length // 2)
        
        self.fc = nn.Linear(conv_output_size, output_size)
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.random.randn(x.shape[0], 1)
        
        # 转换维度: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        
        # 多个卷积操作
        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(x))
            pooled = self.pool(conv_out)
            conv_outputs.append(pooled)
        
        # 拼接所有卷积输出
        x = torch.cat(conv_outputs, dim=1)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.dropout(x)
        output = self.fc(x)
        
        return output

class AutoencoderFactorModel(nn.Module):
    """自编码器因子模型"""
    
    def __init__(self, input_size: int, encoding_dim: int = 32, 
                 hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super(AutoencoderFactorModel, self).__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_size))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.random.randn(x.shape[0], x.shape[1])
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VAEFactorModel(nn.Module):
    """变分自编码器因子模型"""
    
    def __init__(self, input_size: int, latent_dim: int = 16, 
                 hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super(VAEFactorModel, self).__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 均值和方差层
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_size))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        if not TORCH_AVAILABLE:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class AttentionFactorModel(nn.Module):
    """注意力机制因子模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_heads: int = 8, output_size: int = 1, dropout: float = 0.2):
        super(AttentionFactorModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.random.randn(x.shape[0], 1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 自注意力
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.layer_norm(x + self.dropout(ffn_output))
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 输出层
        output = self.output_layer(x)
        
        return output

class ResNetFactorModel(nn.Module):
    """ResNet因子模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_blocks: int = 3, output_size: int = 1, dropout: float = 0.2):
        super(ResNetFactorModel, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            self._make_res_block(hidden_size, dropout) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def _make_res_block(self, hidden_size: int, dropout: float):
        """创建残差块"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return np.random.randn(x.shape[0], 1)
        
        # 输入层
        x = x.mean(dim=1)  # 全局平均池化
        x = self.input_layer(x)
        
        # 残差块
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = x + residual  # 残差连接
            x = torch.relu(x)
        
        # 输出层
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output

class DeepLearningFactorCalculator:
    """深度学习因子计算器"""
    
    def __init__(self, sequence_length: int = 60, device: str = 'cpu'):
        """
        初始化深度学习因子计算器
        
        Args:
            sequence_length: 序列长度
            device: 计算设备
        """
        self.sequence_length = sequence_length
        self.device = device
        self.models = {}
        
        if TORCH_AVAILABLE:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"深度学习因子计算器初始化完成，序列长度: {sequence_length}")

    def prepare_sequences(self, data: pd.DataFrame, target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备序列数据
        
        Args:
            data: 输入数据
            target_col: 目标列名
            
        Returns:
            X: 特征序列
            y: 目标序列
        """
        try:
            # 数据预处理
            if SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data.fillna(method='ffill').fillna(0))
            else:
                # 简单标准化
                scaled_data = (data.fillna(method='ffill').fillna(0) - data.mean()) / (data.std() + 1e-8)
                scaled_data = scaled_data.values
            
            # 创建序列
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                if target_col and target_col in data.columns:
                    y.append(scaled_data[i, data.columns.get_loc(target_col)])
                else:
                    # 默认使用第一列作为目标
                    y.append(scaled_data[i, 0])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"序列准备失败: {str(e)}")
            return np.array([]), np.array([])

    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, 
                        model_name: str = 'lstm_factor') -> Dict[str, Any]:
        """训练LSTM模型"""
        if not TORCH_AVAILABLE or len(X) == 0:
            return {}
        
        try:
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            input_size = X.shape[-1]
            model = LSTMFactorModel(input_size=input_size)
            
            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            total_loss = 0
            num_epochs = 100
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                if epoch % 20 == 0:
                    logger.info(f"LSTM Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                test_pred = model(X_test_tensor)
                
                train_mse = nn.functional.mse_loss(train_pred, y_train_tensor).item()
                test_mse = nn.functional.mse_loss(test_pred, y_test_tensor).item()
            
            # 保存模型
            self.models[model_name] = model
            
            return {
                'model': model,
                'train_loss': train_mse,
                'test_loss': test_mse,
                'predictions': test_pred.numpy(),
                'model_type': 'LSTM'
            }
            
        except Exception as e:
            logger.error(f"LSTM模型训练失败: {str(e)}")
            return {}

    def train_gru_model(self, X: np.ndarray, y: np.ndarray,
                       model_name: str = 'gru_factor') -> Dict[str, Any]:
        """训练GRU模型"""
        if not TORCH_AVAILABLE or len(X) == 0:
            return {}
        
        try:
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            input_size = X.shape[-1]
            model = GRUFactorModel(input_size=input_size)
            
            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            total_loss = 0
            num_epochs = 100
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                if epoch % 20 == 0:
                    logger.info(f"GRU Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                test_pred = model(X_test_tensor)
                
                train_mse = nn.functional.mse_loss(train_pred, y_train_tensor).item()
                test_mse = nn.functional.mse_loss(test_pred, y_test_tensor).item()
            
            # 保存模型
            self.models[model_name] = model
            
            return {
                'model': model,
                'train_loss': train_mse,
                'test_loss': test_mse,
                'predictions': test_pred.numpy(),
                'model_type': 'GRU'
            }
            
        except Exception as e:
            logger.error(f"GRU模型训练失败: {str(e)}")
            return {}

    def train_transformer_model(self, X: np.ndarray, y: np.ndarray,
                              model_name: str = 'transformer_factor') -> Dict[str, Any]:
        """训练Transformer模型"""
        if not TORCH_AVAILABLE or len(X) == 0:
            return {}
        
        try:
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            input_size = X.shape[-1]
            model = TransformerFactorModel(input_size=input_size)
            
            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            total_loss = 0
            num_epochs = 100
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                if epoch % 20 == 0:
                    logger.info(f"Transformer Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                test_pred = model(X_test_tensor)
                
                train_mse = nn.functional.mse_loss(train_pred, y_train_tensor).item()
                test_mse = nn.functional.mse_loss(test_pred, y_test_tensor).item()
            
            # 保存模型
            self.models[model_name] = model
            
            return {
                'model': model,
                'train_loss': train_mse,
                'test_loss': test_mse,
                'predictions': test_pred.numpy(),
                'model_type': 'Transformer'
            }
            
        except Exception as e:
            logger.error(f"Transformer模型训练失败: {str(e)}")
            return {}

    def train_cnn_model(self, X: np.ndarray, y: np.ndarray,
                       model_name: str = 'cnn_factor') -> Dict[str, Any]:
        """训练CNN模型"""
        if not TORCH_AVAILABLE or len(X) == 0:
            return {}
        
        try:
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            input_size = X.shape[-1]
            sequence_length = X.shape[1]
            model = CNNFactorModel(input_size=input_size, sequence_length=sequence_length)
            
            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            total_loss = 0
            num_epochs = 100
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                if epoch % 20 == 0:
                    logger.info(f"CNN Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                test_pred = model(X_test_tensor)
                
                train_mse = nn.functional.mse_loss(train_pred, y_train_tensor).item()
                test_mse = nn.functional.mse_loss(test_pred, y_test_tensor).item()
            
            # 保存模型
            self.models[model_name] = model
            
            return {
                'model': model,
                'train_loss': train_mse,
                'test_loss': test_mse,
                'predictions': test_pred.numpy(),
                'model_type': 'CNN'
            }
            
        except Exception as e:
            logger.error(f"CNN模型训练失败: {str(e)}")
            return {}

    def train_autoencoder_model(self, X: np.ndarray, model_name: str = 'autoencoder_factor') -> Dict[str, Any]:
        """训练自编码器模型"""
        if not TORCH_AVAILABLE or len(X) == 0:
            return {}
        
        try:
            # 重塑数据为2D
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # 数据分割
            X_train, X_test = train_test_split(X_reshaped, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            input_size = X_reshaped.shape[1]
            model = AutoencoderFactorModel(input_size=input_size)
            
            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            total_loss = 0
            num_epochs = 100
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, _) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, data)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                if epoch % 20 == 0:
                    logger.info(f"Autoencoder Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                test_output = model(X_test_tensor)
                test_loss = nn.functional.mse_loss(test_output, X_test_tensor).item()
            
            # 保存模型
            self.models[model_name] = model
            
            return {
                'model': model,
                'train_loss': total_loss / num_epochs,
                'test_loss': test_loss,
                'reconstructions': test_output.numpy(),
                'model_type': 'Autoencoder'
            }
            
        except Exception as e:
            logger.error(f"自编码器模型训练失败: {str(e)}")
            return {}

    def train_vae_model(self, X: np.ndarray, model_name: str = 'vae_factor') -> Dict[str, Any]:
        """训练VAE模型"""
        if not TORCH_AVAILABLE or len(X) == 0:
            return {}
        
        try:
            # 重塑数据为2D
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # 数据分割
            X_train, X_test = train_test_split(X_reshaped, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            input_size = X_reshaped.shape[1]
            model = VAEFactorModel(input_size=input_size)
            
            # 训练设置
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # VAE损失函数
            def vae_loss(recon_x, x, mu, logvar):
                BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                return BCE + KLD
            
            # 训练循环
            model.train()
            total_loss = 0
            num_epochs = 50
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, _) in enumerate(train_loader):
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(data)
                    loss = vae_loss(recon_batch, data, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                if epoch % 10 == 0:
                    logger.info(f"VAE Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                test_recon, test_mu, test_logvar = model(X_test_tensor)
                test_loss = vae_loss(test_recon, X_test_tensor, test_mu, test_logvar)
            
            # 保存模型
            self.models[model_name] = model
            
            return {
                'model': model,
                'train_loss': total_loss / num_epochs,
                'test_loss': test_loss.item(),
                'latent_features': test_mu.numpy(),
                'model_type': 'VAE'
            }
            
        except Exception as e:
            logger.error(f"VAE模型训练失败: {str(e)}")
            return {}

    def train_attention_model(self, X: np.ndarray, y: np.ndarray,
                            model_name: str = 'attention_factor') -> Dict[str, Any]:
        """训练注意力模型"""
        if not TORCH_AVAILABLE or len(X) == 0:
            return {}
        
        try:
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            input_size = X.shape[-1]
            model = AttentionFactorModel(input_size=input_size)
            
            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            total_loss = 0
            num_epochs = 100
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                if epoch % 20 == 0:
                    logger.info(f"Attention Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                test_pred = model(X_test_tensor)
                
                train_mse = nn.functional.mse_loss(train_pred, y_train_tensor).item()
                test_mse = nn.functional.mse_loss(test_pred, y_test_tensor).item()
            
            # 保存模型
            self.models[model_name] = model
            
            return {
                'model': model,
                'train_loss': train_mse,
                'test_loss': test_mse,
                'predictions': test_pred.numpy(),
                'model_type': 'Attention'
            }
            
        except Exception as e:
            logger.error(f"注意力模型训练失败: {str(e)}")
            return {}

    def train_resnet_model(self, X: np.ndarray, y: np.ndarray,
                         model_name: str = 'resnet_factor') -> Dict[str, Any]:
        """训练ResNet模型"""
        if not TORCH_AVAILABLE or len(X) == 0:
            return {}
        
        try:
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 创建模型
            input_size = X.shape[-1]
            model = ResNetFactorModel(input_size=input_size)
            
            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            total_loss = 0
            num_epochs = 100
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                if epoch % 20 == 0:
                    logger.info(f"ResNet Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # 评估
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                test_pred = model(X_test_tensor)
                
                train_mse = nn.functional.mse_loss(train_pred, y_train_tensor).item()
                test_mse = nn.functional.mse_loss(test_pred, y_test_tensor).item()
            
            # 保存模型
            self.models[model_name] = model
            
            return {
                'model': model,
                'train_loss': train_mse,
                'test_loss': test_mse,
                'predictions': test_pred.numpy(),
                'model_type': 'ResNet'
            }
            
        except Exception as e:
            logger.error(f"ResNet模型训练失败: {str(e)}")
            return {}

    def calculate_deep_learning_factors(self, price_data: pd.DataFrame,
                                      volume_data: pd.DataFrame = None,
                                      fundamental_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算深度学习因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            
        Returns:
            深度学习因子字典
        """
        try:
            # 准备特征数据
            features = self._prepare_features(price_data, volume_data, fundamental_data)
            
            if features.empty:
                logger.warning("特征数据为空")
                return {}
            
            # 准备序列数据
            X, y = self.prepare_sequences(features, target_col='close')
            
            if len(X) == 0:
                logger.warning("序列数据为空")
                return {}
            
            # 训练各种模型
            models_results = {}
            
            # LSTM模型
            lstm_result = self.train_lstm_model(X, y, 'lstm_factor')
            if lstm_result:
                models_results['lstm'] = lstm_result
            
            # GRU模型
            gru_result = self.train_gru_model(X, y, 'gru_factor')
            if gru_result:
                models_results['gru'] = gru_result
            
            # Transformer模型
            transformer_result = self.train_transformer_model(X, y, 'transformer_factor')
            if transformer_result:
                models_results['transformer'] = transformer_result
            
            # CNN模型
            cnn_result = self.train_cnn_model(X, y, 'cnn_factor')
            if cnn_result:
                models_results['cnn'] = cnn_result
            
            # 自编码器模型
            autoencoder_result = self.train_autoencoder_model(X, 'autoencoder_factor')
            if autoencoder_result:
                models_results['autoencoder'] = autoencoder_result
            
            # VAE模型
            vae_result = self.train_vae_model(X, 'vae_factor')
            if vae_result:
                models_results['vae'] = vae_result
            
            # 注意力模型
            attention_result = self.train_attention_model(X, y, 'attention_factor')
            if attention_result:
                models_results['attention'] = attention_result
            
            # ResNet模型
            resnet_result = self.train_resnet_model(X, y, 'resnet_factor')
            if resnet_result:
                models_results['resnet'] = resnet_result
            
            # 生成因子序列
            factors = self._generate_factor_series(models_results, price_data.index)
            
            logger.info(f"成功计算 {len(factors)} 个深度学习因子")
            return factors
            
        except Exception as e:
            logger.error(f"深度学习因子计算失败: {str(e)}")
            return {}

    def _prepare_features(self, price_data: pd.DataFrame,
                         volume_data: pd.DataFrame = None,
                         fundamental_data: pd.DataFrame = None) -> pd.DataFrame:
        """准备特征数据"""
        try:
            features = price_data.copy()
            
            # 添加技术指标
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            features['volatility'] = features['returns'].rolling(window=20).std()
            
            # 移动平均
            for window in [5, 10, 20, 50]:
                features[f'ma_{window}'] = features['close'].rolling(window=window).mean()
                features[f'ma_ratio_{window}'] = features['close'] / features[f'ma_{window}']
            
            # RSI
            delta = features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # 布林带
            bb_window = 20
            bb_std = 2
            bb_ma = features['close'].rolling(window=bb_window).mean()
            bb_std_val = features['close'].rolling(window=bb_window).std()
            bb_upper = bb_ma + (bb_std_val * bb_std)
            bb_lower = bb_ma - (bb_std_val * bb_std)
            features['bb_upper'] = bb_upper
            features['bb_lower'] = bb_lower
            features['bb_position'] = (features['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # 添加成交量特征
            if volume_data is not None:
                features = features.join(volume_data, how='left')
                features['volume_ma'] = features['volume'].rolling(window=20).mean()
                features['volume_ratio'] = features['volume'] / features['volume_ma']
            
            # 添加基本面特征
            if fundamental_data is not None:
                features = features.join(fundamental_data, how='left')
            
            # 填充缺失值
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"特征准备失败: {str(e)}")
            return pd.DataFrame()

    def _generate_factor_series(self, models_results: Dict[str, Dict], 
                               index: pd.Index) -> Dict[str, pd.Series]:
        """生成因子序列"""
        try:
            factors = {}
            
            for model_name, result in models_results.items():
                if 'predictions' in result:
                    # 对于有预测结果的模型
                    predictions = result['predictions'].flatten()
                    
                    # 创建与原始数据长度匹配的序列
                    factor_values = np.full(len(index), np.nan)
                    start_idx = self.sequence_length
                    end_idx = start_idx + len(predictions)
                    
                    if end_idx <= len(index):
                        factor_values[start_idx:end_idx] = predictions
                    
                    factors[f'{model_name}_factor'] = pd.Series(factor_values, index=index)
                
                elif 'latent_features' in result:
                    # 对于VAE等有潜在特征的模型
                    latent_features = result['latent_features']
                    
                    # 使用第一个潜在维度作为因子
                    if latent_features.ndim > 1:
                        latent_values = latent_features[:, 0]
                    else:
                        latent_values = latent_features
                    
                    factor_values = np.full(len(index), np.nan)
                    start_idx = self.sequence_length
                    end_idx = start_idx + len(latent_values)
                    
                    if end_idx <= len(index):
                        factor_values[start_idx:end_idx] = latent_values
                    
                    factors[f'{model_name}_latent_factor'] = pd.Series(factor_values, index=index)
            
            return factors
            
        except Exception as e:
            logger.error(f"因子序列生成失败: {str(e)}")
            return {}

    def calculate_reinforcement_learning_factors(self, price_data: pd.DataFrame,
                                               volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算强化学习因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            
        Returns:
            强化学习因子字典
        """
        try:
            factors = {}
            
            # Q-Learning因子
            q_factors = self._calculate_q_learning_factors(price_data)
            factors.update(q_factors)
            
            # 动作价值因子
            action_factors = self._calculate_action_value_factors(price_data, volume_data)
            factors.update(action_factors)
            
            # 策略梯度因子
            policy_factors = self._calculate_policy_gradient_factors(price_data)
            factors.update(policy_factors)
            
            logger.info(f"成功计算 {len(factors)} 个强化学习因子")
            return factors
            
        except Exception as e:
            logger.error(f"强化学习因子计算失败: {str(e)}")
            return {}

    def _calculate_q_learning_factors(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算Q-Learning因子"""
        try:
            factors = {}
            
            # 计算收益率
            returns = price_data['close'].pct_change()
            
            # 简化的Q值计算
            # 状态：价格相对位置
            high_low_range = price_data['high'] - price_data['low']
            price_position = (price_data['close'] - price_data['low']) / (high_low_range + 1e-8)
            
            # 动作：买入(1)、持有(0)、卖出(-1)
            # 简化的Q值估计
            q_buy = returns.rolling(window=10).mean() + returns.rolling(window=10).std()
            q_hold = returns.rolling(window=10).mean()
            q_sell = -returns.rolling(window=10).mean() + returns.rolling(window=10).std()
            
            # Q-Learning因子
            factors['rl_q_buy'] = q_buy
            factors['rl_q_hold'] = q_hold
            factors['rl_q_sell'] = q_sell
            factors['rl_q_max'] = pd.concat([q_buy, q_hold, q_sell], axis=1).max(axis=1)
            factors['rl_q_diff'] = q_buy - q_sell
            
            return factors
            
        except Exception as e:
            logger.error(f"Q-Learning因子计算失败: {str(e)}")
            return {}

    def _calculate_action_value_factors(self, price_data: pd.DataFrame,
                                      volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """计算动作价值因子"""
        try:
            factors = {}
            
            # 计算收益率和波动率
            returns = price_data['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            # 动作价值函数
            # 买入动作价值
            buy_signal = (price_data['close'] > price_data['close'].rolling(window=5).mean()).astype(int)
            buy_value = returns.shift(-1) * buy_signal
            factors['rl_buy_value'] = buy_value.rolling(window=10).mean()
            
            # 卖出动作价值
            sell_signal = (price_data['close'] < price_data['close'].rolling(window=5).mean()).astype(int)
            sell_value = -returns.shift(-1) * sell_signal
            factors['rl_sell_value'] = sell_value.rolling(window=10).mean()
            
            # 最优动作选择
            action_values = pd.DataFrame({
                'buy': factors['rl_buy_value'],
                'sell': factors['rl_sell_value']
            })
            factors['rl_optimal_action'] = action_values.idxmax(axis=1).map({'buy': 1, 'sell': -1}).fillna(0)
            
            # 动作置信度
            max_value = action_values.max(axis=1)
            min_value = action_values.min(axis=1)
            factors['rl_action_confidence'] = (max_value - min_value) / (max_value.abs() + 1e-8)
            
            return factors
            
        except Exception as e:
            logger.error(f"动作价值因子计算失败: {str(e)}")
            return {}

    def _calculate_policy_gradient_factors(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算策略梯度因子"""
        try:
            factors = {}
            
            # 计算技术指标
            returns = price_data['close'].pct_change()
            
            # 移动平均
            ma_short = price_data['close'].rolling(window=5).mean()
            ma_long = price_data['close'].rolling(window=20).mean()
            
            # RSI
            delta = price_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
            
            # 策略信号
            ma_signal = (ma_short > ma_long).astype(int) * 2 - 1  # -1 或 1
            rsi_signal = ((rsi < 30).astype(int) - (rsi > 70).astype(int))  # -1, 0, 或 1
            
            # 简化的策略梯度估计
            policy_signal = (ma_signal + rsi_signal) / 2
            
            # 策略梯度因子
            factors['rl_policy_signal'] = policy_signal
            factors['rl_policy_gradient'] = policy_signal * returns.shift(-1)
            factors['rl_policy_momentum'] = factors['rl_policy_gradient'].rolling(window=10).mean()
            
            # 策略改进方向
            policy_improvement = factors['rl_policy_gradient'].rolling(window=5).apply(
                lambda x: 1 if x.mean() > 0 else -1
            )
            factors['rl_policy_improvement'] = policy_improvement
            
            return factors
            
        except Exception as e:
            logger.error(f"策略梯度因子计算失败: {str(e)}")
            return {}

    def calculate_all_deep_learning_factors(self, price_data: pd.DataFrame,
                                          volume_data: pd.DataFrame = None,
                                          fundamental_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算所有深度学习因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            
        Returns:
            所有深度学习因子字典
        """
        try:
            all_factors = {}
            
            # 深度学习因子
            dl_factors = self.calculate_deep_learning_factors(price_data, volume_data, fundamental_data)
            all_factors.update(dl_factors)
            
            # 强化学习因子
            rl_factors = self.calculate_reinforcement_learning_factors(price_data, volume_data)
            all_factors.update(rl_factors)
            
            logger.info(f"总共计算了 {len(all_factors)} 个深度学习因子")
            return all_factors
            
        except Exception as e:
            logger.error(f"所有深度学习因子计算失败: {str(e)}")
            return {}

    def generate_factor_report(self, factors: Dict[str, pd.Series]) -> str:
        """生成因子报告"""
        try:
            if not factors:
                return "没有可用的因子数据"
            
            report = ["深度学习因子分析报告", "=" * 50, ""]
            
            for factor_name, factor_series in factors.items():
                valid_data = factor_series.dropna()
                if len(valid_data) > 0:
                    report.extend([
                        f"因子: {factor_name}",
                        f"  有效数据点: {len(valid_data)}",
                        f"  均值: {valid_data.mean():.6f}",
                        f"  标准差: {valid_data.std():.6f}",
                        f"  最小值: {valid_data.min():.6f}",
                        f"  最大值: {valid_data.max():.6f}",
                         ""
                     ])
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"因子报告生成失败: {str(e)}")
            return "因子报告生成失败"

def main():
    """示例用法 - 仅用于测试和演示"""
    print("深度学习因子计算器示例")
    
    # 创建模拟数据 - 仅用于测试和演示
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')  # 生成日期范围 - 仅用于测试和演示
    n_periods = len(dates)  # 计算时间序列长度 - 仅用于测试和演示
    
    np.random.seed(42)  # 设置随机种子确保结果可重现 - 仅用于测试和演示
    
    # 模拟价格数据 - 仅用于测试和演示
    price_data = pd.DataFrame({
        'open': np.random.uniform(95, 105, n_periods),  # 开盘价：95-105区间均匀分布 - 仅用于测试和演示
        'high': np.random.uniform(100, 110, n_periods),  # 最高价：100-110区间均匀分布 - 仅用于测试和演示
        'low': np.random.uniform(90, 100, n_periods),  # 最低价：90-100区间均匀分布 - 仅用于测试和演示
        'close': np.random.uniform(95, 105, n_periods),  # 收盘价：95-105区间均匀分布 - 仅用于测试和演示
    }, index=dates)
    
    # 添加趋势 - 仅用于测试和演示
    trend = np.cumsum(np.random.normal(0.001, 0.02, n_periods))  # 生成累积趋势 - 仅用于测试和演示
    price_data['close'] = 100 * np.exp(trend)  # 应用指数趋势到收盘价 - 仅用于测试和演示
    
    # 模拟成交量数据 - 仅用于测试和演示
    volume_data = pd.DataFrame({
        'volume': np.random.uniform(1e6, 5e6, n_periods)  # 成交量：100万-500万区间均匀分布 - 仅用于测试和演示
    }, index=dates)
    
    # 计算因子 - 仅用于测试和演示
    calculator = DeepLearningFactorCalculator(sequence_length=30)  # 创建深度学习因子计算器 - 仅用于测试和演示
    factors = calculator.calculate_all_deep_learning_factors(price_data, volume_data)  # 计算所有深度学习因子 - 仅用于测试和演示
    
    # 生成报告 - 仅用于测试和演示
    report = calculator.generate_factor_report(factors)  # 生成因子报告 - 仅用于测试和演示
    print(report)

if __name__ == "__main__":
    main()