#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件示例
复制此文件为 config.py 并填入您的真实凭据
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class FirstradeConfig:
    """Firstrade 配置"""
    username: str = ""
    password: str = ""
    pin: str = ""
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            username=os.getenv('FIRSTRADE_USERNAME', ''),
            password=os.getenv('FIRSTRADE_PASSWORD', ''),
            pin=os.getenv('FIRSTRADE_PIN', '')
        )

@dataclass
class AlpacaConfig:
    """Alpaca 配置"""
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"  # 默认使用模拟交易
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            api_key=os.getenv('ALPACA_API_KEY', ''),
            secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )

@dataclass
class InteractiveBrokersConfig:
    """Interactive Brokers 配置"""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS 默认端口，IB Gateway 使用 4001
    client_id: int = 1
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            host=os.getenv('IB_HOST', '127.0.0.1'),
            port=int(os.getenv('IB_PORT', '7497')),
            client_id=int(os.getenv('IB_CLIENT_ID', '1'))
        )

@dataclass
class SystemConfig:
    """系统配置"""
    log_level: str = "INFO"
    data_collection_interval: int = 5  # 秒
    max_data_points: int = 1000
    enable_real_trading: bool = False  # 默认禁用真实交易
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            data_collection_interval=int(os.getenv('DATA_COLLECTION_INTERVAL', '5')),
            max_data_points=int(os.getenv('MAX_DATA_POINTS', '1000')),
            enable_real_trading=os.getenv('ENABLE_REAL_TRADING', 'false').lower() == 'true'
        )

# 全局配置实例
config = {
    'firstrade': FirstradeConfig.from_env(),
    'alpaca': AlpacaConfig.from_env(),
    'interactive_brokers': InteractiveBrokersConfig.from_env(),
    'system': SystemConfig.from_env()
}

def load_config_from_file(config_file: str = 'config.py') -> dict:
    """从配置文件加载配置"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return config

# 使用说明
"""
使用方法:

1. 复制此文件为 config.py
2. 填入您的真实凭据:

   # Firstrade 配置
   config['firstrade'].username = "your_username"
   config['firstrade'].password = "your_password"
   config['firstrade'].pin = "your_pin"
   
   # Alpaca 配置
   config['alpaca'].api_key = "your_api_key"
   config['alpaca'].secret_key = "your_secret_key"
   
   # Interactive Brokers 配置
   config['interactive_brokers'].port = 7497  # TWS 端口
   
3. 或者设置环境变量:
   export FIRSTRADE_USERNAME="your_username"
   export FIRSTRADE_PASSWORD="your_password"
   export FIRSTRADE_PIN="your_pin"
   
   export ALPACA_API_KEY="your_api_key"
   export ALPACA_SECRET_KEY="your_secret_key"
   
   export IB_PORT="7497"

4. 启用真实交易 (谨慎使用):
   config['system'].enable_real_trading = True
   # 或设置环境变量: export ENABLE_REAL_TRADING="true"
"""