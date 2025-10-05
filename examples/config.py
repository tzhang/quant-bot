#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 管理券商API凭据和系统设置
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class FirstradeConfig:
    """Firstrade配置"""
    username: str = ""
    password: str = ""
    pin: str = ""
    dry_run: bool = True
    enabled: bool = False

@dataclass
class AlpacaConfig:
    """Alpaca配置"""
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"  # 纸上交易URL
    dry_run: bool = True
    enabled: bool = False

@dataclass
class InteractiveBrokersConfig:
    """Interactive Brokers配置"""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS纸上交易端口
    client_id: int = 1
    dry_run: bool = True
    enabled: bool = False

@dataclass
class SystemConfig:
    """系统配置"""
    # 监控仪表板设置
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080
    
    # 数据收集间隔（秒）
    data_collection_interval: int = 5
    
    # 日志级别
    log_level: str = "INFO"
    
    # 风险管理设置
    max_daily_trades: int = 10
    max_position_size: float = 0.15
    max_daily_loss: float = 0.05
    
    # 默认交易符号
    default_symbols: list = None
    
    def __post_init__(self):
        if self.default_symbols is None:
            self.default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

class Config:
    """主配置类"""
    
    def __init__(self):
        self.firstrade = FirstradeConfig()
        self.alpaca = AlpacaConfig()
        self.interactive_brokers = InteractiveBrokersConfig()
        self.system = SystemConfig()
        
        # 从环境变量加载配置
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # Firstrade配置
        self.firstrade.username = os.getenv("FIRSTRADE_USERNAME", "")
        self.firstrade.password = os.getenv("FIRSTRADE_PASSWORD", "")
        self.firstrade.pin = os.getenv("FIRSTRADE_PIN", "")
        self.firstrade.dry_run = os.getenv("FIRSTRADE_DRY_RUN", "true").lower() == "true"
        self.firstrade.enabled = os.getenv("FIRSTRADE_ENABLED", "false").lower() == "true"
        
        # Alpaca配置
        self.alpaca.api_key = os.getenv("ALPACA_API_KEY", "")
        self.alpaca.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.alpaca.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.alpaca.dry_run = os.getenv("ALPACA_DRY_RUN", "true").lower() == "true"
        self.alpaca.enabled = os.getenv("ALPACA_ENABLED", "false").lower() == "true"
        
        # Interactive Brokers配置
        self.interactive_brokers.host = os.getenv("IB_HOST", "127.0.0.1")
        self.interactive_brokers.port = int(os.getenv("IB_PORT", "7497"))
        self.interactive_brokers.client_id = int(os.getenv("IB_CLIENT_ID", "1"))
        self.interactive_brokers.dry_run = os.getenv("IB_DRY_RUN", "true").lower() == "true"
        self.interactive_brokers.enabled = os.getenv("IB_ENABLED", "false").lower() == "true"
        
        # 系统配置
        self.system.dashboard_host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
        self.system.dashboard_port = int(os.getenv("DASHBOARD_PORT", "8080"))
        self.system.data_collection_interval = int(os.getenv("DATA_COLLECTION_INTERVAL", "5"))
        self.system.log_level = os.getenv("LOG_LEVEL", "INFO")
    
    def get_enabled_brokers(self) -> Dict[str, bool]:
        """获取启用的券商列表"""
        return {
            "firstrade": self.firstrade.enabled and bool(self.firstrade.username and self.firstrade.password),
            "alpaca": self.alpaca.enabled and bool(self.alpaca.api_key and self.alpaca.secret_key),
            "interactive_brokers": self.interactive_brokers.enabled
        }
    
    def has_any_broker_enabled(self) -> bool:
        """检查是否有任何券商被启用"""
        return any(self.get_enabled_brokers().values())
    
    def get_primary_broker(self) -> Optional[str]:
        """获取主要券商（第一个启用的券商）"""
        enabled_brokers = self.get_enabled_brokers()
        for broker, enabled in enabled_brokers.items():
            if enabled:
                return broker
        return None
    
    def get(self, key: str):
        """获取配置属性"""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"配置项 '{key}' 不存在")

# 全局配置实例
config = Config()

def load_config_from_file(config_file: str = "trading_config.env"):
    """从文件加载配置"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        
        # 重新加载配置
        global config
        config = Config()
        print(f"配置已从 {config_file} 加载")
    else:
        print(f"配置文件 {config_file} 不存在，使用默认配置")

def create_sample_config_file(config_file: str = "trading_config.env.sample"):
    """创建示例配置文件"""
    sample_config = """# 交易系统配置文件
# 复制此文件为 trading_config.env 并填入真实凭据

# Firstrade配置
FIRSTRADE_USERNAME=your_username
FIRSTRADE_PASSWORD=your_password
FIRSTRADE_PIN=your_pin
FIRSTRADE_DRY_RUN=true
FIRSTRADE_ENABLED=false

# Alpaca配置
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DRY_RUN=true
ALPACA_ENABLED=false

# Interactive Brokers配置
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
IB_DRY_RUN=true
IB_ENABLED=false

# 系统配置
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DATA_COLLECTION_INTERVAL=5
LOG_LEVEL=INFO
"""
    
    with open(config_file, 'w') as f:
        f.write(sample_config)
    
    print(f"示例配置文件已创建: {config_file}")
    print("请复制为 trading_config.env 并填入真实凭据")

if __name__ == "__main__":
    # 创建示例配置文件
    create_sample_config_file()
    
    # 显示当前配置
    print("\n当前配置:")
    print(f"启用的券商: {config.get_enabled_brokers()}")
    print(f"主要券商: {config.get_primary_broker()}")
    print(f"仪表板地址: {config.system.dashboard_host}:{config.system.dashboard_port}")