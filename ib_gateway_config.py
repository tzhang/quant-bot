#!/usr/bin/env python3
"""
IB Gateway 配置文件
配置Interactive Brokers Gateway API连接参数和真实数据获取设置
"""

import os
from dataclasses import dataclass
from typing import Optional, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IBGatewayConfig:
    """IB Gateway 配置类"""
    # 连接参数
    host: str = "127.0.0.1"
    port: int = 4001  # IB Gateway 模拟交易端口
    client_id: int = 1
    timeout: int = 30
    
    # 真实交易配置
    real_trading_port: int = 4000  # IB Gateway 真实交易端口
    
    # 数据获取配置
    market_data_type: int = 3  # 1=实时, 2=冻结, 3=延迟, 4=延迟冻结
    request_timeout: int = 10
    max_concurrent_requests: int = 50
    
    # 监控股票列表
    watch_symbols: List[str] = None
    
    # 风险管理
    max_daily_loss: float = 5000.0
    max_position_size: float = 0.1  # 10%最大仓位
    
    def __post_init__(self):
        if self.watch_symbols is None:
            self.watch_symbols = [
                "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
                "NVDA", "META", "NFLX", "AMD", "INTC"
            ]
    
    @classmethod
    def from_env(cls) -> 'IBGatewayConfig':
        """从环境变量加载配置"""
        return cls(
            host=os.getenv('IB_HOST', '127.0.0.1'),
            port=int(os.getenv('IB_PORT', '4001')),
            client_id=int(os.getenv('IB_CLIENT_ID', '1')),
            timeout=int(os.getenv('IB_TIMEOUT', '30')),
            real_trading_port=int(os.getenv('IB_REAL_PORT', '4000')),
            market_data_type=int(os.getenv('IB_MARKET_DATA_TYPE', '3')),
            max_daily_loss=float(os.getenv('IB_MAX_DAILY_LOSS', '5000.0')),
            max_position_size=float(os.getenv('IB_MAX_POSITION_SIZE', '0.1'))
        )
    
    def get_connection_string(self, use_real_trading: bool = False) -> str:
        """获取连接字符串"""
        port = self.real_trading_port if use_real_trading else self.port
        return f"{self.host}:{port} (客户端ID: {self.client_id})"
    
    def is_paper_trading(self) -> bool:
        """检查是否为模拟交易"""
        return self.port in [4001, 7497]  # Gateway模拟端口或TWS模拟端口
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 检查端口范围
            if not (1000 <= self.port <= 65535):
                logger.error(f"无效端口号: {self.port}")
                return False
            
            # 检查客户端ID
            if not (0 <= self.client_id <= 32):
                logger.error(f"无效客户端ID: {self.client_id}")
                return False
            
            # 检查超时设置
            if self.timeout <= 0:
                logger.error(f"无效超时设置: {self.timeout}")
                return False
            
            # 检查风险参数
            if self.max_daily_loss <= 0:
                logger.error(f"无效最大日损失: {self.max_daily_loss}")
                return False
            
            if not (0 < self.max_position_size <= 1):
                logger.error(f"无效最大仓位比例: {self.max_position_size}")
                return False
            
            logger.info("✅ IB Gateway配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def print_config(self):
        """打印配置信息"""
        trading_mode = "模拟交易" if self.is_paper_trading() else "真实交易"
        
        print("🔧 IB Gateway 配置信息:")
        print("=" * 40)
        print(f"连接地址: {self.host}:{self.port}")
        print(f"客户端ID: {self.client_id}")
        print(f"交易模式: {trading_mode}")
        print(f"连接超时: {self.timeout}秒")
        print(f"市场数据类型: {self.market_data_type}")
        print(f"最大日损失: ${self.max_daily_loss:,.2f}")
        print(f"最大仓位比例: {self.max_position_size:.1%}")
        print(f"监控股票数量: {len(self.watch_symbols)}")
        print("=" * 40)

# 默认配置实例
DEFAULT_CONFIG = IBGatewayConfig()

def get_config() -> IBGatewayConfig:
    """获取配置实例"""
    # 优先从环境变量加载
    config = IBGatewayConfig.from_env()
    
    # 验证配置
    if not config.validate():
        logger.warning("使用默认配置")
        config = DEFAULT_CONFIG
    
    return config

def main():
    """测试配置"""
    print("🚀 IB Gateway 配置测试")
    
    # 加载配置
    config = get_config()
    config.print_config()
    
    # 显示监控股票
    print(f"\n📊 监控股票列表:")
    for i, symbol in enumerate(config.watch_symbols, 1):
        print(f"  {i:2d}. {symbol}")
    
    print(f"\n🔗 连接信息:")
    print(f"  模拟交易: {config.get_connection_string(False)}")
    print(f"  真实交易: {config.get_connection_string(True)}")

if __name__ == "__main__":
    main()