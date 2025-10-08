"""
NASDAQ 100 股票列表配置
包含NASDAQ 100指数中的主要股票代码
"""

# NASDAQ 100 主要股票列表 (按市值排序)
NASDAQ100_SYMBOLS = [
    # 科技巨头
    'AAPL',   # Apple Inc.
    'MSFT',   # Microsoft Corporation
    'GOOGL',  # Alphabet Inc. Class A
    'GOOG',   # Alphabet Inc. Class C
    'AMZN',   # Amazon.com Inc.
    'TSLA',   # Tesla Inc.
    'META',   # Meta Platforms Inc.
    'NVDA',   # NVIDIA Corporation
    
    # 大型科技公司
    'NFLX',   # Netflix Inc.
    'ADBE',   # Adobe Inc.
    'CRM',    # Salesforce Inc.
    'ORCL',   # Oracle Corporation
    'INTC',   # Intel Corporation
    'AMD',    # Advanced Micro Devices Inc.
    'QCOM',   # QUALCOMM Incorporated
    'AVGO',   # Broadcom Inc.
    
    # 电商和服务
    'PYPL',   # PayPal Holdings Inc.
    'EBAY',   # eBay Inc.
    'ZOOM',   # Zoom Video Communications Inc.
    'DOCU',   # DocuSign Inc.
    
    # 生物技术
    'GILD',   # Gilead Sciences Inc.
    'AMGN',   # Amgen Inc.
    'BIIB',   # Biogen Inc.
    'REGN',   # Regeneron Pharmaceuticals Inc.
    'VRTX',   # Vertex Pharmaceuticals Inc.
    
    # 消费品牌
    'COST',   # Costco Wholesale Corporation
    'SBUX',   # Starbucks Corporation
    'PEP',    # PepsiCo Inc.
    'KO',     # The Coca-Cola Company
    'MDLZ',   # Mondelez International Inc.
    
    # 半导体
    'MU',     # Micron Technology Inc.
    'MRVL',   # Marvell Technology Inc.
    'LRCX',   # Lam Research Corporation
    'AMAT',   # Applied Materials Inc.
    'KLAC',   # KLA Corporation
    
    # 通信和媒体
    'CMCSA',  # Comcast Corporation
    'CHTR',   # Charter Communications Inc.
    'TMUS',   # T-Mobile US Inc.
    'DISH',   # DISH Network Corporation
    
    # 金融科技
    'FISV',   # Fiserv Inc.
    'ADP',    # Automatic Data Processing Inc.
    'PAYX',   # Paychex Inc.
    'INTU',   # Intuit Inc.
    
    # 其他重要股票
    'BKNG',   # Booking Holdings Inc.
    'ABNB',   # Airbnb Inc.
    'UBER',   # Uber Technologies Inc.
    'LYFT',   # Lyft Inc.
    'SNAP',   # Snap Inc.
    'TWTR',   # Twitter Inc. (现为X)
    'PINS',   # Pinterest Inc.
    'ZM',     # Zoom Video Communications
    'ROKU',   # Roku Inc.
    'NTES',   # NetEase Inc.
    'JD',     # JD.com Inc.
    'BIDU',   # Baidu Inc.
    'PDD',    # PDD Holdings Inc.
]

# 按行业分类的股票组
TECH_GIANTS = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA']
SEMICONDUCTOR = ['INTC', 'AMD', 'QCOM', 'AVGO', 'MU', 'MRVL', 'LRCX', 'AMAT', 'KLAC']
BIOTECH = ['GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX']
CONSUMER = ['COST', 'SBUX', 'PEP', 'KO', 'MDLZ']
FINTECH = ['PYPL', 'FISV', 'ADP', 'PAYX', 'INTU']

# 高波动性股票 (适合短期交易)
HIGH_VOLATILITY = ['TSLA', 'NFLX', 'ZOOM', 'SNAP', 'ROKU', 'ABNB']

# 稳定股票 (适合长期投资)
STABLE_STOCKS = ['AAPL', 'MSFT', 'COST', 'PEP', 'KO', 'AMGN']

def get_nasdaq100_symbols():
    """获取完整的NASDAQ100股票列表"""
    return NASDAQ100_SYMBOLS

def get_symbols_by_category(category):
    """根据类别获取股票列表"""
    categories = {
        'tech_giants': TECH_GIANTS,
        'semiconductor': SEMICONDUCTOR,
        'biotech': BIOTECH,
        'consumer': CONSUMER,
        'fintech': FINTECH,
        'high_volatility': HIGH_VOLATILITY,
        'stable': STABLE_STOCKS
    }
    return categories.get(category, [])

def get_top_symbols(count=20):
    """获取前N个最重要的股票"""
    return NASDAQ100_SYMBOLS[:count]