# config.py
import pandas as pd
from datetime import datetime, timedelta

# 分析参数配置
class Config:
    # 时间范围
    START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # VaR参数
    CONFIDENCE_LEVEL = 0.95
    TIME_HORIZON = 1  # 天
    MONTE_CARLO_SIMULATIONS = 10000
    MONTE_CARLO_DAYS = 252
    
    # 资产配置
    DEFAULT_PORTFOLIO = {
        'AAPL': 0.25,   # Apple
        'MSFT': 0.20,   # Microsoft
        'GOOGL': 0.15,  # Google
        'AMZN': 0.15,   # Amazon
        'TSLA': 0.10,   # Tesla
        'SPY': 0.15     # S&P 500 ETF
    }
    
    # 随机种子
    RANDOM_SEED = 42