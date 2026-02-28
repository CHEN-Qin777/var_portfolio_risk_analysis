# config.py
import pandas as pd
from datetime import datetime, timedelta

# Configuration des paramètres d'analyse
class Config:
    # Période
    START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Paramètres de la VaR
    CONFIDENCE_LEVEL = 0.95
    TIME_HORIZON = 1  # 天
    MONTE_CARLO_SIMULATIONS = 10000
    MONTE_CARLO_DAYS = 252
    
    # Répartition des actifs
    DEFAULT_PORTFOLIO = {
        'AAPL': 0.25,   # Apple
        'MSFT': 0.20,   # Microsoft
        'GOOGL': 0.15,  # Google
        'AMZN': 0.15,   # Amazon
        'TSLA': 0.10,   # Tesla
        'SPY': 0.15     # S&P 500 ETF
    }
    
    # Graine aléatoire

    RANDOM_SEED = 42
