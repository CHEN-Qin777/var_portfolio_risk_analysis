```python
# src/data_loader.py
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
    
    def download_market_data(self, symbols):
        """Télécharge les données de marché depuis Yahoo Finance"""
        print(f"Téléchargement des données pour {len(symbols)} actifs...")
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=self.start_date, end=self.end_date)
                if not hist_data.empty:
                    data[symbol] = hist_data['Close']
                    print(f"✓ {symbol}: {len(hist_data)} points de données")
                else:
                    print(f"✗ {symbol}: Aucune donnée")
            except Exception as e:
                print(f"✗ {symbol}: Erreur - {e}")
        
        return pd.DataFrame(data)
    
    def calculate_returns(self, prices):
        """Calcule les rendements quotidiens"""
        returns = prices.pct_change().dropna()
        return returns
    
    def generate_sample_data(self, symbols, portfolio_weights):
        """Génère des données d'exemple"""
        print("Génération des données d'exemple du portefeuille...")
        
        # Téléchargement des données de marché
        prices = self.download_market_data(symbols)
        
        if prices.empty:
            raise ValueError("Impossible de télécharger les données de marché")
        
        # Calcul des rendements
        returns = self.calculate_returns(prices)
        
        # Création des données du portefeuille
        portfolio_data = {
            'prices': prices,
            'returns': returns,
            'weights': portfolio_weights
        }
        
        return portfolio_data
    
    def save_data_to_csv(self, data, filename):
        """Enregistre les données au format CSV"""
        data.to_csv(filename, index=True)
        print(f"Données enregistrées dans {filename}")
```
