# src/var_calculator.py
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class VaRCalculator:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
    
    def historical_var(self, returns, weights, portfolio_value=1000000):
        """Calcul de la VaR par simulation historique"""
        # Calcul du rendement du portefeuille
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calcul de la VaR
        var_historical = -np.percentile(portfolio_returns, 
                                      (1 - self.confidence_level) * 100)
        var_historical_value = var_historical * portfolio_value
        
        return {
            'var': var_historical,
            'var_value': var_historical_value,
            'portfolio_returns': portfolio_returns
        }
    
    def parametric_var(self, returns, weights, portfolio_value=1000000):
        """Calcul de la VaR paramétrique (méthode variance-covariance)"""
        # Calcul du rendement du portefeuille
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calcul de la moyenne et de l'écart-type
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Calcul de la VaR (en utilisant la distribution normale)
        z_score = stats.norm.ppf(self.confidence_level)
        var_parametric = -(mean_return - z_score * std_return)
        var_parametric_value = var_parametric * portfolio_value
        
        return {
            'var': var_parametric,
            'var_value': var_parametric_value,
            'mean': mean_return,
            'std': std_return
        }
    
    def calculate_expected_shortfall(self, portfolio_returns, portfolio_value=1000000):
        """Calcul de l'Expected Shortfall (CVaR)"""
        var_threshold = -np.percentile(portfolio_returns, 
                                     (1 - self.confidence_level) * 100)
        
        # Recherche des pertes dépassant la VaR
        tail_losses = portfolio_returns[portfolio_returns <= -var_threshold]
        
        if len(tail_losses) > 0:
            expected_shortfall = -tail_losses.mean()
            expected_shortfall_value = expected_shortfall * portfolio_value
        else:
            expected_shortfall = var_threshold
            expected_shortfall_value = expected_shortfall * portfolio_value
        
        return {
            'es': expected_shortfall,
            'es_value': expected_shortfall_value,
            'tail_losses': tail_losses
        }
    
    def calculate_portfolio_stats(self, returns, weights, portfolio_value=1000000):
        """Calcul des statistiques du portefeuille"""
        portfolio_returns = (returns * weights).sum(axis=1)
        
        stats_dict = {
            'portfolio_value': portfolio_value,
            'mean_daily_return': portfolio_returns.mean(),
            'volatility': portfolio_returns.std(),
            'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis(),
            'min_return': portfolio_returns.min(),
            'max_return': portfolio_returns.max()
        }
        
        return stats_dict
