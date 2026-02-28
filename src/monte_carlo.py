# src/monte_carlo.py
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulator:
    def __init__(self, n_simulations=10000, time_horizon=252, random_seed=42):
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def simulate_gbm(self, returns, weights, initial_portfolio_value=1000000):
        """Simulation Monte-Carlo utilisant le mouvement brownien géométrique"""
        # Calcul du rendement du portefeuille
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calcul des paramètres
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        dt = 1  # quotidien
        
        # Génération des trajectoires aléatoires
        simulations = np.zeros((self.time_horizon, self.n_simulations))
        simulations[0] = initial_portfolio_value
        
        for t in range(1, self.time_horizon):
            # Génération des chocs aléatoires
            shocks = np.random.normal(mean_return * dt, 
                                    std_return * np.sqrt(dt), 
                                    self.n_simulations)
            simulations[t] = simulations[t-1] * (1 + shocks)
        
        return simulations
    
    def monte_carlo_var(self, simulations, confidence_level=0.95):
        """Calcul de la VaR basée sur la simulation Monte-Carlo"""
        # Calcul de la distribution des valeurs finales
        final_values = simulations[-1, :]
        
        # Calcul des profits/pertes (P&L)
        initial_value = simulations[0, 0]
        pnl = final_values - initial_value
        
        # Calcul de la VaR
        var_mc = -np.percentile(pnl, (1 - confidence_level) * 100)
        var_mc_percentage = var_mc / initial_value
        
        return {
            'var': var_mc_percentage,
            'var_value': var_mc,
            'final_values': final_values,
            'pnl_distribution': pnl,
            'simulations': simulations
        }
    
    def correlated_mc_simulation(self, returns, weights, initial_portfolio_value=1000000):
        """Simulation Monte-Carlo prenant en compte la corrélation des actifs"""
        # Calcul de la matrice de covariance
        cov_matrix = returns.cov()
        
        # Décomposition de Cholesky
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Si la matrice n'est pas définie positive, utiliser la matrice définie positive la plus proche
            from sklearn.covariance import ledoit_wolf
            cov_matrix = ledoit_wolf(returns)[0]
            L = np.linalg.cholesky(cov_matrix)
        
        n_assets = len(weights)
        dt = 1
        
        # Génération des nombres aléatoires corrélés
        simulations = np.zeros((self.time_horizon, self.n_simulations, n_assets))
        portfolio_values = np.zeros((self.time_horizon, self.n_simulations))
        
        for i in range(self.n_simulations):
            # Génération des nombres aléatoires indépendants
            Z = np.random.normal(0, 1, (self.time_horizon, n_assets))
            # Conversion en nombres aléatoires corrélés
            correlated_Z = Z @ L.T
            
            # Simulation des trajectoires de prix pour chaque actif
            asset_paths = np.zeros((self.time_horizon, n_assets))
            asset_paths[0] = initial_portfolio_value * np.array(list(weights.values()))
            
            for t in range(1, self.time_horizon):
                returns_t = (returns.mean().values * dt + 
                           correlated_Z[t] * np.sqrt(dt))
                asset_paths[t] = asset_paths[t-1] * (1 + returns_t)
            
            simulations[:, i, :] = asset_paths
            portfolio_values[:, i] = asset_paths.sum(axis=1)
        
        return portfolio_values, simulations
