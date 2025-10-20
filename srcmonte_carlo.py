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
        """使用几何布朗运动进行蒙特卡洛模拟"""
        # 计算投资组合收益率
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # 计算参数
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        dt = 1  # 每天
        
        # 生成随机路径
        simulations = np.zeros((self.time_horizon, self.n_simulations))
        simulations[0] = initial_portfolio_value
        
        for t in range(1, self.time_horizon):
            # 生成随机 shocks
            shocks = np.random.normal(mean_return * dt, 
                                    std_return * np.sqrt(dt), 
                                    self.n_simulations)
            simulations[t] = simulations[t-1] * (1 + shocks)
        
        return simulations
    
    def monte_carlo_var(self, simulations, confidence_level=0.95):
        """基于蒙特卡洛模拟计算VaR"""
        # 计算最终价值分布
        final_values = simulations[-1, :]
        
        # 计算损益
        initial_value = simulations[0, 0]
        pnl = final_values - initial_value
        
        # 计算VaR
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
        """考虑资产相关性的蒙特卡洛模拟"""
        # 计算协方差矩阵
        cov_matrix = returns.cov()
        
        # Cholesky分解
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # 如果矩阵不是正定的，使用最近的正定矩阵
            from sklearn.covariance import ledoit_wolf
            cov_matrix = ledoit_wolf(returns)[0]
            L = np.linalg.cholesky(cov_matrix)
        
        n_assets = len(weights)
        dt = 1
        
        # 生成相关随机数
        simulations = np.zeros((self.time_horizon, self.n_simulations, n_assets))
        portfolio_values = np.zeros((self.time_horizon, self.n_simulations))
        
        for i in range(self.n_simulations):
            # 生成独立随机数
            Z = np.random.normal(0, 1, (self.time_horizon, n_assets))
            # 转换为相关随机数
            correlated_Z = Z @ L.T
            
            # 模拟每个资产的价格路径
            asset_paths = np.zeros((self.time_horizon, n_assets))
            asset_paths[0] = initial_portfolio_value * np.array(list(weights.values()))
            
            for t in range(1, self.time_horizon):
                returns_t = (returns.mean().values * dt + 
                           correlated_Z[t] * np.sqrt(dt))
                asset_paths[t] = asset_paths[t-1] * (1 + returns_t)
            
            simulations[:, i, :] = asset_paths
            portfolio_values[:, i] = asset_paths.sum(axis=1)
        
        return portfolio_values, simulations