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
        """历史模拟法计算VaR"""
        # 计算投资组合收益率
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # 计算VaR
        var_historical = -np.percentile(portfolio_returns, 
                                      (1 - self.confidence_level) * 100)
        var_historical_value = var_historical * portfolio_value
        
        return {
            'var': var_historical,
            'var_value': var_historical_value,
            'portfolio_returns': portfolio_returns
        }
    
    def parametric_var(self, returns, weights, portfolio_value=1000000):
        """参数法计算VaR（方差-协方差方法）"""
        # 计算投资组合收益率
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # 计算均值和标准差
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # 计算VaR（使用正态分布）
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
        """计算Expected Shortfall (CVaR)"""
        var_threshold = -np.percentile(portfolio_returns, 
                                     (1 - self.confidence_level) * 100)
        
        # 找到超过VaR的损失
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
        """计算投资组合统计信息"""
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