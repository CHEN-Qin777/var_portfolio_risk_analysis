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
        """从Yahoo Finance下载市场数据"""
        print(f"下载 {len(symbols)} 种资产的数据...")
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=self.start_date, end=self.end_date)
                if not hist_data.empty:
                    data[symbol] = hist_data['Close']
                    print(f"✓ {symbol}: {len(hist_data)} 个数据点")
                else:
                    print(f"✗ {symbol}: 无数据")
            except Exception as e:
                print(f"✗ {symbol}: 错误 - {e}")
        
        return pd.DataFrame(data)
    
    def calculate_returns(self, prices):
        """计算日收益率"""
        returns = prices.pct_change().dropna()
        return returns
    
    def generate_sample_data(self, symbols, portfolio_weights):
        """生成示例数据"""
        print("生成示例投资组合数据...")
        
        # 下载市场数据
        prices = self.download_market_data(symbols)
        
        if prices.empty:
            raise ValueError("无法下载市场数据")
        
        # 计算收益率
        returns = self.calculate_returns(prices)
        
        # 创建投资组合数据
        portfolio_data = {
            'prices': prices,
            'returns': returns,
            'weights': portfolio_weights
        }
        
        return portfolio_data
    
    def save_data_to_csv(self, data, filename):
        """保存数据到CSV"""
        data.to_csv(filename, index=True)
        print(f"数据已保存到 {filename}")