import numpy as np
import pandas as pd
import os
from algorithms.momentumV5 import getMyPosition
from replim_analysis.optimize import optimize_parameters

def loadPrices(fn):
    """Load price data"""
    df = pd.read_csv(fn, sep='\s+', header=None)
    return df.values.T

if __name__ == "__main__":
    # Load data
    price_file = os.path.join(os.path.dirname(__file__), '..', 'prices.txt')
    price_data = loadPrices(price_file)
    nInst, nDays = price_data.shape
    
    # Optimize parameters
    if nDays >= 100:
        optimized_params = optimize_parameters(price_data)
        print(f"Optimized Parameters: {optimized_params}")
    else:
        optimized_params = None
        print("Using default parameters (insufficient data)")
    
    # Run strategy
    portfolio_values = []
    start_day = (optimized_params['n_days_lookback'] if optimized_params else 30) + 1
    
    for day in range(start_day, nDays):
        positions = getMyPosition(price_data[:, :day+1], optimized_params)
        pnl = np.sum(positions * (price_data[:, day] - price_data[:, day-1]))
        portfolio_values.append(pnl)
    
    # Print results
    if portfolio_values:
        sharpe = np.mean(portfolio_values)/np.std(portfolio_values)*np.sqrt(252)
        print(f"Annualized Sharpe: {sharpe:.2f}")