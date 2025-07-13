import numpy as np
from sklearn.model_selection import KFold
from algorithms.momentumV5 import getMyPosition  # Import core strategy

def simulate_strategy(prcSoFar, params):
    """
    Simulate strategy with given parameters
    Returns annualized Sharpe ratio
    """
    daily_returns = []
    
    for day in range(params['n_days_lookback'] + 1, prcSoFar.shape[1]):
        positions = getMyPosition(prcSoFar[:, :day+1], params)
        pnl = np.sum(positions * (prcSoFar[:, day] - prcSoFar[:, day-1]))
        daily_returns.append(pnl)
    
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        return np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    return -np.inf

def optimize_parameters(price_data):
    """
    Find optimal parameters using 5-fold CV
    Returns dictionary of best parameters
    """
    lookback_grid = [20, 30, 40, 50]
    capital_grid = [2000, 3000, 4000]
    best_sharpe = -np.inf
    best_params = {'n_days_lookback': 30, 'capital_per_stock': 3000}
    
    kf = KFold(n_splits=5, shuffle=False)
    for train_idx, _ in kf.split(price_data.T):
        prc_train = price_data[:, train_idx]
        for n_days in lookback_grid:
            for capital in capital_grid:
                test_params = {'n_days_lookback': n_days, 'capital_per_stock': capital}
                sharpe = simulate_strategy(prc_train, test_params)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = test_params
    return best_params