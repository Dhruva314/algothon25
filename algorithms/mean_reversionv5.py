import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

nInst = 50
currentPos = np.zeros(nInst)

# Default parameters
DEFAULT_PARAMS = {
    'short_window': 15,
    'long_window': 40,
    'entry_z': 1.2,
    'exit_z': 0.3,
    'max_position_pct': 0.8,
    'volatility_cap': 0.12,
    'min_price': 10.0,
    'trend_threshold': 0.8,
    'position_decay': 0.85,
    'min_trade_size': 200,
    'max_trades': 15
}

# Parameter grid for optimization
param_grid = {
    'short_window': [10, 15, 20],
    'long_window': [30, 40, 50],
    'entry_z': [1.0, 1.2, 1.5],
    'exit_z': [0.3, 0.5],
    'volatility_cap': [0.1, 0.12]
}

def getMyPosition(prcSoFar):
    global currentPos
    
    (nInst, nt) = prcSoFar.shape
    params = dict(DEFAULT_PARAMS)
    
    if nt > 50:  # Only optimize with sufficient data
        try:
            optimized_params = optimize_parameters(prcSoFar)
            params.update({k:v for k,v in optimized_params.items() if k in param_grid})
        except Exception as e:
            print(f"Optimization skipped: {str(e)}")
    
    if nt < max(params['short_window'], params['long_window']) + 5:
        return np.zeros(nInst)
    
    current_prices = prcSoFar[:, -1]
    indicators = calc_indicators(prcSoFar, params)
    
    valid = (
        (current_prices >= params['min_price']) &
        (indicators['volatility'] <= params['volatility_cap']) &
        (np.abs(indicators['trend_strength']) < params['trend_threshold'])
    )
    
    target_positions = generate_signals(indicators, current_prices, currentPos, params, valid)
    
    max_shares = (10000 / current_prices).astype(int)
    target_positions = np.clip(target_positions, -max_shares, max_shares)
    
    currentPos = target_positions.astype(int)
    return currentPos

def calc_indicators(prices, params):
    """Calculate technical indicators"""
    short_prices = prices[:, -params['short_window']:]
    long_prices = prices[:, -params['long_window']:]
    
    short_ma = np.mean(short_prices, axis=1)
    long_ma = np.mean(long_prices, axis=1)
    long_std = np.std(long_prices, axis=1) + 1e-8
    
    returns = np.diff(short_prices, axis=1) / short_prices[:, :-1]
    volatility = np.std(returns, axis=1) * np.sqrt(252)
    trend_strength = short_ma / long_ma - 1
    
    return {
        'z_scores': (prices[:, -1] - long_ma) / long_std,
        'volatility': volatility,
        'trend_strength': trend_strength,
        'short_ma': short_ma,
        'long_ma': long_ma
    }

def generate_signals(indicators, current_prices, currentPos, params, valid):
    """Generate trading signals"""
    target_positions = np.zeros(nInst)
    active_positions = 0
    z_rank = np.argsort(np.abs(indicators['z_scores']))[::-1]
    
    for i in z_rank:
        if not valid[i] or active_positions >= params['max_trades']:
            continue
            
        z = indicators['z_scores'][i]
        price = current_prices[i]
        
        # Exit logic
        if currentPos[i] != 0 and abs(z) < params['exit_z']:
            target_positions[i] = 0
            active_positions -= 1
            continue
            
        # Entry logic
        if abs(z) > params['entry_z']:
            size = -np.sign(z) * min(
                2500 / (indicators['volatility'][i] + 0.05),
                10000 * params['max_position_pct']
            )
            shares = int(size / price)
            
            if abs(shares - currentPos[i]) * price > params['min_trade_size']:
                target_positions[i] = shares
                active_positions += 1
            else:
                target_positions[i] = currentPos[i] * params['position_decay']
        else:
            target_positions[i] = currentPos[i] * params['position_decay']
    
    return target_positions

def optimize_parameters(price_data, n_splits=3):
    """Time-series parameter optimization"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = -np.inf
    best_params = {}
    
    for params in [dict(zip(param_grid.keys(), v)) 
                  for v in product(*param_grid.values())][:10]:
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(price_data.T):
            train_data = price_data[:, train_idx]
            val_data = price_data[:, val_idx]
            
            # Combine with default params
            eval_params = dict(DEFAULT_PARAMS)
            eval_params.update(params)
            
            # Evaluate
            score = evaluate_params(train_data, val_data, eval_params)
            fold_scores.append(score)
        
        avg_score = np.mean(fold_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    return best_params

def evaluate_params(train_data, val_data, params):
    """Evaluate parameter set on validation period"""
    positions = np.zeros(nInst)
    daily_pl = []
    
    for t in range(1, val_data.shape[1]):
        hist_prc = np.hstack([train_data, val_data[:, :t]])
        current_prices = val_data[:, t]
        
        indicators = calc_indicators(hist_prc, params)
        valid = (
            (current_prices >= params['min_price']) &
            (indicators['volatility'] <= params['volatility_cap']) &
            (np.abs(indicators['trend_strength']) < params['trend_threshold'])
        )
        
        new_pos = generate_signals(indicators, current_prices, positions, params, valid)
        pl = np.sum(positions * (current_prices - val_data[:, t-1]))
        daily_pl.append(pl)
        positions = new_pos
    
    return np.mean(daily_pl) - 0.1 * np.std(daily_pl) if daily_pl else -np.inf