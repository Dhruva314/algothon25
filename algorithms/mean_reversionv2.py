import numpy as np

nInst = 50
currentPos = np.zeros(nInst)
lastPrices = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos, lastPrices
    
    (nInst, nt) = prcSoFar.shape
    
    # TUNED PARAMETERS
    params = {
        'short_window': 15,       # More responsive to recent changes
        'long_window': 40,        # Shorter mean reversion period
        'entry_z': 1.2,           # More sensitive entry threshold
        'exit_z': 0.3,            # Earlier exits
        'max_position_pct': 0.8,  # More conservative sizing
        'volatility_cap': 0.12,   # Lower volatility tolerance
        'min_price': 10.0,       # Avoid cheaper, more volatile stocks
        'trend_threshold': 0.8,   # Stricter trend avoidance
        'position_decay': 0.85,   # Faster position reduction
        'min_trade_size': 200,    # Fewer, more meaningful trades
        'max_trades': 15          # Limit simultaneous positions
    }
    
    if nt < max(params['short_window'], params['long_window']) + 5:
        return np.zeros(nInst)
    
    current_prices = prcSoFar[:, -1]
    
    # Calculate indicators
    def calc_indicators(prices):
        short_prices = prices[:, -params['short_window']:]
        long_prices = prices[:, -params['long_window']:]
        
        short_ma = np.mean(short_prices, axis=1)
        long_ma = np.mean(long_prices, axis=1)
        long_std = np.std(long_prices, axis=1) + 1e-8
        
        returns = np.diff(short_prices, axis=1) / short_prices[:, :-1]
        volatility = np.std(returns, axis=1) * np.sqrt(252)
        trend_strength = short_ma / long_ma - 1
        
        return {
            'z_scores': (current_prices - long_ma) / long_std,
            'volatility': volatility,
            'trend_strength': trend_strength
        }
    
    indicators = calc_indicators(prcSoFar)
    
    # Apply strict filters
    valid = (
        (current_prices >= params['min_price']) &
        (indicators['volatility'] <= params['volatility_cap']) &
        (np.abs(indicators['trend_strength']) < params['trend_threshold'])
    )
    
    # Rank instruments by trading signal strength
    z_rank = np.argsort(np.abs(indicators['z_scores']))[::-1]
    
    target_positions = np.zeros(nInst)
    active_positions = 0
    
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
    
    # Enforce position limits
    max_shares = (10000 / current_prices).astype(int)
    target_positions = np.clip(target_positions, -max_shares, max_shares)
    
    currentPos = target_positions.astype(int)
    return currentPos
