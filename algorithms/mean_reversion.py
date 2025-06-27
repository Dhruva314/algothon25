import numpy as np

# Global variables required by eval.py
nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    """
    Main function called by eval.py for position decisions
    
    Args:
        prcSoFar: numpy array of shape (nInst, nt) containing price history
        
    Returns:
        numpy array of integers representing desired positions for each instrument
    """
    global currentPos
    
    (nInst, nt) = prcSoFar.shape
    
    # Need minimum data for reliable calculations
    if nt < 50:
        return np.zeros(nInst)
    
    # Strategy parameters (tune these for optimization)
    LOOKBACK_WINDOW = 40      # Days to calculate rolling mean
    Z_ENTRY_THRESHOLD = 1.5   # Z-score to enter position
    Z_EXIT_THRESHOLD = 0.4    # Z-score to exit position
    MAX_POSITION_PCT = 0.85   # Use 85% of $10k limit
    VOLATILITY_FILTER = 0.12  # Skip stocks with >12% daily volatility
    MIN_TRADE_SIZE = 50       # Minimum dollar amount to trade
    
    # Get current prices and calculate rolling statistics
    current_prices = prcSoFar[:, -1]
    price_window = prcSoFar[:, -LOOKBACK_WINDOW:]
    
    # Calculate rolling mean and standard deviation
    rolling_mean = np.mean(price_window, axis=1)
    rolling_std = np.std(price_window, axis=1) + 1e-8  # Avoid division by zero
    
    # Calculate z-scores (how many std devs from mean)
    z_scores = (current_prices - rolling_mean) / rolling_std
    
    # Calculate volatility filter (annualized daily volatility)
    if nt > 20:
        recent_prices = prcSoFar[:, -20:]
        returns = np.diff(recent_prices, axis=1) / recent_prices[:, :-1]
        volatility = np.std(returns, axis=1) * np.sqrt(252)
    else:
        volatility = np.zeros(nInst)
    
    # Initialize new positions
    new_positions = np.zeros(nInst)
    
    # Position sizing logic for each instrument
    for i in range(nInst):
        current_pos = currentPos[i]
        z_score = z_scores[i]
        price = current_prices[i]
        vol = volatility[i]
        
        # Skip if price is too low (penny stocks) or volatility too high
        if price < 2.0 or vol > VOLATILITY_FILTER:
            new_positions[i] = current_pos  # Keep existing position
            continue
        
        # Exit logic: close position if z-score has reverted
        if current_pos != 0 and abs(z_score) < Z_EXIT_THRESHOLD:
            new_positions[i] = 0
            continue
        
        # Entry logic: open new position if z-score is extreme
        if abs(z_score) > Z_ENTRY_THRESHOLD:
            # Calculate base position size
            # Negative z-score (price below mean) -> Long position
            # Positive z-score (price above mean) -> Short position
            base_dollar_size = -z_score * 3000  # Base sizing
            
            # Adjust for volatility (reduce size for volatile stocks)
            vol_adjustment = max(0.3, min(1.0, 0.08 / (vol + 0.01)))
            adjusted_dollar_size = base_dollar_size * vol_adjustment
            
            # Apply position limits ($10k per stock)
            max_dollar_position = 10000 * MAX_POSITION_PCT
            capped_dollar_size = np.clip(adjusted_dollar_size, 
                                       -max_dollar_position, 
                                       max_dollar_position)
            
            # Convert to integer shares
            target_shares = int(capped_dollar_size / price)
            
            # Only trade if the change is significant enough
            position_change_value = abs(target_shares - current_pos) * price
            if position_change_value > MIN_TRADE_SIZE:
                new_positions[i] = target_shares
            else:
                new_positions[i] = current_pos  # Keep current position
        else:
            # No strong signal, keep current position
            new_positions[i] = current_pos
    
    # Apply final position limits (safety check)
    for i in range(nInst):
        price = current_prices[i]
        max_shares = int(10000 / price)
        new_positions[i] = np.clip(new_positions[i], -max_shares, max_shares)
    
    # Update global position tracker
    currentPos = new_positions.copy()
    
    return new_positions.astype(int)


# Optional: Helper function for debugging (not called by eval.py)
def get_strategy_info(prcSoFar):
    """
    Returns current strategy statistics for analysis
    Only use this for debugging, not in actual competition
    """
    if prcSoFar.shape[1] < 50:
        return "Insufficient data"
    
    current_prices = prcSoFar[:, -1]
    
    # Calculate current portfolio value
    portfolio_value = np.sum(currentPos * current_prices)
    
    # Count positions
    num_long = np.sum(currentPos > 0)
    num_short = np.sum(currentPos < 0)
    num_zero = np.sum(currentPos == 0)
    
    return {
        'portfolio_value': portfolio_value,
        'positions': {'long': num_long, 'short': num_short, 'zero': num_zero},
        'total_absolute_position': np.sum(np.abs(currentPos))
    }