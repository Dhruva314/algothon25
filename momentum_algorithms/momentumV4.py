import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape
    
    # Parameters - optimized for better performance
    momentum_lookback = 30  # Reduced from 45 for more responsive signals
    vol_lookback = 20       # Reduced for more adaptive volatility
    capital = 3000
    
    # Need sufficient data
    if nt < momentum_lookback + 1:
        return np.zeros(nins)
    
    # 1. Multiple momentum signals
    log_returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    
    # A) Short-term momentum (5-day)
    if log_returns.shape[1] >= 5:
        short_mom = np.mean(log_returns[:, -5:], axis=1)
    else:
        short_mom = np.zeros(nins)
    
    # B) Medium-term exponentially weighted momentum
    if log_returns.shape[1] >= momentum_lookback:
        weights = np.exp(-0.1 * np.arange(momentum_lookback))[::-1]
        weights /= np.sum(weights)
        med_mom = np.dot(log_returns[:, -momentum_lookback:], weights)
    else:
        med_mom = np.zeros(nins)
    
    # C) Long-term momentum (if we have enough data)
    long_lookback = min(60, log_returns.shape[1])
    if long_lookback >= 20:
        long_mom = np.mean(log_returns[:, -long_lookback:], axis=1)
    else:
        long_mom = np.zeros(nins)
    
    # Combine momentum signals
    momentum = 0.5 * short_mom + 0.3 * med_mom + 0.2 * long_mom
    
    # 2. Adaptive volatility estimation
    if log_returns.shape[1] >= vol_lookback:
        # Use EWMA for volatility with higher weight on recent observations
        vol_weights = np.exp(-0.15 * np.arange(vol_lookback))[::-1]
        vol_weights /= np.sum(vol_weights)
        squared_returns = log_returns[:, -vol_lookback:] ** 2
        vol = np.sqrt(np.dot(squared_returns, vol_weights)) + 1e-8
    else:
        vol = np.std(log_returns, axis=1) + 1e-8
    
    # 3. Risk-adjusted momentum
    risk_adj_momentum = momentum / vol
    
    # 4. Market neutralization (improved)
    if log_returns.shape[1] >= momentum_lookback:
        # Use equal-weighted market proxy
        market_returns = np.mean(log_returns[:, -momentum_lookback:], axis=0)
        
        # Calculate rolling beta more robustly
        betas = np.zeros(nins)
        for i in range(nins):
            if np.var(market_returns) > 1e-8:
                betas[i] = np.cov(log_returns[i, -momentum_lookback:], market_returns)[0,1] / np.var(market_returns)
        
        # Market-neutral score
        market_neutral_score = risk_adj_momentum - 0.5 * betas  # Reduced beta impact
    else:
        market_neutral_score = risk_adj_momentum
    
    # 5. Cross-sectional ranking and scoring
    # Use ranks instead of z-scores for more stable signals
    ranks = np.argsort(np.argsort(market_neutral_score))
    normalized_ranks = (ranks - (nins - 1) / 2) / (nins - 1)  # Scale to [-0.5, 0.5]
    
    # Apply sigmoid transformation for smoother position sizing
    signals = np.tanh(3 * normalized_ranks)  # Smooth between -1 and 1
    
    # 6. Position sizing with risk management
    # Dollar-neutral constraint
    long_signals = np.maximum(signals, 0)
    short_signals = np.maximum(-signals, 0)
    
    # Normalize to ensure dollar neutrality
    long_weight = np.sum(long_signals)
    short_weight = np.sum(short_signals)
    
    if long_weight > 0 and short_weight > 0:
        long_signals = long_signals / long_weight * 0.5
        short_signals = short_signals / short_weight * 0.5
        final_weights = long_signals - short_signals
    else:
        final_weights = signals / (np.sum(np.abs(signals)) + 1e-8)
    
    # 7. Concentration limits and turnover control
    # Limit maximum position size
    max_weight = 0.15  # Max 15% of capital per position
    final_weights = np.clip(final_weights, -max_weight, max_weight)
    
    # Calculate target positions
    target_dollars = capital * final_weights
    target_positions = target_dollars / prcSoFar[:, -1]
    
    # 8. Turnover control (smooth transitions)
    if np.any(currentPos != 0):
        # Limit position changes to reduce transaction costs
        max_change = 0.3  # Max 30% position change per period
        current_dollars = currentPos * prcSoFar[:, -1]
        current_weights = current_dollars / capital
        
        weight_change = final_weights - current_weights
        weight_change = np.clip(weight_change, -max_change, max_change)
        
        adjusted_weights = current_weights + weight_change
        adjusted_dollars = capital * adjusted_weights
        target_positions = adjusted_dollars / prcSoFar[:, -1]
    
    # Round to integer positions
    currentPos = np.round(target_positions).astype(int)
    
    # Final check: ensure we don't exceed reasonable position limits
    max_pos_size = int(capital * 0.2 / np.mean(prcSoFar[:, -1]))  # Max position size
    currentPos = np.clip(currentPos, -max_pos_size, max_pos_size)
    
    return currentPos

"""
mean(PL): -0.2
return: -0.00026
StdDev(PL): 4.60
annSharpe(PL): -0.66 
totDvolume: 155825 
Score: -0.65
"""