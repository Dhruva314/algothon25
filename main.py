
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

nInst = 50
currentPos = np.zeros(nInst)

# def getMyPositionMeanReversion(prcSoFar):
#     global currentPos
#     nins, nt = prcSoFar.shape
#     lookback = 50  # Increased for better signal
#     vol_lookback = 45
#     capital = 3000
#     max_position_pct = 0.2  # Max 20% per instrument

#     if nt < lookback + 1:
#         return np.zeros(nins)

#     # Calculate log returns
#     log_returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    
#     if log_returns.shape[1] < lookback:
#         return np.zeros(nins)

#     # 1. Multi-timeframe momentum
#     # Short-term (10 days), medium-term (30 days), long-term (60 days)
#     short_mom = np.mean(log_returns[:, -20:], axis=1)
#     med_mom = np.mean(log_returns[:, -40:], axis=1)
#     long_mom = np.mean(log_returns[:, -lookback*5:], axis=1)
    
#     # Weighted combination favoring recent momentum
#     momentum = 0.2 * short_mom + 0.6 * med_mom + 0.2 * long_mom

#     # 2. Enhanced volatility calculation with GARCH-like weighting
#     vol_weights = np.exp(-0.1 * np.arange(vol_lookback))[::-1]
#     vol_weights /= np.sum(vol_weights)
    
#     # Calculate weighted volatility
#     volatility = np.zeros(nins)
#     for i in range(nins):
#         if log_returns.shape[1] >= vol_lookback:
#             recent_returns = log_returns[i, -vol_lookback:]
#             volatility[i] = np.sqrt(np.sum(vol_weights * recent_returns**2))
#     volatility = np.maximum(volatility, 1e-6)

#     # 3. Mean reversion component (contrarian signal)
#     # Look for extreme moves that might revert
#     recent_change = (prcSoFar[:, -1] / prcSoFar[:, -9] - 1)  # 10-day change
#     mean_reversion = -np.tanh(recent_change * 30)  # Contrarian signal

#     # 4. Market regime detection
#     market_returns = np.mean(log_returns, axis=0)
#     market_vol = np.std(market_returns[-30:]) if log_returns.shape[1] >= 30 else 0.02
    
#     # Adjust strategy based on market volatility regime
#     regime_factor = np.clip(0.02 / (market_vol + 1e-6), 0.5, 2.0)

#     # 5. Improved beta calculation with rolling window
#     market_returns_recent = market_returns[-lookback:]
#     betas = np.zeros(nins)
#     for i in range(nins):
#         inst_returns = log_returns[i, -lookback:]
#         if len(inst_returns) == len(market_returns_recent):
#             beta_calc = np.cov(inst_returns, market_returns_recent)[0,1] / (np.var(market_returns_recent) + 1e-8)
#             betas[i] = beta_calc

#     # 6. Combined signal with multiple factors
#     # Risk-adjusted momentum
#     risk_adj_momentum = momentum / volatility
    
#     # Combine momentum and mean reversion based on volatility
#     # High vol -> more mean reversion, Low vol -> more momentum
#     vol_regime = np.percentile(volatility, 50)
#     mr_weight = np.minimum(volatility / (vol_regime + 1e-6), 1.0) * 0.3
#     momentum_weight = 1.0 - mr_weight
    
#     combined_signal = (momentum_weight * risk_adj_momentum + 
#                       mr_weight * mean_reversion)
    
#     # Apply beta neutralization more selectively
#     beta_adj_signal = combined_signal - 0.5 * betas  # Partial beta hedge
    
#     # 7. Improved cross-sectional ranking
#     # Use robust statistics
#     signal_median = np.median(beta_adj_signal)
#     signal_mad = np.median(np.abs(beta_adj_signal - signal_median))
#     robust_zscore = (beta_adj_signal - signal_median) / (signal_mad * 1.4826 + 1e-8)
    
#     # Apply regime factor
#     robust_zscore *= regime_factor
    
#     # More aggressive clipping for better risk control
#     robust_zscore = np.clip(robust_zscore, -2.5, 2.5)

#     # 8. Enhanced position sizing
#     # Apply Kelly criterion-inspired sizing
#     expected_return = np.abs(robust_zscore) * 0.001  # Expected daily return
#     kelly_fraction = expected_return / (volatility**2 + 1e-6)
#     kelly_fraction = np.clip(kelly_fraction, 0, 0.5)  # Max 50% Kelly
    
#     # Combine z-score direction with Kelly sizing
#     position_weights = np.sign(robust_zscore) * kelly_fraction
    
#     # Normalize to dollar neutral
#     long_weights = np.maximum(position_weights, 0)
#     short_weights = np.minimum(position_weights, 0)
    
#     long_sum = np.sum(long_weights)
#     short_sum = np.sum(np.abs(short_weights))
    
#     if long_sum > 0 and short_sum > 0:
#         # Balance long and short sides
#         balance_factor = min(long_sum, short_sum)
#         if long_sum > short_sum:
#             long_weights *= balance_factor / long_sum
#         else:
#             short_weights *= balance_factor / short_sum
#         position_weights = long_weights + short_weights

#     # Final normalization
#     total_weight = np.sum(np.abs(position_weights))
#     if total_weight > 0:
#         position_weights /= total_weight

#     # 9. Position limits and turnover control
#     target_dollars = capital * position_weights
#     target_positions = target_dollars / prcSoFar[:, -1]
    
#     # Limit maximum position size
#     max_shares = capital * max_position_pct / prcSoFar[:, -1]
#     target_positions = np.clip(target_positions, -max_shares, max_shares)
    
#     # Turnover control - limit changes to reduce transaction costs
#     if nt > lookback + 1:  # Not first iteration
#         position_change = target_positions - currentPos
#         max_change = np.abs(currentPos) * 0.3  # Max 30% position change
#         max_change = np.maximum(max_change, 10)  # But at least 10 shares
        
#         position_change = np.clip(position_change, -max_change, max_change)
#         target_positions = currentPos + position_change

#     currentPos = np.round(target_positions).astype(int)
#     return currentPos



##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

MAX_POS = 10000
nInst = 50

cointegration_vectors = [
    [38.53501255, 27.93024285], # Set 1 (50,2)
    [ 16.87897605,  -9.89043988, -13.81384429,  13.84605889, -30.95955299], # Set 2
    [ 15.91622771, -19.98080868] # Set 3 (24,12)
]

weights = pd.DataFrame({
    "vectors": cointegration_vectors,
    "sets": [
        (49, 1),                    # (50-1, 2-1)
        (4, 22, 6, 2, 20),          # (5-1, 23-1, 7-1, 3-1, 21-1)
        (23, 11)                    # (24-1, 12-1)
    ]
})

def getBaselinePositions(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape
    lookback = 50  # Increased for better signal
    vol_lookback = 45
    capital = 3000
    max_position_pct = 0.2  # Max 20% per instrument

    if nt < lookback + 1:
        return np.zeros(nins)

    # Calculate log returns
    log_returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    
    if log_returns.shape[1] < lookback:
        return np.zeros(nins)

    # 1. Multi-timeframe momentum
    # Short-term (10 days), medium-term (30 days), long-term (60 days)
    short_mom = np.mean(log_returns[:, -20:], axis=1)
    med_mom = np.mean(log_returns[:, -40:], axis=1)
    long_mom = np.mean(log_returns[:, -lookback*5:], axis=1)
    
    # Weighted combination favoring recent momentum
    momentum = 0.2 * short_mom + 0.6 * med_mom + 0.2 * long_mom

    # 2. Enhanced volatility calculation with GARCH-like weighting
    vol_weights = np.exp(-0.1 * np.arange(vol_lookback))[::-1]
    vol_weights /= np.sum(vol_weights)
    
    # Calculate weighted volatility
    volatility = np.zeros(nins)
    for i in range(nins):
        if log_returns.shape[1] >= vol_lookback:
            recent_returns = log_returns[i, -vol_lookback:]
            volatility[i] = np.sqrt(np.sum(vol_weights * recent_returns**2))
    volatility = np.maximum(volatility, 1e-6)

    # 3. Mean reversion component (contrarian signal)
    # Look for extreme moves that might revert
    recent_change = (prcSoFar[:, -1] / prcSoFar[:, -9] - 1)  # 10-day change
    mean_reversion = -np.tanh(recent_change * 30)  # Contrarian signal

    # 4. Market regime detection
    market_returns = np.mean(log_returns, axis=0)
    market_vol = np.std(market_returns[-30:]) if log_returns.shape[1] >= 30 else 0.02
    
    # Adjust strategy based on market volatility regime
    regime_factor = np.clip(0.02 / (market_vol + 1e-6), 0.5, 2.0)

    # 5. Improved beta calculation with rolling window
    market_returns_recent = market_returns[-lookback:]
    betas = np.zeros(nins)
    for i in range(nins):
        inst_returns = log_returns[i, -lookback:]
        if len(inst_returns) == len(market_returns_recent):
            beta_calc = np.cov(inst_returns, market_returns_recent)[0,1] / (np.var(market_returns_recent) + 1e-8)
            betas[i] = beta_calc

    # 6. Combined signal with multiple factors
    # Risk-adjusted momentum
    risk_adj_momentum = momentum / volatility
    
    # Combine momentum and mean reversion based on volatility
    # High vol -> more mean reversion, Low vol -> more momentum
    vol_regime = np.percentile(volatility, 50)
    mr_weight = np.minimum(volatility / (vol_regime + 1e-6), 1.0) * 0.3
    momentum_weight = 1.0 - mr_weight
    
    combined_signal = (momentum_weight * risk_adj_momentum + 
                      mr_weight * mean_reversion)
    
    # Apply beta neutralization more selectively
    beta_adj_signal = combined_signal - 0.5 * betas  # Partial beta hedge
    
    # 7. Improved cross-sectional ranking
    # Use robust statistics
    signal_median = np.median(beta_adj_signal)
    signal_mad = np.median(np.abs(beta_adj_signal - signal_median))
    robust_zscore = (beta_adj_signal - signal_median) / (signal_mad * 1.4826 + 1e-8)
    
    # Apply regime factor
    robust_zscore *= regime_factor
    
    # More aggressive clipping for better risk control
    robust_zscore = np.clip(robust_zscore, -2.5, 2.5)

    # 8. Enhanced position sizing
    # Apply Kelly criterion-inspired sizing
    expected_return = np.abs(robust_zscore) * 0.001  # Expected daily return
    kelly_fraction = expected_return / (volatility**2 + 1e-6)
    kelly_fraction = np.clip(kelly_fraction, 0, 0.5)  # Max 50% Kelly
    
    # Combine z-score direction with Kelly sizing
    position_weights = np.sign(robust_zscore) * kelly_fraction
    
    # Normalize to dollar neutral
    long_weights = np.maximum(position_weights, 0)
    short_weights = np.minimum(position_weights, 0)
    
    long_sum = np.sum(long_weights)
    short_sum = np.sum(np.abs(short_weights))
    
    if long_sum > 0 and short_sum > 0:
        # Balance long and short sides
        balance_factor = min(long_sum, short_sum)
        if long_sum > short_sum:
            long_weights *= balance_factor / long_sum
        else:
            short_weights *= balance_factor / short_sum
        position_weights = long_weights + short_weights

    # Final normalization
    total_weight = np.sum(np.abs(position_weights))
    if total_weight > 0:
        position_weights /= total_weight

    # 9. Position limits and turnover control
    target_dollars = capital * position_weights
    target_positions = target_dollars / prcSoFar[:, -1]
    
    # Limit maximum position size
    max_shares = capital * max_position_pct / prcSoFar[:, -1]
    target_positions = np.clip(target_positions, -max_shares, max_shares)
    
    # Turnover control - limit changes to reduce transaction costs
    if nt > lookback + 1:  # Not first iteration
        position_change = target_positions - currentPos
        max_change = np.abs(currentPos) * 0.3  # Max 30% position change
        max_change = np.maximum(max_change, 10)  # But at least 10 shares
        
        position_change = np.clip(position_change, -max_change, max_change)
        target_positions = currentPos + position_change

    currentPos = np.round(target_positions).astype(int)
    return currentPos

def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape
    if nt < 50:
        return np.zeros(nins)

    # === 1. Run your enhanced mean reversion + momentum strategy ===
    baseline_positions = getBaselinePositions(prcSoFar)  # This is your first big getMyPosition function's logic but returns positions
    
    # === 2. Log prices for pairwise trading ===
    log_prices = np.log(prcSoFar)
    
    # Copy baseline positions to adjust with pairs
    currentPos = baseline_positions.copy()
    
    # === 3. Iterate over cointegrated pairs/sets ===
    for _, row in weights.iterrows():
        vec = np.array(row['vectors'])
        subset_indices = list(row['sets'])

        subset = log_prices[subset_indices].T
        spread = subset @ vec
        spread_series = pd.Series(spread)
        
        # Calculate rolling z-score for spread (20-day window)
        rolling_mean = spread_series.rolling(20).mean()
        rolling_std = spread_series.rolling(20).std()
        zscore = (spread_series - rolling_mean) / rolling_std
        
        score = zscore.iloc[-1]

        # Only trade pair if abs(zscore) > 1.5
        if abs(score) > 1.5:
            size_frac = np.sign(score) * min(abs(score), 3) / 3
            latest_prices = prcSoFar[subset_indices, -1]

            # Cointegration hedge weights (reversed for mean reversion)
            hedge_weights = -vec / np.sum(np.abs(vec))

            # Zero out baseline positions on these subset indices
            currentPos[subset_indices] = 0
            
            # Override positions on the pair components
            for i, stock_idx in enumerate(subset_indices):
                price = latest_prices[i]
                weight = hedge_weights[i]
                dollar_target = size_frac * MAX_POS
                units = int(dollar_target * abs(weight) / price)
                currentPos[stock_idx] += units
        
        # If zscore between 1 and 1.5, or between -1.5 and -1, exit positions on the pair
        # elif 1 <= abs(score) <= 1.5:
        #     currentPos[subset_indices] = 0
        # # Else keep baseline positions unchanged on these instruments

    return currentPos
