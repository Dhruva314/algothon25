import numpy as np
import pandas as pd
from itertools import product
from scipy import stats

# Load price data
def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None)
    return df.values.T  # (nInst, nDays)

# Improved momentum strategy implementation
def run_strategy(prcHist, params, test_range):
    nInst = prcHist.shape[0]
    curPos = np.zeros(nInst)
    cash = 0
    value = 0
    dayPL = []

    for t in test_range:
        prcSoFar = prcHist[:, :t]
        if t < prcHist.shape[1]:
            newPos = getMyPosition(prcSoFar, params)
            curPrices = prcSoFar[:, -1]
            deltaPos = newPos - curPos
            dvolume = np.sum(np.abs(deltaPos) * curPrices)
            comm = dvolume * 0.0005
            cash -= np.dot(deltaPos, curPrices) + comm
        else:
            newPos = curPos

        curPrices = prcHist[:, t - 1]
        curPos = newPos
        posValue = np.dot(curPos, curPrices)
        todayVal = cash + posValue
        dayPL.append(todayVal - value)
        value = todayVal

    dayPL = np.array(dayPL)
    if len(dayPL) < 2 or np.std(dayPL) == 0:
        return -np.inf, {}
    
    # Calculate comprehensive metrics
    mean_pl = np.mean(dayPL)
    std_pl = np.std(dayPL)
    sharpe = mean_pl / (std_pl + 1e-8) * np.sqrt(252)  # Annualized Sharpe
    max_drawdown = calculate_max_drawdown(np.cumsum(dayPL))
    
    score = mean_pl - 0.1 * std_pl  # Original scoring function
    
    metrics = {
        'mean_pl': mean_pl,
        'std_pl': std_pl,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'score': score
    }
    
    return score, metrics

def calculate_max_drawdown(cumulative_returns):
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - peak
    return np.min(drawdown)

# Enhanced strategy logic with parameters
def getMyPosition(prcSoFar, params):
    nins, nt = prcSoFar.shape
    
    # Unpack parameters
    lookback = params['lookback']
    vol_lookback = params['vol_lookback']
    capital = params['capital']
    max_position_pct = params['max_position_pct']
    short_mom_days = params['short_mom_days']
    med_mom_days = params['med_mom_days']
    momentum_weights = params['momentum_weights']
    vol_decay = params['vol_decay']
    mr_lookback = params['mr_lookback']
    mr_weight_factor = params['mr_weight_factor']
    beta_hedge_factor = params['beta_hedge_factor']
    regime_vol_days = params['regime_vol_days']
    zscore_clip = params['zscore_clip']
    kelly_max = params['kelly_max']
    turnover_limit = params['turnover_limit']

    if nt < lookback + 1:
        return np.zeros(nins)

    # Calculate log returns
    log_returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    
    if log_returns.shape[1] < lookback:
        return np.zeros(nins)

    # 1. Multi-timeframe momentum
    short_mom = np.mean(log_returns[:, -short_mom_days:], axis=1) if log_returns.shape[1] >= short_mom_days else np.zeros(nins)
    med_mom = np.mean(log_returns[:, -med_mom_days:], axis=1) if log_returns.shape[1] >= med_mom_days else np.zeros(nins)
    long_mom = np.mean(log_returns[:, -lookback:], axis=1)
    
    # Weighted combination
    momentum = (momentum_weights[0] * short_mom + 
               momentum_weights[1] * med_mom + 
               momentum_weights[2] * long_mom)

    # 2. Enhanced volatility calculation
    vol_weights = np.exp(-vol_decay * np.arange(vol_lookback))[::-1]
    vol_weights /= np.sum(vol_weights)
    
    volatility = np.zeros(nins)
    for i in range(nins):
        if log_returns.shape[1] >= vol_lookback:
            recent_returns = log_returns[i, -vol_lookback:]
            volatility[i] = np.sqrt(np.sum(vol_weights * recent_returns**2))
    volatility = np.maximum(volatility, 1e-6)

    # 3. Mean reversion component
    if log_returns.shape[1] >= mr_lookback:
        recent_change = (prcSoFar[:, -1] / prcSoFar[:, -mr_lookback] - 1)
        mean_reversion = -np.tanh(recent_change * 10)
    else:
        mean_reversion = np.zeros(nins)

    # 4. Market regime detection
    market_returns = np.mean(log_returns, axis=0)
    if log_returns.shape[1] >= regime_vol_days:
        market_vol = np.std(market_returns[-regime_vol_days:])
    else:
        market_vol = 0.02
    
    regime_factor = np.clip(0.02 / (market_vol + 1e-6), 0.5, 2.0)

    # 5. Beta calculation
    market_returns_recent = market_returns[-lookback:]
    betas = np.zeros(nins)
    for i in range(nins):
        inst_returns = log_returns[i, -lookback:]
        if len(inst_returns) == len(market_returns_recent):
            beta_calc = np.cov(inst_returns, market_returns_recent)[0,1] / (np.var(market_returns_recent) + 1e-8)
            betas[i] = beta_calc

    # 6. Combined signal
    risk_adj_momentum = momentum / volatility
    
    # Dynamic mean reversion weighting
    vol_regime = np.percentile(volatility, 50)
    mr_weight = np.minimum(volatility / (vol_regime + 1e-6), 1.0) * mr_weight_factor
    momentum_weight = 1.0 - mr_weight
    
    combined_signal = (momentum_weight * risk_adj_momentum + 
                      mr_weight * mean_reversion)
    
    # Apply beta neutralization
    beta_adj_signal = combined_signal - beta_hedge_factor * betas
    
    # 7. Cross-sectional ranking with robust statistics
    signal_median = np.median(beta_adj_signal)
    signal_mad = np.median(np.abs(beta_adj_signal - signal_median))
    robust_zscore = (beta_adj_signal - signal_median) / (signal_mad * 1.4826 + 1e-8)
    
    robust_zscore *= regime_factor
    robust_zscore = np.clip(robust_zscore, -zscore_clip, zscore_clip)

    # 8. Kelly criterion position sizing
    expected_return = np.abs(robust_zscore) * 0.001
    kelly_fraction = expected_return / (volatility**2 + 1e-6)
    kelly_fraction = np.clip(kelly_fraction, 0, kelly_max)
    
    position_weights = np.sign(robust_zscore) * kelly_fraction
    
    # Dollar neutral normalization
    long_weights = np.maximum(position_weights, 0)
    short_weights = np.minimum(position_weights, 0)
    
    long_sum = np.sum(long_weights)
    short_sum = np.sum(np.abs(short_weights))
    
    if long_sum > 0 and short_sum > 0:
        balance_factor = min(long_sum, short_sum)
        if long_sum > short_sum:
            long_weights *= balance_factor / long_sum
        else:
            short_weights *= balance_factor / short_sum
        position_weights = long_weights + short_weights

    total_weight = np.sum(np.abs(position_weights))
    if total_weight > 0:
        position_weights /= total_weight

    # 9. Position sizing and limits
    target_dollars = capital * position_weights
    target_positions = target_dollars / prcSoFar[:, -1]
    
    # Position limits
    max_shares = capital * max_position_pct / prcSoFar[:, -1]
    target_positions = np.clip(target_positions, -max_shares, max_shares)
    
    # Turnover control
    global currentPos
    if 'currentPos' in globals() and nt > lookback + 1:
        position_change = target_positions - currentPos
        max_change = np.maximum(np.abs(currentPos) * turnover_limit, 10)
        position_change = np.clip(position_change, -max_change, max_change)
        target_positions = currentPos + position_change

    currentPos = np.round(target_positions).astype(int)
    return currentPos

# Cross-validation with enhanced parameter search
def cross_validate(prcHist, param_configs, n_folds=5):
    nDays = prcHist.shape[1]
    fold_size = nDays // n_folds
    results = []

    for i, params in enumerate(param_configs):
        print(f"Testing configuration {i+1}/{len(param_configs)}")
        fold_scores = []
        fold_metrics = []
        
        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size
            if end > nDays:
                continue
            
            # Reset global position tracking
            global currentPos
            currentPos = np.zeros(prcHist.shape[0])
            
            score, metrics = run_strategy(prcHist, params, range(start + 1, end))
            fold_scores.append(score)
            fold_metrics.append(metrics)

        if fold_scores:
            avg_score = np.mean(fold_scores)
            avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) 
                          for key in fold_metrics[0].keys()}
            results.append((params, avg_score, avg_metrics))

    best = max(results, key=lambda x: x[1])
    return best, results

# Generate parameter configurations
def generate_param_configs():
    # Define parameter ranges
    lookback_options = [45, 60, 75]
    vol_lookback_options = [30, 45, 60]
    capital_options = [3000, 5000, 7000]
    max_position_pct_options = [0.10, 0.15, 0.20]
    short_mom_days_options = [5, 10, 15]
    med_mom_days_options = [20, 30, 40]
    vol_decay_options = [0.05, 0.1, 0.15]
    mr_lookback_options = [3, 5, 7]
    mr_weight_factor_options = [0.2, 0.3, 0.4]
    beta_hedge_factor_options = [0.3, 0.5, 0.7]
    regime_vol_days_options = [20, 30, 40]
    zscore_clip_options = [2.0, 2.5, 3.0]
    kelly_max_options = [0.3, 0.5, 0.7]
    turnover_limit_options = [0.2, 0.3, 0.4]

    # Generate limited combinations (grid search would be too large)
    configs = []
    
    # Strategy 1: Conservative
    for lookback, vol_lookback, capital in product([45, 60], [30, 45], [3000, 5000]):
        config = {
            'lookback': lookback,
            'vol_lookback': vol_lookback,
            'capital': capital,
            'max_position_pct': 0.10,
            'short_mom_days': 10,
            'med_mom_days': 30,
            'momentum_weights': [0.5, 0.3, 0.2],
            'vol_decay': 0.1,
            'mr_lookback': 5,
            'mr_weight_factor': 0.2,
            'beta_hedge_factor': 0.5,
            'regime_vol_days': 30,
            'zscore_clip': 2.0,
            'kelly_max': 0.3,
            'turnover_limit': 0.2
        }
        configs.append(config)
    
    # Strategy 2: Aggressive
    for lookback, vol_lookback, capital in product([60, 75], [45, 60], [5000, 7000]):
        config = {
            'lookback': lookback,
            'vol_lookback': vol_lookback,
            'capital': capital,
            'max_position_pct': 0.20,
            'short_mom_days': 5,
            'med_mom_days': 20,
            'momentum_weights': [0.6, 0.3, 0.1],
            'vol_decay': 0.15,
            'mr_lookback': 3,
            'mr_weight_factor': 0.4,
            'beta_hedge_factor': 0.3,
            'regime_vol_days': 20,
            'zscore_clip': 3.0,
            'kelly_max': 0.7,
            'turnover_limit': 0.4
        }
        configs.append(config)
        
    # Strategy 3: Balanced
    for lookback, vol_lookback, capital in product([60], [45], [3000, 5000, 7000]):
        config = {
            'lookback': lookback,
            'vol_lookback': vol_lookback,
            'capital': capital,
            'max_position_pct': 0.15,
            'short_mom_days': 10,
            'med_mom_days': 30,
            'momentum_weights': [0.5, 0.3, 0.2],
            'vol_decay': 0.1,
            'mr_lookback': 5,
            'mr_weight_factor': 0.3,
            'beta_hedge_factor': 0.5,
            'regime_vol_days': 30,
            'zscore_clip': 2.5,
            'kelly_max': 0.5,
            'turnover_limit': 0.3
        }
        configs.append(config)

    return configs

# === Main ===
if __name__ == "__main__":
    # Initialize global position tracking
    currentPos = None
    
    prcHist = loadPrices("./prices.txt")
    print(f"Loaded price data: {prcHist.shape[0]} instruments, {prcHist.shape[1]} days")

    # Generate parameter configurations
    param_configs = generate_param_configs()
    print(f"Testing {len(param_configs)} parameter configurations")

    # Run cross-validation
    best_params, all_results = cross_validate(prcHist, param_configs, n_folds=5)

    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    
    print("\n=== Best Parameters ===")
    best_config, best_score, best_metrics = best_params
    for key, value in best_config.items():
        print(f"{key}: {value}")
    print(f"\nBest Score: {best_score:.4f}")
    print(f"Sharpe Ratio: {best_metrics['sharpe']:.4f}")
    print(f"Max Drawdown: {best_metrics['max_drawdown']:.2f}")

    print("\n=== Top 10 Results ===")
    sorted_results = sorted(all_results, key=lambda x: -x[1])
    for i, (params, score, metrics) in enumerate(sorted_results[:10]):
        print(f"\nRank {i+1}: Score = {score:.4f}, Sharpe = {metrics['sharpe']:.4f}")
        print(f"  Lookback: {params['lookback']}, Vol_Lookback: {params['vol_lookback']}")
        print(f"  Capital: {params['capital']}, Max_Pos: {params['max_position_pct']:.2f}")
        print(f"  Kelly_Max: {params['kelly_max']:.2f}, Turnover: {params['turnover_limit']:.2f}")