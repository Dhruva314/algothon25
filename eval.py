#!/usr/bin/env python
# %%
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
# from main import getMyPosition as getPosition

signal_medians = []
signal_mads = []
regime_factors = []

spreads_means = []
spreads_stds = []

dlrPosLimit = 10000

prev = np.zeros(50)

# def pairWise(prcSoFar, stock_A, stock_B, factor) :
#     currentPos = np.zeros(50)
#     hist_A = prcSoFar[stock_A, :]
#     hist_B = prcSoFar[stock_B, :]

#     # _, pval, _ = coint(hist_A, hist_B)

#     # --- Beta regression: regress A on B using least squares ---
#     beta = np.polyfit(hist_B, hist_A, 1)[0]  # Only get the slope (ignore intercept)

#     # --- Compute hedged spread ---
#     spread = hist_A - beta * hist_B * factor
#     mean_spread = np.mean(spread)
#     std_spread = np.std(spread)

#     z_score = (spread[-1] - mean_spread) / std_spread
#     if z_score > 1:
#         currentPos[stock_A] = -1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_A, -1]).astype(int)
#         currentPos[stock_B] = 1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_B, -1]).astype(int)
#     elif z_score < -1:
#         currentPos[stock_A] = 1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_A, -1]).astype(int)
#         currentPos[stock_B] = -1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_B, -1]).astype(int)
#     elif abs(z_score) < 0.2:
#         currentPos[stock_A] = 0
#         currentPos[stock_B] = 0
#     else:
#         currentPos = getMyPosition(prcSoFar[:,:-1])

#     return currentPos

# def getMyPosition(prcSoFar):

# def getPosition(prcSoFar, prev):
#     currentPos = np.zeros(50)
#     (nins, nt) = prcSoFar.shape
#     if (nt < 25):
#         return np.zeros(nins)
    
#     stock_A = 24-1
#     stock_B = 12-1

#     hist_A = prcSoFar[stock_A, :]
#     hist_B = prcSoFar[stock_B, :]

#     # _, pval, _ = coint(hist_A, hist_B)

#     # --- Beta regression: regress A on B using least squares ---
#     beta = np.polyfit(hist_B, hist_A, 1)[0]  # Only get the slope (ignore intercept)

#     # --- Compute hedged spread ---
#     spread = hist_A - beta * hist_B
#     mean_spread = np.mean(spread)
#     std_spread = np.std(spread)

#     spreads_means.append(mean_spread)
#     spreads_stds.append(std_spread)

#     z_score = (spread[-1] - mean_spread) / std_spread
#     if z_score > 1:
#         currentPos[stock_A] = -1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_A, -1]).astype(int)
#         currentPos[stock_B] = 1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_B, -1]).astype(int)
#     elif z_score < -1:
#         currentPos[stock_A] = 1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_A, -1]).astype(int)
#         currentPos[stock_B] = -1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_B, -1]).astype(int)
#     elif abs(z_score) < 0.2:
#         currentPos[stock_A] = 0
#         currentPos[stock_B] = 0
#     else:
#         currentPos = prev
        
#     return currentPos, spread[-1], currentPos[stock_A], currentPos[stock_B]
        
def pairWisePer(prcSoFar, stock_A, stock_B, factor) :
    currentPos = np.zeros(50)
    hist_A = prcSoFar[stock_A, :]
    hist_B = prcSoFar[stock_B, :]

    # _, pval, _ = coint(hist_A, hist_B)

    # --- Beta regression: regress A on B using least squares ---
    beta = np.polyfit(hist_B, hist_A, 1)[0]  # Only get the slope (ignore intercept)

    # --- Compute hedged spread ---
    spread = hist_A - beta * hist_B * factor
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)

    z_score = (spread[-1] - mean_spread) / std_spread
    if z_score > 2:
        currentPos[stock_A] = -1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_A, -1]).astype(int)
        currentPos[stock_B] = 1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_B, -1]).astype(int)
    elif z_score < -2:
        currentPos[stock_A] = 1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_A, -1]).astype(int)
        currentPos[stock_B] = -1 * np.abs(z_score/3) * np.round(dlrPosLimit / prcSoFar[stock_B, -1]).astype(int)
    elif abs(z_score) < 1:
        currentPos[stock_A] = 0
        currentPos[stock_B] = 0
    else:
        currentPos = prev

    # pos_size = np.abs(z_score / 3)

    # # For each stock in the set, determine position size proportional to cointegration weight
    # for i, stock_idx in enumerate(stock_list):
    #     # Calculate position size in shares, scaled by abs(weight)
    #     size = pos_size * np.abs(coint_vector[i]) * np.round(dlrPosLimit / prcSoFar[stock_idx, -1])
    #     size = size.astype(int)
        
    #     if z_score > 1:
    #         # Short if cointegration weight positive, long if negative
    #         currentPos[stock_idx] = -np.sign(coint_vector[i]) * size
    #     elif z_score < -1:
    #         currentPos[stock_idx] = np.sign(coint_vector[i]) * size
    #     elif abs(z_score) < 0.2:
    #         currentPos[stock_idx] = 0
    #     else:
    #         currentPos = prev

    return currentPos

def pairWise(prcSoFar):
    currentPos = np.zeros(50)
    (nins, nt) = prcSoFar.shape
    if (nt < 25):
        return np.zeros(nins)
    
    stock_A = 50-1
    stock_B = 2-1

    stock_C = 24-1
    stock_D = 12-1

    currentPos += pairWisePer(prcSoFar, stock_A, stock_B, -1)
    currentPos += pairWisePer(prcSoFar, stock_C, stock_D, 1)

    set_A = {50-1, 2-1}
    set_B = {24-1,12-1}
    # set_C = {5-1,23-1,7-1,3-1,21-1}

    # currentPos += pairWise(prcSoFar, set_A)
    # currentPos += pairWise(prcSoFar, set_B)
    # currentPos += pairWise(prcSoFar, set_C)
        
    # return currentPos, spread[-1], currentPos[stock_A], currentPos[stock_B]
    return currentPos


def getPosition(prcSoFar):
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
    long_mom = np.mean(log_returns[:, -lookback:], axis=1)
    
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
    recent_change = np.log(prcSoFar[:, -1] / prcSoFar[:, -11])  # 10-day change
    log_returns_hist = np.log(prcSoFar[:, 10:] / prcSoFar[:, :-10])
    log_returns_means = np.mean(log_returns_hist, axis=1)
    mean_reversion = -np.tanh((recent_change - log_returns_means) * 1/20.0)  # Contrarian signal

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
            beta_calc = np.cov(inst_returns, market_returns_recent)[0,1] / (np.var(market_returns_recent) + 1e-6)
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
    
    # # # Apply beta neutralization more selectively
    # beta_adj_signal = combined_signal - 0.5 * betas  # Partial beta hedge
    
    # 7. Improved cross-sectional ranking
    # Use robust statistics
    signal_median = np.median(combined_signal)
    signal_mad = np.median(np.abs(combined_signal - signal_median))
    robust_zscore = (combined_signal - signal_median) / (signal_mad * 1.4826 + 1e-8)
    
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
    
    # # Turnover control - limit changes to reduce transaction costs
    # if nt > lookback + 1:  # Not first iteration
    #     position_change = target_positions - currentPos
    #     max_change = np.abs(currentPos) * 0.3  # Max 30% position change
    #     max_change = np.maximum(max_change, 10)  # But at least 10 shares
        
    #     position_change = np.clip(position_change, -max_change, max_change)
    #     target_positions = currentPos + position_change

    currentPos = np.round(target_positions).astype(int)

    # mr_pos = pairWise(prcSoFar).astype(int)
    # currentPos[np.nonzero(mr_pos)[0]] = 0
    # currentPos += mr_pos

    return currentPos


nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile="./prices.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

spreads = []
posA_list = []
posB_list = []

def calcPL(prcHist, numTestDays, startDay):

    prev = np.zeros(nInst)

    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    # startDay = nt + 1 - numTestDays
    for t in range(startDay+1, startDay+numTestDays+1):
        prcHistSoFar = prcHist[:,startDay:t]
        curPrices = prcHistSoFar[:,-1]
        if (t < numTestDays):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(prcHistSoFar)
            # newPosOrig, spread_val, posA, posB = getPosition(prcHistSoFar)
            prev = newPosOrig
            # spreads.append(spread_val)
            # posA_list.append(posA)
            # posB_list.append(posB)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm
        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        if (t > startDay):
            print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    print("Start Day %0lf done" % startDay)
    return (plmu, ret, plstd, annSharpe, totDVolume)

scores = []
for i in range(0,5):
    (meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll,200,i*200)
    score = meanpl - 0.1*plstd
    scores.append(score)

# print ("=====")
# print ("mean(PL): %.1lf" % meanpl)
# print ("return: %.5lf" % ret)
# print ("StdDev(PL): %.2lf" % plstd)
# print ("annSharpe(PL): %.2lf " % sharpe)
# print ("totDvolume: %.0lf " % dvol)
# print ("Score: %.2lf" % score)

print ("=====")
print ("Mean Score: %.2lf" % np.mean(scores))

# # Plot of certain numbers used.  


# days = list(range(len(signal_medians)))

# plt.figure(figsize=(14, 6))

# plt.plot(days, signal_medians, label='Signal Median')
# plt.plot(days, signal_mads, label='Signal MAD')
# plt.plot(days, regime_factors, label='Regime Factor')

# plt.title("Signal Statistics Over Time")
# plt.xlabel("Day")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.show()


# spread_mean = np.mean(spreads)
# spread_std = np.std(spreads)

# days = list(range(len(spreads)))

# plt.figure(figsize=(15, 6))

# # Plot spread with thresholds
# plt.subplot(2, 1, 1)
# plt.plot(days, spreads, label='Spread = A - β*B', color='blue')
# plt.plot(days, spreads_means, label='Spread Mean', color='black')
# plt.plot(days, spreads_means+spread_std, label='Spread + std', color='red')
# plt.plot(days, spreads_means-spread_std, label='Spread - std', color='green')
# plt.axhline(spread_mean, color='black', linestyle='--', label='Mean')
# plt.axhline(spread_mean + spread_std, color='red', linestyle='--', label='Mean + 1σ')
# plt.axhline(spread_mean - spread_std, color='green', linestyle='--', label='Mean - 1σ')
# plt.title("Hedged Spread Over Time")
# plt.ylabel("Spread")
# plt.grid(True)
# plt.legend()
# # Plot position sizes
# plt.subplot(2, 1, 2)
# plt.plot(days, posA_list, label='Position in A (stock 24)')
# plt.plot(days, posB_list, label='Position in B (stock 12)')
# plt.title("Position Sizes Over Time")
# plt.xlabel("Day")
# plt.ylabel("Position")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()
# %%
