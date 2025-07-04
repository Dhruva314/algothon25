import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape
    lookback = 45
    vol_lookback = 30
    capital = 3000

    if nt < lookback + 1:
        return np.zeros(nins)

    # 1. Compute momentum: exponentially weighted log returns
    log_returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    weights = np.exp(-np.linspace(0, 1, lookback))[::-1]  # newer days weigh more
    weights /= np.sum(weights)
    if log_returns.shape[1] < lookback:
        return np.zeros(nins)
    ew_momentum = np.dot(log_returns[:, -lookback:], weights)

    # 2. Volatility (std dev of log returns)
    vol = np.std(log_returns[:, -vol_lookback:], axis=1) + 1e-8

    # 3. Risk-adjusted momentum
    score = ew_momentum / vol

    # 4. Remove market beta (neutralise broad market exposure)
    market = np.mean(log_returns[:, -lookback:], axis=0)
    beta = np.array([np.cov(log_returns[i, -lookback:], market)[0,1] / np.var(market) 
                     for i in range(nins)])
    score -= beta  # subtract beta to market

    # 5. Cross-sectional z-score
    zscore = (score - np.mean(score)) / (np.std(score) + 1e-8)
    zscore = np.clip(zscore, -2, 2)

    # 6. Capital allocation
    weights = zscore / np.sum(np.abs(zscore))  # dollar-neutral
    target_dollars = capital * weights
    positions = target_dollars / prcSoFar[:, -1]

    currentPos = np.round(positions).astype(int)
    return currentPos

"""
mean(PL): 0.1
return: 0.00016
StdDev(PL): 5.70
annSharpe(PL): 0.26 
totDvolume: 103547 
Score: -0.48
"""