import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    n_days_lookback = 37
    capital_per_stock = 3000
    
    if prcSoFar.shape[1] < n_days_lookback + 1:
        return np.zeros(nInst)
    
    # Calculate returns over lookback period
    returns = prcSoFar[:, -1] / prcSoFar[:, -n_days_lookback - 1] - 1
    
    # Rank instruments by momentum (best to worst)
    ranked = np.argsort(returns)[::-1]
    
    # Go long top 25%, short bottom 25%
    long_stocks = ranked[:nInst//4]
    short_stocks = ranked[-nInst//4:]
    
    # Initialize positions
    positions = np.zeros(nInst)
    
    # Equal dollar allocation ($3000 per position)
    for stock in long_stocks:
        positions[stock] = capital_per_stock / prcSoFar[stock, -1] * 0.7
    
    for stock in short_stocks:
        positions[stock] = -capital_per_stock / prcSoFar[stock, -1] * 0.7
    
    # Round to whole shares and update
    currentPos = np.round(positions).astype(int)
    return currentPos

"""
mean(PL): 10.1
return: 0.00079
StdDev(PL): 107.56
annSharpe(PL): 1.48 
totDvolume: 2502077 
Score: -0.64
"""