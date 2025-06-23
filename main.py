
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

MAX_POS = 10000
nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 20):
        return np.zeros(nins)
    # lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    # lNorm = np.sqrt(lastRet.dot(lastRet))
    # lastRet /= lNorm
    # rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    # currentPos = np.array([int(x) for x in currentPos+rpos])
    # return currentPos

    log_prices = np.log(prcSoFar)
    subset = log_prices[[49, 1]].T
        
    # Code to Implement Trades
    cointegration_vector = np.array([38.53501255, 27.93024285])
    spread = subset @ (cointegration_vector)
    spread = pd.Series(spread)  # convert to Series
    zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()


    # Use raw prices for trading
    latest_prices = prcSoFar[[49, 1]][-1]

    # Determine if you're entering or exiting
    if abs(zscore.iloc[-1]) < 1:
        currentPos = np.zeros(nInst)  # Exit
    else:
        # Direction of trade
        direction = -1 if zscore.iloc[-1] > 1 else 1

        if direction != 0:
            # Cointegration weights (we'll reverse them for mean-reversion)
            hedge_weights = direction * -cointegration_vector

            # Convert hedge ratio to unit positions scaled by $10K per instrument
            subset_indices = [49, 1]  # Stock_50 → index 49, Stock_2 → index 1
            currentPos = np.zeros(nInst)

            for i, stock_idx in enumerate(subset_indices):
                price = latest_prices[i]
                weight = hedge_weights[i]

                # Dollar exposure capped at $10K per stock
                units = int(min(np.abs(weight), 1.0) * MAX_POS / price)

                currentPos[stock_idx] = np.sign(weight) * units
        else:
            currentPos = np.zeros(nInst)
    return currentPos
