
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

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 20):
        return np.zeros(nins)

    currentPos = np.zeros(nInst)
    log_prices = np.log(prcSoFar)
        
    # Code to Implement Trades

    for _, row in weights.iterrows():
        vec = row['vectors']
        vec = np.array(vec)
        subset_indices = list(row['sets'])

        subset = log_prices[subset_indices].T
        spread = subset @ (vec)
        spread = pd.Series(spread)  # convert to Series
        zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()

        score = zscore.iloc[-1]
        # size_frac = (-1 * max(min(3 - score, 3), 0)) if score > 1 else (max(min(3 + score, 3), 0))
        size_frac = np.sign(score)*min(abs(score), 3) / 3

        # Use raw prices for trading
        latest_prices = prcSoFar[subset_indices, -1]

        # Determine if you're entering or exiting
        if abs(zscore.iloc[-1]) < 1:
            for i, stock_idx in enumerate(subset_indices):
                currentPos[stock_idx] = 0  # Exit
        else:

            # Cointegration weights (we'll reverse them for mean-reversion)
            hedge_weights = -vec / np.sum(np.abs(vec))

            for i, stock_idx in enumerate(subset_indices):
                price = latest_prices[i]
                weight = hedge_weights[i]

                # Size adjusted position
                dollar_target = size_frac * MAX_POS
                units = int(dollar_target * abs(weight) / price)

                currentPos[stock_idx] += units
        
    return currentPos
