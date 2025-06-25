



# %%

import numpy as np


##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

loaded_coeff_matrix = np.loadtxt("algorithms/logistic_coeff_matrix.csv", delimiter=",")
print("Loaded matrix shape:", loaded_coeff_matrix.shape)

nInst = 50
currentPos = np.zeros(nInst)
maxPos = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

nInst = 50
currentPos = np.zeros(nInst)
maxPos = 10000

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    
     # Get latest prices at time t (shape: nInst,)
    latest_prices = prcSoFar[:, -1]
    # Add intercept row and multiply to get predictions
    # loaded_coeff_matrix shape: (51, 50)
    # So build input vector with intercept term = 1 followed by latest_prices
    
    X_input = np.hstack(([1], latest_prices))  # shape (51,)
    
    # Predicted next prices for all 50 stocks
    predicted_prices = X_input @ loaded_coeff_matrix  # shape (50,)
    
    # Calculate difference: predicted - current
    diffs = predicted_prices - latest_prices
    
    # Positions: sign of difference (+1 or -1), zero difference treated as zero position
    positions = np.sign(diffs)
    
    # Convert zeros (if any) to 0, otherwise +1 or -1 remain
    positions = positions.astype(int)
    
    # Optionally, scale by maxPos or leave as just signs (1 or -1)
    positions = (positions * (maxPos / latest_prices))/30

    positions = positions.astype(int)
    
    currentPos = positions
    return currentPos

# %%
