



# %%

import numpy as np
# %%
##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################



# print("Loaded matrix shape:", loaded_coeff_matrix.shape)

nInst = 50
currentPos = np.zeros(nInst)
maxPos = 10000

def getMyPosition(prcSoFar):
    loaded_coeff_matrix = np.loadtxt("algorithms/lr_coeff_matrix.csv", delimiter=",")
    rmses = np.loadtxt("algorithms/lr_rmse.csv")

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
    positions = (positions * (maxPos / latest_prices))

    rmses = rmses[1:]
    mask = np.abs(diffs) < (2 * rmses)
    positions[mask] = 0
    
    currentPos = positions.astype(int)
    return currentPos