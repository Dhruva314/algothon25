



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
    
    latest_prices = prcSoFar[:, -1]
    X_input = np.hstack(([1], latest_prices))  # shape (51,)

    # Predict log-odds → convert to probabilities
    probs = sigmoid(X_input @ loaded_coeff_matrix)  # shape (50,)
    
    # If predicted prob > 0.5 → price will go up → long
    positions = np.where(probs > 0.5, 1, -1)

    # Scale by maxPos / price to control dollar exposure
    safe_prices = np.where(latest_prices == 0, 1e-6, latest_prices)
    positions = positions * (maxPos / safe_prices) / 30
    
    currentPos = positions.astype(int)
    return currentPos

# =====
# %%
