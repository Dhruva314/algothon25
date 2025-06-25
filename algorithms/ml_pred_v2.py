# Uses All current prices of the 50 instruments to predict the future value of 1 stock
# Repeat for all stocks (250 variables)

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

df = pd.read_csv("../prices.txt", header = None, delim_whitespace=True)
column_names = [f"Stock_{i}" for i in range(1, 51)]
df.columns = column_names

# Train = rows 0–499, Test = rows 500–749
df_train = df.iloc[0:500]
df_test = df.iloc[500:750]

X_train_all = df_train.iloc[:-1].values     # (499, 50)
X_test_all = df_test.iloc[:-1].values       # (249, 50)
n_stocks = X_train_all.shape[1]

alphas = np.logspace(-3, 3, 20)  # alpha values to try
coeff_matrix = np.zeros((n_stocks + 1, n_stocks))  # rows: intercept + 50 coefficients, cols: 50 target stocks

for target_idx in range(n_stocks):
    # Target stock prices at t+1
    y_train = df_train.iloc[1:, target_idx].values  # (499,)
    y_test = df_test.iloc[1:, target_idx].values    # (249,)

    best_alpha = None
    best_rmse = np.inf

    # Grid search: evaluate RMSE on test set
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_all, y_train)
        y_pred = model.predict(X_test_all)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    # Final model fit with best alpha on training data
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train_all, y_train)

    # Store intercept and coefficients
    coeff_matrix[0, target_idx] = final_model.intercept_
    coeff_matrix[1:, target_idx] = final_model.coef_

    print(f"Stock_{target_idx+1}: Best alpha={best_alpha:.5f}, Test RMSE={best_rmse:.5f}")

print("Coefficient matrix shape:", coeff_matrix.shape)  # Should be (51, 50)
np.savetxt("lr_coeff_matrix.csv", coeff_matrix, delimiter=",")


# # Fit final model with best alpha
# final_model = Ridge(alpha=best_alpha)
# final_model.fit(X, y)
# predictions = final_model.predict(X)
# mse = mean_squared_error(y, predictions)
# rmse = np.sqrt(mse)

# print("Model coefficients (for each stock):", final_model.coef_)
# print("Intercept:", final_model.intercept_)
# print("Training RMSE:", rmse)
# r2 = r2_score(y, predictions)
# print(f"R² (coefficient of determination): {r2:.4f}")

# # Predict next price
# latest_prices = df.iloc[-1].values.reshape(1, -1)
# predicted_next_price = final_model.predict(latest_prices)[0]
# print("Predicted next price of Stock_1:", predicted_next_price)

# # Residuals scatter plot
# residuals = y - predictions
# plt.figure(figsize=(10, 4))
# plt.scatter(range(len(residuals)), residuals, marker='o', alpha=0.7)
# plt.title("Residuals (Dot Plot)")
# plt.xlabel("Index")
# plt.ylabel("Residual")
# plt.show()

# # Q-Q plot of residuals
# plt.figure(figsize=(6, 6))
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("Q-Q Plot of Residuals")
# plt.show()

# %%
##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

loaded_coeff_matrix = np.loadtxt("algorithms/lr_coeff_matrix.csv", delimiter=",")
print("Loaded matrix shape:", loaded_coeff_matrix.shape)

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
    positions = int(positions * (maxPos / latest_prices))/10
    
    currentPos = positions
    return currentPos
