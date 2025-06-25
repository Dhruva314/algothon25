# Uses All current prices of the 50 instruments to predict the future value of 1 stock
# Repeat for all stocks (250 variables)

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../prices.txt", header=None, sep=r'\s+')
column_names = [f"Stock_{i}" for i in range(1, 51)]
df.columns = column_names

# Split into train/test
df_train = df.iloc[0:500]
df_test = df.iloc[500:750]

X_train_all = df_train.iloc[:-1].values     # shape (499, 50)
X_test_all = df_test.iloc[:-1].values       # shape (249, 50)

n_stocks = X_train_all.shape[1]

# Store logistic coefficients: shape (51, 50)
logistic_coeff_matrix = np.zeros((n_stocks + 1, n_stocks))  # intercept + 50 weights per stock

for target_idx in range(n_stocks):
    # Get future price movements for the stock
    y_train_prices = df_train.iloc[1:, target_idx].values
    y_train_prev = df_train.iloc[:-1, target_idx].values
    y_train = (y_train_prices > y_train_prev).astype(int)

    y_test_prices = df_test.iloc[1:, target_idx].values
    y_test_prev = df_test.iloc[:-1, target_idx].values
    y_test = (y_test_prices > y_test_prev).astype(int)

    # Fit logistic regression
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_all, y_train)

    # Evaluate accuracy on test
    y_pred = model.predict(X_test_all)
    acc = accuracy_score(y_test, y_pred)

    # Store intercept and coefficients
    logistic_coeff_matrix[0, target_idx] = model.intercept_[0]
    logistic_coeff_matrix[1:, target_idx] = model.coef_[0]

    print(f"Stock_{target_idx+1}: Test Accuracy = {acc:.4f}")

# Save logistic coefficients
np.savetxt("logistic_coeff_matrix.csv", logistic_coeff_matrix, delimiter=",")
print("Saved logistic_coeff_matrix.csv")

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

loaded_coeff_matrix = np.loadtxt("algorithms/logistic_coeff_matrix.csv", delimiter=",")
print("Loaded matrix shape:", loaded_coeff_matrix.shape)

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
    positions = positions * (maxPos / safe_prices)
    
    currentPos = positions.astype(int)
    return currentPos

# =====
# mean(PL): 6.4
# return: 0.00143
# StdDev(PL): 68.76
# annSharpe(PL): 1.46
# totDvolume: 866728
# Score: -0.51