import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile="prices.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

events = np.zeros(nt, dtype=bool)

stock_A = prcAll[0, :]

window = 50
test_size = 10

for t in range(window + 1, nt):
    log_returns = np.log(stock_A[t - window:t] / stock_A[t - window - 1:t - 1])
    baseline = log_returns[:-test_size]
    test_data = log_returns[-test_size:]
    
    mu_0 = np.mean(baseline)
    t_stat, p_value = stats.ttest_1samp(test_data, mu_0)
    
    events[t] = p_value < 0.05



# --- Parameters ---
momentum_window = 50

# --- Compute Momentum ---
momentum = stock_A[momentum_window:] - stock_A[:-momentum_window]
momentum = np.concatenate((np.full(momentum_window, np.nan), momentum))  # pad with NaNs to align length

# --- Transition Points ---
start_indices = np.where((events[1:] == 1) & (events[:-1] == 0))[0] + 1
end_indices   = np.where((events[1:] == 0) & (events[:-1] == 1))[0] + 1

# --- Plot ---
plt.figure(figsize=(12, 6))

# Price
plt.plot(stock_A, label="Stock A Price", color="black")

# Momentum (scaled to match price range visually)
momentum_scaled = momentum * 0.5 + np.nanmean(stock_A)  # scale and shift to overlay
plt.plot(momentum_scaled, label=f"Momentum ({momentum_window}-day)", color="orange")

# Event markers
plt.scatter(start_indices, stock_A[start_indices], color="red", label="Event Start", zorder=5)
plt.scatter(end_indices, stock_A[end_indices], color="green", label="Event End", zorder=5)

plt.axhline(np.nanmean(stock_A), color='gray', linestyle='--', label='Mean Price')

# Aesthetics
plt.title("Stock A with Event Markers and Momentum Overlay")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()