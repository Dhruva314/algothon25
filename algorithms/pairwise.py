# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

# MAX_POS = 10000
# nInst = 50
# currentPos = np.zeros(nInst)

# df = pd.read_csv("../prices.txt", header = None, delim_whitespace=True)
# column_names = [f"Stock_{i}" for i in range(1, 51)]
# df.columns = column_names

# log_prices = df.apply(np.log)

# # Select a group of correlated stocks (e.g. from your clustering)
# subset = log_prices[['Stock_5', 'Stock_23', 'Stock_7', 'Stock_3', 'Stock_21']]

# # Johansen test: det_order = 0 (no deterministic trend), k_ar_diff = 1 (lags)
# result = coint_johansen(subset, det_order=0, k_ar_diff=1)

# # Display trace statistics and critical values
# print("Trace stats:", result.lr1)
# print("Critical values (90%, 95%, 99%):\n", result.cvt)

# # Use the first cointegration vector (most stationary combination)
# cointegration_vector = result.evec[:, 0]  # first eigenvector

# # Create spread (linear combination)
# spread = subset @ (cointegration_vector/10)

# # Plot spread to check mean-reversion
# spread.plot(title="Johansen Cointegrated Spread")
# plt.axhline(spread.mean(), color='red', linestyle='--')
# plt.show()


# # Code to Implement Trades

# zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()

# # Simple entry/exit
# entry_long = zscore < -1
# entry_short = zscore > 1
# exit = abs(zscore) < 0.1


# # Use raw prices for trading
# latest_prices = df[['Stock_5', 'Stock_23', 'Stock_7', 'Stock_3', 'Stock_21']].iloc[-1].values

# # Determine if you're entering or exiting
# if abs(zscore.iloc[-1]) < 1:
#     currentPos = np.zeros(nInst)  # Exit
# else:
#     # Direction of trade
#     direction = -1 if zscore.iloc[-1] > 1 else 1

#     if direction != 0:
#         # Cointegration weights (we'll reverse them for mean-reversion)
#         hedge_weights = direction * -cointegration_vector

#         # Convert hedge ratio to unit positions scaled by $10K per instrument
#         subset_indices = [49, 1]  # Stock_50 → index 49, Stock_2 → index 1
#         currentPos = np.zeros(nInst)

#         for i, stock_idx in enumerate(subset_indices):
#             price = latest_prices[i]
#             weight = hedge_weights[i]

#             # Dollar exposure capped at $10K per stock
#             units = int(min(np.abs(weight), 1.0) * MAX_POS / price)

#             currentPos[stock_idx] = np.sign(weight) * units
#     else:
#         currentPos = np.zeros(nInst)


# # # Finds size of position for the corr stocks 
# # weights = -cointegration_vector  # negative for mean reversion
# # positions = weights / np.sum(np.abs(weights)) * MAX_POS
# # currentPos[49] = positions[0]
# # currentPos[1] = positions[1]

# # position = np.zeros(nInst)
# # size_frac = (-entry_short * (3 - zscore).clip(upper=3)) + \
# #             (entry_long * (3 + zscore).clip(upper=3))
# # if not exit.iloc[-1]:
# #   position[1] = np.exp(cointegration_vector[1]/10) 
# #   position[49] = np.exp(cointegration_vector[0]/10) 

# # if entry_short.iloc[-1]:
# #   position = -position


# %%
##### TODO #########################################

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

def getMyPosition(prcSoFar, z_stop=1.5):
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
        if abs(zscore.iloc[-1]) < 1 or abs(zscore.iloc[-1]) > z_stop:
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


nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile="C:/Users/dhruv/Coding/algothon25/prices.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

def calcPL(prcHist, numTestDays):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    startDay = nt + 1 - numTestDays
    for t in range(startDay, nt+1):
        prcHistSoFar = prcHist[:,:t]
        curPrices = prcHistSoFar[:,-1]
        if (t < nt):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getMyPosition(prcHistSoFar)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm
        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        if (t > startDay):
            # print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)



(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll,200)
score = meanpl - 0.1*plstd
# print ("=====")
# print ("mean(PL): %.1lf" % meanpl)
# print ("return: %.5lf" % ret)
# print ("StdDev(PL): %.2lf" % plstd)
# print ("annSharpe(PL): %.2lf " % sharpe)
# print ("totDvolume: %.0lf " % dvol)
# print ("Score: %.2lf" % score)

def cross_validate(prcAll, folds=5, zstop_range=np.arange(1.5, 3.1, 0.1)):
    nt = prcAll.shape[1]
    fold_size = nt // folds
    results = []

    for z_stop in zstop_range:
        fold_scores = []
        for k in range(folds):
            start = k * fold_size
            end = (k + 1) * fold_size if k < folds - 1 else nt
            prcFold = prcAll[:, start:end]

            # Inject current z_stop into strategy
            def getMyPositionWrapper(prcSoFar):
                return getMyPosition(prcSoFar, z_stop)

            global getPosition
            getPosition = getMyPositionWrapper

            # Evaluate on fold
            meanpl, _, plstd, _, _ = calcPL(prcFold, fold_size)
            score = meanpl - 0.1 * plstd
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        results.append((z_stop, avg_score))
        print(f"z_stop = {z_stop:.1f}, avg_score = {avg_score:.2f}")

    # Select best
    best_z, best_score = max(results, key=lambda x: x[1])
    print(f"\nBest z_stop: {best_z:.1f} with score: {best_score:.2f}")
    return best_z

best_z = cross_validate(prcAll)
print(best_z)
# %%
