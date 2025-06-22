# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
# from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv("../prices.txt", header = None, delim_whitespace=True)
column_names = [f"Stock_{i}" for i in range(1, 51)]
df.columns = column_names

pct_change_df = df.pct_change()
pct_change_df.drop(0, axis=0, inplace=True)
pct_change_df

Y = pct_change_df["Stock_5"].copy()
Y = (Y >= 0).astype(int)
X = pct_change_df[['Stock_23', 'Stock_7', 'Stock_3', 'Stock_21']]

X = sm.add_constant(X)  # Add intercept
model = sm.Logit(Y, X).fit()
print(model.summary())

# %%
##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


def getMyPostion(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos

# %%i
