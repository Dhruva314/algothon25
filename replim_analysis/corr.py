# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro


# Functions
# Groups clusters of pairs of correlated stocks 
def process_pairs(pairs):
    sets = [set()]  # start with one empty set

    for a, b in pairs:
        found = False

        for s in sets:
            if a in s and b in s:
                # both elements already in the set — do nothing
                found = True
                break
            elif a in s and b not in s:
                s.add(b)
                found = True
                break
            elif b in s and a not in s:
                s.add(a)
                found = True
                break

        if not found:
            # no set contains either element — create new set with both
            sets.append({a, b})

    return sets

#---------------------------------------------------------------------------------


df = pd.read_csv("../prices.txt", header = None, delim_whitespace=True)
column_names = [f"Stock_{i}" for i in range(1, 51)]
df.columns = column_names

# pct_change_df = df.pct_change()
# df[["Stock_2", "Stock_50"]].plot()
# plt.show()

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")

# Create a boolean mask where abs(corr) >= 0.8
mask = (abs(corr_matrix) >= 0.9) & (abs(corr_matrix) != 1)

# Use .stack() to reshape and filter only True values
strong_corr_pairs = mask.stack()
strong_corr_pairs = strong_corr_pairs[strong_corr_pairs]
strong_corr_list = strong_corr_pairs.index.tolist()

# Groups stocks by correlation 
result = process_pairs(strong_corr_list)

for i, s in enumerate(result):
  print(f"Set {i}: {s}")
