# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("../prices.txt", header = None, delim_whitespace=True)
column_names = [f"Stock_{i}" for i in range(1, 51)]
df.columns = column_names
print(df.head())
df["Stock_1"].plot()
plt.show()


corr_matrix = df.corr()
print(corr_matrix)

# %%
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")

# %%
