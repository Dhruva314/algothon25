import pickle
import pandas as pd
import numpy as np

# team_decode.py
file_path = 'team_pos/Team_858308.dump'

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

df = loadPrices(file_path)
# print(df)
print(f"Shape: {df.shape}")

def getMyPosition(prcSoFar):
  (nins, nt) = prcSoFar.shape
  # print(nt)
  currentPos = np.zeros(nInst)
  if (nt > 1000):
     df = loadPrices(file_path)
     currentPos = df[:,nt-1001]

  return currentPos