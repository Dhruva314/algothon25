#!/usr/bin/env python

import numpy as np
import pandas as pd
# from main import getMyPosition as getPosition

nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile="./prices.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

# prev = np.zeros(50)

def pairWisePer(prcSoFar, stock_A, stock_B, factor) :
    currentPos = np.zeros(50)
    hist_A = prcSoFar[stock_A, :]
    hist_B = prcSoFar[stock_B, :]

    # _, pval, _ = coint(hist_A, hist_B)

    # --- Beta regression: regress A on B using least squares ---
    beta = np.polyfit(hist_B, hist_A, 1)[0]  # Only get the slope (ignore intercept)

    # --- Compute hedged spread ---
    spread = hist_A - beta * hist_B * factor
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)

    z_score = (spread[-1] - mean_spread) / std_spread
    z_thesh = 3
    pos_A = np.round(dlrPosLimit / prcSoFar[stock_A, -1]).astype(int)
    pos_B = np.round(dlrPosLimit / prcSoFar[stock_B, -1]).astype(int)
    if z_score > z_thesh:
        currentPos[stock_A] = -1 * np.abs(z_score - z_thesh) * pos_A
        currentPos[stock_B] = 1 * np.abs(z_score - z_thesh) * pos_B
    elif z_score < -z_thesh:
        currentPos[stock_A] = 1 * np.abs(z_score + z_thesh) * pos_A
        currentPos[stock_B] = -1 * np.abs(z_score + z_thesh) * pos_B
    elif abs(z_score) < 0.2:
        currentPos[stock_A] = 0
        currentPos[stock_B] = 0
    # else:
    #     currentPos = prev

    return currentPos

def getPosition(prcSoFar):
    currentPos = np.zeros(50)
    (nins, nt) = prcSoFar.shape
    if (nt < 25):
        return np.zeros(nins)
    
    stock_A = 50-1
    stock_B = 2-1

    stock_C = 24-1
    stock_D = 12-1

    currentPos += pairWisePer(prcSoFar, stock_A, stock_B, -1)
    currentPos += pairWisePer(prcSoFar, stock_C, stock_D, 1)

    return currentPos

def calcPL(prcHist, startDay):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    print("============================================================")
    for t in range(1, nt+1):
        prcHistSoFar = prcHist[:,:t]
        curPrices = prcHistSoFar[:,-1]
        if (t < nt):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(prcHistSoFar)
            # prev = newPosOrig
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
        if (t > 0):
            # print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t+startDay,value, todayPL, totDVolume, ret))
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


# CV Fold-10

np.random.seed(217)

def pick_spaced_points(min_val, max_val, num_points, window):
    points = []
    while len(points) < num_points:
        candidate = np.random.randint(min_val, max_val)
        if all(abs(candidate - p) >= window for p in points):
            points.append(candidate)
        if len(points) > 10000:
            raise RuntimeError("Too many attempts – try reducing spacing or number of points.")
    return np.array(sorted(points))

window = 50
fold = 5
points = [0]
points.extend(pick_spaced_points(0+window, 1000-window, fold-1, window))
points.append(1000)
print(points)

meanpls = []
rets = []
plstds = []
sharpes = []
dvols = []
scores = []

for i in range(0, len(points)-1):
    # a = 0
    # b = 1000
    # if (i == 0):
    #     b = points[i]
    # elif (i == len(points) - 1):
    #     a = points[i]
    # else:
        # a = points[i-1]
        # b = points[i]

    a = points[i]
    b = points[i+1]
    (meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll[:,a:b], a)
    score = meanpl - 0.1*plstd
    
    print ("==============")
    print (f"Fold {i+1} Result")
    print ("==============")
    print ("mean(PL): %.1lf" % meanpl)
    print ("return: %.5lf" % ret)
    print ("StdDev(PL): %.2lf" % plstd)
    print ("annSharpe(PL): %.2lf " % sharpe)
    print ("totDvolume: %.0lf " % dvol)
    print ("Score: %.2lf" % score)

    meanpls.append(meanpl)
    rets.append(ret)
    plstds.append(plstd)
    sharpes.append(sharpe)
    dvols.append(dvol)
    scores.append(score)

print ("===============")
print ("Mean CV Results")
print ("===============")
print ("mean(PL): %.1lf" % np.mean(meanpls))
print ("return: %.5lf" % np.mean(rets))
print ("StdDev(PL): %.2lf" % np.mean(plstds))
print ("annSharpe(PL): %.2lf " % np.mean(sharpes))
print ("totDvolume: %.0lf " % np.mean(dvols))
print ("Score: %.2lf" % np.mean(scores))