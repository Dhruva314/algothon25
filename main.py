
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
maxPos = 10000
currentPos = np.zeros(nInst)

def sigmoidAlt(x):
    return abs(2 / (1 + np.exp(-10*x)) - 1)

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    # lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    # lNorm = np.sqrt(lastRet.dot(lastRet))
    # lastRet /= lNorm
    # rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    # currentPos = np.array([int(x) for x in currentPos+rpos])

    prog_mean = []
    for instrument in prcSoFar:
        instrument_means = []
        for i, value in enumerate(instrument):
            if i < 1:
                instrument_means.append(value)
                continue
            instrument_means.append(np.mean(instrument[:i]))
        prog_mean.append(instrument_means)

    linear_fits = []
    # if current_day <= starting_day:
    #     linear_fits = initial_fits
    # else:
    for i, indicator in enumerate(prog_mean):
        x = np.array(list(range(0, len(indicator))))
        y = np.array(indicator)
        slope, intercept = np.polyfit(x, y, 1)
        linear_fits.append((slope, intercept))

    evs = []
    for i in linear_fits:
        x = len(prcSoFar[0])
        m = i[0]
        c = i[1]
        expected_value = (m * x) + c
        evs.append(expected_value)

    uppers = []
    lowers = []
    for i, indicator_history in enumerate(prcSoFar):
        freedom_factor = 1
        freedom = abs(linear_fits[i][0]) * freedom_factor
        upper = evs[i] + freedom
        lower = evs[i] - freedom
        uppers.append(upper)
        lowers.append(lower)

    positions = np.zeros(nInst)    

    for i in range(len(prcSoFar)):
        current_price = prcSoFar[i][-1]
        
        # Current progressive mean (mean of prices excluding current)
        if len(prcSoFar[i]) < 2:
            continue
        current_prog_mean = np.mean(prcSoFar[i][:-1])
        
        # Compare to bounds
        upper = uppers[i]
        lower = lowers[i]
        
        if current_prog_mean > upper:
            # Price is "too strong" → mean higher than expected → potential sell
            positions[i] = -int(maxPos / current_price * sigmoidAlt(linear_fits[i][0]))   # Go short

        elif current_prog_mean < lower:
            # Price is "too weak" → mean lower than expected → potential buy
            positions[i] = int(maxPos / current_price * sigmoidAlt(linear_fits[i][0]))   # Go long
        
        else:
            positions[i] = 0

    return positions
