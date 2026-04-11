# Bollinger Band Mean Reversion Strategy

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm

df=pd.read_csv('inputData_GLD_USO.csv')
df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
df.set_index('Date', inplace=True)

lookback=20
hedgeRatio=np.full(df.shape[0], np.nan)
for t in np.arange(lookback, len(hedgeRatio)):
    regress_results=sm.ols(formula="USO ~ GLD", data=df[(t-lookback):t]).fit() # Note this can deal with NaN in top row
    hedgeRatio[t-1]=regress_results.params[1]

yport=np.sum(ts.add_constant(-hedgeRatio)[:, [1,0]]*df, axis=1)
yport.plot()

# Bollinger band strategy
entryZscore=1
exitZscore=0

MA=yport.rolling(lookback).mean()
MSTD=yport.rolling(lookback).std()
zScore=(yport-MA)/MSTD

longsEntry=zScore < -entryZscore
longsExit =zScore > -entryZscore

shortsEntry=zScore > entryZscore
shortsExit =zScore < exitZscore

numUnitsLong=np.zeros(longsEntry.shape)
numUnitsLong[:]=np.nan

numUnitsShort=np.zeros(shortsEntry.shape)
numUnitsShort[:]=np.nan

numUnitsLong[0]=0
numUnitsLong[longsEntry]=1
numUnitsLong[longsExit]=0
numUnitsLong=pd.DataFrame(numUnitsLong)
numUnitsLong.fillna(method='ffill', inplace=True)

numUnitsShort[0]=0
numUnitsShort[shortsEntry]=-1
numUnitsShort[shortsExit]=0
numUnitsShort=pd.DataFrame(numUnitsShort)
numUnitsShort.fillna(method='ffill', inplace=True)

numUnits=numUnitsLong+numUnitsShort
positions=pd.DataFrame(np.tile(numUnits.values, [1, 2]) * ts.add_constant(-hedgeRatio)[:, [1,0]] *df.values) #  [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.
pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
pd.DataFrame((np.cumprod(1+ret)-1)).plot()

print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
