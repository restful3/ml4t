# Volatility Futures vs Equity Index Futures

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm

entryThreshold=0.1
onewaytcost=1/10000

# VX futures
vx=pd.read_csv('inputDataDaily_VX_20120507.csv')
vx['Date']=pd.to_datetime(vx['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
vx.set_index('Date', inplace=True)

# VIX index
vix=pd.read_csv('VIX.csv')
vix['Date']=pd.to_datetime(vix['Date'],  format='%Y-%m-%d').dt.date # remove HH:MM:SS
vix.set_index('Date', inplace=True)
vix=vix[['Close']]
vix.rename({'Close': 'VIX'}, axis=1, inplace=True)


# ES backadjusted continuous contract
es=pd.read_csv('inputDataDaily_ES_20120507.csv')
es['Date']=pd.to_datetime(es['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
es.set_index('Date', inplace=True)
es.rename({'Close': 'ES'}, axis=1, inplace=True)

# Merge on common dates
df=pd.merge(vx, vix, how='inner', on='Date')
df=pd.merge(df, es, how='inner', on='Date')

vx=df.drop({'VIX', 'ES'}, axis=1)
vix=df[['VIX']]
es=df[['ES']]

isExpireDate=vx.notnull() & vx.shift(-1).isnull()

# Define front month as 40 days to 10 days before expiration
numDaysStart=40
numDaysEnd=10

positions=np.zeros((vx.shape[0], vx.shape[1]+1))

for c in range(vx.shape[1]-1):
    expireIdx=np.where(isExpireDate.iloc[:,c])
    if c==0:
        startIdx=expireIdx[0]-numDaysStart
        endIdx=expireIdx[0]-numDaysEnd
    else:
        startIdx=np.max((endIdx+1, expireIdx[0]-numDaysStart))
        endIdx=expireIdx[0]-numDaysEnd
        
    if (expireIdx[0] >= 0):
        idx=np.arange(startIdx, endIdx+1)
        dailyRoll=(vx.iloc[idx, c]-vix.iloc[idx, 0])/np.arange(expireIdx-startIdx+1, expireIdx-endIdx, -1)
        positions[idx[dailyRoll > entryThreshold], c]=-1
        positions[idx[dailyRoll > entryThreshold], -1]=-1
        
        positions[idx[dailyRoll < -entryThreshold], c]=1
        positions[idx[dailyRoll < entryThreshold], -1]=1

y=pd.merge(vx*1000, es*50, on='Date')
positions=pd.DataFrame(positions, index=y.index)
pnl=np.sum((positions.shift().values)*(y-y.shift()) - onewaytcost*np.abs(positions.values*y - (positions.values*y).shift()), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs((positions.values*y).shift()), axis=1)

idx=ret.index[ret.index >= pd.datetime(2008, 8, 4).date()]
cumret=(np.cumprod(1+ret[idx[500:]])-1)
cumret.plot()

print('APR=%f Sharpe=%f' % (np.prod(1+ret[idx[500:]])**(252/len(ret[idx[500:]]))-1, np.sqrt(252)*np.mean(ret[idx[500:]])/np.std(ret[idx[500:]])))
from calculateMaxDD import calculateMaxDD
maxDD, maxDDD, i=calculateMaxDD(cumret.fillna(0))
print('Max DD=%f Max DDD in days=%i' % (maxDD, maxDDD))
#APR=0.377914 Sharpe=2.117500
#Max DD=-0.434420 Max DDD in days=73