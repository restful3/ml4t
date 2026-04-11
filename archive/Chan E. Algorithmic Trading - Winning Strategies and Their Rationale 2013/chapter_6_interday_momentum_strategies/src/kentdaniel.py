# Example 6.2: Cross-Sectional Momentum Strategy for Stocks

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm

lookback=252
holddays=25
topN=50

# Stocks
cl=pd.read_csv('inputDataOHLCDaily_20120424_cl.csv')
stocks=pd.read_csv('inputDataOHLCDaily_20120424_stocks.csv')

cl['Var1']=pd.to_datetime(cl['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
cl.columns=np.insert(stocks.values, 0, 'Date')
cl.set_index('Date', inplace=True)

ret=cl.pct_change(periods=lookback)
longs=np.full(cl.shape, False)
shorts=np.full(cl.shape, False)
positions=np.zeros(cl.shape)

for t in range(lookback, cl.shape[0]):
    hasData=np.where(np.isfinite(ret.iloc[t, :]))
    hasData=hasData[0]
    if len(hasData)>0:
        idxSort=np.argsort(ret.iloc[t, hasData])  
        longs[t, hasData[idxSort.values[np.arange(-np.min((topN, len(idxSort))),0)]]]=1
        shorts[t, hasData[idxSort.values[np.arange(0,topN)]]]=1
        
longs=pd.DataFrame(longs)
shorts=pd.DataFrame(shorts)

for h in range(holddays-1):
    long_lag=longs.shift(h).fillna(False)
    short_lag=shorts.shift(h).fillna(False)
    positions[long_lag]=positions[long_lag]+1
    positions[short_lag]=positions[short_lag]-1
    
positions=pd.DataFrame(positions)
ret=pd.DataFrame(np.sum((positions.shift().values)*(cl.pct_change().values), axis=1)/(2*topN)/holddays) # daily P&L of the strategy
cumret=(np.cumprod(1+ret)-1)
cumret.plot()

print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
#from calculateMaxDD import calculateMaxDD
#maxDD, maxDDD, i=calculateMaxDD(cumret.fillna(0))
#print('Max DD=%f Max DDD in days=%i' % (maxDD, maxDDD))