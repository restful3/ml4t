# Example 4.1: Buy-on-Gap Model on SPX Stocks

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm

topN=10 # Max number of positions
entryZscore=1
lookback=20 # for MA

op=pd.read_csv('inputDataOHLCDaily_20120424_op.csv')
hi=pd.read_csv('inputDataOHLCDaily_20120424_hi.csv')
lo=pd.read_csv('inputDataOHLCDaily_20120424_lo.csv')
cl=pd.read_csv('inputDataOHLCDaily_20120424_cl.csv')

stocks=pd.read_csv('inputDataOHLCDaily_20120424_stocks.csv')

op['Var1']=pd.to_datetime(op['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
op.columns=np.insert(stocks.values, 0, 'Date')
op.set_index('Date', inplace=True)

hi['Var1']=pd.to_datetime(hi['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
hi.columns=np.insert(stocks.values, 0, 'Date')
hi.set_index('Date', inplace=True)

lo['Var1']=pd.to_datetime(lo['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
lo.columns=np.insert(stocks.values, 0, 'Date')
lo.set_index('Date', inplace=True)

cl['Var1']=pd.to_datetime(cl['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
cl.columns=np.insert(stocks.values, 0, 'Date')
cl.set_index('Date', inplace=True)

stdretC2C90d=cl.pct_change().rolling(90).std().shift(1)
buyPrice=lo.shift()*(1-entryZscore*stdretC2C90d)

retGap=(op-lo.shift())/lo.shift()

ma=cl.rolling(lookback).mean()
positionsTable=np.zeros(retGap.shape)


for t in np.arange(1, cl.shape[0]):
    hasData=np.where(np.isfinite(retGap.iloc[t, :]) & (op.iloc[t, :] < buyPrice.iloc[t, :]).values & (op.iloc[t, :] > ma.iloc[t, :]).values)
    hasData=hasData[0]
    if len(hasData)>0:
        idxSort=np.argsort(retGap.iloc[t, hasData])  
        positionsTable[t, hasData[idxSort.values[np.arange(-np.min((topN, len(idxSort))),0)]]]=1

retO2C=(cl-op)/op
    
pnl=np.sum(positionsTable*retO2C, axis=1) # daily P&L of the strategy
ret=pnl/topN
(np.cumprod(1+ret)-1).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
