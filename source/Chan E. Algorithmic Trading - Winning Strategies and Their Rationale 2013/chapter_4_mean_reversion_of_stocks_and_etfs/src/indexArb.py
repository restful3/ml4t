# Example 4.2: Arbitrage between SPY and Its Component Stocks

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm

# Stocks
cl=pd.read_csv('inputDataOHLCDaily_20120424_cl.csv')
stocks=pd.read_csv('inputDataOHLCDaily_20120424_stocks.csv')

cl['Var1']=pd.to_datetime(cl['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
cl.columns=np.insert(stocks.values, 0, 'Date')
cl.set_index('Date', inplace=True)

# ETFs
cl_etf=pd.read_csv('inputData_ETF_cl.csv')
etfs=pd.read_csv('inputData_ETF_stocks.csv')

cl_etf['Var1']=pd.to_datetime(cl_etf['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
cl_etf.columns=np.insert(etfs.values, 0, 'Date')
cl_etf.set_index('Date', inplace=True)

# Merge on common dates
df=pd.merge(cl, cl_etf, how='inner', on='Date')

cl_stocks=df[cl.columns]
cl_etf=df[cl_etf.columns]

# Use SPY only
cl_etf=cl_etf['SPY'] # This turns cl_etf into Series

trainDataIdx=df.index[(df.index > pd.datetime(2007, 1, 1).date()) & (df.index <= pd.datetime(2007, 12, 31).date())]
testDataIdx =df.index[df.index > pd.datetime(2007, 12, 31).date()]

isCoint=np.full(stocks.shape[1], False)
for s in range(stocks.shape[1]):
    # Combine the two time series into a matrix y2 for input into Johansen test
    y2=pd.concat([cl_stocks.loc[trainDataIdx].iloc[:, s], cl_etf.loc[trainDataIdx]], axis=1)
    y2=y2.loc[y2.notnull().all(axis=1),]
    
    if (y2.shape[0] > 250):
        # Johansen test
        result=vm.coint_johansen(y2.values, det_order=0, k_ar_diff=1)
        if (result.lr1[0] > result.cvt[0,0]):
            isCoint[s]=True
        
print(isCoint.sum())

yN=cl_stocks.loc[trainDataIdx, isCoint]
logMktVal_long=np.sum(np.log(yN), axis=1) # The net market value of the long-only portfolio is same as the "spread"

# Confirm that the portfolio cointegrates with SPY
ytest=pd.concat([logMktVal_long, np.log(cl_etf.loc[trainDataIdx])], axis=1)
result=vm.coint_johansen(ytest, det_order=0, k_ar_diff=1)
print(result.lr1)
print(result.cvt)
print(result.lr2)
print(result.cvm)

#Apply linear mean-reversion model on test set
yNplus=pd.concat([cl_stocks.loc[testDataIdx, isCoint], pd.DataFrame(cl_etf.loc[testDataIdx])], axis=1)  # Array of stock and ETF prices
weights=np.column_stack((np.full((testDataIdx.shape[0], isCoint.sum()), result.evec[0,0]), np.full((testDataIdx.shape[0], 1), result.evec[1, 0]))) # Array of log market value of stocks and ETF's
   
logMktVal=np.sum(weights*np.log(yNplus), axis=1) # Log market value of long-short portfolio

lookback=5
numUnits =-(logMktVal-logMktVal.rolling(lookback).mean())/logMktVal.rolling(lookback).std() # capital invested in portfolio in dollars.  movingAvg and movingStd are functions from epchan.com/book2
positions=pd.DataFrame(np.expand_dims(numUnits, axis=1)*weights)# results.evec(:, 1)' can be viewed as the capital allocation, while positions is the dollar capital in each ETF.
pnl=np.sum((positions.shift().values)*(np.log(yNplus)-np.log(yNplus.shift()).values), axis=1) # daily P&L of the strategy
ret=pd.DataFrame(pnl.values/np.sum(np.abs(positions.shift()), axis=1).values)
(np.cumprod(1+ret)-1).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
