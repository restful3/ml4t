# Using the CADF test for cointegration

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm

df=pd.read_csv('inputData_EWA_EWC_IGE.csv')
df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
df.set_index('Date', inplace=True)

df.plot()
df.plot.scatter(x='EWA', y='EWC')

results=sm.ols(formula="EWC ~ EWA", data=df[['EWA', 'EWC']]).fit()
print(results.params)
hedgeRatio=results.params['EWA']
print('hedgeRatio=%f' % hedgeRatio)

pd.DataFrame((df['EWC']-hedgeRatio*df['EWA'])).plot()

# cadf test
coint_t, pvalue, crit_value=ts.coint(df['EWA'], df['EWC'])
print('t-statistic=%f' % coint_t)
print('pvalue=%f' % pvalue)
print(crit_value)

# Johansen test
result=vm.coint_johansen(df[['EWA', 'EWC']].values, det_order=0, k_ar_diff=1)
print(result.lr1)
print(result.cvt)
print(result.lr2)
print(result.cvm)

# Add IGE for Johansen test
result=vm.coint_johansen(df.values, det_order=0, k_ar_diff=1)
print(result.lr1)
print(result.cvt)
print(result.lr2)
print(result.cvm)

print(result.eig) # eigenvalues
print(result.evec) # eigenvectors

yport=pd.DataFrame(np.dot(df.values, result.evec[:, 0])) #  (net) market value of portfolio

ylag=yport.shift()
deltaY=yport-ylag
df2=pd.concat([ylag, deltaY], axis=1)
df2.columns=['ylag', 'deltaY']
regress_results=sm.ols(formula="deltaY ~ ylag", data=df2).fit() # Note this can deal with NaN in top row
print(regress_results.params)

halflife=-np.log(2)/regress_results.params['ylag']
print('halflife=%f days' % halflife)

#  Apply a simple linear mean reversion strategy to EWA-EWC-IGE
lookback=np.round(halflife).astype(int) #  setting lookback to the halflife found above
numUnits =-(yport-yport.rolling(lookback).mean())/yport.rolling(lookback).std() # capital invested in portfolio in dollars.  movingAvg and movingStd are functions from epchan.com/book2
positions=pd.DataFrame(np.dot(numUnits.values, np.expand_dims(result.evec[:, 0], axis=1).T)*df.values) # results.evec(:, 1)' can be viewed as the capital allocation, while positions is the dollar capital in each ETF.
pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
pd.DataFrame((np.cumprod(1+ret)-1)).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
# APR=0.125739 Sharpe=1.391310
