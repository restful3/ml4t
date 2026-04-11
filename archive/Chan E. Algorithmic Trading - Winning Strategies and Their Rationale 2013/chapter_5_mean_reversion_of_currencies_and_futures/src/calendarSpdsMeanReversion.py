# Example 5.4: Mean Reversion Trading of Calendar Spreads
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


df=pd.read_csv('inputDataDaily_CL_20120502.csv')
df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
df.set_index('Date', inplace=True)

# Fitting gamma to forward curve
gamma=np.full(df.shape[0], np.nan)
for t in range(df.shape[0]):
    idx=np.where(np.isfinite(df.iloc[t, :]))[0]
    idxDiff=np.array(list(set(idx[1:])-set(idx)))
    if ((len(idx)>=5) & (all(idxDiff[0:4]==1))):
        FT=df.iloc[t, idx[:5]]
        T=sm.add_constant(range(FT.shape[0]))
        model=sm.OLS(np.log(FT.values), T)
        res=model.fit()
        gamma[t]=-12*res.params[1]
          
results=adfuller(gamma[np.where(np.isfinite(gamma))], maxlag=1, regression='c', autolag=None)
print(results)
#(-4.586684184658408, 0.00013666960538551907, 1, 1995, {'1%': -3.4336320721769433, '5%': -2.862989840784964, '10%': -2.56754183359401})

gamma=pd.DataFrame(gamma)
gamma.fillna(method='ffill')

#gammaGood=gamma.iloc[np.where(np.isfinite(gamma)), :]
gammaGood=gamma[gamma.notna().values]
gammalag=gammaGood.shift()
deltaGamma=gammaGood-gammalag
deltaGamma=deltaGamma[1:]
gammalag=gammalag[1:]

X=sm.add_constant(gammalag)
model=sm.OLS(deltaGamma, X)
res=model.fit()
halflife=-np.log(2)/res.params[1]
# 41.095311903707795

lookback=int(halflife)
MA=gamma.rolling(lookback).mean()
MSTD=gamma.rolling(lookback).std()
zScore=(gamma-MA)/MSTD

positions=np.zeros(df.shape)
isExpireDate=np.isfinite(df) & ~np.isfinite(df.shift(-1));
holddays=3*21
numDaysStart=holddays+10
numDaysEnd=10
spreadMonth=12

for c in range(0, df.shape[1]-spreadMonth):
    expireIdx=np.where(isExpireDate.iloc[:,c])[-1]
    if c==0:
        startIdx=max(0, expireIdx-numDaysStart)
        endIdx=expireIdx-numDaysEnd    
    else:
        myStartIdx=endIdx+1
        myEndIdx=expireIdx-numDaysEnd
        if (myEndIdx-myStartIdx >= holddays):
            startIdx=myStartIdx
            endIdx=myEndIdx
        else:
            startIdx=np.Inf
            
    if ((len(expireIdx) > 0) & (endIdx > startIdx)):
        positions[startIdx[0]:endIdx[0], c]=-1
        positions[startIdx[0]:endIdx[0], c+spreadMonth]=1
        
positions[zScore.isna().values.flatten(), :]=0
zScore.fillna(-np.Inf, inplace=True)

positions[zScore.values.flatten() > 0, :]=-positions[zScore.values.flatten() > 0, :]
positions=pd.DataFrame(positions)
pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
(np.cumprod(1+ret)-1).plot()

print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
#APR=0.024347 Sharpe=1.275860
