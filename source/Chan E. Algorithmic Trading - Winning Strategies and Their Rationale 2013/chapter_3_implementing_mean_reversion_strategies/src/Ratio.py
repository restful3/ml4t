# Trading Ratio

import numpy as np
import pandas as pd

df=pd.read_csv('inputData_GLD_USO.csv')
df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
df.set_index('Date', inplace=True)

lookback=20
ratio=df['USO']/df['GLD']
ratio.plot()

#yport=np.sum(ts.add_constant(-hedgeRatio)[:, [1,0]]*np.log(df), axis=1)
#yport.plot()

# Apply a simple linear mean reversion strategy to GLD-USO
numUnits =-(ratio-ratio.rolling(lookback).mean())/ratio.rolling(lookback).std() # capital invested in portfolio in dollars.  movingAvg and movingStd are functions from epchan.com/book2
positions=pd.DataFrame(np.tile(numUnits.values, [2, 1]).T * np.ones((numUnits.shape[0], 2)) * np.array([-1, 1]) ) # positions in dollar invested
pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
# (np.cumprod(1+ret)-1).plot()
pd.DataFrame((np.cumprod(1+ret)-1)).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
