# Example 4.1: Buy-on-Gap Model on SPX Stocks

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm

op=pd.read_csv('inputDataOHLCDaily_20120424_op.csv')
cl=pd.read_csv('inputDataOHLCDaily_20120424_cl.csv')

stocks=pd.read_csv('inputDataOHLCDaily_20120424_stocks.csv')

op['Var1']=pd.to_datetime(op['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
op.columns=np.insert(stocks.values, 0, 'Date')
op.set_index('Date', inplace=True)

cl['Var1']=pd.to_datetime(cl['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
cl.columns=np.insert(stocks.values, 0, 'Date')
cl.set_index('Date', inplace=True)

earnann=pd.read_csv('earnannFile.csv')
earnann['Date']=pd.to_datetime(earnann['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
earnann.set_index('Date', inplace=True)

np.testing.assert_array_equal(stocks.iloc[0,:], earnann.columns)

df=pd.merge(op, cl, how='inner', left_index=True, right_index=True, suffixes=('_op', '_cl'))
df=pd.merge(earnann, df, how='inner', left_index=True, right_index=True)

earnann=df.iloc[:, 0:(earnann.shape[1])].astype(bool)
op=df.iloc[:, (earnann.shape[1]):((earnann.shape[1])+op.shape[1])]
cl=df.iloc[:, ((earnann.shape[1])+op.shape[1]):]

op.columns=stocks.iloc[0,:]
cl.columns=stocks.iloc[0,:]

lookback=90

retC2O=(op-cl.shift())/cl.shift()
stdC2O=retC2O.rolling(lookback).std()

positions=np.zeros(cl.shape) 

longs=  (retC2O >=  0.5*stdC2O) & earnann
shorts= (retC2O <= -0.5*stdC2O) & earnann

positions[longs]=1
positions[shorts]=-1

ret=np.sum(positions*(cl-op)/op, axis=1)/30

cumret=(np.cumprod(1+ret)-1)
cumret.plot()

print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
from calculateMaxDD import calculateMaxDD
maxDD, maxDDD, i=calculateMaxDD(cumret.fillna(0))
print('Max DD=%f Max DDD in days=%i' % (maxDD, maxDDD))
#APR=0.068126 Sharpe=1.494743
#Max DD=-0.026052 Max DDD in days=109