# Example 4.3: Linear Long-Short Model on Stocks

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm

# Stocks
cl_=pd.read_csv('inputDataOHLCDaily_20120424_cl.csv')
stocks=pd.read_csv('inputDataOHLCDaily_20120424_stocks.csv')

cl_['Var1']=pd.to_datetime(cl_['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
cl_.columns=np.insert(stocks.values, 0, 'Date')
cl_.set_index('Date', inplace=True)

cl_=cl_.loc[(cl_.index >= pd.datetime(2007,1, 3).date()) & (cl_.index <= pd.datetime(2011,12,30).date()),:]

op=pd.read_csv('inputDataOHLCDaily_20120424_op.csv')
op['Var1']=pd.to_datetime(op['Var1'],  format='%Y%m%d').dt.date # remove HH:MM:SS
op.columns=np.insert(stocks.values, 0, 'Date')
op.set_index('Date', inplace=True)

op=op.loc[(op.index >= pd.datetime(2007,1, 3).date()) & (op.index <= pd.datetime(2011,12,30).date()),:]

ret=cl_.pct_change() # daily returns

marketRet=np.mean(ret, axis=1) # equal weighted market index return

weights=-(np.array(ret)-np.reshape(marketRet.values, (ret.shape[0], 1)))
weights=weights/pd.DataFrame(np.abs(weights)).sum(axis=1).values.reshape((weights.shape[0], 1))
weights=pd.DataFrame(weights, columns=stocks.values[0], index=np.array(ret.index))

dailyret=(weights.shift()*ret).sum(axis=1) # Capital is always one


((1+dailyret).cumprod()-1).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+dailyret)**(252/len(dailyret))-1, np.sqrt(252)*np.mean(dailyret)/np.std(dailyret)))
# APR=13.7%, Sharpe=1.3

ret=(op-cl_.shift())/cl_.shift() # daily returns

marketRet=np.mean(ret, axis=1) # equal weighted market index return

weights=-(np.array(ret)-np.reshape(marketRet.values, (ret.shape[0], 1)))
weights=weights/pd.DataFrame(np.abs(weights)).sum(axis=1).values.reshape((weights.shape[0], 1))
weights=pd.DataFrame(weights, columns=stocks.values[0], index=np.array(ret.index))

dailyret=(weights*(cl_-op)/op).sum(axis=1) # Capital is always one

((1+dailyret).cumprod()-1).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+dailyret)**(252/len(dailyret))-1, np.sqrt(252)*np.mean(dailyret)/np.std(dailyret)))