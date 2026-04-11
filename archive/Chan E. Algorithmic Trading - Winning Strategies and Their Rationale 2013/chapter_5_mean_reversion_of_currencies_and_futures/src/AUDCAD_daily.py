# Example 5.2: Pair Trading AUD.CAD with Rollover Interests

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm

df=pd.read_csv('inputData_AUDCAD_20120426.csv')
#df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d')
df.set_index('Date', inplace=True)

aud=pd.read_csv('AUD_interestRate.csv')
audindex=pd.PeriodIndex(year=aud.Year, month=aud.Month, freq='M')
#aud.index=audindex.to_timestamp().date
aud.index=audindex.to_timestamp()

cad=pd.read_csv('CAD_interestRate.csv')
cadindex=pd.PeriodIndex(year=cad.Year, month=cad.Month, freq='M')
#cad.index=cadindex.to_timestamp().date
cad.index=cadindex.to_timestamp()

df=pd.merge(df, aud, how='outer', left_index=True, right_index=True)
df.drop({'Year', 'Month'}, axis=1, inplace=True)
df.rename({'Rates': 'AUD_Rates'}, axis=1, inplace=True)

df=pd.merge(df, cad, how='outer', left_index=True, right_index=True)
df.drop({'Year', 'Month'}, axis=1, inplace=True)
df.rename({'Rates': 'CAD_Rates'}, axis=1, inplace=True)

df.fillna(method='ffill', axis=0, inplace=True)
df.loc[:,{'AUD_Rates', 'CAD_Rates'}]=df.loc[:,{'AUD_Rates', 'CAD_Rates'}]/365/100 # convert from annual to daily rates

isWednesday=df.index.weekday==2
df.loc[isWednesday,{'AUD_Rates'}]=df.loc[isWednesday,{'AUD_Rates'}]*3

isThursday=df.index.weekday==3
df.loc[isThursday,{'CAD_Rates'}]=df.loc[isThursday,{'CAD_Rates'}]*3

lookback=20

ma=df['Close'].rolling(lookback).mean()
mstd=df['Close'].rolling(lookback).std()
z=(df['Close']-ma)/mstd  

ret=-np.sign(z).shift()*(np.log(df['Close'])+(-np.log(df['Close'])+np.log(1+df['AUD_Rates'])-np.log(1+df['CAD_Rates'])).shift())
(np.cumprod(1+ret)-1).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
# APR=0.064719 Sharpe=0.610818
