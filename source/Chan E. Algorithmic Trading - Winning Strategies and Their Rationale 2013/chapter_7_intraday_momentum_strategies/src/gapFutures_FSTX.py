# Example 7.1: Opening Gap Strategy for FSTX

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

entryZscore=0.1;


df=pd.read_csv('inputDataDaily_FSTX_20120517.csv')
df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
df.set_index('Date', inplace=True)

stdretC2C90d=df['Close'].pct_change().rolling(90).std().shift()

longs= df['Open'] >= df['High'].shift()*(1+entryZscore*stdretC2C90d)
shorts=df['Open'] >= df['Low'].shift()*(1-entryZscore*stdretC2C90d)

positions=np.zeros(longs.shape)

positions[longs]=1
positions[shorts]=-1

ret=positions*(df['Close']-df['Open'])/df['Open']

cumret=(np.cumprod(1+ret)-1)
cumret.plot()

print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
from calculateMaxDD import calculateMaxDD
maxDD, maxDDD, i=calculateMaxDD(cumret.fillna(0))
print('Max DD=%f Max DDD in days=%i' % (maxDD, maxDDD))
#APR=0.074864 Sharpe=0.494857
#Max DD=-0.233629 Max DDD in days=789