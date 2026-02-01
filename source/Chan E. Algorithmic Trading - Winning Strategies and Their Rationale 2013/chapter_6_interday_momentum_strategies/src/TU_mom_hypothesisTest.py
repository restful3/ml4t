# Hypothesis Testing on a Futures Momentum Strategy

import numpy as np
import pandas as pd
#from scipy.stats import describe
from scipy.stats import pearson3

df=pd.read_csv('TU.csv')
df['Time']=pd.to_datetime(df['Time']).dt.date # remove HH:MM:SS
df.set_index('Time', inplace=True)

lookback=250
holddays=25

longs= df['Close'] > df['Close'].shift()
shorts= df['Close'] < df['Close'].shift()

pos=np.zeros(df.shape[0])

for h in range(0, holddays):
    long_lag=longs.shift(h)
    long_lag[long_lag.isna()]=False
    long_lag=long_lag.astype(bool)

    short_lag=shorts.shift(h)
    short_lag[short_lag.isna()]=False
    short_lag=short_lag.astype(bool)

    pos[long_lag]=pos[long_lag]+1
    pos[short_lag]=pos[short_lag]-1
    
capital=np.nansum(np.array(pd.DataFrame(abs(pos)).shift()), axis=1)
pos[capital==0,]=0
capital[capital==0]=1

marketRet=df['Close'].pct_change()

ret=np.nansum(np.array(pd.DataFrame(pos).shift())*np.array(marketRet), axis=1)/capital/holddays
 
sharpe=np.sqrt(len(ret))*np.nanmean(ret)/np.nanstd(ret)

print("Gaussian Test statistic=%f" % sharpe)
#Gaussian Test statistic=2.769741

# Randomized market returns hypothesis test
# =============================================================================
#_,_,mean,var,skew,kurt=describe(marketRet, nan_policy='omit')
# =============================================================================
skew_, loc_, scale_=pearson3.fit(marketRet[1:]) # First element is NaN
numSampleAvgretBetterOrEqualObserved=0
for sample in range(10000):
    marketRet_sim=pearson3.rvs(skew=skew_, loc=loc_, scale=scale_, size=marketRet.shape[0], random_state=sample)
    cl_sim=np.cumproduct(1+marketRet_sim)-1
    
    longs_sim =cl_sim > pd.Series(cl_sim).shift(lookback)
    shorts_sim=cl_sim < pd.Series(cl_sim).shift(lookback)
    
    pos_sim=np.zeros(cl_sim.shape[0])
    
    for h in range(0, holddays):
        long_sim_lag=longs_sim.shift(h)
        long_sim_lag[long_sim_lag.isna()]=False
        long_sim_lag=long_sim_lag.astype(bool)
    
        short_sim_lag=shorts_sim.shift(h)
        short_sim_lag[short_sim_lag.isna()]=False
        short_sim_lag=short_sim_lag.astype(bool)
    
        pos_sim[long_sim_lag]=pos_sim[long_sim_lag]+1
        pos_sim[short_sim_lag]=pos_sim[short_sim_lag]-1
        
        capital=np.nansum(np.array(pd.DataFrame(abs(pos_sim)).shift()), axis=1)
        pos_sim[capital==0,]=0
        capital[capital==0]=1
               
        ret_sim=np.nansum(np.array(pd.DataFrame(pos_sim).shift())*np.array(marketRet_sim), axis=1)/capital/holddays
        if (np.mean(ret_sim) >= np.mean(ret)):
            numSampleAvgretBetterOrEqualObserved=numSampleAvgretBetterOrEqualObserved+1
            
print("Randomized prices: p-value=%f" % (numSampleAvgretBetterOrEqualObserved/10000))
#Randomized prices: p-value=23.617800

# Randomized entry trades hypothesis test
numSampleAvgretBetterOrEqualObserved=0
for sample in range(10000):
    P=np.random.permutation(len(longs))
    longs_sim=longs[P]
    shorts_sim=shorts[P]
    
    pos_sim=np.zeros(cl_sim.shape[0])

    for h in range(0, holddays):
        long_sim_lag=longs_sim.shift(h)
        long_sim_lag[long_sim_lag.isna()]=False
        long_sim_lag=long_sim_lag.astype(bool)
    
        short_sim_lag=shorts_sim.shift(h)
        short_sim_lag[short_sim_lag.isna()]=False
        short_sim_lag=short_sim_lag.astype(bool)
    
        pos_sim[long_sim_lag]=pos_sim[long_sim_lag]+1
        pos_sim[short_sim_lag]=pos_sim[short_sim_lag]-1
        
        capital=np.nansum(np.array(pd.DataFrame(abs(pos_sim)).shift()), axis=1)
        pos_sim[capital==0,]=0
        capital[capital==0]=1
               
        ret_sim=np.nansum(np.array(pd.DataFrame(pos_sim).shift())*np.array(marketRet), axis=1)/capital/holddays
        if (np.mean(ret_sim) >= np.mean(ret)):
            numSampleAvgretBetterOrEqualObserved=numSampleAvgretBetterOrEqualObserved+1
            
print("Randomized trades: p-value=%f" % (numSampleAvgretBetterOrEqualObserved/10000))
#Randomized trades: p-value=1.365600