# Example 5.3: Estimating Spot and Roll Returns Using the Constant Returns Model
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import statsmodels.api as sm
#import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm

df=pd.read_csv('inputDataDaily_C2_20120813.csv')
df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
df.set_index('Date', inplace=True)

# Find spot prices
spot=df['C_Spot']
df.drop('C_Spot', axis=1, inplace=True)

T=sm.add_constant(range(spot.shape[0]))

model=sm.OLS(np.log(spot), T)
res=model.fit() # Note this can deal with NaN in top row

print('Average annualized spot return=', 252*res.params[1])
#Average annualized spot return= 0.02805562210100287

# Fitting gamma to forward curve
gamma=np.full(df.shape[0], np.nan)
for t in range(df.shape[0]):
    idx=np.where(np.isfinite(df.iloc[t, :]))[0]
    idxDiff = np.roll(idx, -1) - idx 
    all_ones = all(idxDiff[0:4]==1) 

    if (len(idx)>=5) and all_ones:
        FT=df.iloc[t, idx[:5]]
        T=sm.add_constant(range(FT.shape[0]))
        model=sm.OLS(np.log(FT.values), T)
        res=model.fit()
        gamma[t]=-12*res.params[1]
          
       
pd.DataFrame(gamma).plot()

print('Average annualized roll return=', np.nanmean(gamma))
# Average annualized roll return= -0.12775650227459556