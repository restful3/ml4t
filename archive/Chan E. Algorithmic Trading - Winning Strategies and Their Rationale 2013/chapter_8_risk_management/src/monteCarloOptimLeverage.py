# Box 8.1
import numpy as np
import pandas as pd
#from scipy.stats import describe 
from scipy.stats import pearson3
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm
from scipy.optimize import minimize
 
df=pd.read_csv('AUDCAD_unequal_ret.csv')

skew_, loc_, scale_=pearson3.fit(df) 
print('skew=%f loc=%f scale=%f' % (skew_, loc_, scale_))
mean,var,skew,kurt=pearson3.stats(skew_, loc_, scale_, moments='mvks')
print('mean=%f var=%f skew=%f kurt=%f' % (mean, var, skew, kurt))

ret_sim=pearson3.rvs(skew_, loc_, scale_, size=100000, random_state=0)

def g(f, R):
    return np.sum(np.log(1+f*R), axis=0)/R.shape[0]

myf=range(1, 24)
myg=np.full(24, np.NaN)

for f in myf:
    myg[f]=g(f, ret_sim)

myg=myg[1:]
myg=pd.DataFrame(myg)
myg.plot()

minusG = lambda f : -g(f, df)
minusGsim = lambda f : -g(f, ret_sim)

#optimal leverage based on simulated returns
res = minimize(minusGsim, 0, method='Nelder-Mead')
optimalF=res.x
print('Optimal leverage=%f optimal growth rate=%f' % (optimalF, -res.fun))
#Optimal leverage=25.512625 optimal growth rate=0.005767

minR=np.min(ret_sim)
print('minR=%f' % (minR))
#minR=-0.018201

# max drawdown with optimal leverage
from calculateMaxDD import calculateMaxDD
maxDD, maxDDD, i=calculateMaxDD((np.cumproduct(1+optimalF*ret_sim)-1))
print('Max DD=%f with optimal leverage=%f' % (maxDD, optimalF))
#Max DD=-0.996312 with optimal leverage=25.512625

#max drawdown with half of optimal leverage
maxDD, maxDDD, i=calculateMaxDD((np.cumproduct(1+optimalF/2*ret_sim)-1))
print('Max DD=%f with half optimal leverage=%f' % (maxDD, optimalF/2))
#Max DD=-0.900276 with half optimal leverage=12.756313

# max drawdown with 1/7 of optimal leverage
maxDD, maxDDD, i=calculateMaxDD((np.cumproduct(1+optimalF/7*ret_sim)-1))
print('Max DD=%f with half optimal leverage=%f' % (maxDD, optimalF/7))
#Max DD=-0.429629 with half optimal leverage=3.644661

#max drawdown with 1/1.4 of optimal leverage for historical returns
maxDD, maxDDD, i=calculateMaxDD((np.cumproduct(1+optimalF/1.4*df.values)-1))
print('Max DD=%f with historical returns=%f' % (maxDD, optimalF/1.4))
#Max DD=-0.625894 with historical returns=18.223304

D=0.5
print('Growth rate on simulated returns using D=%3.1f of optimal leverage on full account=%f' % (D, -minusGsim(optimalF*D)))
#Growth rate on simulated returns using D=0.5 of optimal leverage on full account=0.004317
maxDD, maxDDD, i=calculateMaxDD((np.cumproduct(1+optimalF*D*ret_sim)-1))
print('MaxDD on simulated returns using D of optimal leverage on full account=%f' % (maxDD))
#MaxDD on simulated returns using D of optimal leverage on full account=-0.900276

# CPPI
g_cppi=0
drawdown=0
for t in range(ret_sim.shape[0]):
    g_cppi+=np.log(1+ret_sim[t]*D*optimalF*(1+drawdown))
    drawdown=min([0, (1+drawdown)*(1+ret_sim[t])-1])
    
g_cppi=g_cppi/len(ret_sim)
print('Growth rate on simulated returns using CPPI=%f' % g_cppi)
#Growth rate on simulated returns using CPPI=0.004264
print('Growth rate on historical returns using D of optimal leverage on full account=%f' % (-minusG(optimalF*D)))
#Growth rate on historical returns using D of optimal leverage on full account=0.004053
maxDD, maxDDD, i=calculateMaxDD((np.cumproduct(1+optimalF*D*df.values)-1))
print('MaxDD on historical returns using D of optimal leverage on full account=%f' % (maxDD))
#MaxDD on historical returns using D of optimal leverage on full account=-0.303448

# CPPI
g_cppi=0
drawdown=0
for t in range(df.shape[0]):
    g_cppi+=np.log(1+df.iloc[t,]*D*optimalF*(1+drawdown))
    drawdown=np.min([0, (1+drawdown)*(1+df.iloc[t,])-1])
    
g_cppi=g_cppi/len(df)
print('Growth rate on historical returns CPPI=%f' % (g_cppi))
