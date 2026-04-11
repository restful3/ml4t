# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:41:31 2018

@author: Ernest
"""

# Example 2: Using ADF Test for Mean Reversion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from genhurst import genhurst 

df=pd.read_csv('inputData_USDCAD.csv')

y=df.loc[df['Time']==1659, 'Close']

plt.plot(y)


results=adfuller(y, maxlag=1, regression='c', autolag=None)
print(results)

# Find Hurst exponent
H, pVal=genhurst(np.log(y))
print("H=%f pValue=%f" % (H, pVal))