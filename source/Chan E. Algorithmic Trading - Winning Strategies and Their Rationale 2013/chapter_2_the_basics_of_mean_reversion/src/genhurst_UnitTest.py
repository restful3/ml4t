# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:48:44 2018

@author: Ernest
"""
import numpy as np
from genhurst import genhurst
import matplotlib.pyplot as plt
np.random.seed(1)

z=2*(np.random.random((1000,1))-0.5) # white noise
H, pVal=genhurst(z)
print('White noise: H=%f pVal=%f' % (H, pVal))


z=np.cumsum(z) # random walk 
plt.plot(z)

H, pVal=genhurst(z)
print('Random walk: H=%f pVal=%f' % (H, pVal))