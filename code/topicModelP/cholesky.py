#!/usr/bin/python
# Author: Suzanna Sia

import numpy as np
import math
import warnings
import pdb

warnings.filterwarnings('error')

def cholupdate(L,x,sign):

    p = np.size(x)
    v = x.copy()

    for k in range(p):

        try:
            if sign == '+':
                r = np.sqrt(L[k,k]**2 + x[k]**2)
            elif sign == '-':
                r = np.sqrt(L[k,k]**2 - x[k]**2)
        except:
            pdb.set_trace()
            #r = np.sqrt(L[k,k]**2 - x[k]**2)
        c = r/L[k,k]
        s = x[k]/L[k,k]
        L[k,k] = r
        if sign == '+':
            L[k,k+1:p] = (L[k,k+1:p] + s*x[k+1:p])/c
            #L[k+1:p, k] = (L[k+1:p, k] + s*x[k+1:p])/c
            
        elif sign == '-':
            # without abs here x explodes..
            L[k,k+1:p] = (L[k,k+1:p] - s*x[k+1:p])/c
            #L[k+1:p, k] = (L[k+1:p, k] - s*x[k+1:p])/c

        x[k+1:p]= c*x[k+1:p] - s*L[k, k+1:p]
        #x[k+1:p]= c*x[k+1:p] - s*L[k+1:p, k]

    return L
