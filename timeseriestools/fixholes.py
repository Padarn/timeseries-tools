""" 
Various methods for fixing holes in time series.
"""

import pandas as pd
from rtools.rtools import *

#---------### R USING FUNCTIONS ###--------------#
""" The functions below use R code to do fitting,
    they are reasonably messy as a concenquence and
    so will not be tested extensively for the time
    being - eventually move to statsmodels """

def kalman_holes_R_seasonal(ts, sfreq= None, hole_inds = None):
    """
    Takes a pandas time series and fixes the holes using smoothing
    from a state space model and kalman filtering. EXPERIMENTAL

    Parameters
    ----------
    ts         - pandas time series object, with nans instead of 
                 missing values: this means it has a frequency
    sfreq      - optional seasonal frequency for the time series
    hole_indes - optional argument to specify the indicies at 
                 which holes will be filled, defaults to all nans


    Output
    ------
    The time series with fixed holes

    NOTES
    -----
    THIS IS EXPERIMENTAL - NO GURANTEES IT WORKS
    """

    if sfreq is None:
        sfreq = 1

    # convert ts to R object 
    tsvals = ts.values
    rAssign(tsvals,"tsvals")
    rAssign(sfreq, "sfreq")
    
    # rcode 
    rCode("x = ts(tsvals,deltat=1/sfreq)")
    rCode("mod = tsSmooth(StructTS(x))")
    if sfreq == 1:
        rCode("fit = ts(rowSums(mod))")
    else:
        rCode("fit = ts(rowSums(mod[,-2]))")
    rCode("tsp(fit) = tsp(x)")
    fixed = np.array(rCode("fit",ret=True))

    if hole_inds is None:
        return pd.Series(index=ts.index,data=fixed)

def arima_holes_R_seasonal(ts, arima_order = None, seasonal_order = None,
                          sfreq= None, hole_inds = None):
    """
    Takes a pandas time series and fixes the holes using smoothing
    from a seasonal arima model. EXPERIMENTAL

    Parameters
    ----------
    ts          - pandas time series object, with nans instead of 
                  missing values: this means it has a frequency
    arima_order - vector of arima order, set to (1,0,1) if None
    seasonal_order - same as above
    sfreq       - optional seasonal frequency for the time series
    hole_indes  - optional argument to specify the indicies at 
                  which holes will be filled, defaults to all nans


    Output
    ------
    The time series with fixed holes

    NOTES
    -----
    THIS IS EXPERIMENTAL - NO GURANTEES IT WORKS
    """

    if sfreq is None:
        sfreq = 1
    if arima_order is None:
        arima_order = np.array([1,0,1])
    if seasonal_order is None:
        seasonal_order = np.array([1,0,1])

    # convert ts to R object 
    tsvals = ts.values
    rAssign(tsvals,"tsvals")
    rAssign(sfreq, "sfreq")
    rAssign(arima_order,"arimaorder")
    rAssign(seasonal_order,"seasonalorder")
    
    # rcode 
    rCode("invisible(library(forecast))")
    rCode("x = ts(tsvals,f=sfreq)")
    rCode("mod = auto.arima(x)")
    fixed = np.array(rCode("KalmanSmooth(x,mod$model)$smooth[,1]",ret=True))

    if hole_inds is None:
        return pd.Series(index=ts.index,data=fixed)



