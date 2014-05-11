"""
Filters for Pandas time series for specific purposes in the rest of
the tool set.
"""

import pandas as pd
import numpy as np
from math import isnan


def remove_incomplete_periods(ts, period='d', freq = None):
    """
    Function takes a time series object which has sampling frequency of at 
    least daily and returns a new time seires with all 'periods' which are 
    incomplete removed.

    Parameters 
    ----
    ts - time series to remove from
    freq - input timeseries frequency expected. If None, try and read
           from ts
    period - the type of period to filter on
    """
    


    if freq == None:
        if ts.index.freq == None:
            raise TypeError("Input time series has no frequency. Supply one")
        else:
            freq = ts.index.freqstr

    start = ts.index[0].to_period(period)
    end = ts.index[-1].to_period(period)
    index = pd.date_range(start=start.to_timestamp(),end=(end+1).to_timestamp(),freq=freq)
    index = index[:-1]
    ts = ts.reindex(index)

    #ts = # reindex from start to end using ''period''
    def hasnan(x):
        if np.any(x.apply(isnan)):
            return np.nan
        else:
            return True
    nan_periods = ts.resample(period,how=hasnan)
    nan_periods = nan_periods.dropna().index.to_period(period)
    
    def validperiod(x):
        if x.to_period(period) in nan_periods:
            return True
        else:
            return False
    groups = ts.groupby(validperiod).groups
    return ts[groups.get(True,[])]