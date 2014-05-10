"""
Filters for Pandas time series for specific purposes in the rest of
the tool set.
"""

import pandas as pd
import numpy as np
from math import isnan


def remove_incomplete_periods(ts, period='d'):
    """
    Function takes a time series object which has sampling frequency of at 
    least daily and returns a new time seires with all 'periods' which are 
    incomplete removed.
    """
    
    freq = ts.index.freqstr
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