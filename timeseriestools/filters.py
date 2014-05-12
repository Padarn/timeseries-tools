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
    period - the type of period to filter on
    freq - input timeseries frequency expected. If None, try and read
           from ts
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

def group_continuous_segments_indices(ts, freq = None):
    """
    Take a pandas time series which may and nan's and or gaps in the record
    and return a list of indices corresponding to the continous subseries.

    Parameter 
    ---------
    ts - time series to remove from
    freq - input timeseries frequency expected. If None, try and read
           from ts
    """

    if freq == None:
        if ts.index.freq == None:
            raise TypeError("Input time series has no frequency. Supply one")
        else:
            freq = ts.index.freqstr

    # reindex so all missing values are now nan
    start = ts.index[0]
    end = ts.index[-1]
    index = pd.date_range(start=start,end=end,freq=freq)
    ts = ts.reindex(index)

    # consume time seires inserting into a list to return
    ts_notnan = 1-ts.apply(isnan)
    ts_notnan[ts_notnan==0]=-1
    diff = ts_notnan[1:].values-ts_notnan[0:-1].values
    downs = np.where(diff==-2)[0]
    ups = np.where(diff==2)[0]+1
    if not isnan(ts[0]):
        ups = np.insert(ups,0,0)
    if not isnan(ts[-1]):
        downs = np.append(downs,len(ts)-1) 
    return [list(x) for x in zip(ups,downs)]

def longest_continuous_segment(ts, freq = None, return_all=False):
    """
    Take a pandas time series which may have nan's and or gaps in the record
    and returns a time series representing the longest avaliable continous 
    subseries.

    Parameters 
    ----------
    ts - time series to get segment from
    freq - input timeseries frequency expected. If None, try and read
           from ts
    return_all - in the case of a tie for the longest, return_all being
                 true means all are returned. Otherwise the first is.

    Notes
    -----
    This function just uses the above 'group_continuous_segments' and takes
    the longest.
    """

    indices = group_continuous_segments_indices(ts, freq = freq)
    lens = [x[1]-x[0] for x in indices]
    maxlens = np.max(lens)
    longest = np.where(lens==maxlens)[0]
    if(len(longest)==1):
        x = indices[longest]
        return ts[x[0]:x[1]+1]
    elif return_all == False:
        x = indices[longest[0]]
        return ts[x[0]:x[1]+1]
    else:
        indices_longest = [indices[x] for x in longest]
        return [ts[x[0]:x[1]+1] for x in indices_longest]



def group_continuous_segments_indices_multivariate(df, freq = None):
    """
    Take a pandas data frame which is indexed by a time series index and
    returns the indices ofcontinous segments over which all of the variables
    of the data frame are avaliable and valid.

    Parameters
    ----------
    df - dataframe to get segment from
    freq - input timeseries frequency expected. If None, try and read
           from ts
    """

    if freq == None:
        if df.index.freq == None:
            raise TypeError("Input time series has no frequency. Supply one")
        else:
            freq = df.index.freqstr

    start = df.index[0]
    end = df.index[-1]
    index = pd.date_range(start=start,end=end,freq=freq)
    df = df.reindex(index)
    ts = df.apply(sum,axis=1)

    return group_continuous_segments_indices(ts, freq)

def longest_continous_segment_multivariate(df, freq = None, return_all = False):
    """
    Take a pandas data frame which is indexed by a time series index and
    returns the longest continous segment over which all of the variables of 
    the data frame are avaliable and valid.

    Parameters
    ----------
    df - dataframe to get segment from
    freq - input timeseries frequency expected. If None, try and read
           from ts
    return_all - in the case of a tie for the longest, return_all being
                true means all are returned. Otherwise the first is.
    """

    if freq == None:
        if df.index.freq == None:
            raise TypeError("Input time series has no frequency. Supply one")
        else:
            freq = df.index.freqstr

    start = df.index[0]
    end = df.index[-1]
    index = pd.date_range(start=start,end=end,freq=freq)
    df = df.reindex(index)
    ts = df.apply(sum,axis=1)

    ts_longest = longest_continuous_segment(ts, freq, return_all)
    return df.reindex(ts_longest.index)




