"""
Filters for Pandas time series for specific purposes in the rest of
the tool set.
"""

import pandas as pd
import numpy as np
from math import isnan


def _find_time_index(i, index1, index2):
    time = index1.index[i]
    newi = np.where(index2.index==time)
    return newi[0][0]

def _incomplete_to_nan(data, period = None, freq = None):
    """
    Takes an Pandas Series or DataFrame and returns a copy of it where any
    missing values at the input frequncy are replaced by NaNs rather than 
    just being excluded. Thus the index of the returned object is a continous
    block of time.

    Parameters 
    ----------
    data    - time indexed Series or DataFrame
    period  - period over which to build new index - defaults to freq
    freq    - frequency of the time index, if None attempted to be read from 
              indexed

    Output
    ------
    returns new reindexed object with nans in any required locations.
    """

    # If no input frequency try and read from the data
    if freq == None:
        if data.index.freq == None:
            raise TypeError("Input time series has no frequency. Supply one")
        else:
            freq = data.index.freqstr

    if period == None:
        period = freq

    # Build new index
    start = data.index[0].to_period(period)
    end = data.index[-1].to_period(period)
    index = pd.date_range(start=start.to_timestamp(),end=(end+1).to_timestamp(),freq=freq)
    index = index[:-1]
    data = data.reindex(index)

    return data

def remove_incomplete_periods(ts, period='d', freq = None):
    """
    Function takes a time series object which has sampling frequency of at 
    least daily and returns a new time seires with all 'periods' which are 
    incomplete removed.

    Parameters 
    ----------
    ts - time series to remove from
    period - the type of period to filter on
    freq - input timeseries frequency expected. If None, try and read
           from ts

    Output
    ------
    new time series object with incomplete periods missing
    """
    
    # replace all missing at 'freq' with nan
    ts = _incomplete_to_nan(ts, period, freq)

    # resample at 'period' by seeing if any elements are nan
    def hasnan(x):
        if np.any(x.apply(isnan)):
            return np.nan
        else:
            return True
    nan_periods = ts.resample(period,how=hasnan)
    nan_periods = nan_periods.dropna().index.to_period(period)
    
    # define a group of all valid periods
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

    Output
    ------
    pairs of indicies for the input series that deliniate the period where
    the series is not nan.
    """

    # convert missing to nan
    ts_wnan = _incomplete_to_nan(ts, freq = freq)

    # find nan-notnan change points in full series
    ts_notnan = 1-ts_wnan.apply(isnan)
    ts_notnan[ts_notnan==0]=-1
    diff = ts_notnan[1:].values-ts_notnan[0:-1].values
    downs = np.where(diff==-2)[0]
    ups = np.where(diff==2)[0]+1
    # account for end points
    if not isnan(ts_wnan[0]):
        ups = np.insert(ups,0,0)
    if not isnan(ts_wnan[-1]):
        downs = np.append(downs,len(ts_wnan)-1) 

    # convert indices back to indices for input series
    ts_ups = [_find_time_index(i, ts_wnan, ts) for i in ups]
    ts_downs = [_find_time_index(i, ts_wnan, ts) for i in downs]

    # return pairs
    return [list(x) for x in zip(ts_ups,ts_downs)]


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

    Output
    ------
    series corresponding to the longest continous period. If return_all
    is true, and there are muliple series, then a list containing all of 
    them is returned.

    Notes
    -----
    This function just uses the above 'group_continuous_segments' and takes
    the longest.
    """

    # Get continous block indices
    indices = group_continuous_segments_indices(ts, freq = freq)
    
    # Find lengths of blocks and select the maximums
    lens = [x[1]-x[0] for x in indices]
    maxlens = np.max(lens)
    longest = np.where(lens==maxlens)[0]
    
    # Return the longest, or the first longest if tied.
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

    Output
    ------
    returns the indicies of the segments of continous non nan values for 
    all columns. That is, segments are segments where all series are non
    nan.
    """

    # turn non continous segments into nans
    df = _incomplete_to_nan(df, freq = freq)

    # combine nans from all rows to get concurrent
    ts = df.apply(sum,axis=1)

    # now works exactly like time series method
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

    Output
    ------
    the dataframe slice of the longest continous period with all series 
    avaliable.
    """

    # turn non continous segments into nans
    df = _incomplete_to_nan(df, freq = freq)

    # combine nans from all rows to get concurrent
    ts = df.apply(sum,axis=1)

    # now indices works exactly like time series method
    ts_longest = longest_continuous_segment(ts, freq, return_all)
    return df.reindex(ts_longest.index)




