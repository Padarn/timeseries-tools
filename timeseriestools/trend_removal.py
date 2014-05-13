from datetime import datetime as dt
import numpy as np
import time
import pandas as pd
import statsmodels.api as sm

def _apply_to_each_col(df, func, param_list):
    """ 
    Generic function to apply a function to each col
    in a df that has key in param_list
    """
    df = df.copy()
    for key in param_list.keys():
        df[key] = func(df[key],param_list[key])
    return df

def _toYearFraction(date):
    """ 
    Converts a date time to a float representation.
    Not intended for use outside of class
    """
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration
    return date.year + fraction

def remove_ts_trend(ts, trend_ts, return_coef = False):
    """
    Remove a constant multiple of the given 'trend_ts' from ts.
    Best fit is found by linear regression.

    Parameters
    ----------
    ts          - time series for which to remove the linear trend.
    trend_ts    - trend time series to remove a multiple of.
    return_coef - if true, also returns the multiplicative coef.

    Output
    -----
    returns the trend removed series, and if return_coef is true
    returns as a second argument the coefficient
    of the fit.
    """
    mod = sm.OLS(ts.values, trend_ts.values)
    res = mod.fit()
    if return_coef:
        return ts - res.fittedvalues, res.params
    else:
        return ts - res.fittedvalues

def remove_linear_trend(ts, return_fit = False):
    """
    Remove a linear trend from a Pandas time series object. The linear
    trend is fit using a least squares fit to the data.

    Parameters
    ----------
    ts         - time series for which to remove the linear trend.
    return_fit - bool, decides if fit is returned as second argument.
                 If true, returns the trend and coefficients:
                       new_ts, [linear_fit, fit]
                 fit is an array fit = [b1,b0], y=b0+b1*x

    Output
    -----
    returns the trend removed series, and if return_coef is true
    returns as a second argument the fitted series and coefficient
    of the fit.
    """
    
    # Scale times
    times = np.array([_toYearFraction(x) for x in ts.index])
    a, b = times[0], times[-1]
    times = (times - a)/(b-a)

    fit = np.polyfit(np.array(times),ts.values,1)
    linear_fit = pd.Series(data = fit[0]*np.array(times)+fit[1],
                           index = ts.index)
    ts= ts-linear_fit
    if return_fit:
        return ts, [linear_fit, fit]
    else:
        return ts

def remove_trig_trend(ts, period, order=1, return_fit = False):
    """
    Remove a trignometric trend from a Pandas time series object. The
    trend is fit using a least squares fit to the data.

    Parameters
    ----------
    ts         - time series for which to remove the trig trend.
    period     - the period of the longest trig cycle.
    order      - the order of the trig fit.
    return_fit - bool, decides if fit is returned as second argument.
                 If true, returns the trend and coefficients:
                       new_ts, [linear_fit, fit]
                 fit is an array fit = [b1,b0], y=b0+b1*x

    Output
    -----
    returns the trend removed series, and if return_coef is true
    returns as a second argument the fitted series and coefficient
    of the fit.
    """

    times = np.array([_toYearFraction(x) for x in ts.index])
    a, b = times[0], times[-1]
    times = (times - a)/(b-a)

    onecycle = _toYearFraction(pd.date_range(start=ts.index[0], periods=2, freq=period)[1])
    onecycle = 2*np.pi/((onecycle-a)/(b-a))

    X = np.ones(len(ts))
    for i in range(order):
        j = i+1
        X = np.column_stack((X, np.sin(onecycle*times*j), np.cos(onecycle*times*j)))

    mod = sm.OLS(ts.values, X)
    res = mod.fit()
    trig_fit = pd.Series(data = res.fittedvalues, index = ts.index)

    if return_fit:
        return ts - trig_fit, [trig_fit, res.params]
    else:
        return ts - trig_fit


def _change_mean(ts, new_mean):
    """
    Performs change_mean on one time series. See doc string of change_mean
    """

    return ts-ts.mean()+new_mean

def change_mean(tsframe, new_mean = None):
    """
    Changes the mean of a time series, or of all columns of a dataframe. 
    If no new mean is specified, changes the mean to zero.

    Parameters
    ----------
    ts_frame - dataframe or time series to change mean of
    new_mean - new mean for data, if dataframe should be dict

    Output
    ------
    the new series/dataframe with changed mean
    """

    if isinstance(tsframe, pd.DataFrame):
        if new_mean is None:
            new_mean = {}
            for key in tsframe.keys():
                new_mean[key]=0
        return _apply_to_each_col(tsframe, _change_mean, new_mean)
    else:
        if new_mean is None:
            return _change_mean(tsframe, 0)
        else:
            return _change_mean(tsframe, new_mean)


def _change_sd(ts, new_sd):
    """
    Performs change_sd on one time series. See doc string of change_sd
    """

    return ts/ts.std()*new_sd

def change_sd(tsframe, new_sd = None):
    """
    Changes the standard deviation of a time series, or of all columns of 
    a dataframe. If no new standard deviation is specified, changes the 
    standard deviation to zero.

    Parameters
    ----------
    ts_frame - dataframe or time series to change standard deviation of
    new_sd   - new standard deviation for data, if dataframe should be dict

    Output
    ------
    the new series/dataframe with changed standard deviation.
    """

    if isinstance(tsframe, pd.DataFrame):
        if new_sd is None:
            new_sd = {}
            for key in tsframe.keys():
                new_sd[key]=1
        return _apply_to_each_col(tsframe, _change_sd, new_sd)
    else:
        if new_sd is None:
            return _change_sd(tsframe, 1)
        else:
            return _change_sd(tsframe, new_sd)


def _standardize(ts, old_values = None, return_old = False):
    """
    Performs standardize on one time series. See doc string of standardize
    """

    if old_values == None:
        mean = ts.mean()
        sd = ts.std()
        if return_old:
            return (ts-mean)/sd, [mean, sd]
        else:
            return (ts-mean)/sd
    else:
        old_mean = old_values[0]
        old_sd = old_values[1]
        return ts*old_sd+old_mean

def standardize(tsframe, old_values = None, return_old = False):
    """
    Standardizes a time series or all columns of a dataframe. That is the mean
    is set to zero and the standard deviation to ones. If old mean and standard
    deviation are provided then the process is reversed instead.

    Parameters
    ----------
    tsframe -    dataframe or time seires to standardize
    old_values - old values of mean and standard deiviation in a dict if is to be
                 reversed.
    return_old - if true, old mean and standard deviations are returned.

    Output
    ------
    the new standardized (or reversed) series/dataframe, and also the old mean 
    and standard deviation as a second return argument is return_old=True
    """

    # NOTE : THIS SHOULD BE GENERALISED SO DON"T NEED ALL THIS 
    # EXTRA STUFF THAT SHOULD BE HANDLED BY  _apply_to_each_col 
    if isinstance(tsframe, pd.DataFrame):
        if old_values is None:
            old_values = {}
            df = tsframe.copy()
            for key in param_list.keys():
                df[key], old_values[key] =  _standardize(df[key], 
                                              return_old = True)
            if return_old:
                return df, old_values
            else:
                return df
    else:
        if old_values is None:
            if return_old:
                return _standardize(tsframe, return_old = True)
            else:
                return _standardize(tsframe)
        else:
            return _standardize(tsframe, old_values = old_values)

