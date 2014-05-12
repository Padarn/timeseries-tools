from datetime import datetime as dt
import numpy as np
import time
import pandas as pd
import statsmodels.api as sm

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
    ts - time series for which to remove the linear trend.
    trend_ts - trend time series to remove a multiple of.
    return_coef - if true, also returns the multiplicative coef.
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
    ts - time series for which to remove the linear trend.
    return_fit - bool, decides if fit is returned as second argument.
                 If true, returns the trend and coefficients:
                       new_ts, [linear_fit, fit]
                 fit is an array fit = [b1,b0], y=b0+b1*x
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
    ts - time series for which to remove the trig trend.
    period - the period of the longest trig cycle.
    order - the order of the trig fit.
    return_fit - bool, decides if fit is returned as second argument.
                 If true, returns the trend and coefficients:
                       new_ts, [linear_fit, fit]
                 fit is an array fit = [b1,b0], y=b0+b1*x
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

