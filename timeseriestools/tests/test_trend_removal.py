"""
Tests for tremd_removal.py interface and results


Notes
-----
Tests are a little verbose as they were developed along side building
the functionality.
"""

import numpy as np
import pandas as pd
import timeseriestools.trend_removal as trend_removal
from nose.tools import raises
from numpy.testing import (assert_, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose,
                           assert_array_almost_equal)
from math import isnan
from datetime import datetime


class TestRemovalFilters(object):

    def __init__(self):
        np.random.seed(1234)
        rng = pd.date_range('1/1/2011', periods=50*24, freq='1h')
        self.ts = pd.Series(np.random.randn(len(rng)), index=rng)
        self.linear = np.linspace(-10,10,len(self.ts))
        self.linear = pd.Series(data=self.linear,index=self.ts.index)
        self.trig = 10*np.sin(np.linspace(0,10*np.pi,len(self.ts)))
        self.trig = pd.Series(data=self.trig,index=self.ts.index)

    def test_remove_linear_without_fit(self):
        ts_plus_linear = self.ts + self.linear
        ts = trend_removal.remove_linear_trend(ts_plus_linear, return_fit = False)
        assert_array_almost_equal(ts,self.ts,1)

    def test_remove_linear_with_fit(self):
        ts_plus_linear = self.ts + self.linear
        ts, [linear_fit, fit] = trend_removal.remove_linear_trend(ts_plus_linear, return_fit = True)
        assert_array_almost_equal(ts,self.ts,1)
        assert_array_almost_equal(linear_fit,self.linear,1)
        assert_array_almost_equal(np.array([20,-10]),fit,1)

    def test_dummy_trig_frame(self):
        dummydf = trend_removal.dummy_trig_trend(self.ts.index, '12M', order = 2)
        assert_equal([u'period_12M_o_1_sin', u'period_12M_o_1_cos', u'period_12M_o_2_sin', u'period_12M_o_2_cos'],
                      dummydf.keys())
        dummydf = trend_removal.dummy_trig_trend(self.ts.index, '1h', order = 1)
        assert_almost_equal(np.ones(len(dummydf)), dummydf.period_1h_o_1_cos,2)
        

    def test_remove_trig_without_fit_order1(self):
        ts_plus_trig = self.ts + self.trig
        ts = trend_removal.remove_trig_trend(ts_plus_trig, period='10d' ,return_fit = False)
        assert_array_almost_equal(ts,self.ts,1)

    def test_remove_linear_without_fit_order2_overspecified(self):
        ts_plus_trig = self.ts + self.trig
        ts, [trig_fit, fit] = trend_removal.remove_trig_trend(ts_plus_trig, period='10d', order=2, return_fit = True)
        ##Note: This seems more sensitve than it should be.
        assert_array_almost_equal(ts[10:20],self.ts[10:20],1) 

    def test_remove_linear_with_fit_order1(self):
        ts_plus_trig = self.ts + self.trig
        ts, [trig_fit, fit] = trend_removal.remove_trig_trend(ts_plus_trig, period='10d' ,return_fit = True)
        assert_array_almost_equal(ts,self.ts,1)
        assert_array_almost_equal(trig_fit,self.trig,1)
        assert_array_almost_equal(np.array([0.0, 10.0, 0]),fit,1)

    def test_remove_trend_linear_no_coef(self):
        ts_plus_linear = self.ts + self.linear
        ts = trend_removal.remove_ts_trend(ts_plus_linear, self.linear, return_coef = False)
        assert_array_almost_equal(ts,self.ts,1)

    def test_remove_trend_linear_coef(self):
        ts_plus_linear = self.ts+ self.linear
        ts, [fit, coef] = trend_removal.remove_ts_trend(ts_plus_linear, self.linear, return_coef = True)
        assert_array_almost_equal(ts,self.ts,1)
        assert_almost_equal(coef,1,1)


class TestStandardizationScaling(object):

    def __init__(self):
        np.random.seed(1234)
        rng = pd.date_range('1/1/2011', periods=24*50, freq='1h')
        self.mu = 5
        self.mu2 = -3
        self.sigma = 2
        self.sigma2 = 3
        self.ts = pd.Series(np.random.randn(len(rng))*self.sigma+self.mu, index=rng)
        self.df = pd.DataFrame(index=self.ts.index)
        self.df['ts1']=pd.Series(np.random.randn(len(rng))*self.sigma+self.mu, index=rng)
        self.df['ts2']=pd.Series(np.random.randn(len(rng))*self.sigma2+self.mu2, index=rng)

    def test_ts_set_hour_mean(self):
        ts = self.ts
        ts = trend_removal.change_mean(ts)
        assert_almost_equal(ts.mean(), 0)
        ts = trend_removal.change_mean(ts, 15)
        assert_almost_equal(ts.mean(),15)

    def test_df_set_sd(self):
        df = self.df
        df = trend_removal.change_sd(df)
        assert_almost_equal(df['ts1'].std(), 1)
        assert_almost_equal(df['ts2'].std(), 1)
        df = trend_removal.change_sd(df, {'ts1':15,'ts2':5})
        assert_almost_equal(df['ts1'].std(), 15)
        assert_almost_equal(df['ts2'].std(), 5)

    def test_ts_standardize(self):
        trend_removal.standardize(self.ts)
        ts = self.ts
        ts, old = trend_removal.standardize(ts, return_old=True)
        assert_almost_equal(ts.mean(), 0)
        assert_almost_equal(ts.std(),1)
        ts = trend_removal.standardize(ts, old_values=old)
        assert_almost_equal(ts.mean(),self.ts.mean())
        assert_almost_equal(ts.std(),self.ts.std())

    def test_hours_standardize(self):
        df = self.df
        df, old = trend_removal.standardize_hours(df,return_old=True)
        gstd = df.groupby(lambda x:x.hour).get_group(5)['ts1']
        g = self.df.groupby(lambda x:x.hour).get_group(5)['ts1']
        assert_almost_equal(gstd.mean(),0)
        assert_almost_equal(gstd.std(),1)
        df = trend_removal.standardize_hours(df,old_values=old)
        gstd = df.groupby(lambda x:x.hour).get_group(5)['ts1']
        assert_almost_equal(gstd.mean(),g.mean())
        assert_almost_equal(gstd.std(),g.std())

    def test_zero_sigma_standardize(self):
        ## TODO - add test that checks that zero series are just 
        # left alone or handled somehow