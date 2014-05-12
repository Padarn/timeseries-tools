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
		ts, coef = trend_removal.remove_ts_trend(ts_plus_linear, self.linear, return_coef = True)
		assert_array_almost_equal(ts,self.ts,1)
		assert_almost_equal(coef,1,1)



