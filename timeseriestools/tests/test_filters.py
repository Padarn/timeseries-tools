"""
Tests for filters.py interface and results


Notes
-----
Tests are a little verbose as they were developed along side building
the functionality.
"""

import numpy as np
import pandas as pd
import timeseriestools.filters as filters
from nose.tools import raises
from numpy.testing import (assert_, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose)
from math import isnan
from datetime import datetime


class TestRemovalFilters(object):

	def __init__(self):
		np.random.seed(1234)
		rng = pd.date_range('1/1/2011', periods=72, freq='1h')
		self.ts = pd.Series(np.random.randn(len(rng)), index=rng)

	def test_remove_one_day(self):
		ts = self.ts
		ts['1-1-2011 05']=np.nan
		ts = filters.remove_incomplete_periods(ts)
		assert(not datetime(2011,1,1) in ts.resample('d'))

	def test_remove_one_hour(self):
		ts = self.ts
		ts['1-1-2011 05']=np.nan
		ts = filters.remove_incomplete_periods(ts,period='1h')
		assert(not datetime(2011,1,1,5) in ts.index)

	def test_remove_nothing(self):
		index = self.ts.index
		ts = self.ts
		ts = filters.remove_incomplete_periods(ts, period='1h')
		assert_equal(ts.index, index)

	def test_remove_all(self):
		ts = self.ts
		ts[ts.resample('d').index]=np.nan
		ts = filters.remove_incomplete_periods(ts,'d')
		assert(len(ts)==0)

	@raises(TypeError)
	def test_no_feq_exception(self):
		ts = self.ts
		ts[0]=np.nan
		ts = filters.remove_incomplete_periods(ts, period='1h')
		ts = filters.remove_incomplete_periods(ts, period='d')

	def test_remove_hour_then_day_beginning(self):
		ts = self.ts
		ts[0]=np.nan
		ts = filters.remove_incomplete_periods(ts, period='1h')
		ts = filters.remove_incomplete_periods(ts, period='d', freq='1h')
		for i in range(24):
			assert(not datetime(2011,1,1,i) in ts.index)


	def test_remove_hour_then_day_middle(self):
		ts = self.ts
		ts[40]=np.nan
		ts = filters.remove_incomplete_periods(ts, period='1h')
		ts = filters.remove_incomplete_periods(ts, period='d', freq='1h')
		for i in range(24):
			assert(not datetime(2011,1,2,i) in ts.index)

	def test_remove_hour_then_day_end(self):
		ts = self.ts
		ts[-1]=np.nan
		ts = filters.remove_incomplete_periods(ts, period='1h')
		ts = filters.remove_incomplete_periods(ts, period='d', freq='1h')
		assert(not datetime(2011,1,3) in ts.resample('d'))


