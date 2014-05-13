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

class TestContinuousSegmentFunctions(object):

	def __init__(self):
		np.random.seed(1234)
		rng = pd.date_range('1/1/2011', periods=500, freq='1h')
		self.ts = pd.Series(np.random.randn(len(rng)), index=rng)

	def test_get_continuous_groups_multiple_missing(self):
		ts = self.ts
		ts = ts[0:100].append(ts[200:500])
		ts[300]=np.nan
		indices = filters.group_continuous_segments_indices(ts, freq='1h')
		assert_equal(indices,[[0, 99], [100, 299], [301, 399]])

	def test_two_tied_return_one(self):
		ts = self.ts
		ts = ts[0:100].append(ts[200:300])
		ts_seg = filters.longest_continuous_segment(ts, freq='1h')
		assert_array_equal(ts_seg.values, ts[0:100].values)

	def test_two_tied_return_all(self):
		ts = self.ts
		ts = ts[0:100].append(ts[200:300])
		ts_segs = filters.longest_continuous_segment(ts, freq='1h', return_all=True)
		assert_array_equal(ts_segs[0].values, self.ts[0:100].values)
		assert_array_equal(ts_segs[1].values, self.ts[200:300].values)

	def test_get_longest_start_nan(self):
		ts = self.ts
		ts[0:100] = np.nan
		ts_seg = filters.longest_continuous_segment(ts, freq='1h')
		assert_array_equal(ts_seg.values, ts[100:].values)

	def test_get_longest_end_nan(self):
		ts = self.ts
		ts[400:] = np.nan
		ts_seg = filters.longest_continuous_segment(ts, freq='1h')
		assert_array_equal(ts_seg.values, ts[:400].values)

	def test_get_longest_multiple_middle_missing(self):
		ts = self.ts
		ts[300:400] = np.nan
		ts = filters.remove_incomplete_periods(ts, period='1h')
		ts[250] = np.nan
		ts[200:202] = np.nan
		ts_seg = filters.longest_continuous_segment(ts, freq='1h')
		assert_array_equal(ts_seg.values, ts[:200].values)

	def test_get_indices_three_series_different_missing(self):
		df = pd.DataFrame(index = self.ts.index)
		df['ser1'] = df['ser2'] = df['ser3'] = self.ts
		df['ser1'][100] = df['ser1'][450] = np.nan
		df['ser2'][400] = np.nan
		df['ser3'][50:200] = np.nan
		df = filters.group_continuous_segments_indices(df, freq = '1h')
		assert_array_equal(df,np.array([[0, 49], [200, 399], [401, 449], [451, 499]]))

	def test_get_longest_three_series_different_missing(self):
		df = pd.DataFrame(index = self.ts.index)
		df['ser1'] = df['ser2'] = df['ser3'] = self.ts
		df_cp = df.copy()
		df['ser1'][100] = df['ser1'][450] = np.nan
		df['ser2'][400] = np.nan
		df['ser3'][50:200] = np.nan
		df = filters.longest_continuous_segment(df, freq = '1h')
		assert_equal(df.values,df_cp[200:400].values)


