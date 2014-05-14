"""
Tests for fixholes.py interface and results


Notes
-----
Tests are a little verbose as they were developed along side building
the functionality.
"""

import numpy as np
import pandas as pd
import timeseriestools.filters as filters
import timeseriestools.fixholes as fixholes
from nose.tools import raises
from numpy.testing import (assert_, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose)
from math import isnan
from datetime import datetime

