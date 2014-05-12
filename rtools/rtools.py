import pandas.rpy.common as com
import rpy2.robjects as robj
import pandas as pd
import rpy2.robjects.numpy2ri as npr
import numpy as np

def rAssign(obj,name):
	rr = robj.R()
	if isinstance(obj,list):
		obj = np.array(obj)
	if isinstance(obj, np.ndarray):
		obj = npr.numpy2ri(obj)
		rr.assign(name,obj)
	elif isinstance(obj,pd.DataFrame):
		obj = com.convert_to_r_dataframe(dframe)
		rr.assign(name, obj)
	else:
		rr.assign(name, obj)

def rPrint(name):
	robj.r("print("+name+")")


def rCode(code, ret=False):
	if ret:
		return robj.r(code)
	else:
		_ = robj.r(code)
