import rtools as rtools
import numpy as np
import pandas as pd

def glsmodel(yname, xlin_names, xspline_names, df, extras = None):

    # Push the data with the given names from df to 
    df = df.dropna()

    rtools.rCode('library(nlme)')
    rtools.rCode('library(splines)')


    rtools.rAssign(df[yname].values,yname)
    for xn in xlin_names:
        rtools.rAssign(df[xn].values,xn)
    for xn in xspline_names:
        rtools.rAssign(df[xn].values,xn)

    # Put together rCode
    rcode = 'gls('+yname+' ~ '
    for xn in xlin_names:
        rcode = rcode + xn + '+'
    for xn in xspline_names:
        rcode = rcode + 'bs(' + xn + ',3)' + '+'
    rcode = rcode[:-1]

    # Add on extras 
    varmean = pd.Series(index=df.index,data = np.ones(len(df.index)))
    varmean = varmean*df[yname].mean()
    rtools.rAssign(varmean.values, 'varmean')
    if extras is not None:
        rcode = rcode + ',' + extras

    rcode = rcode +')'
    

    mod = rtools.rCode(rcode, ret = True)

    return mod

def gammodel(yname, xlin, xspline, df, extras = None):

    # Push the data with the given names from df to 
    df = df.dropna()

    rtools.rCode('library(mgcv)')
    rtools.rCode('library(splines)')
    rtools.rAssign(df, 'dataf')

    # Put together rCode
    rcode = 'gamm(' + yname + ' ~ '
    for xn in xlin:
        rcode = rcode +  xn + '+'
    for xn in xspline:
        rcode = rcode + 'te(hour,' + xn + ')+'
    rcode = rcode[:-1]

    # Add on extras 
    varmean = pd.Series(index=df.index,data = np.ones(len(df.index)))
    varmean = varmean*df[yname].mean()
    rtools.rAssign(varmean.values, 'varmean')
    if extras is not None:
        rcode = rcode + ',' + extras

    rcode = rcode +',data=dataf)'
    print rcode
    mod = rtools.rCode(rcode, ret = True)

    return mod

def modelfitted(model, ts = None):

    rtools.rAssign(model,'mod')
    fit = np.array(rtools.rCode("fitted(mod)",ret=True))
    if ts is None:
        return fit
    else:
        fit = pd.Series(index = ts.dropna().index, data = fit)
        fit = fit.reindex(ts.index)
        return fit

def modelpredict(model, newdata):

    newdatap = newdata.dropna()
    rtools.rAssign(model, 'mod')
    rtools.rAssign(newdatap, 'pdata')
    predict = np.array(rtools.rCode("predict(mod,newdata=pdata)",ret=True))
    predict = pd.Series(data= predict, index=newdatap.index)
    predict = predict.reindex(newdata.index)
    return predict

def modelresid(model, rtype = None, ts = None):

    rtools.rAssign(model,'mod')
    if type is None:
        res =  np.array(rtools.rCode("resid(mod)",ret=True))
    else:
        res =  np.array(rtools.rCode("resid(mod,type='"+rtype+"')",ret=True))
    if ts is None:
        return res
    else:
        res = pd.Series(index = ts.dropna().index, data = res)
        res = res.reindex(ts.index)
        return res
