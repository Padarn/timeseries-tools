import gadata as ga
import timeseriestools as ttools
from pylab import plot,show,legend
import numpy as np

ts = ga.recorded_arc().AirTemp_arc.copy()
ts = ts[200:500]
ts, old_means = ttools.trend_removal.standardize_hours(ts, return_old=True)
ts2 = ts.copy()
ts2[100:100+22] = np.nan
xout = range(100,100+22)

fix = ttools.fixholes.kalman_holes_R_seasonal(ts2,sfreq=24)
fix2 = ttools.fixholes.arima_holes_R_seasonal(ts2,sfreq=24)


# unstandardize
ts = ttools.trend_removal.standardize_hours(ts, old_values=old_means)
ts2 = ttools.trend_removal.standardize_hours(ts2, old_values=old_means)
fix = ttools.trend_removal.standardize_hours(fix, old_values=old_means)
fix2 = ttools.trend_removal.standardize_hours(fix2, old_values=old_means)


fix[xout].plot(linestyle='-',linewidth=2,alpha=0.8,legend='kalman')
fix2[xout].plot(linestyle='-',linewidth=2,alpha=0.8,legend='arima')
ts[xout].plot(marker='+',markersize=8,linewidth=0,c='r',alpha=1,legend='real')
ts2.plot(linewidth=1,alpha=0.5,c='r')
legend(['kalman','arima','real'])
show()