
import itertools
import matplotlib.pyplot as plt

colors = itertools.cycle(['r','g','b','c','y','m','k'])
markers = itertools.cycle(['o','s','v'])

import pylab
from pylab import arange,pi,sin,cos,sqrt
def setup_pylab_fig():
    import pylab
    from pylab import arange,pi,sin,cos,sqrt
    fig_width_pt = 390.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    params = {'backend': 'ps',
       'axes.labelsize': 10,
      'text.fontsize': 10,
      'legend.fontsize': 10,
      'xtick.labelsize': 8,
      'ytick.labelsize': 8,
      'text.usetex': True,
      'figure.figsize': fig_size}
    pylab.rcParams.update(params)
    # Generate data
    # Plot data
    pylab.figure(1)
    pylab.clf()