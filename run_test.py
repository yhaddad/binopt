import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
plt.style.use('physics')

plt.rcParams['axes.grid'       ]  = False
plt.rcParams['xtick.labelsize' ]  = 14
plt.rcParams['ytick.labelsize' ]  = 14
plt.rcParams['axes.labelsize'  ]  = 14
plt.rcParams['legend.fancybox' ]  = False

pd.options.mode.chained_assignment = None

import binopt
from scipy import special as sp

rng = np.random.RandomState(15)  # deterministic random data
s = sp.expit(rng.normal(loc= +4, scale=3.0,size=10000))
b = sp.expit(rng.normal(loc= -2, scale=2.9,size=10000))
ws = rng.normal(loc= 1 , scale=1 ,size=10000)
wb = rng.normal(loc= 500, scale=1,size=10000)

X = np.concatenate([s,b])
Y = np.concatenate([np.ones(s.shape[0]), np.zeros(b.shape[0])])
W = np.concatenate([ws,wb])


binner = binopt.optimize_bin(nbins=4, range=[0,1], drop_last_bin=False,
                             fix_upper=True, fix_lower=False,
                             use_kde_density=False)

print binner.fit(X,Y, sample_weights=W)
