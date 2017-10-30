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
from scipy.special import logit, expit


sevent = 1000
bevent = 10000

np.random.seed(41)

# np.random.RandomState(15)  # deterministic random data
X = np.concatenate((expit(np.random.normal(+2.0, 2.0, sevent)),
                    expit(np.random.normal(-0.5, 2.0, bevent))))
M = np.concatenate((np.random.normal(125, 1.7, sevent),
                    100+np.random.exponential(1/0.05, bevent)))
Y = np.concatenate((np.ones(sevent),
                    np.zeros(bevent)))
W = np.concatenate((0.1*expit(np.random.normal(2, 2, sevent)),
                    5.0*expit(np.random.exponential(2, bevent))))
W = np.concatenate((np.ones(sevent), np.ones(bevent)))

binner = binopt.optimize_bin(nbins=3, range=[0,1],
                             drop_last_bin=True,
                             fix_upper=True,
                             fix_lower=False,
                             use_kde_density=True)
print "AMS 4"
opt = binner.fit(X, Y, sample_weights=W, method="Nelder-Mead",
                 breg=None, fom="AMS2")

s = binner.binned_stats(opt.x)[0]
b = binner.binned_stats(opt.x)[1]
bins = binner.binned_stats(opt.x)[2]

print "bounds : ", opt.x
print "signif : ", binner.binned_score(opt.x)
print "Nsig   : ", s
print "Nbkg   : ", b
print "Z0     : ", s/np.sqrt(b)
print "bins   : ", bins

plt.figure(figsize=(5, 5))
plt.hist(X[Y==1],bins=100, range=[0,1], alpha=1.0, color='red',
         weights=W[Y==1], histtype='step',lw=1, normed=True)
plt.hist(X[Y==0],bins=100, range=[0,1], alpha=1.0, color='blue',
         weights=W[Y==0], histtype='step',lw=1, normed=True)

for ix in opt.x:
    plt.axvline(ix, c="k", ls="--", lw=1.2)

# fig = plt.figure(figsize=(12, 4))
# binner.optimisation_monitoring()
# plt.show()


binner.boundary_scan_2d()
plt.show()
