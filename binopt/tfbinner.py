# -*- coding: utf-8 -*-
"""Core methods."""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import (MaxNLocator, NullLocator, ScalarFormatter)
import tools
from scipy.stats import norm
import numdifftools as nd
from .core import binner_base
import tensorflow as tf


np.set_printoptions(precision=4)


class tfbinner(binner_base):
    """Optimse bining of lableled data."""
    def __init__(self, nbins, range, drop_last_bin=True):
        """Init."""
        self.nbins = nbins
        self.range = range
        self.drop_last_bin = drop_last_bin
        self.X = None
        self.y = None
        self.result = None
        self.x_init = None
        self.fom = "AMS3"
        self.sample_weights = None
        self.scan = np.ones([1, self.nbins+1])

    def _fom_(self, s, b, berr=0, method="AMS2"):
        """
        Builtin figure of metrits for boundary optimisation.
        4 methods are currently implemented :
        - AMS1 : $Z_0=s/\sqrt{s+b+\sigma_b^2}$
        - AMS2 : $Z_0=s/\sqrt{b+\sigma_b^2}$
        - AMS3 : $Z_0=\sqrt{2(s+b+sigma_b^2)\ln(1+\frac{s}{b+\sigma_b^2})-s}$
                 the $\sigma_b$ should be set to 0 for physical results
        - AMS4 : Significance as defined in [0] but including systematic
                 uncertainty on b
        [0] Eur. Phys. J. C (2011) 71: 1554 DOI 10.1140/epjc/s10052-011-1554-0
        """
        c = np.zeros(b.shape[0])
        with np.errstate(divide='ignore', invalid='ignore'):
            if method == "AMS1":
                c = tools.divide(s, np.sqrt(s + b + berr**2))
            elif method == "AMS2":
                c = tools.divide(s, np.sqrt(b + berr**2))
            elif method == "AMS3":
                term_a = (s + b + berr**2)
                term_b = (1 + tools.divide(s, (b + berr**2)))
                c = np.sqrt(2*(term_a * np.log(term_b) - s))
            elif method == "AMS4":
                term_a = tools.divide((s+b)*(b+berr**2), ((s+b)*berr**2)+b**2)
                term_b = 1 + tools.divide(s*berr**2, b*(b+berr**2))
                rad = (s+b)*np.log(term_a)-tools.divide(b**2, berr**2)*np.log(term_b)
                c = np.sqrt(2*rad)
            c[~ np.isfinite(c)] = 0
        return c

    def binned_score(self, x, breg=None):
        """Binned score."""
        _bins_ = np.sort(np.insert(x, [0, x.shape[0]], [self.range]))
        nb_, _ = np.histogram(self.X[self.y == 0], bins=_bins_,
                              range=self.range,
                              weights=self.sample_weights[self.y == 0])
        ns_, _ = np.histogram(self.X[self.y == 1], bins=_bins_,
                              range=self.range,
                              weights=self.sample_weights[self.y == 1])
        wb2_, _ = np.histogram(self.X[self.y == 1], bins=_bins_,
                               range=self.range,
                               weights=self.sample_weights[self.y == 1]**2)
        error_nb_ = np.sqrt(wb2_)
        if nb_.shape != ns_.shape:
            return 0
        else:
            if breg is None:
                return self._fom_(ns_, nb_, berr=error_nb_, method=self.fom)
            else:
                return self._fom_(ns_, nb_, berr=breg, method=self.fom)

    def cost_fun(self, x, breg=None, lower_bound=None, upper_bound=None):
        """Cost function."""
        z = None
        x = np.sort(x)
        # print "cost function : ", x
        if upper_bound is not None:
            x[x >= upper_bound] = upper_bound
        if lower_bound is not None:
            x[x <= lower_bound] = lower_bound
        # print "     function : ", x, lower_bound, upper_bound
        if self.use_kde_density:
            if breg is None:
                z = self.binned_score_density(x, self.breg)
            else:
                z = self.binned_score_density(x, breg)
        else:
            if breg is None:
                z = self.binned_score(x, self.breg)
            else:
                z = self.binned_score(x, breg)
        if self.drop_last_bin:
            # _v_ = np.insert(np.sort(x), 0, [-np.sqrt((z[1:]**2).sum())])
            # self.scan = np.insert(self.scan, 0, _v_, axis=0)
            return -np.sqrt((z[1:]**2).sum())
        else:
            # _v_ = np.insert(np.sort(x), 0, -np.sqrt((z**2).sum()))
            # self.scan = np.insert(self.scan, 0, _v_, axis=0)
            return -np.sqrt((z**2).sum())

    def fit(self, X, y, sample_weights=None, fom="AMS2",
            method="TNC", breg=None, min_args={}):
        """Fitting.
        There figure of merits are supported for now :
        AMS1

        """
        self.X = X
        self.y = y
        self.fom = fom
        self.breg = breg

        if sample_weights is not None:
            self.sample_weights = sample_weights
        else:
            self.sample_weights = np.ones(X.shape[0])

        if self.x_init is None:
            """ remove the fixed bounds """
            self.x_init = np.linspace(self.range[0],
                                      self.range[1],
                                      self.nbins+2)[1:-1]
        np.random.seed(555)
        _bounds_ = np.array([self.range for i in range(self.nbins)])

        if "differential_evolution":
            self.result = optimize.differential_evolution(
                self.cost_fun,
                bounds=_bounds_,
                **min_args
            )
            self.x_init = self.result.x
            self.result.x = np.sort(self.result.x)
        if "Nelder-Mead" in method:
            self.result = optimize.minimize(
                self.cost_fun, self.x_init,
                args=(min(self.range),
                      max(self.range)),
                bounds=_bounds_,
                method='Nelder-Mead'
            )
            self.x_init = self.result.x
            self.result.x = np.sort(self.result.x)
        return self.result

    def optimisation_monitoring(self, fig=None):
        """ Monitoring the convergence of the fit"""
        if fig is None:
            fig = plt.figure(figsize=(15, 5))
        plt.subplots_adjust(hspace=0.001)
        ax1 = plt.subplot(211)
        self.scan = np.matrix(self.scan)
        self.scan = self.scan[np.argsort(self.scan.A[:, 0])[::-1]]
        for ix in range(1, self.nbins):
            ax1.plot(range(self.scan.shape[0]), self.scan[:, ix])

        ax1.set_ylabel('bin boundaries')
        ax2 = plt.subplot(212)
        ax2.plot(range(self.scan.shape[0]), self.scan[:, 0])
        ax2.set_ylabel('cost function')
        ax2.set_xlabel('optimization steps')
        xticklabels = ax1.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        plt.tight_layout()
        return fig

    def boundary_scan_2d(self, fig=None,
                          title_fmt=".2f",
                          max_n_ticks=5, steps=0.01,
                          label='parameter_scan'):
        """Fit result displayed on matrix."""
        if self.nbins <= 1:
            return None
        tx = np.arange(self.range[0], self.range[1], steps)
        ty = np.arange(self.range[0], self.range[1], steps)
        xx, yy = np.meshgrid(tx, ty)
        K = self.nbins
        factor = 2.0
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
        whspace = 0.05          # w/hspace size
        plotdim = factor * K + factor * (K - 1.) * whspace
        dim = lbdim + plotdim + trdim

        if fig is None:
            fig, axes = plt.subplots(K, K, figsize=(dim, dim))
        else:
            try:
                axes = np.array(fig.axes).reshape((K, K))
            except:
                raise ValueError("Provided figure has {0} axes, but data has "
                                 "dimensions K={1}".format(len(fig.axes), K))
        lb = lbdim / dim
        tr = (lbdim + plotdim) / dim
        fig.subplots_adjust(left=lb, bottom=lb,
                            right=tr, top=tr,
                            wspace=whspace, hspace=whspace)

        for i in range(0, self.nbins):
            ax = axes[i, i]

            def _fun_1d(x):
                """Fn 1d."""
                _param_ = [ix for ix in self.result.x]
                _param_[i] = x
                return self.cost_fun(np.array(_param_))
            vec_fun_1d = np.vectorize(_fun_1d)
            z1d = vec_fun_1d(tx)

            ax.plot(tx, z1d, 'red')
            ax.axvline(x=self.result.x[i], color='blue', ls="--")
            ax.set_xlim(self.range)
            ax.yaxis.tick_right()
            # if i > 0:
                # ax.set_yticklabels([])
            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(
                    MaxNLocator(max_n_ticks, prune="lower")
                )
            if i < (self.nbins - 1):
                ax.set_xticklabels([])
            else:
                [l.set_rotation(90) for l in ax.get_xticklabels()]
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=True)
                )
                ax.xaxis.set_label_text("$x_%i$" % i)
            if i == 0:
                [l.set_rotation(0) for l in ax.get_yticklabels()]
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=True)
                )
                ax.yaxis.set_label_text("cost function")

            for j in range(0, self.nbins):
                ax = axes[i, j]
                if j > i:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                elif j == i:
                    continue

                def _fun_(x, y):
                    """Fn 2D."""
                    _param_ = [ix for ix in self.result.x]
                    _param_[i] = x
                    _param_[j] = y
                    if x != y:
                        return self.cost_fun(np.array(_param_))
                    else:
                        return 0.0

                vec_fun_ = np.vectorize(_fun_)
                zz = vec_fun_(xx, yy)
                levels = np.linspace(zz.min(), 0.7*zz.min(), 5)
                ax.contourf(xx, yy, zz,
                            np.linspace(zz.min(), 0.7*zz.min(), 20),
                            cmap=plt.cm.Spectral_r)
                # C = ax.contour(xx, yy, zz, levels,
                #                 linewidth=0.1, colors='black')
                # ax.clabel(C, inline=1, fontsize=5)
                ax.plot(self.result.x[j], self.result.x[i],
                        'ro', label='best fit')
                ax.set_xlim(self.range)
                ax.set_ylim(self.range)
                if max_n_ticks == 0:
                    ax.xaxis.set_major_locator(NullLocator())
                    ax.yaxis.set_major_locator(NullLocator())
                else:
                    ax.xaxis.set_major_locator(
                        MaxNLocator(max_n_ticks, prune="lower")
                    )
                    ax.yaxis.set_major_locator(
                        MaxNLocator(max_n_ticks, prune="lower")
                    )
                if i < self.nbins - 1:
                    ax.set_xticklabels([])
                else:
                    [l.set_rotation(90) for l in ax.get_xticklabels()]
                    ax.xaxis.set_major_formatter(
                        ScalarFormatter(useMathText=True)
                    )
                    ax.xaxis.set_label_text("$x_%i$" % j)
                if j > 0:
                    ax.set_yticklabels([])
                else:
                    [l.set_rotation(0) for l in ax.get_yticklabels()]
                    ax.yaxis.set_major_formatter(
                        ScalarFormatter(useMathText=True)
                    )
                    ax.yaxis.set_label_text("$x_%i$" % i)
        return fig

    def covariance_matrix(self):
        """Covariance matrix."""
        pass
