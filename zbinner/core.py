"""Core methods."""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.ticker import ScalarFormatter
import kde
np.set_printoptions(precision=4)


class binner_base(object):
    """Abstract class for classification based binning."""

    def __init__(self, nbins, range):
        """Init."""
        self.nbins = nbins
        self.range = range
        raise NotImplementedError('Method not implemented')

    def fit(self, X, y, sample_weights=None):
        """Fitting method."""
        raise NotImplementedError('Method not implemented')
        return self


class optimize_bin(binner_base):
    """Optimse bining of lableled data."""

    def __init__(self, nbins, range,
                 drop_last_bin=True,
                 fix_upper=True,
                 fix_lower=False,
                 use_kde_density=True):
        """Init."""
        self.nbins = nbins
        self.range = range
        self.drop_last_bin = drop_last_bin
        self.X = None
        self.y = None
        self.pdf_s = None
        self.pdf_b = None
        self.result = None
        self.fix_upper = fix_upper
        self.fix_lower = fix_lower
        self.breg = 0
        self.use_kde_density = use_kde_density
        self.sample_weights = None
        self.scan = {"bounds": [], "cost": []}

    def _fom_(self, s, b, breg=10):
        """Default figure-of-merit."""
        c = np.zeros(s.shape[0])
        c[(s == 0) & (b == 0)] = 0
        c[(s+b) != 0] = s[(s+b) != 0] / np.sqrt((s+b+breg)[(s+b) != 0])
        return c

    def binned_score(self, x):
        """Binned score."""
        nb_, _ = np.histogram(self.X[self.y == 0], bins=x, range=self.range,
                              weights=self.sample_weights[self.y == 0])
        ns_, _ = np.histogram(self.X[self.y == 1], bins=x, range=self.range,
                              weights=self.sample_weights[self.y == 1])
        if nb_.shape != ns_.shape:
            return 0
        else:
            return self._fom_(ns_, nb_)

    def binned_score_density(self, x):
        """Binned score after KDE estimation of the distributions."""
        ns_ = self.sample_weights[self.y == 1].sum()
        ns_ *= np.array([self.pdf_s.integrate_box_1d(x[i], x[i+1])
                         for i in range(x.shape[0]-1)])
        nb_ = self.sample_weights[self.y == 0].sum()
        nb_ *= np.array([self.pdf_b.integrate_box_1d(x[i], x[i+1])
                         for i in range(x.shape[0]-1)])
        if nb_.shape != ns_.shape:
            return 0
        else:
            return self._fom_(ns_, nb_)

    def cost_fun(self, x, lower_bound=None, upper_bound=None):
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
            z = self.binned_score_density(x)
        else:
            z = self.binned_score(x)

        self.scan['bounds'].append(np.sort(x))
        self.scan['cost'].append(z)

        if self.drop_last_bin:
            return -np.sqrt((z[1:]**2).sum())
        else:
            return -np.sqrt((z**2).sum())

    def fit(self, X, y, sample_weights=None):
        """Fitting."""
        self.X = X
        self.y = y
        if sample_weights is not None:
            self.sample_weights = sample_weights
        else:
            self.sample_weights = np.ones(X.shape[0])

        x_init = np.linspace(self.range[0], self.range[1], self.nbins+1)
        np.random.seed(555)
        _bounds_ = np.array([self.range for i in range(self.nbins + 1)])

        if self.use_kde_density:
            self.pdf_s = kde.gaussian_kde(
                                self.X[self.y == 1],
                                weights=self.sample_weights[self.y == 1])
            self.pdf_b = kde.gaussian_kde(
                                self.X[self.y == 0],
                                weights=self.sample_weights[self.y == 0])
            self.result = optimize.minimize(self.cost_fun, x_init,
                                            args=(min(self.range),
                                                  max(self.range)),
                                            bounds=_bounds_,
                                            method='Nelder-Mead')
            self.result.x = np.sort(self.result.x)
        else:
            min_args = {
                "method": "Powell",
                "args": (self.range[0], self.range[1])
            }
            # bound_up = np.array([max(self.range) for i in range(self.nbins+1)])
            # bound_dw = np.array([min(self.range) for i in range(self.nbins+1)])
            self.result = optimize.basinhopping(self.cost_fun, x_init,
                                                minimizer_kwargs=min_args,
                                                niter=3000)
            self.result.x = np.sort(self.result.x)
        return self.result

    def optimisation_monitoring_(self, fig=None):
        """Motinotor the health of the fit."""
        if fig is None:
            fig = plt.figure(figsize=(10, 2))

        plt.subplots_adjust(hspace=0.001)
        ax1 = plt.subplot(211)
        for ix in range(self.result.x.shape[0]):
            ax1.plot(range(np.array(self.scan['bounds']).shape[0]),
                     np.array(self.scan['bounds'])[:, ix])

        ax1.set_ylabel('bin boundaries')
        ax2 = plt.subplot(212)
        ax2.plot(range(np.array(self.scan['bounds']).shape[0]),
                 np.sort(-np.array(self.scan['cost'])))
        ax2.set_ylabel('cost function')
        ax2.set_xlabel('optimisation steps')
        xticklabels = ax1.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        return fig

    def parameter_scan_2d(self, fig=None,
                          title_fmt=".2f",
                          max_n_ticks=5,
                          label='parameter_scan'):
        """Fit result displayed on matrix."""

        if self.nbins <= 2:
            return None
        tx = np.arange(self.range[0], self.range[1], 0.01)
        ty = np.arange(self.range[0], self.range[1], 0.01)
        xx, yy = np.meshgrid(tx, ty)
        K = self.nbins - 1
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

        for i in range(1, self.nbins):
            ax = axes[i-1, i-1]

            def _fun_1d(x):
                """Fn 1d."""
                _param_ = [self.result.x[k] for k in range(self.nbins+1)]
                _param_[i] = x
                _param_[0] = self.range[0]
                _param_[-1] = self.range[1]
                return self.cost_fun(np.array(_param_))

            vec_fun_1d = np.vectorize(_fun_1d)
            z1d = vec_fun_1d(tx)
            ax.plot(tx, z1d)
            ax.axvline(x=self.result.x[i], color='red', ls="--")
            ax.set_xlim(self.range)
            if i > 1:
                ax.set_yticklabels([])
            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(
                    MaxNLocator(max_n_ticks, prune="lower")
                )
            if i < self.nbins - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(90) for l in ax.get_xticklabels()]
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=True)
                )
                ax.xaxis.set_label_text("$x_%i$" % i)
            if i == 1:
                [l.set_rotation(0) for l in ax.get_yticklabels()]
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=True)
                )
                ax.yaxis.set_label_text("cost function")

            for j in range(1, self.nbins):
                ax = axes[i-1, j-1]
                if j > i:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                elif j == i:
                    continue

                def _fun_(x, y):
                    """Fn 2D."""
                    _param_ = [self.result.x[k] for k in range(self.nbins+1)]
                    _param_[i] = x
                    _param_[j] = y
                    _param_[0] = self.range[0]
                    _param_[-1] = self.range[1]
                    return self.cost_fun(np.array(_param_))

                vec_fun_ = np.vectorize(_fun_)
                zz = vec_fun_(xx, yy)
                levels = np.linspace(zz.min(), 0.95*zz.min(), 5)
                ax.contourf(xx, yy, zz,
                            np.linspace(zz.min(), 0.85*zz.min(), 20),
                            cmap=plt.cm.Spectral_r)
                C = ax.contour(xx, yy, zz, levels,
                               linewidth=0.1, colors='black')
                ax.clabel(C, inline=1, fontsize=5)
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
                if j > 1:
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
