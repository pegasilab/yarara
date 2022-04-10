"""
@author: Cretignier Michael 
@university University of Geneva
"""

import os
import sys

cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])


import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from scipy.interpolate import interp1d
from statsmodels.stats.weightstats import DescrStatsW
from wpca import EMPCA
from wpca import PCA as PCA_BACK
from wpca import WPCA

from . import my_functions as myf
from .my_functions import rm_outliers as rm_out

plt_version = float("".join(matplotlib.__version__.split(".")))


class table(object):
    """this classe has been establish with pandas DataFrame"""

    def __init__(self, array):
        self.table = array
        self.dim = np.shape(array)

    def rms_w(self, weights, axis=1):
        average = np.average(self.table, weights=weights, axis=axis)

        if axis == 1:
            data_recentered = self.table - average[:, np.newaxis]
        if axis == 0:
            data_recentered = (self.table.T - average[:, np.newaxis]).T

        variance = np.average((data_recentered) ** 2, weights=weights, axis=axis)
        self.rms = np.sqrt(variance)

    def WPCA(self, pca, weight=None, comp_max=None, m=2, kind="inter"):
        """from https://github.com/jakevdp/wpca/blob/master/WPCA-Example.ipynb
        enter which pca do yo want either 'pca', 'wpca' or 'empca'
        empca slower than wpca

        """

        # self.replace_outliers(m=m, kind=kind)

        Signal = self.table

        R = Signal.copy()

        if pca == "pca":
            ThisPCA = PCA_BACK
        elif pca == "wpca":
            ThisPCA = WPCA
        elif pca == "empca":
            ThisPCA = EMPCA
        elif pca == "pca_backup":
            # ThisPCA = PCA
            pass

        if (weight is None) | (pca == "pca"):
            kwds = {}
        else:
            kwds = {"weights": np.sqrt(weight)}  # defined as 1/sigma

        if comp_max == None:
            comp_max = len(R.T)

        # Compute the PCA vectors & variance
        pca = ThisPCA(n_components=comp_max).fit(R, **kwds)

        # Reconstruct the data using the PCA model
        self.components = ThisPCA(n_components=comp_max).fit_transform(R, **kwds).T
        self.vec_fitted = ThisPCA(n_components=comp_max).fit_reconstruct(R, **kwds)
        self.vec = pca.components_.T

        norm = np.sign(np.nanmedian(self.vec, axis=0))
        self.vec = self.vec / norm
        self.components = self.components * norm[:, np.newaxis]

        # self.s_values = pca.singular_values_

        # components = abs_coeff*abs(self.components)+(1-abs_coeff)*self.components

        self.phi_components = np.sum(self.components < 0, axis=1) / len(self.components.T)
        self.zscore_components = np.mean(self.components, axis=1) / np.std(self.components, axis=1)

        self.var = pca.explained_variance_
        self.var_ratio = pca.explained_variance_ratio_

        self.wpca_model = pca

    def fit_base(self, base_vec, weight=None, num_sim=1):
        """weights define as 1/sigma**2 self.table = MxT, base_vec = NxT, N the number of basis element"""

        if np.shape(base_vec)[1] != np.shape(self.table)[0]:
            base_vec = np.array(
                [
                    base_vec[i] * np.ones(np.shape(self.table)[0])[:, np.newaxis]
                    for i in range(len(base_vec))
                ]
            )

        if (np.shape(self.table)[0] == np.shape(self.table)[1]) & (len(np.shape(self.table)) == 2):
            base_vec = np.array(
                [
                    base_vec[i] * np.ones(np.shape(self.table)[0])[:, np.newaxis]
                    for i in range(len(base_vec))
                ]
            )

        if weight is None:
            weight = np.ones(np.shape(self.table))

        coeff = np.array(
            [
                np.linalg.lstsq(
                    base_vec[:, i, :].T * np.sqrt(weight[i])[:, np.newaxis],
                    (self.table[i]) * np.sqrt(weight[i]),
                    rcond=None,
                )[0]
                for i in range(len(self.table))
            ]
        )

        vec_bootstrap = np.array(
            [
                (self.table[i] + np.random.randn(num_sim, len(weight[i])) / np.sqrt(weight[i]))
                for i in range(len(self.table))
            ]
        )

        coeff_test = np.array(
            [
                np.linalg.lstsq(
                    base_vec[:, i, :].T * np.sqrt(weight[i])[:, np.newaxis],
                    (vec_bootstrap[i] * np.sqrt(weight[i])).T,
                    rcond=None,
                )[0]
                for i in range(len(self.table))
            ]
        )
        coeff_mean = np.mean(coeff_test, axis=2)
        coeff_std = np.std(coeff_test, axis=2)

        self.vec_resampling = vec_bootstrap
        self.coeff_resampling = coeff_test
        self.coeff_mean = coeff_mean
        self.coeff_std = coeff_std

        vec_fitted = np.array(
            [np.sum(coeff[j] * base_vec[:, j, :].T, axis=1) for j in range(len(self.table))]
        )
        all_vec_fitted = np.array([coeff[j] * base_vec[:, j, :].T for j in range(len(self.table))])
        self.coeff_fitted = coeff
        self.vec_fitted = vec_fitted
        self.all_vec_fitted = all_vec_fitted
        vec_residues = self.table - vec_fitted
        vec_residues[self.table == 0] = 0
        self.vec_residues = vec_residues
        self.weights = weight
        coeff_fitted_std = []
        for i in range(len(self.table)):
            X = np.mat(base_vec[:, i, :]).T
            XX = np.linalg.inv(X.T * X)
            V = np.mat(np.diag(weight[i] ** -1))
            cov = XX * X.T * V * X * XX
            coeff_fitted_std.append(np.sqrt(np.diag(cov)))

        self.coeff_fitted_std = np.array(coeff_fitted_std)

        self.chi2 = np.sum(vec_residues**2 * weight, axis=1)
        coeff_pos = coeff * np.sign(np.median(coeff, axis=0))
        mean_coeff = np.mean(coeff_pos, axis=0)
        if sum(mean_coeff != 0):
            epsilon = 1e-6 * np.min(abs(mean_coeff[mean_coeff != 0]))
        else:
            epsilon = 1e-6
        self.zscore_base = mean_coeff / (np.std(coeff_pos, axis=0) + epsilon)
        self.phi_base = np.sum(coeff_pos < 0, axis=0) / len(coeff_pos)


class tableXY(object):
    def __init__(self, x, y, *yerr):
        self.stats = pd.DataFrame({}, index=[0])
        self.y = np.array(y)  # vector of y

        if x is None:  # for a fast table initialisation
            x = np.arange(len(y))
        self.x = np.array(x)  # vector of x

        try:
            np.sum(self.y)
        except:
            self.y = np.zeros(len(self.x))
            self.yerr = np.ones(len(self.y))

        if len(x) != len(y):
            print("X et Y have no the same lenght")

        if len(yerr) != 0:
            if len(yerr) == 1:
                self.yerr = np.array(yerr[0])
                self.xerr = np.zeros(len(self.x))
            elif len(yerr) == 2:
                self.xerr = np.array(yerr[0])
                self.yerr = np.array(yerr[1])
        else:
            if sum(~np.isnan(self.y.astype("float"))):
                self.yerr = np.ones(len(self.x)) * myf.mad(
                    rm_out(self.y.astype("float"), m=2, kind="sigma")[1]
                )
                if not np.sum(abs(self.yerr)):
                    self.yerr = np.ones(len(self.x))
            else:
                self.yerr = np.ones(len(self.x))
            self.xerr = np.zeros(len(self.x))

    def rms_w(self):
        if len(self.x) > 1:
            self.rms = DescrStatsW(self.y, weights=1.0 / self.yerr**2).std
            self.weighted_average = DescrStatsW(self.y, weights=1.0 / self.yerr**2).mean
            self.stats["rms"] = self.rms
        else:
            self.rms = 0
            self.weighted_average = self.y[0]

    def copy(self):
        return tableXY(self.x.copy(), self.y.copy(), self.xerr.copy(), self.yerr.copy())

    def switch(self):
        self.x, self.y = self.y, self.x
        self.xerr, self.yerr = self.yerr, self.xerr

    def stack(self):
        self.x = np.hstack(self.x)
        self.y = np.hstack(self.y)

    def order(self, order=None):
        if order is None:
            order = self.x.argsort()
        self.order_liste = order
        self.x = self.x[order]
        self.y = self.y[order]
        self.xerr = self.xerr[order]
        self.yerr = self.yerr[order]

    def null(self):
        self.yerr = 0 * self.yerr

    def clip(self, min=[None, None], max=[None, None], replace=True, invers=False):
        """This function seems sometimes to not work without any reason WARNING"""
        min2 = np.array(min).copy()
        max2 = np.array(max).copy()
        if min2[1] == None:
            min2[1] = np.nanmin(self.y) - 1
        if max2[1] == None:
            max2[1] = np.nanmax(self.y) + 1
        masky = (self.y <= max2[1]) & (self.y >= min2[1])
        if min2[0] == None:
            min2[0] = np.nanmin(self.x) - 1
        if max2[0] == None:
            max2[0] = np.nanmax(self.x) + 1
        maskx = (self.x <= max2[0]) & (self.x >= min2[0])
        mask = maskx & masky
        try:
            self.clip_mask = self.clip_mask & mask
        except:
            self.clip_mask = mask

        if invers:
            mask = ~mask
        self.clipped = tableXY(self.x[mask], self.y[mask], self.xerr[mask], self.yerr[mask])
        if replace == True:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.yerr = self.yerr[mask]
            self.xerr = self.xerr[mask]
        else:
            self.clipx = self.x[mask]
            self.clipy = self.y[mask]
            self.clipyerr = self.yerr[mask]
            self.clipxerr = self.xerr[mask]

    def supress_nan(self):
        mask = ~np.isnan(self.x) & ~np.isnan(self.y) & ~np.isnan(self.yerr) & ~np.isnan(self.xerr)
        if sum(~mask) == len(mask):
            self.replace_nan()
        else:
            self.mask_not_nan = mask
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.xerr = self.xerr[mask]
            self.yerr = self.yerr[mask]

    def replace_nan(self):
        self.y[np.isnan(self.y)] = np.random.randn(sum(np.isnan(self.y)))
        self.x[np.isnan(self.x)] = np.random.randn(sum(np.isnan(self.x)))
        self.yerr[np.isnan(self.yerr)] = np.random.randn(sum(np.isnan(self.yerr)))
        self.xerr[np.isnan(self.xerr)] = np.random.randn(sum(np.isnan(self.xerr)))

    def supress_mask(self, mask):
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.xerr = self.xerr[mask]
        self.yerr = self.yerr[mask]

    def center_symmetrise(self, center, replace=False, Plot=False):
        x = self.x
        kernel = self.copy()
        window = np.min([np.max(x) - center, center - np.min(x)])
        nb_elem = len(x) * 10

        new_grid = np.ravel(np.linspace(center - window, center + window, 2 * nb_elem - 1))
        kernel.interpolate(new_grid=new_grid)

        sym_kernel = np.ravel(
            np.mean(np.array([kernel.y[0 : nb_elem - 1], kernel.y[nb_elem:][::-1]]), axis=0)
        )
        sym_kernel = np.hstack([sym_kernel, kernel.y[nb_elem - 1], sym_kernel[::-1]])
        sym_kernel = tableXY(new_grid, sym_kernel)
        sym_kernel.interpolate(new_grid=x)

        if Plot:
            self.plot(ls="-")
            sym_kernel.plot(color="r", ls="-")

        if replace:
            self.y = sym_kernel.y
        else:
            self.y_sym = sym_kernel.y

    def plot(
        self,
        Show=False,
        color="k",
        label="",
        ls="",
        offset=0,
        mask=None,
        capsize=3,
        fmt="o",
        markersize=6,
        zorder=1,
        species=None,
        alpha=1,
        modulo=None,
        modulo_norm=False,
        cmap="viridis",
        new=False,
        phase_mod=0,
        periodic=False,
        frac=1,
        yerr=True,
    ):

        """For the mask give either the first and last index in a list [a,b] or the mask boolean"""

        if (len(self.x) > 10000) & (ls == "") & (modulo is None):
            ls = "-"

        if species is None:
            species = np.ones(len(self.x))

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors_species = [color] + prop_cycle.by_key()["color"]

        for num, selection in enumerate(np.unique(species)):

            color = colors_species[num]

            if num != 0:
                label = None

            if mask is None:
                mask2 = np.ones(len(self.x)).astype("bool")
            elif type(mask[0]) == int:
                mask2 = np.zeros(len(self.x)).astype("bool")
                mask2[mask[0] : mask[1]] = True
            else:
                mask2 = mask

            loc = np.where(species[mask2] == selection)[0]

            sel = np.arange(len(loc))
            if frac != 1:
                sel = np.random.choice(
                    np.arange(len(loc)), size=int(frac * len(loc)), replace=False
                )

            if new:
                plt.figure()

            if ls != "":
                plt.plot(
                    self.x[mask2][loc][sel],
                    self.y[mask2][loc][sel] + offset,
                    ls=ls,
                    zorder=zorder,
                    label=label,
                    color=color,
                    alpha=alpha,
                )
            else:
                if modulo is not None:
                    norm = (1 - modulo_norm) + modulo_norm * modulo
                    new_x = ((self.x[mask2][loc] - phase_mod) % modulo) / norm
                    plt.errorbar(
                        new_x[sel],
                        self.y[mask2][loc][sel] + offset,
                        xerr=self.xerr[mask2][loc][sel],
                        yerr=self.yerr[mask2][loc][sel] * int(yerr),
                        fmt=fmt,
                        color=color,
                        alpha=alpha,
                        capsize=capsize,
                        label=label,
                        markersize=markersize,
                        zorder=zorder,
                    )
                    plt.scatter(
                        new_x[sel],
                        self.y[mask2][loc][sel] + offset,
                        marker="o",
                        c=self.x[mask2][loc][sel],
                        cmap=cmap,
                        s=markersize,
                        zorder=zorder * 100,
                    )
                    if periodic:
                        for i in range(int(np.float(periodic))):
                            plt.errorbar(
                                new_x[sel] - (i + 1) * modulo / norm,
                                self.y[mask2][loc][sel] + offset,
                                xerr=self.xerr[mask2][loc][sel],
                                yerr=self.yerr[mask2][loc][sel],
                                fmt=fmt,
                                color=color,
                                alpha=0.3,
                                capsize=capsize,
                                markersize=markersize,
                                zorder=zorder,
                            )
                            plt.errorbar(
                                new_x[sel] + (i + 1) * modulo / norm,
                                self.y[mask2][loc][sel] + offset,
                                xerr=self.xerr[mask2][loc][sel],
                                yerr=self.yerr[mask2][loc][sel],
                                fmt=fmt,
                                color=color,
                                alpha=0.3,
                                capsize=capsize,
                                markersize=markersize,
                                zorder=zorder,
                            )
                else:
                    plt.errorbar(
                        self.x[mask2][loc][sel],
                        self.y[mask2][loc][sel] + offset,
                        xerr=self.xerr[mask2][loc][sel],
                        yerr=self.yerr[mask2][loc][sel] * int(yerr),
                        fmt=fmt,
                        color=color,
                        alpha=alpha,
                        capsize=capsize,
                        label=label,
                        markersize=markersize,
                        zorder=zorder,
                    )
        if Show == True:
            plt.legend()
            plt.show()

    def find_max(self, vicinity=3):
        self.index_max, self.y_max = myf.local_max(self.y, vicinity=vicinity)
        self.index_max = self.index_max.astype("int")
        self.x_max = self.x[self.index_max.astype("int")]
        self.max_extremum = tableXY(self.x_max, self.y_max)

    def find_min(self, vicinity=3):
        self.index_min, self.y_min = myf.local_max(-self.y, vicinity=vicinity)
        self.index_min = self.index_min.astype("int")
        self.y_min *= -1
        self.x_min = self.x[self.index_min.astype("int")]
        self.min_extremum = tableXY(self.x_min, self.y_min)

    def smooth(self, box_pts=5, shape="rectangular", replace=True):
        self.y_smoothed = myf.smooth(self.y, box_pts, shape=shape)

        self.smoothed = tableXY(self.x, self.y_smoothed, self.xerr, self.yerr)

        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = self.y_smoothed

    def diff(self, replace=True):
        diff = np.diff(self.y) / np.diff(self.x)
        new = tableXY(self.x[0:-1] + np.diff(self.x) / 2, diff)
        new.interpolate(new_grid=self.x, replace=True)

        self.deri = tableXY(self.x, new.y, self.xerr, self.yerr)

        if replace:
            self.y_backup = self.y
            self.y = new.y

    def substract_polyfit(self, deg, replace=False, Draw=False):
        model = None
        self.replace_nan()

        if sum(self.yerr):
            w = 1 / (self.yerr + np.max(self.yerr[self.yerr != 0]) * 100) ** 2
            if len(np.unique(self.yerr)) == 1:
                w = np.ones(len(self.yerr))
        else:
            if np.nanstd(self.y):
                w = 1 / (np.nanstd(self.y) / 2 + self.y * 0) ** 2
            else:
                model = 1 + self.y * 0

        if model is None:
            self.poly_coefficient = np.polyfit(self.x, self.y, deg, w=np.sqrt(w))
            model = np.polyval(self.poly_coefficient, self.x)
        sub_model = self.y - model
        self.detrend_poly = tableXY(self.x, sub_model, self.xerr, self.yerr)

        if Draw == True:
            plt.plot(self.x, model)
        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = sub_model
        else:
            self.sub_model = sub_model
        self.poly_curve = model
        self.model = model

    def rolling(self, window=1, quantile=None, median=True, iq=True):
        if median:
            self.roll_median = np.ravel(
                pd.DataFrame(self.y).rolling(window, min_periods=1, center=True).quantile(0.50)
            )
        if iq:
            self.roll_Q1 = np.ravel(
                pd.DataFrame(self.y).rolling(window, min_periods=1, center=True).quantile(0.25)
            )
            self.roll_Q3 = np.ravel(
                pd.DataFrame(self.y).rolling(window, min_periods=1, center=True).quantile(0.75)
            )
            self.roll_IQ = self.roll_Q3 - self.roll_Q1
        if quantile is not None:
            self.roll = np.ravel(
                pd.DataFrame(self.y).rolling(window, min_periods=1, center=True).quantile(quantile)
            )

    def fit_poly(self, Draw=False, d=2, color="r", cov=True):
        if np.sum(self.yerr) != 0:
            weights = self.yerr
        else:
            weights = np.ones(len(self.x))
        if cov:
            coeff, V = np.polyfit(self.x, self.y, d, w=1 / weights, cov=cov)
            self.cov = V
            self.err = np.sqrt(np.diag(V))
        else:
            coeff = np.polyfit(self.x, self.y, d, w=1 / weights, cov=cov)
        self.poly_coefficient = coeff
        self.chi2 = np.sum((self.y - np.polyval(coeff, self.x)) ** 2) / np.sum(self.yerr**2)
        self.bic = self.chi2 + (d + 1) * np.log(len(self.x))
        if Draw == True:
            new_x = np.linspace(self.x.min(), self.x.max(), 10000)
            plt.plot(
                new_x,
                np.polyval(coeff, new_x),
                linestyle="-.",
                color=color,
                linewidth=1,
            )
            # uncertainty = np.sqrt(np.sum([(err[j]*new_x**j)**2 for j in range(len(err))],axis=0))
            # plt.fill_between(new_x,np.polyval(coeff, new_x)-uncertainty/2,np.polyval(coeff, new_x)+uncertainty/2,alpha=0.4,color=color)

    def fit_sinus(self, Draw=False, d=0, guess=[0, 1, 0, 0, 0, 0], p_max=500, report=False, c="r"):
        gmodel = Model(myf.sinus)
        fit_params = Parameters()
        fit_params.add("amp", value=guess[0])
        fit_params.add("period", value=guess[1], min=0, max=p_max)
        fit_params.add("phase", value=guess[2], min=-0.1, max=np.pi + 0.1)
        fit_params.add("c", value=guess[3])
        fit_params.add("a", value=guess[4])
        fit_params["a"].vary = False
        fit_params.add("b", value=guess[5])
        fit_params["b"].vary = False
        if d == 1:
            fit_params["b"].vary = True
        if d > 1:
            fit_params["b"].vary = True
            fit_params["a"].vary = True

        result2 = gmodel.fit(self.y, fit_params, x=self.x)

        if Draw:
            newx = np.linspace(self.x.min(), self.x.max(), 1000)
            y_fit = gmodel.eval(result2.params, x=newx)
            plt.plot(newx, y_fit, color=c)

        if report:
            print(result2.fit_report())
        self.sinus_fitted = gmodel.eval(result2.params, x=self.x)

        self.lmfit = result2
        self.params = result2.params

    def rm_outliers(self, who="Y", m=2, kind="inter", bin_length=0, replace=True):
        vec = self.copy()
        if bin_length:
            self.night_stack(bin_length=bin_length, replace=False)
            vec_binned = self.stacked.copy()
        else:
            vec_binned = self.copy()

        if who == "Xerr":
            mask = rm_out(self.xerr, m=m, kind=kind)[0]
            vec_binned = vec_binned.xerr
            vec = vec.xerr
        if who == "Yerr":
            mask = rm_out(self.yerr, m=m, kind=kind)[0]
            vec_binned = vec_binned.yerr
            vec = vec.yerr
        if who == "Y":
            mask = rm_out(self.y, m=m, kind=kind)[0]
            vec_binned = vec_binned.y
            vec = vec.y
        if who == "X":
            mask = rm_out(self.x, m=m, kind=kind)[0]
            vec_binned = vec_binned.x
            vec = vec.x

        if bin_length:
            outputs = rm_out(vec_binned, m=m, kind=kind, return_borders=True)
            mask = (vec >= outputs[-1]) & (vec <= outputs[-2])

        if who == "both":
            mask = rm_out(self.x, m=m, kind=kind)[0] & rm_out(self.y, m=m, kind=kind)[0]
        self.mask = mask
        if replace:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.yerr = self.yerr[mask]
            self.xerr = self.xerr[mask]

    def fit_gaussian(self, guess=None, Plot=True, color="r", free_offset=True):
        """guess = [amp,cen,width,offset]"""
        if guess is None:
            guess = [-0.5, 0, 3, 1]

        gmodel = Model(myf.gaussian)
        fit_params = Parameters()
        fit_params.add("amp", value=guess[0], min=-1, max=0)
        fit_params.add("cen", value=guess[1])
        fit_params.add("wid", value=guess[2], min=1)
        fit_params.add("offset", value=guess[3], min=0, max=2)
        if not free_offset:
            fit_params["offset"].vary = False
        result1 = gmodel.fit(self.y, fit_params, 1 / self.yerr**2, x=self.x)
        self.lmfit = result1
        self.params = result1.params
        if Plot:
            newx = np.linspace(np.min(self.x), np.max(self.x), 10 * len(self.x))
            plt.plot(newx, gmodel.eval(result1.params, x=newx), color=color)

    def interpolate(
        self,
        new_grid="auto",
        method="cubic",
        replace=True,
        interpolate_x=True,
        fill_value="extrapolate",
        scale="lin",
    ):

        if scale != "lin":
            self.inv()

        if type(new_grid) == str:
            new_grid = np.linspace(self.x.min(), self.x.max(), 10 * len(self.x))
        if type(new_grid) == int:
            new_grid = np.linspace(self.x.min(), self.x.max(), new_grid * len(self.x))

        if np.sum(new_grid != self.x) != 0:
            if replace:
                self.x_backup = self.x.copy()
                self.y_backup = self.y.copy()
                self.xerr_backup = self.xerr.copy()
                self.yerr_backup = self.yerr.copy()
                self.y = interp1d(
                    self.x,
                    self.y,
                    kind=method,
                    bounds_error=False,
                    fill_value=fill_value,
                )(new_grid)
                if np.sum(abs(self.yerr)):
                    self.yerr = interp1d(
                        self.x,
                        self.yerr,
                        kind=method,
                        bounds_error=False,
                        fill_value=fill_value,
                    )(new_grid)
                else:
                    self.yerr = np.zeros(len(new_grid))
                if (interpolate_x) & (bool(np.sum(abs(self.xerr)))):
                    self.xerr = interp1d(
                        self.x,
                        self.xerr,
                        kind=method,
                        bounds_error=False,
                        fill_value=fill_value,
                    )(new_grid)
                else:
                    self.xerr = np.zeros(len(new_grid))
                self.x = new_grid

                if scale != "lin":
                    self.inv()

            else:
                self.y_interp = interp1d(
                    self.x,
                    self.y,
                    kind=method,
                    bounds_error=False,
                    fill_value=fill_value,
                )(new_grid)
                if np.sum(abs(self.yerr)):
                    self.yerr_interp = interp1d(
                        self.x,
                        self.yerr,
                        kind=method,
                        bounds_error=False,
                        fill_value=fill_value,
                    )(new_grid)
                else:
                    self.yerr_interp = np.zeros(len(new_grid))
                if (interpolate_x) & (bool(np.sum(abs(self.xerr)))):
                    self.xerr_interp = interp1d(
                        self.x,
                        self.xerr,
                        kind=method,
                        bounds_error=False,
                        fill_value=fill_value,
                    )(new_grid)
                else:
                    self.xerr_interp = np.zeros(len(new_grid))
                self.x_interp = new_grid
                self.interpolated = tableXY(
                    self.x_interp, self.y_interp, self.xerr_interp, self.yerr_interp
                )

                if scale != "lin":
                    self.interpolated.inv()
                    self.inv()
