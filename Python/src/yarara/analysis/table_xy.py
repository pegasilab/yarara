"""
This modules does XXX
"""
from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional, Union, overload

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from numpy import float64, ndarray
from numpy.typing import ArrayLike, NDArray
from pandas.core.series import Series
from scipy.interpolate import interp1d
from statsmodels.stats.weightstats import DescrStatsW

from ..mathfun import gaussian, sinus
from ..stats import find_nearest, identify_nearest, local_max, mad
from ..stats import rm_outliers as rm_out
from ..stats import smooth
from ..stats.nearest import match_nearest
from ..util import assert_never, ccf_fun, doppler_r


class tableXY(object):
    """
    Describes a scatter plot (x, y)
    """

    @overload
    def __init__(self, x: Optional[ArrayLike], y: ArrayLike, /) -> None:
        pass

    @overload
    def __init__(self, x: Optional[ArrayLike], y: ArrayLike, yerr: ArrayLike, /) -> None:
        pass

    @overload
    def __init__(
        self, x: Optional[ArrayLike], y: ArrayLike, xerr: ArrayLike, yerr: ArrayLike, /
    ) -> None:
        pass

    # TODO: check how the Sphinx documentation shows this constructor

    def __init__(
        self,
        x: Union[List[float64], Series, ndarray],
        y: Union[List[float64], Series, List[Union[float64, float]], ndarray],
        *errs: ArrayLike,
    ) -> None:
        """Creates a tableXY

        The constructor takes additional error vectors.

        - If no error vector is provided, the y error is estimated using the median absolute deviation
        - If one error vector is provided, it is attributed to the y error
        - If two error vectors are provided, they correspond to the x and y errors respectively

        Args:
            x: X values, can be omitted by providing None (will be replaced by a 0 to (n-1) range instead)
            y: Y values, must be provided
            errs: See description

        Raises:
            ArgumentError: If more that two error vectors are provided
        """
        y = np.array(y).astype(np.float64)  # vector of y

        if x is None:  # for a fast table initialisation
            x = np.arange(len(y))
        x = np.array(x).astype(np.float64)  # vector of x

        assert len(x) == len(y), "X and Y must have the same length"
        n = len(x)

        if len(errs) == 0:
            xerr = np.zeros(n)
            if sum(~np.isnan(y)):
                yerr = np.ones(n) * mad(rm_out(y, m=2, kind="sigma")[1])
                if not np.sum(abs(yerr)):
                    yerr = np.ones(n)
            else:
                yerr = np.ones(n)
        elif len(errs) == 1:
            xerr = np.zeros(n)
            yerr = np.array(errs[0])
        elif len(errs) == 2:
            xerr = np.array(errs[0])
            yerr = np.array(errs[1])
        else:
            raise ValueError("Maximum two errors arguments")

        self.x: NDArray[np.float64] = x
        self.y: NDArray[np.float64] = y
        self.xerr: NDArray[np.float64] = xerr
        self.yerr: NDArray[np.float64] = yerr
        self.rms: float = None  # type: ignore
        self.weighted_average: float = None  # type: ignore
        self.clip_mask: Any = None
        self.clipx: Any = None
        self.clipy: Any = None
        self.clipxerr: Any = None
        self.clipyerr: Any = None
        self.y_sym: Any = None
        self.index_max: Any = None
        self.x_max: Any = None
        self.max_extremum: Any = None
        self.index_min: Any = None
        self.y_min: Any = None
        self.x_min: Any = None
        self.min_extremum: Any = None
        self.y_smoothed: Any = None
        self.x_backup: Any = None
        self.y_backup: Any = None
        self.xerr_backup: Any = None
        self.yerr_backup: Any = None

    def match_x(self, table_xy, replace=False):
        match = match_nearest(self.x, table_xy.x)[:, 0:2].astype("int")
        if replace:
            self.x, self.y, self.xerr, self.yerr = (
                self.x[match[:, 0]],
                self.y[match[:, 0]],
                self.xerr[match[:, 0]],
                self.yerr[match[:, 0]],
            )
            table_xy.x, table_xy.y, table_xy.xerr, table_xy.yerr = (
                table_xy.x[match[:, 1]],
                table_xy.y[match[:, 1]],
                table_xy.xerr[match[:, 1]],
                table_xy.yerr[match[:, 1]],
            )
        else:
            v1 = tableXY(
                self.x[match[:, 0]],
                self.y[match[:, 0]],
                self.xerr[match[:, 0]],
                self.yerr[match[:, 0]],
            )
            v2 = tableXY(
                table_xy.x[match[:, 1]],
                table_xy.y[match[:, 1]],
                table_xy.xerr[match[:, 1]],
                table_xy.yerr[match[:, 1]],
            )
            return v1, v2

    def myscatter(
        self,
        num: bool = True,
        liste: Optional[List[int]] = None,
        factor: int = 30,
        color: str = "b",
        alpha: int = 1,
        x_offset: int = 0,
        y_offset: int = 0,
        color_text: str = "k",
        modulo: None = None,
    ) -> None:
        n = np.arange(len(self.x)).astype("str")
        if modulo is not None:
            newx = self.x % modulo
        else:
            newx = self.x
        plt.scatter(np.array(newx), np.array(self.y), color=color, alpha=alpha)
        ax = plt.gca()
        dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) / factor
        dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) / factor
        if num:
            for i, txt in enumerate(n):
                plt.annotate(
                    txt,
                    (np.array(newx)[i] + x_offset, np.array(self.y)[i] + y_offset),
                    color=color_text,
                )
        if liste is not None:
            for i, txt in enumerate(liste):
                plt.annotate(
                    txt,
                    (np.array(newx)[i] + dx + x_offset, np.array(self.y)[i] + dy + y_offset),
                    color=color_text,
                )

    def recenter(self, who: Literal["X", "Y", "x", "y", "both"] = "both", weight=False):
        if (who == "X") | (who == "both") | (who == "x"):
            self.xmean = np.nanmean(self.x)
            self.x = self.x - np.nanmean(self.x)
        if (who == "Y") | (who == "both") | (who == "y"):
            self.ymean = np.nanmean(self.y)
            self.y = self.y - np.nanmean(self.y)

    def species_recenter(self, species: ndarray, ref: None = None, replace: bool = True) -> None:

        spe = np.unique(species)
        shift = np.zeros(len(self.y))

        if (len(spe) > 1) & (len(species) == len(self.y)):
            val_median = np.array([np.nanmedian(self.y[np.where(species == s)[0]]) for s in spe])

            if ref is None:
                ref = 0
            else:
                ref = val_median[ref]

            val_median -= ref

            for k, s in enumerate(spe):
                shift[species == s] += val_median[k]
        else:
            shift += np.nanmedian(self.y)

        newy = self.y - shift

        self.species = species
        if replace == True:
            self.y = newy
        else:
            self.species_recentered = tableXY(self.x, newy, self.xerr, self.yerr)

    def night_stack(self, db: int = 0, bin_length: int = 1, replace: bool = False) -> None:

        jdb = self.x
        vrad = self.y
        vrad_std = self.yerr.copy()

        if not np.sum(vrad_std):  # to avoid null vector
            vrad_std += 1

        vrad_std[vrad_std == 0] = np.nanmax(vrad_std[vrad_std != 0] * 10)

        weights = 1 / (vrad_std) ** 2

        if bin_length:
            groups = ((jdb - db) // bin_length).astype("int")
            groups -= groups[0]
            group = np.unique(groups)
        else:
            group = np.arange(len(jdb))
            groups = np.arange(len(jdb))

        mean_jdb = []
        mean_vrad = []
        mean_svrad = []

        for j in group:
            g = np.where(groups == j)[0]
            mean_jdb.append(np.sum(jdb[g] * weights[g]) / np.sum(weights[g]))
            mean_svrad.append(1 / np.sqrt(np.sum(weights[g])))
            mean_vrad.append(np.sum(vrad[g] * weights[g]) / np.sum(weights[g]))

        mean_jdb = np.array(mean_jdb)
        mean_vrad = np.array(mean_vrad)
        mean_svrad = np.array(mean_svrad)

        if replace:
            self.x, self.y, self.xerr, self.yerr = (
                mean_jdb,
                mean_vrad,
                0.0 * mean_svrad,
                mean_svrad,
            )
        else:
            self.stacked = tableXY(mean_jdb, mean_vrad, mean_svrad)

    def rms_w(self) -> None:
        if len(self.x) > 1:
            self.rms = DescrStatsW(self.y, weights=1.0 / self.yerr**2).std
            self.weighted_average = DescrStatsW(self.y, weights=1.0 / self.yerr**2).mean
        else:
            self.rms = 0
            self.weighted_average = self.y[0]

    def copy(self) -> "tableXY":
        return tableXY(self.x.copy(), self.y.copy(), self.xerr.copy(), self.yerr.copy())

    def switch(self):
        """Switches, in place, the x and y coordinates"""
        self.x, self.y = self.y, self.x
        self.xerr, self.yerr = self.yerr, self.xerr

    def order(self, order: None = None) -> None:
        if order is None:
            order = self.x.argsort()
        else:
            order = np.array(order).astype(np.int64)
        self.order_liste = order
        self.x = self.x[order]
        self.y = self.y[order]
        self.xerr = self.xerr[order]
        self.yerr = self.yerr[order]

    def null(self) -> None:
        self.yerr = 0 * self.yerr

    def clip(
        self,
        min: List[Optional[Union[float64, int, float]]] = [None, None],
        max: List[Optional[Union[float64, int, float]]] = [None, None],
        replace: bool = True,
        invers: bool = False,
    ) -> None:
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

    def suppress_nan(self) -> None:
        mask = ~np.isnan(self.x) & ~np.isnan(self.y) & ~np.isnan(self.yerr) & ~np.isnan(self.xerr)
        if sum(~mask) == len(mask):
            self.replace_nan()
        else:
            self.mask_not_nan = mask
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.xerr = self.xerr[mask]
            self.yerr = self.yerr[mask]

    def replace_nan(self) -> None:
        self.y[np.isnan(self.y)] = np.random.randn(sum(np.isnan(self.y)))
        self.x[np.isnan(self.x)] = np.random.randn(sum(np.isnan(self.x)))
        self.yerr[np.isnan(self.yerr)] = np.random.randn(sum(np.isnan(self.yerr)))
        self.xerr[np.isnan(self.xerr)] = np.random.randn(sum(np.isnan(self.xerr)))

    def suppress_mask(self, mask: ndarray) -> None:
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.xerr = self.xerr[mask]
        self.yerr = self.yerr[mask]

    def center_symmetrise(
        self, center: ndarray, replace: bool = False, Plot: bool = False
    ) -> None:
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
        Show: bool = False,
        color: str = "k",
        label: str = "",
        ls: str = "",
        offset: int = 0,
        mask: None = None,
        capsize: int = 3,
        fmt: str = "o",
        markersize: int = 6,
        zorder: int = 1,
        species: Optional[ndarray] = None,
        alpha: float = 1.0,
        modulo: Optional[Union[float, float64]] = None,
        modulo_norm: bool = False,
        cmap: str = "viridis",
        new: bool = False,
        phase_mod: int = 0,  # TODO: really int?
        periodic: bool = False,
        frac: float = 1.0,
        yerr: bool = True,
    ) -> None:

        """For the mask give either the first and last index in a list [a,b] or the mask boolean"""

        if (len(self.x) > 10000) & (ls == "") & (modulo is None):
            ls = "-"

        if species is None:
            species = np.ones(len(self.x))

        prop_cycle = plt.rcParams["axes.prop_cycle"]  # type: ignore
        colors_species = [color] + prop_cycle.by_key()["color"]  # type: ignore

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
                        marker="o",  # type: ignore
                        c=self.x[mask2][loc][sel],
                        cmap=cmap,  # type: ignore
                        s=markersize,
                        zorder=zorder * 100,
                    )
                    if periodic:
                        for i in range(int(float(periodic))):
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

    def find_max(self, vicinity: int = 3) -> None:
        self.index_max, self.y_max = local_max(self.y, vicinity=vicinity)
        self.index_max = self.index_max.astype("int")
        self.x_max = self.x[self.index_max.astype("int")]
        self.max_extremum = tableXY(self.x_max, self.y_max)

    def find_min(self, vicinity: int = 3) -> None:
        self.index_min, self.y_min = local_max(-self.y, vicinity=vicinity)
        self.index_min = self.index_min.astype("int")
        self.y_min *= -1
        self.x_min = self.x[self.index_min.astype("int")]
        self.min_extremum = tableXY(self.x_min, self.y_min)

    def smooth(
        self,
        box_pts: int = 5,
        shape: str = "rectangular",
        replace: bool = True,
    ) -> None:
        self.y_smoothed = smooth(self.y, box_pts, shape=shape)

        self.smoothed = tableXY(self.x, self.y_smoothed, self.xerr, self.yerr)

        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = self.y_smoothed

    def diff(self, replace: bool = True) -> None:
        diff = np.diff(self.y) / np.diff(self.x)
        new = tableXY(self.x[0:-1] + np.diff(self.x) / 2, diff)
        new.interpolate(new_grid=self.x, replace=True)

        self.deri = tableXY(self.x, new.y, self.xerr, self.yerr)

        if replace:
            self.y_backup = self.y
            self.y = new.y

    def substract_polyfit(
        self, deg: Union[float64, int], replace: bool = False, Draw: bool = False
    ) -> None:
        model = None
        self.replace_nan()

        w: np.ndarray = np.array([])
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

    def rolling(
        self, window: int = 1, quantile: Optional[int] = None, median: bool = True, iq: bool = True
    ) -> None:
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
            self.roll_IQ = self.roll_Q3 - self.roll_Q1  # type: ignore
        if quantile is not None:
            self.roll = np.ravel(
                pd.DataFrame(self.y).rolling(window, min_periods=1, center=True).quantile(quantile)
            )

    def fit_poly(self, Draw: bool = False, d: int = 2, color: str = "r", cov: bool = True) -> None:
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

    def fit_sinus(
        self,
        Draw: bool = False,
        d: int = 0,
        guess: List[Union[int, float]] = [0, 1, 0, 0, 0, 0],
        p_max: int = 500,
        report: bool = False,
        c: str = "r",
    ) -> None:
        gmodel = Model(sinus)
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

    def rm_outliers(
        self,
        who: str = "Y",
        m: int = 2,
        kind: str = "inter",
        replace: bool = True,
    ) -> None:

        # TODO: I've removed the binning thing
        if who == "Xerr":
            mask = rm_out(self.xerr, m=m, kind=kind)[0]
        elif who == "Yerr":
            mask = rm_out(self.yerr, m=m, kind=kind)[0]
        elif who == "Y":
            mask = rm_out(self.y, m=m, kind=kind)[0]
        elif who == "X":
            mask = rm_out(self.x, m=m, kind=kind)[0]
        elif who == "both":
            mask = rm_out(self.x, m=m, kind=kind)[0] & rm_out(self.y, m=m, kind=kind)[0]
        else:
            assert_never(who)
        self.mask = mask
        if replace:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.yerr = self.yerr[mask]
            self.xerr = self.xerr[mask]

    def fit_gaussian(
        self,
        guess: None = None,
        Plot: bool = True,
        color: str = "r",
        free_offset: bool = True,
    ) -> None:
        """guess = [amp,cen,width,offset]"""
        if guess is None:
            guess = [-0.5, 0, 3, 1]
        if not isinstance(guess, np.ndarray):
            guess = np.array(guess)
        gmodel = Model(gaussian)
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

    # TODO: remove the "scale" argument
    def interpolate(
        self,
        new_grid: Union[str, ndarray] = "auto",
        method: str = "cubic",
        replace: bool = True,
        interpolate_x: bool = True,
        fill_value: Union[float, str] = "extrapolate",
        scale: str = "lin",
    ) -> None:
        if isinstance(new_grid, int):
            new_grid = np.linspace(self.x.min(), self.x.max(), new_grid * len(self.x))
        elif isinstance(new_grid, str) and new_grid == "auto":
            new_grid = np.linspace(self.x.min(), self.x.max(), 10 * len(self.x))

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

    def ccf(
        self,
        mask,
        rv_sys=0,
        rv_range=15,
        weighted=True,
        ccf_oversampling=1,
        wave_min=None,
        wave_max=None,
    ):

        if len(np.shape(mask)) < 2:
            mask = np.hstack([mask[:, np.newaxis], np.ones(len(mask))[:, np.newaxis]])

        grid = self.x
        flux = self.y[:, np.newaxis].T
        flux_err = self.yerr[:, np.newaxis].T

        if rv_sys:
            mask[:, 0] = doppler_r(mask[:, 0], rv_sys)[0]

        mask_shifted = doppler_r(mask[:, 0], (rv_range + 5) * 1000)

        mask = mask[
            (doppler_r(mask[:, 0], 30000)[0] < grid.max())
            & (doppler_r(mask[:, 0], 30000)[1] > grid.min()),
            :,
        ]  # suppress line farther than 30kms
        if wave_min is not None:
            mask = mask[mask[:, 0] > wave_min, :]
        if wave_max is not None:
            mask = mask[mask[:, 0] < wave_max, :]

        mask_min = np.min(mask[:, 0])
        mask_max = np.max(mask[:, 0])

        grid_min = int(find_nearest(grid, doppler_r(mask_min, -100000)[0])[0])
        grid_max = int(find_nearest(grid, doppler_r(mask_max, 100000)[0])[0])
        grid = grid[grid_min:grid_max]

        log_grid = np.linspace(np.log10(grid).min(), np.log10(grid).max(), len(grid))
        dgrid = log_grid[1] - log_grid[0]
        # dv = (10**(dgrid)-1)*299.792e6

        used_region = ((10**log_grid) >= mask_shifted[1][:, np.newaxis]) & (
            (10**log_grid) <= mask_shifted[0][:, np.newaxis]
        )
        used_region = (np.sum(used_region, axis=0) != 0).astype("bool")
        logging.info(
            f"Percentage of the spectrum used : {100 * sum(used_region) / len(grid):.1f} [%] \n"
        )

        mask_wave = np.log10(mask[:, 0])
        mask_contrast = mask[:, 1] * weighted + (1 - weighted)

        log_grid_mask = np.arange(
            log_grid.min() - 10 * dgrid, log_grid.max() + 10 * dgrid + dgrid / 10, dgrid / 11
        )
        log_mask = np.zeros(len(log_grid_mask))

        match = identify_nearest(mask_wave, log_grid_mask)
        for j in np.arange(-5, 6, 1):
            log_mask[match + j] = (mask_contrast) ** 2

        all_flux = []
        all_flux.append(
            interp1d(
                np.log10(self.x),
                flux[0],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )(log_grid)
        )
        flux = np.array(all_flux)

        all_flux_err = []
        all_flux_err.append(
            interp1d(
                np.log10(self.x),
                flux_err[0],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(log_grid)
        )
        flux_err = np.array(all_flux_err)

        log_template = interp1d(
            log_grid_mask, log_mask, kind="linear", bounds_error=False, fill_value="extrapolate"
        )(log_grid)

        vrad, ccf_power, ccf_power_std = ccf_fun(
            log_grid[used_region],
            flux[:, used_region],
            log_template[used_region],
            rv_range=rv_range,
            oversampling=ccf_oversampling,
            spec1_std=flux_err[:, used_region],
        )  # to compute on all the ccf simultaneously

        self.ccf_profile = tableXY(vrad, np.ravel(ccf_power))
        self.ccf_profile.yerr /= np.max(self.ccf_profile.y)
        self.ccf_profile.y /= np.max(self.ccf_profile.y)

        ccf_profile = self.ccf_profile

        plt.figure(figsize=(18, 6))
        plt.axes((0.05, 0.1, 0.58, 0.75))
        plt.plot(self.x, self.y, color="k")
        plt.xlim(np.min(mask[:, 0]), np.max(mask[:, 0]))
        for j in mask[:, 0]:
            plt.axvline(x=j, color="b", alpha=0.5)
        plt.axes((0.68, 0.1, 0.3, 0.75))
        plt.scatter(ccf_profile.x, ccf_profile.y, color="k", marker="o")  # type: ignore
        plt.axvline(x=0, ls=":", color="k")
        plt.axhline(y=1, ls=":", color="k")
        plt.ylim(0, 1.1)

        amp = np.percentile(ccf_profile.y, 95) - np.percentile(ccf_profile.y, 5)
        xmin = np.argmin(ccf_profile.y)
        x1 = find_nearest(ccf_profile.y[0:xmin], 1 - amp / 2)[0][0]
        x2 = find_nearest(ccf_profile.y[xmin:], 1 - amp / 2)[0][0]
        width = ccf_profile.x[xmin:][x2] - ccf_profile.x[0:xmin][x1]
        center = ccf_profile.x[xmin]
        guess = [float(-amp), float(center), float(width), 1.0]
        ccf_profile.fit_gaussian(guess=guess, Plot=True)
        plt.axvline(x=ccf_profile.params["cen"].value, color="r", alpha=0.3)

        self.ccf_params = ccf_profile.params

        plt.title(
            "C = %.2f +/- %.2f [%%] \n FWHM = %.2f +/- %.2f [km/s] \n RV = %.2f +/- %.2f [m/s]"
            % (
                100 * (-ccf_profile.params["amp"].value),
                ccf_profile.params["amp"].stderr,
                ccf_profile.params["wid"].value / 1000 * 2.355,
                ccf_profile.params["wid"].stderr / 1000 * 2.355,
                ccf_profile.params["cen"].value,
                ccf_profile.params["cen"].stderr,
            )
        )
