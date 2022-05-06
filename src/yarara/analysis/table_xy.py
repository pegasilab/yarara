"""
This modules does XXX
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Union, overload

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from statsmodels.stats.weightstats import DescrStatsW

from ..mathfun import gaussian, sinus
from ..stats import local_max, mad
from ..stats import rm_outliers as rm_out
from ..stats import smooth
from ..util import assert_never


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
        x: Optional[ArrayLike],
        y: ArrayLike,
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

    def rms_w(self) -> None:
        if len(self.x) > 1:
            self.rms = DescrStatsW(self.y, weights=1.0 / self.yerr**2).std
            self.weighted_average = DescrStatsW(self.y, weights=1.0 / self.yerr**2).mean
        else:
            self.rms = 0
            self.weighted_average = self.y[0]

    def copy(self) -> tableXY:
        return tableXY(self.x.copy(), self.y.copy(), self.xerr.copy(), self.yerr.copy())

    def switch(self):
        """Switches, in place, the x and y coordinates"""
        self.x, self.y = self.y, self.x
        self.xerr, self.yerr = self.yerr, self.xerr

    def order(self, order: Optional[ArrayLike] = None) -> None:
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
        min: List[Optional[Union[float, int]]] = [None, None],
        max: List[Optional[Union[float, int]]] = [None, None],
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

    def supress_nan(self) -> None:
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

    def supress_mask(self, mask: np.ndarray) -> None:
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.xerr = self.xerr[mask]
        self.yerr = self.yerr[mask]

    def center_symmetrise(
        self, center: np.ndarray, replace: bool = False, Plot: bool = False
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
        label: Optional[str] = "",
        ls: str = "",
        offset: int = 0,
        mask: None = None,
        capsize: int = 3,
        fmt: str = "o",
        markersize: int = 6,
        zorder: int = 1,
        species: Optional[NDArray[np.float64]] = None,
        alpha: int = 1,
        modulo: Optional[float] = None,
        modulo_norm: bool = False,
        cmap: str = "viridis",
        new: bool = False,
        phase_mod: int = 0,
        periodic: bool = False,
        frac: int = 1,
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
        shape: Union[int, Literal["savgol", "rectangular", "gaussian"]] = "rectangular",
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

    def substract_polyfit(self, deg: int, replace: bool = False, Draw: bool = False) -> None:
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
        who: Union[
            Literal["X"], Literal["Xerr"], Literal["Y"], Literal["Yerr"], Literal["both"]
        ] = "Y",
        m: int = 2,
        kind: Union[Literal["inter"], Literal["sigma"], Literal["mad"]] = "inter",
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
        guess: Optional[ArrayLike] = None,
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
        new_grid: Union[np.ndarray, Literal["auto"], int] = "auto",
        method: str = "cubic",
        replace: bool = True,
        interpolate_x: bool = True,
        fill_value: Union[float, str] = "extrapolate",
        scale: Literal["lin"] = "lin",
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
