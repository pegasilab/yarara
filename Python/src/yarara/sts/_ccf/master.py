from __future__ import annotations

import datetime
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from numpy import ndarray
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from ... import iofun, util
from ...analysis import tableXY
from ...iofun import pickle_dump
from ...paths import paths, root
from ...stats import IQ, find_nearest, identify_nearest
from ...util import assert_never, ccf_fun, doppler_r

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_master_ccf(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    name_ext: str = "",
    rvs_: Optional[ndarray] = None,
) -> None:
    self.import_table()

    vrad, ccfs = (self.all_ccf_saved[sub_dico][0], self.all_ccf_saved[sub_dico][1])

    if rvs_ is None:
        rvs: NDArray[np.float64] = self.ccf_rv.y.copy()
    else:
        rvs = rvs_

    med_rv = np.nanmedian(rvs)
    rvs -= med_rv

    new_ccf = []
    for j in range(len(ccfs.T)):
        ccf = tableXY(vrad - rvs[j], ccfs[:, j])
        ccf.interpolate(new_grid=vrad, method="linear", fill_value=np.nan)
        new_ccf.append(ccf.y)
    new_ccf = np.array(new_ccf)
    new_vrad = vrad - med_rv
    stack = np.sum(new_ccf, axis=0)
    stack /= np.nanpercentile(stack, 95)
    half = 0.5 * (1 + np.nanmin(stack))

    master_ccf = tableXY(new_vrad, stack)
    master_ccf.suppress_nan()
    master_ccf.interpolate(replace=True, method="cubic")

    new_vrad = master_ccf.x
    stack = master_ccf.y

    v1 = new_vrad[new_vrad < 0][find_nearest(stack[new_vrad < 0], half)[0][0]]
    v2 = new_vrad[new_vrad > 0][find_nearest(stack[new_vrad > 0], half)[0][0]]

    vmin = np.nanmin(new_vrad[~np.isnan(stack)])
    vmax = np.nanmax(new_vrad[~np.isnan(stack)])

    vlim = np.min([abs(vmin), abs(vmax)])
    vmin = -vlim
    vmax = vlim

    contrast = 1 - np.nanmin(stack)

    plt.figure()
    plt.plot(new_vrad, stack, color="k", label="Contrast = %.1f %%" % (100 * contrast))

    extension = ["YARARA", "HARPS", "telluric"][int(name_ext != "") + int(name_ext == "_telluric")]

    if extension == "YARARA":
        self.fwhm = np.round((v2 - v1) / 1000, 2)
        iofun.pickle_dump(
            {"vrad": new_vrad, "ccf_power": stack},
            open(self.dir_root + "MASTER/MASTER_CCF_KITCAT.p", "wb"),
        )
        try:
            old = pd.read_pickle(self.dir_root + "MASTER/MASTER_CCF_HARPS.p")
            plt.plot(old["vrad"], old["ccf_power"], alpha=0.5, color="k", ls="--")
        except:
            pass
    elif extension == "HARPS":
        iofun.pickle_dump(
            {"vrad": new_vrad, "ccf_power": stack},
            open(self.dir_root + "MASTER/MASTER_CCF_HARPS.p", "wb"),
        )
    elif extension == "telluric":
        try:
            old = pd.read_pickle(self.dir_root + "MASTER/MASTER_CCF" + name_ext + ".p")
            plt.plot(old["vrad"], old["ccf_power"], alpha=0.5, color="k", ls="--")
        except:
            pass
        iofun.pickle_dump(
            {"vrad": new_vrad, "ccf_power": stack},
            open(self.dir_root + "MASTER/MASTER_CCF" + name_ext + ".p", "wb"),
        )
    else:
        raise ValueError("Cannot happen")

    plt.xlim(vmin, vmax)

    plt.plot(
        [v1, v2],
        [half, half],
        color="r",
        label="FHWM = %.2f kms" % ((v2 - v1) / 1000),
    )
    plt.scatter([v1, v2], [half, half], color="r", edgecolor="k", zorder=10)
    plt.scatter([0], [np.nanmin(stack)], color="k", edgecolor="k", zorder=10)
    plt.axvline(x=0, ls=":", color="k", alpha=0.5)
    plt.legend()
    plt.grid()
    plt.xlabel("RV [m/s]", fontsize=13)
    plt.ylabel("Flux normalised", fontsize=13)
    plt.title("%s" % (self.starname), fontsize=14)

    self.yarara_star_info(Contrast=[extension, np.round(contrast, 3)])
    self.yarara_star_info(Fwhm=[extension, np.round((v2 - v1) / 1000, 2)])

    plt.savefig(self.dir_root + "IMAGES/MASTER_CCF" + name_ext + ".pdf")
