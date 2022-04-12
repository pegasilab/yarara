from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pylab as plt
import numpy as np
from numpy import float64, ndarray

from ..analysis import tableXY
from ..util import map_rnr

if TYPE_CHECKING:
    from ..my_rassine_tools import spec_time_series


# =============================================================================
# EXTRACT BERV AT A SPECIFIC TIME
# =============================================================================


def yarara_get_berv_value(
    self: spec_time_series,
    time_value: float,
    Draw: bool = False,
    new: bool = True,
    light_graphic: bool = False,
    save_fig: bool = True,
) -> float64:
    """Return the berv value for a given jdb date"""

    self.import_table()
    tab = self.table
    berv = tableXY(tab["jdb"], tab["berv"], tab["berv"] * 0 + 0.3)
    berv.fit_sinus(guess=[30, 365.25, 0, 0, 0, 0], Draw=False)
    amp = berv.lmfit.params["amp"].value
    period = berv.lmfit.params["period"].value
    phase = berv.lmfit.params["phase"].value
    offset = berv.lmfit.params["c"].value

    berv_value = amp * np.sin(2 * np.pi * time_value / period + phase) + offset
    if Draw == True:
        t = np.linspace(0, period, 365)
        b = amp * np.sin(2 * np.pi * t / period + phase) + offset

        if new:
            plt.figure(figsize=(8.5, 7))
        plt.title(
            "BERV min : %.1f | BERV max : %.1f | BERV mean : %.1f"
            % (np.min(berv.y), np.max(berv.y), np.mean(berv.y)),
            fontsize=13,
        )
        berv.plot(modulo=period)
        plt.plot(t, b, color="k")
        if not light_graphic:
            plt.axvline(x=time_value % period, color="gray")
            plt.axhline(
                y=berv_value,
                color="gray",
                label="BERV = %.1f [km/s]" % (berv_value),
            )
            plt.axhline(y=0, ls=":", color="k")
            plt.legend()
        plt.xlabel("Time %% %.2f" % (period), fontsize=16)
        plt.ylabel("BERV [km/s]", fontsize=16)
        if save_fig:
            plt.savefig(self.dir_root + "IMAGES/berv_values_summary.pdf")
    return berv_value


def yarara_get_orders(self: spec_time_series) -> ndarray:
    self.import_material()
    mat = self.material
    orders = np.array(mat["orders_rnr"])
    orders = map_rnr(orders)
    orders = np.round(orders, 0)
    self.orders = orders
    return orders


# extract
def yarara_get_pixels(self: spec_time_series) -> ndarray:
    self.import_material()
    mat = self.material
    pixels = np.array(mat["pixels_rnr"])
    pixels = map_rnr(pixels)
    pixels = np.round(pixels, 0)
    self.pixels = pixels
    return pixels
