from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pylab as plt
import numpy as np
from numpy.typing import ArrayLike

from ... import iofun
from ...stats import clustering
from ...util import print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_correct_borders_pxl(
    self: spec_time_series,
    pixels_to_reject: ArrayLike = [2, 4095],
    min_shift: int = -30,
    max_shift: int = 30,
) -> None:
    """Produce a brute mask to flag lines crossing pixels according to min-max shift

    Parameters
    ----------
    pixels_to_reject : List of pixels
    min_shift : min shist value in km/s
    max_shift : max shist value in km/s
    """

    print_box("\n---- RECIPE : CREATE PIXELS BORDERS MASK ----\n")

    self.import_material()
    load = self.material

    wave = np.array(load["wave"])
    dwave = np.mean(np.diff(wave))
    pxl = self.yarara_get_pixels()
    orders = self.yarara_get_orders()

    pxl *= orders != 0

    pixels_rejected = np.array(pixels_to_reject)

    pxl[pxl == 0] = np.max(pxl) * 2

    dist = np.zeros(len(pxl)).astype("bool")
    for i in np.arange(np.shape(pxl)[1]):
        dist = dist | (np.min(abs(pxl[:, i] - pixels_rejected[:, np.newaxis]), axis=0) == 0)

    # idx1, dust, dist1 = find_nearest(pixels_rejected,pxl[:,0])
    # idx2, dust, dist2 = find_nearest(pixels_rejected,pxl[:,1])

    # dist = (dist1<=1)|(dist2<=1)

    f = np.where(dist == 1)[0]
    plt.figure()
    for i in np.arange(np.shape(pxl)[1]):
        plt.scatter(pxl[f, i], orders[f, i])

    val, cluster = clustering(dist, 0.5, 1)
    val = np.array([np.product(v) for v in val])
    cluster = cluster[val.astype("bool")]

    left = np.round(wave[cluster[:, 0]] * min_shift / 3e5 / dwave, 0).astype("int")
    right = np.round(wave[cluster[:, 1]] * max_shift / 3e5 / dwave, 0).astype("int")
    # length = right-left+1

    # wave_flagged = wave[f]
    # left = doppler_r(wave_flagged,min_shift*1000)[0]
    # right = doppler_r(wave_flagged,max_shift*1000)[0]

    # idx_left = find_nearest(wave,left)[0]
    # idx_right = find_nearest(wave,right)[0]

    idx_left = cluster[:, 0] + left
    idx_right = cluster[:, 1] + right

    flag_region = np.zeros(len(wave)).astype("int")

    for l, r in zip(idx_left, idx_right):
        flag_region[l : r + 1] = 1

    load["borders_pxl"] = flag_region.astype("int")
    iofun.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))
