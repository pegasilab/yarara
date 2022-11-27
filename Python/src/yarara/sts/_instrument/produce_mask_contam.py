from __future__ import annotations

import glob as glob
import time
from typing import TYPE_CHECKING, List, Optional

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from colorama import Fore
from numpy import ndarray
from tqdm import tqdm

from ... import iofun
from ...analysis import table, tableXY
from ...paths import root
from ...plots import my_colormesh, plot_color_box
from ...stats import (
    clustering,
    find_nearest,
    flat_clustering,
    match_nearest,
    merge_borders,
    smooth2d,
)
from ...util import doppler_r, flux_norm_std, print_box, sphinx

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_produce_mask_contam(
    self: spec_time_series, frog_file: str = root + "/Python/Material/Contam_HARPN.p"
) -> None:
    """
    Creation of the stitching mask on the spectrum

    Parameters
    ----------
    frog_file : files containing the wavelength of the stitching
    """

    directory = self.directory
    kw = "_planet" * self.planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    print_box("\n---- RECIPE : PRODUCTION CONTAM MASK ----\n")

    print("\n [INFO] FROG file used : %s" % (frog_file))
    self.import_table()
    self.import_material()
    load = self.material

    grid = np.array(load["wave"])

    # extract frog table
    frog_table = pd.read_pickle(frog_file)
    # stitching

    print("\n [INFO] Producing the contam mask...")

    wave_contam = np.hstack(frog_table["wave"])
    contam = np.hstack(frog_table["contam"])

    vec = tableXY(wave_contam, contam)
    vec.order()
    vec.interpolate(new_grid=np.array(load["wave"]), method="linear")

    load["contam"] = vec.y.astype("int")
    iofun.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))
