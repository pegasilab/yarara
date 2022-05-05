from __future__ import annotations

import datetime
import glob as glob
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from colorama import Fore
from numpy import ndarray
from scipy.interpolate import interp1d
from tqdm import tqdm

from yarara.analysis import tableXY

from ... import io, util
from ...analysis import table, tableXY
from ...paths import root
from ...plots import auto_axis, my_colormesh
from ...stats import IQ, find_nearest, identify_nearest, match_nearest, smooth2d
from ...util import ccf as ccf_fun
from ...util import doppler_r, get_phase, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_retropropagation_correction(
    self: spec_time_series,
    correction_map: str = "matching_smooth",
    sub_dico: str = "matching_cosmics",
    continuum: str = "linear",
) -> None:

    # we introduce the continuum correction (post-processing of rassine normalisation) inside the cosmics correction
    # it allow to not lose the output product of rassine (no need of backup)
    # allow to rerun the code iteratively from the beginning
    # do not use matching_diff + substract_map['cosmics','smooth'] simultaneously otherwise 2 times corrections
    # rurunning the cosmics recipes will kill this correction, therefore a sphinx warning is included in the recipes
    # when a loop is rerun (beginning at fourier correction or water correction), make sure to finish completely the loop

    print_box("\n---- RECIPE : RETROPROPAGATION CORRECTION MAP ----\n")

    directory = self.directory

    planet = self.planet

    self.import_material()
    self.import_table()
    file_test = self.import_spectrum()

    try:
        hl = file_test["parameters"]["hole_left"]
    except:
        hl = None
    try:
        hr = file_test["parameters"]["hole_right"]
    except:
        hr = None

    wave = np.array(self.material["wave"])
    if hl is not None:
        i1 = int(find_nearest(wave, hl)[0])
        i2 = int(find_nearest(wave, hr)[0])

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    correction_retro = pd.read_pickle(
        self.dir_root + "CORRECTION_MAP/map_" + correction_map + ".p"
    )["correction_map"]

    m = pd.read_pickle(
        self.dir_root + "CORRECTION_MAP/map_" + sub_dico + ".p"
    )  # allow the iterative process to be run
    m["correction_map"] += correction_retro
    io.pickle_dump(m, open(self.dir_root + "CORRECTION_MAP/map_" + sub_dico + ".p", "wb"))

    print("\nComputation of the new continua, wait ... \n")
    time.sleep(0.5)
    count_file = -1
    for j in tqdm(files):
        count_file += 1
        file = pd.read_pickle(j)
        conti = file["matching_cosmics"]["continuum_" + continuum]
        flux = file["flux" + kw]

        flux_norm_corrected = flux / conti - correction_retro[count_file]
        new_conti = flux / (flux_norm_corrected + epsilon)  # flux/flux_norm_corrected

        new_continuum = new_conti.copy()
        new_continuum[flux == 0] = conti[flux == 0]
        new_continuum[new_continuum != new_continuum] = conti[
            new_continuum != new_continuum
        ]  # to supress mystic nan appearing
        new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]
        new_continuum[new_continuum == 0] = conti[new_continuum == 0]
        if hl is not None:
            new_continuum[i1:i2] = conti[i1:i2]

        file[sub_dico]["continuum_" + continuum] = new_continuum
        io.save_pickle(j, file)
