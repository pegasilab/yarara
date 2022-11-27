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

from ... import iofun, util
from ...analysis import table, tableXY
from ...paths import root
from ...plots import auto_axis, my_colormesh
from ...stats import IQ, find_nearest, identify_nearest, match_nearest, smooth2d
from ...util import ccf_fun, doppler_r, get_phase, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_retropropagation_correction(
    self: spec_time_series,
    correction_map: str = "matching_smooth",
    sub_dico: str = "matching_cosmics",
) -> None:

    # we introduce the continuum correction (post-processing of rassine normalisation) inside the cosmics correction
    # it allow to not lose the output product of rassine (no need of backup)
    # allow to rerun the code iteratively from the beginning
    # do not use matching_diff + substract_map['cosmics','smooth'] simultaneously otherwise 2 times corrections
    # rurunning the cosmics recipes will kill this correction, therefore a sphinx warning is included in the recipes
    # when a loop is rerun (beginning at fourier correction or water correction), make sure to finish completely the loop

    logging.info("RECIPE : RETROPROPAGATION CORRECTION MAP")

    directory = self.directory

    planet = self.planet

    self.import_material()
    self.import_table()

    file_test = self.import_spectrum()
    hl: float = file_test["parameters"]["hole_left"]
    hr: float = file_test["parameters"]["hole_right"]
    wave = np.array(self.material["wave"])

    i1 = int(find_nearest(wave, hl)[0])
    i2 = int(find_nearest(wave, hr)[0])

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        logging.info("PLANET ACTIVATED")

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    correction_retro = np.load(self.dir_root + "CORRECTION_MAP/map_" + correction_map + ".npy")
    # correction_retro = np.load(self.dir_root+'CORRECTION_MAP/map_'+correction_map+'.npy')

    m = np.load(
        self.dir_root + "CORRECTION_MAP/map_" + sub_dico + ".npy"
    )  # allow the iterative process to be run
    m += correction_retro
    # myf.pickle_dump(m,open(self.dir_root+'CORRECTION_MAP/map_'+sub_dico+'.p','wb'))
    np.save(self.dir_root + "CORRECTION_MAP/map_" + sub_dico + ".npy", m)

    logging.info("Computation of the new continua, wait ...")

    flux, conti = self.import_sts_flux(load=["flux" + kw, sub_dico])

    flux_norm_corrected = flux / conti - correction_retro
    new_conti = flux / (flux_norm_corrected + epsilon)  # flux/flux_norm_corrected

    new_continuum = new_conti.copy()
    new_continuum[flux == 0] = conti[flux == 0]
    new_continuum[new_continuum != new_continuum] = conti[
        new_continuum != new_continuum
    ]  # to suppress mystic nan appearing
    new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]
    new_continuum[new_continuum == 0] = conti[new_continuum == 0]
    if hl is not None:
        new_continuum[i1:i2] = conti[i1:i2]

    fname = self.dir_root + "WORKSPACE/CONTINUUM/Continuum_%s.npy" % (sub_dico)
    np.save(fname, new_continuum.astype("float32"))
