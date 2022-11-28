from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from colorama import Fore
from numpy import ndarray
from tqdm import tqdm

from ... import iofun
from ...analysis import tableXY
from ...plots import my_colormesh
from ...stats import IQ, find_nearest, flat_clustering, smooth
from ...util import doppler_r, flux_norm_std, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_poissonian_noise(
    self: spec_time_series,
    noise_wanted: float = 1 / 100,
    wave_ref: None = None,
    flat_snr: bool = True,
    seed: int = 9,
) -> Tuple[ndarray, ndarray]:
    self.import_table()
    self.import_material()

    if noise_wanted:
        master = np.sqrt(
            np.array(self.material.reference_spectrum * self.material.correction_factor)
        )  # used to scale the snr continuum into errors bars
        snrs = pd.read_pickle(self.dir_root + "WORKSPACE/Analyse_snr.p")

        if not flat_snr:
            if wave_ref is None:
                current_snr = np.array(self.table.snr_computed)
            else:
                i = find_nearest(snrs["wave"], wave_ref)[0]
                current_snr = snrs["snr_curve"][:, i]

            curves = current_snr * np.ones(len(master))[:, np.newaxis]
        else:
            curves = snrs["snr_curve"]  # snr curves representing the continuum snr

        snr_wanted = 1 / noise_wanted
        diff = 1 / snr_wanted**2 - 1 / curves**2
        diff[diff < 0] = 0
        # snr_to_degrate = 1/np.sqrt(diff)

        noise = np.sqrt(diff)
        noise[np.isnan(noise)] = 0

        noise_values = noise * master[:, np.newaxis].T
        np.random.seed(seed=seed)
        matrix_noise = np.random.randn(len(self.table.jdb), len(self.material.wave))
        matrix_noise *= noise_values
    else:
        matrix_noise = np.zeros((len(self.table.jdb), len(self.material.wave)))
        noise_values = np.zeros((len(self.table.jdb), len(self.material.wave)))

    return matrix_noise, noise_values
