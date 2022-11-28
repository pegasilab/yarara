from __future__ import annotations

import glob as glob
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy import float64, int64, ndarray
from pandas.core.frame import DataFrame
from tqdm import tqdm

from ... import iofun, util
from ...analysis import tableXY

if TYPE_CHECKING:
    from .. import spec_time_series


def spectrum(
    self: spec_time_series,
    num: int = 0,
    sub_dico: str = "matching_diff",
    norm: bool = False,
    planet: bool = False,
    color_correction: bool = False,
) -> tableXY:
    """
    Produce a tableXY spectrum by specifying its index number

    Args:
        num : index of the spectrum to extract
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (either linear or cubic)
        norm : True/False button to normalize the spectrum

    Returns:
        The tableXY spectrum object

    """

    if color_correction:
        self.import_material()
        color_corr = np.array(self.material.correction_factor)
    else:
        color_corr = 1

    array = self.import_spectrum(num=num)
    kw = "_planet" * planet

    wave = array["wave"]
    flux, flux_std, conti1, continuum, continuum_std = self.import_sts_flux(
        load=["flux" + kw, "flux_err", "matching_diff", sub_dico, "continuum_err"], num=num
    )
    correction = conti1 / continuum
    spectrum = tableXY(wave, flux * correction * color_corr, flux_std * correction * color_corr)
    if norm:
        flux_norm, flux_norm_std = util.flux_norm_std(flux, flux_std, continuum, continuum_std)
        spectrum_norm = tableXY(wave, flux_norm * color_corr, flux_norm_std * color_corr)
        return spectrum_norm
    else:
        return spectrum


def import_spectrum(self: spec_time_series, num: int = 0) -> Dict[str, Any]:
    """
    Import a pickle file of a spectrum to get fast common information shared by all spectra

    Args:
        num: Index of the spectrum to extract (if None random selection)

    Returns:
        The opened pickle file
    """

    directory = self.directory
    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)
    return pd.read_pickle(files[num])


def yarara_get_bin_length(self):
    bin_length = float(
        "".join(glob.glob(self.dir_root + "WORKSPACE/RASSINE*")[0].split("_B")[1].split("_D")[0])
    )
    return bin_length
