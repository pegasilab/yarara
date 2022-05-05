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





# =============================================================================
# VISUALISE THE RASSINE TIMESERIES AND ITS CORRELATION WITH A PROXY
# =============================================================================


def yarara_map(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    continuum: str = "linear",
    planet: bool = False,
    modulo: None = None,
    unit: float = 1.0,
    wave_min: None = 4000,
    wave_max: None = 4300,
    time_min: None = None,
    time_max: None = None,
    index: str = "index",
    ratio: bool = False,
    reference: bool = "median",
    berv_shift: bool = False,
    rv_shift: bool = False,
    new: bool = True,
    Plot: bool = True,
    substract_map: List[Any] = [],
    add_map: List[Any] = [],
    correction_factor: bool = True,
    p_noise: float = 1 / np.inf,
) -> Tuple[ndarray, ndarray, ndarray]:

    """
    Display the time-series spectra with proxies and its correlation

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    wave_min : Minimum x axis limit
    wave_max : Maximum x axis limit
    time_min : Minimum y axis limit
    time_max : Maximum y axis limit
    index : 'index' or 'time' if 'time', display the time-series with blank color if no spectra at specific time  (need roughly equidistant time-series spectra)
    zoom : int-type, to improve the resolution of the 2D plot
    smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
    reference : 'median', 'snr' or 'master' to select the reference normalised spectrum usedin the difference
    berv_shift : keyword column to use to move in terrestrial rest-frame, km/s speed
    rv_shift : keyword column to use to shift spectra, m/s speed
    cmap : cmap of the 2D plot
    low_cmap : vmin cmap colorbar
    high_cmap : vmax cmap colorbar

    """

    directory = self.directory
    self.import_material()
    load = self.material
    wave = np.array(load["wave"])
    self.import_table()

    jdb = np.array(self.table["jdb"])
    snr = np.array(self.table["snr"])

    if type(berv_shift) != np.ndarray:
        try:
            berv = np.array(self.table[berv_shift])
        except:
            berv = 0 * jdb
    else:
        berv = berv_shift

    if type(rv_shift) != np.ndarray:
        try:
            rv = np.array(self.table[rv_shift])
        except:
            rv = 0 * jdb
    else:
        rv = rv_shift

    rv = rv - np.median(rv)

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    zoom = self.zoom
    smooth_map = self.smooth_map
    cmap = self.cmap
    low_cmap = self.low_cmap
    high_cmap = self.high_cmap

    files = glob.glob(directory + "RASSI*.p")
    files = np.array(self.table["filename"])  # updated 29.10.21 to allow ins_merged
    files = np.sort(files)

    epsilon = 1e-12

    all_flux, all_flux_err, conti, conti_err = self.import_sts_flux(
        load=["flux" + kw, "flux_err", sub_dico, "continuum_err"]
    )
    flux, err_flux = util.flux_norm_std(all_flux, all_flux_err, conti + epsilon, conti_err)

    if correction_factor:
        flux *= np.array(load["correction_factor"])
        err_flux *= np.array(load["correction_factor"])

    for maps in substract_map:
        flux = self.yarara_substract_map(flux, maps, correction_factor=correction_factor)

    for maps in add_map:
        flux = self.yarara_add_map(flux, maps, correction_factor=correction_factor)

    idx_min = 0
    idx_max = len(wave)
    idx2_min = 0
    idx2_max = len(flux)

    if wave_min is not None:
        idx_min, val, dist = find_nearest(wave, wave_min)
        if val < wave_min:
            idx_min += 1
    if wave_max is not None:
        idx_max, val, dist = find_nearest(wave, wave_max)
        idx_max += 1
        if val > wave_max:
            idx_max -= 1

    if time_min is not None:
        idx2_min = time_min
    if time_max is not None:
        idx2_max = time_max + 1

    if (idx_min == 0) & (idx_max == 0):
        idx_max = find_nearest(wave, np.min(wave) + (wave_max - wave_min))[0]

    noise_matrix, noise_values = self.yarara_poissonian_noise(
        noise_wanted=p_noise, wave_ref=None, flat_snr=True
    )
    flux += noise_matrix
    err_flux = np.sqrt(err_flux**2 + noise_values**2)

    reverse = 1
    if idx_min > idx_max:
        idx_min, idx_max = idx_max, idx_min
        reverse = -1

    old_length = len(wave)
    wave = wave[int(idx_min) : int(idx_max)][::reverse]
    flux = flux[int(idx2_min) : int(idx2_max), int(idx_min) : int(idx_max)]
    err_flux = err_flux[int(idx2_min) : int(idx2_max), int(idx_min) : int(idx_max)]

    snr = snr[int(idx2_min) : int(idx2_max)]
    jdb = jdb[int(idx2_min) : int(idx2_max)]
    berv = berv[int(idx2_min) : int(idx2_max)]

    if np.sum(abs(rv)) != 0:
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, flux[j], 0 * wave)
            test.x = doppler_r(test.x, rv[j])[1]
            test.interpolate(new_grid=wave, method="cubic", replace=True)
            flux[j] = test.y

    if reference == "snr":
        ref = flux[snr.argmax()]
    elif reference == "median":
        ref = np.median(flux, axis=0)
    elif reference == "master":
        ref = (np.array(load["reference_spectrum"]) * np.array(load["correction_factor"]))[
            int(idx_min) : int(idx_max)
        ]
    elif type(reference) == int:
        ref = flux[reference]
    elif type(reference) == np.ndarray:
        ref = reference.copy()
        if len(ref) == old_length:
            ref = ref[int(idx_min) : int(idx_max)]
    else:
        ref = 0 * np.median(flux, axis=0)
        low_cmap = 0
        high_cmap = 1

    if low_cmap is None:
        low_cmap = np.percentile(flux - ref, 2.5)
    if high_cmap is None:
        high_cmap = np.percentile(flux - ref, 97.5)

    if ratio:
        diff = smooth2d(flux / (ref + epsilon), smooth_map)
        low_cmap = 1 - 0.005
        high_cmap = 1 + 0.005
    else:
        diff = smooth2d(flux - ref, smooth_map)

    if modulo is not None:
        diff = self.yarara_map_folded(diff, modulo=modulo, jdb=jdb)[0]

    if np.sum(abs(berv)) != 0:
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, diff[j], 0 * wave)
            test.x = doppler_r(test.x, berv[j] * 1000)[1]
            test.interpolate(new_grid=wave, method="cubic", replace=True)
            diff[j] = test.y

    self.map = (wave, diff)

    if index != "index":
        dtime = np.median(np.diff(jdb))
        liste_time = np.arange(jdb.min(), jdb.max() + dtime, dtime)
        match_time = match_nearest(liste_time, jdb)

        snr2 = np.nan * np.ones(len(liste_time))
        jdb2 = np.nan * np.ones(len(liste_time))
        diff2 = np.median(diff) * np.ones((len(liste_time), len(wave)))

        snr2[match_time[:, 0].astype("int")] = snr[match_time[:, 1].astype("int")]
        jdb2[match_time[:, 0].astype("int")] = jdb[match_time[:, 1].astype("int")]
        diff2[match_time[:, 0].astype("int"), :] = diff[match_time[:, 1].astype("int"), :]

        snr = snr2
        jdb = jdb2
        diff = diff2
    if Plot:
        if new:
            fig = plt.figure(figsize=(24, 6))
            plt.axes([0.1, 0.1, 0.8, 0.8])
        my_colormesh(
            wave,
            np.arange(len(diff)),
            diff * unit,
            zoom=zoom,
            vmin=low_cmap * unit,
            vmax=high_cmap * unit,
            cmap=cmap,
        )
        plt.tick_params(direction="in", top=True, right=True)
        plt.ylabel("Spectra  indexes (time)", fontsize=14)
        plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
        plt.ylim(0, None)
        if new:
            cbaxes = fig.add_axes([0.86 + 0.04, 0.1, 0.01, 0.8])
            ax = plt.colorbar(cax=cbaxes)
            ax.ax.set_ylabel(r"$\Delta$ flux normalised")
    return diff, err_flux, wave
