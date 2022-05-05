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

from .. import io, util
from ..analysis import table, tableXY
from ..paths import root
from ..plots import auto_axis, my_colormesh
from ..stats import IQ, find_nearest, identify_nearest, match_nearest, smooth2d
from ..util import ccf as ccf_fun
from ..util import doppler_r, get_phase, print_box

if TYPE_CHECKING:
    from . import spec_time_series


# =============================================================================
# COMPUTE ALL THE TEMPERATURE SENSITIVE RATIO
# =============================================================================


def yarara_activity_index(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    continuum: str = "linear",
    plot: bool = True,
    debug: bool = False,
    calib_std: int = 0,
    optimize: bool = False,
    substract_map: List[Any] = [],
    add_map: List[Any] = [],
    p_noise: float = 1 / np.inf,
    save: bool = True,
) -> None:
    """
    Produce the activity proxy time-series. Need to cancel the RV systemic of the star

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    rv : The RV of the star system
    plot : True/False, Plot the proxies time-series
    debug : True/False, Plot the intermediate graphicwith the spectrum and area extraction for the proxies
    ron : read-out-noise error injected by reading pixels
    calib_std : std error due to flat-field photon noise (5.34e-4 or 10.00e-4 Cretignier+20)
    """

    print_box("\n---- RECIPE : ACTIVITY PROXIES EXTRACTION ----\n")

    directory = self.directory
    rv_sys = self.rv_sys
    self.import_table()

    self.import_material()
    load = self.material
    kw = "_planet" * self.planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")
    time.sleep(1)

    epsilon = 1e-12
    save_kw = save

    try:
        self.import_star_info()
        bv = self.star_info["BV"]["fixed"]
        if np.isnan(bv):
            bv = 0
            print(Fore.YELLOW + " [WARNING] No BV value given for the star" + Fore.RESET)
        # conv_offset = -0.60 - 5.99*bv + 2.51*bv**2 #old calib
        # conv_slope = 4.55 - 7.30*bv + 3.61*bv**2 #old calib

        conv_offset = 1.13 - 10.13 * bv + 4.97 * bv**2  # old calib
        conv_slope = 5.76 - 10.0 * bv + 5.10 * bv**2  # old calib

    except:
        conv_offset = 0
        conv_slope = 1

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    # [center, half-window, hole_size, half-window-continuum,database_kw, subplot]
    Ca2H = [3968.47, 0.45, 0, 1.5, "CaIIH", None]
    Ca2K = [3933.66, 0.45, 0, 1.5, "CaIIK", None]
    Ca1 = [4226.72, 1.50, 0, 1.5, "CaI", 2]
    Mg1a = [5167.32, 0.50, 0, 1.5, "MgIa", 6]
    Mg1b = [5172.68, 0.50, 0, 1.5, "MgIb", 7]
    Mg1c = [5183.60, 0.50, 0, 1.5, "MgIc", 8]
    NaDl = [5889.95, 0.50, 0, 0, "NaD1", 3]
    NaDr = [5895.92, 0.50, 0, 0, "NaD2", 4]
    Ha = [6562.79, 0.35, 0, 0, "Ha", 9]
    Hb = [4861.35, 0.35, 0, 1.5, "Hb", 10]
    Hc = [4340.47, 0.35, 0, 1.5, "Hc", 11]
    Hd = [4101.73, 0.15, 0, 1.5, "Hd", 12]
    Heps = [3889.04, 0.10, 0, 1.5, "Heps", None]
    He1D3 = [5875.62, 0.15, 0, 0, "HeID3", 5]

    all_proxies = [
        Ca2H,
        Ca2K,
        Ca1,
        Mg1a,
        Mg1b,
        Mg1c,
        NaDl,
        NaDr,
        Ha,
        Hb,
        Hc,
        Hd,
        Heps,
        He1D3,
    ]

    # FeXIV = 5302.86 ;

    # calib_std = 5.34e-4 #from Cretignier+20 precision linear + clustering
    # calib_std = 10.00e-4 #from Cretignier+20 precision linear + clustering (new sun) related to flat field SNR

    fluxes = []
    err_fluxes = []
    jdb = []
    snrs = []
    count = 0

    file_random = self.import_spectrum()
    waves = file_random["wave"]
    all_prox_names = np.array(all_proxies)[:, 4]
    proxy_found = ((np.array(all_proxies)[:, 0] - np.nanmin(waves[0])) > 0) & (
        (np.nanmax(waves[0])) > 0
    )
    all_proxies = list(np.array(all_proxies)[proxy_found])

    dgrid = np.mean(np.diff(waves))
    for j in tqdm(files):
        count += 1
        file = pd.read_pickle(j)
        try:
            jdb.append(file["parameters"]["jdb"])
        except:
            jdb.append(count)

        snrs.append(file["parameters"]["SNR_5500"])

        f = file["flux" + kw]
        f_std = file["flux_err"]
        c = file[sub_dico]["continuum_" + continuum]
        c_std = file["continuum_err"]

        f_norm, f_norm_std = util.flux_norm_std(f, f_std, c, c_std)
        dustbin, f_norm_std = util.flux_norm_std(
            f, f_std, file["matching_diff"]["continuum_" + continuum], c_std
        )

        fluxes.append(f_norm)
        err_fluxes.append(f_norm_std)

    waves = np.array(waves)
    flux = np.array(fluxes)
    err_flux = np.array(err_fluxes)
    flux *= np.array(load["correction_factor"])
    err_flux *= np.array(load["correction_factor"])
    snrs = np.array(snrs)

    noise_matrix, noise_values = self.yarara_poissonian_noise(
        noise_wanted=p_noise, wave_ref=None, flat_snr=True
    )
    flux += noise_matrix
    err_flux = np.sqrt(err_flux**2 + noise_values**2)

    # TODO: restore those two methods or remove those arguments

    # for maps in substract_map:
    #     flux = self.yarara_substract_map(flux, maps, correction_factor=True)

    # for maps in add_map:
    #     flux = self.yarara_add_map(flux, maps, correction_factor=True)

    jdb = np.array(jdb)
    ref = snrs.argmax()
    wave_ref = waves
    wave = waves

    flux_ref = np.median(flux, axis=0)
    ratio = flux / (flux_ref + epsilon)
    try:
        mask_ghost = np.array(load["ghost_a"]).astype("bool")
    except:
        mask_ghost = np.zeros(len(flux_ref)).astype("bool")
    ratio[:, mask_ghost] = np.nan

    if rv_sys is None:
        if file_random["parameters"]["RV_sys"] is not None:
            rv_sys = 1000 * file_random["parameters"]["RV_sys"]
        else:
            rv_sys = 0
    else:
        rv_sys *= 1000

    def find_proxy(vec):
        center = doppler_r(vec[0], rv_sys)[0]
        left = doppler_r(vec[0] - vec[1], rv_sys)[0]
        right = doppler_r(vec[0] + vec[1], rv_sys)[0]

        center_idx_proxy = find_nearest(wave, center)[0]
        left_idx_proxy = find_nearest(wave, left)[0]
        right_idx_proxy = find_nearest(wave, right)[0]

        left = doppler_r(vec[0] - vec[2], rv_sys)[0]
        right = doppler_r(vec[0] + vec[2], rv_sys)[0]

        left_idx_hole = find_nearest(wave, left)[0]
        right_idx_hole = find_nearest(wave, right)[0]

        left = doppler_r(vec[0] - vec[3], rv_sys)[0]
        right = doppler_r(vec[0] + vec[3], rv_sys)[0]

        left_idx_cont = find_nearest(wave, left)[0]
        right_idx_cont = find_nearest(wave, right)[0]

        return (
            int(center_idx_proxy),
            int(left_idx_proxy),
            int(right_idx_proxy),
            int(left_idx_hole),
            int(right_idx_hole),
            int(left_idx_cont),
            int(right_idx_cont),
        )

    def extract_proxy(vec, kernel=None):
        c, l, r, l_hole, r_hole, l_cont, r_cont = find_proxy(vec)
        if kernel is None:
            continuum = 1
            if r != l:
                r += 1
            if l_hole != r_hole:
                r_hole += 1
            if l_cont != l:
                r_cont += 1
                continuum = np.hstack([ratio[:, l_cont:l], ratio[:, r:r_cont]])
                continuum = np.nanmedian(continuum, axis=1)
                continuum[np.isnan(continuum)] = 1
            proxy = np.sum(flux[:, l:r], axis=1) - np.sum(flux[:, l_hole:r_hole], axis=1)
            proxy_std = np.sum((err_flux[:, l:r]) ** 2, axis=1) - np.sum(
                (err_flux[:, l_hole:r_hole]) ** 2, axis=1
            )
            proxy_std = np.sqrt(proxy_std)
            norm_proxy = (r - l) - (r_hole - l_hole)

            proxy /= continuum
            proxy_std /= continuum
        else:
            kernel /= np.sum(abs(kernel))
            proxy = np.sum((flux - np.median(flux, axis=0)) * kernel, axis=1)
            proxy_std = np.sum((kernel * err_flux) ** 2, axis=1)
            proxy_std = np.sqrt(proxy_std)
            norm_proxy = 1

        prox = tableXY(jdb, proxy, proxy_std)
        prox.rms_w()
        proxy_rms = prox.rms
        windex = int(1 / dgrid)
        mask_proxy = abs(np.arange(len(flux.T)) - c) < windex
        slope = np.median(
            (flux[:, mask_proxy] - np.mean(flux[:, mask_proxy], axis=0))
            / ((proxy - np.mean(proxy))[:, np.newaxis]),
            axis=0,
        )

        s = tableXY(wave[mask_proxy], slope)
        s.smooth(box_pts=7, shape="savgol")
        s.center_symmetrise(doppler_r(vec[0], rv_sys)[0], replace=True)
        slope = s.y

        t = table(flux[:, mask_proxy] - np.mean(flux[:, mask_proxy], axis=0))
        t.rms_w(1 / err_flux[:, mask_proxy] ** 2, axis=0)
        rslope = np.zeros(len(flux.T))
        rslope[mask_proxy] = slope
        rms = np.ones(len(flux.T))
        rms[mask_proxy] = t.rms

        rcorr = (
            rslope * proxy_rms / (rms + epsilon)
        )  # need good weighting of the proxy and the flux

        if norm_proxy:
            proxy /= norm_proxy
            proxy_std /= norm_proxy
            return proxy, proxy_std, rcorr, rslope
        else:
            return 0 * proxy, 0 * proxy_std, 0 * wave, 0 * wave

    save = {"null": 0}
    mask_activity = np.zeros(len(waves))
    all_rcorr = []
    all_rslope = []
    for p in all_proxies:
        c, l, r, lh, rh, lc, rc = find_proxy(p)
        mask_activity[l:r] = 1
        proxy, proxy_std, rcorr, rslope = extract_proxy(p)
        all_rcorr.append(rcorr)
        all_rslope.append(rslope)
        save[p[4]] = proxy
        save[p[4] + "_std"] = proxy_std
    del save["null"]

    all_rcorr = np.array(all_rcorr)
    all_rslope = np.array(all_rslope)

    for n in all_prox_names:
        if n not in save.keys():
            save[n] = np.zeros(len(jdb))
            save[n + "_std"] = np.zeros(len(jdb))

    save_backup = save.copy()

    if optimize:
        save = {"null": 0}
        mask_activity = np.zeros(len(waves))
        for i, p in enumerate(all_proxies):
            mask_activity[all_rslope[i] != 0] = 1
            proxy, proxy_std, dust, dust = extract_proxy(p, kernel=all_rslope[i])
            save[p[4]] = proxy
            save[p[4] + "_std"] = proxy_std
        del save["null"]

        self.all_kernels = all_rslope

    del ratio

    if debug:
        plt.figure(figsize=(18, 9))
        plt.subplot(3, 1, 1)
        plt.plot(wave_ref, all_rslope.T, color="k")
        plt.axhline(y=0, color="r")
        ax = plt.gca()
        plt.subplot(3, 1, 2, sharex=ax)
        plt.plot(wave_ref, all_rcorr.T, color="k")
        plt.axhline(y=0, color="r")
        plt.ylim(-1, 1)
        plt.subplot(3, 1, 3, sharex=ax)
        plt.plot(wave_ref, flux_ref, color="k")
        for p in all_proxies:
            center = doppler_r(p[0], rv_sys)[0]
            hw = p[1]
            plt.axvspan(xmin=center - hw, xmax=center + hw, alpha=0.5, color="r")
            plt.axvline(x=center, color="r")

    load["activity_proxies"] = mask_activity.astype("int")
    io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    if not np.sum(abs(save["CaIIK"])):
        save["CaIIK"] = save["Ha"] + 0.01
        save["CaIIH"] = save["Ha"] + 0.01
        save["CaIIK_std"] = save["Ha_std"]
        save["CaIIH_std"] = save["Ha_std"]
        save_backup["CaIIK"] = save_backup["Ha"]
        save_backup["CaIIH"] = save_backup["Ha"]
        save_backup["CaIIK_std"] = save_backup["Ha_std"]
        save_backup["CaIIH_std"] = save_backup["Ha_std"]

    save["CaII"] = 0.5 * (save["CaIIK"] + save["CaIIH"])
    save["CaII_std"] = 0.5 * np.sqrt((save["CaIIK_std"]) ** 2 + (save["CaIIH_std"]) ** 2)

    save["NaD"] = 0.5 * (save["NaD1"] + save["NaD2"])
    save["NaD_std"] = 0.5 * np.sqrt((save["NaD1_std"]) ** 2 + (save["NaD2_std"]) ** 2)

    save["MgI"] = 0.5 * (save["MgIa"] + save["MgIb"] + save["MgIc"])
    save["MgI_std"] = 0.5 * np.sqrt(
        (save["MgIa_std"]) ** 2 + (save["MgIb_std"]) ** 2 + (save["MgIc_std"]) ** 2
    )

    save_backup["CaII"] = 0.5 * (save_backup["CaIIK"] + save_backup["CaIIH"])
    save_backup["CaII_std"] = 0.5 * np.sqrt(
        (save_backup["CaIIK_std"]) ** 2 + (save_backup["CaIIH_std"]) ** 2
    )

    shift = np.mean(save["CaII"]) - np.mean(save_backup["CaII"])

    save["RHK"] = conv_slope * np.log10(save["CaII"].copy() - shift) + conv_offset
    save["RHK_std"] = (
        conv_slope
        * (save["CaII_std"].copy() + calib_std)
        / abs(save["CaII"].copy() - shift)
        / np.log(10)
    )

    save_backup["RHK"] = conv_slope * np.log10(save_backup["CaII"].copy()) + conv_offset
    save_backup["RHK_std"] = (
        conv_slope
        * (save_backup["CaII_std"].copy() + calib_std)
        / abs(save_backup["CaII"].copy())
        / np.log(10)
    )

    self.all_proxies = save
    self.all_proxies_name = list(save.keys())[::2]

    all_proxies.append([0, 0, 0, 0, "CaII", None])
    all_proxies.append([0, 0, 0, 0, "NaD", None])
    all_proxies.append([0, 0, 0, 0, "MgI", None])
    all_proxies.append([0, 0, 0, 0, "RHK", 1])

    for j in list(save.keys()):
        if len(j.split("_std")) == 2:
            save[j] += calib_std

    if save_kw:
        print("\n Saving activity proxies...")
        if False:  # no more used 02.07.21
            for i, j in enumerate(files):
                file = pd.read_pickle(j)
                # print('File (%.0f/%.0f) %s SNR %.0f reduced'%(i+1,len(files),j,snrs[i]))
                for p in all_proxies:
                    file["parameters"][p[4]] = save[p[4]][i]
                    file["parameters"][p[4] + "_std"] = save[p[4] + "_std"][i]
                io.pickle_dump(file, open(j, "wb"))

            self.yarara_analyse_summary()
        else:
            self.yarara_obs_info(kw=pd.DataFrame(save))

    self.ca2k = tableXY(jdb, save["CaIIK"], save["CaIIK_std"] + calib_std)
    self.ca2h = tableXY(jdb, save["CaIIH"], save["CaIIH_std"] + calib_std)
    self.ca2 = tableXY(jdb, save["CaII"], save["CaII_std"] + calib_std)
    self.rhk = tableXY(jdb, save["RHK"], save["RHK_std"])
    self.mg1 = tableXY(jdb, save["MgI"], save["MgI_std"] + calib_std)
    self.mga = tableXY(jdb, save["MgIa"], save["MgIa_std"] + calib_std)
    self.mgb = tableXY(jdb, save["MgIb"], save["MgIb_std"] + calib_std)
    self.mgc = tableXY(jdb, save["MgIc"], save["MgIc_std"] + calib_std)
    self.nad = tableXY(jdb, save["NaD"], save["NaD_std"] + calib_std)
    self.nad1 = tableXY(jdb, save["NaD1"], save["NaD1_std"] + calib_std)
    self.nad2 = tableXY(jdb, save["NaD2"], save["NaD2_std"] + calib_std)
    self.ha = tableXY(jdb, save["Ha"], save["Ha_std"] + calib_std)
    self.hb = tableXY(jdb, save["Hb"], save["Hb_std"] + calib_std)
    self.hc = tableXY(jdb, save["Hc"], save["Hc_std"] + calib_std)
    self.hd = tableXY(jdb, save["Hd"], save["Hd_std"] + calib_std)
    self.heps = tableXY(jdb, save["Heps"], save["Heps_std"] + calib_std)
    self.hed3 = tableXY(jdb, save["HeID3"], save["HeID3_std"] + calib_std)
    self.ca1 = tableXY(jdb, save["CaI"], save["CaI_std"] + calib_std)

    self.infos["latest_dico_activity"] = sub_dico

    if plot:
        phase = get_phase(np.array(self.table.jdb), 365.25)
        for name, modulo, phase_mod in zip(["", "_1year"], [None, 365.25], [None, phase]):
            titles = [
                "CaII H&K",
                "CaI",
                "NaD_left",
                "NaD_right",
                "HeID3",
                "MgIa",
                "MgIb",
                "MgIc",
                r"$H_\alpha$",
                r"$H_\beta$",
                r"$H_\gamma$",
                r"$H_\delta$",
            ]
            plt.figure(figsize=(20, 10))
            plt.subplot(3, 4, 1)
            ax = plt.gca()
            for p in all_proxies:
                if p[5] is not None:
                    num = p[5]
                    plt.subplot(3, 4, num, sharex=ax)
                    plt.title(titles[num - 1])

                    vec = tableXY(
                        jdb,
                        save_backup[p[4]],
                        save_backup[p[4] + "_std"] + calib_std,
                    )
                    vec.plot(
                        capsize=0,
                        zorder=1,
                        color=["k", "r"][int(optimize)],
                        modulo=modulo,
                        phase_mod=phase,
                    )

                    if optimize:
                        vec2 = tableXY(jdb, save[p[4]], save[p[4] + "_std"] + calib_std)
                        vec2.y -= np.mean(vec2.y)
                        vec2.y += np.mean(vec.y)
                        vec2.plot(
                            capsize=0,
                            color="k",
                            zorder=10,
                            modulo=modulo,
                            phase_mod=phase,
                        )

                    auto_axis(vec.y, m=5)
                    plt.xlabel("Time")
                    plt.ylabel("Proxy [unit arb.]")
            plt.subplots_adjust(
                left=0.07,
                right=0.93,
                top=0.95,
                bottom=0.08,
                wspace=0.3,
                hspace=0.35,
            )
            plt.savefig(self.dir_root + "IMAGES/all_proxies" + name + ".pdf")


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


def uncorrect_hole(
    self: spec_time_series,
    conti: ndarray,
    conti_ref: ndarray,
    values_forbidden: List[Union[int, float]] = [0, np.inf],
) -> ndarray:
    file_test = self.import_spectrum()
    wave = np.array(file_test["wave"])
    hl = file_test["parameters"]["hole_left"]
    hr = file_test["parameters"]["hole_right"]

    if hl != -99.9:
        i1 = int(find_nearest(wave, hl)[0])
        i2 = int(find_nearest(wave, hr)[0])
        conti[:, i1 - 1 : i2 + 2] = conti_ref[:, i1 - 1 : i2 + 2].copy()

    for l in values_forbidden:
        conti[conti == l] = conti_ref[conti == l].copy()

    return conti
