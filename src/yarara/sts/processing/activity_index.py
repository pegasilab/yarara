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
from ...util import doppler_r, flux_norm_std, get_phase, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


# =============================================================================
# COMPUTE ALL THE TEMPERATURE SENSITIVE RATIO
# =============================================================================


def yarara_activity_index(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    plot: bool = True,
    debug: bool = False,
    calib_std: int = 0,
    optimize: bool = False,
    p_noise: float = 1 / np.inf,
    save_kw: bool = True,
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
    jdb = np.array(self.table["jdb"])
    snrs = np.array(self.table["snr"])

    self.import_material()
    load = self.material
    kw = "_planet" * self.planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")
    time.sleep(1)

    epsilon = 1e-12

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

    file_random = self.import_spectrum()
    waves = file_random["wave"]
    all_prox_names = np.array(all_proxies)[:, 4]
    proxy_found = ((np.array(all_proxies)[:, 0] - np.nanmin(waves[0])) > 0) & (
        (np.nanmax(waves[0])) > 0
    )
    all_proxies = list(np.array(all_proxies)[proxy_found])

    dgrid = np.mean(np.diff(waves))
    all_flux, all_flux_err, conti, conti2, conti_err = self.import_sts_flux(
        load=["flux" + kw, "flux_err", sub_dico, "matching_diff", "continuum_err"]
    )
    flux, err_flux = flux_norm_std(all_flux, all_flux_err, conti + epsilon, conti_err)
    dust, err_flux = flux_norm_std(all_flux, all_flux_err, conti2 + epsilon, conti_err)

    waves = np.array(waves)
    flux *= np.array(load["correction_factor"])
    err_flux *= np.array(load["correction_factor"])
    snrs = np.array(snrs)

    noise_matrix, noise_values = self.yarara_poissonian_noise(
        noise_wanted=p_noise, wave_ref=None, flat_snr=True
    )
    flux += noise_matrix
    err_flux = np.sqrt(err_flux**2 + noise_values**2)

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

    save: Dict[str, Any] = {"null": 0}
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

    def non_neg(prox, prox_std):
        mask = prox <= 0
        prox[mask] = np.median(prox[~mask])
        prox_std[mask] = np.median(prox[~mask]) * 0.99
        return prox, prox_std

    for kw in save.keys():
        if kw[-3:] != "std":
            save[kw], save[kw + "_std"] = non_neg(save[kw], save[kw + "_std"])

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
