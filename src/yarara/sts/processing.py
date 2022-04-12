from __future__ import annotations

import datetime
import glob as glob
import os
import time
from typing import TYPE_CHECKING

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from colorama import Fore
from scipy.interpolate import interp1d
from tqdm import tqdm

from .. import io
from ..analysis import table, tableXY
from ..paths import root
from ..plots import auto_axis, my_colormesh
from ..stats import IQ, find_nearest, identify_nearest, match_nearest, smooth2d
from ..util import doppler_r, flux_norm_std, get_phase, print_box

if TYPE_CHECKING:
    from ..my_rassine_tools import spec_time_series


# =============================================================================
# COMPUTE ALL THE TEMPERATURE SENSITIVE RATIO
# =============================================================================


def yarara_activity_index(
    self: spec_time_series,
    sub_dico="matching_diff",
    continuum="linear",
    plot=True,
    debug=False,
    calib_std=0,
    optimize=False,
    substract_map=[],
    add_map=[],
    p_noise=1 / np.inf,
    save=True,
):
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

        f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)
        dustbin, f_norm_std = flux_norm_std(
            f, f_std, file["matching_diff"]["continuum_" + continuum], c_std
        )

        fluxes.append(f_norm)
        err_fluxes.append(f_norm_std)
        self.debug = j

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

    for maps in substract_map:
        flux = self.yarara_substract_map(flux, maps, correction_factor=True)

    for maps in add_map:
        flux = self.yarara_add_map(flux, maps, correction_factor=True)

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
# COMPUTE THE CCF OF THE RASSINE SPECTRUM
# =============================================================================


def yarara_ccf(
    self: spec_time_series,
    sub_dico="matching_diff",
    continuum="linear",
    mask=None,
    mask_name=None,
    ccf_name=None,
    mask_col="weight_rv",
    treshold_telluric=1,
    ratio=False,
    element=None,
    reference=True,
    weighted=True,
    plot=False,
    display_ccf=False,
    save=True,
    save_ccf_profile=False,
    normalisation="left",
    del_outside_max=False,
    bis_analysis=False,
    ccf_oversampling=1,
    rv_range=None,
    rv_borders=None,
    delta_window=5,
    debug=False,
    rv_sys=None,
    rv_shift=None,
    speed_up=True,
    force_brute=False,
    wave_min=None,
    wave_max=None,
    squared=True,
    p_noise=1 / np.inf,
    substract_map=[],
    add_map=[],
):
    """
    Compute the CCF of a spectrum, reference to use always the same continuum (matching_anchors highest SNR).
    Display_ccf to plot all the individual CCF. Plot to plot the FWHM, contrast and RV.

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    mask : The line mask used to cross correlate with the spectrum (mask should be located in MASK_CCF otherwise KITCAT dico)
    mask_col : Column of the KitCat column to use for the weight
    threshold_telluric : Maximum telluric contamination to keep a stellar line in the mask
    reference : True/False or 'norm', True use the matching anchors of reference, False use the continuum of each spectrum, norm use the continuum normalised spectrum (not )
    plot : True/False to plot the RV time-series
    display_ccf : display all the ccf subproduct
    save : True/False to save the informations iun summary table
    normalisation : 'left' or 'slope'. if left normalise the CCF by the most left value, otherwise fit a line between the two highest point
    del_outside maximam : True/False to delete the CCF outside the two bump in personal mask
    speed_up : remove region from the CCF not crossed by a line in the mask to speed up the code
    force_brute : force to remove the region excluded by the brute mask

    """

    directory = self.directory
    planet = self.planet

    def replace_none(y, yerr):
        if yerr is None:
            return np.nan, 1e6
        else:
            return y, yerr

    if rv_range is None:
        rv_range = int(3 * self.fwhm)
        print("\n [INFO] RV range updated to : %.1f kms" % (rv_range))

    if rv_borders is None:
        rv_borders = int(2 * self.fwhm)
        print("\n [INFO] RV borders updated to : %.1f kms" % (rv_borders))

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    self.import_table()

    files = glob.glob(directory + "RASSI*.p")
    files = np.array(self.table["filename"])  # updated 29.10.21 to allow ins_merged
    files = np.sort(files)

    flux = []
    snr = []
    conti = []
    flux_err = []
    jdb = []
    berv = []

    epsilon = 1e-12

    file_random = self.import_spectrum()
    self.import_table()
    self.import_material()
    load = self.material

    mask_loc = mask_name
    if mask_name is None:
        mask_name = "No name"
        mask_loc = "No name"

    if mask is None:
        mask = self.mask_harps
        mask_name = self.mask_harps

    if type(mask) == str:
        if mask.split(".")[-1] == "p":
            loc_mask = self.dir_root + "KITCAT/"
            mask_name = mask
            mask_loc = loc_mask + mask
            dico = pd.read_pickle(mask_loc)["catalogue"]
            dico = dico.loc[dico["rel_contam"] < treshold_telluric]
            if "valid" in dico.keys():
                dico = dico.loc[dico["valid"]]
            if element is not None:
                dico = dico.loc[dico["element"] == element]
            mask = np.array([np.array(dico["freq_mask0"]), np.array(dico[mask_col])]).T
            mask = mask[mask[:, 1] != 0]
            print("\n [INFO] Nb lines in the CCF mask : %.0f" % (len(dico)))

        else:
            mask_name = mask
            mask_loc = root + "/Python/MASK_CCF/" + mask + ".txt"
            mask = np.genfromtxt(mask_loc)
            mask = np.array([0.5 * (mask[:, 0] + mask[:, 1]), mask[:, 2]]).T

    if type(mask) == pd.core.frame.DataFrame:
        dico = mask
        mask = np.array(
            [
                np.array(mask["freq_mask0"]).astype("float"),
                np.array(mask[mask_col]).astype("float"),
            ]
        ).T

    print("\n [INFO] CCF mask selected : %s \n" % (mask_loc))

    if rv_sys is None:
        if file_random["parameters"]["RV_sys"] is not None:
            rv_sys = 1000 * file_random["parameters"]["RV_sys"]
        else:
            rv_sys = 0
    else:
        rv_sys *= 1000

    if rv_shift is None:
        rv_shift = np.zeros(len(files))

    print("\n [INFO] RV sys : %.2f [km/s] \n" % (rv_sys / 1000))

    mask[:, 0] = doppler_r(mask[:, 0], rv_sys)[0]

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        if not i:
            wave = file["wave"]
        snr.append(file["parameters"]["SNR_5500"])
        try:
            jdb.append(file["parameters"]["jdb"])
        except:
            jdb.append(i)
        try:
            berv.append(file["parameters"]["berv"])
        except:
            berv.append(0)

        f = file["flux" + kw]
        f_std = file["flux_err"]
        c = file[sub_dico]["continuum_" + continuum] + epsilon
        c_std = file["continuum_err"]

        f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)

        flux.append(f_norm)
        flux_err.append(f_norm_std)
        conti.append(c - epsilon)

    wave = np.array(wave)
    flux = np.array(flux)
    flux_err = np.array(flux_err)
    snr = np.array(snr)
    conti = np.array(conti)
    jdb = np.array(jdb)
    berv = np.array(berv)
    grid = wave

    flux, flux_err, wave = self.yarara_map(
        sub_dico=sub_dico,
        planet=self.planet,
        wave_min=None,
        wave_max=None,
        Plot=False,
        reference=False,
        substract_map=substract_map,
        add_map=add_map,
        correction_factor=False,
    )  # 04.08.21 in order to include add and substract map

    flux *= np.array(load["correction_factor"])
    flux_err *= np.array(load["correction_factor"])

    noise_matrix, noise_values = self.yarara_poissonian_noise(
        noise_wanted=p_noise, wave_ref=None, flat_snr=True
    )
    flux += noise_matrix
    flux_err = np.sqrt(flux_err**2 + noise_values**2)

    if reference == True:
        norm_factor = np.array(load["color_template"])
        flux *= norm_factor
        flux_err *= norm_factor
    elif reference == "norm":
        if ratio:
            norm_factor = np.array(load["reference_spectrum"]) * np.array(
                load["correction_factor"]
            )
            norm_factor[norm_factor == 0] = 1
            flux /= norm_factor
            flux_err /= norm_factor
        else:
            pass
    elif reference == "master_snr":
        norm_factor = np.array(load["master_snr_curve"]) ** 2
        norm_factor[np.isnan(norm_factor)] = 1
        norm_factor *= np.nanmean(np.array(load["color_template"])) / np.nanmean(norm_factor)
        flux *= norm_factor
        flux_err *= norm_factor
    else:
        flux *= conti
        flux_err *= conti

    if sub_dico == "matching_brute":
        force_brute = True

    mask_shifted = doppler_r(mask[:, 0], (rv_range + 5) * 1000)

    if force_brute:
        brute_mask = np.array(load["mask_brute"])
        used_region = ((grid) >= mask_shifted[1][:, np.newaxis]) & (
            (grid) <= mask_shifted[0][:, np.newaxis]
        )
        line_killed = np.sum(brute_mask * used_region, axis=1) == 0
        mask = mask[line_killed]
        mask_shifted = doppler_r(mask[:, 0], (rv_range + 5) * 1000)

    mask = mask[
        (doppler_r(mask[:, 0], 30000)[0] < grid.max())
        & (doppler_r(mask[:, 0], 30000)[1] > grid.min()),
        :,
    ]  # supres line farther than 30kms
    if wave_min is not None:
        mask = mask[mask[:, 0] > wave_min, :]
    if wave_max is not None:
        mask = mask[mask[:, 0] < wave_max, :]

    print("\n [INFO] Nb lines in the mask : %.0f \n" % (len(mask)))

    mask_min = np.min(mask[:, 0])
    mask_max = np.max(mask[:, 0])

    # supress useless part of the spectra to speed up the CCF
    grid_min = int(find_nearest(grid, doppler_r(mask_min, -100000)[0])[0])
    grid_max = int(find_nearest(grid, doppler_r(mask_max, 100000)[0])[0])
    grid = grid[grid_min:grid_max]

    log_grid = np.linspace(np.log10(grid).min(), np.log10(grid).max(), len(grid))
    dgrid = log_grid[1] - log_grid[0]
    # dv = (10**(dgrid)-1)*299.792e6

    # computation of region free of spectral line to increase code speed

    if speed_up:
        used_region = ((10**log_grid) >= mask_shifted[1][:, np.newaxis]) & (
            (10**log_grid) <= mask_shifted[0][:, np.newaxis]
        )
        used_region = (np.sum(used_region, axis=0) != 0).astype("bool")
        print(
            "\n [INFO] Percentage of the spectrum used : %.1f [%%] (%.0f) \n"
            % (100 * sum(used_region) / len(grid), len(grid))
        )
        time.sleep(1)
    else:
        used_region = np.ones(len(grid)).astype("bool")

    if (
        not os.path.exists(self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits")
    ) | (force_brute):
        print(
            "\n [INFO] CCF mask reduced for the first time, wait for the static mask producing...\n"
        )
        time.sleep(1)
        mask_wave = np.log10(mask[:, 0])
        mask_contrast = mask[:, 1] * weighted + (1 - weighted)

        mask_hole = (mask[:, 0] > doppler_r(file_random["parameters"]["hole_left"], -30000)[0]) & (
            mask[:, 0] < doppler_r(file_random["parameters"]["hole_right"], 30000)[0]
        )
        mask_contrast[mask_hole] = 0

        log_grid_mask = np.arange(
            log_grid.min() - 10 * dgrid,
            log_grid.max() + 10 * dgrid + dgrid / 10,
            dgrid / 11,
        )
        log_mask = np.zeros(len(log_grid_mask))

        # mask_contrast /= np.sqrt(np.nansum(mask_contrast**2)) #UPDATE 04.05.21 (DOES NOT WORK)

        match = identify_nearest(mask_wave, log_grid_mask)
        for j in np.arange(-delta_window, delta_window + 1, 1):
            log_mask[match + j] = (mask_contrast) ** (1 + int(squared))

        plt.figure()
        plt.plot(log_grid_mask, log_mask)

        if (not force_brute) & (mask_name != "No name"):
            hdu = fits.PrimaryHDU(np.array([log_grid_mask, log_mask]).T)
            hdul = fits.HDUList([hdu])
            hdul.writeto(self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits")
            print(
                "\n [INFO] CCF mask saved under : %s"
                % (self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits")
            )
    else:
        print(
            "\n [INFO] CCF mask found : %s"
            % (self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits")
        )
        log_grid_mask, log_mask = fits.open(
            self.dir_root + "CCF_MASK/CCF_" + mask_name.split(".")[0] + ".fits"
        )[0].data.T

    log_template = interp1d(
        log_grid_mask,
        log_mask,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(log_grid)

    flux = flux[:, grid_min:grid_max]
    flux_err = flux_err[:, grid_min:grid_max]

    amplitudes = []
    amplitudes_std = []
    rvs = []
    rvs_std = []
    fwhms = []
    fwhms_std = []
    ew = []
    ew_std = []
    centers = []
    centers_std = []
    depths = []
    depths_std = []
    bisspan = []
    bisspan_std = []
    b0s = []
    b1s = []
    b2s = []
    b3s = []
    b4s = []

    if display_ccf:
        plt.figure()

    now = datetime.datetime.now()
    print(
        "\n Computing CCF (Current time %.0fh%.0fm%.0fs) \n" % (now.hour, now.minute, now.second)
    )

    all_flux = []
    for j, i in enumerate(files):
        all_flux.append(
            interp1d(
                np.log10(doppler_r(grid, rv_shift[j])[0]),
                flux[j],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )(log_grid)
        )
    all_flux = np.array(all_flux)

    all_flux_err = []
    for j, i in enumerate(files):
        all_flux_err.append(
            interp1d(
                np.log10(doppler_r(grid, rv_shift[j])[0]),
                flux_err[j],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(log_grid)
        )
    all_flux_err = np.array(all_flux_err)

    vrad, ccf_power, ccf_power_std = ccf(
        log_grid[used_region],
        all_flux[:, used_region],
        log_template[used_region],
        rv_range=rv_range,
        oversampling=ccf_oversampling,
        spec1_std=all_flux_err[:, used_region],
    )  # to compute on all the ccf simultaneously

    now = datetime.datetime.now()
    print("")
    print("\n CCF computed (Current time %.0fh%.0fm%.0fs) \n" % (now.hour, now.minute, now.second))

    try:
        self.all_ccf_saved[sub_dico] = (vrad, ccf_power, ccf_power_std)
    except AttributeError:
        self.all_ccf_saved = {sub_dico: (vrad, ccf_power, ccf_power_std)}

    ccf_ref = np.median(ccf_power, axis=1)
    continuum_ccf = np.argmax(ccf_ref)
    top_ccf = np.argsort(ccf_ref)[
        -int(len(ccf_ref) / 2) :
    ]  # roughly half of a CCF is made of the continuum

    ccf_snr = 1 / (
        np.std((ccf_power - ccf_ref[:, np.newaxis])[top_ccf], axis=0)
        / np.mean(ccf_power[continuum_ccf])
    )
    print(" [INFO] SNR CCF continuum median : %.0f\n" % (np.median(ccf_snr)))

    # noise_ccf = ccf_power_std
    # w = noise_ccf/(np.gradient(ccf_ref)/np.gradient(vrad)+epsilon)[:,np.newaxis]
    # w[w==0] = np.min(w[w!=0])/10
    # svrad_phot = 1/np.sqrt(np.sum(1/w**2,axis=0))
    # scaling = 820/np.mean(np.gradient(vrad)) #to penalize oversampling in vrad
    # svrad_phot*=scaling
    # self.svrad_phot = svrad_phot

    noise_ccf = [
        (np.sqrt(ccf_ref / np.max(ccf_ref)) * ccf_ref[continuum_ccf])[:, np.newaxis] / ccf_snr,
        ccf_power_std,
    ][
        int(ratio)
    ]  # assume that the noise in the continuum is white (okay for matching_mad but wrong when tellurics are still there)
    sigma_rv = noise_ccf / (abs(np.gradient(ccf_ref)) / np.gradient(vrad))[:, np.newaxis]
    w_rv = (1 / sigma_rv) ** 2
    svrad_phot = 1 / np.sqrt(np.sum(w_rv, axis=0))
    scaling = np.sqrt(820 / np.mean(np.gradient(vrad)))  # to penalize oversampling in vrad
    svrad_phot *= scaling

    svrad_phot[svrad_phot == 0] = 2 * np.max(svrad_phot)  # in case of null values

    print(" [INFO] Photon noise RV median : %.2f m/s\n " % (np.median(svrad_phot)))

    svrad_phot2 = {}
    svrad_phot2["rv"] = 10 ** (
        0.98 * np.log10(svrad_phot) - 3.08
    )  # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2["contrast"] = 10 ** (
        0.98 * np.log10(svrad_phot) - 3.58
    )  # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2["fwhm"] = 10 ** (
        0.98 * np.log10(svrad_phot) - 2.94
    )  # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2["center"] = 10 ** (
        0.98 * np.log10(svrad_phot) - 2.83
    )  # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2["depth"] = 10 ** (
        0.97 * np.log10(svrad_phot) - 3.62
    )  # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2["ew"] = 10 ** (
        0.97 * np.log10(svrad_phot) - 3.47
    )  # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2["vspan"] = 10 ** (
        0.98 * np.log10(svrad_phot) - 2.95
    )  # from photon noise simulations Photon_noise_CCF.py

    self.svrad_phot = svrad_phot2["rv"]

    print(
        " [INFO] Photon noise RV from calibration : %.2f m/s\n "
        % (np.median(svrad_phot2["rv"]) * 1000)
    )

    for j, i in enumerate(files):
        if bis_analysis:
            print("File (%.0f/%.0f) %s SNR %.0f reduced" % (j + 1, len(files), i, snr[j]))
        file = pd.read_pickle(i)
        # log_spectrum = interp1d(np.log10(grid), flux[j], kind='cubic', bounds_error=False, fill_value='extrapolate')(log_grid)
        # vrad, ccf_power_old = ccf2(log_grid, log_spectrum,  log_grid_mask, log_mask)
        # vrad, ccf_power_old = ccf(log_grid, log_spectrum, log_template, rv_range=45, oversampling=ccf_oversampling)
        ccf_power_old = ccf_power[:, j]
        ccf_power_old_std = ccf_power_std[:, j]
        ccf = tableXY(vrad / 1000, ccf_power_old, ccf_power_old_std)
        ccf.yerr = np.sqrt(abs(ccf.y))

        ccf.y *= -1
        ccf.find_max(vicinity=5)

        ccf.diff(replace=False)
        ccf.deri.y = np.abs(ccf.deri.y)
        for jj in range(3):
            ccf.deri.find_max(vicinity=4 - jj)
            if len(ccf.deri.x_max) > 1:
                break

        first_max = ccf.deri.x_max[np.argsort(ccf.deri.y_max)[-1]]
        second_max = ccf.deri.x_max[np.argsort(ccf.deri.y_max)[-2]]

        ccf.y *= -1
        if (np.min(abs(ccf.x_max - 0.5 * (first_max + second_max))) < 5) & (self.fwhm < 15):
            center = ccf.x_max[np.argmin(abs(ccf.x_max - 0.5 * (first_max + second_max)))]
        else:
            center = ccf.x[ccf.y.argmin()]
        ccf.x -= center

        if not del_outside_max:
            mask = (ccf.x > -rv_borders) & (ccf.x < rv_borders)
            ccf.supress_mask(mask)
        else:
            ccf.find_max(vicinity=10)
            ccf.index_max = np.sort(ccf.index_max)
            mask = np.zeros(len(ccf.x)).astype("bool")
            mask[ccf.index_max[0] : ccf.index_max[1] + 1] = True
            ccf.supress_mask(mask)

        if normalisation == "left":
            norm = ccf.y[0]
        else:
            max1 = np.argmax(ccf.y[0 : int(len(ccf.y) / 2)])
            max2 = np.argmax(ccf.y[int(len(ccf.y) / 2) :]) + int(len(ccf.y) / 2)
            fmax1 = ccf.y[max1]
            fmax2 = ccf.y[max2]
            norm = (fmax2 - fmax1) / (max2 - max1) * (np.arange(len(ccf.y)) - max2) + fmax2
        ccf.yerr /= norm
        ccf.y /= norm

        if ratio:
            ccf.yerr *= 0
            ccf.yerr += 0.01

        if display_ccf:
            ccf.plot(color=None)

        # bis #interpolated on 10 m/s step
        moments = np.zeros(5)
        b0 = []
        b1 = []
        b2 = []
        b3 = []
        b4 = []

        if bis_analysis:
            ccf.x *= 1000
            drv = 10
            border = np.min([abs(ccf.x.min()), abs(ccf.x.max())])
            border = (border // drv) * drv
            grid_vrad = np.arange(-border, border + drv, drv)
            ccf.interpolate(new_grid=grid_vrad, replace=False)
            ccf.interpolated.yerr *= 0
            ccf.interpolated.y = (ccf.interpolated.y - ccf.interpolated.y.min()) / (
                ccf.interpolated.y.max() - ccf.interpolated.y.min()
            )
            ccf.interpolated.my_bisector(oversampling=10, between_max=True)
            ccf.interpolated.bis.clip(min=[0.01, None], max=[0.7, None])
            bis = ccf.interpolated.bis
            bis.interpolate(new_grid=np.linspace(0.01, 0.7, 20))
            bis.y -= bis.y[0]
            if save:
                save_bis = {
                    "bis_flux": bis.x,
                    "bis_rv": bis.y,
                    "bis_rv_std": bis.yerr,
                }
                file["ccf_bis"] = save_bis
            bis.y = np.gradient(bis.y)
            for p in range(5):
                moments[p] = np.sum(bis.x**p * bis.y)
            b0.append(moments[0])
            b1.append(moments[1])
            b2.append(moments[2])
            b3.append(moments[3])
            b4.append(moments[4])
            ccf.x /= 1000
        else:
            b0.append(0)
            b1.append(0)
            b2.append(0)
            b3.append(0)
            b4.append(0)

        ccf.clip(min=[-0.5, None], max=[0.5, None], replace=False)
        if len(ccf.clipped.x) < 7:
            ccf.clip(min=[-2, None], max=[2, None], replace=False)
        ccf.clipped.fit_poly()
        a, b, c = ccf.clipped.poly_coefficient
        para_center = -b / (2 * a) + center
        para_depth = a * (-b / (2 * a)) ** 2 + b * (-b / (2 * a)) + c
        centers.append(para_center)
        depths.append(1 - para_depth)

        EW = np.sum(ccf.y - 1) / len(ccf.y)
        ew.append(EW)
        save_ccf = {
            "ccf_flux": ccf.y,
            "ccf_flux_std": ccf.yerr,
            "ccf_rv": ccf.x + center,
            "reference": reference,
            "ew": EW,
        }

        para_ccf = {"para_rv": para_center, "para_depth": para_depth}

        ccf.fit_gaussian(Plot=False)  # ,guess=[-self.contrast,0,self.fwhm/2.355,1])

        rv_ccf = ccf.params["cen"].value + center
        rv_ccf_std = ccf.params["cen"].stderr
        rv_ccf, rv_ccf_std = replace_none(rv_ccf, rv_ccf_std)
        rv_ccf_std = svrad_phot2["rv"][j]
        factor = rv_ccf_std / abs(rv_ccf)
        scaling_noise = {
            "amp": 0.32,
            "wid": 1.33,
            "depth": 0.29,
            "center": 1.79,
            "bisspan": 1.37,
            "ew": 0.42,
        }

        contrast_ccf = -ccf.params["amp"].value
        contrast_ccf_std = ccf.params["amp"].stderr
        contrast_ccf, contrast_ccf_std = replace_none(contrast_ccf, contrast_ccf_std)
        contrast_ccf_std = svrad_phot2["contrast"][
            j
        ]  # abs(contrast_ccf)*factor*scaling_noise['amp']

        wid_ccf = ccf.params["wid"].value
        wid_ccf_std = ccf.params["wid"].stderr
        wid_ccf, wid_ccf_std = replace_none(wid_ccf, wid_ccf_std)
        wid_ccf_std = svrad_phot2["fwhm"][j]  # abs(wid_ccf)*factor*scaling_noise['wid']

        offset_ccf = ccf.params["offset"].value
        offset_ccf_std = ccf.params["offset"].stderr
        offset_ccf, offset_ccf_std = replace_none(offset_ccf, offset_ccf_std)

        amplitudes.append(contrast_ccf)
        amplitudes_std.append(contrast_ccf_std)
        rvs.append(rv_ccf)
        rvs_std.append(rv_ccf_std)
        fwhms.append(wid_ccf)
        fwhms_std.append(wid_ccf_std)
        bisspan.append(rv_ccf - para_center)
        bisspan_ccf_std = svrad_phot2["vspan"][
            j
        ]  # abs(rv_ccf - para_center)*factor*scaling_noise['bisspan']
        bisspan_std.append(bisspan_ccf_std)

        ew_std.append(svrad_phot2["ew"][j])  # abs(EW)*factor*scaling_noise['ew'])
        centers_std.append(
            svrad_phot2["center"][j]
        )  # abs(para_center)*factor*scaling_noise['center'])
        depths_std.append(
            svrad_phot2["depth"][j]
        )  # abs(1-para_depth)*factor*scaling_noise['depth'])

        save_ccf["ew_std"] = ew_std
        para_ccf["para_rv_std"] = centers_std
        para_ccf["para_depth_std"] = depths_std

        file["ccf"] = save_ccf
        file["ccf_parabola"] = para_ccf

        b0s.append(moments[0])
        b1s.append(moments[1])
        b2s.append(moments[2])
        b3s.append(moments[3])
        b4s.append(moments[4])
        if save:
            save_gauss = {
                "contrast": contrast_ccf,
                "contrast_std": contrast_ccf_std,
                "rv": rv_ccf,
                "rv_std": rv_ccf_std,
                "rv_std_phot": svrad_phot2["rv"][j],
                "fwhm": wid_ccf,
                "fwhm_std": wid_ccf_std,
                "offset": offset_ccf,
                "offset_std": offset_ccf_std,
                "vspan": rv_ccf - para_center,
                "vspan_std": bisspan_std,
                "b0": moments[0],
                "b1": moments[1],
                "b2": moments[2],
                "b3": moments[3],
                "b4": moments[4],
            }

            file["ccf_gaussian"] = save_gauss
            #
            #                    ccf.my_bisector(between_max=True,oversampling=50)
            #                    bis = tableXY(ccf.bisector[5::50,1],ccf.bisector[5::50,0]+center,ccf.bisector[5::50,2])
            #
            #                    save_bis = {'bis_flux':bis.x,'bis_rv':bis.y,'bis_rv_std':bis.yerr}
            #                    file['ccf_bis'] = save_bis

            io.pickle_dump(file, open(i, "wb"))

    # try:
    #     rvs_std = np.array(self.table['rv_dace_std'])/1000
    # except:
    #     pass

    rvs_std_backup = np.array(self.table["rv_dace_std"]) / 1000
    rvs_std = svrad_phot2["rv"]
    rvs_std[rvs_std == 0] = rvs_std_backup[rvs_std == 0]

    fwhms = np.array(fwhms).astype("float") * 2.355
    fwhms_std = np.array(fwhms_std).astype("float") * 2.355

    self.warning_rv_borders = False
    if np.median(fwhms) > (rv_borders / 1.5):
        print("[WARNING] The CCF is larger than the RV borders for the fit")
        self.warning_rv_borders = True

    self.ccf_rv = tableXY(jdb, np.array(rvs) * 1000, np.array(rvs_std) * 1000)
    self.ccf_centers = tableXY(jdb, np.array(centers) * 1000, np.array(centers_std) * 1000)
    self.ccf_contrast = tableXY(jdb, amplitudes, amplitudes_std)
    self.ccf_depth = tableXY(jdb, depths, depths_std)
    self.ccf_fwhm = tableXY(jdb, fwhms, fwhms_std)
    self.ccf_vspan = tableXY(jdb, np.array(bisspan) * 1000, np.array(bisspan_std) * 1000)
    self.ccf_ew = tableXY(jdb, np.array(ew), np.array(ew_std))
    self.ccf_bis0 = tableXY(jdb, b0s, np.sqrt(2) * np.array(rvs_std) * 1000)
    self.ccf_timeseries = np.array(
        [
            ew,
            ew_std,
            amplitudes,
            amplitudes_std,
            rvs,
            rvs_std,
            svrad_phot2["rv"],
            fwhms,
            fwhms_std,
            centers,
            centers_std,
            depths,
            depths_std,
            b0s,
            bisspan,
            bisspan_std,
        ]
    )
    self.ccf_rv.rms_w()
    self.ccf_centers.rms_w()
    self.ccf_rv_shift = center

    ccf_infos = pd.DataFrame(
        self.ccf_timeseries.T,
        columns=[
            "ew",
            "ew_std",
            "contrast",
            "contrast_std",
            "rv",
            "rv_std",
            "rv_std_phot",
            "fwhm",
            "fwhm_std",
            "center",
            "center_std",
            "depth",
            "depth_std",
            "b0",
            "bisspan",
            "bisspan_std",
        ],
    )
    ccf_infos["jdb"] = jdb
    ccf_infos = {
        "table": ccf_infos,
        "creation_date": datetime.datetime.now().isoformat(),
    }

    if not os.path.exists(self.directory + "Analyse_ccf.p"):
        ccf_summary = {"star_info": {"name": self.starname}}
        io.pickle_dump(ccf_summary, open(self.directory + "/Analyse_ccf.p", "wb"))

    if ccf_name is None:
        ccf_name = sub_dico

    if save:
        if mask_name != "No name":
            file_summary_ccf = pd.read_pickle(self.directory + "Analyse_ccf.p")
            try:
                file_summary_ccf["CCF_" + mask_name.split(".")[0]][ccf_name] = ccf_infos
            except KeyError:
                file_summary_ccf["CCF_" + mask_name.split(".")[0]] = {ccf_name: ccf_infos}

            io.pickle_dump(file_summary_ccf, open(self.directory + "/Analyse_ccf.p", "wb"))

    self.infos["latest_dico_ccf"] = ccf_name

    self.yarara_analyse_summary()

    if save_ccf_profile:
        self.yarara_ccf_save(mask_name.split(".")[0], sub_dico)

    if plot:
        plt.figure(figsize=(12, 10))
        plt.subplot(4, 2, 1)
        self.ccf_rv.plot(
            label=r"rms : %.2f | $\sigma_{\gamma}$ : %.2f"
            % (self.ccf_rv.rms, np.median(self.svrad_phot) * 1000)
        )
        plt.legend()
        plt.title("RV", fontsize=14)
        ax = plt.gca()

        plt.subplot(4, 2, 3, sharex=ax)  # .scatter(jdb,ew,color='k')
        self.ccf_ew.plot()
        plt.title("EW", fontsize=14)
        plt.ylim(
            np.nanpercentile(ew, 25) - 1.5 * IQ(ew),
            np.nanpercentile(ew, 75) + 1.5 * IQ(ew),
        )

        plt.subplot(4, 2, 5, sharex=ax)  # .scatter(jdb,amplitudes,color='k')
        self.ccf_contrast.plot()
        plt.title("Contrast", fontsize=14)
        plt.ylim(
            np.nanpercentile(amplitudes, 25) - 1.5 * IQ(amplitudes),
            np.nanpercentile(amplitudes, 75) + 1.5 * IQ(amplitudes),
        )

        plt.subplot(4, 2, 4, sharex=ax)  # .scatter(jdb,fwhms,color='k')
        self.ccf_fwhm.plot()
        plt.title("FWHM", fontsize=14)
        plt.ylim(
            np.nanpercentile(fwhms, 25) - 1.5 * IQ(fwhms),
            np.nanpercentile(fwhms, 75) + 1.5 * IQ(fwhms),
        )

        plt.subplot(4, 2, 6, sharex=ax)  # .scatter(jdb,depths,color='k')
        self.ccf_depth.plot()
        plt.title("Depth", fontsize=14)
        plt.ylim(
            np.nanpercentile(depths, 25) - 1.5 * IQ(depths),
            np.nanpercentile(depths, 75) + 1.5 * IQ(depths),
        )

        plt.subplot(4, 2, 2, sharex=ax, sharey=ax)
        self.ccf_centers.plot(label="rms : %.2f" % (self.ccf_centers.rms))
        plt.legend()
        plt.title("Center", fontsize=14)

        plt.subplot(4, 2, 7, sharex=ax).scatter(jdb, b0s, color="k")
        plt.title("BIS", fontsize=14)

        plt.subplot(4, 2, 8, sharex=ax)  # .scatter(jdb,bisspan,color='k')
        self.ccf_vspan.plot()
        plt.title(r"RV $-$ Center (VSPAN)", fontsize=14)
        plt.ylim(
            np.nanpercentile(self.ccf_vspan.y, 25) - 1.5 * IQ(self.ccf_vspan.y),
            np.nanpercentile(self.ccf_vspan.y, 75) + 1.5 * IQ(self.ccf_vspan.y),
        )
        plt.subplots_adjust(left=0.07, right=0.93, top=0.95, bottom=0.08, wspace=0.3, hspace=0.3)

        if bis_analysis:
            plt.figure(figsize=(12, 10))
            plt.subplot(3, 2, 1).scatter(jdb, b0s, color="k")
            ax = plt.gca()
            plt.title("B0", fontsize=14)
            plt.subplot(3, 2, 2, sharex=ax).scatter(
                jdb, self.ccf_rv.y - self.ccf_centers.y, color="k"
            )
            plt.title("RV-Center", fontsize=14)
            plt.subplot(3, 2, 3, sharex=ax).scatter(jdb, b1s, color="k")
            plt.title("B1", fontsize=14)
            plt.subplot(3, 2, 4, sharex=ax).scatter(jdb, b2s, color="k")
            plt.title("B2", fontsize=14)
            plt.subplot(3, 2, 5, sharex=ax).scatter(jdb, b3s, color="k")
            plt.title("B3", fontsize=14)
            plt.subplot(3, 2, 6, sharex=ax).scatter(jdb, b4s, color="k")
            plt.title("B4", fontsize=14)

    return {
        "rv": self.ccf_rv,
        "cen": self.ccf_centers,
        "contrast": self.ccf_contrast,
        "fwhm": self.ccf_fwhm,
        "vspan": self.ccf_vspan,
    }


# =============================================================================
# VISUALISE THE RASSINE TIMESERIES AND ITS CORRELATION WITH A PROXY
# =============================================================================


def yarara_map(
    self: spec_time_series,
    sub_dico="matching_diff",
    continuum="linear",
    planet=False,
    modulo=None,
    unit=1.0,
    wave_min=4000,
    wave_max=4300,
    time_min=None,
    time_max=None,
    index="index",
    ratio=False,
    reference="median",
    berv_shift=False,
    rv_shift=False,
    new=True,
    Plot=True,
    substract_map=[],
    add_map=[],
    correction_factor=True,
    p_noise=1 / np.inf,
):

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

    self.import_table()

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

    flux = []
    err_flux = []
    snr = []
    jdb = []
    berv = []
    rv = []

    epsilon = 1e-12

    for i, j in enumerate(files):
        self.current_file = j
        file = pd.read_pickle(j)
        if not i:
            wave = file["wave"]
        f = file["flux" + kw]
        f_std = file["flux_err"]
        c = file[sub_dico]["continuum_" + continuum] + epsilon
        c_std = file["continuum_err"]
        f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)

        snr.append(file["parameters"]["SNR_5500"])
        flux.append(f_norm)
        err_flux.append(f_norm_std)

        try:
            jdb.append(file["parameters"]["jdb"])
        except:
            jdb.append(i)
        if type(berv_shift) != np.ndarray:
            try:
                berv.append(file["parameters"][berv_shift])
            except:
                berv.append(0)
        else:
            berv = berv_shift
        if type(rv_shift) != np.ndarray:
            try:
                rv.append(file["parameters"][rv_shift])
            except:
                rv.append(0)
        else:
            rv = rv_shift

    del self.current_file

    wave = np.array(wave)
    flux = np.array(flux)
    err_flux = np.array(err_flux)
    snr = np.array(snr)
    jdb = np.array(jdb)
    berv = np.array(berv)
    rv = np.array(rv)
    rv = rv - np.median(rv)

    self.debug = flux

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

    old_length = len(wave)
    wave = wave[int(idx_min) : int(idx_max)]
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
    correction_map="matching_smooth",
    sub_dico="matching_cosmics",
    continuum="linear",
):

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


def uncorrect_hole(self: spec_time_series, conti, conti_ref, values_forbidden=[0, np.inf]):
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
