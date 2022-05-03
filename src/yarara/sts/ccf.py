from __future__ import annotations

import datetime
import glob as glob
import logging
import os
import time
from dataclasses import dataclass
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
from ..analysis import tableXY
from ..paths import root
from ..stats import IQ, find_nearest, identify_nearest, match_nearest, smooth2d
from ..util import ccf as ccf_fun
from ..util import doppler_r, get_phase, print_box

if TYPE_CHECKING:
    from . import spec_time_series


@dataclass(frozen=True)
class CCFResult:
    rv: tableXY
    centers: tableXY
    contrast: tableXY
    depth: tableXY
    fwhm: tableXY
    vspan: tableXY
    ew: tableXY
    bis0: tableXY
    jdb: np.ndarray
    rv_photon_noise: np.ndarray
    rv_shift: float

    def timeseries(self) -> np.ndarray:
        return np.array(
            [
                self.ew.y,
                self.ew.yerr,
                self.contrast.y,
                self.contrast.yerr,
                self.rv.y / 1000,
                self.rv.yerr / 1000,
                self.rv_photon_noise,
                self.fwhm.y,
                self.fwhm.yerr,
                self.centers.y / 1000,
                self.centers.yerr / 1000,
                self.depth.y,
                self.depth.yerr,
                self.bis0.y,
                self.vspan.y / 1000,
                self.vspan.yerr / 1000,
            ]
        )

    def infos(self) -> pd.DataFrame:
        res = pd.DataFrame(
            self.timeseries().T,
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
        res["jdb"] = self.jdb


def yarara_ccf(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    mask: Union[ndarray, str, None] = None,
    mask_name: None = None,
    ccf_name: None = None,
    mask_col: str = "weight_rv",
    treshold_telluric: int = 1,
    ratio: bool = False,
    element: None = None,
    reference: Union[bool, str] = True,
    weighted: bool = True,
    plot: bool = False,
    display_ccf: bool = False,
    save: bool = True,
    save_ccf_profile: bool = False,
    normalisation: str = "left",
    del_outside_max: bool = False,
    bis_analysis: bool = False,
    ccf_oversampling: int = 1,
    rv_range: Optional[float] = None,
    rv_borders: Optional[float] = None,
    delta_window: int = 5,
    rv_sys: Optional[float] = None,
    rv_shift: Optional[ndarray] = None,
    speed_up: bool = True,
    force_brute: bool = False,
    wave_min: None = None,
    wave_max: None = None,
    time_min: Optional[float] = None,
    time_max: Optional[float] = None,
    squared: bool = True,
    p_noise: float = 1 / np.inf,
    substract_map: List[Any] = [],
    add_map: List[Any] = [],
) -> Dict[str, tableXY]:
    """
    Compute the CCF of a spectrum, reference to use always the same continuum (matching_anchors highest SNR).
    Display_ccf to plot all the individual CCF. Plot to plot the FWHM, contrast and RV.

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
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
        logging.info("RV range updated to : %.1f kms" % (rv_range))

    if rv_borders is None:
        rv_borders = int(2 * self.fwhm)
        logging.info("RV borders updated to : %.1f kms" % (rv_borders))

    kw = "_planet" * planet
    if kw != "":
        logging.info("PLANET ACTIVATED")

    self.import_table()

    files = glob.glob(directory + "RASSI*.p")
    files = np.array(self.table["filename"])  # updated 29.10.21 to allow ins_merged
    files = np.sort(files)

    flux = []
    conti = []
    flux_err = []

    epsilon = 1e-12

    file_random = self.import_spectrum()
    self.import_table()
    self.import_material()
    load = self.material

    wave = np.array(load["wave"])
    jdb = np.array(self.table["jdb"])
    snr = np.array(self.table["snr"])

    mask_loc = mask_name
    if mask_name is None:
        mask_name = "No name"
        mask_loc = "No name"

    if mask is None:
        mask = self.mask_harps
        mask_name = self.mask_harps

    if isinstance(mask, str):
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

    if mask is not None and isinstance(mask, pd.core.frame.DataFrame):
        dico = mask
        mask = np.array(
            [
                np.array(mask["freq_mask0"]).astype("float"),
                np.array(mask[mask_col]).astype("float"),
            ]
        ).T

    assert isinstance(mask, np.ndarray)
    logging.info("CCF mask selected : %s \n" % (mask_loc))

    if ccf_name is None:
        ccf_name = sub_dico

    if rv_sys is None:
        if file_random["parameters"]["RV_sys"] is not None:
            rv_sys = 1000.0 * file_random["parameters"]["RV_sys"]
        else:
            rv_sys = 0.0
    else:
        rv_sys *= 1000

    assert isinstance(rv_sys, float)

    if not isinstance(rv_shift, np.ndarray):
        if (rv_shift is None) | (rv_shift == False):
            rv_shift = np.zeros(len(files))
        elif rv_shift == True:
            rv_shift = np.array(self.table["rv_shift"]) * 1000

    logging.info("RV sys : %.2f [km/s] \n" % (rv_sys / 1000))

    mask[:, 0] = doppler_r(mask[:, 0], rv_sys)[0]

    all_flux, all_flux_err, conti, conti_err = self.import_sts_flux(
        load=["flux" + kw, "flux_err", sub_dico, "continuum_err"]
    )
    flux, flux_err = util.flux_norm_std(all_flux, all_flux_err, conti + epsilon, conti_err)
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
        norm_factor = np.array(load["master_snr_curve"] * load["ratio_factor_snr"]) ** 2
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
        if plot:
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

    idx2_min = 0
    idx2_max = len(flux)
    if time_min is not None:
        idx2_min = int(find_nearest(jdb, time_min)[0][0])
    if time_max is not None:
        idx2_max = int(find_nearest(jdb, time_max)[0][0] + 1)

    flux = flux[idx2_min:idx2_max, grid_min:grid_max]
    flux_err = flux_err[idx2_min:idx2_max, grid_min:grid_max]
    files = files[idx2_min:idx2_max]
    rv_shift = rv_shift[idx2_min:idx2_max]
    jdb = jdb[idx2_min:idx2_max]
    snr = snr[idx2_min:idx2_max]

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
    logging.info(
        "Computing CCF (Current time %.0fh%.0fm%.0fs) \n" % (now.hour, now.minute, now.second)
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

    vrad, ccf_power, ccf_power_std = ccf_fun(
        log_grid[used_region],
        all_flux[:, used_region],
        log_template[used_region],
        rv_range=rv_range,
        oversampling=ccf_oversampling,
        spec1_std=all_flux_err[:, used_region],
    )  # to compute on all the ccf simultaneously

    now = datetime.datetime.now()
    logging.info(
        "CCF computed (Current time %.0fh%.0fm%.0fs)" % (now.hour, now.minute, now.second)
    )
    logging.info("CCF velocity step : %.0f m/s\n" % (np.median(np.diff(vrad))))

    if not hasattr(self, "all_ccf_saved"):
        self.all_ccf_saved = {}
    self.all_ccf_saved[ccf_name] = (vrad, ccf_power, ccf_power_std)

    ccf_ref = np.median(ccf_power, axis=1)
    continuum_ccf = np.argmax(ccf_ref)
    top_ccf = np.argsort(ccf_ref)[
        -int(len(ccf_ref) / 2) :
    ]  # roughly half of a CCF is made of the continuum

    ccf_snr = 1 / (
        np.std((ccf_power - ccf_ref[:, np.newaxis])[top_ccf], axis=0)
        / np.mean(ccf_power[continuum_ccf])
    )
    logging.info("SNR CCF continuum median : %.0f\n" % (np.median(ccf_snr)))

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

    logging.info("Photon noise RV median : %.2f m/s\n " % (np.median(svrad_phot)))

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

    logging.info(
        "Photon noise RV from calibration : %.2f m/s\n " % (np.median(svrad_phot2["rv"]) * 1000)
    )

    for j, i in enumerate(files):
        if bis_analysis:
            print("File (%.0f/%.0f) %s SNR %.0f reduced" % (j + 1, len(files), i, snr[j]))
        file = pd.read_pickle(i)
        # log_spectrum = interp1d(np.log10(grid), flux[j], kind='cubic', bounds_error=False, fill_value='extrapolate')(log_grid)
        # vrad, ccf_power_old = ccf2(log_grid, log_spectrum,  log_grid_mask, log_mask)
        # vrad, ccf_power_old = ccf_fun(log_grid, log_spectrum, log_template, rv_range=45, oversampling=ccf_oversampling)
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

        save_ccf["ew_std"] = svrad_phot2["ew"][j]
        para_ccf["para_rv_std"] = svrad_phot2["center"][j]
        para_ccf["para_depth_std"] = svrad_phot2["depth"][j]

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
                "vspan_std": bisspan_ccf_std,
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

    rvs_std_backup = np.array(self.table["rv_dace_std"])[idx2_min:idx2_max] / 1000
    rvs_std = svrad_phot2["rv"]
    rvs_std[rvs_std == 0] = rvs_std_backup[rvs_std == 0]

    fwhms = np.array(fwhms).astype("float") * 2.355
    fwhms_std = np.array(fwhms_std).astype("float") * 2.355

    if np.median(fwhms) > (rv_borders / 1.5):
        logging.warn("The CCF is larger than the RV borders for the fit")

    ccf_result = CCFResult(
        rv=tableXY(jdb, np.array(rvs) * 1000, np.array(rvs_std) * 1000),
        centers=tableXY(jdb, np.array(centers) * 1000, np.array(centers_std) * 1000),
        contrast=tableXY(jdb, amplitudes, amplitudes_std),
        depth=tableXY(jdb, depths, depths_std),
        fwhm=tableXY(jdb, fwhms, fwhms_std),
        vspan=tableXY(jdb, np.array(bisspan) * 1000, np.array(bisspan_std) * 1000),
        ew=tableXY(jdb, np.array(ew), np.array(ew_std)),
        bis0=tableXY(jdb, b0s, np.sqrt(2) * np.array(rvs_std) * 1000),
        jdb=jdb,
        rv_photon_noise=svrad_phot2["rv"],
        rv_shift=center,
    )
    ccf_result.rv.rms_w()
    ccf_result.centers.rms_w()

    ccf_infos = {
        "table": ccf_result.infos(),
        "creation_date": datetime.datetime.now().isoformat(),
    }

    if not os.path.exists(self.directory + "Analyse_ccf.p"):
        ccf_summary = {"star_info": {"name": self.starname}}
        io.pickle_dump(ccf_summary, open(self.directory + "/Analyse_ccf.p", "wb"))

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
        self.yarara_ccf_save(mask_name.split(".")[0], ccf_name)

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


def yarara_master_ccf(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    name_ext: str = "",
    rvs: Optional[ndarray] = None,
) -> None:
    self.import_table()

    vrad, ccfs = (self.all_ccf_saved[sub_dico][0], self.all_ccf_saved[sub_dico][1])

    if rvs is None:
        rvs = self.ccf_rv.y.copy()

    med_rv = np.nanmedian(rvs)
    rvs -= med_rv

    new_ccf = []
    for j in range(len(ccfs.T)):
        ccf = tableXY(vrad - rvs[j], ccfs[:, j])
        ccf.interpolate(new_grid=vrad, method="linear", fill_value=np.nan)
        new_ccf.append(ccf.y)
    new_ccf = np.array(new_ccf)
    new_vrad = vrad - med_rv
    stack = np.sum(new_ccf, axis=0)
    stack /= np.nanpercentile(stack, 95)
    half = 0.5 * (1 + np.nanmin(stack))

    master_ccf = tableXY(new_vrad, stack)
    master_ccf.supress_nan()
    master_ccf.interpolate(replace=True, method="cubic")

    new_vrad = master_ccf.x
    stack = master_ccf.y

    v1 = new_vrad[new_vrad < 0][find_nearest(stack[new_vrad < 0], half)[0][0]]
    v2 = new_vrad[new_vrad > 0][find_nearest(stack[new_vrad > 0], half)[0][0]]

    vmin = np.nanmin(new_vrad[~np.isnan(stack)])
    vmax = np.nanmax(new_vrad[~np.isnan(stack)])

    vlim = np.min([abs(vmin), abs(vmax)])
    vmin = -vlim
    vmax = vlim

    contrast = 1 - np.nanmin(stack)

    plt.figure()
    plt.plot(new_vrad, stack, color="k", label="Contrast = %.1f %%" % (100 * contrast))

    extension = ["YARARA", "HARPS", "telluric"][int(name_ext != "") + int(name_ext == "_telluric")]

    if extension == "YARARA":
        self.fwhm = np.round((v2 - v1) / 1000, 2)
        io.pickle_dump(
            {"vrad": new_vrad, "ccf_power": stack},
            open(self.dir_root + "MASTER/MASTER_CCF_KITCAT.p", "wb"),
        )
        try:
            old = pd.read_pickle(self.dir_root + "MASTER/MASTER_CCF_HARPS.p")
            plt.plot(old["vrad"], old["ccf_power"], alpha=0.5, color="k", ls="--")
        except:
            pass
    elif extension == "HARPS":
        io.pickle_dump(
            {"vrad": new_vrad, "ccf_power": stack},
            open(self.dir_root + "MASTER/MASTER_CCF_HARPS.p", "wb"),
        )
    elif extension == "telluric":
        try:
            old = pd.read_pickle(self.dir_root + "MASTER/MASTER_CCF" + name_ext + ".p")
            plt.plot(old["vrad"], old["ccf_power"], alpha=0.5, color="k", ls="--")
        except:
            pass
        io.pickle_dump(
            {"vrad": new_vrad, "ccf_power": stack},
            open(self.dir_root + "MASTER/MASTER_CCF" + name_ext + ".p", "wb"),
        )
    else:
        raise ValueError("Cannot happen")

    plt.xlim(vmin, vmax)

    plt.plot(
        [v1, v2],
        [half, half],
        color="r",
        label="FHWM = %.2f kms" % ((v2 - v1) / 1000),
    )
    plt.scatter([v1, v2], [half, half], color="r", edgecolor="k", zorder=10)
    plt.scatter([0], [np.nanmin(stack)], color="k", edgecolor="k", zorder=10)
    plt.axvline(x=0, ls=":", color="k", alpha=0.5)
    plt.legend()
    plt.grid()
    plt.xlabel("RV [m/s]", fontsize=13)
    plt.ylabel("Flux normalised", fontsize=13)
    plt.title("%s" % (self.starname), fontsize=14)

    self.yarara_star_info(Contrast=[extension, np.round(contrast, 3)])
    self.yarara_star_info(Fwhm=[extension, np.round((v2 - v1) / 1000, 2)])

    plt.savefig(self.dir_root + "IMAGES/MASTER_CCF" + name_ext + ".pdf")
