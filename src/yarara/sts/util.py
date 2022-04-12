from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from colorama import Fore
from numpy import ndarray
from tqdm import tqdm

from .. import io
from ..analysis import tableXY
from ..paths import root
from ..plots import my_colormesh
from ..stats import IQ, find_nearest, flat_clustering, smooth
from ..util import doppler_r, flux_norm_std, print_box

if TYPE_CHECKING:
    from ..my_rassine_tools import spec_time_series


# =============================================================================
# YARARA NONE ZERO FLUX
# =============================================================================


def yarara_non_zero_flux(
    self: spec_time_series, spectrum: Optional[ndarray] = None, min_value: None = None
) -> ndarray:
    file_test = self.import_spectrum()
    hole_left = file_test["parameters"]["hole_left"]
    hole_right = file_test["parameters"]["hole_right"]
    grid = file_test["wave"]
    mask = (grid < hole_left) | (grid > hole_right)

    directory = self.directory

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    if spectrum is None:
        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            flux = file["flux"]
            zero = flux == 0
            if min_value is None:
                min_value = np.min(flux[flux != 0])
            flux[mask & zero] = min_value
            io.pickle_dump(file, open(j, "wb"))
    else:
        print("[INFO] Removing null values of the spectrum")
        zero = spectrum <= 0
        if min_value is None:
            min_value = np.min(spectrum[spectrum > 0])
        spectrum[mask & zero] = min_value
        return spectrum


# =============================================================================
#     CREATE MEDIAN SPECTRUM TELLURIC SUPRESSED
# =============================================================================


def yarara_median_master_backup(
    self: spec_time_series,
    sub_dico: Optional[str] = "matching_diff",
    method: str = "mean",
    continuum: str = "linear",
    supress_telluric: bool = True,
    shift_spectrum: bool = False,
    telluric_tresh: float = 0.001,
    wave_min: int = 5750,
    wave_max: int = 5900,
    jdb_range: List[int] = [-100000, 100000, 1],
    mask_percentile: List[Optional[int]] = [None, 50],
    save: bool = True,
) -> None:
    """
    Produce a median master by masking region of the spectrum

    Parameters
    ----------

    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    telluric_tresh : Treshold used to cover the position of the contaminated wavelength
    wave_min : The minimum xlim axis
    wave_max : The maximum xlim axis

    """

    mask_percentile = [None, 50]

    print_box("\n---- RECIPE : PRODUCE MASTER MEDIAN SPECTRUM ----\n")

    self.import_table()
    self.import_material()
    load = self.material
    epsilon = 1e-6

    planet = self.planet

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    print("\n---- DICO %s used ----" % (sub_dico))

    if self.table["telluric_fwhm"][0] is None:
        fwhm = np.array([3.0] * len(self.table.jdb))
    else:
        try:
            fwhm = np.array(self.table["telluric_fwhm"])
        except:
            fwhm = np.array([3.0] * len(self.table.jdb))

    fwhm_max = [5, 8][self.instrument[:-2] == "CORALIE"]
    fwhm_min = [2, 3][self.instrument[:-2] == "CORALIE"]
    fwhm_default = [3, 5][self.instrument[:-2] == "CORALIE"]
    if np.percentile(fwhm, 95) > fwhm_max:
        logging.warn(
            "[WARNING] FWHM of tellurics larger than %.0f km/s (%.1f), reduced to default value of %.0f km/s"
            % (fwhm_max, np.percentile(fwhm, 95), fwhm_default)
        )
        fwhm = np.array([fwhm_default] * len(self.table.jdb))
    if np.percentile(fwhm, 95) < fwhm_min:
        logging.warn(
            "FWHM of tellurics smaller than %.0f km/s (%.1f), increased to default value of %.0f km/s"
            % (fwhm_min, np.percentile(fwhm, 95), fwhm_default)
        )
        fwhm = np.array([fwhm_default] * len(self.table.jdb))

    print("\n [INFO] FWHM of tellurics : %.1f km/s" % (np.percentile(fwhm, 95)))

    all_flux = []
    all_conti = []
    for i, name in enumerate(np.array(self.table["filename"])):
        file = pd.read_pickle(name)
        self.debug = name
        if not i:
            wavelength = file["wave"]
            self.wave = wavelength

        f = file["flux" + kw]
        f_std = file["flux_err"]
        c = file[sub_dico]["continuum_" + continuum]
        c_std = file["continuum_err"]

        f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)

        all_flux.append(f)
        all_conti.append(c)

    all_flux = np.array(all_flux)
    all_conti = np.array(all_conti)
    all_flux_norm = all_flux / all_conti

    mask = np.ones(len(self.table.jdb)).astype("bool")
    if jdb_range[2]:
        mask = (np.array(self.table.jdb) > jdb_range[0]) & (
            np.array(self.table.jdb) < jdb_range[1]
        )
    else:
        mask = (np.array(self.table.jdb) < jdb_range[0]) | (
            np.array(self.table.jdb) > jdb_range[1]
        )

    if sum(mask) < 40:
        print(
            Fore.YELLOW
            + "\n [WARNING] Not enough spectra %s the specified temporal range"
            % (["inside", "outside"][jdb_range[2] == 0])
            + Fore.RESET
        )
        mask = np.ones(len(self.table.jdb)).astype("bool")
    else:
        print(
            "\n [INFO] %.0f spectra %s the specified temporal range can be used for the median\n"
            % (sum(mask), ["inside", "outside"][jdb_range[2] == 0])
        )

    all_flux = all_flux[mask]
    all_conti = all_conti[mask]
    all_flux_norm = all_flux_norm[mask]

    berv = np.array(self.table["berv" + kw])[mask]
    rv_shift = np.array(self.table["rv_shift"])[mask]
    berv = berv - rv_shift

    model = pd.read_pickle(root + "/Python/Material/model_telluric.p")
    grid = model["wave"]
    spectre = model["flux_norm"]
    telluric = tableXY(grid, spectre)
    telluric.find_min()

    all_min = np.array([telluric.x_min, telluric.y_min, telluric.index_min]).T
    all_min = all_min[1 - all_min[:, 1] > telluric_tresh]
    all_width = np.round(
        all_min[:, 0] * fwhm[:, np.newaxis] / 3e5 / np.median(np.diff(wavelength)),
        0,
    )
    all_width = np.nanpercentile(all_width, 95, axis=0)

    if supress_telluric:
        borders = np.array([all_min[:, 2] - all_width, all_min[:, 2] + all_width]).T
        telluric_mask = flat_clustering(len(grid), borders) != 0
        all_mask2 = []
        for j in tqdm(berv):
            mask = tableXY(doppler_r(grid, j * 1000)[0], telluric_mask, 0 * grid)
            mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
            all_mask2.append(mask.y != 0)
        all_mask2 = np.array(all_mask2).astype("float")
    else:
        all_mask2 = np.zeros(np.shape(all_flux_norm))

    #        i=-1
    #        for j in tqdm(berv):
    #            i+=1
    #            borders = np.array([all_min[:,2]-all_width[i],all_min[:,2]+all_width[i]]).T
    #            telluric_mask = flat_clustering(len(grid),borders)!=0
    #            mask = tableXY(doppler_r(grid,j*1000)[0],telluric_mask)
    #            mask.interpolate(new_grid=wavelength,method='linear')
    #            all_mask2.append(mask.y!=0)
    #         all_mask2 = np.array(all_mask2).astype('float')

    if supress_telluric:
        telluric_mask = telluric.y < (1 - telluric_tresh)
        all_mask = []
        for j in tqdm(berv):
            mask = tableXY(doppler_r(grid, j * 1000)[0], telluric_mask, 0 * grid)
            mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
            all_mask.append(mask.y != 0)
        all_mask = np.array(all_mask).astype("float")
    else:
        all_mask = np.zeros(np.shape(all_flux_norm))

    if shift_spectrum:
        rv_star = np.array(self.table["ccf_rv"])
        rv_star[np.isnan(rv_star)] = np.nanmedian(rv_star)
        rv_star -= np.median(rv_star)
        i = -1
        if method == "median":
            for j in tqdm(rv_star):
                i += 1
                mask = tableXY(
                    doppler_r(wavelength, j * 1000)[1],
                    all_flux_norm[i],
                    0 * wavelength,
                )
                mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                all_flux_norm[i] = mask.y.copy()
        else:
            # print(len(rv))
            # print(np.shape(all_flux))
            for j in tqdm(rv_star):
                i += 1
                mask = tableXY(
                    doppler_r(wavelength, j * 1000)[1],
                    all_flux[i],
                    0 * wavelength,
                )
                mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                all_flux[i] = mask.y.copy()
                mask = tableXY(
                    doppler_r(wavelength, j * 1000)[1],
                    all_conti[i],
                    0 * wavelength,
                )
                mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                all_conti[i] = mask.y.copy()

    # plt.plot(wavelength,np.product(all_mask,axis=0))
    # plt.plot(wavelength,np.product(all_mask2,axis=0))
    print(
        "Percent always contaminated metric1 : %.3f %%"
        % (np.sum(np.product(all_mask, axis=0)) / len(wavelength) * 100)
    )
    print(
        "Percent always contaminated metric2 : %.3f %%"
        % (np.sum(np.product(all_mask2, axis=0)) / len(wavelength) * 100)
    )

    all_mask_nan1 = 1 - all_mask
    all_mask_nan1[all_mask_nan1 == 0] = np.nan

    all_mask_nan2 = 1 - all_mask2
    all_mask_nan2[all_mask_nan2 == 0] = np.nan

    print(mask_percentile)
    if mask_percentile[0] is None:
        mask_percentile[0] = np.ones(len(wavelength)).astype("bool")

    print(
        np.shape(wavelength),
        np.shape(all_flux_norm),
        np.shape(mask_percentile[0]),
        np.shape(mask_percentile[1]),
    )

    med = np.zeros(len(wavelength))
    med[mask_percentile[0]] = np.nanpercentile(
        all_flux_norm[:, mask_percentile[0]], mask_percentile[1], axis=0
    )
    med[~mask_percentile[0]] = np.nanpercentile(all_flux_norm[:, ~mask_percentile[0]], 50, axis=0)

    del mask_percentile

    # med1 = np.nanmedian(all_flux_norm*all_mask_nan1,axis=0)
    # med2 = np.nanmedian(all_flux_norm*all_mask_nan2,axis=0)

    mean = np.nansum(all_flux, axis=0) / np.nansum(all_conti, axis=0)
    mean1 = np.nansum(all_flux * all_mask_nan1, axis=0) / (
        np.nansum(all_conti * all_mask_nan1, axis=0) + epsilon
    )
    mean2 = np.nansum(all_flux * all_mask_nan2, axis=0) / (
        np.nansum(all_conti * all_mask_nan2, axis=0) + epsilon
    )
    mean2[mean2 == 0] = np.nan
    mean1[mean1 == 0] = np.nan
    mean1[mean1 != mean1] = mean2[mean1 != mean1]
    mean1[mean1 != mean1] = med[mean1 != mean1]
    mean2[mean2 != mean2] = mean1[mean2 != mean2]
    # med1[med1!=med1] = mean1[med1!=med1]
    # med2[med2!=med2] = mean1[med2!=med2]
    all_flux_diff_med = all_flux_norm - med
    tresh = 1.5 * IQ(np.ravel(all_flux_diff_med)) + np.nanpercentile(all_flux_diff_med, 75)

    mean1[mean1 > (1 + tresh)] = 1

    if method != "median":
        self.reference = (wavelength, mean1)
    else:
        self.reference = (wavelength, med)

    plt.figure(figsize=(16, 8))
    plt.plot(wavelength, med, color="b", ls="-", label="median")
    plt.plot(wavelength, mean1, color="g", ls="-", label="mean1")
    plt.plot(wavelength, mean2, color="r", ls="-", label="mean2")
    plt.legend(loc=2)
    # plt.plot(wavelength,med,color='b',ls='-.')
    # plt.plot(wavelength,med1,color='g',ls='-.')
    # plt.plot(wavelength,med2,color='r',ls='-.')

    all_flux_diff_mean = all_flux_norm - mean
    all_flux_diff_med = all_flux_norm - med
    all_flux_diff1_mean = all_flux_norm - mean1
    # all_flux_diff1_med = all_flux_norm - med1
    # all_flux_diff2_mean = all_flux_norm - mean2
    # all_flux_diff2_med = all_flux_norm - med2

    idx_min = int(find_nearest(wavelength, wave_min)[0])
    idx_max = int(find_nearest(wavelength, wave_max)[0])

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 2)
    plt.imshow(
        all_flux_diff_mean[::-1, idx_min:idx_max],
        aspect="auto",
        vmin=-0.005,
        vmax=0.005,
        cmap="plasma",
    )
    plt.imshow(all_mask2[::-1, idx_min:idx_max], aspect="auto", alpha=0.2, cmap="Reds")
    ax = plt.gca()
    plt.subplot(2, 1, 1, sharex=ax, sharey=ax)
    plt.imshow(
        all_flux_diff_mean[::-1, idx_min:idx_max],
        aspect="auto",
        vmin=-0.005,
        vmax=0.005,
        cmap="plasma",
    )
    plt.imshow(all_mask[::-1, idx_min:idx_max], aspect="auto", alpha=0.2, cmap="Reds")

    if len(berv) > 15:
        plt.figure(figsize=(16, 8))
        plt.subplot(2, 1, 1)

        plt.title("Median")
        my_colormesh(
            wavelength[idx_min : idx_max + 1],
            np.arange(len(all_mask)),
            all_flux_diff_med[:, idx_min : idx_max + 1],
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )
        ax = plt.gca()

        plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
        plt.title("Masked mean")
        my_colormesh(
            wavelength[idx_min : idx_max + 1],
            np.arange(len(all_mask)),
            all_flux_diff1_mean[:, idx_min : idx_max + 1],
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )

    load["wave"] = wavelength
    if method != "median":
        load["reference_spectrum"] = mean1
    else:
        load["reference_spectrum"] = med - np.median(all_flux_diff_med)

    ref = np.array(load["reference_spectrum"])
    ref = self.yarara_non_zero_flux(spectrum=ref, min_value=None)
    load["reference_spectrum"] = ref

    load.loc[load["reference_spectrum"] < 0, "reference_spectrum"] = 0

    if save:
        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))
    else:
        return np.array(load["reference_spectrum"])


def yarara_median_master(
    self: spec_time_series,
    sub_dico: Optional[str] = "matching_diff",
    continuum: str = "linear",
    method: str = "max",
    smooth_box: int = 7,
    supress_telluric: bool = True,
    shift_spectrum: bool = False,
    wave_min: int = 5750,
    wave_max: int = 5900,
    bin_berv: int = 10,
    bin_snr: Optional[int] = None,
    telluric_tresh: float = 0.001,
    jdb_range: List[int] = [-100000, 100000, 1],
    mask_percentile: List[Optional[int]] = [None, 50],
    save: bool = True,
) -> None:
    """
    Produce a median master by masking region of the spectrum

    Parameters
    ----------

    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    telluric_tresh : Treshold used to cover the position of the contaminated wavelength
    wave_min : The minimum xlim axis
    wave_max : The maximum xlim axis

    """
    if method:
        self.yarara_median_master_backup(
            sub_dico=sub_dico,
            method=method,
            continuum=continuum,
            telluric_tresh=telluric_tresh,
            wave_min=wave_min,
            wave_max=wave_max,
            jdb_range=jdb_range,
            supress_telluric=supress_telluric,
            shift_spectrum=shift_spectrum,
            mask_percentile=mask_percentile,
            save=save,
        )

    if method == "max":
        print_box("\n---- RECIPE : PRODUCE MASTER MAX SPECTRUM ----\n")

        self.import_table()
        self.import_material()
        load = self.material
        tab = self.table
        planet = self.planet

        epsilon = 1e-12

        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        if sub_dico is None:
            sub_dico = self.dico_actif
        print("---- DICO %s used ----" % (sub_dico))

        all_flux = []
        all_conti = []
        snr = np.array(tab["snr"])
        for i, name in enumerate(np.array(tab["filename"])):
            file = pd.read_pickle(name)
            if not i:
                wavelength = file["wave"]
                self.wave = wavelength

            f = file["flux" + kw]
            f_std = file["flux_err"]
            c = file[sub_dico]["continuum_" + continuum]
            c_std = file["continuum_err"]

            f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)

            all_flux.append(f_norm)
            all_conti.append(file["matching_diff"]["continuum_" + continuum])

        all_conti = np.array(all_conti)
        all_flux = np.array(all_flux) * all_conti.copy()
        all_flux_backup = all_flux / all_conti
        med = np.nanmedian(all_flux_backup, axis=0)

        berv = np.array(tab["berv"])
        rv_shift = np.array(tab["rv_shift"])
        berv = berv - rv_shift

        sort = np.argsort(berv)
        snr = np.array(tab["snr"])
        snr_sorted = snr[sort]

        if bin_snr is not None:
            val = int(np.sum(snr_sorted**2) // (bin_snr**2))
            if val > 4:
                bin_berv = val
            else:
                print(
                    "The expected SNR cannot be reached since the total SNR is about : %.0f"
                    % (np.sqrt(np.sum(snr_sorted**2)))
                )
                print("The maximum value allowed : %.0f" % (np.sqrt(np.sum(snr_sorted**2) / 5)))
                bin_snr = np.sqrt(np.sum(snr_sorted**2) / 5) - 50
                bin_berv = int(np.sum(snr_sorted**2) // (bin_snr**2))

        # plt.plot(np.sqrt(np.cumsum(snr_sorted**2)))
        snr_lim = np.linspace(0, np.sum(snr_sorted**2), bin_berv + 1)
        berv_bin = berv[sort][find_nearest(np.cumsum(snr_sorted**2), snr_lim)[0]]
        berv_bin[0] -= 1  # to ensure the first point to be selected

        mask_bin = (berv > berv_bin[0:-1][:, np.newaxis]) & (berv <= berv_bin[1:][:, np.newaxis])
        berv_bin = (
            berv_bin[:-1][np.sum(mask_bin, axis=1) != 0]
            + np.diff(berv_bin)[np.sum(mask_bin, axis=1) != 0] / 2
        )
        mask_bin = mask_bin[np.sum(mask_bin, axis=1) != 0]

        snr_stacked = []
        all_flux_norm = []
        all_snr = []
        for j in range(len(mask_bin)):
            all_flux_norm.append(
                np.sum(all_flux[mask_bin[j]], axis=0)
                / (np.sum(all_conti[mask_bin[j]], axis=0) + epsilon)
            )
            snr_stacked.append(np.sqrt(np.sum((snr[mask_bin[j]]) ** 2)))
            all_snr.append(snr[mask_bin[j]])
        all_flux_norm = np.array(all_flux_norm)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        self.yarara_get_berv_value(0, Draw=True, new=False, save_fig=False, light_graphic=True)
        ax = plt.gca()
        for j in berv_bin:
            plt.axhline(y=j, color="k", alpha=0.2)
        plt.axhline(y=0, color="k", ls=":")

        plt.subplot(1, 2, 2, sharey=ax)
        plt.axhline(y=0, color="k", ls=":")

        plt.plot(snr_stacked, berv_bin, "bo-", alpha=0.3)
        curve = tableXY(snr_stacked, berv_bin)
        curve.myscatter(
            num=False,
            liste=[len(all_snr[j]) for j in range(len(all_snr))],
            color="k",
            factor=50,
        )
        plt.xlabel("SNR stacked", fontsize=13)
        plt.ylabel("BERV [km/s]", fontsize=13)

        print("SNR of binned spetcra around %.0f" % (np.mean(snr_stacked)))

        for j in range(len(all_flux_norm)):
            all_flux_norm[j] = smooth(all_flux_norm[j], shape="savgol", box_pts=smooth_box)

        mean1 = np.max(all_flux_norm, axis=0)

        self.reference_max = (wavelength, mean1)

        mean1 -= np.median(mean1 - self.reference[1])

        all_flux_diff_mean = all_flux_backup - mean1
        all_flux_diff_med = all_flux_backup - med
        all_flux_diff1_mean = all_flux_backup - self.reference[1]

        idx_min = int(find_nearest(wavelength, wave_min)[0])
        idx_max = int(find_nearest(wavelength, wave_max)[0])

        plt.figure(figsize=(16, 8))
        plt.subplot(3, 1, 1)

        plt.title("Median")
        my_colormesh(
            wavelength[idx_min : idx_max + 1],
            np.arange(len(berv)),
            all_flux_diff_med[sort][:, idx_min : idx_max + 1],
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )
        ax = plt.gca()

        plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
        plt.title("Max")
        my_colormesh(
            wavelength[idx_min : idx_max + 1],
            np.arange(len(berv)),
            all_flux_diff_mean[sort][:, idx_min : idx_max + 1],
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )

        plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
        plt.title("Masked weighted mean")
        my_colormesh(
            wavelength[idx_min : idx_max + 1],
            np.arange(len(berv)),
            all_flux_diff1_mean[sort][:, idx_min : idx_max + 1],
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )

        load["wave"] = wavelength
        load["reference_spectrum"] = mean1
        io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))


# =============================================================================
# CUT SPECTRUM
# =============================================================================


def yarara_cut_spectrum(
    self: spec_time_series, wave_min: None = None, wave_max: Optional[int] = None
) -> None:
    """Cut the spectrum time-series borders to reach the specified wavelength limits (included)
    There is no way to cancel this step ! Use it wisely."""

    print_box("\n---- RECIPE : SPECTRA CROPING ----\n")

    directory = self.directory
    self.import_material()
    load = self.material

    if wave_min == "auto":
        w0 = np.min(load["wave"])
        a, b, c = self.yarara_map(
            "matching_diff",
            reference="norm",
            wave_min=w0,
            wave_max=w0 + 1000,
            Plot=False,
        )
        a = a[np.sum(a < 1e-5, axis=1) > 100]  # only kept spectra with more than 10 values below 0

        if len(a):
            t = np.nanmean(np.cumsum(a < 1e-5, axis=1).T / np.sum(a < 1e-5, axis=1), axis=1)
            i0 = find_nearest(t, 0.99)[0][0]  # supress wavelength range until 99% of nan values
            if i0:
                wave_min = load["wave"][i0]
                print(
                    " [INFO] Automatic detection of the blue edge spectrum found at %.2f AA\n"
                    % (wave_min)
                )
            else:
                wave_min = None
        else:
            wave_min = None

    maps = glob.glob(self.dir_root + "CORRECTION_MAP/*.p")
    if len(maps):
        for name in maps:
            correction_map = pd.read_pickle(name)
            old_wave = correction_map["wave"]
            length = len(old_wave)
            idx_min = 0
            idx_max = len(old_wave)
            if wave_min is not None:
                idx_min = int(find_nearest(old_wave, wave_min)[0])
            if wave_max is not None:
                idx_max = int(find_nearest(old_wave, wave_max)[0] + 1)
            correction_map["wave"] = old_wave[idx_min:idx_max]
            correction_map["correction_map"] = correction_map["correction_map"][:, idx_min:idx_max]
            io.pickle_dump(correction_map, open(name, "wb"))
            print("%s modified" % (name.split("/")[-1]))

    time.sleep(1)

    old_wave = np.array(load["wave"])
    length = len(old_wave)
    idx_min = 0
    idx_max = len(old_wave)
    if wave_min is not None:
        idx_min = int(find_nearest(old_wave, wave_min)[0])
    if wave_max is not None:
        idx_max = int(find_nearest(old_wave, wave_max)[0] + 1)

    new_wave = old_wave[idx_min:idx_max]
    wave_min = np.min(new_wave)
    wave_max = np.max(new_wave)

    load = load[idx_min:idx_max]
    load = load.reset_index(drop=True)
    io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    try:
        file_kitcat = pd.read_pickle(self.dir_root + "KITCAT/kitcat_spectrum.p")
        wave_kit = file_kitcat["wave"]
        idx_min = 0
        idx_max = len(wave_kit)
        if wave_min is not None:
            idx_min = int(find_nearest(wave_kit, wave_min)[0])
        if wave_max is not None:
            idx_max = int(find_nearest(wave_kit, wave_max)[0] + 1)
        for kw in [
            "wave",
            "flux",
            "correction_factor",
            "flux_telluric",
            "flux_uncorrected",
            "continuum",
        ]:
            file_kitcat[kw] = np.array(file_kitcat[kw])[idx_min:idx_max]
        io.pickle_dump(file_kitcat, open(self.dir_root + "KITCAT/kitcat_spectrum.p", "wb"))
    except:
        pass

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    file_ref = self.import_spectrum()
    old_wave = np.array(file_ref["wave"])
    length = len(old_wave)
    idx_min = 0
    idx_max = len(old_wave)
    if wave_min is not None:
        idx_min = int(find_nearest(old_wave, wave_min)[0])
    if wave_max is not None:
        idx_max = int(find_nearest(old_wave, wave_max)[0] + 1)

    new_wave = old_wave[idx_min:idx_max]
    wave_min = np.min(new_wave)
    wave_max = np.max(new_wave)

    for j in tqdm(files):
        file = pd.read_pickle(j)
        file["parameters"]["wave_min"] = wave_min
        file["parameters"]["wave_max"] = wave_max

        anchors_wave = file["matching_anchors"]["anchor_wave"]
        mask = (anchors_wave >= wave_min) & (anchors_wave <= wave_max)
        file["matching_anchors"]["anchor_index"] = (
            file["matching_anchors"]["anchor_index"][mask] - idx_min
        )
        file["matching_anchors"]["anchor_flux"] = file["matching_anchors"]["anchor_flux"][mask]
        file["matching_anchors"]["anchor_wave"] = file["matching_anchors"]["anchor_wave"][mask]

        anchors_wave = file["output"]["anchor_wave"]
        mask = (anchors_wave >= wave_min) & (anchors_wave <= wave_max)
        file["output"]["anchor_index"] = file["output"]["anchor_index"][mask] - idx_min
        file["output"]["anchor_flux"] = file["output"]["anchor_flux"][mask]
        file["output"]["anchor_wave"] = file["output"]["anchor_wave"][mask]

        fields = file.keys()
        for field in fields:
            if type(file[field]) == dict:
                sub_fields = file[field].keys()
                for sfield in sub_fields:
                    if type(file[field][sfield]) == np.ndarray:
                        if len(file[field][sfield]) == length:
                            file[field][sfield] = file[field][sfield][idx_min:idx_max]
            elif type(file[field]) == np.ndarray:
                if len(file[field]) == length:
                    file[field] = file[field][idx_min:idx_max]
        io.save_pickle(j, file)


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
