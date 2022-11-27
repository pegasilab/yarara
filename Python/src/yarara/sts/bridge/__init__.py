from __future__ import annotations

import glob
import logging
import os
import time
from typing import TYPE_CHECKING, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from scipy.interpolate import interp1d
from tqdm import tqdm

from ... import analysis
from ...analysis import tableXY
from ...io import pickle_dump, save_pickle, touch_pickle
from ...paths import root
from ...plots import plot_color_box, plot_copy_time, transit_draw
from ...stats import IQ, rm_outliers
from ...stats.misc import mad
from ...stats.nearest import find_nearest
from ...util import doppler_r, flux_norm_std, map_rnr

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_simbad_query(self: spec_time_series, starname=None) -> None:
    """
    Reads information about the star from the SIMBAD table and populates the Stellar_info_ pickle
    """
    self.import_star_info()
    if starname is None:
        starname = self.starname

    table_simbad = touch_pickle(root + "/Python/database/SIMBAD/table_stars.p")

    dico = self.star_info

    if starname in table_simbad.keys():
        logging.info("Star found in the SIMBAD table")
        dico2 = table_simbad[starname]

        for kw in dico2.keys():
            if isinstance(dico2[kw], dict):
                for kw2 in dico2[kw].keys():
                    dico[kw][kw2] = dico2[kw][kw2]

    pickle_dump(
        dico,
        open(self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p", "wb"),
    )

    sp = dico["Sp_type"]["fixed"]
    self.mask_harps = ["G2", "K5", "M2"][int((sp[0] == "K") | (sp[0] == "M")) + int(sp[0] == "M")]

    del dico["Name"]

    print("\nValue after SIMBAD query")
    print("----------------\n")

    dataframe = pd.DataFrame(
        [dico[i]["fixed"] for i in dico.keys()],
        index=[i for i in dico.keys()],
        columns=[self.starname],
    )
    print(dataframe)


def flux_error(
    self: spec_time_series, ron=11
) -> None:  # temporary solution because no error with harps s1d
    """
    Produce flux errors on the pickles files defined as the ron plus square root of flux

    Args:
        ron : Read-out-Noise value of the reading detector errors

    """

    self.import_material()
    file = self.import_spectrum()
    correction = np.array(self.material["blaze_correction"])

    directory = self.directory
    files = glob.glob(directory + "RASSI*.p")
    for j in tqdm(files):
        file = pd.read_pickle(j)
        print(j)
        if not np.median(abs(file["flux_err"])):
            file["flux_err"] = ron + np.sqrt(abs(file["flux"])) * correction
            save_pickle(j, file)


def continuum_error(self: spec_time_series):
    """
    Produce error on the continuum by interpolating the error flux value at anchor points

    Parameters
    ----------

    Returns
    -------

    """
    directory = self.directory
    files = glob.glob(directory + "RASSI*.p")
    for j in tqdm(files):
        file = pd.read_pickle(j)
        vec = tableXY(
            file["matching_anchors"]["anchor_wave"],
            file["flux_err"][file["matching_anchors"]["anchor_index"]],
        )
        vec.interpolate(new_grid=file["wave"], method="linear", interpolate_x=False)
        file["continuum_err"] = vec.yerr
        save_pickle(j, file)


def yarara_check_rv_sys(self: spec_time_series):
    self.import_star_info()
    self.import_table()
    file_test: tableXY = self.spectrum(num=int(np.argmax(self.table.snr)), norm=True)

    rv_sys2 = self.import_spectrum(num=0)["parameters"]["RV_sys"]
    rv_sys1 = self.star_info["Rv_sys"]["fixed"]

    if 2 * abs(rv_sys2 - rv_sys1) / abs(rv_sys1 + rv_sys2 + 1e-6) * 100 > 20:
        logging.warning(
            "RV_sys incompatible between star_info (%.1f) and RASSINE output (%.1f)",
            rv_sys1,
            rv_sys2,
        )
        mask = np.genfromtxt(root + "/Python/MASK_CCF/" + self.mask_harps + ".txt")
        mask = np.array([0.5 * (mask[:, 0] + mask[:, 1]), mask[:, 2]]).T

        rv_range = [15, self.star_info["FWHM"]["fixed"]][int(self.star_info["FWHM"]["fixed"] > 15)]

        file_test.ccf(
            mask,
            weighted=True,
            rv_range=rv_range + max(abs(rv_sys1), abs(rv_sys2)),
        )

        rv_sys_fit = file_test.ccf_params["cen"].value
        file_test.ccf(mask, weighted=True, rv_range=rv_range * 1.5, rv_sys=rv_sys_fit)

        rv_sys_fit += file_test.ccf_params["cen"].value

        rv_sys_fit = np.round(rv_sys_fit / 1000, 2)
        contrast_fit = 100 * (abs(file_test.ccf_params["amp"].value))

        y_min = 1 - 2 * contrast_fit / 100
        if y_min < 0:
            y_min = 0
        plt.ylim(y_min, 1.1)

        logging.info("RV_sys value fitted as %.2f", rv_sys_fit)

        plt.savefig(self.dir_root + "IMAGES/RV_sys_fitting.pdf")

        self.yarara_star_info(Rv_sys=["fixed", rv_sys_fit])
        self.yarara_star_info(Contrast=["fixed", np.round(contrast_fit / 100, 3)])
        df = pd.DataFrame(np.ones(len(self.table)) * rv_sys_fit, columns=["RV_sys"])
        self.yarara_obs_info(df)
    else:
        logging.info("Both RV_sys match : %.2f/%.2f kms", rv_sys1, rv_sys2)


def yarara_check_fwhm(self: spec_time_series, delta_window: int = 5):

    warning_rv_borders = True
    iteration = -1
    while warning_rv_borders:
        iteration += 1
        output = self.yarara_ccf(
            mask_name=self.mask_harps,
            mask=self.read_ccf_mask(self.mask_harps),
            plot=False,
            save=False,
            sub_dico="matching_diff",
            ccf_oversampling=1,
            rv_range=None,
            rv_borders=None,
            delta_window=delta_window,
        )

        new_fwhm = np.nanmedian(output["fwhm"].y)

        warning_rv_borders = self.warning_rv_borders
        if iteration == 4:
            warning_rv_borders = True
            new_fwhm = 6
            logging.warning("The algorithm does not found a proper FWHM")

        self.fwhm = np.round(new_fwhm, 1)

    self.yarara_star_info(Fwhm=["YARARA", np.round(self.fwhm, 2)])

    logging.info("FWHM measured as %.1f kms", self.fwhm)


def yarara_correct_secular_acc(self: spec_time_series, update_rv: bool = False):
    """
    Compute the secular drift and correction the fluxes to cancel the drift

    Parameters
    ----------

    update_rv : True/False to update the flux value in order to cancel the secular drift

    """

    directory = self.directory
    self.import_table()
    self.import_star_info()

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    dist = self.star_info["Dist_pc"]["fixed"]
    pma = self.star_info["Pma"]["fixed"]
    pmd = self.star_info["Pmd"]["fixed"]

    distance_m = dist * 3.08567758e16
    mu_radps = (
        np.sqrt(pma**2 + pmd**2) * 2 * np.pi / (360.0 * 1000.0 * 3600.0 * 86400.0 * 365.25)
    )
    acc_sec = distance_m * 86400.0 * mu_radps**2  # rv secular drift in m/s per days

    # file_random = self.import_spectrum()
    all_acc_sec = []
    all_rv_sec = []
    for j in tqdm(files):
        file = pd.read_pickle(j)
        # acc_sec = file_random['parameters']['acc_sec']
        all_acc_sec.append(acc_sec)
        file["parameters"]["RV_sec"] = file["parameters"]["jdb"] * acc_sec
        all_rv_sec.append(file["parameters"]["RV_sec"])
        save_pickle(j, file)
    all_rv = np.array(all_rv_sec)

    plt.figure()
    plt.scatter(
        self.table["jdb"],
        (self.table["ccf_rv"] - np.median(self.table["ccf_rv"])) * 1000,
    )
    plt.plot(self.table["jdb"], all_rv - np.median(all_rv), color="k")

    if update_rv:
        print("Modification of the files to cancel the secular acceleration")
        time.sleep(1)
        all_rv -= np.median(all_rv)
        i = -1
        for j in tqdm(files):
            i += 1
            file = pd.read_pickle(j)
            file["parameters"]["RV_sec"] = all_rv[i]
            try:
                flux = file["flux_backup"]
            except KeyError:
                flux = file["flux"]
            wave = file["wave"]
            file["flux_backup"] = flux.copy()
            if all_rv[i]:
                flux_shifted = interp1d(
                    doppler_r(wave, all_rv[i])[1],
                    flux,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(wave)
                file["flux"] = flux_shifted
            save_pickle(j, file)

    self.yarara_analyse_summary()


def scale_cmap(self: spec_time_series):
    a, b, c = self.yarara_map(
        reference="median",
        sub_dico="matching_diff",
        wave_min=3000.0,
        wave_max=5000.0,
        plot=False,
    )
    med = np.nanmedian(b)
    Q3 = np.nanpercentile(b, 75)
    Q1 = np.nanpercentile(b, 25)
    clim = np.max([Q3 - med, med - Q1]) + 1.5 * (Q3 - Q1)
    print(" [INFO] Zscale color scaled to : %.3f" % (clim))
    self.low_cmap = -clim
    self.high_cmap = clim


def suppress_low_snr_spectra(self: spec_time_series, suppress: bool = False):
    """
    Suppress spectra under fixed treshold in SNR

    Args:
        suppress : True/False to delete of simply hide the files

    """
    snr = tableXY(self.table.jdb, self.table.snr)

    lim_inf = np.nanpercentile(snr.y, 50) - 1.5 * IQ(snr.y)

    for lim in [100, 75, 50, 35, 20]:
        if (lim_inf < lim) & (np.nanpercentile(snr.y, 16) > lim):
            lim_inf = lim
            break

    if lim_inf < 0:
        lim_inf = 35

    logging.info("Spectra under SNR %.0f suppressed", lim_inf)

    files_to_process = np.sort(glob.glob(self.directory + "R*.p"))
    mask = np.zeros(len(files_to_process))
    self.import_table()
    mask = np.array(self.table["snr"]) < lim_inf

    self.suppress_time_spectra(mask=mask, suppress=suppress)
    os.system("rm " + self.directory + "Analyse_summary.p")
    os.system("rm " + self.directory + "Analyse_summary.csv")
    self.yarara_analyse_summary()
    mask = mask.astype("bool")
    self.suppress_time_RV(mask)


def yarara_suppress_doubtful_spectra(self: spec_time_series, suppress=False):
    """
    Suppress outlier spectra


    Args:
        suppress: Whether to hide or remove the file. Defaults to False.
    """
    file_test = self.import_spectrum()
    grid = file_test["wave"]
    maps, _, _ = self.yarara_map(
        wave_min=np.min(grid), wave_max=np.max(grid), reference="zeros", plot=False
    )

    nb_out = np.sum(maps < 0.05, axis=1).astype("float")
    non_outliers_mask, _ = rm_outliers(nb_out, m=5, kind="inter")
    outliers_mask = ~non_outliers_mask

    if np.any(outliers_mask):
        logging.info(
            "%.0f Spectra are doubtful and will be deleted: %s",
            np.sum(outliers_mask),
            np.where(outliers_mask),
        )
        self.suppress_time_spectra(mask=outliers_mask, suppress=suppress)


# def suppress_time_spectra(
#     self: spec_time_series,
#     liste=None,
#     jdb_min=None,
#     jdb_max=None,
#     num_min=None,
#     num_max=None,
#     suppress=False,
#     name_ext="temp",
# ):
#     """
#     Suppress spectra according to time

#     Args:
#         jdb or num are both inclusive in lower and upper limit
#         snr_cutoff: treshold value of the cutoff below which spectra are removed
#         suppress: True/False to delete of simply hide the files

#     """

#     self.import_table()
#     jdb = self.table["jdb"]
#     name = self.table["filename"]
#     directory = "/".join(np.array(name)[0].split("/")[:-1])

#     if num_min is not None:
#         jdb_min = jdb[num_min]

#     if num_max is not None:
#         jdb_max = jdb[num_max]

#     if jdb_min is None:
#         jdb_min = jdb_max

#     if jdb_max is None:
#         jdb_max = jdb_min

#     if (num_min is None) & (num_max is None) & (jdb_min is None) & (jdb_max is None):
#         jdb_min = 1e9
#         jdb_max = -1e9

#     if liste is None:
#         mask = np.array((jdb >= jdb_min) & (jdb <= jdb_max))
#     else:
#         if type(liste[0]) == np.bool_:
#             mask = liste
#         else:
#             mask = np.in1d(np.arange(len(jdb)), liste)

#     if sum(mask):
#         idx = np.arange(len(jdb))[mask]
#         print(" [INFO] Following spectrum indices will be suppressed : ", idx)
#         print(" [INFO] Number of spectrum suppressed : %.0f \n" % (sum(mask)))
#         maps = glob.glob(self.dir_root + "CORRECTION_MAP/*.npy")
#         if len(maps):
#             for names in maps:
#                 correction_map = np.load(names)
#                 correction_map = np.delete(correction_map, idx, axis=0)
#                 np.save(names, correction_map.astype("float32"))
#                 print("%s modified" % (names.split("/")[-1]))

#         name = name[mask]

#         maps = glob.glob(self.directory + "FLUX/*.npy")
#         if len(maps):
#             for names in maps:
#                 flux_map = np.load(names)
#                 flux_map = np.delete(flux_map, idx, axis=0)
#                 np.save(names, flux_map.astype("float32"))
#                 print("%s modified" % (names.split("/")[-1]))

#         maps = glob.glob(self.directory + "CONTINUUM/*.npy")
#         if len(maps):
#             for names in maps:
#                 continuum_map = np.load(names)
#                 continuum_map = np.delete(continuum_map, idx, axis=0)
#                 np.save(names, continuum_map.astype("float32"))
#                 print("%s modified" % (names.split("/")[-1]))

#         files_to_process = np.sort(np.array(name))
#         for j in files_to_process:
#             if suppress:
#                 print("File deleted : %s " % (j))
#                 os.system("rm " + j)
#             else:
#                 new_name = name_ext + "_" + j.split("/")[-1]
#                 print("File renamed : %s " % (directory + "/" + new_name))
#                 os.system("mv " + j + " " + directory + "/" + new_name)

#         os.system("rm " + directory + "/Analyse_summary.p")
#         os.system("rm " + directory + "/Analyse_summary.csv")
#         self.yarara_analyse_summary()

#         self.suppress_time_RV(mask)


def suppress_time_RV(self: spec_time_series, liste):

    self.import_ccf()

    if sum(liste):
        mask = list(self.table_ccf.keys())[1:]
        try:
            for m in mask:
                for d in self.table_ccf[m].keys():
                    self.table_ccf[m][d]["table"] = self.table_ccf[m][d]["table"][
                        ~liste
                    ].reset_index(drop=True)

            pickle_dump(self.table_ccf, open(self.directory + "Analyse_ccf.p", "wb"))

            logging.info("CCF table modifed")
        except:
            logging.error("CCF cannot be modified")


def yarara_map_1d_to_2d(self: spec_time_series, instrument="HARPS03"):
    self.import_material()
    mat = self.material
    wave = np.array(mat["wave"])
    wave_matrix = fits.open(root + "/Python/Material/" + instrument + "_WAVE_MATRIX_A.fits")[
        0
    ].data

    jdb = (
        fits.open(root + "/Python/Material/" + instrument + "_WAVE_MATRIX_A.fits")[0].header[
            "MJD-OBS"
        ]
        + 0.5
    )
    berv_file = self.yarara_get_berv_value(jdb)
    shape = np.shape(wave_matrix)
    wave_matrix = np.reshape(doppler_r(np.ravel(wave_matrix), 0 * berv_file * 1000)[0], shape)
    dim1, dim2 = np.shape(wave_matrix)
    dim1 += 1
    dim2 += 1

    try:
        blaze = fits.open(root + "/Python/Material/" + instrument + "_BLAZE.fits")[0].data
    except:
        blaze = 0 * wave_matrix

    mapping_pixels = np.zeros((len(wave), len(wave_matrix)))
    mapping_orders = np.zeros((len(wave), len(wave_matrix)))
    mapping_blaze = np.zeros((len(wave), len(wave_matrix)))

    index = np.arange(len(wave))
    for order in tqdm(range(len(wave_matrix))):
        index_cut = index[
            (wave > np.min(wave_matrix[order])) & (wave < np.max(wave_matrix[order]))
        ]
        wave_cut = wave[index_cut]
        match = find_nearest(wave_matrix[order], wave_cut)
        mapping_orders[index_cut, order] = order + 1
        mapping_pixels[index_cut, order] = match[0] + 1
        mapping_blaze[index_cut, order] = blaze[order, match[0]]

    max_overlapping = np.max(np.sum(mapping_orders != 0, axis=1))
    logging.info("Maximum %.0f orders overlapped", max_overlapping)

    sort = np.argsort(mapping_pixels, axis=1)[:, ::-1]
    map_pix = np.array([mapping_pixels[i, sort[i]][0:max_overlapping] for i in range(len(sort))])
    map_ord = np.array([mapping_orders[i, sort[i]][0:max_overlapping] for i in range(len(sort))])
    map_blaze = np.array([mapping_blaze[i, sort[i]][0:max_overlapping] for i in range(len(sort))])
    blaze_correction = np.sqrt(map_blaze[:, 0] + map_blaze[:, 1])
    blaze_correction[blaze_correction == 0] = 1
    blaze_correction = 1 / np.sqrt(blaze_correction)

    zone_merged = np.sum(map_pix != 0, axis=1) - 1

    coded_pixels = np.zeros(len(wave))
    coded_orders = np.zeros(len(wave))

    logging.info("Encoding pixels and orders by R%.0f to R mapping\n", max_overlapping)
    time.sleep(1)

    for i in tqdm(range(len(wave))):
        coded_pixels[i] = map_rnr(map_pix[i], val_max=dim2, n=max_overlapping)
        coded_orders[i] = map_rnr(map_ord[i], val_max=dim1, n=max_overlapping)

    mat["pixels_rnr"] = coded_pixels
    mat["orders_rnr"] = coded_orders
    mat["merged"] = zone_merged
    mat["blaze_correction"] = blaze_correction

    pickle_dump(mat, open(self.directory + "Analyse_material.p", "wb"))


def yarara_flux_constant(self: spec_time_series):
    """
    Compute the flux constant which allow to preserve the bolometric integrated flux
    """

    directory = self.directory

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    all_flux = []
    snr = []

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        all_flux.append(file["flux"])
        snr.append(file["parameters"]["SNR_5500"])

    all_flux = np.array(all_flux)
    snr = np.array(snr)

    constants = np.sum(all_flux, axis=1)
    constants = constants / constants[snr.argmax()]

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        file["parameters"]["flux_balance"] = constants[i]
        pickle_dump(file, open(j, "wb"))

    self.yarara_analyse_summary()


def yarara_color_template(
    self: spec_time_series,
    sub_dico: str = "matching_anchors",
    continuum: Literal["linear"] = "linear",
):
    """
    Define the color template used in the weighting of the lines for the CCF.
    The color is defined as the matching_anchors continuum of the best SNR spectra.
    The product is saved in the Material file.

    Args:
        sub_dico : The sub_dictionnary used to  select the continuum
        continuum : The continuum to select (can only be linear)
    """

    self.import_table()
    self.import_material()
    load = self.material
    tab = self.table
    snr = np.array(tab["snr"]).argmax()

    file_ref = self.import_spectrum(num=int(snr))
    wave = file_ref["wave"]
    continuum_ref = file_ref[sub_dico]["continuum_" + continuum]

    load["wave"] = wave
    load["color_template"] = continuum_ref
    load["master_snr_curve"] = np.sqrt(
        continuum_ref
    )  # assuming flux units are in photon noise units
    pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))


def yarara_berv_summary(
    self: spec_time_series,
    sub_dico: Optional[str] = "matching_diff",
    dbin_berv: float = 0.3,
    nb_plot=3,
    telluric_fwhm: float = 3.5,
):
    """
    Produce a berv summary

    Parameters
    ----------

    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    telluric_tresh : Treshold used to cover the position of the contaminated wavelength
    wave_min : The minimum xlim axis
    wave_max : The maximum xlim axis

    """

    logging.info("\n---- RECIPE : PRODUCE BERV SUMMARY ----\n")

    self.import_table()
    self.import_material()
    tab = self.table

    if sub_dico is None:
        sub_dico = self.dico_actif
    logging.info("---- DICO %s used ----" % (sub_dico))

    snr = np.array(tab["snr"])
    fluxes, fluxes_std, all_conti, all_conti_err = self.import_sts_flux(
        load=["flux", "flux_err", sub_dico, "continuum_err"]
    )
    all_flux, all_flux_std = flux_norm_std(fluxes, fluxes_std, all_conti, all_conti_err)

    all_flux = np.array(all_flux) * all_conti.copy()

    berv = np.array(tab["berv"])
    rv_shift = np.array(tab["rv_shift"])
    berv = berv - rv_shift

    window = np.arange(-350, 351, 1)
    window = np.where(
        abs(window) < telluric_fwhm * 10 / 2, 1, 0
    )  # compute the contamination for the berv observations
    windows = np.array([np.roll(window, int(i)) for i in berv * 10])
    windows_contam = int(100 * sum(np.median(windows, axis=0)) / sum(window))

    qc = int(windows_contam < 25)  # less than 25% of the telluric covered the full time
    check = ["r", "g"][qc]

    berv_bin = np.arange(
        np.min(np.round(berv, 0)) - dbin_berv,
        np.max(np.round(berv, 0)) + dbin_berv + 0.01,
        dbin_berv,
    )
    mask_bin = (berv > berv_bin[0:-1][:, np.newaxis]) & (berv < berv_bin[1:][:, np.newaxis])
    sum_mask = mask_bin[np.sum(mask_bin, axis=1) != 0]
    if sum(np.sum(sum_mask, axis=1) != 0) < 5:  # need ast least 5 bins
        dbin_berv = np.round((np.max(berv) - np.min(berv)) / 15, 1)
        print("Bin size changed for %.1f km/s because not enough bins" % (dbin_berv))
        if not dbin_berv:
            dbin_berv = 0.05
        berv_bin = np.arange(
            np.min(np.round(berv, 0)) - dbin_berv,
            np.max(np.round(berv, 0)) + dbin_berv + 0.01,
            dbin_berv,
        )
        mask_bin = (berv > berv_bin[0:-1][:, np.newaxis]) & (berv < berv_bin[1:][:, np.newaxis])

    plt.figure(figsize=(6 * nb_plot, 6))
    plt.subplot(1, nb_plot, 1)
    self.yarara_get_berv_value(0, Draw=True, new=False, save_fig=False, light_graphic=True)

    ax = plt.gca()
    for j in berv_bin:
        plt.axhline(y=j, color="k", alpha=0.2)
    plt.axhline(y=0, color="k", ls=":")

    berv_bin = berv_bin[:-1][np.sum(mask_bin, axis=1) != 0] + dbin_berv / 2
    mask_bin = mask_bin[np.sum(mask_bin, axis=1) != 0]
    logging.info("Nb bins full : %.0f", (sum(np.sum(mask_bin, axis=1) != 0)))

    all_conti_ref = []
    snr_binned = []
    snr_stacked = []
    all_snr = []
    for j in range(len(mask_bin)):
        all_flux[j] = np.sum(all_flux[mask_bin[j]], axis=0)
        all_conti[j] = np.sum(all_conti[mask_bin[j]], axis=0)
        all_conti_ref.append(np.mean(all_conti[mask_bin[j]], axis=0))
        snr_binned.append(np.sqrt(np.mean((snr[mask_bin[j]]) ** 2)))
        snr_stacked.append(np.sqrt(np.sum((snr[mask_bin[j]]) ** 2)))
        all_snr.append(list(snr[mask_bin[j]]))
    all_flux = all_flux[0 : len(mask_bin)]
    all_conti = all_conti[0 : len(mask_bin)]
    all_conti_ref = np.array(all_conti_ref)
    snr_binned = np.array(snr_binned)

    if nb_plot == 3:
        plt.subplot(1, nb_plot, 2)
        plt.boxplot(
            all_snr,
            positions=berv_bin,
            widths=dbin_berv / 2,
            vert=False,
            labels=[len(all_snr[j]) for j in range(len(all_snr))],
        )
        plt.scatter(snr_binned, berv_bin, marker="x", color="r")
        plt.ylabel("Nb spectra", fontsize=16)

        # plt.tick_params(labelleft=False)
        plt.xlabel("SNR", fontsize=16)
        plt.ylim(ax.get_ylim())

    plt.subplot(1, nb_plot, nb_plot, sharey=ax)
    plt.axhline(y=0, color="k", ls=":")
    plt.title("Window contam covered = %.0f %%" % (windows_contam))
    plot_color_box(color=check)

    self.yarara_star_info(Contam_BERV=["fixed", int(windows_contam)])

    plt.plot(snr_stacked, berv_bin, "bo-", alpha=0.3)
    curve = tableXY(snr_stacked, berv_bin)
    curve.myscatter(
        num=False,
        liste=[len(all_snr[j]) for j in range(len(all_snr))],
        color="k",
        factor=50,
    )
    plt.xlabel("SNR stacked", fontsize=16)
    plt.ylabel("BERV [km/s]", fontsize=16)

    plt.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.10)

    plt.savefig(self.dir_root + "IMAGES/berv_statistic_summary.pdf")


def snr_statistic(self: spec_time_series, version=1):
    self.import_table()
    if version == 1:
        print(self.table["snr"].describe())
        snr = np.array(self.table["snr"])
    else:
        print(self.table["snr_computed"].describe())
        snr = np.array(self.table["snr_computed"])
    plt.figure(figsize=(8, 7))
    plt.title(
        "\n Nb spectra : %.0f\nMin : %.0f   |   Q1 : %.0f   |   Q2 : %.0f   |   Q3 : %.0f   |   Max : %.0f\n"
        % (
            len(snr),
            np.min(snr),
            np.percentile(snr, 25),
            np.percentile(snr, 50),
            np.percentile(snr, 75),
            np.max(snr),
        ),
        fontsize=16,
    )
    plt.hist(snr, bins=40, histtype="step", color="k")
    plt.hist(snr, bins=40, alpha=0.2, color="b")
    plt.axvline(x=np.median(snr), ls="-", color="k")
    plt.axvline(
        x=np.percentile(snr, 16),
        ls=":",
        color="k",
        label=r"$16^{th}$ percentile = %.0f" % (np.percentile(snr, 16)),
    )
    plt.axvline(
        x=np.percentile(snr, 84),
        ls=":",
        color="k",
        label=r"$84^{th}$ percentile = %.0f" % (np.percentile(snr, 84)),
    )
    plt.legend(prop={"size": 14})

    crit = int(np.percentile(snr, 50) > 75)
    check = ["r", "g"][crit]  # median higher than snr 75 in at 5500 angstrom
    plt.xlabel(r"$SNR_{5500}$", fontsize=16, fontweight="bold", color=check)
    plot_color_box(color=check)

    plt.savefig(self.dir_root + "IMAGES/snr_statistic_%.0f.pdf" % (version))


def dace_statistic(
    self: spec_time_series, substract_model=False, ymin=None, ymax=None, return_ylim=False
):
    self.import_table()
    vec = self.import_dace_sts(substract_model=substract_model)
    # vec.recenter(who='Y')

    species = np.array(self.table["ins"])
    vec.species_recenter(species=species)

    vec.substract_polyfit(2, replace=False)
    vec.rms_w()
    vec.detrend_poly.rms_w()
    vec.night_stack()

    plt.figure(figsize=(15, 6))
    vec.plot(color="gray", alpha=0.25, capsize=0, label="rms : %.2f m/s" % (vec.rms))
    vec.detrend_poly.plot(
        color="k",
        capsize=0,
        label="rms2 : %.2f m/s" % (vec.detrend_poly.rms),
        species=species,
    )
    plt.xlabel(r"Time BJD [days]", fontsize=16)
    plt.ylabel(r"RV [m/s]", fontsize=16)
    plt.legend(prop={"size": 14})
    plot_copy_time()

    if ymin is not None:
        plt.ylim(ymin, ymax)

    mini = np.min(vec.x)
    maxi = np.max(vec.x)
    plt.title(
        "%s\n  Nb measurements : %.0f | Nb nights : %.0f | Time span : %.0f days \n   Min : %.0f (%s)  |  Max : %.0f (%s)\n   rms : %.2f m/s   |   rms2 : %.2f m/s   |   $\\sigma_{\\gamma}$ : %.2f m/s\n"
        % (
            self.starname,
            len(vec.x),
            len(vec.stacked.x),
            maxi - mini,
            mini,
            Time(mini + 2400000, format="jd").iso.split(" ")[0],
            maxi,
            Time(maxi + 2400000, format="jd").iso.split(" ")[0],
            vec.rms,
            vec.detrend_poly.rms,
            np.nanmedian(vec.yerr),
        ),
        fontsize=16,
        va="top",
    )
    plt.subplots_adjust(left=0.06, right=0.96, top=0.72)
    plt.savefig(self.dir_root + "IMAGES/RV_statistic.pdf")
    if return_ylim:
        ax = plt.gca()
        return ax.get_ylim()


def yarara_transit_def(
    self: spec_time_series,
    period: float = 100000.0,
    T0: float = 55000.0,
    duration: float = 2.0,
    auto=False,
):
    """period in days, T0 transits center in jdb - 2'400'000, duration in hours"""

    self.import_table()
    self.import_star_info()

    time = np.sort(np.array(self.table.jdb))

    if auto:
        table_transit = pd.read_csv(root + "/Python/Material/transits.csv", index_col=0)
        star_transit_properties = table_transit.loc[table_transit["starname"] == self.starname]
        if len(star_transit_properties):
            period = np.array(star_transit_properties["period"]).astype("float")
            T0 = np.array(star_transit_properties["T0"]).astype("float")
            duration = np.array(star_transit_properties["dt"]).astype("float")
            teff = int(star_transit_properties["Teff"].values[0])
            fwhm = int(star_transit_properties["FWHM"].values[0])
            if teff:
                print(
                    "\n [INFO] Effective temperature upated from %.0f to %.0f K"
                    % (self.star_info["Teff"]["fixed"], teff)
                )
                self.yarara_star_info(Teff=["fixed", int(teff)])
            if fwhm:
                print(
                    "\n [INFO] FWHM upated from %.0f to %.0f km/s"
                    % (self.star_info["FWHM"]["fixed"], fwhm)
                )
                self.yarara_star_info(Fwhm=["fixed", int(fwhm)])
                self.fwhm = fwhm

            for p, t0, dt in zip(period, T0, duration):
                print(
                    "\n [INFO] Star %s found in the table : P = %.2f days | T0 = %.2f JDB | T14 = %.2f hours | Teff = %.0f K | FWHM = %.0f km/s"
                    % (self.starname, p, t0, dt, teff, fwhm)
                )
        else:
            print("\n [WARNING] Star %s not found in the transits.csv table" % (self.starname))
            if (period != 100000) & (period != 0):
                period = np.array([period])
                T0 = np.array([T0])
                duration = np.array([duration])
            else:
                period = np.array([])
                T0 = np.array([])
                duration = np.array([])
    else:
        if (period != 100000) & (period != 0):
            period = np.array([period])
            T0 = np.array([T0])
            duration = np.array([duration])
        else:
            period = np.array([])
            T0 = np.array([])
            duration = np.array([])

    duration /= 24

    transits = np.zeros(len(time))

    c = 0
    for p, t0, dt in zip(period, T0, duration):
        c += 1
        phases = ((time - t0) % p + p / 2) % p - p / 2
        transits += c * (abs(phases) < (dt / 2)).astype("int")

    if np.sum(transits):
        plt.figure(figsize=(15, 5))
        plt.subplot(2, 1, 1)
        plt.ylabel("SNR", fontsize=14)
        plt.scatter(time, self.table.snr, c=transits, zorder=10)

        transit_draw(period, T0, duration)

        length = (np.max(time) - np.min(time)) * 0.1
        plt.xlim(np.min(time) - length, np.max(time) + length)

        plt.title(
            "%s In/Out spectra : %.0f/%.0f"
            % (
                self.starname,
                sum(transits != 0),
                len(transits) - sum(transits != 0),
            )
        )
        ax = plt.gca()

        plt.subplot(2, 1, 2, sharex=ax)
        plt.xlabel("Jdb - 2,400,000 [days]", fontsize=14)
        plt.ylabel("RV", fontsize=14)
        plt.scatter(time, self.table.rv_dace, c=transits, zorder=10)

        transit_draw(period, T0, duration)

        length = (np.max(time) - np.min(time)) * 0.1
        plt.xlim(np.min(time) - length, np.max(time) + length)

        plt.savefig(self.dir_root + "IMAGES/Transit_definition.pdf")

        plt.show(block=False)
    df = pd.DataFrame(transits, columns=["transit_in"])
    self.yarara_obs_info(df)


def yarara_get_first_wave(self: spec_time_series):
    m, m_std, wave = self.yarara_map(
        sub_dico="matching_diff",
        wave_min=3800,
        wave_max=4600,
        plot=False,
        reference="zeros",
    )
    snrs = np.median(m, axis=0) / (mad(m, axis=0) + 1e-6)
    bad_wave = wave[snrs < 1]
    if len(bad_wave):
        min_wave = np.percentile(bad_wave, 95)
    else:
        min_wave = np.nanmin(wave)
    logging.info("Minimum wavelength such than SNR>1 defined at %.2f AA", min_wave)
    return min_wave


def import_dace_sts(
    self: spec_time_series, substract_model=False
):  # subtract_model used by default
    self.import_table()

    model = np.array(self.table["rv_shift"]) * 1000.0
    vector = np.array(self.table["rv_dace"])

    coeff = np.argmin(
        [
            mad(vector - model),
            mad(vector - 0.0 * model),
            mad(vector + model),
        ]
    )
    logging.info("Coefficient selected:", coeff)

    vec = tableXY(
        self.table["jdb"],
        self.table["rv_dace"] + (coeff - 1) * model,
        self.table["rv_dace_std"],
    )
    return vec
