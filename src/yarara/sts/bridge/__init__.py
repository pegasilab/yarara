import glob
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from yarara.stats import rm_outliers
from yarara.util import doppler_r

from ...analysis import tableXY
from ...io import pickle_dump, save_pickle, touch_pickle
from ...paths import root
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


def suppress_low_snr_spectra(
    self: spec_time_series, snr_cutoff: float = 100.0, suppress: bool = False
):
    """
    Supress spectra under fixed treshold in SNR

    Args:
        snr_cutoff : treshold value of the cutoff below which spectra are removed
        suppress : True/False to delete of simply hide the files

    """

    files_to_process = np.sort(glob.glob(self.directory + "R*.p"))
    mask = np.zeros(len(files_to_process))
    self.import_table()
    mask = np.array(self.table["snr"]) < snr_cutoff

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
