#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:24:29 2020

@author: cretignier
"""

from dace.spectroscopy import Spectroscopy
import pandas as pd
import numpy as np
import my_classes as myc
import my_functions as myf
import matplotlib.pylab as plt
import getopt
import sys
import os
from dace.sun import Sun
from tqdm import tqdm

cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])

# =============================================================================
# FUNCTIONS
# =============================================================================


def fileroot_format(vec, code):
    if code == "0":
        return vec
    if code == "1":
        vec = np.array(
            [
                string.split("T")[0] + "T" + string.split("T")[1].replace("-", ":")
                for string in vec
            ]
        )
        return vec


# =============================================================================
#
# =============================================================================

instrument = "HARPS03"
drs_version = None
instrument_mode = None
DRS_version = "old"
force_reduction = False
raw_directory = True

star = "HD20794"
import_raw = "s1d"
sigma_clipping = 3

if len(sys.argv) > 1:
    optlist, args = getopt.getopt(sys.argv[1:], "s:i:d:m:c:D:f:o:")
    for j in optlist:
        if j[0] == "-s":
            star = j[1]
        elif j[0] == "-i":
            instrument = j[1]
        elif j[0] == "-m":
            instrument_mode = j[1]
        elif j[0] == "-d":
            drs_version = j[1]
        elif j[0] == "-c":
            sigma_clipping = float(j[1])
        elif j[0] == "-D":
            DRS_version = j[1]
        elif j[0] == "-f":
            force_reduction = bool(int(j[1]))
        elif j[0] == "-o":
            raw_directory = bool(int(j[1]))

print(
    "\n[INFO] Parameters used : star(%s), instrument(%s), sig_clipping(%.1f), DRS(%s)"
    % (star, instrument, sigma_clipping, DRS_version)
)

temporary_solution = False
if (instrument[0:5] == "HARPS") & (DRS_version == "new"):
    temporary_solution = False


# also change the import_dace_query function inside YARARA <----
if DRS_version == "new":
    dico = {
        "HARPS03": [
            "HARPS",
            "2.3.1",
            "/projects/astro/HARPSNEWDRS/DRS-2.3.1/reduced",
            "r.",
            "_" + import_raw.upper() + "_A.fits",
            "0",
        ],
        "HARPS15": [
            "HARPS",
            "2.3.1",
            "/projects/astro/HARPSNEWDRS/DRS-2.3.1/reduced",
            "r.",
            "_" + import_raw.upper() + "_A.fits",
            "0",
        ],
        "HARPN": [
            "HARPN",
            "2.3.5",
            "/projects/astro/HARPNNEWDRS/DRS-2.3.5/reduced",
            "r.",
            "_" + import_raw.upper() + "_A.fits",
            "0",
        ],
        "ESPRESSO18": [
            "SINGLEHR11",
            "2.2.8-HR11",
            "/hpcstorage/cretigni/ESPRESSO/reduced",
            "r.",
            "_" + import_raw.upper() + "_A.fits",
            "0",
        ],
        "ESPRESSO19": [
            "SINGLEHR11",
            "2.2.8-HR11",
            "/hpcstorage/cretigni/ESPRESSO/reduced",
            "r.",
            "_" + import_raw.upper() + "_A.fits",
            "0",
        ],
        "CORALIE98": [
            "CORALIE",
            "3.3",
            "/hpcstorage/cretigni/CORALIE/reduced",
            "r.",
            "_" + import_raw.upper() + "_A.fits",
            "0",
        ],
        "CORALIE07": [
            "CORALIE",
            "3.4",
            "/hpcstorage/cretigni/CORALIE/reduced",
            "r.",
            "_" + import_raw.upper() + "_A.fits",
            "0",
        ],
        "CORALIE14": [
            "CORALIE",
            "3.8",
            "/hpcstorage/cretigni/CORALIE/reduced",
            "r.",
            "_" + import_raw.upper() + "_A.fits",
            "0",
        ],
    }
    use_drift = 0

elif DRS_version != "new":
    dico = {
        "HARPS03": [
            "HARPS",
            "3.5",
            "/projects/astro/HARPSDRS/DRS-3.5/reduced",
            "",
            "_" + import_raw + "_A.fits",
            "0",
        ],
        "HARPS15": [
            "HARPS",
            "3.5",
            "/projects/astro/HARPSDRS/DRS-3.5/reduced",
            "",
            "_" + import_raw + "_A.fits",
            "0",
        ],
        "HARPN": [
            "HARPN",
            "3.7",
            "/projects/astro/HARPNDRS/DRS-3.7/reduced",
            "",
            "_" + import_raw + "_A.fits",
            "0",
        ],
        "CORALIE98": [
            "CORALIE",
            "3.3",
            "/hpcstorage/cretigni/CORALIE/reduced",
            "",
            "_" + import_raw + "_A.fits",
            "0",
        ],
        "CORALIE07": [
            "CORALIE",
            "3.4",
            "/hpcstorage/cretigni/CORALIE/reduced",
            "",
            "_" + import_raw + "_A.fits",
            "0",
        ],
        "CORALIE14": [
            "CORALIE",
            "3.8",
            "/hpcstorage/cretigni/CORALIE/reduced",
            "",
            "_" + import_raw + "_A.fits",
            "0",
        ],
    }
    use_drift = 1


if instrument_mode is None:
    instrument_mode = dico[instrument][0]

if drs_version is None:
    drs_version = dico[instrument][1]

if raw_directory:
    raw_directory = dico[instrument][2]
else:
    raw_directory = (
        root + "/Yarara/" + star + "/data/" + import_raw + "/spectroDownload"
    )
pre_ext = dico[instrument][3]
post_ext = dico[instrument][4]
replace_code = dico[instrument][5]


directory_to_yarara = root + "/Yarara/" + star + "/data/" + import_raw + "/"

if not os.path.exists(root + "/Yarara/" + star):
    os.system("mkdir " + root + "/Yarara/" + star)

if not os.path.exists(root + "/Yarara/" + star + "/data/"):
    os.system("mkdir " + root + "/Yarara/" + star + "/data/")

if not os.path.exists(root + "/Yarara/" + star + "/data/" + import_raw + "/"):
    os.system("mkdir " + root + "/Yarara/" + star + "/data/" + import_raw + "/")

if not os.path.exists(
    root + "/Yarara/" + star + "/data/" + import_raw + "/" + instrument + "/"
):
    os.system(
        "mkdir "
        + root
        + "/Yarara/"
        + star
        + "/data/"
        + import_raw
        + "/"
        + instrument
        + "/"
    )

night_selected = None
if len(star.split("_N")) > 1:
    night_selected = star.split("_N")[1]
    night_selected = (
        night_selected[0:4] + "-" + night_selected[4:6] + "-" + night_selected[6:]
    )
    old_star = star
    star = star.split("_N")[0]
    print(
        "\n[INFO] Night %s selected to be reduce for the star %s !"
        % (night_selected, star)
    )


print(
    "------------------------------------------------\n STAR LOADED FROM DACE : %s (%s)\n------------------------------------------------"
    % (star, instrument)
)

if star == "Sun":
    if False:
        print("[INFO] Read the dynamic solar table by DACE query")
        data_extract = Sun.get_timeseries()
        data_extract = pd.DataFrame(data_extract)
    else:
        data_extract = pd.read_csv(
            "/hpcstorage/cretigni/Python/Material/Sun_table.csv", index_col=0
        )
        data_extract = data_extract.loc[data_extract["rejected"] == 0]
        print("[INFO] Read the static solar table")
    data_extract = data_extract.loc[data_extract["obs_quality"] > 0.90]
    data_extract = data_extract.loc[
        data_extract["airmass"] < 2.25
    ]  # correction <= 1m/s due to extinction

    if night_selected is not None:
        data_extract = data_extract.loc[data_extract["date_night"] == night_selected]
        star = old_star

    data_extract["filename"] = np.array(
        [i.replace(":", "-") for i in np.array(data_extract["filename"])]
    )
    data_extract["raw_file"] = data_extract["filename"]
    data_extract["sindex"] = data_extract["smw"]
    data_extract["sindex_err"] = data_extract["smw_err"]
    data_extract["drs_qc"] = data_extract["obs_quality"] > 0.90
    data_extract["spectroFluxSn50"] = data_extract["sn_order_50"]
    data_extract["rjd"] = data_extract["date_bjd"]
    data_extract["bispan"] = data_extract["bis_span"]
    data_extract["berv"] /= 1000

    for k in [
        "drift_used",
        "haindex",
        "haindex_err",
        "naindex",
        "naindex_err",
        "caindex",
        "caindex_err",
        "bispan_err",
    ]:
        data_extract[k] = 0
    data_extract = data_extract.reset_index(drop=True)
else:
    if temporary_solution:
        path_table = "/hpcstorage/dumusque/ESPRESSO_DRS_for_HARPS/extract_FITS_data/"
        data_extract = myf.convert_to_dace(
            star, instrument, drs_version, instrument_mode, path_table
        )
    else:
        data_extract = Spectroscopy.get_timeseries(star)[instrument][drs_version][
            instrument_mode
        ]
        for kw in list(
            data_extract.keys()
        ):  # some new exotic error with DACE query... 16.11.21
            if type(kw) == tuple:
                del data_extract[kw]

    if instrument[0:8] == "ESPRESSO":
        if not len(data_extract):
            drs_version = drs_version.replace("HR11", "HR21")
            instrument_mode = "SINGLEHR21"
            data_extract = Spectroscopy.get_timeseries(star)[instrument][drs_version][
                instrument_mode
            ]
            if len(data_extract):
                print(
                    "[INFO] mode %s detected for ESPRESSO drs version %s"
                    % (instrument_mode, drs_version)
                )
            else:
                drs_version = "2.2.8-HR21"
                instrument_mode = "SINGLEHR21"
                data_extract = Spectroscopy.get_timeseries(star)[instrument][
                    drs_version
                ][instrument_mode]
                if len(data_extract):
                    print(
                        "[INFO] mode %s detected for ESPRESSO drs version %s"
                        % (instrument_mode, drs_version)
                    )

    data_extract = pd.DataFrame(data_extract)
    data_extract = data_extract[
        [
            "rjd",
            "berv",
            "raw_file",
            "drs_qc",
            "spectroFluxSn50",
            "drift_used",
            "rv",
            "rv_err",
            "fwhm",
            "fwhm_err",
            "contrast",
            "contrast_err",
            "bispan",
            "bispan_err",
            "rhk",
            "rhk_err",
            "sindex",
            "sindex_err",
        ]
    ]  # ,'haindex','haindex_err','naindex','naindex_err','caindex','caindex_err']]


if len(data_extract):

    print(" Total number of files found : %.0f" % (len(data_extract)))
    print(" K-IQ clipping will be performed with k = %.1f" % (sigma_clipping))

    data_extract["raw_file"] = np.array(
        [i.split("/")[-1] for i in data_extract["raw_file"]]
    )
    first = np.where(np.array([i for i in data_extract["raw_file"][0]]) == ".")[0]

    if DRS_version == "new":
        b1 = first[0] + 1
        b2 = first[1] + 1
    else:
        b1 = 0
        b2 = first[0] + 1
    data_extract["dace_data"] = data_extract["raw_file"].str[b1:-5]
    time_raw = data_extract["raw_file"].str[b2 : b2 + 23]

    # Remove 0.5 days, so that even after midnight, keeps the same night (works in Chile only, not HARPN)

    if instrument == "HARPN":
        time_raw = np.array(
            [
                i.split("T")[0] + "T" + i.split("T")[1].replace("-", ":")
                for i in time_raw
            ]
        )
        data_extract["berv"] = np.array(data_extract["berv"]) / 1000

    data_extract["night"] = pd.to_datetime(time_raw) - pd.Timedelta(
        24 * 3600 * 1e9 * 0.5
    )
    data_extract["night"] = np.array([str(i)[:10] for i in data_extract["night"]])
    data_extract = data_extract.loc[~data_extract.duplicated("dace_data")].reset_index(
        drop=True
    )

    if instrument == "HARPS03":
        data_extract = data_extract.loc[data_extract["rjd"] > 53500].reset_index(
            drop=True
        )

    # =============================================================================
    # supression of data doubtful
    # =============================================================================

    index_spectrum = np.arange(len(data_extract))
    index = np.zeros((len(index_spectrum), 9))

    def index_selection(col="rv", Plot=False):
        index2 = np.arange(len(data_extract))
        vec = myc.tableXY(data_extract["rjd"], data_extract[col].astype("float"))
        vec.y[np.isnan(vec.y)] = np.random.randn(sum(np.isnan(vec.y)))
        vec.yerr[vec.yerr == 0] = 10 * np.nanmax(vec.yerr)
        vec.recenter(who="Y")

        vec.night_stack(bin_length=1, replace=False)
        if len(vec.stacked.y) > 5:
            dust, dust, sup, inf = myf.rm_outliers(
                vec.stacked.y, kind="inter", m=5, return_borders=True
            )
            vec.masked((vec.y <= sup) & (vec.y >= inf))

        vec.night_stack(bin_length=1, replace=False)
        if (sum(abs(vec.stacked.y)) != 0) & (len(vec.stacked.y) > 1):
            if instrument != "HARPN":
                vec.stacked.substract_polyfit(5, replace=False, Draw=False)
            else:
                vec.stacked.substract_polymorceau(
                    [56738], degree=3, replace=False, Draw=False
                )
            model = myc.tableXY(vec.stacked.x, vec.stacked.model)
            model.interpolate(new_grid=np.array(vec.x), method="linear")
            vec.y -= model.y
            mask, dust = myf.rm_outliers(vec.y, kind="inter", m=5)
            vec.masked(mask)
            vec.y += model.y[mask]

        trend = np.zeros(len(data_extract["rjd"]))
        if sum(abs(vec.y)):
            for j in range(2):  # two iteration
                vec.rm_outliers(kind="inter", m=5)
                if instrument != "HARPN":
                    vec.substract_polyfit(5, replace=False, Draw=False)
                else:
                    vec.substract_polymorceau(
                        [56738], degree=3, replace=False, Draw=False
                    )
                model = myc.tableXY(vec.x, vec.model)
                vec.y -= model.y
                model.interpolate(
                    new_grid=np.array(data_extract["rjd"]), method="linear"
                )
                trend += model.y

        vec = myc.tableXY(data_extract["rjd"], data_extract[col].astype("float"))
        vec.y[np.isnan(vec.y)] = np.random.randn(sum(np.isnan(vec.y)))
        vec.y -= trend

        vec.night_stack(bin_length=1, replace=False)
        if len(vec.stacked.y) > 5:
            dust, dust, sup, inf = myf.rm_outliers(
                vec.stacked.y, kind="inter", m=5, return_borders=True
            )
            mask_out = (vec.y <= sup) & (vec.y >= inf)
            vec.masked(mask_out)
            index2 = index2[mask_out]

        vec.rm_outliers(kind="inter", m=sigma_clipping)

        # vec.y += vec.sub_model[vec.mask]
        index2 = index2[vec.mask]
        if Plot:
            trend = myc.tableXY(data_extract["rjd"], trend)
            vec = myc.tableXY(data_extract["rjd"], data_extract[col].astype("float"))
            vec.y[np.isnan(vec.y)] = np.random.randn(sum(np.isnan(vec.y)))
            shift = np.nanmedian(vec.y[index2] - trend.y[index2])
            trend.interpolate(replace=True)
            vec.plot(color="gray", capsize=0)
            vec.plot(mask=index2, capsize=0)
            trend.y += shift
            trend.plot(color="r", ls="-", zorder=10)
            iq = myf.IQ(vec.y)
            if not iq:
                iq = 1
            plt.ylim(
                np.nanpercentile(vec.y, 25) - 5 * iq,
                np.nanpercentile(vec.y, 25) + 5 * iq,
            )
            plt.ylabel(col)
            plt.xlabel("Time")
        return index2

    if sigma_clipping:
        Plot = True
        plt.figure(figsize=(12, 8))
        plt.subplot(5, 2, 1)
        index_rv = index_selection("rv", Plot=Plot)
        plt.subplot(5, 2, 2)
        index_rv_std = index_selection("rv_err", Plot=Plot)
        plt.subplot(5, 2, 3)
        index_fwhm = index_selection("fwhm", Plot=Plot)
        plt.subplot(5, 2, 4)
        index_fwhm_std = index_selection("fwhm_err", Plot=Plot)
        plt.subplot(5, 2, 5)
        index_contrast = index_selection("contrast", Plot=Plot)
        plt.subplot(5, 2, 6)
        index_contrast_std = index_selection("contrast_err", Plot=Plot)
        plt.subplot(5, 2, 7)
        index_sindex = index_selection("sindex", Plot=Plot)
        plt.subplot(5, 2, 8)
        index_sindex_std = index_selection("sindex_err", Plot=Plot)

        index[index_rv, 0] = 1
        index[index_rv_std, 1] = 1
        index[index_sindex, 6] = 1
        index[index_sindex_std, 7] = 1

        index[index_fwhm, 2] = 1
        index[index_fwhm_std, 3] = 1
        index[index_contrast, 4] = 1
        index[index_contrast_std, 5] = 1

        index[:, 5] = 1

        # time clipping
        time = myc.tableXY(
            data_extract["rjd"].astype("float"), data_extract["rjd"].astype("float")
        )
        time.rm_outliers(m=200, kind="inter", bin_length=1, replace=False)
        plt.subplot(5, 2, 9)
        time.plot(color="gray")
        time.plot(color="k", mask=time.mask)
        plt.ylabel("Time")
        index[:, 8] = time.mask.astype("int")
        index_kept = index_spectrum[np.product(index, axis=1).astype("bool")]
    else:
        print(" [INFO] Sigma clipping skipped")
        index_kept = index_spectrum

    if not os.path.exists(directory_to_yarara + "/ALL_OBSERVATIONS"):
        os.system("mkdir " + directory_to_yarara + "/ALL_OBSERVATIONS")
    data_extract.to_csv(
        directory_to_yarara
        + "/ALL_OBSERVATIONS/All_observations_%s_%s_DRS-%s.rdb"
        % (star, instrument, drs_version.replace(".", "-")),
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    data_extract_selected = data_extract.loc[index_kept].copy()
    data_extract_selected = data_extract_selected.loc[data_extract["drs_qc"]]
    # supress the 1% spectra with lowest snr50
    data_extract_selected = data_extract_selected.loc[
        data_extract["spectroFluxSn50"]
        > np.nanpercentile(data_extract["spectroFluxSn50"], 1)
    ]

    print(
        " [INFO] %.0f spectrum selected as valid on the pool of %.0f spectra"
        % (len(index_kept), len(index_spectrum))
    )

    rv = myc.tableXY(data_extract["rjd"], data_extract["rv"], data_extract["rv_err"])
    plt.subplot(5, 2, 10)
    rv.plot(
        color="gray",
        label="all(%.0f-%.0fn)"
        % (len(data_extract), len(np.unique(data_extract["night"]))),
        capsize=0,
    )
    rv = myc.tableXY(
        data_extract_selected["rjd"],
        data_extract_selected["rv"],
        data_extract_selected["rv_err"],
    )
    rv.plot(
        color="k",
        label="kept(%.0f-%.0fn)"
        % (len(data_extract_selected), len(np.unique(data_extract_selected["night"]))),
        capsize=0,
    )
    iq = myf.IQ(rv.y)
    plt.ylim(np.nanpercentile(rv.y, 25) - 5 * iq, np.nanpercentile(rv.y, 25) + 5 * iq)
    plt.subplots_adjust(
        left=0.08, right=0.95, top=0.95, bottom=0.09, hspace=0.3, wspace=0.3
    )
    plt.legend()
    plt.savefig(directory_to_yarara + "/Selections_observations_%s.jpg" % (instrument))

    # =============================================================================
    # check temporal baseline and berv span
    # =============================================================================

    treshold = 1
    nights = np.sort(np.unique(data_extract_selected["night"]))
    nb_night = len(nights)
    nb_files = len(data_extract_selected)
    if (nb_files < treshold) & (not force_reduction):
        print(
            " [ERROR] Not enough days of observations for the star : %s, current value of %.0f(%.0f) is below the treshold of %.0f days"
            % (star, nb_files, nb_night, treshold)
        )
    else:
        print(
            " [INFO] %.0f days of observations (%.0f) for the star : %s"
            % (nb_night, nb_files, star)
        )
        print(" [INFO] First night : %s, Last night : %s" % (nights[0], nights[-1]))
        berv = myc.tableXY(
            data_extract_selected["rjd"], data_extract_selected["berv"].astype("float")
        )
        rv = myc.tableXY(
            data_extract_selected["rjd"],
            data_extract_selected["rv"].astype("float"),
            data_extract_selected["rv_err"].astype("float"),
        )
        fwhm = myc.tableXY(
            data_extract_selected["rjd"],
            data_extract_selected["fwhm"].astype("float"),
            data_extract_selected["fwhm_err"].astype("float"),
        )
        contrast = myc.tableXY(
            data_extract_selected["rjd"],
            data_extract_selected["contrast"].astype("float"),
            data_extract_selected["contrast_err"].astype("float"),
        )
        rhk = myc.tableXY(
            data_extract_selected["rjd"],
            data_extract_selected["rhk"].astype("float"),
            data_extract_selected["rhk_err"].astype("float"),
        )
        acti = r"$\log R\prime_{HK}$"

        if not np.nansum(abs(rhk.y)):
            acti = "S-index"
            rhk = myc.tableXY(
                data_extract_selected["rjd"],
                data_extract_selected["sindex"].astype("float"),
                data_extract_selected["sindex_err"].astype("float"),
            )

        rv.supress_nan()
        rv.not_null()
        fwhm.supress_nan()
        fwhm.not_null()
        contrast.supress_nan()
        contrast.not_null()
        rhk.supress_nan()
        rhk.not_null()

        rv.binning(bin_width=1)
        fwhm.binning(bin_width=1)
        contrast.binning(bin_width=1)
        rhk.binning(bin_width=1)

        plt.figure(figsize=(18, 9))
        plt.subplot(2, 2, 1)
        plt.ylabel("RV [m/s]", fontsize=16)
        rv.binned.plot(capsize=0)
        plt.subplot(2, 2, 2)
        plt.ylabel("FWHM [km/s]", fontsize=16)
        fwhm.binned.plot(capsize=0)
        plt.subplot(2, 2, 3)
        plt.ylabel("Contrast [arb.unit]", fontsize=16)
        contrast.binned.plot(capsize=0)
        plt.subplot(2, 2, 4)
        plt.ylabel(acti, fontsize=16)
        rhk.binned.plot(capsize=0)
        plt.subplots_adjust(
            left=0.06, right=0.93, top=0.93, bottom=0.08, hspace=0.3, wspace=0.3
        )
        plt.savefig(
            directory_to_yarara + "/Summary_observations_CCF_%s.pdf" % (instrument)
        )

        berv.fit_sinus(guess=[30, 365.25, 0, 0, 0, 0], report=False)
        berv_amplitude = abs(berv.params["amp"].value)
        berv_span = np.max(berv.y) - np.min(berv.y)
        rv_span = np.nanpercentile(rv.y, 97.5) - np.nanpercentile(rv.y, 2.5)

        if berv_span < 3:
            print(
                " [WARNING] BERV span unsufficiently covered for the star : %s, current value of %.1f km/s is below the treshold of 3 km/s"
                % (star, berv_span)
            )
        else:
            print(
                " [INFO] BERV span of %.1f km/s for the star : %s" % (berv_span, star)
            )

        if rv_span > 50:
            print(
                " [WARNING] RV span too large (binary?,large amplitude planet?, transit?) for the star : %s, current value of %.1f m/s"
                % (star, rv_span)
            )

        data_extract_selected = data_extract_selected.rename(
            columns={
                "rv": "vrad",
                "rv_err": "svrad",
                "fwhm_err": "sig_fwhm",
                "contrast_err": "sig_contrast",
                "rhk_err": "sig_rhk",
                "sindex": "s_mw",
                "sindex_err": "sig_s",
                #'haindex':'ha',
                #'haindex_err':'sig_ha',
                #'naindex':'na',
                #'naindex_err':'sig_na',
                #'caindex':'ca',
                #'caindex_err':'sig_ca',
                "bispan": "bis_span",
                "bispan_err": "sig_bis_span",
                "drift_used": "drift_used",
            }
        )

        kw = [
            "rjd",
            "vrad",
            "svrad",
            "fwhm",
            "sig_fwhm",
            "bis_span",
            "sig_bis_span",
            "contrast",
            "sig_contrast",
            "s_mw",
            "sig_s",
            #'ha', 'sig_ha', 'na', 'sig_na', 'ca', 'sig_ca',
            "rhk",
            "sig_rhk",
            "berv",
            "drift_used",
        ]

        data_extract_selected = data_extract_selected.reset_index(drop=True)
        data_extract_selected["drift_used"] = (
            data_extract_selected["drift_used"].astype("float") * use_drift
        )
        data_extract_selected = data_extract_selected[
            kw
            + list(
                np.setdiff1d(np.array(list(data_extract_selected.keys())), np.array(kw))
            )
        ]

        kw_rm = []
        for i in data_extract_selected.keys():
            try:
                if sum(np.isnan(data_extract_selected[i].astype("float"))):
                    kw_rm.append(i)
                    data_extract_selected[i] = 0  # update 3.12.20
                else:
                    data_extract_selected[i] = np.round(
                        data_extract_selected[i].astype("float"), 6
                    )
            except:
                pass

        # data_extract_selected = data_extract_selected.drop(columns=kw_rm) #update 3.12.20

        n = np.array(data_extract_selected["night"])
        f = np.array(data_extract_selected["dace_data"])

        f = fileroot_format(f, replace_code)
        complete_fileroot = raw_directory + "/" + n + "/" + pre_ext + f + post_ext
        for number in np.arange(len(complete_fileroot)):
            if os.path.exists(
                complete_fileroot[number].replace(
                    import_raw.upper(), import_raw.upper() + "_SKYSUB"
                )
            ):
                complete_fileroot[number] = complete_fileroot[number].replace(
                    import_raw.upper(), import_raw.upper() + "_SKYSUB"
                )

        data_extract_selected["fileroot"] = complete_fileroot

        file_exist = []
        night_saved = []
        fileroot_saved = []
        counter = 0
        # for index in tqdm(np.array(data_extract_selected.index)):
        for f, old_night, index in zip(
            np.array(data_extract_selected["fileroot"]),
            np.array(data_extract_selected["night"]),
            np.array(data_extract_selected.index),
        ):
            # f = data_extract_selected.loc[index,'fileroot']
            # counter+=1 ; print(counter)
            if os.path.exists(f):
                file_exist.append(True)
                night_saved.append(old_night)
                fileroot_saved.append(f)
            else:
                if instrument[0] == "E":  # for espresso copy the data on the machine
                    if not os.path.exists(raw_directory + "/" + old_night + "/"):
                        os.system("mkdir " + raw_directory + "/" + old_night + "/")
                    error = os.system(
                        "scp espresso_drs@espressodrs:/data/ESPRESSODRS/DRS-2.2.8/reduced/"
                        + old_night
                        + "/"
                        + f.split("/")[-1]
                        + " "
                        + raw_directory
                        + "/"
                        + old_night
                        + "/"
                    )
                    if not error:
                        print("File %s copied" % (f.split("/")[-1]))
                        fileroot_saved.append(f)
                        file_exist.append(True)
                        night_saved.append(old_night)
                else:  # for solar observations
                    # old_night = data_extract_selected.loc[index,'night']
                    new_night = str(
                        pd.to_datetime(old_night) + pd.Timedelta(24 * 3600 * 1e9)
                    )[0:10]
                    f2 = f.replace(old_night, new_night)
                    if os.path.exists(f2):
                        file_exist.append(True)
                        night_saved.append(new_night)
                        fileroot_saved.append(f2)
                    else:
                        file_exist.append(False)
                        night_saved.append(old_night)
                        fileroot_saved.append(f)
                        # print(old_night,f)

        file_exist = np.array(file_exist)
        night_saved = np.array(night_saved)
        fileroot_saved = np.array(fileroot_saved)

        data_extract_selected["fileroot"] = fileroot_saved
        data_extract_selected["night"] = night_saved
        file_missing = np.array(data_extract_selected["fileroot"])[
            np.array(~file_exist)
        ]

        print(
            " [INFO] %.0f file(s) not found on the pool of %.0f files"
            % (sum(~file_exist), len(file_exist))
        )
        if len(file_missing) > 0:
            np.savetxt(
                directory_to_yarara + "/Files_not_founded.txt", file_missing, fmt="%s"
            )
            print(
                " [INFO] file(s) not found saved in %s"
                % (directory_to_yarara + "/Files_not_founded_%s.txt" % (instrument))
            )
        data_extract_selected = data_extract_selected.loc[file_exist].reset_index(
            drop=True
        )
        print(
            " [INFO] %.0f file(s) can be reduced in the pipeline"
            % (len(data_extract_selected))
        )

        data_extract_selected = data_extract_selected.drop(columns=["dace_data"])

        # os.system('cp %s/%s/*%s*%s.fits %s'%(raw_directory, n, f, import_raw, directory_to_yarara))

        data_extract_selected.loc[-1] = data_extract_selected.loc[0].copy()

        for i in data_extract_selected.keys():
            data_extract_selected.loc[-1, i] = "-" * len(i)

        data_extract_selected.index = data_extract_selected.index + 1

        data_extract_selected = data_extract_selected.sort_index()

        data_extract_selected.to_csv(
            directory_to_yarara
            + "/%s_%s_DRS-%s.rdb" % (star, instrument, drs_version.replace(".", "-")),
            sep="\t",
            index=False,
            float_format="%.6f",
        )

        print(
            " [INFO] file saved %s"
            % (
                directory_to_yarara
                + "/%s_%s_DRS-%s.rdb"
                % (star, instrument, drs_version.replace(".", "-"))
            )
        )

    plt.close("all")
else:
    print(" No data found for %s with %s instrument" % (star, instrument))
