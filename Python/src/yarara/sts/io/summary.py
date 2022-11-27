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

from ... import io, util
from ...analysis import tableXY

if TYPE_CHECKING:
    from .. import spec_time_series


# =============================================================================
# MAKE SUMMARY
# =============================================================================

# io
def yarara_analyse_summary(self: spec_time_series, rm_old: bool = False) -> None:
    """
    Produce a summary table with the RASSINE files of the specified directory

    """

    directory = self.directory
    if rm_old:
        os.system("rm " + directory + "Analyse_summary.p")
        os.system("rm " + directory + "Analyse_summary.csv")

    test, names = self.import_rassine_output(return_name=True)

    # info obs

    ins = np.array([test[j]["parameters"].get("instrument") for j in range(len(test))])
    snr = np.array([test[j]["parameters"].get("SNR_5500") for j in range(len(test))])
    snr2 = np.array([test[j]["parameters"].get("SNR_computed") for j in range(len(test))])
    jdb = np.array([test[j]["parameters"].get("jdb") for j in range(len(test))])
    mjd = np.array([test[j]["parameters"].get("mjd") for j in range(len(test))])
    berv = np.array([test[j]["parameters"].get("berv") for j in range(len(test))])
    berv_planet = np.array([test[j]["parameters"].get("berv_planet") for j in range(len(test))])
    lamp_offset = np.array([test[j]["parameters"].get("lamp_offset") for j in range(len(test))])
    rv = np.array([test[j]["parameters"].get("rv") for j in range(len(test))])
    rv_shift = np.array([test[j]["parameters"].get("RV_shift") for j in range(len(test))])
    rv_sec = np.array([test[j]["parameters"].get("RV_sec") for j in range(len(test))])
    rv_moon = np.array([test[j]["parameters"].get("RV_moon") for j in range(len(test))])
    airmass = np.array([test[j]["parameters"].get("airmass") for j in range(len(test))])
    texp = np.array([test[j]["parameters"].get("texp") for j in range(len(test))])
    seeing = np.array([test[j]["parameters"].get("seeing") for j in range(len(test))])
    humidity = np.array([test[j]["parameters"].get("humidity") for j in range(len(test))])
    rv_planet = np.array([test[j]["parameters"].get("rv_planet") for j in range(len(test))])
    rv_dace = np.array([test[j]["parameters"].get("rv_dace") for j in range(len(test))])
    rv_dace_std = np.array([test[j]["parameters"].get("rv_dace_std") for j in range(len(test))])
    flux_balance = np.array([test[j]["parameters"].get("flux_balance") for j in range(len(test))])
    transit = np.array([test[j]["parameters"].get("transit_in") for j in range(len(test))])
    night_drift = np.array([test[j]["parameters"].get("drift_used") for j in range(len(test))])
    night_drift_std = np.array(
        [test[j]["parameters"].get("drift_used_std") for j in range(len(test))]
    )

    # info activity

    rhk = np.array([test[j]["parameters"].get("RHK") for j in range(len(test))])
    rhk_std = np.array([test[j]["parameters"].get("RHK_std") for j in range(len(test))])
    kernel_rhk = np.array([test[j]["parameters"].get("Kernel_CaII") for j in range(len(test))])
    kernel_rhk_std = np.array(
        [test[j]["parameters"].get("Kernel_CaII_std") for j in range(len(test))]
    )
    ca2k = np.array([test[j]["parameters"].get("CaIIK") for j in range(len(test))])
    ca2k_std = np.array([test[j]["parameters"].get("CaIIK_std") for j in range(len(test))])
    ca2h = np.array([test[j]["parameters"].get("CaIIH") for j in range(len(test))])
    ca2h_std = np.array([test[j]["parameters"].get("CaIIH_std") for j in range(len(test))])
    ca2 = np.array([test[j]["parameters"].get("CaII") for j in range(len(test))])
    ca2_std = np.array([test[j]["parameters"].get("CaII_std") for j in range(len(test))])
    ca1 = np.array([test[j]["parameters"].get("CaI") for j in range(len(test))])
    ca1_std = np.array([test[j]["parameters"].get("CaI_std") for j in range(len(test))])
    mg1 = np.array([test[j]["parameters"].get("MgI") for j in range(len(test))])
    mg1_std = np.array([test[j]["parameters"].get("MgI_std") for j in range(len(test))])
    mg1a = np.array([test[j]["parameters"].get("MgIa") for j in range(len(test))])
    mg1a_std = np.array([test[j]["parameters"].get("MgIa_std") for j in range(len(test))])
    mg1b = np.array([test[j]["parameters"].get("MgIb") for j in range(len(test))])
    mg1b_std = np.array([test[j]["parameters"].get("MgIb_std") for j in range(len(test))])
    mg1c = np.array([test[j]["parameters"].get("MgIc") for j in range(len(test))])
    mg1c_std = np.array([test[j]["parameters"].get("MgIc_std") for j in range(len(test))])
    d3 = np.array([test[j]["parameters"].get("HeID3") for j in range(len(test))])
    d3_std = np.array([test[j]["parameters"].get("HeID3_std") for j in range(len(test))])
    nad = np.array([test[j]["parameters"].get("NaD") for j in range(len(test))])
    nad_std = np.array([test[j]["parameters"].get("NaD_std") for j in range(len(test))])
    nad1 = np.array([test[j]["parameters"].get("NaD1") for j in range(len(test))])
    nad1_std = np.array([test[j]["parameters"].get("NaD1_std") for j in range(len(test))])
    nad2 = np.array([test[j]["parameters"].get("NaD2") for j in range(len(test))])
    nad2_std = np.array([test[j]["parameters"].get("NaD2_std") for j in range(len(test))])
    ha = np.array([test[j]["parameters"].get("Ha") for j in range(len(test))])
    ha_std = np.array([test[j]["parameters"].get("Ha_std") for j in range(len(test))])
    hb = np.array([test[j]["parameters"].get("Hb") for j in range(len(test))])
    hb_std = np.array([test[j]["parameters"].get("Hb_std") for j in range(len(test))])
    hc = np.array([test[j]["parameters"].get("Hc") for j in range(len(test))])
    hc_std = np.array([test[j]["parameters"].get("Hc_std") for j in range(len(test))])
    hd = np.array([test[j]["parameters"].get("Hd") for j in range(len(test))])
    hd_std = np.array([test[j]["parameters"].get("Hd_std") for j in range(len(test))])
    heps = np.array([test[j]["parameters"].get("Heps") for j in range(len(test))])
    heps_std = np.array([test[j]["parameters"].get("Heps_std") for j in range(len(test))])

    wbk = np.array([test[j]["parameters"].get("WB_K") for j in range(len(test))])
    wbk_std = np.array([test[j]["parameters"].get("WB_K_std") for j in range(len(test))])
    wbh = np.array([test[j]["parameters"].get("WB_H") for j in range(len(test))])
    wbh_std = np.array([test[j]["parameters"].get("WB_H_std") for j in range(len(test))])
    wb = np.array([test[j]["parameters"].get("WB") for j in range(len(test))])
    wb_std = np.array([test[j]["parameters"].get("WB_std") for j in range(len(test))])
    wb_h1 = np.array([test[j]["parameters"].get("WB_H1") for j in range(len(test))])
    wb_h1_std = np.array([test[j]["parameters"].get("WB_H1_std") for j in range(len(test))])
    kernel_wb = np.array([test[j]["parameters"].get("Kernel_WB") for j in range(len(test))])
    kernel_wb_std = np.array(
        [test[j]["parameters"].get("Kernel_WB_std") for j in range(len(test))]
    )

    cb = np.array([test[j]["parameters"].get("CB") for j in range(len(test))])
    cb_std = np.array([test[j]["parameters"].get("CB_std") for j in range(len(test))])
    cb2 = np.array([test[j]["parameters"].get("CB2") for j in range(len(test))])
    cb2_std = np.array([test[j]["parameters"].get("CB2_std") for j in range(len(test))])

    # pca shells coefficient

    shell_compo = np.array([test[j]["parameters"].get("shell_fitted") for j in range(len(test))])
    shell1 = np.array([test[j]["parameters"].get("shell_v1") for j in range(len(test))])
    shell2 = np.array([test[j]["parameters"].get("shell_v2") for j in range(len(test))])
    shell3 = np.array([test[j]["parameters"].get("shell_v3") for j in range(len(test))])
    shell4 = np.array([test[j]["parameters"].get("shell_v4") for j in range(len(test))])
    shell5 = np.array([test[j]["parameters"].get("shell_v5") for j in range(len(test))])

    # pca rv vectors
    pca_compo = np.array([test[j]["parameters"].get("pca_fitted") for j in range(len(test))])
    pca1 = np.array([test[j]["parameters"].get("pca_v1") for j in range(len(test))])
    pca2 = np.array([test[j]["parameters"].get("pca_v2") for j in range(len(test))])
    pca3 = np.array([test[j]["parameters"].get("pca_v3") for j in range(len(test))])
    pca4 = np.array([test[j]["parameters"].get("pca_v4") for j in range(len(test))])
    pca5 = np.array([test[j]["parameters"].get("pca_v5") for j in range(len(test))])

    # telluric ccf

    vec_background = np.array(
        [test[j]["parameters"].get("proxy_background") for j in range(len(test))]
    )

    # telluric ccf

    tell_ew = np.array([test[j]["parameters"].get("telluric_ew") for j in range(len(test))])
    tell_contrast = np.array(
        [test[j]["parameters"].get("telluric_contrast") for j in range(len(test))]
    )
    tell_rv = np.array([test[j]["parameters"].get("telluric_rv") for j in range(len(test))])
    tell_fwhm = np.array([test[j]["parameters"].get("telluric_fwhm") for j in range(len(test))])
    tell_center = np.array(
        [test[j]["parameters"].get("telluric_center") for j in range(len(test))]
    )
    tell_depth = np.array([test[j]["parameters"].get("telluric_depth") for j in range(len(test))])

    # telluric ccf h2o

    h2o_ew = np.array([test[j]["parameters"].get("h2o_ew") for j in range(len(test))])
    h2o_contrast = np.array([test[j]["parameters"].get("h2o_contrast") for j in range(len(test))])
    h2o_rv = np.array([test[j]["parameters"].get("h2o_rv") for j in range(len(test))])
    h2o_fwhm = np.array([test[j]["parameters"].get("h2o_fwhm") for j in range(len(test))])
    h2o_center = np.array([test[j]["parameters"].get("h2o_center") for j in range(len(test))])
    h2o_depth = np.array([test[j]["parameters"].get("h2o_depth") for j in range(len(test))])

    # telluric ccf o2

    o2_ew = np.array([test[j]["parameters"].get("o2_ew") for j in range(len(test))])
    o2_contrast = np.array([test[j]["parameters"].get("o2_contrast") for j in range(len(test))])
    o2_rv = np.array([test[j]["parameters"].get("o2_rv") for j in range(len(test))])
    o2_fwhm = np.array([test[j]["parameters"].get("o2_fwhm") for j in range(len(test))])
    o2_center = np.array([test[j]["parameters"].get("o2_center") for j in range(len(test))])
    o2_depth = np.array([test[j]["parameters"].get("o2_depth") for j in range(len(test))])

    # teff

    t_eff = np.array([test[j]["parameters"].get("Teff") for j in range(len(test))])
    t_eff_std = np.array([test[j]["parameters"].get("Teff_std") for j in range(len(test))])

    # ghost

    ghost = np.array([test[j]["parameters"].get("ghost") for j in range(len(test))])

    # sas
    sas = np.array([test[j]["parameters"].get("SAS") for j in range(len(test))])
    sas_std = np.array([test[j]["parameters"].get("SAS_std") for j in range(len(test))])
    sas1y = np.array([test[j]["parameters"].get("SAS1Y") for j in range(len(test))])
    sas1y_std = np.array([test[j]["parameters"].get("SAS1Y_std") for j in range(len(test))])
    bis = np.array([test[j]["parameters"].get("BIS") for j in range(len(test))])
    bis_std = np.array([test[j]["parameters"].get("BIS_std") for j in range(len(test))])
    bis2 = np.array([test[j]["parameters"].get("BIS2") for j in range(len(test))])
    bis2_std = np.array([test[j]["parameters"].get("BIS2_std") for j in range(len(test))])

    # parabola ccf

    try:
        para_rv = np.array([test[j]["ccf_parabola"].get("para_rv") for j in range(len(test))])
        para_depth = np.array(
            [test[j]["ccf_parabola"].get("para_depth") for j in range(len(test))]
        )
    except KeyError:
        para_rv = np.zeros(len(snr))
        para_depth = np.zeros(len(snr))

    # gaussian ccf
    try:
        ccf_ew = np.array([test[j]["ccf"].get("ew") for j in range(len(test))])
        ccf_contrast = np.array(
            [test[j]["ccf_gaussian"].get("contrast") for j in range(len(test))]
        )
        ccf_contrast_std = np.array(
            [test[j]["ccf_gaussian"].get("contrast_std") for j in range(len(test))]
        )
        ccf_rv = np.array([test[j]["ccf_gaussian"].get("rv") for j in range(len(test))])
        ccf_rv_std = np.array([test[j]["ccf_gaussian"].get("rv_std") for j in range(len(test))])
        ccf_fwhm = (
            np.array([test[j]["ccf_gaussian"].get("fwhm") for j in range(len(test))]) * 2.355
        )
        ccf_fwhm_std = (
            np.array([test[j]["ccf_gaussian"].get("fwhm_std") for j in range(len(test))]) * 2.355
        )
        ccf_offset = np.array([test[j]["ccf_gaussian"].get("offset") for j in range(len(test))])
        ccf_offset_std = np.array(
            [test[j]["ccf_gaussian"].get("offset_std") for j in range(len(test))]
        )
        ccf_vspan = np.array([test[j]["ccf_gaussian"].get("vspan") for j in range(len(test))])
        ccf_vspan_std = np.array([test[j]["ccf_gaussian"].get("rv_std") for j in range(len(test))])
        ccf_svrad_phot = np.array(
            [test[j]["ccf_gaussian"].get("rv_std_phot") for j in range(len(test))]
        )

    except KeyError:
        ccf_ew = np.zeros(len(snr))
        ccf_contrast = np.zeros(len(snr))
        ccf_contrast_std = np.zeros(len(snr))
        ccf_rv = np.zeros(len(snr))
        ccf_rv_std = np.zeros(len(snr))
        ccf_fwhm = np.zeros(len(snr))
        ccf_fwhm_std = np.zeros(len(snr))
        ccf_offset = np.zeros(len(snr))
        ccf_offset_std = np.zeros(len(snr))
        ccf_vspan = np.zeros(len(snr))
        ccf_vspan_std = np.zeros(len(snr))
        ccf_svrad_phot = np.zeros(len(snr))

    dico = pd.DataFrame(
        {
            "filename": names,
            "ins": ins,
            "snr": snr,
            "snr_computed": snr2,
            "flux_balance": flux_balance,
            "jdb": jdb,
            "mjd": mjd,
            "berv": berv,
            "berv_planet": berv_planet,
            "lamp_offset": lamp_offset,
            "rv": rv,
            "rv_shift": rv_shift,
            "rv_sec": rv_sec,
            "rv_moon": rv_moon,
            "rv_planet": rv_planet,
            "rv_dace": rv_dace,
            "rv_dace_std": rv_dace_std,
            "rv_drift": night_drift,
            "rv_drift_std": night_drift_std,
            "airmass": airmass,
            "texp": texp,
            "seeing": seeing,
            "humidity": humidity,
            "transit_in": transit,
            "RHK": rhk,
            "RHK_std": rhk_std,
            "Kernel_CaII": kernel_rhk,
            "Kernel_CaII_std": kernel_rhk_std,
            "CaIIK": ca2k,
            "CaIIK_std": ca2k_std,
            "CaIIH": ca2h,
            "CaIIH_std": ca2h_std,
            "CaII": ca2,
            "CaII_std": ca2_std,
            "CaI": ca1,
            "CaI_std": ca1_std,
            "MgI": mg1,
            "MgI_std": mg1_std,
            "MgIa": mg1a,
            "MgIa_std": mg1a_std,
            "MgIb": mg1b,
            "MgIb_std": mg1b_std,
            "MgIc": mg1c,
            "MgIc_std": mg1c_std,
            "HeID3": d3,
            "HeID3_std": d3_std,
            "NaD": nad,
            "NaD_std": nad_std,
            "NaD1": nad1,
            "NaD1_std": nad1_std,
            "NaD2": nad2,
            "NaD2_std": nad2_std,
            "Ha": ha,
            "Ha_std": ha_std,
            "Hb": hb,
            "Hb_std": hb_std,
            "Hc": hc,
            "Hc_std": hc_std,
            "Hd": hd,
            "Hd_std": hd_std,
            "Heps": heps,
            "Heps_std": heps_std,
            "WBK": wbk,
            "WBK_std": wbk_std,
            "WBH": wbh,
            "WBH_std": wbh_std,
            "WB": wb,
            "WB_std": wb_std,
            "WB_H1": wb_h1,
            "WB_H1_std": wb_h1_std,
            "Kernel_WB": kernel_wb,
            "Kernel_WB_std": kernel_wb_std,
            "CB": cb,
            "CB_std": cb_std,
            "CB2": cb2,
            "CB2_std": cb2_std,
            "shell_fitted": shell_compo,
            "shell1": shell1,
            "shell2": shell2,
            "shell3": shell3,
            "shell4": shell4,
            "shell5": shell5,
            "pca_fitted": pca_compo,
            "pca1": pca1,
            "pca2": pca2,
            "pca3": pca3,
            "pca4": pca4,
            "pca5": pca5,
            "telluric_ew": tell_ew,
            "telluric_contrast": tell_contrast,
            "telluric_rv": tell_rv,
            "telluric_fwhm": tell_fwhm,
            "telluric_center": tell_center,
            "telluric_depth": tell_depth,
            "h2o_ew": h2o_ew,
            "h2o_contrast": h2o_contrast,
            "h2o_rv": h2o_rv,
            "h2o_fwhm": h2o_fwhm,
            "h2o_center": h2o_center,
            "h2o_depth": h2o_depth,
            "o2_ew": o2_ew,
            "o2_contrast": o2_contrast,
            "o2_rv": o2_rv,
            "o2_fwhm": o2_fwhm,
            "o2_center": o2_center,
            "o2_depth": o2_depth,
            "proxy_background": vec_background,
            "ccf_rv_para": para_rv,
            "ccf_depth_para": para_depth,
            "ccf_ew": ccf_ew,
            "ccf_contrast": ccf_contrast,
            "ccf_contrast_std": ccf_contrast_std,
            "ccf_vspan": ccf_vspan,
            "ccf_vspan_std": ccf_vspan_std,
            "ccf_rv": ccf_rv,
            "ccf_rv_std": ccf_rv_std,
            "ccf_rv_std_phot": ccf_svrad_phot,
            "ccf_fwhm": ccf_fwhm,
            "ccf_fwhm_std": ccf_fwhm_std,
            "ccf_offset": ccf_offset,
            "ccf_offset_std": ccf_offset_std,
            "Teff": t_eff,
            "Teff_std": t_eff_std,
            "ghost": ghost,
            "SAS": sas,
            "SAS_std": sas_std,
            "SAS1Y": sas1y,
            "SAS1Y_std": sas1y_std,
            "BIS": bis,
            "BIS_std": bis_std,
            "BIS2": bis2,
            "BIS2_std": bis2_std,
        }
    )

    if os.path.exists(directory + "Analyse_summary.p"):
        summary = pd.read_pickle(directory + "Analyse_summary.p")
        merge = summary.merge(dico, on="filename", how="outer", indicator=True)
        update = pd.DataFrame({"filename": np.array(merge["filename"])})
        merge = merge.drop(columns="filename")
        summary = summary.drop(columns="filename")
        dico = dico.drop(columns="filename")

        all_keys = np.array(summary.keys())
        for j in all_keys:
            try:
                merge.loc[merge["_merge"] != "left_only", j + "_x"] = merge.loc[
                    merge["_merge"] != "left_only", j + "_y"
                ]
                update[j] = merge[j + "_x"]
            except KeyError:
                update[j] = merge[j]

        all_keys = np.setdiff1d(np.array(dico.keys()), np.array(summary.keys()))
        if len(all_keys) != 0:
            for j in all_keys:
                update[j] = merge[j]
        dico = update

    dico = dico.sort_values(by="jdb").reset_index(drop=True)

    io.save_pickle(directory + "/Analyse_summary.p", dico)
    dico.to_csv(directory + "/Analyse_summary.csv")

    print("\n [INFO] Summary table updated")
