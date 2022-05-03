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

from .. import analysis, io, util

if TYPE_CHECKING:
    from . import spec_time_series


def spectrum(
    self: spec_time_series,
    num: int = 0,
    sub_dico: str = "matching_diff",
    continuum: str = "linear",
    norm: bool = False,
    planet: bool = False,
    color_correction: bool = False,
) -> tableXY:
    """
    Produce a tableXY spectrum by specifying its index number

    Parameters
    ----------
    num : index of the spectrum to extract
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    norm : True/False button to normalise the spectrum

    Returns
    -------
    Return the tableXY spectrum object

    """

    if color_correction:
        self.import_material()
        color_corr = np.array(self.material.correction_factor)
    else:
        color_corr = 1

    array = self.import_spectrum(num=num)
    kw = "_planet" * planet

    flux = array["flux" + kw]
    flux_std = array["flux_err"]
    wave = array["wave"]
    conti1, continuum = self.import_sts_flux(load=["matching_diff", sub_dico], num=num)

    correction = conti1 / continuum
    spectrum = analysis.tableXY(
        wave, flux * correction * color_corr, flux_std * correction * color_corr
    )
    if norm:
        continuum_std = array["continuum_err"]
        flux_norm, flux_norm_std = util.flux_norm_std(flux, flux_std, continuum, continuum_std)
        spectrum_norm = analysis.tableXY(wave, flux_norm * color_corr, flux_norm_std * color_corr)
        return spectrum_norm
    else:
        return spectrum


# =============================================================================
# IMPORT ALL RASSINE DICTIONNARY
# =============================================================================


def import_rassine_output(
    self: spec_time_series, return_name: bool = False, kw1: None = None, kw2: None = None
) -> Any:
    """
    Import all the RASSINE dictionnaries in a list

    Parameters
    ----------
    return_name : True/False to also return the filenames

    Returns
    -------
    Return the list containing all thedictionnary

    """

    directory = self.directory

    files = glob.glob(directory + "RASSI*.p")
    if len(files) <= 1:  # 1 when merged directory
        print("No RASSINE file found in the directory : %s" % (directory))
        if return_name:
            return [], []
        else:
            return []
    else:
        files = np.sort(files)
        file = []
        for i, j in enumerate(files):
            file.append(pd.read_pickle(j))

            if kw1 is not None:
                file[-1] = file[-1][kw1]

            if kw2 is not None:
                file[-1] = file[-1][kw2]

        if return_name:
            return file, files
        else:
            return file


# region INFO REDUCTION


def import_info_reduction(self: spec_time_series) -> None:
    if self.info_reduction_ut != os.path.getmtime(
        self.dir_root + "REDUCTION_INFO/Info_reduction.p"
    ):
        self.info_reduction = pd.read_pickle(self.dir_root + "REDUCTION_INFO/Info_reduction.p")
        self.info_reduction_ut = os.path.getmtime(
            self.dir_root + "REDUCTION_INFO/Info_reduction.p"
        )


def update_info_reduction(self: spec_time_series) -> None:
    io.pickle_dump(
        self.info_reduction, open(self.dir_root + "REDUCTION_INFO/Info_reduction.p", "wb")
    )


# endregion


def import_sts_flux(
    self: spec_time_series,
    load: Sequence[str] = ["flux", "flux_err", "matching_diff"],
    num: Optional[int] = None,
) -> Sequence[np.ndarray]:
    all_elements = []
    for l in load:
        if len(l.split("matching_")) > 1:
            all_elements.append(np.load(self.directory + "CONTINUUM/Continuum_%s.npy" % (l)))
        else:
            all_elements.append(np.load(self.directory + "FLUX/Flux_%s.npy" % (l)))

    if num is not None:
        all_elements = [elem[num] for elem in all_elements]
    return all_elements


def yarara_add_step_dico(
    self: spec_time_series,
    sub_dico: str,
    step: int,
    sub_dico_used: Optional[str] = None,
    chain: bool = False,
) -> None:
    """Add the step kw for the dico chain, if chain is set to True numbered the full chain"""
    self.import_table()
    self.import_info_reduction()

    table = self.table
    if chain:
        self.import_dico_chain(sub_dico)
        sub_dico = self.dico_chain[::-1]
        step = np.arange(len(sub_dico))
    else:
        sub_dico = [sub_dico]
        step = [step]
        sub_dico_used = [sub_dico_used]

    for sub, s, sb in zip(sub_dico, step, sub_dico_used):
        self.info_reduction[sub] = {"sub_dico_used": sb, "step": s, "valid": True}
    self.update_info_reduction()


def yarara_add_ccf_entry(self, kw, default_value=1):
    self.import_ccf()
    for mask in list(self.table_ccf.keys()):
        if mask != "star_info":
            for sb in list(self.table_ccf[mask].keys()):
                self.table_ccf[mask][sb]["table"][kw] = default_value
    io.pickle_dump(self.table_ccf, open(self.directory + "Analyse_ccf.p", "wb"))


# region IMPORT SUMMARY TABLE


def import_star_info(self: spec_time_series) -> None:
    self.star_info = pd.read_pickle(
        self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"
    )


def import_table(self: spec_time_series) -> None:
    if self.table_ut != os.path.getmtime(self.directory + "Analyse_summary.p"):
        self.table = pd.read_pickle(self.directory + "Analyse_summary.p")
        self.table_ut = os.path.getmtime(self.directory + "Analyse_summary.p")


def import_material(self: spec_time_series) -> None:
    if self.material_ut != os.path.getmtime(self.directory + "Analyse_material.p"):
        self.material = pd.read_pickle(self.directory + "Analyse_material.p")
        self.material_ut = os.path.getmtime(self.directory + "Analyse_material.p")


# endregion

# region IMPORT THE FULL DICO CHAIN


def import_dico_tree(self: spec_time_series) -> None:
    self.import_info_reduction()
    file_test = self.info_reduction
    kw = list(file_test.keys())
    kw_kept = []
    kw_chain = []
    for k in kw:
        if len(k.split("matching_")) == 2:
            kw_kept.append(k)
    kw_kept = np.array(kw_kept)

    info = []
    for n in kw_kept:
        try:
            s = file_test[n]["step"]
            dico = file_test[n]["sub_dico_used"]
            info.append([n, s, dico])
        except:
            pass
    info = pd.DataFrame(info, columns=["dico", "step", "dico_used"])
    self.dico_tree = info.sort_values(by="step")


# endregion

# =============================================================================
# IMPORT a RANDOM SPECTRUM
# =============================================================================


def import_spectrum(self: spec_time_series, num: Optional[int64] = None) -> Dict[str, Any]:
    """
    Import a pickle file of a spectrum to get fast common information shared by all spectra

    Parameters
    ----------
    num : index of the spectrum to extract (if None random selection)

    Returns
    -------
    Return the open pickle file

    """

    directory = self.directory
    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)
    if not len(files):
        files = glob.glob(directory.replace("WORKSPACE", "TEMP") + "RASSI*.p")

    if num is None:
        try:
            num = np.random.choice(np.arange(len(files)), 1)
            file = files[num][0]
        except:
            file = files[0]

    else:
        try:
            file = files[num]
        except:
            file = files[0]
    return pd.read_pickle(file)


def yarara_star_info(
    self: spec_time_series,
    Rv_sys: None = None,
    simbad_name: None = None,
    magB: None = None,
    magV: None = None,
    magR: None = None,
    BV: None = None,
    VR: None = None,
    sp_type: None = None,
    Mstar: None = None,
    Rstar: None = None,
    Vsini: None = None,
    Vmicro: None = None,
    Teff: None = None,
    log_g: None = None,
    FeH: None = None,
    Prot: None = None,
    Fwhm: Optional[List[Union[str, float64]]] = None,
    Contrast: Optional[List[Union[str, float64]]] = None,
    CCF_delta: None = None,
    Pmag: None = None,
    stellar_template: None = None,
) -> None:

    kw = [
        "Rv_sys",
        "Simbad_name",
        "Sp_type",
        "magB",
        "magV",
        "magR",
        "BV",
        "VR",
        "Mstar",
        "Rstar",
        "Vsini",
        "Vmicro",
        "Teff",
        "Log_g",
        "FeH",
        "Prot",
        "FWHM",
        "Contrast",
        "Pmag",
        "stellar_template",
        "CCF_delta",
    ]
    val = [
        Rv_sys,
        simbad_name,
        sp_type,
        magB,
        magV,
        magR,
        BV,
        VR,
        Mstar,
        Rstar,
        Vsini,
        Vmicro,
        Teff,
        log_g,
        FeH,
        Prot,
        Fwhm,
        Contrast,
        Pmag,
        stellar_template,
        CCF_delta,
    ]

    self.import_star_info()
    self.import_table()
    self.import_material()

    table = self.table
    snr = np.array(table["snr"]).argmax()
    file_test = self.import_spectrum(num=snr)

    for i, j in zip(kw, val):
        if j is not None:
            if type(j) != list:
                j = ["fixed", j]
            if i in self.star_info.keys():
                self.star_info[i][j[0]] = j[1]
            else:
                self.star_info[i] = {j[0]: j[1]}

    # TODO:
    # Here, we removed the Gray temperature and the MARCS atlas atmospheric model
    # initialization

    io.pickle_dump(
        self.star_info,
        open(self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p", "wb"),
    )


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


# ===
# Other things
# ===


def yarara_obs_info(
    self: spec_time_series,
    kw: DataFrame = [None, None],
    jdb: None = None,
    berv: None = None,
    rv: None = None,
    airmass: None = None,
    texp: None = None,
    seeing: None = None,
    humidity: None = None,
) -> None:
    """
    Add some observationnal information in the RASSINE files and produce a summary table

    Parameters
    ----------
    kw: list-like with format [keyword,array]
    jdb : array-like with same size than the number of files in the directory
    berv : array-like with same size than the number of files in the directory
    rv : array-like with same size than the number of files in the directory
    airmass : array-like with same size than the number of files in the directory
    texp : array-like with same size than the number of files in the directory
    seeing : array-like with same size than the number of files in the directory
    humidity : array-like with same size than the number of files in the directory

    """

    directory = self.directory

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    nb = len(files)

    if type(kw) == pd.core.frame.DataFrame:  # in case of a pandas dataframe
        kw = [list(kw.keys()), [i for i in np.array(kw).T]]
    else:
        try:
            if len(kw[1]) == 1:
                kw[1] = [kw[1][0]] * nb
        except TypeError:
            kw[1] = [kw[1]] * nb

        kw[0] = [kw[0]]
        kw[1] = [kw[1]]

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        for kw1, kw2 in zip(kw[0], kw[1]):
            if kw1 is not None:
                if len(kw1.split("ccf_")) - 1:
                    file["ccf_gaussian"][kw1.split("ccf_")[1]] = kw2[i]
                else:
                    file["parameters"][kw1] = kw2[i]
        if jdb is not None:
            file["parameters"]["jdb"] = jdb[i]
        if berv is not None:
            file["parameters"]["berv"] = berv[i]
        if rv is not None:
            file["parameters"]["rv"] = rv[i]
        if airmass is not None:
            file["parameters"]["airmass"] = airmass[i]
        if texp is not None:
            file["parameters"]["texp"] = texp[i]
        if seeing is not None:
            file["parameters"]["seeing"] = seeing[i]
        if humidity is not None:
            file["parameters"]["humidity"] = humidity[i]
        io.save_pickle(j, file)

    self.yarara_analyse_summary()


def supress_time_spectra(
    self: spec_time_series,
    liste: Optional[ndarray] = None,
    jdb_min: None = None,
    jdb_max: None = None,
    num_min: None = None,
    num_max: None = None,
    supress: bool = False,
    name_ext: str = "temp",
) -> None:
    """
    Supress spectra according to time

    Parameters
    ----------

    jdb or num are both inclusive in lower and upper limit
    snr_cutoff : treshold value of the cutoff below which spectra are removed
    supress : True/False to delete of simply hide the files

    """

    self.import_table()
    jdb = self.table["jdb"]
    name = self.table["filename"]
    directory = "/".join(np.array(name)[0].split("/")[:-1])

    if num_min is not None:
        jdb_min = jdb[num_min]

    if num_max is not None:
        jdb_max = jdb[num_max]

    if jdb_min is None:
        jdb_min = jdb_max

    if jdb_max is None:
        jdb_max = jdb_min

    if (num_min is None) & (num_max is None) & (jdb_min is None) & (jdb_max is None):
        jdb_min = 1e9
        jdb_max = -1e9

    if liste is None:
        mask = np.array((jdb >= jdb_min) & (jdb <= jdb_max))
    else:
        if type(liste[0]) == np.bool_:
            mask = liste
        else:
            mask = np.in1d(np.arange(len(jdb)), liste)

    if sum(mask):
        idx = np.arange(len(jdb))[mask]
        print(" [INFO] Following spectrum indices will be supressed : ", idx)
        print(" [INFO] Number of spectrum supressed : %.0f \n" % (sum(mask)))
        maps = glob.glob(self.dir_root + "CORRECTION_MAP/*.p")
        if len(maps):
            for names in maps:
                correction_map = pd.read_pickle(names)
                correction_map["correction_map"] = np.delete(
                    correction_map["correction_map"], idx, axis=0
                )
                io.pickle_dump(correction_map, open(names, "wb"))
                print("%s modified" % (names.split("/")[-1]))

        name = name[mask]

        files_to_process = np.sort(np.array(name))
        for j in files_to_process:
            if supress:
                print("File deleted : %s " % (j))
                os.system("rm " + j)
            else:
                new_name = name_ext + "_" + j.split("/")[-1]
                print("File renamed : %s " % (directory + "/" + new_name))
                os.system("mv " + j + " " + directory + "/" + new_name)

        os.system("rm " + directory + "/Analyse_summary.p")
        os.system("rm " + directory + "/Analyse_summary.csv")
        self.yarara_analyse_summary()

        self.supress_time_RV(mask)


# region new_database_format


def yarara_exploding_pickle(self: spec_time_series) -> None:
    self.import_table()
    file_test = self.import_spectrum()

    sub_dico = util.string_contained_in(list(file_test.keys()), "matching")[1]

    sub_dico_to_ventile = []
    sub_dico_to_delete = []
    for sb in sub_dico:
        if "continuum_linear" in file_test[sb].keys():
            sub_dico_to_ventile.append(sb)
        else:
            sub_dico_to_delete.append(sb)

    files = np.array(self.table["filename"])

    c = -1
    for sb in sub_dico_to_ventile:
        c += 1
        print(
            "\n [INFO] Venting sub_dico %s, number of dico remaining : %.0f \n"
            % (sb, len(sub_dico_to_ventile) - c)
        )
        continua = []
        for f in tqdm(files):
            file = pd.read_pickle(f)
            continua.append(file[sb]["continuum_linear"])
            if (sb != "matching_anchors") & (sb != "matching_diff"):
                del file[sb]
            io.pickle_dump(file, open(f, "wb"))
        continua = np.array(continua)
        fname = self.dir_root + "WORKSPACE/CONTINUUM/Continuum_%s.npy" % (sb)
        np.save(fname, continua)

    c = -1
    for sb in sub_dico_to_delete:
        c += 1
        print(
            "\n [INFO] Deleting sub_dico %s, number of dico remaining : %.0f \n"
            % (sb, len(sub_dico_to_delete) - c)
        )
        for f in tqdm(files):
            file = pd.read_pickle(f)
            del file[sb]
            io.pickle_dump(file, open(f, "wb"))

    kw = list(util.string_contained_in(list(file_test.keys()), "flux")[1])
    kw2 = list(util.string_contained_in(list(file_test.keys()), "continuum")[1])

    c = -1
    for sb in kw + kw2:
        c += 1
        print("\n [INFO] Venting key word %s \n" % (sb))
        flux = []
        for f in tqdm(files):
            file = pd.read_pickle(f)
            flux.append(file[sb])
            io.pickle_dump(file, open(f, "wb"))
        flux = np.array(flux)
        fname = self.dir_root + "WORKSPACE/FLUX/Flux_%s.npy" % (sb)
        np.save(fname, flux)


# endregion
