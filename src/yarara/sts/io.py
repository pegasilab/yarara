from __future__ import annotations

import glob as glob
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .. import io

if TYPE_CHECKING:
    from ..my_rassine_tools import spec_time_series

# =============================================================================
# IMPORT ALL RASSINE DICTIONNARY
# =============================================================================


def import_rassine_output(self: spec_time_series, return_name=False, kw1=None, kw2=None):
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
            self.debug = j
            file.append(pd.read_pickle(j))

            if kw1 is not None:
                file[-1] = file[-1][kw1]

            if kw2 is not None:
                file[-1] = file[-1][kw2]

        if return_name:
            return file, files
        else:
            return file


# =============================================================================
# IMPORT SUMMARY TABLE
# =============================================================================


def import_star_info(self: spec_time_series) -> None:
    self.star_info = pd.read_pickle(
        self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"
    )


def import_table(self: spec_time_series) -> None:
    self.table = pd.read_pickle(self.directory + "Analyse_summary.p")


def import_material(self: spec_time_series) -> None:
    self.material = pd.read_pickle(self.directory + "Analyse_material.p")


# =============================================================================
# IMPORT THE FULL DICO CHAIN
# =============================================================================

# io
def import_dico_tree(self: spec_time_series):
    file_test = self.import_spectrum()
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
            s = file_test[n]["parameters"]["step"]
            dico = file_test[n]["parameters"]["sub_dico_used"]
            info.append([n, s, dico])
        except:
            pass
    info = pd.DataFrame(info, columns=["dico", "step", "dico_used"])
    self.dico_tree = info.sort_values(by="step")


# =============================================================================
# IMPORT a RANDOM SPECTRUM
# =============================================================================


def import_spectrum(self: spec_time_series, num=None):
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
    Rv_sys=None,
    simbad_name=None,
    magB=None,
    magV=None,
    magR=None,
    BV=None,
    VR=None,
    sp_type=None,
    Mstar=None,
    Rstar=None,
    Vsini=None,
    Vmicro=None,
    Teff=None,
    log_g=None,
    FeH=None,
    Prot=None,
    Fwhm=None,
    Contrast=None,
    CCF_delta=None,
    Pmag=None,
    stellar_template=None,
):

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


# ===
# Other things
# ===


def yarara_obs_info(
    self: spec_time_series,
    kw=[None, None],
    jdb=None,
    berv=None,
    rv=None,
    airmass=None,
    texp=None,
    seeing=None,
    humidity=None,
):
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
