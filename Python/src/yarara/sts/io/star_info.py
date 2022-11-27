from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ... import io, util
from ...analysis import tableXY
from ...paths import root

if TYPE_CHECKING:
    from .. import spec_time_series


def import_star_info(self: spec_time_series) -> None:
    self.star_info = pd.read_pickle(
        self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"
    )


def find_stellar_mass_radius(Teff, sp_type="G2V"):
    """Habets 1981 calibration curve"""
    lim = 0
    for k in sp_type[::-1]:
        try:
            int(k)
            break
        except:
            lim += 1

    class_lum = sp_type[len(sp_type) - lim :]
    if class_lum == "":
        class_lum = "V"
    if class_lum != "V":
        class_lum = "IV"
    calib = pd.read_pickle(root + "/Python/Material/logT_logM_logR.p")[class_lum]
    curve_mass = tableXY(10 ** calib["log(T)"], 10 ** calib["log(M/Ms)"])
    curve_radius = tableXY(10 ** calib["log(T)"], 10 ** calib["log(R/Rs)"])
    curve_mass.interpolate(new_grid=np.array([Teff]))
    curve_radius.interpolate(new_grid=np.array([Teff]))
    m = curve_mass.y
    r = curve_radius.y
    log_g = 2 + np.log10(6.67e-11 * (m * 1.98e30) / (r * 696342000) ** 2)
    return m[0], r[0], log_g[0]


def yarara_star_info(
    self,
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
    Contam_BERV=None,
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
        "Contam_BERV",
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
        Contam_BERV,
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

    try:
        self.star_info["Teff"]["Gray"] = file_test["parameters"]["Teff_gray"]
        M, R, logg = find_stellar_mass_radius(
            file_test["parameters"]["Teff_gray"],
            sp_type=self.star_info["Sp_type"]["fixed"],
        )
        self.star_info["Mstar"]["Gray"] = np.round(M, 2)
        self.star_info["Rstar"]["Gray"] = np.round(R, 2)
        self.star_info["Log_g"]["Gray"] = np.round(logg, 2)

    except:
        pass

    try:
        m = self.model_atmos["MARCS"]
        a = self.model_atmos["ATLAS"]

        self.star_info["Teff"]["MARCS"] = int(m[0].split("_")[0][1:])
        self.star_info["Log_g"]["MARCS"] = np.round(float(m[0].split("_")[1][1:]), 2)

        M, R, logg = find_stellar_mass_radius(
            int(m[0].split("_")[0][1:]), sp_type=self.star_info["Sp_type"]["fixed"]
        )
        if abs(logg - np.round(m[0].split("_")[1][1:], 2)) / logg < 0.2:
            self.star_info["Mstar"]["MARCS"] = np.round(M, 2)
            self.star_info["Rstar"]["MARCS"] = np.round(R, 2)

        self.star_info["Teff"]["ATLAS"] = int(a[0].split("_")[0][1:])
        self.star_info["Log_g"]["ATLAS"] = np.round(float(a[0].split("_")[1][1:]), 2)

        M, R, logg = find_stellar_mass_radius(
            int(a[0].split("_")[0][1:]), sp_type=self.star_info["Sp_type"]["fixed"]
        )
        if abs(logg - np.round(m[0].split("_")[1][1:], 2)) / logg < 0.2:
            self.star_info["Mstar"]["ATLAS"] = np.round(M, 2)
            self.star_info["Rstar"]["ATLAS"] = np.round(R, 2)
    except:
        pass

    io.pickle_dump(
        self.star_info,
        open(self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p", "wb"),
    )
