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
from ...paths import root

if TYPE_CHECKING:
    from .. import spec_time_series


def import_star_info(self: spec_time_series) -> None:
    self.star_info = pd.read_pickle(
        self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"
    )


def yarara_star_info(
    self: spec_time_series,
    Rv_sys: Optional[List[Union[str, float64]]] = None,
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
    snr: int = np.array(table["snr"]).argmax()  # type: ignore
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

