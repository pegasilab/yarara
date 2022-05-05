from __future__ import annotations

import datetime
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from numpy import ndarray
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from ... import io, util
from ...analysis import tableXY
from ...io import pickle_dump
from ...paths import paths, root
from ...stats import IQ, find_nearest, identify_nearest
from ...util import assert_never
from ...util import ccf as ccf_fun
from ...util import doppler_r

if TYPE_CHECKING:
    from .. import spec_time_series


def read_ccf_mask(self: spec_time_series, mask_name: str) -> NDArray[np.float64]:
    logging.info("Reading CCF mask : %s \n" % (mask_name))
    mask_path = root + "/Python/MASK_CCF/" + mask_name + ".txt"
    mask = np.genfromtxt(mask_path)
    mask = np.array([0.5 * (mask[:, 0] + mask[:, 1]), mask[:, 2]]).T
    return mask


def yarara_ccf_save(self: spec_time_series, mask: str, sub_dico: str):
    self.import_ccf()
    table = self.table_ccf["CCF_" + mask]

    all_ccf_saved = self.all_ccf_saved[sub_dico]
    vrad = all_ccf_saved[0]
    ccfs = all_ccf_saved[1]
    ccfs_std = all_ccf_saved[2]
    creation_date = table[sub_dico]["creation_date"]
    table_ccf_moments = table[sub_dico]["table"][
        [
            "jdb",
            "rv",
            "rv_std",
            "contrast",
            "contrast_std",
            "fwhm",
            "fwhm_std",
            "bisspan",
            "bisspan_std",
            "ew",
            "ew_std",
        ]
    ]

    if not os.path.exists(self.directory + "Analyse_ccf_saved.p"):
        file_to_save = {}
    else:
        file_to_save = pd.read_pickle(self.directory + "/Analyse_ccf_saved.p")

    ccf_infos = {
        "ccf_vrad": vrad,
        "ccf_flux": ccfs,
        "ccf_flux_std": ccfs_std,
        "table": table_ccf_moments,
        "creation_date": creation_date,
    }

    try:
        file_to_save["CCF_" + mask][sub_dico] = ccf_infos
    except KeyError:
        file_to_save["CCF_" + mask] = {sub_dico: ccf_infos}

    pickle_dump(file_to_save, open(self.directory + "/Analyse_ccf_saved.p", "wb"))
