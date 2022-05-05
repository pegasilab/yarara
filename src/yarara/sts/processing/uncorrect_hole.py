from __future__ import annotations

import datetime
import glob as glob
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from colorama import Fore
from numpy import ndarray
from scipy.interpolate import interp1d
from tqdm import tqdm

from yarara.analysis import tableXY

from ... import io, util
from ...analysis import table, tableXY
from ...paths import root
from ...plots import auto_axis, my_colormesh
from ...stats import IQ, find_nearest, identify_nearest, match_nearest, smooth2d
from ...util import ccf as ccf_fun
from ...util import doppler_r, get_phase, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def uncorrect_hole(
    self: spec_time_series,
    conti: ndarray,
    conti_ref: ndarray,
    values_forbidden: List[Union[int, float]] = [0, np.inf],
) -> ndarray:
    file_test = self.import_spectrum()
    wave = np.array(file_test["wave"])
    hl = file_test["parameters"]["hole_left"]
    hr = file_test["parameters"]["hole_right"]

    if hl != -99.9:
        i1 = int(find_nearest(wave, hl)[0])
        i2 = int(find_nearest(wave, hr)[0])
        conti[:, i1 - 1 : i2 + 2] = conti_ref[:, i1 - 1 : i2 + 2].copy()

    for l in values_forbidden:
        conti[conti == l] = conti_ref[conti == l].copy()

    return conti
