from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from .. import io
from .. import my_classes as myc
from .. import my_functions as myf

if TYPE_CHECKING:
    from ..my_rassine_tools import spec_time_series


def uncorrect_hole(self: spec_time_series, conti, conti_ref, values_forbidden=[0, np.inf]):
    file_test = self.import_spectrum()
    wave = np.array(file_test["wave"])
    hl = file_test["parameters"]["hole_left"]
    hr = file_test["parameters"]["hole_right"]

    if hl != -99.9:
        i1 = int(myf.find_nearest(wave, hl)[0])
        i2 = int(myf.find_nearest(wave, hr)[0])
        conti[:, i1 - 1 : i2 + 2] = conti_ref[:, i1 - 1 : i2 + 2].copy()

    for l in values_forbidden:
        conti[conti == l] = conti_ref[conti == l].copy()

    return conti
