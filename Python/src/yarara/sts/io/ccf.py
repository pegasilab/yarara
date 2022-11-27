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


def yarara_add_ccf_entry(self, kw, default_value=1):
    self.import_ccf()
    for mask in list(self.table_ccf.keys()):
        if mask != "star_info":
            for sb in list(self.table_ccf[mask].keys()):
                self.table_ccf[mask][sb]["table"][kw] = default_value
    io.pickle_dump(self.table_ccf, open(self.directory + "Analyse_ccf.p", "wb"))


def import_ccf(self):
    if self.table_ccf_ut != os.path.getmtime(self.directory + "Analyse_ccf.p"):
        self.table_ccf = pd.read_pickle(self.directory + "Analyse_ccf.p")
        self.table_ccf_ut = os.path.getmtime(self.directory + "Analyse_ccf.p")
    else:
        pass
