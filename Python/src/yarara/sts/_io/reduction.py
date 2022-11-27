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

from ... import iofun, util
from ...analysis import tableXY

if TYPE_CHECKING:
    from .. import spec_time_series


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
    iofun.pickle_dump(
        self.info_reduction, open(self.dir_root + "REDUCTION_INFO/Info_reduction.p", "wb")
    )
