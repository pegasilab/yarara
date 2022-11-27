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


def import_table(self: spec_time_series) -> None:
    if self.table_ut != os.path.getmtime(self.directory + "Analyse_summary.p"):
        self.table = pd.read_pickle(self.directory + "Analyse_summary.p")
        self.table_ut = os.path.getmtime(self.directory + "Analyse_summary.p")
