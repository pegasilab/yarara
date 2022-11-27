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


def import_material(self: spec_time_series) -> None:
    if self.material_ut != os.path.getmtime(self.directory + "Analyse_material.p"):
        self.material = pd.read_pickle(self.directory + "Analyse_material.p")
        self.material_ut = os.path.getmtime(self.directory + "Analyse_material.p")
