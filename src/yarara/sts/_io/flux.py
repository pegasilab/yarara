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


def import_sts_flux(
    self: spec_time_series,
    load: Sequence[str] = ["flux", "flux_err", "matching_diff"],
    num: Optional[int] = None,
) -> Sequence[np.ndarray]:
    all_elements = []
    for l in load:
        if len(l.split("matching_")) > 1:
            all_elements.append(np.load(self.directory + f"CONTINUUM/Continuum_{l}.npy"))
        else:
            all_elements.append(np.load(self.directory + f"FLUX/Flux_{l}.npy"))

    if num is not None:
        all_elements = [elem[num] for elem in all_elements]
    return all_elements
