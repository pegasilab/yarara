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


def yarara_obs_info(self: spec_time_series, df: DataFrame) -> None:
    """
    Add some observationnal information in the RASSINE files and produce a summary table

    Parameters
    ----------
    kw: list-like with format [keyword,array]

    """

    directory = self.directory

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    kw: List[List[str]] = [list(df.keys()), [i for i in np.array(df).T]]
    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        for kw1, kw2 in zip(kw[0], kw[1]):
            if kw1 is not None:
                if len(kw1.split("ccf_")) - 1:
                    file["ccf_gaussian"][kw1.split("ccf_")[1]] = kw2[i]
                else:
                    file["parameters"][kw1] = kw2[i]
        io.save_pickle(j, file)

    self.yarara_analyse_summary()
