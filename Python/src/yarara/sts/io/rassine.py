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


# =============================================================================
# IMPORT ALL RASSINE DICTIONNARY
# =============================================================================


def import_rassine_output(
    self: spec_time_series, return_name: bool = False, kw1: None = None, kw2: None = None
) -> Any:
    """
    Import all the RASSINE dictionnaries in a list

    Parameters
    ----------
    return_name : True/False to also return the filenames

    Returns
    -------
    Return the list containing all thedictionnary

    """

    directory = self.directory

    files = glob.glob(directory + "RASSI*.p")
    if len(files) <= 1:  # 1 when merged directory
        print("No RASSINE file found in the directory : %s" % (directory))
        if return_name:
            return [], []
        else:
            return []
    else:
        files = np.sort(files)
        file = []
        for i, j in enumerate(files):
            file.append(pd.read_pickle(j))

            if kw1 is not None:
                file[-1] = file[-1][kw1]

            if kw2 is not None:
                file[-1] = file[-1][kw2]

        if return_name:
            return file, files
        else:
            return file
