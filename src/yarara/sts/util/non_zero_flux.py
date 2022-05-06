from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd

from ...io import pickle_dump

if TYPE_CHECKING:
    from .. import spec_time_series


@overload
def yarara_non_zero_flux(
    self: spec_time_series, spectrum: None = None, min_value: Optional[float] = None
) -> None:
    pass


@overload
def yarara_non_zero_flux(
    self: spec_time_series, spectrum: np.ndarray, min_value: Optional[float] = None
) -> np.ndarray:
    pass


def yarara_non_zero_flux(
    self: spec_time_series,
    spectrum: Optional[np.ndarray] = None,
    min_value: Optional[float] = None,
) -> Optional[np.ndarray]:
    file_test = self.import_spectrum()
    hole_left = file_test["parameters"]["hole_left"]
    hole_right = file_test["parameters"]["hole_right"]
    grid = file_test["wave"]
    mask = (grid < hole_left) | (grid > hole_right)

    directory = self.directory

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    if spectrum is None:
        for i, j in enumerate(files):
            file = pd.read_pickle(j)
            flux = file["flux"]
            zero = flux == 0
            if min_value is None:
                min_value = np.min(flux[flux != 0])
            flux[mask & zero] = min_value
            pickle_dump(file, open(j, "wb"))
    else:
        logging.info("Removing null values of the spectrum")
        zero = spectrum <= 0
        if min_value is None:
            min_value = np.min(spectrum[spectrum > 0])
        spectrum[mask & zero] = min_value
        return spectrum
