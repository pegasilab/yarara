from __future__ import annotations

import glob as glob
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_substract_map(
    self, flux_init: NDArray[np.float64], map_name: str, correction_factor: bool = False
):
    self.import_material()
    load = self.material
    correction_map = 0
    corr = np.array(load["correction_factor"]) * int(correction_factor) + (
        1 - int(correction_factor)
    )
    file_map = glob.glob(self.dir_root + "CORRECTION_MAP/map_matching_" + map_name + ".npy")
    if len(file_map):
        correction_map = np.load(file_map[0])
    else:
        print(" [WARNING] Correction map %s does not exist !" % ("matching_" + map_name))
    flux_modified = flux_init - correction_map * corr
    return flux_modified


def yarara_add_map(
    self, flux_init: NDArray[np.float64], map_name: str, correction_factor: bool = False
):
    self.import_material()
    load = self.material
    correction_map = 0
    corr = np.array(load["correction_factor"]) * int(correction_factor) + (
        1 - int(correction_factor)
    )
    file_map = glob.glob(self.dir_root + "CORRECTION_MAP/map_matching_" + map_name + ".npy")
    if len(file_map):
        correction_map = np.load(file_map[0])
    else:
        print(" [WARNING] Correction map %s does not exist !" % ("matching_" + map_name))
    flux_modified = flux_init + correction_map * corr
    return flux_modified
