from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
from numpy import float64
from tqdm import tqdm

from ... import iofun
from ...stats import find_nearest

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_cut_spectrum(
    self: spec_time_series,
    wave_min: Optional[float64] = None,
    wave_max: Optional[Union[float, int]] = None,
) -> None:
    """Cut the spectrum time-series borders to reach the specified wavelength limits (included)

    There is no way to cancel this step ! Use it wisely."""

    logging.info("RECIPE : SPECTRA CROPING")

    directory = self.directory
    self.import_material()
    load = self.material
    old_wave = np.array(load["wave"])

    length = len(old_wave)
    idx_min = 0
    idx_max = len(old_wave)
    if wave_min is not None:
        idx_min = int(find_nearest(old_wave, wave_min)[0])
    if wave_max is not None:
        idx_max = int(find_nearest(old_wave, wave_max)[0] + 1)

    maps = glob.glob(self.dir_root + "CORRECTION_MAP/*.npy")
    if len(maps):
        for name in maps:
            correction_map = np.load(name)
            np.save(name, correction_map[:, idx_min:idx_max].astype("float32"))
            print("%s modified" % (name.split("/")[-1]))

    maps = glob.glob(self.dir_root + "WORKSPACE/CONTINUUM/*.npy")
    if len(maps):
        for name in maps:
            correction_map = np.load(name)
            np.save(name, correction_map[:, idx_min:idx_max].astype("float32"))
            print("%s modified" % (name.split("/")[-1]))

    maps = glob.glob(self.dir_root + "WORKSPACE/FLUX/*.npy")
    if len(maps):
        for name in maps:
            correction_map = np.load(name)
            np.save(name, correction_map[:, idx_min:idx_max].astype("float32"))
            print("%s modified" % (name.split("/")[-1]))

    new_wave = old_wave[idx_min:idx_max]
    wave_min = np.min(new_wave)
    wave_max = np.max(new_wave)

    load = load[idx_min:idx_max]
    load = load.reset_index(drop=True)
    iofun.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    file_ref = self.import_spectrum()
    old_wave = np.array(file_ref["wave"])
    length = len(old_wave)
    idx_min = 0
    idx_max = len(old_wave)
    if wave_min is not None:
        idx_min = int(find_nearest(old_wave, wave_min)[0])
    if wave_max is not None:
        idx_max = int(find_nearest(old_wave, wave_max)[0] + 1)

    new_wave = old_wave[idx_min:idx_max]
    wave_min = np.min(new_wave)
    wave_max = np.max(new_wave)

    for j in tqdm(files):
        file = pd.read_pickle(j)
        file["parameters"]["wave_min"] = wave_min
        file["parameters"]["wave_max"] = wave_max

        anchors_wave = file["matching_anchors"]["anchor_wave"]
        mask = (anchors_wave >= wave_min) & (anchors_wave <= wave_max)
        file["matching_anchors"]["anchor_index"] = (
            file["matching_anchors"]["anchor_index"][mask] - idx_min
        )
        file["matching_anchors"]["anchor_flux"] = file["matching_anchors"]["anchor_flux"][mask]
        file["matching_anchors"]["anchor_wave"] = file["matching_anchors"]["anchor_wave"][mask]

        anchors_wave = file["output"]["anchor_wave"]
        mask = (anchors_wave >= wave_min) & (anchors_wave <= wave_max)
        file["output"]["anchor_index"] = file["output"]["anchor_index"][mask] - idx_min
        file["output"]["anchor_flux"] = file["output"]["anchor_flux"][mask]
        file["output"]["anchor_wave"] = file["output"]["anchor_wave"][mask]

        fields = file.keys()
        for field in fields:
            if type(file[field]) == dict:
                sub_fields = file[field].keys()
                for sfield in sub_fields:
                    if type(file[field][sfield]) == np.ndarray:
                        if len(file[field][sfield]) == length:
                            file[field][sfield] = file[field][sfield][idx_min:idx_max]
            elif type(file[field]) == np.ndarray:
                if len(file[field]) == length:
                    file[field] = file[field][idx_min:idx_max]
        iofun.save_pickle(j, file)
