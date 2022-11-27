from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ... import iofun
from ...analysis import tableXY
from ...paths import root
from ...stats import find_nearest, match_nearest
from ...util import doppler_r, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_produce_mask_frog(
    self: spec_time_series, frog_file: str = root + "/Python/Material/Ghost_HARPS03.p"
) -> None:
    """
    Correction of the stitching/ghost on the spectrum by PCA fitting

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    frog_file : files containing the wavelength of the stitching
    """

    print_box("\n---- RECIPE : MASK GHOST/STITCHING/THAR WITH FROG ----\n")

    directory = self.directory
    kw = "_planet" * self.planet
    if kw != "":
        logging.info("PLANET ACTIVATED")

    self.import_table()
    self.import_material()
    load = self.material
    dwave = np.nanmedian(np.diff(load.wave))

    file_test = self.import_spectrum()
    grid = file_test["wave"]

    berv_max = self.table["berv" + kw].max()
    berv_min = self.table["berv" + kw].min()
    imin = find_nearest(grid, doppler_r(grid[0], np.max(abs(self.table.berv)) * 1000)[0])[0][0] + 1
    imax = find_nearest(grid, doppler_r(grid[-1], np.max(abs(self.table.berv)) * 1000)[1])[0][0]

    # extract frog table
    frog_table = pd.read_pickle(frog_file)
    frog_table_jdb: NDArray[np.float64] = frog_table["jdb"]
    berv_file: NDArray[np.float64] = self.yarara_get_berv_value(frog_table_jdb)

    # ghost
    for correction in ["stitching", "ghost_a", "ghost_b", "thar"]:
        if correction in frog_table.keys():
            if correction == "stitching":
                logging.info("Producing the stitching mask...")

                wave2d = frog_table["wave"]
                pxl2d = (
                    np.ones(np.shape(wave2d)[0])
                    * np.arange(1, 1 + np.shape(wave2d)[1])[:, np.newaxis]
                ).T
                wave_stitching = np.hstack(wave2d)
                gap_stitching = np.hstack(frog_table["stitching"])
                pxl_stitching = np.hstack(pxl2d)  # type: ignore

                mask = gap_stitching != 0

                wave_stitching = wave_stitching[mask]
                gap_stitching = gap_stitching[mask]
                pxl_stitching = pxl_stitching[mask]

                vec = tableXY(wave_stitching, gap_stitching)
                vec.order()
                stitching = vec.x

                vec = tableXY(wave_stitching, pxl_stitching)
                vec.order()
                pixels = vec.y

                stitching_b0 = doppler_r(stitching, 0 * berv_file * 1000)[0]
                # all_stitch = myf.doppler_r(stitching_b0, berv*1000)[0]

                match_stitching = match_nearest(grid, stitching_b0)
                match_stitching = match_stitching[abs(match_stitching[:, -1]) < dwave * 2]

                indext = match_stitching[:, 0].astype("int")

                wavet_delta = np.zeros(len(grid))
                wavet_delta[indext] = 1
                wavet_pxl = np.zeros(len(grid))
                wavet_pxl[indext] = pixels[match_stitching[:, 1].astype("int")]

                wavet = grid[indext]
                max_t = wavet * ((1 + 1.55e-8) * (1 + (berv_max - 0.0 * berv_file) / 299792.458))
                min_t = wavet * ((1 + 1.55e-8) * (1 + (berv_min - 0.0 * berv_file) / 299792.458))

                mask_stitching = np.sum(
                    (grid > min_t[:, np.newaxis]) & (grid < max_t[:, np.newaxis]),
                    axis=0,
                ).astype("bool")
                self.stitching_zones = mask_stitching

                mask_stitching[0:imin] = 0
                mask_stitching[imax:] = 0

                load["stitching"] = mask_stitching.astype("int")
                load["stitching_delta"] = wavet_delta.astype("int")
                load["stitching_pxl"] = wavet_pxl.astype("int")
            else:
                if correction == "ghost_a":
                    print("\n [INFO] Producing the ghost mask A...")
                elif correction == "ghost_b":
                    print("\n [INFO] Producing the ghost mask B...")
                elif correction == "thar":
                    print("\n [INFO] Producing the thar mask...")

                contam = frog_table[correction]
                mask = np.zeros(len(grid))
                wave_s2d = []
                order_s2d = []
                for order in np.arange(len(contam)):
                    vec = tableXY(
                        doppler_r(frog_table["wave"][order], 0.0 * berv_file * 1000)[0],
                        contam[order],
                        0 * contam[order],
                    )
                    vec.order()
                    vec.y[0:2] = 0
                    vec.y[-2:] = 0
                    begin = int(find_nearest(grid, vec.x[0])[0])
                    end = int(find_nearest(grid, vec.x[-1])[0])
                    sub_grid = grid[begin:end]
                    vec.interpolate(new_grid=sub_grid, method="linear", interpolate_x=False)
                    model = np.array(
                        load["reference_spectrum"][begin:end]
                        * load["correction_factor"][begin:end]
                    )
                    model[model == 0] = 1
                    contam_cumu = vec.y / model
                    if sum(contam_cumu != 0) != 0:
                        mask[begin:end] += np.nanmean(contam_cumu[contam_cumu != 0]) * (
                            contam_cumu != 0
                        )
                        order_s2d.append((vec.y != 0) * (1 + order / len(contam) / 20))
                        wave_s2d.append(sub_grid)

                mask[0:imin] = 0
                mask[imax:] = 0
                load[correction] = mask
        else:
            load[correction] = np.zeros(len(grid))

    iofun.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))
