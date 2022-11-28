from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING

import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ... import iofun
from ...analysis import tableXY
from ...paths import root
from ...plots import my_colormesh
from ...stats import clustering, find_nearest, flat_clustering, smooth
from ...util import flux_norm_std, print_box, sphinx, yarara_artefact_suppressed

if TYPE_CHECKING:
    from .. import spec_time_series


# =============================================================================
# SUPRESS VARIATION RELATIF TO MEDIAN-MAD SPECTRUM (COSMIC PEAK WITH VALUE > 1)
# =============================================================================
def yarara_correct_cosmics(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    k_sigma: int = 3,
    bypass_warning: bool = True,
) -> None:

    """
    Supress flux value outside k-sigma mad clipping

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)

    """

    print_box("\n---- RECIPE : CORRECTION COSMICS ----\n")

    directory = self.directory
    planet = self.planet
    self.import_dico_tree()
    self.import_info_reduction()
    self.import_material()
    self.import_table()
    epsilon = 1e-12

    reduction_accepted = True

    if not bypass_warning:
        if "matching_smooth" in list(self.dico_tree["dico"]):
            logging.warn(
                "Launch that recipes will remove the smooth correction of the previous loop iteration."
            )
            answer = sphinx("Do you want to purchase (y/n) ?", rep=["y", "n"])
            if answer == "n":
                reduction_accepted = False

    if reduction_accepted:
        grid = np.array(self.material["wave"])
        jdb = np.array(self.table["jdb"])
        all_snr = np.array(self.table["snr"])
        files = np.array(self.table["filename"])
        files = np.sort(files)
        kw = "_planet" * planet
        if kw != "":
            print("\n---- PLANET ACTIVATED ----")

        if sub_dico is None:
            sub_dico = self.dico_actif
        print(f"---- DICO {sub_dico} used ----")

        all_flux, conti = self.import_sts_flux(load=["flux" + kw, sub_dico])
        all_flux_norm = all_flux / conti

        step = self.info_reduction[sub_dico]["step"]

        med = np.median(all_flux_norm, axis=0)
        mad = 1.48 * np.median(abs(all_flux_norm - med), axis=0)
        all_flux_corrected = all_flux_norm.copy()
        level = (med + k_sigma * mad) * np.ones(len(jdb))[:, np.newaxis]
        mask = (all_flux_norm > 1) & (all_flux_norm > level)

        print(
            f"\n [INFO] Percentage of cosmics detected with k-sigma {k_sigma:.0f} : {100 * np.sum(mask) / len(mask.T) / len(mask):.2f}% \n"
        )

        med_map = med * np.ones(len(jdb))[:, np.newaxis]

        plt.figure(figsize=(10, 10))
        plt.scatter(
            all_snr,
            np.sum(mask, axis=1) * 100 / len(mask.T),
            edgecolor="k",
            c=jdb,  # type: ignore
            cmap="brg",  # type: ignore
        )
        ax = plt.colorbar()
        plt.yscale("log")
        plt.ylim(0.001, 100)
        plt.xlabel("SNR", fontsize=13)
        plt.ylabel("Percent of the spectrum flagged as cosmics [%]", fontsize=13)
        plt.grid()
        ax.ax.set_ylabel("Jdb", fontsize=13)

        plt.figure(figsize=(20, 5))
        all_flux_corrected[mask] = med_map[mask]
        for j in range(len(jdb)):
            plt.plot(grid, all_flux_corrected[j] - 1.5, color="b", alpha=0.3)
            plt.plot(grid, all_flux_norm[j], color="k", alpha=0.3)
        plt.ylim(-2, 2)
        plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
        plt.ylabel(r"Flux normalised", fontsize=14)
        plt.savefig(self.dir_root + "IMAGES/Correction_cosmics.png")

        correction_cosmics = all_flux_norm - all_flux_corrected
        to_be_saved = {"wave": grid, "correction_map": correction_cosmics}
        np.save(
            self.dir_root + "CORRECTION_MAP/map_matching_cosmics.npy",
            to_be_saved["correction_map"],
        )

        new_continuum = all_flux / (all_flux_corrected + epsilon)
        new_continuum[all_flux == 0] = conti[all_flux == 0]
        new_continuum[new_continuum != new_continuum] = conti[
            new_continuum != new_continuum
        ]  # to suppress mystic nan appearing

        self.info_reduction["matching_cosmics"] = {
            "sub_dico_used": sub_dico,
            "k_sigma": k_sigma,
            "step": step + 1,
            "valid": True,
        }
        self.update_info_reduction()

        fname = self.dir_root + f"WORKSPACE/CONTINUUM/Continuum_{'matching_cosmics'}.npy"
        np.save(fname, new_continuum)

        self.dico_actif = "matching_cosmics"
