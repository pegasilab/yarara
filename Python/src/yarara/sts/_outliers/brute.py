from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, Literal, Optional, Union

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
from ...util import assert_never, flux_norm_std, print_box, sphinx, yarara_artefact_suppressed

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_correct_brute(
    self: spec_time_series,
    sub_dico: str = "matching_mad",
    reference: Union[
        int, Literal["snr"], Literal["median"], Literal["master"], Literal["zeros"]
    ] = "median",
    win_roll: int = 1000,
    min_length: int = 5,
    percent_removed: Optional[float] = 10,
    k_sigma: int = 2,
    extended: int = 10,
    borders_pxl: bool = False,
) -> None:
    """
    Brutal suppression of flux value with variance to high (final solution)

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    reference : 'median', 'snr' or 'master' to select the reference normalised spectrum used in the difference
    win_roll : window size of the rolling algorithm
    min_length : minimum cluster length to be flagged
    k_sigma : k_sigma of the rolling mad clipping
    extended : extension of the cluster size
    low : lowest cmap value
    high : highest cmap value
    cmap : cmap of the 2D plot
    """

    print_box("\n---- RECIPE : CORRECTION BRUTE ----\n")

    directory = self.directory

    cmap = self.cmap
    planet = self.planet
    low_cmap = self.low_cmap
    high_cmap = self.high_cmap

    self.import_info_reduction()
    self.import_material()
    load = self.material
    grid = np.array(load["wave"])
    self.import_table()
    jdb = np.array(self.table["jdb"])
    snr = np.array(self.table["snr"])

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    print("\n---- DICO %s used ----\n" % (sub_dico))

    all_flux, conti = self.import_sts_flux(load=["flux" + kw, sub_dico])
    all_flux = all_flux / conti

    step = self.info_reduction[sub_dico]["step"]

    if isinstance(reference, int):
        logging.info("Reference spectrum : spectrum %.0f" % (reference))
        ref = all_flux[reference]
    elif reference == "snr":
        ref = all_flux[snr.argmax()]
    elif reference == "median":
        logging.info("Reference spectrum : median")
        ref = np.median(all_flux, axis=0)
    elif reference == "master":
        logging.info("Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif reference == "zeros":
        ref = 0 * np.median(all_flux, axis=0)
    else:
        assert_never(reference)

    all_flux = all_flux - ref
    metric = np.std(all_flux, axis=0)
    smoothed_med = np.ravel(
        pd.DataFrame(metric).rolling(win_roll, center=True, min_periods=1).quantile(0.5)
    )
    smoothed_mad = np.ravel(
        pd.DataFrame(abs(metric - smoothed_med))
        .rolling(win_roll, center=True, min_periods=1)
        .quantile(0.5)
    )
    mask = (metric - smoothed_med) > smoothed_mad * 1.48 * k_sigma

    clus = clustering(mask, 0.5, 1)[0]
    clus = np.array([np.product(j) for j in clus])
    cluster = clustering(mask, 0.5, 1)[-1]
    cluster = np.hstack([cluster, clus[:, np.newaxis]])
    cluster = cluster[cluster[:, 3] == 1]
    cluster = cluster[cluster[:, 2] >= min_length]

    cluster2 = cluster.copy()
    sum_mask = []
    all_flat = []
    for j in tqdm(range(200)):
        cluster2[:, 0] -= extended
        cluster2[:, 1] += extended
        flat_vec = flat_clustering(len(grid), cluster2[:, 0:2])
        flat_vec = flat_vec >= 1
        all_flat.append(flat_vec)
        sum_mask.append(np.sum(flat_vec))
    sum_mask = 100 * np.array(sum_mask) / len(grid)
    all_flat = np.array(all_flat)

    loc = find_nearest(sum_mask, np.arange(5, 26, 5))[0]

    plt.figure(figsize=(16, 16))

    plt.subplot(3, 1, 1)
    plt.plot(grid, metric - smoothed_med, color="k")
    plt.plot(grid, smoothed_mad * 1.48 * k_sigma, color="r")
    plt.ylim(0, 0.01)
    ax = plt.gca()

    plt.subplot(3, 1, 2, sharex=ax)
    for i, j, k in zip(["5%", "10%", "15%", "20%", "25%"], loc, [1, 1.05, 1.1, 1.15, 1.2]):
        plt.plot(grid, all_flat[j] * k, label=i)
    plt.legend()

    plt.subplot(3, 2, 5)
    b = tableXY(np.arange(len(sum_mask)) * 5, sum_mask)
    b.null()
    b.plot()
    plt.xlabel("Extension of rejection zones", fontsize=14)
    plt.ylabel("Percent of the spectrum rejected [%]", fontsize=14)

    for j in loc:
        plt.axhline(y=b.y[j], color="k", ls=":")

    ax = plt.gca()
    plt.subplot(3, 2, 6, sharex=ax)
    b.diff(replace=False)
    b.deri.plot()
    for j in loc:
        plt.axhline(y=b.deri.y[j], color="k", ls=":")

    if percent_removed is None:
        percent_removed = int(sphinx("Select the percentage of spectrum removed"))

    loc_select = find_nearest(sum_mask, percent_removed)[0]

    final_mask = np.ravel(all_flat[loc_select]).astype("bool")

    if borders_pxl:
        borders_pxl_mask = np.array(load["borders_pxl"]).astype("bool")
    else:
        borders_pxl_mask = np.zeros(len(final_mask)).astype("bool")

    final_mask = final_mask | borders_pxl_mask

    load["mask_brute"] = final_mask
    iofun.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    all_flux2 = all_flux.copy()
    all_flux2[:, final_mask] = 0

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(all_flux, aspect="auto", vmin=low_cmap, vmax=high_cmap, cmap=cmap)
    ax = plt.gca()
    plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
    plt.imshow(all_flux2, aspect="auto", vmin=low_cmap, vmax=high_cmap, cmap=cmap)
    ax = plt.gca()

    new_conti = conti * (all_flux + ref) / (all_flux2 + ref + epsilon)
    new_continuum = new_conti.copy()
    new_continuum[all_flux == 0] = conti[all_flux == 0]
    new_continuum[new_continuum == 0] = conti[new_continuum == 0]
    new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]

    print("\nComputation of the new continua, wait ... \n")

    self.info_reduction["matching_brute"] = {
        "reference_spectrum": reference,
        "sub_dico_used": sub_dico,
        "k_sigma": k_sigma,
        "rolling_window": win_roll,
        "minimum_length_cluster": min_length,
        "percentage_removed": percent_removed,
        "step": step + 1,
        "valid": True,
    }
    self.update_info_reduction()

    fname = self.dir_root + "WORKSPACE/CONTINUUM/Continuum_%s.npy" % ("matching_brute")
    np.save(fname, new_continuum.astype("float32"))

    self.dico_actif = "matching_brute"
