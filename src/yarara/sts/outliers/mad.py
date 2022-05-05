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

from ... import io
from ...analysis import tableXY
from ...paths import root
from ...plots import my_colormesh
from ...stats import clustering, find_nearest, flat_clustering, smooth
from ...util import flux_norm_std, print_box, sphinx, yarara_artefact_suppressed

if TYPE_CHECKING:
    from .. import spec_time_series


# =========================================================================================
# SUPRESS VARIATION RELATIF TO MEDIAN-MAD SPECTRUM (OUTLIERS CORRECTION + ECCENTRIC PLANET)
# =========================================================================================


def yarara_correct_mad(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    continuum: str = "linear",
    k_sigma: int = 2,
    k_mad: int = 2,
    n_iter: int = 1,
    ext: str = "0",
) -> np.ndarray:

    """
    Supress flux value outside k-sigma mad clipping

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)

    """

    print_box("\n---- RECIPE : CORRECTION MAD ----\n")

    directory = self.directory
    planet = self.planet

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    print("\n---- DICO %s used ----\n" % (sub_dico))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    all_flux = []
    all_flux_std = []
    all_snr = []
    jdb = []

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        if not i:
            grid = file["wave"]

        f = file["flux" + kw]
        f_std = file["flux_err"]
        c = file[sub_dico]["continuum_" + continuum]
        c_std = file["continuum_err"]
        f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)
        all_flux.append(f_norm)
        all_flux_std.append(f_norm_std)
        all_snr.append(file["parameters"]["SNR_5500"])
        jdb.append(file["parameters"]["jdb"])

    step = file[sub_dico]["parameters"]["step"]

    all_flux = np.array(all_flux)
    all_flux_std = np.array(all_flux_std)
    all_snr = np.array(all_snr)
    snr_max = all_snr.argmax()
    jdb = np.array(jdb)

    # plt.subplot(3,1,1)
    # for j in range(len(all_flux)):
    #    plt.plot(grid,all_flux[j])
    # ax = plt.gca()
    # plt.plot(grid,np.median(all_flux,axis=0),color='k',zorder=1000)

    all_flux[np.isnan(all_flux)] = 0

    all_flux2 = all_flux.copy()
    # all_flux2[np.isnan(all_flux2)] = 0

    med = np.median(all_flux2.copy(), axis=0).copy()
    mean = np.mean(all_flux2.copy(), axis=0).copy()
    sup = np.percentile(all_flux2.copy(), 84, axis=0).copy()
    inf = np.percentile(all_flux2.copy(), 16, axis=0).copy()
    ref = all_flux2[snr_max].copy()

    ok = "y"

    save = []
    count = 0
    while ok == "y":

        mad = 1.48 * np.median(
            abs(all_flux2 - np.median(all_flux2, axis=0)), axis=0
        )  # mad transformed in sigma
        mad[mad == 0] = 100
        counter_removed = []
        cum_curve = []
        for j in tqdm(range(len(all_flux2))):
            sigma = tableXY(grid, (abs(all_flux2[j] - med) - all_flux_std[j] * k_sigma) / mad)
            sigma.smooth(box_pts=6, shape="rectangular")
            # sigma.rolling(window=100,quantile=0.50)
            # sig = (sigma.y>(sigma.roll_Q1+3*sigma.roll_IQ))
            # sig = sigma.roll_Q1+3*sigma.roll_IQ

            # sigma.y *= -1
            # sigma.find_max(vicinity=5)
            # loc_min = sigma.index_max.copy()
            # sigma.y *= -1

            mask = sigma.y > k_mad

            # sigma.find_max(vicinity=5)
            # loc_max = np.array([sigma.y_max, sigma.index_max]).T
            # loc_max = loc_max[loc_max[:,0]>k_mad] # only keep sigma higher than k_sigma
            # loc_max = loc_max[:,-1]

            # diff = loc_max - loc_min[:,np.newaxis]
            # diff1 = diff.copy()
            # #diff2 = diff.copy()
            # diff1[diff1<0] = 1000 #arbitrary large value
            # #diff2[diff2>0] = -1000 #arbitrary small value
            # left = np.argmin(diff1,axis=1)
            # left = np.unique(left)
            # mask = np.zeros(len(grid)).astype('bool')
            # for k in range(len(left)-1):
            #     mask[int(sigma.index_max[left[k]]):int(sigma.index_max[left[k]+1])+1] = True

            # all_flux2[j][sigma.y>3] = med[sigma.y>3]

            all_flux2[j][mask] = med[mask]
            counter_removed.append(100 * np.sum(mask * (ref < 0.9)) / np.sum(ref < 0.9))
            cum_curve.append(100 * np.cumsum(mask * (ref < 0.9)) / np.sum(ref < 0.9))

        counter_mad_removed = np.array(counter_removed)
        cum_curves = np.array(cum_curve)
        cum_curves[cum_curves[:, -1] == 0, -1] = 1

        med2 = np.median(all_flux2, axis=0)
        mean2 = np.mean(all_flux2, axis=0)
        sup2 = np.percentile(all_flux2, 84, axis=0)
        inf2 = np.percentile(all_flux2, 16, axis=0)
        ref2 = all_flux2[snr_max].copy()

        save.append((mean - mean2).copy())

        if n_iter is None:
            plt.subplot(3, 1, 1)
            plt.plot(grid, med, color="k")
            plt.plot(grid, ref, color="k", alpha=0.4)
            plt.fill_between(grid, sup, y2=inf, alpha=0.5, color="b")
            ax = plt.gca()
            plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
            plt.plot(grid, med2, color="k")
            plt.plot(grid, ref2, color="k", alpha=0.4)
            plt.fill_between(grid, sup2, y2=inf2, alpha=0.5, color="g")
            plt.subplot(3, 1, 3, sharex=ax)
            plt.plot(grid, ref - ref2, color="k", alpha=0.4)
            for k in range(len(save)):
                plt.plot(grid, save[k])
            plt.axhline(y=0, color="r")

            plt.show(block=False)
            ok = sphinx(
                " Do you want to iterate one more time (y), quit (n) or save (s) ? (y/n/s)",
                rep=["y", "n", "s"],
            )
            plt.close()
        else:
            if n_iter == 1:
                ok = "s"
            else:
                n_iter -= 1
                ok = "y"

        if ok != "y":
            break
        else:
            count += 1
    if ok == "s":
        plt.figure(figsize=(23, 16))
        plt.subplot(2, 3, 1)
        plt.axhline(
            y=0.15,
            color="k",
            ls=":",
            label="rejection criterion  (%.0f)" % (sum(counter_mad_removed > 0.15)),
        )
        plt.legend()
        plt.scatter(jdb, counter_mad_removed, c=jdb, cmap="jet")
        plt.xlabel("Time", fontsize=13)
        plt.ylabel("Percent of the spectrum removed [%]", fontsize=13)
        ax = plt.colorbar()
        ax.ax.set_ylabel("Time")
        plt.subplot(2, 3, 4)
        plt.axhline(
            y=0.15,
            color="k",
            ls=":",
            label="rejection criterion (%.0f)" % (sum(counter_mad_removed > 0.15)),
        )
        plt.scatter(all_snr, counter_mad_removed, c=jdb, cmap="jet")
        plt.xlabel("SNR", fontsize=13)
        plt.ylabel("Percent of the spectrum removed [%]", fontsize=13)
        ax = plt.colorbar()
        ax.ax.set_ylabel("Time")

        jet = plt.get_cmap("jet")
        vmin = np.min(jdb)
        vmax = np.max(jdb)

        cNorm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        plt.subplot(2, 3, 2)
        for j in range(len(jdb)):
            colorVal = scalarMap.to_rgba(jdb[j])
            plt.plot(grid[::500], cum_curves[j][::500], color=colorVal, alpha=0.5)
        plt.xlabel("Wavelength", fontsize=13)
        plt.ylabel("Cumulative of spectrum removed [%]", fontsize=13)

        plt.subplot(2, 3, 5)
        for j in range(len(jdb)):
            colorVal = scalarMap.to_rgba(jdb[j])
            plt.plot(
                grid[::500],
                cum_curves[j][::500] / cum_curves[j][-1] * 100,
                color=colorVal,
                alpha=0.3,
            )
        plt.xlabel("Wavelength", fontsize=13)
        plt.ylabel("Normalised cumulative spectrum removed [%]", fontsize=13)

        plt.subplot(2, 3, 3)
        for j in range(len(jdb)):
            plt.plot(grid[::500], cum_curves[j][::500], color="k", alpha=0.3)
        plt.xlabel("Wavelength", fontsize=13)
        plt.ylabel("Cumulative of spectrum removed [%]", fontsize=13)

        plt.subplot(2, 3, 6)
        for j in range(len(jdb)):
            colorVal = scalarMap.to_rgba(jdb[j])
            plt.plot(
                grid[::500],
                cum_curves[j][::500] / cum_curves[j][-1] * 100,
                color="k",
                alpha=0.3,
            )
        plt.xlabel("Wavelength", fontsize=13)
        plt.ylabel("Normalised cumulative spectrum removed [%]", fontsize=13)
        plt.subplots_adjust(left=0.07, right=0.97)
        plt.savefig(self.dir_root + "IMAGES/mad_statistics_iter_%s.png" % (ext))

        correction_mad = all_flux - all_flux2
        to_be_saved = {"wave": grid, "correction_map": correction_mad}
        io.pickle_dump(
            to_be_saved,
            open(self.dir_root + "CORRECTION_MAP/map_matching_mad.p", "wb"),
        )

        print("\nComputation of the new continua, wait ... \n")
        time.sleep(0.5)
        count_file = -1

        for j in tqdm(files):
            count_file += 1
            file = pd.read_pickle(j)
            all_flux2[count_file][file["flux" + kw] == 0] = 1
            all_flux2[count_file][all_flux2[count_file] == 0] = 1
            new_flux = file["flux" + kw] / all_flux2[count_file]
            new_flux[(new_flux == 0) | (new_flux != new_flux)] = 1
            mask = yarara_artefact_suppressed(
                file[sub_dico]["continuum_" + continuum],
                new_flux,
                larger_than=50,
                lower_than=-50,
            )
            new_flux[mask] = file[sub_dico]["continuum_" + continuum][mask]
            output = {"continuum_" + continuum: new_flux}
            file["matching_mad"] = output
            file["matching_mad"]["parameters"] = {
                "iteration": count + 1,
                "sub_dico_used": sub_dico,
                "k_sigma": k_sigma,
                "k_mad": k_mad,
                "step": step + 1,
            }
            io.save_pickle(j, file)

        self.dico_actif = "matching_mad"
    return counter_mad_removed
