from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING
from typing import Literal as Shape

import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from nptyping import Float, NDArray
from tqdm import tqdm

from .. import io
from ..analysis import tableXY
from ..paths import root
from ..plots import my_colormesh
from ..stats import clustering, find_nearest, flat_clustering, smooth
from ..util import flux_norm_std, print_box, sphinx, yarara_artefact_suppressed

if TYPE_CHECKING:
    from . import spec_time_series


def yarara_correct_smooth(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    continuum: str = "linear",
    reference: str = "median",
    wave_min: int = 4200,
    wave_max: int = 4300,
    window_ang: int = 5,
) -> None:

    print_box("\n---- RECIPE : CORRECTION SMOOTH ----\n")

    directory = self.directory

    zoom = self.zoom
    smooth_map = self.smooth_map
    low_cmap = self.low_cmap * 100
    high_cmap = self.high_cmap * 100
    cmap = self.cmap
    planet = self.planet

    self.import_material()
    self.import_table()
    load = self.material

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    print("\n---- DICO %s used ----\n" % (sub_dico))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    flux = []
    # flux_err = []
    conti = []
    snr = []
    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        if not i:
            wave = file["wave"]
            dgrid = file["parameters"]["dwave"]
            try:
                hl = file["parameters"]["hole_left"]
            except:
                hl = None
            try:
                hr = file["parameters"]["hole_right"]
            except:
                hr = None

        snr.append(file["parameters"]["SNR_5500"])

        f = file["flux" + kw]
        f_std = file["flux_err"]
        c = file[sub_dico]["continuum_" + continuum]
        c_std = file["continuum_err"]
        f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)
        flux.append(f_norm)
        # flux_err.append(f_norm_std)
        conti.append(c)

    step = file[sub_dico]["parameters"]["step"]

    snr = np.array(snr)
    wave = np.array(wave)
    all_flux = np.array(flux)
    # all_flux_std = np.array(flux_err)
    conti = np.array(conti)

    if reference == "snr":
        ref = all_flux[snr.argmax()]
    elif reference == "median":
        print("[INFO] Reference spectrum : median")
        ref = np.median(all_flux, axis=0)
    elif reference == "master":
        print("[INFO] Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif type(reference) == int:
        print("[INFO] Reference spectrum : spectrum %.0f" % (reference))
        ref = all_flux[reference]
    else:
        ref = 0 * np.median(all_flux, axis=0)

    diff_ref = all_flux.copy() - ref

    flux_ratio = (all_flux.copy() / (ref + epsilon)) - 1

    box_pts = int(window_ang / dgrid)

    for k in tqdm(range(len(all_flux))):
        spec_smooth = smooth(
            smooth(flux_ratio[k], box_pts=box_pts, shape=50),
            box_pts=box_pts,
            shape="savgol",
        )
        if hl is not None:
            i1 = int(find_nearest(wave, hl)[0])
            i2 = int(find_nearest(wave, hr)[0])
            spec_smooth[i1 - box_pts : i2 + box_pts] = 0

        flux_ratio[k] -= spec_smooth

    flux_ratio += 1
    flux_ratio *= ref

    diff_ref2 = flux_ratio - ref

    del flux_ratio

    correction_smooth = diff_ref - diff_ref2

    new_conti = conti * (diff_ref + ref) / (diff_ref2 + ref + epsilon)
    new_continuum = new_conti.copy()
    new_continuum[flux == 0] = conti[flux == 0].copy()
    new_continuum[new_continuum != new_continuum] = conti[
        new_continuum != new_continuum
    ].copy()  # to supress mystic nan appearing
    new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)].copy()
    new_continuum[new_continuum == 0] = conti[new_continuum == 0].copy()
    new_continuum = self.uncorrect_hole(new_continuum, conti)

    idx_min = 0
    idx_max = len(wave)

    if wave_min is not None:
        idx_min = find_nearest(wave, wave_min)[0]
    if wave_max is not None:
        idx_max = find_nearest(wave, wave_max)[0] + 1

    if (idx_min == 0) & (idx_max == 1):
        idx_max = find_nearest(wave, np.min(wave) + 200)[0] + 1

    new_wave = wave[int(idx_min) : int(idx_max)]

    fig = plt.figure(figsize=(21, 9))

    plt.axes([0.05, 0.66, 0.90, 0.25])
    my_colormesh(
        new_wave,
        np.arange(len(diff_ref)),
        100 * diff_ref[:, int(idx_min) : int(idx_max)],
        vmin=low_cmap,
        vmax=high_cmap,
        cmap=cmap,
    )
    plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
    plt.ylabel("Spectra  indexes (time)", fontsize=14)
    plt.ylim(0, None)
    ax = plt.gca()
    cbaxes = fig.add_axes([0.95, 0.66, 0.01, 0.25])
    ax1 = plt.colorbar(cax=cbaxes)
    ax1.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

    plt.axes([0.05, 0.375, 0.90, 0.25], sharex=ax, sharey=ax)
    my_colormesh(
        new_wave,
        np.arange(len(diff_ref)),
        100 * diff_ref2[:, int(idx_min) : int(idx_max)],
        vmin=low_cmap,
        vmax=high_cmap,
        cmap=cmap,
    )
    plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
    plt.ylabel("Spectra  indexes (time)", fontsize=14)
    plt.ylim(0, None)
    ax = plt.gca()
    cbaxes2 = fig.add_axes([0.95, 0.375, 0.01, 0.25])
    ax2 = plt.colorbar(cax=cbaxes2)
    ax2.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

    plt.axes([0.05, 0.09, 0.90, 0.25], sharex=ax, sharey=ax)
    my_colormesh(
        new_wave,
        np.arange(len(diff_ref)),
        100
        * (diff_ref[:, int(idx_min) : int(idx_max)] - diff_ref2[:, int(idx_min) : int(idx_max)]),
        vmin=low_cmap,
        vmax=high_cmap,
        cmap=cmap,
    )
    plt.tick_params(direction="in", top=True, right=True, labelbottom=True)
    plt.ylabel("Spectra  indexes (time)", fontsize=14)
    plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
    plt.ylim(0, None)
    ax = plt.gca()
    cbaxes3 = fig.add_axes([0.95, 0.09, 0.01, 0.25])
    ax3 = plt.colorbar(cax=cbaxes3)
    ax3.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

    if sub_dico == "matching_diff":
        plt.savefig(self.dir_root + "IMAGES/Correction_diff.png")
        spec = self.import_spectrum()
        name = "diff"
        recenter = spec[sub_dico]["parameters"]["recenter"]
        ref_name = spec[sub_dico]["parameters"]["reference_continuum"]
        savgol_window = spec[sub_dico]["parameters"]["savgol_window"]
        sub_dico = spec[sub_dico]["parameters"]["sub_dico_used"]
    else:
        plt.savefig(self.dir_root + "IMAGES/Correction_smooth.png")
        to_be_saved = {"wave": wave, "correction_map": correction_smooth}
        io.pickle_dump(
            to_be_saved,
            open(self.dir_root + "CORRECTION_MAP/map_matching_smooth.p", "wb"),
        )
        name = "smooth"
        recenter = False
        ref_name = str(reference)
        savgol_window = 0

    # diff_ref2 = flux_ratio - ref
    # correction_smooth = diff_ref - diff_ref2
    # new_conti = conti*(diff_ref+ref)/(diff_ref2+ref+epsilon)
    # new_continuum = new_conti.copy()
    # new_continuum = self.uncorrect_hole(new_continuum,conti)

    print("Computation of the new continua, wait ... \n")
    time.sleep(0.5)
    count_file = -1
    for j in tqdm(files):
        count_file += 1
        file = pd.read_pickle(j)
        conti = new_continuum[count_file]
        mask = yarara_artefact_suppressed(
            file[sub_dico]["continuum_" + continuum],
            conti,
            larger_than=50,
            lower_than=-50,
        )
        conti[mask] = file[sub_dico]["continuum_" + continuum][mask]
        output = {"continuum_" + continuum: conti}
        file["matching_" + name] = output
        file["matching_" + name]["parameters"] = {
            "reference_continuum": ref_name,
            "sub_dico_used": sub_dico,
            "savgol_window": savgol_window,
            "window_ang": window_ang,
            "step": step + 1,
            "recenter": recenter,
        }
        io.save_pickle(j, file)


# =============================================================================
# SUPRESS VARIATION RELATIF TO MEDIAN-MAD SPECTRUM (COSMIC PEAK WITH VALUE > 1)
# =============================================================================
def yarara_correct_cosmics(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    continuum: str = "linear",
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
        print("---- DICO %s used ----" % (sub_dico))

        all_flux, conti = self.import_sts_flux(load=["flux" + kw, sub_dico])
        all_flux_norm = all_flux / conti

        step = self.info_reduction[sub_dico]["step"]

        med = np.median(all_flux_norm, axis=0)
        mad = 1.48 * np.median(abs(all_flux_norm - med), axis=0)
        all_flux_corrected = all_flux_norm.copy()
        level = (med + k_sigma * mad) * np.ones(len(jdb))[:, np.newaxis]
        mask = (all_flux_norm > 1) & (all_flux_norm > level)

        print(
            "\n [INFO] Percentage of cosmics detected with k-sigma %.0f : %.2f%% \n"
            % (k_sigma, 100 * np.sum(mask) / len(mask.T) / len(mask))
        )

        med_map = med * np.ones(len(jdb))[:, np.newaxis]

        plt.figure(figsize=(10, 10))
        plt.scatter(
            all_snr,
            np.sum(mask, axis=1) * 100 / len(mask.T),
            edgecolor="k",
            c=jdb,
            cmap="brg",
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
        ]  # to supress mystic nan appearing

        self.info_reduction["matching_cosmics"] = {
            "sub_dico_used": sub_dico,
            "k_sigma": k_sigma,
            "step": step + 1,
            "valid": True,
        }
        self.update_info_reduction()

        fname = self.dir_root + "WORKSPACE/CONTINUUM/Continuum_%s.npy" % ("matching_cosmics")
        np.save(fname, new_continuum)

        self.dico_actif = "matching_cosmics"


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


def yarara_correct_brute(
    self: spec_time_series,
    sub_dico: str = "matching_mad",
    continuum: str = "linear",
    reference: str = "median",
    win_roll: int = 1000,
    min_length: int = 5,
    percent_removed: int = 10,
    k_sigma: int = 2,
    extended: int = 10,
    ghost2: bool = "HARPS03",
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

    self.import_material()
    load = self.material

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    print("\n---- DICO %s used ----\n" % (sub_dico))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    all_flux = []
    snr = []
    jdb = []
    conti = []

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        if not i:
            grid = file["wave"]
        all_flux.append(file["flux" + kw] / file[sub_dico]["continuum_" + continuum])
        conti.append(file[sub_dico]["continuum_" + continuum])
        jdb.append(file["parameters"]["jdb"])
        snr.append(file["parameters"]["SNR_5500"])

    step = file[sub_dico]["parameters"]["step"]
    all_flux = np.array(all_flux)
    conti = np.array(conti)

    if reference == "snr":
        ref = all_flux[snr.argmax()]
    elif reference == "median":
        print("[INFO] Reference spectrum : median")
        ref = np.median(all_flux, axis=0)
    elif reference == "master":
        print("[INFO] Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif type(reference) == int:
        print("[INFO] Reference spectrum : spectrum %.0f" % (reference))
        ref = all_flux[reference]
    else:
        ref = 0 * np.median(all_flux, axis=0)

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
        percent_removed = sphinx("Select the percentage of spectrum removed")

    percent_removed = int(percent_removed)

    loc_select = find_nearest(sum_mask, percent_removed)[0]

    final_mask = np.ravel(all_flat[loc_select]).astype("bool")

    if borders_pxl:
        borders_pxl_mask = np.array(load["borders_pxl"]).astype("bool")
    else:
        borders_pxl_mask = np.zeros(len(final_mask)).astype("bool")

    if ghost2:
        g = pd.read_pickle(root + "/Python/Material/Ghost2_" + ghost2 + ".p")
        ghost = tableXY(g["wave"], g["ghost2"], 0 * g["wave"])
        ghost.interpolate(new_grid=grid, replace=True, method="linear", interpolate_x=False)
        ghost_brute_mask = ghost.y.astype("bool")
    else:
        ghost_brute_mask = np.zeros(len(final_mask)).astype("bool")
    load["ghost2"] = ghost_brute_mask.astype("int")
    io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

    final_mask = final_mask | ghost_brute_mask | borders_pxl_mask

    load["mask_brute"] = final_mask
    io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))

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
    time.sleep(0.5)

    i = -1
    for j in tqdm(files):
        i += 1
        file = pd.read_pickle(j)
        output = {"continuum_" + continuum: new_continuum[i]}
        file["matching_brute"] = output
        file["matching_brute"]["parameters"] = {
            "reference_spectrum": reference,
            "sub_dico_used": sub_dico,
            "k_sigma": k_sigma,
            "rolling_window": win_roll,
            "minimum_length_cluster": min_length,
            "percentage_removed": percent_removed,
            "step": step + 1,
        }
        io.save_pickle(j, file)

    self.dico_actif = "matching_brute"
