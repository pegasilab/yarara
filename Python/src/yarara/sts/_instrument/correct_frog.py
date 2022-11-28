from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import matplotlib.pylab as plt
import numpy as np
from colorama import Fore
from tqdm import tqdm

from ...analysis import table, tableXY
from ...paths import root
from ...plots import my_colormesh, plot_color_box
from ...stats import clustering, find_nearest, flat_clustering, merge_borders
from ...util import assert_never, doppler_r, flux_norm_std, print_box, sphinx

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_correct_frog(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    correction: Union[
        Literal["stitching"],
        Literal["ghost_a"],
        Literal["ghost_b"],
        Literal["thar"],
        Literal["contam"],
    ] = "stitching",
    wave_min: float = 3800.0,
    wave_max: float = 3975.0,
    wave_min_train: float = 3700.0,
    wave_max_train: float = 6000.0,
    complete_analysis: bool = False,
    reference: Union[
        int, Literal["snr"], Literal["median"], Literal["master"], Literal["zeros"]
    ] = "median",  # union literal
    equal_weight: bool = True,
    nb_pca_comp: int = 10,
    pca_comp_kept_: Optional[int] = None,
    rcorr_min: float = 0.0,
    threshold_contam: float = 0.5,
    algo_pca: Union[Literal["pca"], Literal["wpca"], Literal["empca"]] = "empca",
) -> None:

    """
    Correction of the stitching/ghost on the spectrum by PCA fitting

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    reference : 'median', 'snr' or 'master' to select the reference normalised spectrum used in the difference
    extended : extension of the cluster size
    """

    print_box(f"\n---- RECIPE : CORRECTION {correction.upper()} WITH FROG ----\n")

    directory = self.directory
    self.import_table()
    self.import_material()
    self.import_info_reduction()
    load = self.material
    grid = np.array(load["wave"])

    step = self.info_reduction[sub_dico]["step"]

    jdb = np.array(self.table["jdb"])
    snr = np.array(self.table["snr"])
    berv = np.array(self.table["berv"])

    file_ref = self.import_spectrum()
    hole_left = file_ref["parameters"]["hole_left"]
    hole_right = file_ref["parameters"]["hole_right"]

    cmap = self.cmap
    planet = self.planet
    low_cmap = self.low_cmap * 100
    high_cmap = self.high_cmap * 100

    kw = "_planet" * planet
    if kw != "":
        logging.info("PLANET ACTIVATED")

    epsilon = 1e-12

    if sub_dico is None:
        sub_dico = self.dico_actif
    logging.info(f"DICO {sub_dico} used")

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    flux, flux_err, conti, conti_err = self.import_sts_flux(
        load=["flux" + kw, "flux_err", sub_dico, "continuum_err"]
    )
    all_flux, all_flux_std = flux_norm_std(flux, flux_err, conti + epsilon, conti_err)

    if isinstance(reference, int):
        logging.info(f"Reference spectrum : spectrum {reference}")
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

    berv_max = self.table["berv" + kw].max()
    berv_min = self.table["berv" + kw].min()

    diff = all_flux - ref

    diff_backup = diff.copy()

    if np.sum(abs(berv)) != 0:
        for j in tqdm(np.arange(len(all_flux))):
            test = tableXY(grid, diff[j], all_flux_std[j])
            test.x = doppler_r(test.x, berv[j] * 1000)[1]
            test.interpolate(new_grid=grid, method="cubic", replace=True, interpolate_x=False)
            diff[j] = test.y
            all_flux_std[j] = test.yerr

    # extract frog table
    # frog_table = pd.read_pickle(frog_file)
    berv_file = 0  # self.yarara_get_berv_value(frog_table['jdb'])

    mask = np.array(load[correction])

    loc_ghost = mask != 0

    # mask[mask<treshold_contam] = 0
    val, borders = clustering(loc_ghost, 0.5, 1)
    val = np.array([np.product(v) for v in val])
    borders = borders[val == 1]

    min_t = grid[borders[:, 0]] * ((1 + 1.55e-8) * (1 + (berv_min - 0 * berv_file) / 299792.458))
    max_t = grid[borders[:, 1]] * ((1 + 1.55e-8) * (1 + (berv_max - 0 * berv_file) / 299792.458))

    if (correction == "ghost_a") | (correction == "ghost_b"):
        for j in range(3):
            if np.sum(mask > threshold_contam) < 200:
                threshold_contam *= 0.75
                logging.warning(
                    f"Not enough wavelength in the mask, threshold contamination reduced down to {threshold_contam:.2f}"
                )

    mask_ghost = np.sum(
        (grid > min_t[:, np.newaxis]) & (grid < max_t[:, np.newaxis]), axis=0
    ).astype("bool")
    mask_ghost_extraction = (
        mask_ghost
        & (mask > threshold_contam)
        & (ref < 1)
        & (np.array(1 - load["activity_proxies"]).astype("bool"))
        & (grid < wave_max_train)
        & (grid > wave_min_train)
    )  # extract everywhere

    if correction == "stitching":
        self.stitching = mask_ghost
        self.stitching_extracted = mask_ghost_extraction
    elif correction == "ghost_a":
        self.ghost_a = mask_ghost
        self.ghost_a_extracted = mask_ghost_extraction
    elif correction == "ghost_b":
        self.ghost_b = mask_ghost
        self.ghost_b_extracted = mask_ghost_extraction
    elif correction == "thar":
        self.thar = mask_ghost
        self.thar_extracted = mask_ghost_extraction
    elif correction == "contam":
        self.contam = mask_ghost
        self.contam_extracted = mask_ghost_extraction
    else:
        assert_never(correction)

    # compute pca

    if correction == "stitching":
        logging.info("Computation of PCA vectors for stitching correction...")
        diff_ref = diff[:, mask_ghost]
        subflux = diff[:, (mask_ghost) & (np.array(load["ghost_a"]) == 0)]
        subflux_std = all_flux_std[:, (mask_ghost) & (np.array(load["ghost_a"]) == 0)]
        lab = "Stitching"
        name = "stitching"
    elif correction == "ghost_a":
        logging.info("Computation of PCA vectors for ghost correction...")
        diff_ref = diff[:, mask_ghost]
        subflux = diff[:, (np.array(load["stitching"]) == 0) & (mask_ghost_extraction)]
        subflux_std = all_flux_std[:, (np.array(load["stitching"]) == 0) & (mask_ghost_extraction)]
        lab = "Ghost_a"
        name = "ghost_a"
    elif correction == "ghost_b":
        logging.info("Computation of PCA vectors for ghost correction...")
        diff_ref = diff[:, mask_ghost]
        subflux = diff[:, (load["thar"] == 0) & (mask_ghost_extraction)]
        subflux_std = all_flux_std[:, (load["thar"] == 0) & (mask_ghost_extraction)]
        lab = "Ghost_b"
        name = "ghost_b"
    elif correction == "thar":
        logging.info("Computation of PCA vectors for thar correction...")
        diff_ref = diff.copy()
        subflux = diff[:, mask_ghost_extraction]
        subflux_std = all_flux_std[:, mask_ghost_extraction]
        lab = "Thar"
        name = "thar"
    elif correction == "contam":
        logging.info("Computation of PCA vectors for contam correction...")
        diff_ref = diff[:, mask_ghost]
        subflux = diff[:, mask_ghost_extraction]
        subflux_std = all_flux_std[:, mask_ghost_extraction]
        lab = "Contam"
        name = "contam"

    subflux_std = subflux_std[:, np.std(subflux, axis=0) != 0]
    subflux = subflux[:, np.std(subflux, axis=0) != 0]

    if not len(subflux[0]):
        subflux = diff[:, 0:10]
        subflux_std = all_flux_std[:, 0:10]

    plt.figure(2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(subflux, aspect="auto", vmin=-0.005, vmax=0.005)
    plt.title(lab + " lines")
    plt.xlabel("Pixels extracted", fontsize=14)
    plt.ylabel("Time", fontsize=14)
    ax = plt.gca()
    plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
    plt.imshow(
        subflux / (epsilon + np.std(subflux, axis=0)),
        aspect="auto",
        vmin=-0.005,
        vmax=0.005,
    )
    plt.title(lab + " lines equalized")
    plt.xlabel("Pixels extracted", fontsize=14)
    plt.ylabel("Time", fontsize=14)

    c = int(equal_weight)

    X_train = (subflux / ((1 - c) + epsilon + c * np.std(subflux, axis=0))).T
    X_train_std = (subflux_std / ((1 - c) + epsilon + c * np.std(subflux, axis=0))).T

    # io.pickle_dump({'jdb':np.array(self.table.jdb),'ratio_flux':X_train,'ratio_flux_std':X_train_std},open(root+'/Python/datasets/telluri_cenB.p','wb'))

    test2 = table(X_train)
    test2.WPCA(algo_pca, weight=1 / X_train_std**2, comp_max=nb_pca_comp)

    phase_mod = np.arange(365)[
        np.argmin(
            np.array([np.max((jdb - k) % 365.25) - np.min((jdb - k) % 365.25) for k in range(365)])
        )
    ]

    plt.figure(4, figsize=(10, 14))
    plt.subplot(3, 1, 1)
    plt.xlabel("# PCA components", fontsize=13)
    plt.ylabel("Variance explained", fontsize=13)
    plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.var_ratio)
    plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.var_ratio)
    plt.subplot(3, 1, 2)
    plt.xlabel("# PCA components", fontsize=13)
    plt.ylabel("Z score", fontsize=13)
    plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.zscore_components)
    plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.zscore_components)
    z_max = test2.zscore_components[-5:].max()
    z_min = test2.zscore_components[-5:].min()
    vec_relevant = np.arange(len(test2.zscore_components)) * (
        (test2.zscore_components > z_max) | (test2.zscore_components < z_min)
    )
    plt.axhspan(ymin=z_min, ymax=z_max, alpha=0.2, color="k")
    pca_comp_kept2 = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])
    plt.subplot(3, 1, 3)
    plt.xlabel("# PCA components", fontsize=13)
    plt.ylabel(r"$\Phi(0)$", fontsize=13)
    plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.phi_components)
    plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.phi_components)
    plt.axhline(y=0.5, color="k")  # type: ignore
    phi_max = test2.phi_components[-5:].max()
    phi_min = test2.phi_components[-5:].min()
    plt.axhspan(ymin=phi_min, ymax=phi_max, alpha=0.2, color="k")
    vec_relevant = np.arange(len(test2.phi_components)) * (
        (test2.phi_components > phi_max) | (test2.phi_components < phi_min)
    )
    if pca_comp_kept_ is None:
        pca_comp_kept = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])
        pca_comp_kept = np.max([pca_comp_kept, pca_comp_kept2])
    else:
        pca_comp_kept = pca_comp_kept_

    plt.savefig(self.dir_root + "IMAGES/" + name + "_PCA_variances.pdf")

    plt.figure(figsize=(15, 10))
    for j in range(pca_comp_kept):
        if j == 0:
            plt.subplot(pca_comp_kept, 2, 2 * j + 1)
            ax = plt.gca()
        else:
            plt.subplot(pca_comp_kept, 2, 2 * j + 1, sharex=ax)
        plt.scatter(jdb, test2.vec[:, j])
        plt.subplot(pca_comp_kept, 2, 2 * j + 2)
        plt.scatter((jdb - phase_mod) % 365.25, test2.vec[:, j])
    plt.subplots_adjust(top=0.95, bottom=0.07, left=0.07, right=0.95, hspace=0)
    plt.savefig(self.dir_root + "IMAGES/" + name + "_PCA_vectors.pdf")

    if correction == "stitching":
        self.vec_pca_stitching = test2.vec[:, 0:pca_comp_kept]
    elif correction == "ghost_a":
        self.vec_pca_ghost_a = test2.vec[:, 0:pca_comp_kept]
    elif correction == "ghost_b":
        self.vec_pca_ghost_b = test2.vec[:, 0:pca_comp_kept]
    elif correction == "thar":
        self.vec_pca_thar = test2.vec[:, 0:pca_comp_kept]
    elif correction == "contam":
        self.vec_pca_contam = test2.vec[:, 0:pca_comp_kept]

    to_be_fit = diff / (np.std(diff, axis=0) + epsilon)

    rcorr = np.zeros(len(grid))
    for j in range(pca_comp_kept):
        proxy1 = test2.vec[:, j]
        rslope1 = np.median(
            (to_be_fit - np.mean(to_be_fit, axis=0)) / ((proxy1 - np.mean(proxy1))[:, np.newaxis]),
            axis=0,
        )
        rcorr1 = abs(rslope1 * np.std(proxy1) / (np.std(to_be_fit, axis=0) + epsilon))
        rcorr = np.nanmax([rcorr1, rcorr], axis=0)
    rcorr[np.isnan(rcorr)] = 0

    val, borders = clustering(mask_ghost, 0.5, 1)
    val = np.array([np.product(j) for j in val])
    borders = borders[val.astype("bool")]
    borders = merge_borders(borders)
    flat_mask = flat_clustering(len(grid), borders, extended=50).astype("bool")
    rcorr_free = rcorr[~flat_mask]
    rcorr_contaminated = rcorr[flat_mask]

    if correction == "thar":
        mask_ghost = np.ones(len(grid)).astype("bool")

    plt.figure(figsize=(8, 6))
    bins_contam, bins, dust = plt.hist(
        rcorr_contaminated,
        label="contaminated region",
        bins=list(np.linspace(0, 1, 100)),
        alpha=0.5,
        density=True,
    )
    bins_control, bins, dust = plt.hist(
        rcorr_free,
        bins=list(np.linspace(0, 1, 100)),
        label="free region",
        alpha=0.5,
        density=True,
    )
    plt.yscale("log")
    plt.legend()
    bins = bins[0:-1] + np.diff(bins) * 0.5  # type: ignore
    sum_a = np.sum(bins_contam[bins > 0.40])  # type: ignore
    sum_b = np.sum(bins_control[bins > 0.40])  # type: ignore
    crit = int(sum_a > (2 * sum_b))
    check = ["r", "g"][crit]  # three times more correlation than in the control group
    plt.xlabel(r"|$\mathcal{R}_{pearson}$|", fontsize=14, fontweight="bold", color=check)
    plt.title("Density", color=check)
    plot_color_box(color=check)

    plt.savefig(self.dir_root + "IMAGES/" + name + "_control_check.pdf")
    logging.info(f"{sum_a:.0f} versus {sum_b:.0f}")

    # TODO: what about failed steps?

    if not crit:
        logging.warning(
            f"Control check failed. Correction may be poorly performed for: {name}"
        )

    # TODO: reformat
    if True:
        logging.info(f"Control check sucessfully performed: {name}")

        diff_ref[np.isnan(diff_ref)] = 0

        idx_min = find_nearest(grid, wave_min)[0]
        idx_max = find_nearest(grid, wave_max)[0] + 1

        new_wave = grid[int(idx_min) : int(idx_max)]

        if complete_analysis:
            plt.figure(figsize=(18, 12))
            plt.subplot(pca_comp_kept // 2 + 1, 2, 1)
            my_colormesh(
                new_wave,
                np.arange(len(diff)),
                diff[:, int(idx_min) : int(idx_max)],
                vmin=low_cmap / 100,
                vmax=high_cmap / 100,
                cmap=cmap,
            )
            ax = plt.gca()
            for nb_vec in tqdm(range(1, pca_comp_kept)):
                correction2 = np.zeros((len(grid), len(jdb)))
                collection = table(diff_ref.T)
                base_vec = np.vstack([np.ones(len(diff)), test2.vec[:, 0:nb_vec].T])
                collection.fit_base(base_vec, num_sim=1)
                correction2[mask_ghost] = collection.coeff_fitted.dot(base_vec)
                correction2 = np.transpose(correction2)
                diff_ref2 = diff - correction2
                plt.subplot(pca_comp_kept // 2 + 1, 2, nb_vec + 1, sharex=ax, sharey=ax)
                plt.title(f"Vec PCA fitted = {nb_vec:0.f}")
                my_colormesh(
                    new_wave,
                    np.arange(len(diff)),
                    diff_ref2[:, int(idx_min) : int(idx_max)],
                    vmin=low_cmap / 100,
                    vmax=high_cmap / 100,
                    cmap=cmap,
                )
            plt.subplots_adjust(top=0.95, bottom=0.07, left=0.07, right=0.95, hspace=0.3)
            plt.subplot(pca_comp_kept // 2 + 1, 2, pca_comp_kept + 1, sharex=ax)
            plt.plot(new_wave, mask[int(idx_min) : int(idx_max)])
            plt.plot(
                new_wave,
                mask_ghost_extraction[int(idx_min) : int(idx_max)],
                color="k",
            )
            if correction == "stitching":
                plt.plot(new_wave, ref[int(idx_min) : int(idx_max)], color="gray")
        else:
            correction_map = np.zeros((len(grid), len(jdb)))
            collection = table(diff_ref.T)
            base_vec = np.vstack([np.ones(len(diff)), test2.vec[:, 0:pca_comp_kept].T])
            collection.fit_base(base_vec, num_sim=1)
            correction_map[mask_ghost] = collection.coeff_fitted.dot(base_vec)
            correction_map = np.transpose(correction_map)
            correction_map[:, rcorr < rcorr_min] = 0

            if np.sum(abs(berv)) != 0:
                for j in tqdm(np.arange(len(all_flux))):
                    test = tableXY(grid, correction_map[j], 0 * grid)
                    test.x = doppler_r(test.x, berv[j] * 1000)[0]
                    test.interpolate(
                        new_grid=grid,
                        method="cubic",
                        replace=True,
                        interpolate_x=False,
                    )
                    correction_map[j] = test.y

                index_min_backup = int(find_nearest(grid, doppler_r(grid[0], 30000)[0])[0])
                index_max_backup = int(find_nearest(grid, doppler_r(grid[-1], -30000)[0])[0])
                correction_map[:, 0 : index_min_backup * 2] = 0
                correction_map[:, index_max_backup * 2 :] = 0
                index_hole_right = int(
                    find_nearest(grid, hole_right + 1)[0]
                )  # correct 1 angstrom band due to stange artefact at the border of the gap
                index_hole_left = int(
                    find_nearest(grid, hole_left - 1)[0]
                )  # correct 1 angstrom band due to stange artefact at the border of the gap
                correction_map[:, index_hole_left : index_hole_right + 1] = 0

            diff_ref2 = diff_backup - correction_map

            new_conti = conti * (diff_backup + ref) / (diff_ref2 + ref + epsilon)
            new_continuum = new_conti.copy()
            new_continuum[all_flux == 0] = conti[all_flux == 0]
            new_continuum[new_continuum != new_continuum] = conti[
                new_continuum != new_continuum
            ]  # to suppress mystic nan appearing
            new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]
            new_continuum[new_continuum == 0] = conti[new_continuum == 0]
            new_continuum = self.uncorrect_hole(new_continuum, conti)

            # plot end

            if (name == "thar") | (name == "stitching"):
                max_var = grid[np.std(correction_map, axis=0).argsort()[::-1]]
                if name == "thar":
                    max_var = max_var[max_var < 4400][0]
                else:
                    max_var = max_var[max_var < 6700][0]
                wave_min = find_nearest(grid, max_var - 15)[1]  # type: ignore
                wave_max = find_nearest(grid, max_var + 15)[1]  # type: ignore

                idx_min = find_nearest(grid, wave_min)[0]
                idx_max = find_nearest(grid, wave_max)[0] + 1

            new_wave = grid[int(idx_min) : int(idx_max)]

            fig = plt.figure(figsize=(21, 9))

            plt.axes((0.05, 0.66, 0.90, 0.25))
            my_colormesh(
                new_wave,
                np.arange(len(diff)),
                100 * diff_backup[:, int(idx_min) : int(idx_max)],
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

            plt.axes((0.05, 0.375, 0.90, 0.25), sharex=ax, sharey=ax)
            my_colormesh(
                new_wave,
                np.arange(len(diff)),
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

            plt.axes((0.05, 0.09, 0.90, 0.25), sharex=ax, sharey=ax)
            my_colormesh(
                new_wave,
                np.arange(len(diff)),
                100 * diff_backup[:, int(idx_min) : int(idx_max)]
                - 100 * diff_ref2[:, int(idx_min) : int(idx_max)],
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

            plt.savefig(self.dir_root + "IMAGES/Correction_" + name + ".png")

            if name == "ghost_b":
                self.import_dico_tree()

                sub = np.array(
                    self.dico_tree.loc[self.dico_tree["dico"] == "matching_ghost_a", "dico_used"]
                )[0]
                flux, flux_err, conti, conti_err = self.import_sts_flux(
                    load=["flux" + kw, "flux_err", sub, "continuum_err"]
                )
                f_norm, f_norm_std = flux_norm_std(flux, flux_err, conti + epsilon, conti_err)
                diff_backup = f_norm - ref

                fig = plt.figure(figsize=(21, 9))

                plt.axes((0.05, 0.66, 0.90, 0.25))
                my_colormesh(
                    new_wave,
                    np.arange(len(diff)),
                    100 * diff_backup[:, int(idx_min) : int(idx_max)],
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

                plt.axes((0.05, 0.375, 0.90, 0.25), sharex=ax, sharey=ax)
                my_colormesh(
                    new_wave,
                    np.arange(len(diff)),
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

                plt.axes((0.05, 0.09, 0.90, 0.25), sharex=ax, sharey=ax)
                my_colormesh(
                    new_wave,
                    np.arange(len(diff)),
                    100 * diff_backup[:, int(idx_min) : int(idx_max)]
                    - 100 * diff_ref2[:, int(idx_min) : int(idx_max)],
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

                plt.savefig(self.dir_root + "IMAGES/Correction_ghost.png")

            to_be_saved = {"wave": grid, "correction_map": correction_map}
            # myf.pickle_dump(to_be_saved,open(self.dir_root+'CORRECTION_MAP/map_matching_'+name+'.p','wb'))
            np.save(
                self.dir_root + "CORRECTION_MAP/map_matching_" + name + ".npy",
                to_be_saved["correction_map"].astype("float32"),
            )

            self.info_reduction["matching_" + name] = {
                "reference_spectrum": reference,
                "sub_dico_used": sub_dico,
                "equal_weight": equal_weight,
                "pca_comp_kept": pca_comp_kept,
                "step": step + 1,
                "valid": True,
            }
            self.update_info_reduction()

            fname = self.dir_root + f"WORKSPACE/CONTINUUM/Continuum_{'matching_' + name}.npy"
            np.save(fname, new_continuum.astype("float32"))

            self.yarara_analyse_summary()

            self.dico_actif = "matching_" + name

            plt.show(block=False)
