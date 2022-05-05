from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from colorama import Fore
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from ... import io
from ...analysis import table as table_cls
from ...analysis import tableXY
from ...paths import paths, root
from ...plots import my_colormesh, plot_color_box
from ...stats import IQ as IQ_fun
from ...stats import clustering, find_nearest, mad, merge_borders, smooth, smooth2d
from ...util import assert_never, doppler_r, flux_norm_std, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_correct_telluric_gradient(
    self: spec_time_series,
    sub_dico_detection: str = "matching_fourier",
    sub_dico_correction: Optional[str] = "matching_oxygen",  # was None
    wave_min_train: float = 4200.0,
    wave_max_train: float = 5000.0,
    wave_min_correction: float = 4400,
    wave_max_correction: float = 6600,
    smooth_map: int = 1,
    reference: str = "master",
    inst_resolution: int = 110000,
    debug: bool = False,
    equal_weight: bool = True,
    nb_pca_comp: int = 20,
    nb_pca_comp_kept: None = None,
    nb_pca_max_kept: int = 5,
    calib_std: float = 1e-3,
) -> None:

    """
    Display the time-series spectra with proxies and its correlation

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    wave_min : Minimum x axis limit
    wave_max : Maximum x axis limit
    smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
    berv_shift : True/False to move in terrestrial rest-frame
    cmap : cmap of the 2D plot
    low_cmap : vmin cmap colorbar
    high_cmap : vmax cmap colorbar

    """

    print_box("\n---- RECIPE : CORRECTION TELLURIC PCA ----\n")

    directory = self.directory

    zoom = self.zoom
    smooth_map = self.smooth_map
    cmap = self.cmap
    planet = self.planet
    low_cmap = self.low_cmap * 100
    high_cmap = self.high_cmap * 100
    self.import_material()
    self.import_table()
    self.import_info_reduction()

    load = self.material
    wave = np.array(load["wave"])

    jdb = np.array(self.table["jdb"])
    snr = np.array(self.table["snr"])
    berv = np.array(self.table["berv"])
    rv_shift = np.array(self.table["rv_shift"])

    mean_berv = np.mean(berv)
    berv = berv - mean_berv - rv_shift

    file_ref = self.import_spectrum()
    hole_left = file_ref["parameters"]["hole_left"]
    hole_right = file_ref["parameters"]["hole_right"]

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico_correction is None:
        sub_dico_correction = self.dico_actif
    print("\n---- DICO %s used ----\n" % (sub_dico_correction))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    all_flux, all_flux_err, conti, conti2, conti_err = self.import_sts_flux(
        load=[
            "flux" + kw,
            "flux_err",
            sub_dico_correction,
            sub_dico_detection,
            "continuum_err",
        ]
    )
    flux_backup = all_flux.copy()
    flux = all_flux.copy() / conti2
    flux_to_correct, flux_err = flux_norm_std(all_flux, all_flux_err, conti + epsilon, conti_err)
    flux_err += calib_std

    step = self.info_reduction[sub_dico_correction]["step"]

    if len(snr) < nb_pca_comp:
        nb_pca_comp = len(snr) - 1
        print(
            "Nb component too high compared to number of observations, nc reduced to %.0f"
            % (len(snr) - 2)
        )

    def idx_wave(wavelength):
        return int(find_nearest(wave, wavelength)[0])

    if reference == "snr":
        ref = flux[snr.argmax()]
    elif reference == "median":
        logging.info("Reference spectrum : median")
        ref = np.median(flux, axis=0)
    elif reference == "master":
        logging.info("Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif type(reference) == int:
        logging.info("Reference spectrum : spectrum %.0f" % (reference))
        ref = flux[reference]
    else:
        ref = 0 * np.median(flux, axis=0)

    diff = smooth2d(flux, smooth_map)
    diff_ref = smooth2d(flux - ref, smooth_map)
    diff_ref_to_correct = smooth2d(flux_to_correct - ref, smooth_map)
    ratio_ref = smooth2d(flux_to_correct / (ref + epsilon), smooth_map)
    diff_backup = diff_ref.copy()
    ratio_backup = ratio_ref.copy()

    del flux_to_correct

    if np.sum(abs(berv)) != 0:
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, diff[j], 0 * wave)
            test.x = doppler_r(test.x, berv[j] * 1000)[1]
            test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
            diff[j] = test.y
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, diff_ref[j], 0 * wave)
            test.x = doppler_r(test.x, berv[j] * 1000)[1]
            test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
            diff_ref[j] = test.y
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, ratio_ref[j], flux_err[j] / (ref + epsilon))
            test.x = doppler_r(test.x, berv[j] * 1000)[1]
            test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
            ratio_ref[j] = test.y
            flux_err[j] = test.yerr

    med_wave_gradient_f = np.median(np.gradient(diff)[1], axis=0)
    med_time_gradient_f = np.median(np.gradient(diff)[0], axis=0)
    mad_time_gradient_f = np.median(np.gradient(diff)[0] - med_time_gradient_f, axis=0)

    med_wave_gradient_df = np.median(np.gradient(diff_ref)[1], axis=0)
    med_time_gradient_df = np.median(np.gradient(diff_ref)[0], axis=0)
    mad_time_gradient_df = np.median(np.gradient(diff_ref)[0] - med_time_gradient_df, axis=0)

    med_df = np.median(diff_ref, axis=0)
    med_f = np.median(diff, axis=0)

    par1 = np.log10(abs(med_time_gradient_f + med_wave_gradient_f) + 1e-6)
    par2 = np.log10(abs(med_time_gradient_df + med_wave_gradient_df) + 1e-6)

    par3 = np.log10(abs(med_df) + 1e-6)
    par4 = np.log10(1 - abs(med_f) + 1e-6)

    par5 = np.log10(abs(med_wave_gradient_f / (med_time_gradient_f + 1e-4) + 1e-2))
    par6 = np.log10(abs(med_wave_gradient_f / (mad_time_gradient_f + 1e-4) + 1e-2))

    par7 = np.log10(abs(med_wave_gradient_df / (med_time_gradient_df + 1e-4) + 1e-6))
    par8 = np.log10(abs(med_wave_gradient_df / (mad_time_gradient_df + 1e-4) + 1e-6))

    par9 = np.log10(abs(med_time_gradient_f / (mad_time_gradient_f + 1e-4) + 1e-2))
    par10 = np.log10(abs(med_time_gradient_df / (mad_time_gradient_df + 1e-4) + 1e-2))

    table = pd.DataFrame(
        {
            "wave": wave,
            "med_df": med_df,
            "med_f": med_f,
            "gt_f": med_time_gradient_f,
            "gl_f": med_wave_gradient_f,
            "gt_df": med_time_gradient_df,
            "gl_df": med_wave_gradient_df,
            "par1": par1,
            "par2": par2,
            "par3": par3,
            "par4": par4,
            "par5": par5,
            "par6": par6,
            "par7": par7,
            "par8": par8,
            "par9": par9,
            "par10": par10,
        }
    )

    table["classes"] = "full_spectrum"
    table.loc[
        (table["wave"] > wave_min_train) & (table["wave"] < wave_max_train),
        "classes",
    ] = "telluric_free"

    med1 = np.median(10 ** np.array(table.loc[table["classes"] == "telluric_free", "par2"]))
    mad1 = 1.48 * np.median(
        abs(med1 - 10 ** np.array(table.loc[table["classes"] == "telluric_free", "par2"]))
    )

    med2 = np.median(10 ** np.array(table.loc[table["classes"] == "telluric_free", "par3"]))
    mad2 = 1.48 * np.median(
        abs(med1 - 10 ** np.array(table.loc[table["classes"] == "telluric_free", "par3"]))
    )

    med3 = np.median(10 ** np.array(table.loc[table["classes"] == "telluric_free", "par8"]))
    mad3 = 1.48 * np.median(
        abs(med1 - 10 ** np.array(table.loc[table["classes"] == "telluric_free", "par8"]))
    )

    crit1 = 10**par2 - med1
    crit2 = 10**par3 - med2
    crit3 = 10**par8 - med3

    z1 = crit1 / mad1
    z2 = crit2 / mad2
    z3 = crit3 / mad3

    ztot = z1 + z2 + z3
    ztot -= np.median(ztot)

    dint = int(3 / np.mean(np.diff(wave)))
    criterion = tableXY(wave, ztot)
    criterion.rolling(window=dint)
    iq = np.percentile(ztot, 75) - np.percentile(ztot, 25)

    # telluric detection

    inside = (
        smooth(ztot, 5, shape="savgol") > 1.5 * iq
    )  # &(criterion.y>criterion.roll_median) # hyperparameter
    pos_peaks = (inside > np.roll(inside, 1)) & (inside > np.roll(inside, -1))
    inside = inside * (1 - pos_peaks)
    neg_peaks = (inside < np.roll(inside, 1)) & (inside < np.roll(inside, -1))
    inside = inside + neg_peaks

    # comparison with Molecfit

    model = pd.read_pickle(root + "/Python/Material/model_telluric.p")
    wave_model = doppler_r(model["wave"], mean_berv * 1000)[0]
    telluric_model = model["flux_norm"]
    model = tableXY(wave_model, telluric_model, 0 * wave_model)
    model.interpolate(new_grid=wave, interpolate_x=False)
    model.find_min(vicinity=5)
    mask_model = np.zeros(len(wave))
    mask_model[model.index_min.astype("int")] = 1 - model.y_min
    mask_model[mask_model < 1e-5] = 0  # hyperparameter

    inside_model = (1 - model.y) > 3e-4
    pos_peaks = (inside_model > np.roll(inside_model, 1)) & (
        inside_model > np.roll(inside_model, -1)
    )
    inside_model = inside_model * (1 - pos_peaks)

    completness = mask_model * inside
    completness = completness[completness != 0]
    completness_max = mask_model[mask_model != 0]
    completness2 = mask_model * (1 - inside)
    completness2 = completness2[completness2 != 0]

    # self.debug1 = completness, completness_max, completness2, inside

    plt.figure()
    val = plt.hist(
        np.log10(completness_max),
        bins=np.linspace(-5, 0, 50),
        cumulative=-1,
        histtype="step",
        lw=3,
        color="k",
        label="telluric model",
    )
    val2 = plt.hist(
        np.log10(completness),
        bins=np.linspace(-5, 0, 50),
        cumulative=-1,
        histtype="step",
        lw=3,
        color="r",
        label="telluric detected",
    )
    val3 = plt.hist(
        np.log10(completness2),
        bins=np.linspace(-5, 0, 50),
        cumulative=1,
        histtype="step",
        lw=3,
        color="g",
        label="telluric undetected",
    )
    plt.close()

    # comp_percent = 100*(1 - (val[0]-val2[0])/(val[0]+1e-12)) #update 10.06.21 to complicated metric
    comp_percent = val3[0] * 100 / np.max(val3[0])
    tel_depth_grid = val[1][0:-1] + 0.5 * (val[1][1] - val[1][0])

    plt.figure(12, figsize=(8.5, 7))
    plt.plot(tel_depth_grid, comp_percent, color="k")
    plt.axhline(y=100, color="b", ls="-.")
    plt.grid()
    if len(np.where(comp_percent == 100)[0]) > 0:
        plt.axvline(
            x=tel_depth_grid[np.where(comp_percent == 100)[0][0]],
            color="b",
            label="100%% Completeness : %.2f [%%]"
            % (100 * 10 ** (tel_depth_grid[np.where(comp_percent == 100)[0][0]])),
        )
        plt.axvline(
            x=tel_depth_grid[find_nearest(comp_percent, 90)[0]],
            color="b",
            ls=":",
            label="90%% Completeness : %.2f [%%]"
            % (100 * 10 ** (tel_depth_grid[find_nearest(comp_percent, 90)[0]])),
        )
    plt.ylabel("Completness [%]", fontsize=16)
    plt.xlabel(r"$\log_{10}$(Telluric depth)", fontsize=16)
    plt.title("Telluric detection completeness versus MolecFit model", fontsize=16)
    plt.ylim(-5, 105)
    plt.legend(prop={"size": 14})
    plt.savefig(self.dir_root + "IMAGES/telluric_detection.pdf")

    # extraction telluric

    telluric_location = inside.copy()
    telluric_location[wave < wave_min_correction] = 0  # reject shorter wavelength
    telluric_location[wave > wave_max_correction] = 0  # reject band

    # self.debug = telluric_location

    # extraction of uncontaminated telluric

    plateau, cluster = clustering(telluric_location, 0.5, 1)
    plateau = np.array([np.product(j) for j in plateau]).astype("bool")
    cluster = cluster[plateau]
    # med_width = np.median(cluster[:,-1])
    # mad_width = np.median(abs(cluster[:,-1] - med_width))*1.48
    telluric_kept = cluster  # cluster[(cluster[:,-1]>med_width-mad_width)&(cluster[:,-1]<med_width+mad_width),:]
    telluric_kept[:, 1] += 1
    # telluric_kept = np.hstack([telluric_kept,wave[telluric_kept[:,0],np.newaxis]])
    # plt.figure();plt.hist(telluric_kept[:,-1],bins=100)
    telluric_kept = telluric_kept[
        telluric_kept[:, -1] > np.nanmedian(telluric_kept[:, -1]) - mad(telluric_kept[:, -1])
    ]
    min_telluric_size = wave / inst_resolution / np.gradient(wave)
    telluric_kept = telluric_kept[min_telluric_size[telluric_kept[:, 0]] < telluric_kept[:, -1]]

    if debug:
        plt.figure(1)
        plt.subplot(3, 2, 1)
        plt.plot(wave, ref, color="k")
        (l4,) = plt.plot(5500 * np.ones(2), [0, 1], color="r")
        ax = plt.gca()
        plt.subplot(2, 2, 2, sharex=ax)
        plt.plot(wave, ztot, color="k")
        ax = plt.gca()
        border_y = ax.get_ylim()
        (l,) = plt.plot(5500 * np.ones(2), border_y, color="r")
        idx = find_nearest(wave, 5500)[0].astype("int")
        plt.ylim(border_y)
        for j in range(len(telluric_kept)):
            plt.axvspan(
                xmin=wave[telluric_kept[j, 0].astype("int")],
                xmax=wave[telluric_kept[j, 1].astype("int")],
                alpha=0.3,
                color="r",
            )
        plt.subplot(2, 2, 4, sharex=ax)
        plt.imshow(
            ratio_ref,
            aspect="auto",
            cmap="plasma",
            vmin=0.99,
            vmax=1.01,
            extent=[wave[0], wave[-1], 0, len(jdb)],
        )

        (l2,) = plt.plot(5500 * np.ones(2), [0, len(jdb)], color="k")

        plt.subplot(3, 2, 5)
        l5, (), (bars5,) = plt.errorbar(
            jdb % 365.25, ratio_ref[:, idx], 0.001 * np.ones(len(jdb)), fmt="ko"
        )
        plt.ylim(0.99, 1.01)
        ax3 = plt.gca()

        plt.subplot(3, 2, 3)
        l3, (), (bars3,) = plt.errorbar(
            jdb, ratio_ref[:, idx], 0.001 * np.ones(len(jdb)), fmt="ko"
        )
        plt.ylim(0.99, 1.01)
        ax4 = plt.gca()

        class Index:
            def update_data(self: spec_time_series, newx, newy):
                idx = find_nearest(wave, newx)[0].astype("int")
                l.set_xdata(newx * np.ones(len(l.get_xdata())))
                l2.set_xdata(newx * np.ones(len(l.get_xdata())))
                l4.set_xdata(newx * np.ones(len(l.get_xdata())))
                l3.set_ydata(ratio_ref[:, idx])
                l5.set_ydata(ratio_ref[:, idx])
                new_segments = [
                    np.array([[x, yt], [x, yb]])
                    for x, yt, yb in zip(jdb, ratio_ref[:, idx] + 0.001, ratio_ref[:, idx] - 0.001)
                ]
                bars3.set_segments(new_segments)
                bars5.set_segments(new_segments)
                ax3.set_ylim(
                    np.min(ratio_ref[:, idx]) - 0.002,
                    np.max(ratio_ref[:, idx]) + 0.002,
                )
                ax4.set_ylim(
                    np.min(ratio_ref[:, idx]) - 0.002,
                    np.max(ratio_ref[:, idx]) + 0.002,
                )
                plt.gcf().canvas.draw_idle()

        t = Index()

        def onclick(event):
            newx = event.xdata
            newy = event.ydata
            if event.dblclick:
                print(newx)
                t.update_data(newx, newy)

        plt.gcf().canvas.mpl_connect("button_press_event", onclick)
    else:
        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.plot(ref, color="k")
        ax = plt.gca()
        plt.subplot(3, 1, 2, sharex=ax)
        plt.plot(ztot, color="k")
        for j in range(len(telluric_kept)):
            plt.axvspan(
                xmin=telluric_kept[j, 0].astype("int"),
                xmax=telluric_kept[j, 1].astype("int"),
                alpha=0.3,
                color="r",
            )
        plt.subplot(3, 1, 3, sharex=ax)
        plt.imshow(diff_ref, aspect="auto", vmin=-0.005, vmax=0.005)

    telluric_extracted_ratio_ref = []
    telluric_extracted_ratio_ref_std = []

    ratio_ref_std2 = flux_err**2
    for j in range(len(telluric_kept)):

        norm = telluric_kept[j, 1] + 1 - telluric_kept[j, 0]
        val = np.nanmean(ratio_ref[:, telluric_kept[j, 0] : telluric_kept[j, 1] + 1], axis=1)
        val_std = (
            np.sqrt(
                np.nansum(
                    ratio_ref_std2[:, telluric_kept[j, 0] : telluric_kept[j, 1] + 1],
                    axis=1,
                )
            )
            / norm
        )

        telluric_extracted_ratio_ref.append(val)
        telluric_extracted_ratio_ref_std.append(val_std)

    telluric_extracted_ratio_ref = np.array(telluric_extracted_ratio_ref).T
    telluric_extracted_ratio_ref_std = np.array(telluric_extracted_ratio_ref_std).T
    telluric_extracted_ratio_ref -= np.median(telluric_extracted_ratio_ref, axis=0)

    plt.figure(2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(telluric_extracted_ratio_ref, aspect="auto", vmin=-0.005, vmax=0.005)
    plt.title("Water lines")
    plt.xlabel("Pixels extracted", fontsize=14)
    plt.ylabel("Time", fontsize=14)

    plt.subplot(1, 2, 2)
    plt.imshow(
        telluric_extracted_ratio_ref / np.std(telluric_extracted_ratio_ref, axis=0),
        aspect="auto",
        vmin=-0.005,
        vmax=0.005,
    )
    plt.title("Water lines")
    plt.xlabel("Pixels extracted", fontsize=14)
    plt.ylabel("Time", fontsize=14)

    c = int(equal_weight)

    X_train = (
        telluric_extracted_ratio_ref / ((1 - c) + c * np.std(telluric_extracted_ratio_ref, axis=0))
    ).T
    X_train_std = (
        telluric_extracted_ratio_ref_std
        / ((1 - c) + c * np.std(telluric_extracted_ratio_ref, axis=0))
    ).T

    # self.debug = (X_train, X_train_std)
    # io.pickle_dump({'jdb':np.array(self.table.jdb),'ratio_flux':X_train,'ratio_flux_std':X_train_std},open(root+'/Python/datasets/telluri_cenB.p','wb'))

    test2 = table_cls(X_train)

    test2.WPCA("wpca", weight=1 / X_train_std**2, comp_max=nb_pca_comp)

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
    z_max = test2.zscore_components[-10:].max()
    z_min = test2.zscore_components[-10:].min()
    vec_relevant = np.arange(len(test2.zscore_components)) * (
        (test2.zscore_components > z_max) | (test2.zscore_components < z_min)
    )
    pca_comp_kept2 = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])

    plt.axhspan(ymin=z_min, ymax=z_max, alpha=0.2, color="k")
    plt.axhline(y=0, color="k")
    plt.subplot(3, 1, 3)
    plt.xlabel("# PCA components", fontsize=13)
    plt.ylabel(r"$\Phi(0)$", fontsize=13)
    plt.plot(np.arange(1, len(test2.phi_components) + 1), test2.phi_components)
    plt.scatter(np.arange(1, len(test2.phi_components) + 1), test2.phi_components)
    plt.axhline(y=0.5, color="k")  # type: ignore
    phi_max = test2.phi_components[-10:].max()
    phi_min = test2.phi_components[-10:].min()
    plt.axhspan(ymin=phi_min, ymax=phi_max, alpha=0.2, color="k")
    vec_relevant = np.arange(len(test2.phi_components)) * (
        (test2.phi_components > phi_max) | (test2.phi_components < phi_min)
    )
    pca_comp_kept = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])
    pca_comp_kept = np.max([pca_comp_kept, pca_comp_kept2])

    if nb_pca_comp_kept is not None:
        pca_comp_kept = nb_pca_comp_kept

    if pca_comp_kept > nb_pca_max_kept:
        pca_comp_kept = nb_pca_max_kept

    logging.info("Nb PCA comp kept : %.0f" % (pca_comp_kept))

    plt.savefig(self.dir_root + "IMAGES/telluric_PCA_variances.pdf")

    plt.figure(figsize=(15, 10))
    for j in range(pca_comp_kept):
        plt.subplot(pca_comp_kept, 2, 2 * j + 1)
        plt.scatter(jdb, test2.vec[:, j])
        plt.subplot(pca_comp_kept, 2, 2 * j + 2)
        plt.scatter((jdb - phase_mod) % 365.25, test2.vec[:, j])
    plt.subplots_adjust(top=0.95, bottom=0.07, left=0.07, right=0.95, hspace=0)
    plt.savefig(self.dir_root + "IMAGES/telluric_PCA_vectors.pdf")

    to_be_fit = ratio_ref / (np.std(ratio_ref, axis=0) + epsilon)

    rcorr = np.zeros(len(wave))
    for j in range(pca_comp_kept):
        proxy1 = test2.vec[:, j]
        rslope1 = np.median(
            (to_be_fit - np.mean(to_be_fit, axis=0)) / ((proxy1 - np.mean(proxy1))[:, np.newaxis]),
            axis=0,
        )

        rcorr1 = abs(rslope1 * np.std(proxy1) / (np.std(to_be_fit, axis=0) + epsilon))
        rcorr = np.nanmax([rcorr1, rcorr], axis=0)
    rcorr[np.isnan(rcorr)] = 0
    rcorr_telluric_free = rcorr[
        int(find_nearest(wave, 4800)[0]) : int(find_nearest(wave, 5000)[0])
    ]
    rcorr_telluric = rcorr[int(find_nearest(wave, 5800)[0]) : int(find_nearest(wave, 6000)[0])]

    plt.figure(figsize=(8, 6))
    bins_contam, bins, dust = plt.hist(
        rcorr_telluric,
        label="contaminated region",
        bins=np.linspace(0, 1, 100),
        alpha=0.5,
    )
    bins_control, bins, dust = plt.hist(
        rcorr_telluric_free,
        bins=np.linspace(0, 1, 100),
        label="free region",
        alpha=0.5,
    )
    plt.legend()
    plt.yscale("log")
    bins = bins[0:-1] + np.diff(bins) * 0.5
    sum_a = np.sum(bins_contam[bins > 0.40])
    sum_b = np.sum(bins_control[bins > 0.40])
    crit = int(sum_a > (2 * sum_b))
    check = ["r", "g"][crit]  # five times more correlation than in the control group
    plt.xlabel(r"|$\mathcal{R}_{pearson}$|", fontsize=14, fontweight="bold", color=check)
    plt.title("Density", color=check)
    plot_color_box(color=check)

    plt.savefig(self.dir_root + "IMAGES/telluric_control_check.pdf")
    logging.info("%.0f versus %.0f" % (sum_a, sum_b))

    if crit:
        logging.info("Control check sucessfully performed: telluric")
    else:
        print(
            Fore.YELLOW
            + " [WARNING] Control check failed. Correction may be poorly performed for: telluric"
            + Fore.RESET
        )

    collection = table_cls(
        ratio_ref.T[telluric_location.astype("bool")]
    )  # do fit only on flag position

    weights = 1 / (flux_err) ** 2
    weights = weights.T[telluric_location.astype("bool")]
    IQ = IQ_fun(collection.table, axis=1)
    Q1 = np.nanpercentile(collection.table, 25, axis=1)
    Q3 = np.nanpercentile(collection.table, 75, axis=1)
    sup = Q3 + 1.5 * IQ
    inf = Q1 - 1.5 * IQ
    out = (collection.table > sup[:, np.newaxis]) | (collection.table < inf[:, np.newaxis])
    weights[out] = np.min(weights) / 100

    # base_vec = np.vstack([np.ones(len(flux)), jdb-np.median(jdb), test2.vec[:,0:pca_comp_kept].T])
    base_vec = np.vstack([np.ones(len(flux)), test2.vec[:, 0:pca_comp_kept].T])
    collection.fit_base(base_vec, weight=weights, num_sim=1)
    # collection.coeff_fitted[:,3] = 0 #supress the linear trend fitted

    del weights
    del flux_err

    correction = np.zeros((len(wave), len(jdb)))
    correction[telluric_location.astype("bool")] = collection.coeff_fitted.dot(base_vec)
    correction = np.transpose(correction)

    correction[correction == 0] = 1
    correction_backup = correction.copy()

    if np.sum(abs(berv)) != 0:
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, correction[j], 0 * wave)
            test.x = doppler_r(test.x, berv[j] * 1000)[0]
            test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
            correction_backup[j] = test.y

    index_min_backup = int(find_nearest(wave, doppler_r(wave[0], berv.max() * 1000)[0])[0])
    index_max_backup = int(find_nearest(wave, doppler_r(wave[-1], berv.min() * 1000)[0])[0])
    correction_backup[:, 0:index_min_backup] = 1
    correction_backup[:, index_max_backup:] = 1
    index_hole_right = int(
        find_nearest(wave, hole_right + 1)[0]
    )  # correct 1 angstrom band due to stange artefact at the border of the gap
    index_hole_left = int(
        find_nearest(wave, hole_left - 1)[0]
    )  # correct 1 angstrom band due to stange artefact at the border of the gap
    correction_backup[:, index_hole_left : index_hole_right + 1] = 1

    del correction

    ratio2_backup = ratio_backup - correction_backup + 1

    del correction_backup

    new_conti = flux_backup / (ref * ratio2_backup + epsilon)
    new_continuum = new_conti.copy()
    new_continuum[flux == 0] = conti[flux == 0]
    new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]
    new_continuum[new_continuum == 0] = conti[new_continuum == 0]

    del ratio2_backup
    del ratio_backup

    diff2_backup = flux_backup / new_continuum - ref

    idx_min = find_nearest(wave, 5700)[0]
    idx_max = find_nearest(wave, 5900)[0] + 1

    new_wave = wave[int(idx_min) : int(idx_max)]

    fig = plt.figure(figsize=(21, 9))
    plt.axes([0.05, 0.55, 0.90, 0.40])
    ax = plt.gca()
    my_colormesh(
        new_wave,
        np.arange(len(diff_ref)),
        100 * diff_backup[:, int(idx_min) : int(idx_max)],
        zoom=zoom,
        vmin=low_cmap,
        vmax=high_cmap,
        cmap=cmap,
    )
    plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
    plt.ylabel("Spectra  indexes (time)", fontsize=16)
    plt.ylim(0, None)
    ax = plt.gca()
    cbaxes = fig.add_axes([0.95, 0.55, 0.01, 0.40])
    ax1 = plt.colorbar(cax=cbaxes)
    ax1.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

    plt.axes([0.05, 0.1, 0.90, 0.40], sharex=ax, sharey=ax)
    my_colormesh(
        new_wave,
        np.arange(len(diff_ref)),
        100 * diff2_backup[:, int(idx_min) : int(idx_max)],
        zoom=zoom,
        vmin=low_cmap,
        vmax=high_cmap,
        cmap=cmap,
    )
    plt.tick_params(direction="in", top=True, right=True, labelbottom=True)
    plt.ylabel("Spectra  indexes (time)", fontsize=16)
    plt.xlabel(r"Wavelength [$\AA$]", fontsize=16)
    plt.ylim(0, None)
    ax = plt.gca()
    cbaxes2 = fig.add_axes([0.95, 0.1, 0.01, 0.40])
    ax2 = plt.colorbar(cax=cbaxes2)
    ax2.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

    plt.savefig(self.dir_root + "IMAGES/Correction_telluric.png")

    correction_pca = diff_ref_to_correct - diff2_backup
    to_be_saved = {"wave": wave, "correction_map": correction_pca}

    # myf.pickle_dump(to_be_saved,open(self.dir_root+'CORRECTION_MAP/map_matching_pca.p','wb'))
    np.save(
        self.dir_root + "CORRECTION_MAP/map_matching_pca.npy",
        to_be_saved["correction_map"].astype("float32"),
    )

    print(" Computation of the new continua, wait ... \n")
    time.sleep(0.5)

    self.info_reduction["matching_pca"] = {
        "reference_spectrum": reference,
        "sub_dico_used": sub_dico_correction,
        "nb_pca_component": pca_comp_kept,
        "step": step + 1,
        "valid": True,
    }
    self.update_info_reduction()

    fname = self.dir_root + "WORKSPACE/CONTINUUM/Continuum_%s.npy" % ("matching_pca")
    np.save(fname, new_continuum.astype("float32"))

    self.dico_actif = "matching_pca"

    plt.show(block=False)
