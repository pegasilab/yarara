from __future__ import annotations

import glob as glob
import logging
from typing import TYPE_CHECKING, Literal, Optional, Union

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from ...paths import root
from ...plots import my_colormesh
from ...stats import find_nearest, smooth
from ...util import assert_never, flux_norm_std, print_box, yarara_artefact_suppressed

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_correct_smooth(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    reference: str = "median",
    wave_min: float = 4200.0,
    wave_max: float = 4300.0,
    window_ang: int = 5.0,
) -> None:

    print_box("\n---- RECIPE : CORRECTION SMOOTH ----\n")

    directory = self.directory

    low_cmap = self.low_cmap * 100
    high_cmap = self.high_cmap * 100
    cmap = self.cmap
    planet = self.planet

    self.import_material()
    self.import_table()
    self.import_info_reduction()

    snr = np.array(self.table["snr"])

    load = self.material
    wave = np.array(load["wave"])

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        logging.info("PLANET ACTIVATED")

    if sub_dico is None:
        sub_dico = self.dico_actif

    logging.info(f"DICO {sub_dico} used")

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    file_ref = self.import_spectrum()
    hl = file_ref["parameters"]["hole_left"]
    hr = file_ref["parameters"]["hole_right"]
    dgrid = file_ref["parameters"]["dwave"]

    flux, all_flux_err, conti, conti_err = self.import_sts_flux(
        load=["flux" + kw, "flux_err", sub_dico, "continuum_err"]
    )
    all_flux, _ = flux_norm_std(flux, all_flux_err, conti + epsilon, conti_err)

    step = self.info_reduction[sub_dico]["step"]

    if isinstance(reference, int):
        logging.info("Reference spectrum : spectrum %d" % (reference))
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
    ].copy()  # to suppress mystic nan appearing
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

    plt.axes((0.05, 0.66, 0.90, 0.25))
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

    plt.axes((0.05, 0.375, 0.90, 0.25), sharex=ax, sharey=ax)
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

    plt.axes((0.05, 0.09, 0.90, 0.25), sharex=ax, sharey=ax)
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
        np.save(
            self.dir_root + "CORRECTION_MAP/map_matching_smooth.npy",
            to_be_saved["correction_map"].astype("float32"),
        )
        name = "smooth"
        recenter = False
        ref_name = str(reference)
        savgol_window = 0

    self.info_reduction["matching_" + name] = {
        "reference_continuum": ref_name,
        "sub_dico_used": sub_dico,
        "savgol_window": savgol_window,
        "window_ang": window_ang,
        "step": step + 1,
        "recenter": recenter,
        "valid": True,
    }
    self.update_info_reduction()

    count_file = -1
    for j in tqdm(files):
        count_file += 1
        mask = yarara_artefact_suppressed(
            conti[count_file],
            new_continuum[count_file],
            larger_than=50,
            lower_than=-50,
        )
        new_continuum[count_file][mask] = conti[count_file][mask]

    fname = self.dir_root + f"WORKSPACE/CONTINUUM/Continuum_{'matching_smooth'}.npy"
    np.save(fname, new_continuum.astype("float32"))

    logging.info("Computation of the new continua, wait ...")
