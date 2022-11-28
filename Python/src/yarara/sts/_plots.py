from __future__ import annotations

import glob
from typing import TYPE_CHECKING, Literal

import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np

from ..stats import find_nearest
from ..util import assert_never, doppler_r

if TYPE_CHECKING:
    from . import spec_time_series


def yarara_plot_all(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    wave_min: float = 4400.0,
    wave_max: float = 4500.0,
    plot_median: bool = False,
    berv_keys: str = "none",
    cmap: str = "brg",
    color: str = "CaII",
    relatif: bool = False,
    new: bool = True,
    substract_map=[],
    p_noise: float = 1 / np.inf,
):
    """
    Plot all the RASSINE spectra in the same plot

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    wave_min : The minimum xlim axis
    wave_max : The maximum xlim axis
    plot_median : True/False, Plot the median of the spectrum
    berv_keys : The berv keyword from RASSINE dictionnary to remove the berv from the spectra (berv in kms)
    planet : True/False to use the flux containing the injected planet or not

    """

    directory = self.directory
    planet = self.planet
    self.import_table()
    self.import_material()
    load = self.material
    wave = np.array(load["wave"])
    jdb = np.array(self.table.jdb)
    try:
        berv = np.array(self.table[berv_keys])
    except:
        berv = 0 * jdb
    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)
    median_spec = []

    jet = plt.get_cmap(cmap)
    index = self.table[color]
    cNorm = mplcolors.Normalize(
        vmin=float(np.percentile(index, 16)), vmax=float(np.percentile(index, 84))
    )
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    maps = np.zeros((len(self.table["jdb"]), len(load["wave"])))
    for m in substract_map:
        maps += np.load(self.dir_root + "CORRECTION_MAP/map_matching_" + m + ".npy")

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    noise_matrix, noise_values = self.yarara_poissonian_noise(
        noise_wanted=p_noise, wave_ref=None, flat_snr=True
    )

    all_flux, all_conti = self.import_sts_flux(load=["flux" + kw, sub_dico])

    begin = int(find_nearest(wave, wave_min)[0][0])
    end = int(find_nearest(wave, wave_max)[0][0])

    wave_cut = wave[begin:end]
    all_flux = all_flux[:, begin:end]
    all_conti = all_conti[:, begin:end]
    noise_matrix = noise_matrix[:, begin:end]
    maps = maps[:, begin:end]
    median = np.array(load["reference_spectrum"])[begin:end]

    flux_norm = all_flux / all_conti - relatif * median - maps

    median_spec = np.median(flux_norm, axis=0)

    if new:
        plt.figure()
    for i, j in enumerate(files):
        colorVal = scalarMap.to_rgba(index[i])
        plt.plot(
            doppler_r(wave_cut, berv[i] * 1000)[1],
            flux_norm[i] + noise_matrix[i],
            color=colorVal,
            alpha=0.4,
        )

    if plot_median:
        median_spec = np.array(median_spec)
        plt.plot(wave_cut, median_spec, color="k")


def yarara_comp_all(
    self: spec_time_series,
    sub_dico1: str = "matching_cosmics",
    sub_dico2: str = "matching_mad",
    analysis: Literal[
        "h2o_1", "h2o_2", "h2o_3", "o2_1", "o2_2", "o2_3", "activity", "ha"
    ] = "h2o_1",
):
    if analysis == "h2o_1":
        wave_min = 5874.9
        wave_max = 5925.1
        color = "telluric_contrast"
        cmap = "brg"
    elif analysis == "h2o_2":
        wave_min = 6459.9
        wave_max = 6500.1
        color = "telluric_contrast"
        cmap = "brg"
    elif analysis == "h2o_3":
        wave_min = 7159.9
        wave_max = 7250.1
        color = "telluric_contrast"
        cmap = "brg"
    elif analysis == "o2_1":
        wave_min = 6274.9
        wave_max = 6310.1
        color = "telluric_contrast"
        cmap = "brg"
    elif analysis == "o2_2":
        wave_min = 6859.9
        wave_max = 7050.1
        color = "telluric_contrast"
        cmap = "brg"
    elif analysis == "o2_3":
        wave_min = 7584.9
        wave_max = 7760.1
        color = "telluric_contrast"
        cmap = "brg"
    elif analysis == "activity":
        wave_min = 4199.9
        wave_max = 4240.1
        color = "CaII"
        cmap = "brg"
    elif analysis == "ha":
        wave_min = 6549.9
        wave_max = 6580.1
        color = "telluric_contrast"
        cmap = "brg"
    else:
        assert_never(analysis)

    wave = self.spectrum(norm=True, num=1).x
    if wave_min < np.min(wave):
        wave_min = np.min(wave)
    if wave_min > np.max(wave):
        wave_min = np.max(wave)
    if wave_max > np.max(wave):
        wave_max = np.max(wave)
    if wave_max < np.min(wave):
        wave_max = np.min(wave)

    if wave_min != wave_max:
        plt.figure(figsize=(18, 7))
        plt.subplot(2, 1, 1)
        plt.title(f"Before YARARA ({sub_dico1})", fontsize=16)
        self.yarara_plot_all(
            wave_min=wave_min,
            wave_max=wave_max,
            sub_dico=sub_dico1,
            color=color,
            new=False,
            cmap=cmap,
            plot_median=True,
        )
        ax = plt.gca()
        plt.xlabel(r"Wavelength [$\AA$]", fontsize=16)
        plt.ylabel(r"Flux", fontsize=16)

        plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
        plt.title(f"After YARARA ({sub_dico2})", fontsize=16)
        self.yarara_plot_all(
            wave_min=wave_min,
            wave_max=wave_max,
            sub_dico=sub_dico2,
            color=color,
            new=False,
            cmap=cmap,
            plot_median=True,
        )
        plt.xlabel(r"Wavelength [$\AA$]", fontsize=16)
        plt.ylabel(r"Flux", fontsize=16)
        plt.xlim(wave_min, wave_max)
        plt.ylim(-0.01, 1.09)
        plt.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.10, hspace=0.40)

        plt.savefig(self.dir_root + f"IMAGES/Correction_1d_{analysis}.png")
