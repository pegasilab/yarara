from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Union

import matplotlib.pylab as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from ... import io
from ...analysis import table as table_cls
from ...analysis import tableXY
from ...paths import paths, root
from ...plots import my_colormesh
from ...stats import IQ as IQ_fun
from ...stats import find_nearest, smooth2d
from ...util import assert_never, doppler_r, flux_norm_std, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_correct_oxygen(
    self: spec_time_series,
    sub_dico: str = "matching_telluric",
    reference: Union[
        int, Literal["snr"], Literal["median"], Literal["master"], Literal["zeros"]
    ] = "master",
    wave_min: float = 5760.0,
    wave_max: float = 5850.0,
    oxygene_bands: Sequence[Sequence[float]] = [
        [5787.0, 5835.0],
        [6275.0, 6340.0],
        [6800.0, 6950.0],
    ],
) -> None:

    """
    Display the time-series spectra with proxies and its correlation

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    wave_min : Minimum x axis limit
    wave_max : Maximum x axis limit
    smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
    """

    print_box("\n---- RECIPE : CORRECTION TELLURIC OXYGEN ----\n")

    directory = self.directory

    zoom = self.zoom
    smooth_map = self.smooth_map
    cmap = self.cmap
    planet = self.planet
    low_cmap = self.low_cmap * 100
    high_cmap = self.high_cmap * 100
    self.import_material()
    self.import_table()
    load = self.material
    wave = np.array(load["wave"])

    self.import_info_reduction()

    step = self.info_reduction[sub_dico]["step"]

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
        logging.info("PLANET ACTIVATED")

    if sub_dico is None:
        sub_dico = self.dico_actif
    logging.info(f"DICO {sub_dico}")

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    all_flux, all_flux_err, conti, conti_err = self.import_sts_flux(
        load=["flux" + kw, "flux_err", sub_dico, "continuum_err"]
    )
    flux, flux_err = flux_norm_std(all_flux, all_flux_err, conti + epsilon, conti_err)

    def idx_wave(wavelength):
        return int(find_nearest(wave, wavelength)[0])

    if isinstance(reference, int):
        ref = flux[reference]
    elif reference == "snr":
        ref = flux[snr.argmax()]
    elif reference == "median":
        ref = np.median(flux, axis=0)
    elif reference == "master":
        ref = np.array(load["reference_spectrum"])
    elif reference == "zeros":
        ref = 0.0 * np.median(flux, axis=0)
    else:
        assert_never(reference)

    diff_ref = smooth2d(flux - ref, smooth_map)
    ratio_ref = smooth2d(flux / (ref + epsilon), smooth_map)

    diff_backup = diff_ref.copy()
    ratio_backup = ratio_ref.copy()

    if np.sum(abs(berv)) != 0:
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, ratio_ref[j], flux_err[j] / (ref + epsilon))
            test.x = doppler_r(test.x, berv[j] * 1000)[1]
            test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
            ratio_ref[j] = test.y
            flux_err[j] = test.yerr

    inside_oxygene_mask = np.zeros(len(ratio_ref.T))
    for k in range(len(oxygene_bands)):
        first = find_nearest(wave, oxygene_bands[k][0])[0]
        last = find_nearest(wave, oxygene_bands[k][1])[0]
        inside_oxygene_mask[int(first) : int(last)] = 1
    # inside_oxygene[wave>6600] = 0  #reject band [HYPERPARAMETER HARDCODED]
    inside_oxygene = inside_oxygene_mask.astype("bool")

    vec = ratio_ref.T[inside_oxygene]
    collection = table_cls(vec)

    print(np.shape(flux_err))

    weights = 1 / (flux_err) ** 2
    weights = weights.T[inside_oxygene]
    IQ = IQ_fun(collection.table, axis=1)
    Q1 = np.nanpercentile(collection.table, 25, axis=1)
    Q3 = np.nanpercentile(collection.table, 75, axis=1)
    sup = Q3 + 1.5 * IQ
    inf = Q1 - 1.5 * IQ
    out = (collection.table > sup[:, np.newaxis]) | (collection.table < inf[:, np.newaxis])
    weights[out] = np.min(weights) / 100

    base_vec = np.vstack(
        [np.ones(len(flux)), jdb - np.median(jdb), jdb**2 - np.median(jdb**2)]
    )  # fit offset + para trend par oxygene line (if binary drift substract)
    collection.fit_base(base_vec, weight=weights, num_sim=1)
    correction = np.zeros((len(wave), len(jdb)))
    correction[inside_oxygene] = collection.coeff_fitted.dot(base_vec)
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

    del flux_err
    del weights
    del correction

    ratio2_backup = ratio_backup - correction_backup + 1  # type: ignore

    del correction_backup
    del ratio_backup

    new_conti = conti * flux / (ref * ratio2_backup + epsilon)
    new_continuum = new_conti.copy()
    new_continuum[flux == 0] = conti[flux == 0]

    del ratio2_backup

    diff2_backup = flux * conti / new_continuum - ref

    # plot end

    idx_min = 0
    idx_max = len(wave)

    if wave_min is not None:
        idx_min = find_nearest(wave, wave_min)[0]
    if wave_max is not None:
        idx_max = find_nearest(wave, wave_max)[0] + 1

    new_wave = wave[int(idx_min) : int(idx_max)]

    fig = plt.figure(figsize=(21, 9))

    plt.axes((0.05, 0.66, 0.90, 0.25))
    my_colormesh(
        new_wave,
        np.arange(len(diff_backup)),
        100 * diff_backup[:, int(idx_min) : int(idx_max)],
        zoom=zoom,
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
        np.arange(len(diff_backup)),
        100 * diff2_backup[:, int(idx_min) : int(idx_max)],
        zoom=zoom,
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
        np.arange(len(diff_backup)),
        100 * (diff_backup - diff2_backup)[:, int(idx_min) : int(idx_max)],
        zoom=zoom,
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

    plt.savefig(self.dir_root + "IMAGES/Correction_oxygen.png")

    pre_map = np.zeros(np.shape(diff_backup))
    if sub_dico == "matching_oxygen":
        sub_dico = self.info_reduction[sub_dico]["sub_dico_used"]
        step -= 1
        # pre_map = pd.read_pickle(self.dir_root+'CORRECTION_MAP/map_matching_oxygen.p')['correction_map']
        pre_map = np.load(self.dir_root + "CORRECTION_MAP/map_matching_oxygen.npy")

    correction_oxygen = diff_backup - diff2_backup
    to_be_saved = {"wave": wave, "correction_map": correction_oxygen + pre_map}
    # myf.pickle_dump(to_be_saved,open(self.dir_root+'CORRECTION_MAP/map_matching_oxygen.p','wb'))
    np.save(
        self.dir_root + "CORRECTION_MAP/map_matching_oxygen.npy",
        to_be_saved["correction_map"].astype("float32"),
    )

    logging.info("Computation of the new continua, wait")
    time.sleep(0.5)

    self.info_reduction["matching_oxygen"] = {
        "reference_spectrum": reference,
        "sub_dico_used": sub_dico,
        "step": step + 1,
        "valid": True,
    }
    self.update_info_reduction()

    fname = self.dir_root + "WORKSPACE/CONTINUUM/Continuum_%s.npy" % ("matching_oxygen")
    np.save(fname, new_continuum.astype("float32"))

    self.dico_actif = "matching_oxygen"
