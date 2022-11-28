from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import matplotlib.pylab as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from ... import iofun
from ...analysis import table as table_cls
from ...analysis import tableXY
from ...paths import paths, root
from ...plots import my_colormesh
from ...stats import IQ as IQ_fun
from ...stats import clustering, find_nearest, flat_clustering, merge_borders, smooth2d
from ...util import assert_never, doppler_r, flux_norm_std, print_box

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_correct_telluric_proxy(
    self: spec_time_series,
    sub_dico: str = "matching_fourier",
    sub_dico_output: str = "telluric",
    wave_min: float = 5700.0,
    wave_max: float = 5900.0,
    reference: str = "master",
    smooth_corr: int = 1,
    proxies_corr: List[str] = ["h2o_depth", "h2o_fwhm"],
    proxies_detrending_: None = None,
    wave_min_correction_: Union[float, int] = 4400.0,
    wave_max_correction_: None = None,  # TODO: correct before
    min_r_corr_: float = 0.40,
    sigma_ext: int = 2,
) -> None:

    """
    Display the time-series spectra with proxies and its correlation

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    wave_min : Minimum x axis limit
    wave_max : Maximum x axis limit
    zoom : int-type, to improve the resolution of the 2D plot
    smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
    reference : 'median', 'snr' or 'master' to select the reference normalised spectrum usedin the difference
    proxy1_corr : keyword  of the first proxies from RASSINE dictionnary to use in the correlation
    proxy1_detrending : Degree of the polynomial fit to detrend the proxy
    proxy2_corr : keyword  of the second proxies from RASSINE dictionnary to use in the correlation
    proxy2_detrending : Degree of the polynomial fit to detrend the proxy
    cmap : cmap of the 2D plot
    min_wave_correction : wavelength limit above which to correct
    min_r_corr : minimum correlation coefficient of one of the two proxies to consider a line as telluric
    dwin : window correction increase by dwin to slightly correct above around the peak of correlation
    positive_coeff : The correction can only be absorption line profile moving and no positive
    """

    if sub_dico_output == "telluric":
        print_box("\n---- RECIPE : CORRECTION TELLURIC WATER ----\n")
        name = "water"
    elif sub_dico_output == "oxygen":
        print_box("\n---- RECIPE : CORRECTION TELLURIC OXYGEN ----\n")
        name = "oxygen"
    else:
        print_box("\n---- RECIPE : CORRECTION TELLURIC PROXY ----\n")
        name = "telluric"

    directory = self.directory

    zoom = self.zoom
    smooth_map = self.smooth_map
    low_cmap = self.low_cmap
    high_cmap = self.high_cmap
    cmap = self.cmap
    planet = self.planet

    self.import_material()
    self.import_table()
    self.import_info_reduction()

    step = self.info_reduction[sub_dico]["step"]

    jdb = np.array(self.table["jdb"])
    snr = np.array(self.table["snr"])
    berv = np.array(self.table["berv"])
    rv_shift = np.array(self.table["rv_shift"])

    mean_berv = np.mean(berv)
    berv = berv - mean_berv - rv_shift

    proxy = np.array([np.array(self.table[proxy_name]) for proxy_name in proxies_corr]).T

    load = self.material
    wave = np.array(load["wave"])

    file_ref = self.import_spectrum()
    hole_left = file_ref["parameters"]["hole_left"]
    hole_right = file_ref["parameters"]["hole_right"]
    dgrid = file_ref["parameters"]["dwave"]

    epsilon = 1e-12

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    print(f"\n---- DICO {sub_dico} used ----\n")

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    all_flux, all_flux_err, conti, conti_err = self.import_sts_flux(
        load=["flux" + kw, "flux_err", sub_dico, "continuum_err"]
    )
    flux, err_flux = flux_norm_std(all_flux, all_flux_err, conti + epsilon, conti_err)

    if proxies_detrending_ is None:
        proxies_detrending: NDArray[np.float64] = np.zeros((len(proxies_corr),))
    else:
        proxies_detrending = np.array(proxies_detrending_)

    for k in range(len(proxies_corr)):
        proxy1 = tableXY(jdb, proxy[:, k])
        proxy1.substract_polyfit(proxies_detrending[k])
        proxy[:, k] = proxy1.detrend_poly.y

    if reference == "snr":
        ref = flux[snr.argmax()]
    elif reference == "median":
        logging.info("Reference spectrum : median")
        ref = np.median(flux, axis=0)
    elif reference == "master":
        logging.info("Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif type(reference) == int:
        logging.info(f"Reference spectrum : spectrum {reference:.0f}")
        ref = flux[reference]
    else:
        ref = 0 * np.median(flux, axis=0)

    ratio = smooth2d(flux / (ref + 1e-6), smooth_map)
    ratio_backup = ratio.copy()

    diff_backup = smooth2d(flux - ref, smooth_map)

    if np.sum(abs(berv)) != 0:
        for j in tqdm(np.arange(len(flux))):
            test = tableXY(wave, ratio[j], err_flux[j] / (ref + 1e-6))
            test.x = doppler_r(test.x, berv[j] * 1000)[1]
            test.interpolate(new_grid=wave, method="cubic", replace=True, interpolate_x=False)
            ratio[j] = test.y
            err_flux[j] = test.yerr

    t = table_cls(ratio)
    t.rms_w(1 / (err_flux) ** 2, axis=0)

    rslope = []
    rcorr = []
    for k in range(len(proxies_corr)):
        rslope.append(
            np.median(
                (ratio - np.mean(ratio, axis=0))
                / ((proxy[:, k] - np.mean(proxy[:, k]))[:, np.newaxis]),
                axis=0,
            )
        )
        rcorr.append(abs(rslope[-1] * np.std(proxy[:, k]) / (t.rms + epsilon)))

    # rcorr1 = abs(rslope1*np.std(proxy1)/np.std(ratio,axis=0))
    # rcorr2 = abs(rslope2*np.std(proxy2)/np.std(ratio,axis=0))

    rslope = np.array(rslope)
    rcorr = np.array(rcorr)

    rcorr = np.max(rcorr, axis=0)
    r_corr = tableXY(wave, rcorr)
    r_corr.smooth(box_pts=smooth_corr, shape="savgol", replace=True)
    rcorr = r_corr.y

    wave_min_correction: float = (
        np.min(wave) if wave_min_correction_ is None else wave_min_correction_
    )
    wave_max_correction: float = (
        np.max(wave) if wave_max_correction_ is None else wave_max_correction_
    )

    if min_r_corr_ is None:
        min_r_corr: float = np.percentile(rcorr[wave < 5400], 75) + 1.5 * IQ_fun(
            rcorr[wave < 5400]
        )  # type: ignore
        logging.info(
            f"Significative R Pearson detected as {min_r_corr:.2f} based on wavelength smaller than 5400"
        )
    else:
        min_r_corr = min_r_corr_

    first_guess_position = (
        (rcorr > min_r_corr) & (wave > wave_min_correction) & (wave < wave_max_correction)
    )  # only keep >0.4 and redder than 4950 AA
    second_guess_position = first_guess_position

    # fwhm_telluric = np.median(self.table['telluric_fwhm'])
    fwhm_telluric = self.star_info["FWHM"]["telluric"]  # 09.08.21
    val, borders = clustering(first_guess_position.astype(np.float64), 0.5, 1)
    val = np.array([np.product(v) for v in val]).astype("bool")
    borders = borders[val]
    wave_tel = wave[(0.5 * (borders[:, 0] + borders[:, 1])).astype("int")]
    extension = np.round(sigma_ext * fwhm_telluric / 3e5 * wave_tel / dgrid, 0).astype("int")
    borders[:, 0] -= extension
    borders[:, 1] += extension
    borders[:, 2] = borders[:, 1] - borders[:, 0] + 1
    borders = merge_borders(borders)
    second_guess_position = flat_clustering(len(wave), borders).astype("bool")

    guess_position = np.arange(len(second_guess_position))[second_guess_position]

    correction = np.zeros((len(wave), len(jdb)))

    len_segment = 10000
    print("\n")
    for k in range(len(guess_position) // len_segment + 1):
        print(
            f" [INFO] Segment {k + 1:.0f}/{len(guess_position) // len_segment + 1:.0f} being reduced\n"
        )
        second_guess_position = guess_position[k * len_segment : (k + 1) * len_segment]
        # print(second_guess_position)

        collection = table_cls(ratio.T[second_guess_position])

        base_vec = np.vstack(
            [np.ones(len(flux))] + [proxy[:, k] for k in range(len(proxies_corr))]
        )
        # rm outliers and define weight for the fit
        weights = (1 / (err_flux / (ref + 1e-6)) ** 2).T[second_guess_position]
        IQ = IQ_fun(collection.table, axis=1)
        Q1 = np.nanpercentile(collection.table, 25, axis=1)
        Q3 = np.nanpercentile(collection.table, 75, axis=1)
        sup = Q3 + 1.5 * IQ
        inf = Q1 - 1.5 * IQ
        out = (collection.table > sup[:, np.newaxis]) | (collection.table < inf[:, np.newaxis])
        weights[out] = np.min(weights) / 100

        collection.fit_base(base_vec, weight=weights, num_sim=1)

        correction[second_guess_position] = collection.coeff_fitted.dot(base_vec)

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

    #        if positive_coeff:
    #            correction_backup[correction_backup>0] = 0

    ratio2_backup = ratio_backup - correction_backup + 1  # type: ignore

    del correction_backup
    del correction
    del err_flux

    new_conti = conti * flux / (ref * ratio2_backup + epsilon)
    new_continuum = new_conti.copy()
    new_continuum[flux == 0] = conti[flux == 0]

    del ratio2_backup
    del ratio_backup

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
        diff_backup[:, int(idx_min) : int(idx_max)],
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
    plt.colorbar(cax=cbaxes)

    plt.axes((0.05, 0.375, 0.90, 0.25), sharex=ax, sharey=ax)
    my_colormesh(
        new_wave,
        np.arange(len(diff_backup)),
        diff2_backup[:, int(idx_min) : int(idx_max)],
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
    plt.colorbar(cax=cbaxes2)

    plt.axes((0.05, 0.09, 0.90, 0.25), sharex=ax, sharey=ax)
    my_colormesh(
        new_wave,
        np.arange(len(diff_backup)),
        (diff_backup - diff2_backup)[:, int(idx_min) : int(idx_max)],
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
    plt.colorbar(cax=cbaxes3)

    plt.savefig(self.dir_root + "IMAGES/Correction_" + name + ".png")

    correction_water = diff_backup - diff2_backup
    to_be_saved = {"wave": wave, "correction_map": correction_water}

    np.save(
        self.dir_root + "CORRECTION_MAP/map_matching_" + sub_dico_output + ".npy",
        to_be_saved["correction_map"].astype("float32"),
    )

    print("\nComputation of the new continua, wait ... \n")
    time.sleep(0.5)

    if sub_dico == "matching_" + sub_dico_output:
        sub_dico = self.info_reduction[sub_dico]["sub_dico_used"]

    self.info_reduction["matching_" + sub_dico_output] = {
        "reference_spectrum": reference,
        "sub_dico_used": sub_dico,
        "proxies": proxies_corr,
        "min_wave_correction ": wave_min_correction,
        "minimum_r_corr": min_r_corr,
        "step": step + 1,
        "valid": True,
    }
    self.update_info_reduction()

    fname = self.dir_root + f"WORKSPACE/CONTINUUM/Continuum_{'matching_' + sub_dico_output}.npy"
    np.save(fname, new_continuum.astype("float32"))

    self.dico_actif = "matching_" + sub_dico_output
