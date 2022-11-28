from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Literal, Optional, Union, cast

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ... import iofun, materials
from ...analysis import tableXY
from ...plots import my_colormesh
from ...stats import IQ, find_nearest, flat_clustering
from ...util import doppler_r

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_median_master(
    self: spec_time_series,
    *,
    sub_dico: Optional[str] = "matching_diff",
    method: Union[Literal["mean"], Literal["median"]] = "mean",
    suppress_telluric: bool = True,
    shift_spectrum: bool = False,
    telluric_tresh: float = 0.001,
    wave_min: float = 5750.0,
    wave_max: float = 5900.0,
    jdb_range: List[int] = [-100000, 100000, 1],
) -> None:
    """
    Produce a median master by masking region of the spectrum

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    telluric_tresh : Treshold used to cover the position of the contaminated wavelength
    wave_min : The minimum xlim axis
    wave_max : The maximum xlim axis
    """

    logging.info("RECIPE : PRODUCE MASTER MEDIAN SPECTRUM")

    self.import_table()
    self.import_material()
    load = self.material
    epsilon = 1e-6

    wavelength = np.array(load["wave"])
    self.wave = wavelength

    planet = self.planet

    kw = "_planet" * planet
    if kw != "":
        logging.info("PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    logging.info(f"DICO {sub_dico} used ----")

    if self.table["telluric_fwhm"][0] is None:
        fwhm = np.array([3.0] * len(self.table.jdb))
    else:
        try:
            fwhm = np.array(self.table["telluric_fwhm"])
        except:
            fwhm = np.array([3.0] * len(self.table.jdb))

    fwhm_max = [5, 8][self.instrument[:-2] == "CORALIE"]
    fwhm_min = [2, 3][self.instrument[:-2] == "CORALIE"]
    fwhm_default = [3, 5][self.instrument[:-2] == "CORALIE"]
    if np.percentile(fwhm, 95) > fwhm_max:
        logging.warning(
            f"FWHM of tellurics larger than {fwhm_max:.0f} km/s ({np.percentile(fwhm, 95):.1f}), reduced to default value of {fwhm_default:.0f} km/s"
        )
        fwhm = np.array([fwhm_default] * len(self.table.jdb))
    if np.percentile(fwhm, 95) < fwhm_min:
        logging.warning(
            f"FWHM of tellurics smaller than {fwhm_min:.0f} km/s ({np.percentile(fwhm, 95):.1f}), increased to default value of {fwhm_default:.0f} km/s"
        )
        fwhm = np.array([fwhm_default] * len(self.table.jdb))

    logging.info(f"FWHM of tellurics : {np.percentile(fwhm, 95):.1f} km/s")

    all_flux, all_conti = self.import_sts_flux(load=["flux" + kw, sub_dico])
    all_flux_norm = all_flux / all_conti

    mask = np.ones(len(self.table.jdb)).astype("bool")
    if jdb_range[2]:
        mask = (np.array(self.table.jdb) > jdb_range[0]) & (
            np.array(self.table.jdb) < jdb_range[1]
        )
    else:
        mask = (np.array(self.table.jdb) < jdb_range[0]) | (
            np.array(self.table.jdb) > jdb_range[1]
        )

    if sum(mask) < 40:
        logging.warning(
            f"Not enough spectra {['inside', 'outside'][jdb_range[2] == 0]} the specified temporal range"
        )
        mask = np.ones(len(self.table.jdb)).astype("bool")
    else:
        logging.info(
            f"{sum(mask):.0f} spectra {['inside', 'outside'][jdb_range[2] == 0]} the specified temporal range can be used for the median"
        )

    all_flux = all_flux[mask]
    all_conti = all_conti[mask]
    all_flux_norm = all_flux_norm[mask]

    berv = np.array(self.table["berv" + kw])[mask]
    rv_shift = np.array(self.table["rv_shift"])[mask]
    berv = berv - rv_shift

    model = cast(
        materials.Telluric_spectrum, pd.read_pickle(str(self.material_folder / "model_telluric.p"))
    )
    grid = model["wave"]
    spectre = model["flux_norm"]
    telluric = tableXY(grid, spectre)
    telluric.find_min()

    all_min = np.array([telluric.x_min, telluric.y_min, telluric.index_min]).T
    all_min = all_min[1 - all_min[:, 1] > telluric_tresh]
    all_width = np.round(
        all_min[:, 0] * fwhm[:, np.newaxis] / 3e5 / np.median(np.diff(wavelength)),
        0,
    )
    all_width = np.nanpercentile(all_width, 95, axis=0)

    if suppress_telluric:
        borders = np.array([all_min[:, 2] - all_width, all_min[:, 2] + all_width]).T
        telluric_mask = flat_clustering(len(grid), borders) != 0
        all_mask2 = []
        for j in tqdm(berv):
            mask = tableXY(doppler_r(grid, j * 1000)[0], telluric_mask, 0 * grid)
            mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
            all_mask2.append(mask.y != 0)
        all_mask2 = np.array(all_mask2).astype("float")
    else:
        all_mask2 = np.zeros(np.shape(all_flux_norm))

    #        i=-1
    #        for j in tqdm(berv):
    #            i+=1
    #            borders = np.array([all_min[:,2]-all_width[i],all_min[:,2]+all_width[i]]).T
    #            telluric_mask = flat_clustering(len(grid),borders)!=0
    #            mask = tableXY(doppler_r(grid,j*1000)[0],telluric_mask)
    #            mask.interpolate(new_grid=wavelength,method='linear')
    #            all_mask2.append(mask.y!=0)
    #         all_mask2 = np.array(all_mask2).astype('float')

    if suppress_telluric:
        telluric_mask = telluric.y < (1 - telluric_tresh)
        all_mask = []
        for j in tqdm(berv):
            mask = tableXY(doppler_r(grid, j * 1000)[0], telluric_mask, 0 * grid)
            mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
            all_mask.append(mask.y != 0)
        all_mask = np.array(all_mask).astype("float")
    else:
        all_mask = np.zeros(np.shape(all_flux_norm))

    if shift_spectrum:
        rv_star = np.array(self.table["ccf_rv"])
        rv_star[np.isnan(rv_star)] = np.nanmedian(rv_star)
        rv_star -= np.median(rv_star)
        i = -1
        if method == "median":
            for j in tqdm(rv_star):
                i += 1
                mask = tableXY(
                    doppler_r(wavelength, j * 1000)[1],
                    all_flux_norm[i],
                    0 * wavelength,
                )
                mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                all_flux_norm[i] = mask.y.copy()
        else:
            # print(len(rv))
            # print(np.shape(all_flux))
            for j in tqdm(rv_star):
                i += 1
                mask = tableXY(
                    doppler_r(wavelength, j * 1000)[1],
                    all_flux[i],
                    0 * wavelength,
                )
                mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                all_flux[i] = mask.y.copy()
                mask = tableXY(
                    doppler_r(wavelength, j * 1000)[1],
                    all_conti[i],
                    0 * wavelength,
                )
                mask.interpolate(new_grid=wavelength, method="linear", interpolate_x=False)
                all_conti[i] = mask.y.copy()

    # plt.plot(wavelength,np.product(all_mask,axis=0))
    # plt.plot(wavelength,np.product(all_mask2,axis=0))
    logging.info(
        f"Percent always contaminated metric1 : {np.sum(np.product(all_mask, axis=0)) / len(wavelength) * 100:.3f} %"
    )
    logging.info(
        f"Percent always contaminated metric2 : {np.sum(np.product(all_mask2, axis=0)) / len(wavelength) * 100:.3f} %"
    )

    all_mask_nan1 = 1 - all_mask
    all_mask_nan1[all_mask_nan1 == 0] = np.nan

    all_mask_nan2 = 1 - all_mask2
    all_mask_nan2[all_mask_nan2 == 0] = np.nan

    mask_percentile_0 = np.ones(len(wavelength)).astype("bool")
    mask_percentile_1 = 50.0
    print(" ", np.shape(wavelength), np.shape(all_flux_norm))

    med = np.zeros(len(wavelength))
    med[mask_percentile_0] = np.nanpercentile(
        all_flux_norm[:, mask_percentile_0], mask_percentile_1, axis=0
    )
    med[~mask_percentile_0] = np.nanpercentile(all_flux_norm[:, ~mask_percentile_0], 50, axis=0)

    mean = np.nansum(all_flux, axis=0) / np.nansum(all_conti, axis=0)
    mean1 = np.nansum(all_flux * all_mask_nan1, axis=0) / (
        np.nansum(all_conti * all_mask_nan1, axis=0) + epsilon
    )
    mean2 = np.nansum(all_flux * all_mask_nan2, axis=0) / (
        np.nansum(all_conti * all_mask_nan2, axis=0) + epsilon
    )
    mean2[mean2 == 0] = np.nan
    mean1[mean1 == 0] = np.nan
    mean1[mean1 != mean1] = mean2[mean1 != mean1]
    mean1[mean1 != mean1] = med[mean1 != mean1]
    mean2[mean2 != mean2] = mean1[mean2 != mean2]
    # med1[med1!=med1] = mean1[med1!=med1]
    # med2[med2!=med2] = mean1[med2!=med2]
    all_flux_diff_med = all_flux_norm - med
    tresh = 1.5 * IQ(np.ravel(all_flux_diff_med)) + np.nanpercentile(all_flux_diff_med, 75)

    mean1[mean1 > (1 + tresh)] = 1

    if method != "median":
        self.reference = (wavelength, mean1)
    else:
        self.reference = (wavelength, med)

    plt.figure(figsize=(16, 8))
    plt.plot(wavelength, med, color="b", ls="-", label="median")
    plt.plot(wavelength, mean1, color="g", ls="-", label="mean1")
    plt.plot(wavelength, mean2, color="r", ls="-", label="mean2")
    plt.legend(loc=2)
    # plt.plot(wavelength,med,color='b',ls='-.')
    # plt.plot(wavelength,med1,color='g',ls='-.')
    # plt.plot(wavelength,med2,color='r',ls='-.')

    all_flux_diff_mean = all_flux_norm - mean
    all_flux_diff_med = all_flux_norm - med
    all_flux_diff1_mean = all_flux_norm - mean1
    # all_flux_diff1_med = all_flux_norm - med1
    # all_flux_diff2_mean = all_flux_norm - mean2
    # all_flux_diff2_med = all_flux_norm - med2

    idx_min = int(find_nearest(wavelength, wave_min)[0])
    idx_max = int(find_nearest(wavelength, wave_max)[0])

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 2)
    plt.imshow(
        all_flux_diff_mean[::-1, idx_min:idx_max],
        aspect="auto",
        vmin=-0.005,
        vmax=0.005,
        cmap="plasma",
    )
    plt.imshow(all_mask2[::-1, idx_min:idx_max], aspect="auto", alpha=0.2, cmap="Reds")
    ax = plt.gca()
    plt.subplot(2, 1, 1, sharex=ax, sharey=ax)
    plt.imshow(
        all_flux_diff_mean[::-1, idx_min:idx_max],
        aspect="auto",
        vmin=-0.005,
        vmax=0.005,
        cmap="plasma",
    )
    plt.imshow(all_mask[::-1, idx_min:idx_max], aspect="auto", alpha=0.2, cmap="Reds")

    if len(berv) > 15:
        plt.figure(figsize=(16, 8))
        plt.subplot(2, 1, 1)

        plt.title("Median")
        my_colormesh(
            wavelength[idx_min : idx_max + 1],
            np.arange(len(all_mask)),
            all_flux_diff_med[:, idx_min : idx_max + 1],
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )
        ax = plt.gca()

        plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
        plt.title("Masked mean")
        my_colormesh(
            wavelength[idx_min : idx_max + 1],
            np.arange(len(all_mask)),
            all_flux_diff1_mean[:, idx_min : idx_max + 1],
            vmin=-0.005,
            vmax=0.005,
            cmap="plasma",
        )

    load["wave"] = wavelength
    if method != "median":
        load["reference_spectrum"] = mean1
    else:
        load["reference_spectrum"] = med - np.median(all_flux_diff_med)

    ref = np.array(load["reference_spectrum"])
    ref = self.yarara_non_zero_flux(spectrum=ref, min_value=None)
    load["reference_spectrum"] = ref

    load.loc[load["reference_spectrum"] < 0, "reference_spectrum"] = 0

    iofun.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))
