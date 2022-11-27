from __future__ import annotations

import glob as glob
import logging
import time
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from ... import iofun
from ...analysis import tableXY
from ...paths import root
from ...plots import my_colormesh
from ...stats import clustering, find_nearest, smooth, smooth2d
from ...util import doppler_r, flux_norm_std, print_box, sphinx

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_correct_pattern(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    wave_min: float = 6000.0,
    wave_max: float = 6100.0,
    reference: Union[
        int, Literal["snr"], Literal["median"], Literal["master"], Literal["zeros"]
    ] = "median",
    width_range: Tuple[float, float] = (0.1, 20.0),
    correct_blue: bool = True,
    correct_red: bool = True,
    jdb_range: Optional[Tuple[int, int]] = None,
) -> None:

    """
    Suppress interferency pattern produced by a material of a certain width by making a Fourier filtering.
    Width_min is the minimum width possible for the material in mm.

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    wave_min : Minimum x axis limit for the plot
    wave_max : Maximum x axis limit for the plot
    reference : 'median', 'snr' or 'master' to select the reference normalised spectrum usedin the difference
    width_min : minimum width in mm above which to search for a peak in Fourier space
    correct_blue : enable correction of the blue detector for HARPS
    correct_red : enable correction of the red detector for HARPS
    """

    directory = self.directory

    zoom = self.zoom
    smooth_map = self.smooth_map
    cmap = self.cmap
    planet = self.planet
    epsilon = 1e-6
    self.import_material()
    self.import_table()
    self.import_info_reduction()

    load = self.material
    wave = np.array(load["wave"])

    jdb = np.array(self.table["jdb"])
    snr = np.array(self.table["snr"])

    logging.info("RECIPE : CORRECTION FRINGING")

    kw = "_planet" * planet
    if kw != "":
        logging.info("PLANET ACTIVATED")

    if sub_dico is None:
        sub_dico = self.dico_actif
    logging.info("DICO %s used ----" % (sub_dico))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    file_ref = self.import_spectrum()
    hl: Optional[float] = file_ref["parameters"]["hole_left"]
    hr: Optional[float] = file_ref["parameters"]["hole_right"]

    all_flux, conti = self.import_sts_flux(load=["flux" + kw, sub_dico])
    flux = all_flux / conti

    step = self.info_reduction[sub_dico]["step"]

    idx_min = int(find_nearest(wave, wave_min)[0])
    idx_max = int(find_nearest(wave, wave_max)[0])

    mask = np.zeros(len(snr)).astype("bool")
    if jdb_range is not None:
        mask = (np.array(self.table.jdb) > jdb_range[0]) & (
            np.array(self.table.jdb) < jdb_range[1]
        )

    if reference == "median":
        if sum(~mask) < 50:
            logging.info("Not enough spectra out of the temporal specified range")
            mask = np.zeros(len(snr)).astype("bool")
        else:
            logging.info(
                f"{sum(~mask):.0f} spectra out of the specified temporal range can be used for the median"
            )

    if isinstance(reference, int):
        logging.info("Reference spectrum : spectrum %.0f" % (reference))
        ref: NDArray[np.float64] = flux[reference]
    elif reference == "snr":
        ref = flux[snr.argmax()]
    elif reference == "median":
        logging.info("Reference spectrum : median")
        ref = np.median(flux[~mask], axis=0)
    elif reference == "master":
        logging.info("Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif reference == "zeros":
        ref = 0.0 * np.median(flux, axis=0)

    logging.info("Pattern analysis for range mm : ", width_range)
    # low = np.percentile(flux-ref,2.5)
    # high = np.percentile(flux-ref,97.5)
    old_diff = smooth2d(flux - ref, smooth_map)
    low: float = np.percentile(flux / (ref + epsilon), 2.5)  # type: ignore
    high: float = np.percentile(flux / (ref + epsilon), 97.5)  # type: ignore

    diff = smooth2d(flux / (ref + epsilon), smooth_map) - 1  # changed for a ratio 21-01-20
    diff[diff == -1] = 0
    diff_backup = diff.copy()

    if jdb_range is None:
        fig = plt.figure(figsize=(18, 6))

        plt.axes((0.06, 0.28, 0.7, 0.65))

        my_colormesh(
            wave[idx_min:idx_max],
            np.arange(len(diff)),
            diff[:, idx_min:idx_max],
            zoom=zoom,
            vmin=low,
            vmax=high,
            cmap=cmap,
        )
        plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
        plt.ylabel("Spectra  indexes (time)", fontsize=14)
        ax = plt.gca()
        cbaxes = fig.add_axes([0.7 + 0.06, 0.28, 0.01, 0.65])
        plt.colorbar(cax=cbaxes)

        plt.axes((0.82, 0.28, 0.07, 0.65), sharey=ax)
        plt.plot(snr, np.arange(len(snr)), "k-")
        plt.tick_params(direction="in", top=True, right=True, labelleft=False)
        plt.xlabel("SNR", fontsize=14)

        plt.axes((0.90, 0.28, 0.07, 0.65), sharey=ax)
        plt.plot(jdb, np.arange(len(snr)), "k-")
        plt.tick_params(direction="in", top=True, right=True, labelleft=False)
        plt.xlabel("jdb", fontsize=14)

        plt.axes((0.06, 0.08, 0.7, 0.2), sharex=ax)
        plt.plot(wave[idx_min:idx_max], flux[snr.argmax()][idx_min:idx_max], color="k")
        plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
        plt.ylabel("Flux normalised", fontsize=14)
        plt.tick_params(direction="in", top=True, right=True)

        plt.show(block=False)
        index_sphinx = int(sphinx("Which index present a clear pattern ?"))
        index_sphinx2 = int(sphinx("Which index present no pattern ?"))
        plt.close()
    else:
        snr = np.array(self.table.snr)

        if sum(mask):
            i1 = np.argmax(snr[mask])
            index_sphinx = np.arange(len(snr))[mask][i1]

            if sum(mask) == len(mask):
                index_sphinx2 = -1
            else:
                i2 = np.argmax(snr[~mask])
                index_sphinx2 = np.arange(len(snr))[~mask][i2]
        else:
            index_sphinx = -1
            index_sphinx2 = -1
            logging.info("No spectrum contaminated by interference pattern")

    print("Index spectrum containing a pattern : %.0f" % (index_sphinx))
    print("Index spectrum not containing a pattern : %.0f" % (index_sphinx2))

    time.sleep(1)

    if index_sphinx >= 0:

        for j in tqdm(range(len(diff))):
            diff_pattern = tableXY(wave, diff[j].copy(), 0 * wave)
            diff_pattern.rolling(
                window=10000, median=False
            )  # HARDCODE PARAMETER rolling filter to remove telluric power in the fourrier space
            diff_pattern.y[abs(diff_pattern.y) > 3 * diff_pattern.roll_IQ] = (
                0
                * np.random.randn(np.sum(abs(diff_pattern.y) > 3 * diff_pattern.roll_IQ))
                * np.median(diff_pattern.roll_IQ)
            )
            diff[j] = diff_pattern.y

        diff_pattern = tableXY(2 / wave[::-1], diff[index_sphinx][::-1])

        if index_sphinx2 >= 0:
            diff_flat = tableXY(2 / wave[::-1], diff[index_sphinx2][::-1], 0 * wave)
        else:
            diff_flat = tableXY(2 / wave[::-1], np.median(diff, axis=0)[::-1], 0 * wave)

        new_grid = np.linspace(diff_pattern.x.min(), diff_pattern.x.max(), len(diff_pattern.x))
        diff_pattern.interpolate(new_grid=new_grid, interpolate_x=False)
        diff_flat.interpolate(new_grid=new_grid, interpolate_x=False)

        # diff_pattern.rolling(window=1000) #HARDCODE PARAMETER rolling filter to remove telluric power in the fourrier space
        # diff_pattern.y[abs(diff_pattern.y)>3*diff_pattern.roll_IQ] = np.median(diff_pattern.y)
        # diff_flat.rolling(window=1000) #rolling filter to remove telluric power in the fourrier space
        # diff_flat.y[abs(diff_flat.y)>3*diff_flat.roll_IQ] = np.median(diff_flat.y)

        dl = np.diff(new_grid)[0]

        fft_pattern = np.fft.fft(diff_pattern.y)
        fft_flat = np.fft.fft(diff_flat.y)

        plt.figure()
        plt.plot(np.abs(fft_pattern))
        plt.plot(np.abs(fft_flat))

        diff_fourrier = np.abs(fft_pattern) - np.abs(fft_flat)  # type: ignore
        diff_fourrier_pos = diff_fourrier[0 : int(len(diff_fourrier) / 2) + 1]

        # TODO: check what this below is doing, code seems broken
        width = (
            1e-7
            * abs(np.fft.fftfreq(len(diff_fourrier), d=1.0))[0 : int(len(diff_fourrier) / 2) + 1]
            / dl
        )  # transformation of the frequency in mm of the material

        maximum = np.argmax(diff_fourrier_pos[(width > width_range[0]) & (width < width_range[1])])
        freq_maximum_ref = width[(width > width_range[0]) & (width < width_range[1])][maximum]

        plt.figure()
        plt.plot(width, diff_fourrier_pos, color="k")
        logging.info(
            "The interference pattern is produced by material with a width of %.3f mm"
            % (freq_maximum_ref)
        )

        new_diff = diff.copy()
        hard_window = 50  # window extraction of the fourier power excess

        index_corrected_pattern_red = []
        index_corrected_pattern_blue = []
        timer_red = -1
        timer_blue = -1

        left: float = 0.0
        right: float = 0.0
        for j in range(len(diff)):

            diff_pattern = tableXY(2 / wave[::-1], diff[j][::-1], 0 * wave)
            diff_pattern.interpolate(new_grid=new_grid, interpolate_x=False)
            diff_pattern_backup = diff_pattern.y.copy()
            # diff_pattern.rolling(window=1000)  #HARDCODE PARAMETER rolling filter to remove telluric power in the fourrier space
            # diff_pattern.y[abs(diff_pattern.y)>3*diff_pattern.roll_IQ] = np.median(diff_pattern.y)
            emergency = 1

            if (hl == -99.9) | (hr == -99.9):
                emergency = 0
                highest = [0, 0, 0]
            else:
                dstep_clust = abs(np.mean(np.diff(diff[j])))
                if dstep_clust == 0:
                    dstep_clust = np.mean(abs(np.mean(np.diff(diff, axis=1), axis=1)))
                mask = clustering(np.cumsum(diff[j]), dstep_clust, 0)[-1]
                highest = mask[mask[:, 2].argmax()]

                left = (
                    2 / wave[int(highest[0] - 1.0 / np.diff(wave)[0])]
                )  # add 1.0 angstrom of security around the gap
                right = (
                    2 / wave[int(highest[1] + 1.0 / np.diff(wave)[0])]
                )  # add 1.0 angstrom of security around the gap

                if hr is not None:
                    if (hr - wave[int(highest[1])]) > 10:
                        print(
                            "The border right of the gap is not the same than the one of the header"
                        )
                        emergency = 0
                if hl is not None:
                    if (hl - wave[int(highest[0])]) > 10:
                        print(
                            "The border left of the gap is not the same than the one of the header"
                        )
                        emergency = 0

            if (highest[2] > 1000) & (emergency):  # because gap between ccd is large
                left = find_nearest(diff_pattern.x, left)[0][0]
                right = find_nearest(diff_pattern.x, right)[0][0]

                left, right = (right, left)  # because xaxis is reversed in 1/lambda space

                fft_pattern_left = np.fft.fft(diff_pattern.y[0:left])
                fft_flat_left = np.fft.fft(diff_flat.y[0:left])
                diff_fourrier_left = np.abs(fft_pattern_left) - np.abs(fft_flat_left)  # type: ignore

                fft_pattern_right = np.fft.fft(diff_pattern.y[right:])
                fft_flat_right = np.fft.fft(diff_flat.y[right:])
                diff_fourrier_right = np.abs(fft_pattern_right) - np.abs(fft_flat_right)  # type: ignore

                width_right = (
                    1e-7
                    * abs(np.fft.fftfreq(len(diff_fourrier_right), d=1.0))[
                        0 : int(len(diff_fourrier_right) / 2) + 1
                    ]
                    / dl
                )  # transformation of the frequency in mm of the material
                width_left = (
                    1e-7
                    * abs(np.fft.fftfreq(len(diff_fourrier_left), d=1.0))[
                        0 : int(len(diff_fourrier_left) / 2) + 1
                    ]
                    / dl
                )  # transformation of the frequency in mm of the material

                diff_fourrier_pos_left = diff_fourrier_left[
                    0 : int(len(diff_fourrier_left) / 2) + 1
                ]
                diff_fourrier_pos_right = diff_fourrier_right[
                    0 : int(len(diff_fourrier_right) / 2) + 1
                ]

                maxima_left = tableXY(
                    width_left[(width_left > width_range[0]) & (width_left < width_range[1])],
                    smooth(
                        diff_fourrier_pos_left[
                            (width_left > width_range[0]) & (width_left < width_range[1])
                        ],
                        3,
                    ),
                )
                maxima_left.find_max(vicinity=int(hard_window / 2))

                maxima_right = tableXY(
                    width_right[(width_right > width_range[0]) & (width_right < width_range[1])],
                    smooth(
                        diff_fourrier_pos_right[
                            (width_right > width_range[0]) & (width_right < width_range[1])
                        ],
                        3,
                    ),
                )
                maxima_right.find_max(vicinity=int(hard_window / 2))

                five_maxima_left = maxima_left.x_max[np.argsort(maxima_left.y_max)[::-1]][0:10]
                five_maxima_right = maxima_right.x_max[np.argsort(maxima_right.y_max)[::-1]][0:10]

                five_maxima_left_y = maxima_left.y_max[np.argsort(maxima_left.y_max)[::-1]][0:10]
                five_maxima_right_y = maxima_right.y_max[np.argsort(maxima_right.y_max)[::-1]][
                    0:10
                ]

                thresh_left = 10 * np.std(diff_fourrier_pos_left)
                thresh_right = 10 * np.std(diff_fourrier_pos_right)

                five_maxima_left = five_maxima_left[five_maxima_left_y > thresh_left]
                five_maxima_right = five_maxima_right[five_maxima_right_y > thresh_right]

                maximum_left: NDArray[np.int64] = None  # type: ignore
                maximum_right: NDArray[np.int64] = None  # type: ignore
                if len(five_maxima_left) > 0:
                    where, freq_maximum_left, dust = find_nearest(
                        five_maxima_left, freq_maximum_ref
                    )
                    maximum_left = maxima_left.index_max[np.argsort(maxima_left.y_max)[::-1]][0:10]
                    maximum_left = maximum_left[five_maxima_left_y > thresh_left][where]
                else:
                    freq_maximum_left = 0
                if len(five_maxima_right) > 0:
                    where, freq_maximum_right, dust = find_nearest(
                        five_maxima_right, freq_maximum_ref
                    )
                    maximum_right = maxima_right.index_max[np.argsort(maxima_right.y_max)[::-1]][
                        0:10
                    ]
                    maximum_right = maximum_right[five_maxima_right_y > thresh_right][where]
                else:
                    freq_maximum_right = 0

                offset_left = np.where(
                    ((width_left > width_range[0]) & (width_left < width_range[1])) == True
                )[0][0]
                offset_right = np.where(
                    ((width_right > width_range[0]) & (width_right < width_range[1])) == True
                )[0][0]

                if ((abs(freq_maximum_left - freq_maximum_ref) / freq_maximum_ref) < 0.05) & (
                    correct_red
                ):
                    logging.info(
                        "Correcting night %.0f (r). Interference produced a width of %.3f mm"
                        % (j, freq_maximum_left)
                    )
                    timer_red += 1
                    index_corrected_pattern_red.append(timer_red)

                    # left
                    smooth_ = tableXY(
                        np.arange(len(diff_fourrier_pos_left)),
                        np.ravel(
                            pd.DataFrame(diff_fourrier_pos_left)
                            .rolling(hard_window, min_periods=1, center=True)
                            .std()
                        ),
                    )
                    smooth_.find_max(vicinity=int(hard_window / 2))
                    maxi = find_nearest(smooth_.x_max, maximum_left + offset_left)[1]

                    smooth_.diff(replace=False)
                    smooth_.deri.rm_outliers(m=5, kind="sigma")

                    loc_slope = np.arange(len(smooth_.deri.mask))[~smooth_.deri.mask]
                    cluster = clustering(loc_slope, hard_window / 2, 0)[
                        0
                    ]  # half size of the rolling window
                    dist = np.ravel([np.mean(k) - maxi for k in cluster])
                    closest = np.sort(np.abs(dist).argsort()[0:2])

                    mini1 = cluster[closest[0]].min()
                    mini2 = cluster[closest[1]].max()

                    fft_pattern_left[mini1:mini2] = fft_flat_left[mini1:mini2]
                    fft_pattern_left[-mini2:-mini1] = fft_flat_left[-mini2:-mini1]
                    diff_pattern.y[0:left] = np.real(np.fft.ifft(fft_pattern_left))
                if ((abs(freq_maximum_right - freq_maximum_ref) / freq_maximum_ref) < 0.05) & (
                    correct_blue
                ):
                    logging.info(
                        f"Correcting night {j:.0f} (b). Interference produced a width of {freq_maximum_right:.3f} mm"
                    )
                    # right
                    timer_blue += 1
                    index_corrected_pattern_blue.append(timer_blue)

                    smooth_ = tableXY(
                        np.arange(len(diff_fourrier_pos_right)),
                        np.ravel(
                            pd.DataFrame(diff_fourrier_pos_right)
                            .rolling(hard_window, min_periods=1, center=True)
                            .std()
                        ),
                    )
                    smooth_.find_max(vicinity=int(hard_window / 2))
                    maxi = find_nearest(smooth_.x_max, maximum_right + offset_right)[1]
                    smooth_.diff(replace=False)
                    smooth_.deri.rm_outliers(
                        m=5, kind="sigma"
                    )  # find peak in fourier space and width from derivative

                    loc_slope = np.arange(len(smooth_.deri.mask))[~smooth_.deri.mask]
                    cluster = clustering(loc_slope, hard_window / 2, 0)[
                        0
                    ]  # half size of the rolling window
                    dist = np.ravel([np.mean(k) - maxi for k in cluster])
                    closest = np.sort(np.abs(dist).argsort()[0:2])

                    mini1 = cluster[closest[0]].min()
                    mini2 = cluster[closest[1]].max()

                    fft_pattern_right[mini1:mini2] = fft_flat_right[mini1:mini2]
                    fft_pattern_right[-mini2:-mini1] = fft_flat_right[-mini2:-mini1]
                    diff_pattern.y[right:] = np.real(np.fft.ifft(fft_pattern_right))

                    # final

                correction = tableXY(
                    diff_pattern.x,
                    diff_pattern_backup - diff_pattern.y,
                    0 * diff_pattern.x,
                )
                correction.interpolate(new_grid=2 / wave[::-1], interpolate_x=False)
                correction.x = 2 / correction.x[::-1]
                correction.y = correction.y[::-1]
                # diff_pattern.x = 2/diff_pattern.x[::-1]
                # diff_pattern.y = diff_pattern.y[::-1]
                # diff_pattern.interpolate(new_grid=wave)
                new_diff[j] = diff_backup[j] - correction.y

            else:
                fft_pattern = np.fft.fft(diff_pattern.y)
                diff_fourrier = np.abs(fft_pattern) - np.abs(fft_flat)  # type: ignore
                diff_fourrier_pos = diff_fourrier[0 : int(len(diff_fourrier) / 2) + 1]
                width = (
                    1e-7
                    * abs(np.fft.fftfreq(len(diff_fourrier), d=1.0))[
                        0 : int(len(diff_fourrier) / 2) + 1
                    ]
                    / dl
                )  # transformation of the frequency in mm of the material

                maxima = tableXY(
                    width[(width > width_range[0]) & (width < width_range[1])],
                    diff_fourrier_pos[(width > width_range[0]) & (width < width_range[1])],
                )
                maxima.find_max(vicinity=50)
                five_maxima = maxima.x_max[np.argsort(maxima.y_max)[::-1]][0:5]
                match = int(np.argmin(abs(five_maxima - freq_maximum_ref)))
                freq_maximum = five_maxima[match]
                maximum = maxima.index_max[np.argsort(maxima.y_max)[::-1]][0:5][match]
                offset = np.where(((width > width_range[0]) & (width < width_range[1])) == True)[
                    0
                ][0]

                if (abs(freq_maximum - freq_maximum_ref) / freq_maximum) < 0.10:
                    print(
                        "[INFO] Correcting night %.0f. The interference pattern is produced by material with a width of %.3f mm"
                        % (j, freq_maximum)
                    )
                    timer_blue += 1
                    index_corrected_pattern_red.append(timer_red)

                    smooth_ = tableXY(
                        np.arange(len(diff_fourrier_pos)),
                        np.ravel(
                            pd.DataFrame(diff_fourrier_pos)
                            .rolling(100, min_periods=1, center=True)
                            .std()
                        ),
                    )
                    smooth_.find_max(vicinity=30)
                    maxi = find_nearest(smooth_.x_max, maximum + offset)[1]

                    smooth_.diff(replace=False)
                    smooth_.deri.rm_outliers(m=5, kind="sigma")

                    loc_slope = np.arange(len(smooth_.deri.mask))[~smooth_.deri.mask]
                    cluster = clustering(loc_slope, 50, 0)[0]  # half size of the rolling window
                    dist = np.ravel([np.mean(k) - maxi for k in cluster])

                    closest = np.sort(np.abs(dist).argsort()[0:2])

                    mini1 = cluster[closest[0]].min()
                    if len(closest) > 1:
                        mini2 = cluster[closest[1]].max()
                    else:
                        mini2 = mini1 + 1

                    fft_pattern[mini1:mini2] = fft_flat[mini1:mini2]
                    fft_pattern[-mini2:-mini1] = fft_flat[-mini2:-mini1]
                    diff_pattern.y = np.real(np.fft.ifft(fft_pattern))
                    diff_pattern.x = 2 / diff_pattern.x[::-1]
                    diff_pattern.y = diff_pattern.y[::-1]
                    diff_pattern.interpolate(new_grid=wave, interpolate_x=False)
                    new_diff[j] = diff_pattern.y

        self.index_corrected_pattern_red = index_corrected_pattern_red
        self.index_corrected_pattern_blue = index_corrected_pattern_blue

        correction = diff_backup - new_diff

        ratio2_backup = new_diff + 1

        new_conti = conti * flux / (ref * ratio2_backup + epsilon)
        new_continuum = new_conti.copy()
        new_continuum[flux == 0] = conti[flux == 0]

        diff2_backup = flux * conti / new_continuum - ref

        new_conti = flux * conti / (diff2_backup + ref + epsilon)

        new_continuum = new_conti.copy()
        new_continuum[flux == 0] = conti[flux == 0]

        low_cmap = self.low_cmap * 100
        high_cmap = self.high_cmap * 100

        fig = plt.figure(figsize=(21, 9))
        plt.axes((0.05, 0.66, 0.90, 0.25))
        my_colormesh(
            wave[idx_min:idx_max],
            np.arange(len(diff)),
            100 * old_diff[:, idx_min:idx_max],
            zoom=zoom,
            vmin=low_cmap,
            vmax=high_cmap,
            cmap=cmap,
        )
        plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
        plt.ylabel("Spectra  indexes (time)", fontsize=14)
        ax = plt.gca()
        cbaxes = fig.add_axes([0.95, 0.66, 0.01, 0.25])
        ax1 = plt.colorbar(cax=cbaxes)
        ax1.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

        plt.axes((0.05, 0.375, 0.90, 0.25), sharex=ax, sharey=ax)
        my_colormesh(
            wave[idx_min:idx_max],
            np.arange(len(new_diff)),
            100 * diff2_backup[:, idx_min:idx_max],
            zoom=zoom,
            vmin=low_cmap,
            vmax=high_cmap,
            cmap=cmap,
        )
        plt.tick_params(direction="in", top=True, right=True, labelbottom=False)
        plt.ylabel("Spectra  indexes (time)", fontsize=14)
        ax = plt.gca()
        cbaxes2 = fig.add_axes([0.95, 0.375, 0.01, 0.25])
        ax2 = plt.colorbar(cax=cbaxes2)
        ax2.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

        plt.axes((0.05, 0.09, 0.90, 0.25), sharex=ax, sharey=ax)
        my_colormesh(
            wave[idx_min:idx_max],
            np.arange(len(new_diff)),
            100 * old_diff[:, idx_min:idx_max] - 100 * diff2_backup[:, idx_min:idx_max],
            vmin=low_cmap,
            vmax=high_cmap,
            zoom=zoom,
            cmap=cmap,
        )
        plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
        plt.ylabel("Spectra  indexes (time)", fontsize=14)
        plt.ylim(0, None)
        ax = plt.gca()
        cbaxes3 = fig.add_axes([0.95, 0.09, 0.01, 0.25])
        ax3 = plt.colorbar(cax=cbaxes3)
        ax3.ax.set_ylabel(r"$\Delta$ flux normalised [%]", fontsize=14)

        plt.savefig(self.dir_root + "IMAGES/Correction_pattern.png")

        pre_map = np.zeros(np.shape(diff2_backup))
        if sub_dico == "matching_fourier":
            sub_dico = self.info_reduction[sub_dico]["sub_dico_used"]
            step -= 1
            pre_map = np.load(self.dir_root + "CORRECTION_MAP/map_matching_fourier.npy")

        correction_pattern = old_diff - diff2_backup
        to_be_saved = {"wave": wave, "correction_map": correction_pattern + pre_map}
        # myf.pickle_dump(to_be_saved,open(self.dir_root+'CORRECTION_MAP/map_matching_fourier.p','wb'))
        np.save(
            self.dir_root + "CORRECTION_MAP/map_matching_fourier.npy",
            to_be_saved["correction_map"].astype("float32"),
        )

        print("\nComputation of the new continua, wait ... \n")
        time.sleep(0.5)

        self.info_reduction["matching_fourier"] = {
            "pattern_width": freq_maximum_ref,
            "width_cutoff": width_range,
            "index_corrected_red": np.array(index_corrected_pattern_red),
            "index_corrected_blue": np.array(index_corrected_pattern_blue),
            "reference_pattern": index_sphinx,
            "reference_flat": index_sphinx2,
            "reference_spectrum": reference,
            "sub_dico_used": sub_dico,
            "step": step + 1,
            "valid": True,
        }
        self.update_info_reduction()

        fname = self.dir_root + "WORKSPACE/CONTINUUM/Continuum_%s.npy" % ("matching_fourier")
        np.save(fname, new_continuum.astype("float32"))

        self.dico_actif = "matching_fourier"

        plt.show(block=False)

        self.fft_output = np.array([diff, new_diff, conti, new_continuum])
    else:
        self.info_reduction["matching_fourier"] = {
            "sub_dico_used": sub_dico,
            "step": step + 1,
            "valid": False,
        }
        self.update_info_reduction()
