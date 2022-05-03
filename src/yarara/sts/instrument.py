from __future__ import annotations

import glob as glob
import time
from typing import TYPE_CHECKING, List, Optional

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from colorama import Fore
from numpy import ndarray
from tqdm import tqdm

from .. import io
from ..analysis import table, tableXY
from ..paths import root
from ..plots import my_colormesh, plot_color_box
from ..stats import (
    clustering,
    find_nearest,
    flat_clustering,
    match_nearest,
    merge_borders,
    smooth2d,
)
from ..util import doppler_r, flux_norm_std, print_box, sphinx

if TYPE_CHECKING:
    from . import spec_time_series


# =============================================================================
# INTERFERENCE CORRECTION
# =============================================================================


def yarara_correct_pattern(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    continuum: str = "linear",
    wave_min: int = 6000,
    wave_max: int = 6100,
    reference: str = "median",
    width_range: List[float] = [0.1, 20],
    correct_blue: bool = True,
    correct_red: bool = True,
    jdb_range: Optional[List[int]] = None,
) -> None:

    """
    Suppress interferency pattern produced by a material of a certain width by making a Fourier filtering.
    Width_min is the minimum width possible for the material in mm.

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    wave_min : Minimum x axis limit for the plot
    wave_max : Maximum x axis limit for the plot
    zoom : int-type, to improve the resolution of the 2D plot
    smooth_map = int-type, smooth the 2D plot by gaussian 2D convolution
    reference : 'median', 'snr' or 'master' to select the reference normalised spectrum usedin the difference
    cmap : cmap of the 2D plot
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
    load = self.material

    print_box("\n---- RECIPE : CORRECTION FRINGING ----\n")

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    if sub_dico is None:
        sub_dico = self.dico_actif
    print("\n---- DICO %s used ----" % (sub_dico))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    flux = []
    conti = []
    snr = []
    jdb = []
    hl = []
    hr = []

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        if not i:
            wave = file["wave"]
        snr.append(file["parameters"]["SNR_5500"])
        try:
            jdb.append(file["parameters"]["jdb"])
        except:
            jdb.append(i)
        try:
            hl.append(file["parameters"]["hole_left"])
        except:
            hl.append(None)
        try:
            hr.append(file["parameters"]["hole_right"])
        except:
            hr.append(None)

        flux.append(file["flux" + kw] / file[sub_dico]["continuum_" + continuum])
        conti.append(file[sub_dico]["continuum_" + continuum])

    step = file[sub_dico]["parameters"]["step"]

    wave = np.array(wave)
    flux = np.array(flux)
    conti = np.array(conti)
    snr = np.array(snr)
    jdb = np.array(jdb)

    idx_min = int(find_nearest(wave, wave_min)[0])
    idx_max = int(find_nearest(wave, wave_max)[0])

    mask = np.zeros(len(snr)).astype("bool")
    if jdb_range is not None:
        mask = (np.array(self.table.jdb) > jdb_range[0]) & (
            np.array(self.table.jdb) < jdb_range[1]
        )

    if reference == "median":
        if sum(~mask) < 50:
            print("Not enough spectra out of the temporal specified range")
            mask = np.zeros(len(snr)).astype("bool")
        else:
            print(
                "%.0f spectra out of the specified temporal range can be used for the median"
                % (sum(~mask))
            )

    if reference == "snr":
        ref = flux[snr.argmax()]
    elif reference == "median":
        print("[INFO] Reference spectrum : median")
        ref = np.median(flux[~mask], axis=0)
    elif reference == "master":
        print("[INFO] Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif type(reference) == int:
        print("[INFO] Reference spectrum : spectrum %.0f" % (reference))
        ref = flux[reference]
    else:
        ref = 0 * np.median(flux, axis=0)

    print("[INFO] Pattern analysis for range mm : ", width_range)
    # low = np.percentile(flux-ref,2.5)
    # high = np.percentile(flux-ref,97.5)
    old_diff = smooth2d(flux - ref, smooth_map)
    low = np.percentile(flux / (ref + epsilon), 2.5)
    high = np.percentile(flux / (ref + epsilon), 97.5)

    diff = smooth2d(flux / (ref + epsilon), smooth_map) - 1  # changed for a ratio 21-01-20
    diff[diff == -1] = 0
    diff_backup = diff.copy()

    if jdb_range is None:
        fig = plt.figure(figsize=(18, 6))

        plt.axes([0.06, 0.28, 0.7, 0.65])

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

        plt.axes([0.82, 0.28, 0.07, 0.65], sharey=ax)
        plt.plot(snr, np.arange(len(snr)), "k-")
        plt.tick_params(direction="in", top=True, right=True, labelleft=False)
        plt.xlabel("SNR", fontsize=14)

        plt.axes([0.90, 0.28, 0.07, 0.65], sharey=ax)
        plt.plot(jdb, np.arange(len(snr)), "k-")
        plt.tick_params(direction="in", top=True, right=True, labelleft=False)
        plt.xlabel("jdb", fontsize=14)

        plt.axes([0.06, 0.08, 0.7, 0.2], sharex=ax)
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
            print("[INFO] No spectrum contaminated by interference pattern")

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

        if np.float(index_sphinx2) >= 0:
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

        diff_fourrier = np.abs(fft_pattern) - np.abs(fft_flat)
        diff_fourrier_pos = diff_fourrier[0 : int(len(diff_fourrier) / 2) + 1]

        width = (
            1e-7
            * abs(np.fft.fftfreq(len(diff_fourrier)))[0 : int(len(diff_fourrier) / 2) + 1]
            / dl
        )  # transformation of the frequency in mm of the material

        maximum = np.argmax(diff_fourrier_pos[(width > width_range[0]) & (width < width_range[1])])
        freq_maximum_ref = width[(width > width_range[0]) & (width < width_range[1])][maximum]

        plt.figure()
        plt.plot(width, diff_fourrier_pos, color="k")
        print(
            "\n[INFO] The interference pattern is produced by material with a width of %.3f mm"
            % (freq_maximum_ref)
        )

        new_diff = diff.copy()
        hard_window = 50  # window extraction of the fourier power excess

        index_corrected_pattern_red = []
        index_corrected_pattern_blue = []
        timer_red = -1
        timer_blue = -1

        for j in range(len(diff)):

            diff_pattern = tableXY(2 / wave[::-1], diff[j][::-1], 0 * wave)
            diff_pattern.interpolate(new_grid=new_grid, interpolate_x=False)
            diff_pattern_backup = diff_pattern.y.copy()
            # diff_pattern.rolling(window=1000)  #HARDCODE PARAMETER rolling filter to remove telluric power in the fourrier space
            # diff_pattern.y[abs(diff_pattern.y)>3*diff_pattern.roll_IQ] = np.median(diff_pattern.y)
            emergency = 1

            if (hl[j] == -99.9) | (hr[j] == -99.9):
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

                if hr[j] is not None:
                    if (hr[j] - wave[int(highest[1])]) > 10:
                        print(
                            "The border right of the gap is not the same than the one of the header"
                        )
                        emergency = 0
                if hl[j] is not None:
                    if (hl[j] - wave[int(highest[0])]) > 10:
                        print(
                            "The border left of the gap is not the same than the one of the header"
                        )
                        emergency = 0

            if (highest[2] > 1000) & (emergency):  # because gap between ccd is large
                left = find_nearest(diff_pattern.x, left)[0]
                right = find_nearest(diff_pattern.x, right)[0]

                left, right = (
                    right[0],
                    left[0],
                )  # because xaxis is reversed in 1/lambda space

                fft_pattern_left = np.fft.fft(diff_pattern.y[0:left])
                fft_flat_left = np.fft.fft(diff_flat.y[0:left])
                diff_fourrier_left = np.abs(fft_pattern_left) - np.abs(fft_flat_left)

                fft_pattern_right = np.fft.fft(diff_pattern.y[right:])
                fft_flat_right = np.fft.fft(diff_flat.y[right:])
                diff_fourrier_right = np.abs(fft_pattern_right) - np.abs(fft_flat_right)

                width_right = (
                    1e-7
                    * abs(np.fft.fftfreq(len(diff_fourrier_right)))[
                        0 : int(len(diff_fourrier_right) / 2) + 1
                    ]
                    / dl
                )  # transformation of the frequency in mm of the material
                width_left = (
                    1e-7
                    * abs(np.fft.fftfreq(len(diff_fourrier_left)))[
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
                    print(
                        "[INFO] Correcting night %.0f (r). Interference produced a width of %.3f mm"
                        % (j, freq_maximum_left)
                    )
                    timer_red += 1
                    index_corrected_pattern_red.append(timer_red)

                    # left
                    smooth = tableXY(
                        np.arange(len(diff_fourrier_pos_left)),
                        np.ravel(
                            pd.DataFrame(diff_fourrier_pos_left)
                            .rolling(hard_window, min_periods=1, center=True)
                            .std()
                        ),
                    )
                    smooth.find_max(vicinity=int(hard_window / 2))
                    maxi = find_nearest(smooth.x_max, maximum_left + offset_left)[1]

                    smooth.diff(replace=False)
                    smooth.deri.rm_outliers(m=5, kind="sigma")

                    loc_slope = np.arange(len(smooth.deri.mask))[~smooth.deri.mask]
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
                    print(
                        "[INFO] Correcting night %.0f (b). Interference produced a width of %.3f mm"
                        % (j, freq_maximum_right)
                    )
                    # right
                    timer_blue += 1
                    index_corrected_pattern_blue.append(timer_blue)

                    smooth = tableXY(
                        np.arange(len(diff_fourrier_pos_right)),
                        np.ravel(
                            pd.DataFrame(diff_fourrier_pos_right)
                            .rolling(hard_window, min_periods=1, center=True)
                            .std()
                        ),
                    )
                    smooth.find_max(vicinity=int(hard_window / 2))
                    maxi = find_nearest(smooth.x_max, maximum_right + offset_right)[1]
                    smooth.diff(replace=False)
                    smooth.deri.rm_outliers(
                        m=5, kind="sigma"
                    )  # find peak in fourier space and width from derivative

                    loc_slope = np.arange(len(smooth.deri.mask))[~smooth.deri.mask]
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
                diff_fourrier = np.abs(fft_pattern) - np.abs(fft_flat)
                diff_fourrier_pos = diff_fourrier[0 : int(len(diff_fourrier) / 2) + 1]
                width = (
                    1e-7
                    * abs(np.fft.fftfreq(len(diff_fourrier)))[0 : int(len(diff_fourrier) / 2) + 1]
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

                    smooth = tableXY(
                        np.arange(len(diff_fourrier_pos)),
                        np.ravel(
                            pd.DataFrame(diff_fourrier_pos)
                            .rolling(100, min_periods=1, center=True)
                            .std()
                        ),
                    )
                    smooth.find_max(vicinity=30)
                    maxi = find_nearest(smooth.x_max, maximum + offset)[1]

                    smooth.diff(replace=False)
                    smooth.deri.rm_outliers(m=5, kind="sigma")

                    loc_slope = np.arange(len(smooth.deri.mask))[~smooth.deri.mask]
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
        plt.axes([0.05, 0.66, 0.90, 0.25])
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

        plt.axes([0.05, 0.375, 0.90, 0.25], sharex=ax, sharey=ax)
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

        plt.axes([0.05, 0.09, 0.90, 0.25], sharex=ax, sharey=ax)
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
            spec = self.import_spectrum()
            sub_dico = spec[sub_dico]["parameters"]["sub_dico_used"]
            step -= 1
            pre_map = pd.read_pickle(self.dir_root + "CORRECTION_MAP/map_matching_fourier.p")[
                "correction_map"
            ]

        correction_pattern = old_diff - diff2_backup
        to_be_saved = {"wave": wave, "correction_map": correction_pattern + pre_map}
        io.pickle_dump(
            to_be_saved,
            open(self.dir_root + "CORRECTION_MAP/map_matching_fourier.p", "wb"),
        )

        print("\nComputation of the new continua, wait ... \n")
        time.sleep(0.5)

        i = -1
        for j in tqdm(files):
            i += 1
            file = pd.read_pickle(j)
            output = {"continuum_" + continuum: new_continuum[i]}
            file["matching_fourier"] = output
            file["matching_fourier"]["parameters"] = {
                "pattern_width": freq_maximum_ref,
                "width_cutoff": width_range,
                "index_corrected_red": np.array(index_corrected_pattern_red),
                "index_corrected_blue": np.array(index_corrected_pattern_blue),
                "reference_pattern": index_sphinx,
                "reference_flat": index_sphinx2,
                "reference_spectrum": reference,
                "sub_dico_used": sub_dico,
                "step": step + 1,
            }
            io.save_pickle(j, file)

        self.dico_actif = "matching_fourier"

        plt.show(block=False)

        self.fft_output = np.array([diff, new_diff, conti, new_continuum])


# =============================================================================
# CORRECTION OF FROG (GHOST AND STITCHING)
# =============================================================================


def yarara_produce_mask_contam(
    self: spec_time_series, frog_file: str = root + "/Python/Material/Contam_HARPN.p"
) -> None:
    """
    Creation of the stitching mask on the spectrum

    Parameters
    ----------
    frog_file : files containing the wavelength of the stitching
    """

    directory = self.directory
    kw = "_planet" * self.planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    print_box("\n---- RECIPE : PRODUCTION CONTAM MASK ----\n")

    print("\n [INFO] FROG file used : %s" % (frog_file))
    self.import_table()
    self.import_material()
    load = self.material

    grid = np.array(load["wave"])

    # extract frog table
    frog_table = pd.read_pickle(frog_file)
    # stitching

    print("\n [INFO] Producing the contam mask...")

    wave_contam = np.hstack(frog_table["wave"])
    contam = np.hstack(frog_table["contam"])

    vec = tableXY(wave_contam, contam)
    vec.order()
    vec.interpolate(new_grid=np.array(load["wave"]), method="linear")

    load["contam"] = vec.y.astype("int")
    io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))


def yarara_produce_mask_frog(
    self: spec_time_series, frog_file: str = root + "/Python/Material/Ghost_HARPS03.p"
) -> None:
    """
    Correction of the stitching/ghost on the spectrum by PCA fitting

    Parameters
    ----------
    sub_dico : The sub_dictionnary used to  select the continuum
    continuum : The continuum to select (either linear or cubic)
    reference : 'median', 'snr' or 'master' to select the reference normalised spectrum used in the difference
    extended : extension of the cluster size
    frog_file : files containing the wavelength of the stitching
    """

    print_box("\n---- RECIPE : MASK GHOST/STITCHING/THAR WITH FROG ----\n")

    directory = self.directory
    kw = "_planet" * self.planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    self.import_table()
    self.import_material()
    load = self.material

    file_test = self.import_spectrum()
    grid = file_test["wave"]

    berv_max = self.table["berv" + kw].max()
    berv_min = self.table["berv" + kw].min()
    imin = find_nearest(grid, doppler_r(grid[0], np.max(abs(self.table.berv)) * 1000)[0])[0][0] + 1
    imax = find_nearest(grid, doppler_r(grid[-1], np.max(abs(self.table.berv)) * 1000)[1])[0][0]

    # extract frog table
    frog_table = pd.read_pickle(frog_file)
    berv_file = self.yarara_get_berv_value(frog_table["jdb"])

    # ghost
    for correction in ["stitching", "ghost_a", "ghost_b", "thar"]:
        if correction in frog_table.keys():
            if correction == "stitching":
                print("\n [INFO] Producing the stitching mask...")

                wave_stitching = np.hstack(frog_table["wave"])
                gap_stitching = np.hstack(frog_table["stitching"])

                vec = tableXY(wave_stitching, gap_stitching)
                vec.order()
                stitching = vec.x[vec.y != 0]

                stitching_b0 = doppler_r(stitching, 0 * berv_file * 1000)[0]
                # all_stitch = doppler_r(stitching_b0, berv*1000)[0]

                match_stitching = match_nearest(grid, stitching_b0)
                indext = match_stitching[:, 0].astype("int")

                wavet_delta = np.zeros(len(grid))
                wavet_delta[indext] = 1

                wavet = grid[indext]
                max_t = wavet * ((1 + 1.55e-8) * (1 + (berv_max - 0 * berv_file) / 299792.458))
                min_t = wavet * ((1 + 1.55e-8) * (1 + (berv_min - 0 * berv_file) / 299792.458))

                mask_stitching = np.sum(
                    (grid > min_t[:, np.newaxis]) & (grid < max_t[:, np.newaxis]),
                    axis=0,
                ).astype("bool")
                self.stitching_zones = mask_stitching

                mask_stitching[0:imin] = 0
                mask_stitching[imax:] = 0

                load["stitching"] = mask_stitching.astype("int")
                load["stitching_delta"] = wavet_delta.astype("int")
            else:
                if correction == "ghost_a":
                    print("\n [INFO] Producing the ghost mask A...")
                elif correction == "ghost_b":
                    print("\n [INFO] Producing the ghost mask B...")
                elif correction == "thar":
                    print("\n [INFO] Producing the thar mask...")

                contam = frog_table[correction]
                mask = np.zeros(len(grid))
                wave_s2d = []
                order_s2d = []
                for order in np.arange(len(contam)):
                    vec = tableXY(
                        doppler_r(frog_table["wave"][order], 0 * berv_file * 1000)[0],
                        contam[order],
                        0 * contam[order],
                    )
                    vec.order()
                    vec.y[0:2] = 0
                    vec.y[-2:] = 0
                    begin = int(find_nearest(grid, vec.x[0])[0])
                    end = int(find_nearest(grid, vec.x[-1])[0])
                    sub_grid = grid[begin:end]
                    vec.interpolate(new_grid=sub_grid, method="linear", interpolate_x=False)
                    model = np.array(
                        load["reference_spectrum"][begin:end]
                        * load["correction_factor"][begin:end]
                    )
                    model[model == 0] = 1
                    contam_cumu = vec.y / model
                    if sum(contam_cumu != 0) != 0:
                        mask[begin:end] += np.nanmean(contam_cumu[contam_cumu != 0]) * (
                            contam_cumu != 0
                        )
                        order_s2d.append((vec.y != 0) * (1 + order / len(contam) / 20))
                        wave_s2d.append(sub_grid)

                mask[0:imin] = 0
                mask[imax:] = 0
                load[correction] = mask
        else:
            load[correction] = np.zeros(len(grid))

    io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))


def yarara_correct_borders_pxl(
    self: spec_time_series,
    pixels_to_reject: ndarray = [2, 4095],
    min_shift: int = -30,
    max_shift: int = 30,
) -> None:
    """Produce a brute mask to flag lines crossing pixels according to min-max shift

    Parameters
    ----------
    pixels_to_reject : List of pixels
    min_shift : min shist value in km/s
    max_shift : max shist value in km/s
    """

    print_box("\n---- RECIPE : CREATE PIXELS BORDERS MASK ----\n")

    self.import_material()
    load = self.material

    wave = np.array(load["wave"])
    dwave = np.mean(np.diff(wave))
    pxl = self.yarara_get_pixels()
    orders = self.yarara_get_orders()

    pxl *= orders != 0

    pixels_rejected = np.array(pixels_to_reject)

    pxl[pxl == 0] = np.max(pxl) * 2

    dist = np.zeros(len(pxl)).astype("bool")
    for i in np.arange(np.shape(pxl)[1]):
        dist = dist | (np.min(abs(pxl[:, i] - pixels_rejected[:, np.newaxis]), axis=0) == 0)

    # idx1, dust, dist1 = find_nearest(pixels_rejected,pxl[:,0])
    # idx2, dust, dist2 = find_nearest(pixels_rejected,pxl[:,1])

    # dist = (dist1<=1)|(dist2<=1)

    f = np.where(dist == 1)[0]
    plt.figure()
    for i in np.arange(np.shape(pxl)[1]):
        plt.scatter(pxl[f, i], orders[f, i])

    val, cluster = clustering(dist, 0.5, 1)
    val = np.array([np.product(v) for v in val])
    cluster = cluster[val.astype("bool")]

    left = np.round(wave[cluster[:, 0]] * min_shift / 3e5 / dwave, 0).astype("int")
    right = np.round(wave[cluster[:, 1]] * max_shift / 3e5 / dwave, 0).astype("int")
    # length = right-left+1

    # wave_flagged = wave[f]
    # left = doppler_r(wave_flagged,min_shift*1000)[0]
    # right = doppler_r(wave_flagged,max_shift*1000)[0]

    # idx_left = find_nearest(wave,left)[0]
    # idx_right = find_nearest(wave,right)[0]

    idx_left = cluster[:, 0] + left
    idx_right = cluster[:, 1] + right

    flag_region = np.zeros(len(wave)).astype("int")

    for l, r in zip(idx_left, idx_right):
        flag_region[l : r + 1] = 1

    load["borders_pxl"] = flag_region.astype("int")
    io.pickle_dump(load, open(self.directory + "Analyse_material.p", "wb"))


def yarara_correct_frog(
    self: spec_time_series,
    sub_dico: str = "matching_diff",
    continuum: str = "linear",
    correction: str = "stitching",
    berv_shift: str = False,
    wave_min: int = 3800,
    wave_max: int = 3975,
    wave_min_train: int = 3700,
    wave_max_train: int = 6000,
    complete_analysis: bool = False,
    reference: str = "median",
    equal_weight: bool = True,
    nb_pca_comp: int = 10,
    pca_comp_kept: None = None,
    rcorr_min: int = 0,
    treshold_contam: float = 0.5,
    algo_pca: str = "empca",
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

    print_box("\n---- RECIPE : CORRECTION %s WITH FROG ----\n" % (correction.upper()))

    directory = self.directory
    self.import_table()
    self.import_material()
    load = self.material

    cmap = self.cmap
    planet = self.planet
    low_cmap = self.low_cmap * 100
    high_cmap = self.high_cmap * 100

    kw = "_planet" * planet
    if kw != "":
        print("\n---- PLANET ACTIVATED ----")

    epsilon = 1e-12

    if sub_dico is None:
        sub_dico = self.dico_actif
    print("\n---- DICO %s used ----\n" % (sub_dico))

    files = glob.glob(directory + "RASSI*.p")
    files = np.sort(files)

    all_flux = []
    all_flux_std = []
    snr = []
    jdb = []
    conti = []
    berv = []

    for i, j in enumerate(files):
        file = pd.read_pickle(j)
        if not i:
            grid = file["wave"]
            hole_left = file["parameters"]["hole_left"]
            hole_right = file["parameters"]["hole_right"]
        f = file["flux" + kw]
        f_std = file["flux_err"]
        c = file[sub_dico]["continuum_" + continuum]
        c_std = file["continuum_err"]
        f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)
        all_flux.append(f_norm)
        all_flux_std.append(f_norm_std)
        conti.append(c)
        jdb.append(file["parameters"]["jdb"])
        snr.append(file["parameters"]["SNR_5500"])
        if type(berv_shift) != np.ndarray:
            try:
                berv.append(file["parameters"][berv_shift])
            except:
                berv.append(0)
        else:
            berv = berv_shift

    step = file[sub_dico]["parameters"]["step"]

    all_flux = np.array(all_flux)
    all_flux_std = np.array(all_flux_std)
    conti = np.array(conti)
    jdb = np.array(jdb)
    snr = np.array(snr)
    berv = np.array(berv)

    if reference == "snr":
        ref = all_flux[snr.argmax()]
    elif reference == "median":
        print(" [INFO] Reference spectrum : median")
        ref = np.median(all_flux, axis=0)
    elif reference == "master":
        print(" [INFO] Reference spectrum : master")
        ref = np.array(load["reference_spectrum"])
    elif type(reference) == int:
        print(" [INFO] Reference spectrum : spectrum %.0f" % (reference))
        ref = all_flux[reference]
    else:
        ref = 0 * np.median(all_flux, axis=0)

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
            if np.sum(mask > treshold_contam) < 200:
                print(
                    Fore.YELLOW
                    + " [WARNING] Not enough wavelength in the mask, treshold contamination reduced down to %.2f"
                    % (treshold_contam)
                    + Fore.RESET
                )
                treshold_contam *= 0.75

    mask_ghost = np.sum(
        (grid > min_t[:, np.newaxis]) & (grid < max_t[:, np.newaxis]), axis=0
    ).astype("bool")
    mask_ghost_extraction = (
        mask_ghost
        & (mask > treshold_contam)
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

    # compute pca

    if correction == "stitching":
        print(" [INFO] Computation of PCA vectors for stitching correction...")
        diff_ref = diff[:, mask_ghost]
        subflux = diff[:, (mask_ghost) & (np.array(load["ghost_a"]) == 0)]
        subflux_std = all_flux_std[:, (mask_ghost) & (np.array(load["ghost_a"]) == 0)]
        lab = "Stitching"
        name = "stitching"
    elif correction == "ghost_a":
        print(" [INFO] Computation of PCA vectors for ghost correction...")
        diff_ref = diff[:, mask_ghost]
        subflux = diff[:, (np.array(load["stitching"]) == 0) & (mask_ghost_extraction)]
        subflux_std = all_flux_std[:, (np.array(load["stitching"]) == 0) & (mask_ghost_extraction)]
        lab = "Ghost_a"
        name = "ghost_a"
    elif correction == "ghost_b":
        print(" [INFO] Computation of PCA vectors for ghost correction...")
        diff_ref = diff[:, mask_ghost]
        subflux = diff[:, (load["thar"] == 0) & (mask_ghost_extraction)]
        subflux_std = all_flux_std[:, (load["thar"] == 0) & (mask_ghost_extraction)]
        lab = "Ghost_b"
        name = "ghost_b"
    elif correction == "thar":
        print(" [INFO] Computation of PCA vectors for thar correction...")
        diff_ref = diff.copy()
        subflux = diff[:, mask_ghost_extraction]
        subflux_std = all_flux_std[:, mask_ghost_extraction]
        lab = "Thar"
        name = "thar"
    elif correction == "contam":
        print(" [INFO] Computation of PCA vectors for contam correction...")
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
    plt.axhline(y=0.5, color="k")
    phi_max = test2.phi_components[-5:].max()
    phi_min = test2.phi_components[-5:].min()
    plt.axhspan(ymin=phi_min, ymax=phi_max, alpha=0.2, color="k")
    vec_relevant = np.arange(len(test2.phi_components)) * (
        (test2.phi_components > phi_max) | (test2.phi_components < phi_min)
    )
    if pca_comp_kept is None:
        pca_comp_kept = int(np.where(vec_relevant != np.arange(len(vec_relevant)))[0][0])
        pca_comp_kept = np.max([pca_comp_kept, pca_comp_kept2])

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
        bins=np.linspace(0, 1, 100),
        alpha=0.5,
        density=True,
    )
    bins_control, bins, dust = plt.hist(
        rcorr_free,
        bins=np.linspace(0, 1, 100),
        label="free region",
        alpha=0.5,
        density=True,
    )
    plt.yscale("log")
    plt.legend()
    bins = bins[0:-1] + np.diff(bins) * 0.5
    sum_a = np.sum(bins_contam[bins > 0.40])
    sum_b = np.sum(bins_control[bins > 0.40])
    crit = int(sum_a > (2 * sum_b))
    check = ["r", "g"][crit]  # three times more correlation than in the control group
    plt.xlabel(r"|$\mathcal{R}_{pearson}$|", fontsize=14, fontweight="bold", color=check)
    plt.title("Density", color=check)
    plot_color_box(color=check)

    plt.savefig(self.dir_root + "IMAGES/" + name + "_control_check.pdf")
    print(" [INFO] %.0f versus %.0f" % (sum_a, sum_b))

    if crit:
        print(" [INFO] Control check sucessfully performed: %s" % (name))
    else:
        print(
            Fore.YELLOW
            + " [WARNING] Control check failed. Correction may be poorly performed for: %s"
            % (name)
            + Fore.RESET
        )

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
            plt.title("Vec PCA fitted = %0.f" % (nb_vec))
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
        plt.plot(new_wave, mask_ghost_extraction[int(idx_min) : int(idx_max)], color="k")
        if correction == "stitching":
            plt.plot(new_wave, ref[int(idx_min) : int(idx_max)], color="gray")
    else:
        correction = np.zeros((len(grid), len(jdb)))
        collection = table(diff_ref.T)
        base_vec = np.vstack([np.ones(len(diff)), test2.vec[:, 0:pca_comp_kept].T])
        collection.fit_base(base_vec, num_sim=1)
        correction[mask_ghost] = collection.coeff_fitted.dot(base_vec)
        correction = np.transpose(correction)
        correction[:, rcorr < rcorr_min] = 0

        if np.sum(abs(berv)) != 0:
            for j in tqdm(np.arange(len(all_flux))):
                test = tableXY(grid, correction[j], 0 * grid)
                test.x = doppler_r(test.x, berv[j] * 1000)[0]
                test.interpolate(new_grid=grid, method="cubic", replace=True, interpolate_x=False)
                correction[j] = test.y

            index_min_backup = int(find_nearest(grid, doppler_r(grid[0], 30000)[0])[0])
            index_max_backup = int(find_nearest(grid, doppler_r(grid[-1], -30000)[0])[0])
            correction[:, 0 : index_min_backup * 2] = 0
            correction[:, index_max_backup * 2 :] = 0
            index_hole_right = int(
                find_nearest(grid, hole_right + 1)[0]
            )  # correct 1 angstrom band due to stange artefact at the border of the gap
            index_hole_left = int(
                find_nearest(grid, hole_left - 1)[0]
            )  # correct 1 angstrom band due to stange artefact at the border of the gap
            correction[:, index_hole_left : index_hole_right + 1] = 0

        diff_ref2 = diff_backup - correction

        new_conti = conti * (diff_backup + ref) / (diff_ref2 + ref + epsilon)
        new_continuum = new_conti.copy()
        new_continuum[all_flux == 0] = conti[all_flux == 0]
        new_continuum[new_continuum != new_continuum] = conti[
            new_continuum != new_continuum
        ]  # to supress mystic nan appearing
        new_continuum[np.isnan(new_continuum)] = conti[np.isnan(new_continuum)]
        new_continuum[new_continuum == 0] = conti[new_continuum == 0]
        new_continuum = self.uncorrect_hole(new_continuum, conti)

        # plot end

        if (name == "thar") | (name == "stitching"):
            max_var = grid[np.std(correction, axis=0).argsort()[::-1]]
            if name == "thar":
                max_var = max_var[max_var < 4400][0]
            else:
                max_var = max_var[max_var < 6700][0]
            wave_min = find_nearest(grid, max_var - 15)[1]
            wave_max = find_nearest(grid, max_var + 15)[1]

            idx_min = find_nearest(grid, wave_min)[0]
            idx_max = find_nearest(grid, wave_max)[0] + 1

        new_wave = grid[int(idx_min) : int(idx_max)]

        fig = plt.figure(figsize=(21, 9))

        plt.axes([0.05, 0.66, 0.90, 0.25])
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

        plt.axes([0.05, 0.375, 0.90, 0.25], sharex=ax, sharey=ax)
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

        plt.axes([0.05, 0.09, 0.90, 0.25], sharex=ax, sharey=ax)
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
            diff_backup = []
            self.import_dico_tree()
            sub = np.array(
                self.dico_tree.loc[self.dico_tree["dico"] == "matching_ghost_a", "dico_used"]
            )[0]
            for i, j in enumerate(files):
                file = pd.read_pickle(j)
                f = file["flux" + kw]
                f_std = file["flux_err"]
                c = file[sub]["continuum_" + continuum]
                c_std = file["continuum_err"]
                f_norm, f_norm_std = flux_norm_std(f, f_std, c, c_std)
                diff_backup.append(f_norm - ref)
            diff_backup = np.array(diff_backup)

            fig = plt.figure(figsize=(21, 9))

            plt.axes([0.05, 0.66, 0.90, 0.25])
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

            plt.axes([0.05, 0.375, 0.90, 0.25], sharex=ax, sharey=ax)
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

            plt.axes([0.05, 0.09, 0.90, 0.25], sharex=ax, sharey=ax)
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

        to_be_saved = {"wave": grid, "correction_map": correction}
        io.pickle_dump(
            to_be_saved,
            open(self.dir_root + "CORRECTION_MAP/map_matching_" + name + ".p", "wb"),
        )

        print("\nComputation of the new continua, wait ... \n")
        time.sleep(0.5)
        i = -1
        for j in tqdm(files):
            i += 1
            file = pd.read_pickle(j)
            output = {"continuum_" + continuum: new_continuum[i]}
            file["matching_" + name] = output
            file["matching_" + name]["parameters"] = {
                "reference_spectrum": reference,
                "sub_dico_used": sub_dico,
                "equal_weight": equal_weight,
                "pca_comp_kept": pca_comp_kept,
                "step": step + 1,
            }
            io.save_pickle(j, file)

        self.yarara_analyse_summary()

        self.dico_actif = "matching_" + name

        plt.show(block=False)
