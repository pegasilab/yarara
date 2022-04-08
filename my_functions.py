"""
@author: Cretignier Michael 
@university University of Geneva
"""

import datetime
import glob as glob
import json
import math
import multiprocessing as multicpu
import os
import pickle
import sys
import threading
import time
from itertools import combinations, compress, product

import astropy.coordinates as astrocoord
import astropy.time as Time
import astropy.visualization.hist as astrohist
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import psutil
import scipy.special as sse
import scipy.stats as stats
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_moon, get_sun
from astropy.io import fits
from astropy.modeling.models import Voigt1D
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator
from matplotlib.widgets import Button, RadioButtons, Slider
from PyAstronomy import pyasl
from scipy import ndimage, signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.spatial import Delaunay
from scipy.stats import boxcox, norm
from tqdm import tqdm

pickle_protocol_version = 3

# astronomical constant

Mass_sun = 1.99e30
Mass_earth = 5.97e24
Mass_jupiter = 1.89e27

radius_sun = 696343 * 1000
radius_earth = 6352 * 100
G_cst = 6.67e-11
au_m = 149597871 * 1000

cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors_cycle_mpl = prop_cycle.by_key()["color"]

# statistical


def pickle_dump(obj, obj_file, protocol=None):
    if protocol is None:
        protocol = pickle_protocol_version
    pickle.dump(obj, obj_file, protocol=protocol)


def parabole(x, a, b, c):
    return a + b * x + c * x**2


def mad(array, axis=0, sigma_conv=True):
    """"""
    if axis == 0:
        step = abs(array - np.nanmedian(array, axis=axis))
    else:
        step = abs(array - np.nanmedian(array, axis=axis)[:, np.newaxis])
    return np.nanmedian(step, axis=axis) * [1, 1.48][int(sigma_conv)]


all_alias = np.array(
    [
        -2 - 1 / 365.25,
        2 + 1 / 365.25,
        -2,
        2,
        -1 - 1 / 365.25,
        1 + 1 / 365.25,
        -1,
        1,
        -1 / 365.25,
        1 / 365.25,
        -1 / (2 * 365.25),
        1 / (2 * 365.25),
    ]
)


def voigt(x, amp, cen, wid, wid2):
    func = Voigt1D(x_0=cen, amplitude_L=2, fwhm_L=wid2, fwhm_G=wid)(x)
    return 1 + amp * func / func.max()


def combination(items):
    output = sum([list(map(list, combinations(items, i))) for i in range(len(items) + 1)], [])
    return output
 
 
def local_max(spectre, vicinity):
    vec_base = spectre[vicinity:-vicinity]
    maxima = np.ones(len(vec_base))
    for k in range(1, vicinity):
        maxima *= (
            0.5
            * (1 + np.sign(vec_base - spectre[vicinity - k : -vicinity - k]))
            * 0.5
            * (1 + np.sign(vec_base - spectre[vicinity + k : -vicinity + k]))
        )

    index = np.where(maxima == 1)[0] + vicinity
    if len(index) == 0:
        index = np.array([0, len(spectre) - 1])
    flux = spectre[index]
    return np.array([index, flux])


def smooth2d(y, box_pts, borders=True, mode="same"):
    if box_pts > 1:
        box = 1 / (box_pts**2) * np.ones(box_pts) * np.ones(box_pts)[:, np.newaxis]
        y_smooth = signal.convolve2d(y, box, mode=mode)
        if borders:
            y_smooth[0 : int(box_pts / 2 + 1), :] = y_smooth[int(box_pts / 2 + 2), :]
            y_smooth[-int(box_pts / 2 + 1) :, :] = y_smooth[-int(box_pts / 2 + 2), :]
            y_smooth[:, 0 : int(box_pts / 2 + 1)] = y_smooth[:, int(box_pts / 2 + 2)][
                :, np.newaxis
            ]
            y_smooth[:, -int(box_pts / 2 + 1) :] = y_smooth[:, -int(box_pts / 2 + 2)][
                :, np.newaxis
            ]
        else:
            y_smooth[0 : int(box_pts / 2 + 1), :] = y[0 : int(box_pts / 2 + 1), :]
            y_smooth[-int(box_pts / 2 + 1) :, :] = y[-int(box_pts / 2 + 1) :, :]
            y_smooth[:, 0 : int(box_pts / 2 + 1)] = y[:, 0 : int(box_pts / 2 + 1)]
            y_smooth[:, -int(box_pts / 2 + 1) :] = y[:, -int(box_pts / 2 + 1) :]
    else:
        y_smooth = y
    return y_smooth


def smooth(y, box_pts, shape="rectangular"):  # rectangular kernel for the smoothing
    box2_pts = int(2 * box_pts - 1)
    if type(shape) == int:
        y_smooth = np.ravel(
            pd.DataFrame(y).rolling(box_pts, min_periods=1, center=True).quantile(shape / 100)
        )

    elif shape == "savgol":
        if box2_pts >= 5:
            y_smooth = savgol_filter(y, box2_pts, 3)
        else:
            y_smooth = y
    else:
        if shape == "rectangular":
            box = np.ones(box2_pts) / box2_pts
        if shape == "gaussian":
            vec = np.arange(-25, 26)
            box = norm.pdf(vec, scale=(box2_pts - 0.99) / 2.35) / np.sum(
                norm.pdf(vec, scale=(box2_pts - 0.99) / 2.35)
            )
        y_smooth = np.convolve(y, box, mode="same")
        y_smooth[0 : int((len(box) - 1) / 2)] = y[0 : int((len(box) - 1) / 2)]
        y_smooth[-int((len(box) - 1) / 2) :] = y[-int((len(box) - 1) / 2) :]
    return y_smooth


def find_nearest(array, value, dist_abs=True):
    if type(array) != np.ndarray:
        array = np.array(array)
    if type(value) != np.ndarray:
        value = np.array([value])

    array[np.isnan(array)] = 1e16

    idx = np.argmin(np.abs(array - value[:, np.newaxis]), axis=1)
    distance = abs(array[idx] - value)
    if dist_abs == False:
        distance = array[idx] - value
    return idx, array[idx], distance


def IQ(array, axis=None):
    return np.nanpercentile(array, 75, axis=axis) - np.nanpercentile(array, 25, axis=axis)


def sinus(x, amp, period, phase, a, b, c):
    return amp * np.sin(x * 2 * np.pi / period + phase) + a * x**2 + b * x + c


def get_phase(array, period):
    new_array = np.sort((array % period))
    j0 = np.min(new_array) + (period - np.max(new_array))
    diff = np.diff(new_array)
    if np.max(diff) > j0:
        return 0.5 * (new_array[np.argmax(diff)] + new_array[np.argmax(diff) + 1])
    else:
        return 0


def make_sound(sentence, voice="Victoria"):
    if True:
        try:
            os.system('say -v %s "' % (voice) + sentence + '"')
        except:
            print("\7")
    else:
        print("\7")


def my_colormesh(
    x,
    y,
    z,
    cmap="seismic",
    vmin=None,
    vmax=None,
    zoom=1,
    shading="auto",
    return_output=False,
    order=3,
    smooth_box=1,
):

    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    x, y = np.meshgrid(x, y)

    x = np.hstack([x, x[:, -1][:, np.newaxis] + dx])
    x = np.vstack([x, x[-1, :]])

    y = np.hstack([y, y[:, -1][:, np.newaxis]])
    y = np.vstack([y, y[-1, :] + dy])

    z = np.hstack([z, z[:, -1][:, np.newaxis]])
    z = np.vstack([z, z[-1, :]])

    z = smooth2d(z, smooth_box, borders=False)

    Z = ndimage.zoom(z, zoom, order=order)
    X = ndimage.zoom(x, zoom, order=order)
    Y = ndimage.zoom(y, zoom, order=order)

    if return_output:
        return X, Y, Z
    else:
        plt.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)


def clustering(array, tresh, num):
    difference = abs(np.diff(array))
    cluster = difference < tresh
    if len(cluster) > 0:
        indice = np.arange(len(cluster))[cluster]

        j = 0
        border_left = [indice[0]]
        border_right = []
        while j < len(indice) - 1:
            if indice[j] == indice[j + 1] - 1:
                j += 1
            else:
                border_right.append(indice[j])
                border_left.append(indice[j + 1])
                j += 1
        border_right.append(indice[-1])
        border = np.array([border_left, border_right]).T
        border = np.hstack([border, (1 + border[:, 1] - border[:, 0])[:, np.newaxis]])

        kept = []
        for j in range(len(border)):
            if border[j, -1] >= num:
                kept.append(array[border[j, 0] : border[j, 1] + 2])
        return np.array(kept), border
    else:
        print("no cluster found with such treshhold")


def merge_borders(cluster_output):
    matrix1 = cluster_output.copy()
    for j in range(10):  # to converge
        matrix = matrix1.copy()
        c = matrix[0, 1]
        n = 0
        while matrix[n + 1, 0] != cluster_output[-1, 0]:
            if matrix[n + 1, 0] < c:
                matrix[n, 1] = matrix[n + 1, 1]
                matrix = np.delete(matrix, n + 1, axis=0)
            else:
                n += 1
                c = matrix[n, 1]
        matrix[:, -1] = matrix[:, 1] - matrix[:, 0] + 1
        matrix1 = matrix.copy()
    return matrix1


def flat_clustering(length, cluster_output, extended=0, elevation=1):
    vec = np.arange(length)
    if type(elevation) == int:
        elevation = np.ones(len(cluster_output)) * elevation

    larger = (vec >= (cluster_output[:, 0][:, np.newaxis] - extended)).astype("int") * elevation[
        :, np.newaxis
    ]
    smaller = (vec <= (cluster_output[:, 1][:, np.newaxis] + 1 + extended)).astype(
        "int"
    ) * elevation[:, np.newaxis]
    flat = np.sqrt(np.sum(larger * smaller, axis=0))
    return flat


def identify_nearest(array1, array2):
    """identify the closest elements in array2 of array1"""
    array1 = np.sort(array1)
    array2 = np.sort(array2)

    identification = []

    begin = 0
    for value in tqdm(array1):
        begin2 = find_nearest(array2[begin:], value)[0]
        identification.append(begin2 + begin)
        begin = int(begin2)
    return np.ravel(identification)


def my_ruler(mini, maxi, dmini, dmaxi):
    """make a list from mini to maxi with initial step dmini linearly growing to dmaxi"""
    m = (dmaxi - dmini) / (maxi - mini)
    p = dmini - m * mini

    a = [mini]
    b = mini
    while b < maxi:
        b = a[-1] + (p + m * a[-1])
        a.append(b)
    a = np.array(a)
    a[-1] = maxi
    return a


def match_nearest(array1, array2, fast=True, max_dist=None):
    """return a table [idx1,idx2,num1,num2,distance] matching the closest element from two arrays. Remark : algorithm very slow by conception if the arrays are too large."""
    if type(array1) != np.ndarray:
        array1 = np.array(array1)
    if type(array2) != np.ndarray:
        array2 = np.array(array2)
    if not (np.product(~np.isnan(array1)) * np.product(~np.isnan(array2))):
        print(
            "there is a nan value in your list, remove it first to be sure of the algorithme reliability"
        )
    index1 = np.arange(len(array1))[~np.isnan(array1)]
    index2 = np.arange(len(array2))[~np.isnan(array2)]
    array1 = array1[~np.isnan(array1)]
    array2 = array2[~np.isnan(array2)]
    liste1 = np.arange(len(array1))[:, np.newaxis] * np.hstack(
        [np.ones(len(array1))[:, np.newaxis], np.zeros(len(array1))[:, np.newaxis]]
    )
    liste2 = np.arange(len(array2))[:, np.newaxis] * np.hstack(
        [np.ones(len(array2))[:, np.newaxis], np.zeros(len(array2))[:, np.newaxis]]
    )
    liste1 = liste1.astype("int")
    liste2 = liste2.astype("int")

    if fast:
        # ensure that the probability for two close value to be the same is null
        if len(array1) > 1:
            dmin = np.diff(np.sort(array1)).min()
        else:
            dmin = 0
        if len(array2) > 1:
            dmin2 = np.diff(np.sort(array2)).min()
        else:
            dmin2 = 0
        array1_r = array1 + 0.001 * dmin * np.random.randn(len(array1))
        array2_r = array2 + 0.001 * dmin2 * np.random.randn(len(array2))
        # match nearest
        m = abs(array2_r - array1_r[:, np.newaxis])
        arg1 = np.argmin(m, axis=0)
        arg2 = np.argmin(m, axis=1)
        mask = np.arange(len(arg1)) == arg2[arg1]
        liste_idx1 = arg1[mask]
        liste_idx2 = arg2[arg1[mask]]
        array1_k = array1[liste_idx1]
        array2_k = array2[liste_idx2]

        mat = np.hstack(
            [
                liste_idx1[:, np.newaxis],
                liste_idx2[:, np.newaxis],
                array1_k[:, np.newaxis],
                array2_k[:, np.newaxis],
                (array1_k - array2_k)[:, np.newaxis],
            ]
        )

        if max_dist is not None:
            mat = mat[(abs(mat[:, -1]) < max_dist)]

        return mat

    else:
        for num, j in enumerate(array1):
            liste1[num, 1] = int(find_nearest(array2, j)[0])
        for num, j in enumerate(array2):
            liste2[num, 1] = int(find_nearest(array1, j)[0])

        save = liste2[:, 0].copy()
        liste2[:, 0] = liste2[:, 1].copy()
        liste2[:, 1] = save.copy()

        liste1 = np.vstack([liste1, liste2])
        liste = []
        for j in np.unique(liste1, axis=0):
            if np.sum(np.product(liste1 == j.astype(tuple), axis=1)) == 2:
                liste.append(j)
        liste = np.array(liste)
        distance = []
        for j in liste[:, 0]:
            distance.append(find_nearest(array2, array1[j], dist_abs=False)[2])

        liste_idx1 = index1[liste[:, 0]]
        liste_idx2 = index2[liste[:, 1]]

        mat = np.hstack(
            [
                liste_idx1[:, np.newaxis],
                liste_idx2[:, np.newaxis],
                array1[liste[:, 0], np.newaxis],
                array2[liste[:, 1], np.newaxis],
                np.array(distance)[:, np.newaxis],
            ]
        )

        if max_dist is not None:
            mat = mat[(abs(mat[:, -1]) < max_dist)]

        return mat


def map_rnr(array, val_max=None, n=2):
    """val_max must be strictly larger than all number in the array, n smaller than 10"""
    if type(array) != np.ndarray:
        array = np.hstack([array])

    if val_max is not None:
        if sum(array > val_max):
            print("The array cannot be higher than %.s" % (str(val_max)))

        # sort = np.argsort(abs(array))[::-1]
        # array = array[sort]
        array = (array / val_max).astype("str")
        min_len = np.max([len(k.split(".")[-1]) for k in array])
        array = np.array([k.split(".")[-1] for k in array])
        array = np.array([k + "0" * (min_len - len(k)) for k in array])
        if len(array) < n:
            array = np.hstack([array, ["0" * min_len] * (n - len(array))])

        new = ""
        for k in range(min_len):
            for l in range(len(array)):
                new += array[l][k]

        concat = str(n) + str(val_max) + "." + new

        return np.array([concat]).astype("float64")
    else:
        decoded = []
        for i in range(len(array)):
            string = str(array[i])
            code, num = string.split(".")
            n = int(code[0])
            val_max = np.float(code[1:])
            vec = []
            for k in range(n):
                vec.append("0." + num[k::n])
            vec = np.array(vec).astype("float")
            vec *= val_max
            # vec = np.sort(vec)
            decoded.append(vec)
        decoded = np.array(decoded)
        return decoded


#


def flux_norm_std(flux, flux_std, continuum, continuum_std):
    flux_norm = flux / continuum
    flux_norm_std = np.sqrt(
        (flux_std / continuum) ** 2 + (flux * continuum_std / continuum**2) ** 2
    )
    mask = flux_norm_std > flux_norm
    flux_norm_std[mask] = abs(
        flux_norm[mask]
    )  # impossible to get larger error than the point value
    return flux_norm, flux_norm_std


def ccf(wave, spec1, spec2, extended=1500, rv_range=45, oversampling=10, spec1_std=None):
    "CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask"
    dwave = np.median(np.diff(wave))

    if spec1_std is None:
        spec1_std = np.zeros(np.shape(spec1))

    if len(np.shape(spec1)) == 1:
        spec1 = spec1[:, np.newaxis].T
    if len(np.shape(spec1_std)) == 1:
        spec1_std = spec1_std[:, np.newaxis].T
    # spec1 = np.hstack([np.ones(extended),spec1,np.ones(extended)])

    spec1 = np.hstack([np.ones((len(spec1), extended)), spec1, np.ones((len(spec1), extended))])
    spec2 = np.hstack([np.zeros(extended), spec2, np.zeros(extended)])
    spec1_std = np.hstack(
        [
            np.zeros((len(spec1_std), extended)),
            spec1_std,
            np.zeros((len(spec1_std), extended)),
        ]
    )
    wave = np.hstack(
        [
            np.arange(-extended * dwave + wave.min(), wave.min(), dwave),
            wave,
            np.arange(wave.max() + dwave, (extended + 1) * dwave + wave.max(), dwave),
        ]
    )
    shift = np.linspace(0, dwave, oversampling + 1)[:-1]
    shift_save = []
    sum_spec = np.nansum(spec2)
    convolution = []
    convolution_std = []

    rv_max = int(np.log10((rv_range / 299.792e3) + 1) / dwave)
    for j in tqdm(shift):
        new_spec = interp1d(
            wave + j, spec2, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )(wave)
        for k in np.arange(-rv_max, rv_max + 1, 1):
            new_spec2 = np.hstack([new_spec[-k:], new_spec[:-k]])
            convolution.append(np.nansum(new_spec2 * spec1, axis=1) / sum_spec)
            convolution_std.append(
                np.sqrt(np.abs(np.nansum(new_spec2 * spec1_std**2, axis=1))) / sum_spec
            )
            shift_save.append(j + k * dwave)
    shift_save = np.array(shift_save)
    sorting = np.argsort(shift_save)
    return (
        (299.792e6 * 10 ** shift_save[sorting]) - 299.792e6,
        np.array(convolution)[sorting],
        np.array(convolution_std)[sorting],
    )


def gaussian(x, cen, amp, offset, wid):
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2)) + offset


def lorentzian(x, amp, cen, offset, wid):
    return amp * wid**2 / ((x - cen) ** 2 + wid**2) + offset


def rm_outliers(array, m=1.5, kind="sigma", axis=0, return_borders=False):
    if type(array) != np.ndarray:
        array = np.array(array)

    if m != 0:
        array[array == np.inf] = np.nan
        # array[array!=array] = np.nan

        if kind == "inter":
            interquartile = np.nanpercentile(array, 75, axis=axis) - np.nanpercentile(
                array, 25, axis=axis
            )
            inf = np.nanpercentile(array, 25, axis=axis) - m * interquartile
            sup = np.nanpercentile(array, 75, axis=axis) + m * interquartile
            mask = (array >= inf) & (array <= sup)
        if kind == "sigma":
            sup = np.nanmean(array, axis=axis) + m * np.nanstd(array, axis=axis)
            inf = np.nanmean(array, axis=axis) - m * np.nanstd(array, axis=axis)
            mask = abs(array - np.nanmean(array, axis=axis)) <= m * np.nanstd(array, axis=axis)
        if kind == "mad":
            median = np.nanmedian(array, axis=axis)
            mad = np.nanmedian(abs(array - median), axis=axis)
            sup = median + m * mad * 1.48
            inf = median - m * mad * 1.48
            mask = abs(array - median) <= m * mad * 1.48
    else:
        mask = np.ones(len(array)).astype("bool")

    if return_borders:
        return mask, array[mask], sup, inf
    else:
        return mask, array[mask]


def ratio_line(l1, l2, grid, spectrei, continuum, window=3):
    """index  of the  grid element ofline  1,2 plus the  grid the spectrumand the continuum"""

    subgrid1 = grid[l1 - window : l1 + window + 1]
    subgrid2 = grid[l2 - window : l2 + window + 1]
    subspectre1 = spectrei[l1 - window : l1 + window + 1]
    subspectre2 = spectrei[l2 - window : l2 + window + 1]
    subcont1 = continuum[l1 - window : l1 + window + 1]
    subcont2 = continuum[l2 - window : l2 + window + 1]

    coeff = np.polyfit(subgrid1 - np.mean(subgrid1), subspectre1 / subcont1, 2)
    coeff2 = np.polyfit(subgrid2 - np.mean(subgrid2), subspectre2 / subcont2, 2)
    d1 = 1 - np.polyval(coeff, -coeff[1] * 0.5 / coeff[0])
    d2 = 1 - np.polyval(coeff2, -coeff2[1] * 0.5 / coeff2[0])

    mini1 = np.argmin(subspectre1)
    mini2 = np.argmin(subspectre2)

    std_d1 = (
        np.sqrt(subspectre1[mini1] * (1 + (subspectre1[mini1] / subcont1[mini1]) ** 2))
        / subcont1[mini1]
    )
    std_d2 = (
        np.sqrt(subspectre2[mini2] * (1 + (subspectre2[mini2] / subcont2[mini2]) ** 2))
        / subcont2[mini2]
    )

    l3, l4 = np.min([l1, l2]), np.max([l1, l2])
    std_cont = 1 / np.sqrt(np.percentile(spectrei[l3 - 8 * window : l4 + 8 * window], 95))

    return (
        d1,
        d2,
        std_d1,
        std_d2,
        d1 / d2,
        std_cont * np.sqrt(2 + (1.0 / (1 - d1)) ** 2 + (1.0 / (1 - d2)) ** 2),
        np.sqrt((std_d1 / d2) ** 2 + (std_d2 * d1 / d2**2) ** 2),
    )


## Useful tools ##


def auto_axis(vec, axis="y", m=3):
    iq = IQ(vec)
    q1 = np.nanpercentile(vec, 25)
    q3 = np.nanpercentile(vec, 75)
    ax = plt.gca()
    if axis == "y":
        val1 = [ax.get_ylim()[0], q1 - m * iq][q1 - m * iq > ax.get_ylim()[0]]
        val2 = [ax.get_ylim()[1], q3 + m * iq][q3 + m * iq < ax.get_ylim()[1]]
        plt.ylim(val1, val2)
    else:
        val1 = [ax.get_xlim()[0], q1 - m * iq][q1 - m * iq > ax.get_xlim()[0]]
        val2 = [ax.get_xlim()[1], q3 + m * iq][q3 + m * iq < ax.get_xlim()[1]]
        plt.xlim(val1, val2)


## console interactions ##


def print_box(sentence):
    print("\n")
    print("L" * len(sentence))
    print(sentence)
    print("T" * len(sentence))
    print("\n")


def doppler_r(lamb, v):
    """Relativistic Doppler. Take (wavelenght, velocity in [m/s]) and return lambda observed and lambda source"""
    c = 299.792e6
    button = False
    factor = np.sqrt((1 + v / c) / (1 - v / c))
    if type(factor) != np.ndarray:
        button = True
        factor = np.array([factor])
    lambo = lamb * factor[:, np.newaxis]
    lambs = lamb * (factor ** (-1))[:, np.newaxis]
    if button:
        return lambo[0], lambs[0]
    else:
        return lambo, lambs


def plot_color_box(color="r", font="bold", lw=2, ax=None, side="all", ls="-"):
    if ls == "-":
        ls = "solid"

    if ax is None:
        ax = plt.gca()
    if side == "all":
        side = ["top", "bottom", "left", "right"]
    else:
        side = [side]
    for axis in side:
        ax.spines[axis].set_linewidth(lw)
        ax.spines[axis].set_color(color)
        if ax.spines[axis].get_linestyle() != ls:  # to win a but of time
            ax.spines[axis].set_linestyle(ls)

    ax.tick_params(axis="x", which="both", colors=color)
    ax.tick_params(axis="y", which="both", colors=color)

    if font == "bold":
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight("bold")

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight("bold")
