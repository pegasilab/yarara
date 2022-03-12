"""
@author: Cretignier Michael 
@university University of Geneva
"""

import numpy as np
import sys, os, psutil
import scipy.stats as stats
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import norm, boxcox
from itertools import compress, product, combinations
from scipy import signal
import scipy.special as sse
from astropy.io import fits
import math
from scipy.spatial import Delaunay
import matplotlib.pylab as plt
from scipy.signal import savgol_filter
import multiprocessing as multicpu
import threading
import astropy.visualization.hist as astrohist
from scipy import ndimage
import pandas as pd
import pickle
from astropy.modeling.models import Voigt1D
import glob as glob
from PyAstronomy import pyasl
from matplotlib.widgets import Slider, Button, RadioButtons
import astropy.time as Time
import datetime
import json
from astropy import units as u
from astropy.coordinates import get_sun, get_moon, SkyCoord, EarthLocation, AltAz
import astropy.coordinates as astrocoord
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import time

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


def current_time():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def days_from_now(array):
    array = np.array(array)
    noww = current_time()

    dt = []
    for i in array:
        if type(i) == str:
            dt.append(
                Time.Time(noww, format="isot").mjd - Time.Time(i, format="isot").mjd
            )
        else:
            dt.append(np.nan)
    dt = np.array(dt)
    return dt


def pickle_dump(obj, obj_file, protocol=None):
    if protocol is None:
        protocol = pickle_protocol_version
    pickle.dump(obj, obj_file, protocol=protocol)


def kernel(x, a, b, cen, amp, offset, wid):
    vec = (1 - (abs(x - cen) / wid) ** a) ** b
    vec[((x - cen) < -wid) | ((x - cen) > wid)] = 0
    vec = vec / np.nansum(vec)
    return amp * vec + offset


def parabole(x, a, b, c):
    return a + b * x + c * x**2


def current_memory():
    process = psutil.Process(os.getpid())
    value = process.memory_info().rss
    order = int(np.log10(value) // 3)
    units = ["b", "kb", "Mb", "Gb"][order]
    val = value / (10**3) ** (order)
    print("\n [INFO] Current memory usage : %.2f %s\n" % (val, units))  # in bytes
    return val / (10**3) ** (3 - order)


def mad(array, axis=0, sigma_conv=True):
    """"""
    if axis == 0:
        step = abs(array - np.nanmedian(array, axis=axis))
    else:
        step = abs(array - np.nanmedian(array, axis=axis)[:, np.newaxis])
    return np.nanmedian(step, axis=axis) * [1, 1.48][int(sigma_conv)]


def fit_expnorm(x, y):
    def fit_func(x, l, s, m):
        return (
            0.5
            * l
            * np.exp(0.5 * l * (2 * m + l * s * s - 2 * x))
            * sse.erfc((m + l * s * s - x) / (np.sqrt(2) * s))
        )  # exponential gaussian

    popt, pcov = curve_fit(fit_func, x, y)
    plt.plot(fit_func(x, popt[0], popt[1], popt[2]))
    plt.scatter(x, y)
    return popt


def transform_cumu(x, limites=30, inv=False):
    if type(limites) == int:
        limites = np.linspace(np.min(x), np.max(x), limites)
    cumu = np.sum(np.sign(0.5 - int(inv)) * (x - limites[:, np.newaxis]) <= 0, axis=1)
    return limites, cumu


def rm_diagonal(array, k=0, put_nan=False):
    diag = np.diag(np.ones(len(array) - abs(k)), k=k)
    if put_nan:
        diag[diag == 1] = np.nan
    mat = np.ones(np.shape(array)) - diag
    return array * mat


def rm_sym_diagonal(array2, k, put_nan=False):
    array = array2.copy()
    for j in range(abs(k) + 1):
        if not j:
            array = rm_diagonal(array, k=j, put_nan=put_nan)
            array = rm_diagonal(array, k=-j, put_nan=put_nan)
        else:
            array = rm_diagonal(array, k=j, put_nan=put_nan)
    return array


def transform_prim(x, y):
    x = np.array(x)
    y = np.array(y)
    y -= np.nanmin(y)
    y /= np.nansum(y)
    x = x[np.argsort(y)]
    y = np.cumsum(np.sort(y))
    y = y[np.argsort(x)]
    x = np.sort(x)
    return x, y


def transform_min_max(y):
    y = np.array(y).astype("float")
    y -= np.nanmin(y)
    y /= np.nanmax(y)
    return y


def transform_logi(y, mu=0.5, sig=0):
    if sig < 0:
        sig = 0
    if sig > 100:
        sig = 100
    return sigmoid(y, mu, 10 + sig)


def homogenenity(array):
    diff = np.gradient(array)


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


def optimal_plot_axis(x, y, percent=10):
    percent /= 100
    int_x = np.max(x) - np.min(x)
    int_y = np.max(y) - np.min(y)

    return (np.min(x) - percent * int_x, np.max(x) + percent * int_x), (
        np.min(y) - percent * int_y,
        np.max(y) + percent * int_y,
    )


def format_number(nb, digit=3):
    nb_log = int(np.round(np.log10(nb) - 0.5, 0))
    nb_digit = digit - nb_log
    if nb_digit < 0:
        nb_digit = 0

    output = [
        "%.0f" % (nb),
        "%.1f" % (nb),
        "%.2f" % (nb),
        "%.3f" % (nb),
        "%.4f" % (nb),
        "%.5f" % (nb),
        "%.6f" % (nb),
    ][nb_digit]

    return output


def touch_dir(path):
    if not os.path.exists(path):
        os.system("mkdir " + path)


def touch_pickle(filename):
    if not os.path.exists(filename):
        pickle_dump({}, open(filename, "wb"))
        return {}
    else:
        return pd.read_pickle(filename)


def calculate_alias(x):
    if type(x) != np.ndarray:
        x = np.array([x])
    return abs(1 / (all_alias + 1 / x[:, np.newaxis]))


def calculate_peaks_that_gives_this_alias(x):
    if type(x) != np.ndarray:
        x = np.array([x])
    return 1 / (-all_alias + 1 / x[:, np.newaxis])


def boxcox_transformation(array, Draw=False):
    if Draw:
        plt.subplot(1, 2, 1)
        plt.hist(array, bins=100)
    transformed_array, lambdaa = boxcox(array)
    if Draw:
        plt.subplot(1, 2, 2)
        plt.hist(transformed_array, bins=100)
    return transformed_array, lambdaa


def polar_err(phi, r, phi_err, r_err, color="k", marker="o"):
    plt.errorbar(
        phi * np.pi / 180, r, yerr=r_err, capsize=0, color=color, marker=marker
    )
    for th, _r in zip([phi], [r]):
        local_theta = np.linspace(-phi_err, phi_err, 72) + th
        local_r = np.ones(72) * _r
        plt.plot(local_theta * np.pi / 180, local_r, color=color, marker="")
    return local_theta


def detect_outliers_horn(array, Draw=False):
    shift = np.min(array)
    if shift < 0:
        iq = np.nanpercentile(array, 75) - np.nanpercentile(array, 25)
        array -= shift - iq / 5  # to be positive
    transformed_array, lambdaa = boxcox(array)
    Q1 = np.nanpercentile(transformed_array, 25)
    Q3 = np.nanpercentile(transformed_array, 75)
    IQ = Q3 - Q1
    mask_outliers = (transformed_array > Q3 + 1.5 * IQ) | (
        transformed_array < Q1 - 1.5 * IQ
    )
    left = np.min(array[~mask_outliers])
    right = np.max(array[~mask_outliers])
    if Draw:
        plt.subplot(2, 1, 1)
        plt.hist(array, bins=100)
        plt.axvline(x=left)
        plt.axvline(x=right)
        plt.subplot(2, 1, 2)
        plt.hist(transformed_array, bins=100)
    return mask_outliers


def multiprocess(nthreads, liste, functions):
    """def functions(liste):
    save = []
    for j in liste:
        save.append([j, j**2])
    return save"""

    if nthreads >= multicpu.cpu_count():
        print(
            "Your number of cpu (%s) is smaller than the number your entered (%s), enter a smaller value please"
            % (multicpu.cpu_count(), nthreads)
        )
    else:
        chunks = np.array_split(liste, nthreads)
        pool = multicpu.Pool(processes=nthreads)
        res = pool.map(functions, chunks)
        return res


def voigt(x, amp, cen, wid, wid2):
    func = Voigt1D(x_0=cen, amplitude_L=2, fwhm_L=wid2, fwhm_G=wid)(x)
    return 1 + amp * func / func.max()


def combination(items):
    output = sum(
        [list(map(list, combinations(items, i))) for i in range(len(items) + 1)], []
    )
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


def maximum2d(spectre, vicinity):
    index = []
    flux = []
    for j in tqdm(range(np.shape(spectre)[0] - vicinity - 1)):
        for i in range(np.shape(spectre)[1] - vicinity - 1):

            maxi = np.product(
                spectre[j + vicinity // 2, i + vicinity // 2]
                >= spectre[j : j + vicinity + 1, i : i + vicinity + 1]
            )
            if maxi:  # local maxima in 5 points vicinity
                index.append([i + vicinity // 2, j + vicinity // 2])
                flux.append(spectre[i + vicinity // 2, j + vicinity // 2])
    index, flux = np.array(index), np.array(flux)
    return index, flux


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


def divergence(G):
    return np.nansum(np.gradient(G), axis=0)


def laplacien(G):
    return np.gradient(np.gradient(G)[1])[1] + np.gradient(np.gradient(G)[0])[0]


def norm_laplacien(laplacien):
    vec = np.ravel(laplacien)
    vec = vec[~np.isnan(vec)]
    vec = rm_outliers(vec, m=3, kind="inter")[1]
    return np.nanmean(abs(vec))


def plot_warmup():
    CCD_warmups = [
        "2014-06-18",
        "2014-10-17",
        "2015-02-03",
        "2015-02-19",
        "2015-02-23",
        "2015-05-19",
        "2015-10-13",
        "2016-03-31",
        "2016-10-26",
        "2017-04-11",
        "2017-11-14",
        "2018-04-23",
        "2018-10-21",
        "2019-03-12",
        "2019-07-19",
        "2019-12-22",
        "2020-05-28",
    ]

    focus_change = ["2014-03-22", "2020-02-19", "2020-06-01"]

    CCD_warmups_jdb = np.array(
        [Time.Time(i, format="iso").jd - 2400000 for i in CCD_warmups]
    )
    focus_jdb = np.array(
        [Time.Time(i, format="iso").jd - 2400000 for i in focus_change]
    )

    for j in CCD_warmups_jdb:
        plt.axvline(x=j, color="b", alpha=0.5)

    for j in focus_jdb:
        plt.axvline(x=j, color="r", alpha=0.5)


def plot_TH_changed(color="b", ls="-", alpha=1, time=None):

    focus_change = [
        "2003-10-30",
        "2007-07-31",
        "2012-08-29",
        "2015-10-07",
        "2018-11-28",
    ]

    focus_jdb = np.array(
        [Time.Time(i, format="iso").jd - 2400000 for i in focus_change]
    )

    if time is None:
        time = np.array([np.min(focus_jdb), np.max(focus_jdb)])
    for j in focus_jdb:
        if (j >= np.min(time)) & (j <= np.max(time)):
            plt.axvline(x=j, color=color, alpha=alpha, ls=ls)


def conv_iso_mjd(time, fmt="isot"):
    time = time.reset_index(drop=True)
    time2 = time.dropna()
    t = np.array([Time.Time(i, format=fmt).jd - 2400000 for i in time2])
    new_time = np.array(time)
    new_time[list(time2.index)] = t
    return new_time


def today():
    today = datetime.datetime.now().isoformat()
    print(today)
    print(Time.Time(today, format="isot").jd - 2400000)


def best_grid_size(length):
    square = math.ceil(np.sqrt(length))
    l2 = math.ceil(length / square)
    return square, l2


def produce_transparent(images, val1=0, val2=0, val3=0, val4=0):
    alpha = np.zeros((np.shape(images)[0], np.shape(images)[1], 4)).astype("int")
    if val1:
        alpha[:, :, 0] = int(val1)
    if val2:
        alpha[:, :, 1] = int(val2)
    if val3:
        alpha[:, :, 2] = int(val3)
    if val4:
        alpha[:, :, 3] = int(val4)
    return alpha


def first_transgression(array, treshold, relation=1):
    nb_comp = 0
    for j in array:
        if j * relation >= treshold * relation:
            nb_comp += 1
        else:
            break
    return nb_comp


def kcluster(array, tresh_dist):
    k_indices = []
    length = len(array)
    indices = np.arange(length)
    while len(indices):
        indice = [np.array([0, 1]), np.array([indices[0]])]
        while len(indice[-1]) != len(indice[-2]):
            new_indice = np.where(np.min(array[indice[-1]], axis=0) < tresh_dist)[0]
            indice.append(new_indice)
        k_indices.append(new_indice)
        indices = indices[~np.in1d(indices, new_indice)]
    return k_indices


def block_matrix(array, r_min=0, offset=0, debug=False):
    matrix = array.copy()
    old_order = np.arange(len(matrix))
    for j in range(len(matrix) - 1 - offset):
        if np.max(np.abs(matrix[j, j + 1 + offset :])) >= r_min:
            order = np.argsort(np.abs(matrix[j, j + 1 + offset :]))[::-1]
            matrix[j + 1 + offset :] = matrix[j + 1 + offset :][order]
            matrix[:, j + 1 + offset :] = matrix[:, j + 1 + offset :][:, order]
            old_order[j + 1 + offset :] = old_order[j + 1 + offset :][order]

    if debug:
        binary_matrix = abs(array[old_order][:, old_order]) > r_min
        cluster_loc = []
        for j in range(1, len(binary_matrix)):
            if not np.sum(binary_matrix[j, 0:j]):
                cluster_loc.append(j)

        cluster_loc = [0] + cluster_loc + [len(binary_matrix)]
    else:
        cluster_loc = None

    return old_order, cluster_loc


def block_matrix2(array, debug=False):
    matrix = array.copy()
    old_order = np.arange(len(matrix))

    matrix = abs(matrix)
    order_row = np.argsort(matrix, axis=1)[:, ::-1]
    order_col = np.argsort(matrix, axis=0)[::-1]

    new_order = [0]
    for j in range(len(matrix) - 1):
        i = 0
        if j % 2:
            while True:
                if order_row[new_order[-1]][i] not in new_order:
                    new_order.append(order_row[new_order[-1]][i])
                    break
                else:
                    i += 1
        else:
            while True:
                if order_col[:, new_order[-1]][i] not in new_order:
                    new_order.append(order_col[:, new_order[-1]][i])
                    break
                else:
                    i += 1

    new_order = np.array(new_order)
    new_order = np.hstack([new_order, np.setdiff1d(old_order, new_order)])

    if debug:
        v1 = new_order[:-1:2]
        v2 = new_order[1::2]

        all_x = np.hstack([[i, i] for i in v1] + [v1[-1]])
        all_y = np.hstack([v1[0]] + [[i, i] for i in v2])

        plt.figure(99)
        for m, n in enumerate([2, 3, 4, len(matrix)]):
            plt.subplot(2, 3, m + 1)
            plt.title("Iteration=%.0f" % (n - 1))
            plt.imshow(matrix)
            plt.plot(all_x[0:n], all_y[0:n], "r.-")

        plt.subplot(2, 3, 5)
        plt.title("Matrix ordered", fontsize=13)
        plt.imshow(matrix[new_order][:, new_order])

        plt.figure(100)
        for n, i in enumerate(new_order):
            if n % 2:
                plt.plot(matrix[:, i] + n)
                if n != (len(new_order) - 1):
                    plt.scatter(new_order[n + 1], matrix[new_order[n], i] + n)
            else:
                plt.plot(matrix[i] + n)
                if n != (len(new_order) - 1):
                    plt.scatter(new_order[n + 1], matrix[i, new_order[n]] + n)

    return new_order, 0


def smooth(y, box_pts, shape="rectangular"):  # rectangular kernel for the smoothing
    box2_pts = int(2 * box_pts - 1)
    if type(shape) == int:
        y_smooth = np.ravel(
            pd.DataFrame(y)
            .rolling(box_pts, min_periods=1, center=True)
            .quantile(shape / 100)
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


def import_dace_keplerian(filename, Mstar=1):
    fin = open(filename, "r")
    lines = fin.readlines()
    parameters = []
    parameters_err = []
    mass = []
    while len(lines[-1].split("KEPLERIAN")) > 1:
        sub = []
        sub_err = []
        for j in range(5):
            sub.append(lines[-j - 1].split("\t")[1])
            sub_err.append(lines[-j - 1].split("\t")[2])
        mass.append(lines[-6].split(" = ")[1][:-2])
        parameters.append(sub)
        parameters_err.append(sub_err)
        lines = lines[:-7]

    mass = np.array(mass).astype("float")
    parameters = np.array(parameters).astype("float")
    parameters_err = np.array(parameters_err).astype("float")
    parameters = parameters[:, ::-1]
    parameters_err = parameters_err[:, ::-1]

    parameters_err = parameters_err.T
    parameters = parameters.T
    periods = parameters[0]

    parameters = np.vstack(
        [parameters, (Mstar * (periods / 365.25) ** 2) ** (1 / 3), mass, 0 * mass]
    )

    dico = {
        k: a
        for k, a in zip(["p", "k", "e", "peri", "long", "a", "mass", "i"], parameters)
    }
    dico_err_inf = {
        k: a
        for k, a in zip(
            ["p_std_inf", "k_std_inf", "e_std_inf", "peri_std_inf", "long_std_inf"],
            parameters_err,
        )
    }
    dico_err_sup = {
        k: a
        for k, a in zip(
            ["p_std_sup", "k_std_sup", "e_std_sup", "peri_std_sup", "long_std_sup"],
            parameters_err,
        )
    }

    dico = pd.DataFrame(dico)
    dico_err_sup = pd.DataFrame(dico_err_sup)
    dico_err_inf = pd.DataFrame(dico_err_inf)

    dico = pd.concat([dico, dico_err_sup, dico_err_inf], axis=1)

    dico = dico.sort_values(by="p")
    dico = dico.reset_index(drop=True)

    return dico


def format_keplerian_dace_table(planets=[[365.25, 0.10, 0, 0, 123, 0]]):
    """[period,semi-amplitude,eccentricity,periastron,ascending node,t0]"""
    matrix = np.array(planets)
    table = pd.DataFrame(
        matrix,
        columns=["p", "k", "e", "peri", "long", "t0"],
        index=["planet %.0f" % (i) for i in np.arange(len(matrix)) + 1],
    )
    return table


def convert_to_dace(star, instrument, drs_version, instrument_mode, path):
    file = glob.glob(
        path
        + "comparison_"
        + star
        + "_rv_old_versus_new_"
        + instrument_mode.lower()
        + "drs02.csv"
    )[0]
    dataframe = pd.read_csv(file)

    if instrument == "HARPS03":
        range_jdb = [0, 57161.5]
    elif instrument == "HARPS15":
        range_jdb = [57161.5, 100000]

    dataframe = dataframe.loc[
        (dataframe["jdb"] < range_jdb[1]) & (dataframe["jdb"] > range_jdb[0])
    ]

    new = {
        "texp": dataframe["exptime"],  # not needed
        "texp_err": dataframe["exptime"] * 0,  # not needed
        "bispan": dataframe["new_bis_span"],
        "bispan_err": dataframe["new_sig_bis_span"],
        "drift_noise": dataframe["new_sig_vrad"],  # not needed
        "drift_noise_err": dataframe["new_sig_vrad"] * 0,  # not needed
        "rjd": dataframe["new_bjd"] - 2400000,
        "rjd_err": dataframe["new_jdb"] * 0,  # not needed
        "cal_therror": dataframe["th_cal_error"],  # not needed
        "cal_therror_err": dataframe["th_cal_error"] * 0,  # not needed
        "fwhm": 1000 * (dataframe["new_fwhm"]),
        "fwhm_err": 1000 * (dataframe["new_sig_fwhm"]),
        "rv": 1000 * (dataframe["new_vrad"]),
        "rv_err": 1000 * (dataframe["new_sig_vrad"]),
        "berv": dataframe["new_berv"],
        "berv_err": dataframe["new_berv"] * 0,  # not needed
        "ccf_noise": dataframe["new_vrad"],  # not needed
        "ccf_noise_err": dataframe["new_sig_vrad"],  # not needed
        "rhk": dataframe["rhk"],  # not needed
        "rhk_err": dataframe["sig_rhk"],  # not needed
        "contrast": dataframe["new_contrast"],
        "contrast_err": dataframe["new_sig_contrast"],
        "cal_thfile": dataframe["new_th_file"],  # not needed
        "spectroFluxSn50": dataframe["new_sn50"],
        "spectroFluxSn50_err": dataframe["new_sn50"] * 0,  # not needed
        "protm08": dataframe["prot_m08"],  # not needed
        "protm08_err": dataframe["sig_prot_m08"],  # not needed
        "caindex": dataframe["s_mw"],  # not needed
        "caindex_err": dataframe["sig_s"],  # not needed
        "pub_reference": [""] * len(dataframe),  # not needed
        "drs_qc": dataframe["new_qc"].astype("bool"),
        "haindex": dataframe["s_mw"],  # not needed
        "haindex_err": dataframe["sig_s"],  # not needed
        "protn84": dataframe["prot_n84"],  # not needed
        "protn84_err": dataframe["sig_prot_n84"],  # not needed
        "naindex": dataframe["s_mw"],  # not needed
        "naindex_err": dataframe["sig_s"],  # not needed
        "snca2": dataframe["sn_CaII"],  # not needed
        "snca2_err": dataframe["sn_CaII"] * 0,  # not needed
        "mask": dataframe["mask"],  # not needed
        "public": [False] * len(dataframe),  # not needed
        "spectroFluxSn20": dataframe["new_sn20"],  # not needed
        "spectroFluxSn20_err": dataframe["new_sn20"] * 0,  # not needed
        "sindex": dataframe["s_mw"],
        "sindex_err": dataframe["sig_s"],
        "drift_used": dataframe["new_drift_ccf"],
        "drift_used_err": dataframe["new_sig_drift_ccf"],  # not needed
        "ccf_asym": dataframe["new_bis_span"],  # not needed
        "ccf_asym_err": dataframe["new_sig_bis_span"],  # not needed
        "date_night": dataframe["night"],
        "raw_file": "r." + dataframe["file_root"] + ".fits",
    }

    return new


def convert_bib_morgane(filename):
    fin = open(filename, "r")
    lines = fin.readlines()
    nb = []
    type_doc = ""
    for n in np.arange(len(lines)):
        l = lines[n]

        if l[0] == "@":
            line1 = l.split("{")
            type_doc = line1[0][1:]
            nb.append(len(line1))
            if len(line1[1].split("_")) > 2:
                name = "_".join(line1[1].split("_")[0:-2])
                new_name = name[0].upper() + name[1:]
                lines[n] = line1[0] + "{" + new_name + "_" + line1[1].split("_")[-1]
        elif l[0:5] == "\tdate":
            lines[n] = "\tyear = {%s},\n" % (l[9:13])
        elif l[0:5] == "\tnote":
            if type_doc == "article":
                lines[n] = "\t\n"
        # elif l[0:4]=='\tdoi':
        #     if type_doc=='article':
        #         lines[n] = '\t\n'
        elif l[0:9] == "\tlocation":
            if type_doc != "article":
                lines[n] = l.replace("\tlocation", "\taddress")

        else:
            lines[n] = l.replace("\tjournaltitle", "\tjournal")

    for k, n in enumerate(lines):
        with open(filename.split(".bib")[0] + "_new.bib", "a") as f:
            f.writelines(n)


def import_dace_mcmc_gael(filename, sigma=1):
    fin = open(filename, "r")
    lines = fin.readlines()
    name_parameters = []
    parameters = []
    parameters_err = []
    parameters_err_sup = []
    parameters_err_inf = []
    mass = []
    string = []

    loc = np.where(np.array([len(i.split("textbf")) for i in lines]) == 2)[0]
    sections = [i.split("\\textbf{")[1].split("}")[0] for i in np.array(lines)[loc]]
    loc = np.array(list(loc) + [len(lines) - 4])

    nb_planet = 0
    all_lines = []
    for begin, end, section in zip(loc[:-1], loc[1:], sections):
        if (
            (section != "Star")
            & (section != "Offset")
            & (section != "Drift")
            & (section != "Noise")
            & (section != "Activity cycle")
        ):
            nb_planet += 1
            string.append(section)
        paragraph = lines[begin + 2 : end - 1]

        l0 = []
        l1 = []
        l2 = []
        l3 = []
        l4 = []

        for l in np.arange(len(paragraph)):
            l0.append(paragraph[l].split(" & ")[0])
            l1.append(paragraph[l].split(" & ")[6])
            l2.append(paragraph[l].split(" & ")[5])
            bracket = paragraph[l].split(" & ")[7 + sigma - 1]
            bracket = bracket[1:-1]
            nb1 = len(bracket.split("--")) - 1
            nb2 = len(bracket.split("-")) - 1
            if (nb1 == 0) & (nb2 == 1):
                l3.append(bracket.split("-")[0])
                l4.append(bracket.split("-")[1])
            if (nb1 == 0) & (nb2 == 2):
                l3.append("-" + bracket.split("-")[1])
                l4.append(bracket.split("-")[2])
            if (nb1 == 1) & (nb2 == 2):
                l3.append(bracket.split("--")[0])
                l4.append("-" + bracket.split("--")[2])
            if (nb1 == 1) & (nb2 == 3):
                l3.append("-" + bracket.split("-")[1])
                l4.append("-" + bracket.split("--")[1])

        name_parameters.append(l0)
        parameters.append(l1)
        parameters_err.append(l2)
        parameters_err_inf.append(l3)
        parameters_err_sup.append(l4)

        all_lines.append([paragraph, l0, l1, l2, l3, l4])

    cols = []
    dico = {}
    dico_err = {}
    dico_err_sup = {}
    dico_err_inf = {}
    dico_kep = pd.DataFrame({})
    dico_kep_err = pd.DataFrame({})
    dico_kep_err_sup = pd.DataFrame({})
    dico_kep_err_inf = pd.DataFrame({})
    pla_name = list(string)  # [1:1+nb_planet]
    pla_itr = 0
    for i, section in enumerate(sections):
        (
            paragraph,
            name_parameters,
            parameters,
            parameters_err,
            parameters_err_inf,
            parameters_err_sup,
        ) = all_lines[i]

        parameters = np.array(parameters).astype("float")
        parameters_err = np.array(parameters_err).astype("float")
        parameters_err_sup = np.array(parameters_err_sup).astype("float")
        parameters_err_inf = np.array(parameters_err_inf).astype("float")

        parameters_err_sup = parameters_err_sup - parameters
        parameters_err_inf = parameters - parameters_err_inf

        if section == "Star":
            name = ["m_star", "plx"]
            dico_temp = {k: [a] for k, a in zip(name, parameters)}
            dico_err_temp = {
                k: [a] for k, a in zip([n + "_std" for n in name], parameters_err)
            }
            dico_err_sup_temp = {
                k: [a]
                for k, a in zip([n + "_std_sup" for n in name], parameters_err_sup)
            }
            dico_err_inf_temp = {
                k: [a]
                for k, a in zip([n + "_std_inf" for n in name], parameters_err_inf)
            }
            cols += name
        elif section == "Offset":
            name = [
                "offset_" + i.split("\gamma_{\mathrm{")[1].split("}")[0]
                for i in name_parameters
            ]
            dico_temp = {k: [a] for k, a in zip(name, parameters)}
            dico_err_temp = {
                k: [a] for k, a in zip([n + "_std" for n in name], parameters_err)
            }
            dico_err_sup_temp = {
                k: [a]
                for k, a in zip([n + "_std_sup" for n in name], parameters_err_sup)
            }
            dico_err_inf_temp = {
                k: [a]
                for k, a in zip([n + "_std_inf" for n in name], parameters_err_inf)
            }
            cols += name
        elif section == "Drift":
            name = ["drift_" + str(i + 1) for i in range(len(parameters))]
            dico_temp = {k: [a] for k, a in zip(name, parameters)}
            dico_err_temp = {
                k: [a] for k, a in zip([n + "_std" for n in name], parameters_err)
            }
            dico_err_sup_temp = {
                k: [a]
                for k, a in zip([n + "_std_sup" for n in name], parameters_err_sup)
            }
            dico_err_inf_temp = {
                k: [a]
                for k, a in zip([n + "_std_inf" for n in name], parameters_err_inf)
            }
            cols += name
        elif section == "Activity cycle":
            name = name_parameters
            dico_temp = {k: [a] for k, a in zip(name, parameters)}
            dico_err_temp = {
                k: [a] for k, a in zip([n + "_std" for n in name], parameters_err)
            }
            dico_err_sup_temp = {
                k: [a]
                for k, a in zip([n + "_std_sup" for n in name], parameters_err_sup)
            }
            dico_err_inf_temp = {
                k: [a]
                for k, a in zip([n + "_std_inf" for n in name], parameters_err_inf)
            }
            cols += name
        elif section == "Noise":
            name = [
                "noise_" + i.split("\sigma_{\mathrm{")[1].split("}")[0]
                for i in name_parameters
            ]
            dico_temp = {k: [a] for k, a in zip(name, parameters)}
            dico_err_temp = {
                k: [a] for k, a in zip([n + "_std" for n in name], parameters_err)
            }
            dico_err_sup_temp = {
                k: [a]
                for k, a in zip([n + "_std_sup" for n in name], parameters_err_sup)
            }
            dico_err_inf_temp = {
                k: [a]
                for k, a in zip([n + "_std_inf" for n in name], parameters_err_inf)
            }
            cols += name
        else:
            if "\log P" in paragraph[0]:
                parameters = np.delete(parameters, [11, 12]).T
                parameters_err = np.delete(parameters_err, [11, 12]).T
                parameters_err_sup = np.delete(parameters_err_sup, [11, 12]).T
                parameters_err_inf = np.delete(parameters_err_inf, [11, 12]).T

                dico_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "logP",
                                "logK",
                                "sqrt(e)cosw",
                                "sqrt(e)sinw",
                                "L0",
                                "a_s",
                                "a",
                                "e",
                                "K",
                                "w",
                                "m_p",
                                "P",
                                "Tc",
                                "Tp",
                            ],
                            np.append(np.array([pla_name[pla_itr]]), parameters),
                        )
                    }
                )
                dico_err_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "logP_std",
                                "logK_std",
                                "sqrt(e)cosw_std",
                                "sqrt(e)sinw_std",
                                "L0_std",
                                "a_s_std",
                                "a_std",
                                "e_std",
                                "K_std",
                                "w_std",
                                "m_p_std",
                                "P_std",
                                "Tc_std",
                                "Tp_std",
                            ],
                            np.append(np.array([pla_name[pla_itr]]), parameters_err),
                        )
                    }
                )
                dico_err_sup_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "logP_std_sup",
                                "logK_std_sup",
                                "sqrt(e)cosw_std_sup",
                                "sqrt(e)sinw_std_sup",
                                "L0_std_sup",
                                "a_s_std_sup",
                                "a_std_sup",
                                "e_std_sup",
                                "K_std_sup",
                                "w_std_sup",
                                "m_p_std_sup",
                                "P_std_sup",
                                "Tc_std_sup",
                                "Tp_std_sup",
                            ],
                            np.append(
                                np.array([pla_name[pla_itr]]), parameters_err_sup
                            ),
                        )
                    }
                )
                dico_err_inf_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "logP_std_inf",
                                "logK_std_inf",
                                "sqrt(e)cosw_std_inf",
                                "sqrt(e)sinw_std_inf",
                                "L0_std_inf",
                                "a_s_std_inf",
                                "a_std_inf",
                                "e_std_inf",
                                "K_std_inf",
                                "w_std_inf",
                                "m_p_std_inf",
                                "P_std_inf",
                                "Tc_std_inf",
                                "Tp_std_inf",
                            ],
                            np.append(
                                np.array([pla_name[pla_itr]]), parameters_err_inf
                            ),
                        )
                    }
                )
                cols_kep = [
                    "planet",
                    "P",
                    "K",
                    "e",
                    "w",
                    "Tp",
                    "a",
                    "m_p",
                    "a_s",
                    "Tc",
                    "logP",
                    "logK",
                    "sqrt(e)cosw",
                    "sqrt(e)sinw",
                    "L0",
                ]
            elif "\lambda_0" in paragraph[0]:
                parameters = np.delete(parameters, [7, 8]).T
                parameters_err = np.delete(parameters_err, [7, 8]).T
                parameters_err_sup = np.delete(parameters_err_sup, [7, 8]).T
                parameters_err_inf = np.delete(parameters_err_inf, [7, 8]).T

                dico_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "L0",
                                "a_s",
                                "a",
                                "e",
                                "K",
                                "w",
                                "m_p",
                                "P",
                                "Tc",
                                "Tp",
                            ],
                            np.append(np.array([pla_name[pla_itr]]), parameters),
                        )
                    }
                )
                dico_err_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "L0_std",
                                "a_s_std",
                                "a_std",
                                "e_std",
                                "K_std",
                                "w_std",
                                "m_p_std",
                                "P_std",
                                "Tc_std",
                                "Tp_std",
                            ],
                            np.append(np.array([pla_name[pla_itr]]), parameters_err),
                        )
                    }
                )
                dico_err_sup_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "L0_std_sup",
                                "a_s_std_sup",
                                "a_std_sup",
                                "e_std_sup",
                                "K_std_sup",
                                "w_std_sup",
                                "m_p_std_sup",
                                "P_std_sup",
                                "Tc_std_sup",
                                "Tp_std_sup",
                            ],
                            np.append(
                                np.array([pla_name[pla_itr]]), parameters_err_sup
                            ),
                        )
                    }
                )
                dico_err_inf_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "L0_std_inf",
                                "a_s_std_inf",
                                "a_std_inf",
                                "e_std_inf",
                                "K_std_inf",
                                "w_std_inf",
                                "m_p_std_inf",
                                "P_std_inf",
                                "Tc_std_inf",
                                "Tp_std_inf",
                            ],
                            np.append(
                                np.array([pla_name[pla_itr]]), parameters_err_inf
                            ),
                        )
                    }
                )
                cols_kep = [
                    "planet",
                    "L0",
                    "P",
                    "K",
                    "e",
                    "w",
                    "Tp",
                    "a",
                    "m_p",
                    "a_s",
                    "Tc",
                ]

            else:
                parameters = np.delete(parameters, [8, 9]).T
                parameters_err = np.delete(parameters_err, [8, 9]).T
                parameters_err_sup = np.delete(parameters_err_sup, [8, 9]).T
                parameters_err_inf = np.delete(parameters_err_inf, [8, 9]).T

                dico_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "P",
                                "K",
                                "e",
                                "w",
                                "L0",
                                "a_s",
                                "a",
                                "m_p",
                                "Tc",
                                "Tp",
                            ],
                            np.append(np.array([pla_name[pla_itr]]), parameters),
                        )
                    }
                )
                dico_err_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "P_std",
                                "K_std",
                                "e_std",
                                "w_std",
                                "L0_std",
                                "a_s_std",
                                "a_std",
                                "m_p_std",
                                "Tc_std",
                                "Tp_std",
                            ],
                            np.append(np.array([pla_name[pla_itr]]), parameters_err),
                        )
                    }
                )
                dico_err_sup_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "P_std_sup",
                                "K_std_sup",
                                "e_std_sup",
                                "w_std_sup",
                                "L0_std_sup",
                                "a_s_std_sup",
                                "a_std_sup",
                                "m_p_std_sup",
                                "Tc_std_sup",
                                "Tp_std_sup",
                            ],
                            np.append(
                                np.array([pla_name[pla_itr]]), parameters_err_sup
                            ),
                        )
                    }
                )
                dico_err_inf_temp = pd.DataFrame(
                    {
                        k: [a]
                        for k, a in zip(
                            [
                                "planet",
                                "P_std_inf",
                                "K_std_inf",
                                "e_std_inf",
                                "w_std_inf",
                                "L0_std_inf",
                                "a_s_std_inf",
                                "a_std_inf",
                                "m_p_std_inf",
                                "Tc_std_inf",
                                "Tp_std_inf",
                            ],
                            np.append(
                                np.array([pla_name[pla_itr]]), parameters_err_inf
                            ),
                        )
                    }
                )
                cols_kep = [
                    "planet",
                    "P",
                    "K",
                    "e",
                    "w",
                    "L0",
                    "a",
                    "a_s",
                    "m_p",
                    "Tc",
                    "Tp",
                ]

        if (
            (section == "Star")
            | (section == "Offset")
            | (section == "Drift")
            | (section == "Noise")
            | (section == "Activity cycle")
        ):
            dico.update(dico_temp)
            dico_err.update(dico_err_temp)
            dico_err_sup.update(dico_err_sup_temp)
            dico_err_inf.update(dico_err_inf_temp)

        else:
            dico_kep = dico_kep.append(dico_temp)
            dico_kep_err = dico_kep_err.append(dico_err_temp)
            dico_kep_err_sup = dico_kep_err_sup.append(dico_err_sup_temp)
            dico_kep_err_inf = dico_kep_err_inf.append(dico_err_inf_temp)

            pla_itr += 1

    # dico = pd.DataFrame(dico)
    # dico_err = pd.DataFrame(dico_err)
    # dico_err_sup = pd.DataFrame(dico_err_sup)
    # dico_err_inf = pd.DataFrame(dico_err_inf)

    # print(dico)
    # idx = get_column_index(dico, cols)
    # dico = dico[dico.columns[idx]]
    # dico_err = dico_err[dico_err.columns[idx]]
    # dico_err_sup = dico_err_sup[dico_err_sup.columns[idx]]
    # dico_err_inf = dico_err_inf[dico_err_inf.columns[idx]]

    # dico = pd.concat([dico, dico_err, dico_err_sup, dico_err_inf],axis=1)
    # idxtot = []
    # for i in range(len(cols)):
    #     idxtot.extend([i,i+len(cols),i+2*len(cols),i+3*len(cols)])
    # dico = dico[dico.columns[idxtot]]
    # dico = dico.reset_index(drop=True)

    # idx = get_column_index(dico_kep, cols_kep)
    # dico_kep = dico_kep[cols_kep]
    # dico_kep_err = dico_kep_err[cols_kep]
    # dico_kep_err_sup = dico_kep_err_sup[cols_kep]
    # dico_kep_err_inf = dico_kep_err_inf[cols_kep]

    dico_kep = pd.concat(
        [dico_kep, dico_kep_err, dico_kep_err_sup, dico_kep_err_inf], axis=1
    )
    idxtot = []
    for i in range(len(cols_kep)):
        idxtot.extend(
            [i, i + len(cols_kep), i + 2 * len(cols_kep), i + 3 * len(cols_kep)]
        )

    dico_kep = dico_kep[dico_kep.columns[idxtot]]
    dico_kep = dico_kep.loc[:, ~dico_kep.columns.duplicated()]
    dico_kep = dico_kep.reset_index(drop=True)
    dico_kep.index = dico_kep["planet"]
    dico_kep = dico_kep.drop(columns=["planet"])
    dico_kep = dico_kep.astype("float")
    dico_kep = dico_kep.sort_values(by="P")
    dico_kep.index = ["planet %.0f" % (i) for i in range(1, 1 + len(dico_kep))]
    dico_kep["i"] = 0

    return dico_kep


def import_dace_mcmc(filename):
    fin = open(filename, "r")
    lines = fin.readlines()
    parameters = []
    parameters_err_sup = []
    parameters_err_inf = []
    mass = []

    loc = np.where([i == "          \\hline\n" for i in lines])[0]
    loc = np.array([0] + list(loc) + [len(lines) - 5])
    star = loc[0:2]
    planets = loc[2:]

    nb_planet = 0
    for begin, end in zip(planets[::2], planets[1::2]):
        nb_planet += 1
        paragraph = lines[begin + 1 : end]
        l1 = []
        l2 = []
        l3 = []
        for l in np.arange(len(paragraph)):
            l1.append(paragraph[l].split(" & ")[6])
            bracket = paragraph[l].split(" & ")[7]
            bracket = bracket[1:-1]
            nb1 = len(bracket.split("--")) - 1
            nb2 = len(bracket.split("-")) - 1
            if (nb1 == 0) & (nb2 == 1):
                l2.append(bracket.split("-")[0])
                l3.append(bracket.split("-")[1])
            if (nb1 == 0) & (nb2 == 2):
                l2.append("-" + bracket.split("-")[1])
                l3.append(bracket.split("-")[2])
            if (nb1 == 1) & (nb2 == 2):
                l2.append(bracket.split("--")[0])
                l3.append("-" + bracket.split("--")[2])
            if (nb1 == 1) & (nb2 == 3):
                l2.append("-" + bracket.split("-")[1])
                l3.append("-" + bracket.split("--")[1])

        parameters.append(l1)
        parameters_err_inf.append(l2)
        parameters_err_sup.append(l3)

    parameters = np.array(parameters).astype("float")
    parameters_err_sup = np.array(parameters_err_sup).astype("float")
    parameters_err_inf = np.array(parameters_err_inf).astype("float")

    parameters_err_sup = parameters_err_sup - parameters
    parameters_err_inf = parameters - parameters_err_inf

    parameters = np.delete(parameters, [5, 8, 9, 11], axis=1).T
    parameters_err_sup = np.delete(parameters_err_sup, [5, 8, 9, 11], axis=1).T
    parameters_err_inf = np.delete(parameters_err_inf, [5, 8, 9, 11], axis=1).T

    parameters = np.vstack([parameters, 0 * parameters[0]])

    dico = {
        k: a
        for k, a in zip(
            ["p", "k", "e", "peri", "long", "a", "mass", "tc", "i"], parameters
        )
    }
    dico_err_sup = {
        k: a
        for k, a in zip(
            [
                "p_std_sup",
                "k_std_sup",
                "e_std_sup",
                "peri_std_sup",
                "long_std_sup",
                "a_std_sup",
                "mass_std_sup",
                "tc_std_sup",
            ],
            parameters_err_sup,
        )
    }
    dico_err_inf = {
        k: a
        for k, a in zip(
            [
                "p_std_inf",
                "k_std_inf",
                "e_std_inf",
                "peri_std_inf",
                "long_std_inf",
                "a_std_inf",
                "mass_std_inf",
                "tc_std_inf",
            ],
            parameters_err_inf,
        )
    }

    dico = pd.DataFrame(dico)
    dico_err_sup = pd.DataFrame(dico_err_sup)
    dico_err_inf = pd.DataFrame(dico_err_inf)

    dico = pd.concat([dico, dico_err_sup, dico_err_inf], axis=1)

    dico = dico.sort_values(by="p")
    dico = dico.reset_index(drop=True)

    return dico


def extract_berv_fits(directory):
    """produce the average or flux weighted average of a parameter in s1d header"""
    all_files = np.sort(glob.glob(directory + "*.fits"))
    save = []
    for j in tqdm(all_files):
        header = fits.getheader(j)  # load the fits header
        spectre = fits.getdata(j).astype("float64")  # the flux of your spectrum
        bolo = np.sum(spectre)
        berv = header["HIERARCH ESO DRS BERV"]
        mjd = header["MJD-OBS"]
        save.append([bolo, berv, mjd])
    save = np.array(save)
    return save


def read_str_angle(array):
    array = np.array(array)
    new_array = []
    for i in array:
        if type(i) == str:
            val = [
                np.float(j) / 60**k
                for k, j in zip(np.arange(len(i.split(" "))), i.split(" "))
            ]
            new_array.append(np.sum(val))
        else:
            new_array.append(np.nan)
    return np.array(new_array)


def average_weight(array, weights, axis=0):
    if axis == 1:
        array = array.T
        weights = weights.T
    new_array = []
    new_weights = []
    for j in range(np.shape(array)[0]):
        if np.sum(weights[:, j]) == 0:
            new_weights.append(0)
            new_array.append(0)
        else:
            new_weights.append(np.sum(weights[:, j]))
            new_array.append(
                np.sum(array[:, j] * weights[:, j]) / np.sum(weights[:, j])
            )
    new_array = np.array(new_array)
    new_weights = np.array(new_weights)
    return new_array, new_weights


def sigmoid2(x, x0, amp, sig, c):
    y = amp / (1 + np.exp(-sig * (x - x0))) + c
    return y


def conv_rhk_prot(log_rhk, bv):
    """From Noyes 1984 and Mamajek 2008"""
    x = 1 - bv
    y = log_rhk + 5
    log_t = int(x > 0) * (1.362 - 0.166 * x + 0.025 * x**2 - 5.323 * x**3) + int(
        x < 0
    ) * (1.362 - 0.14 * x)
    log_p = log_t + 0.324 - 0.4 * y - 0.283 * y**2 - 1.325 * y**3

    prot_n84 = 10**log_p
    sig_prot_n84 = np.log(10) * 0.08 * prot_n84

    prot_m08 = (0.808 - 2.966 * (log_rhk + 4.52)) * 10**log_t
    sig_prot_m08 = 4.4 * bv * 1.7 - 1.7

    if (prot_m08 > 0.0) & (bv >= 0.50):
        age_m08 = 1e-3 * (prot_m08 / 0.407 / (bv - 0.495) ** 0.325) ** (1.0 / 0.566)
        sig_age_m08 = 0.05 * np.log(10) * age_m08
    else:
        age_m08 = 0.0
        sig_age_m08 = 0.0

    return prot_n84, sig_prot_n84, prot_m08, sig_prot_m08, age_m08, sig_age_m08


def plot_weight(y, yerr="none", normed=False, bin_method="scott", alpha=0.6, color="b"):
    if normed:
        alpha1 = 0
    else:
        alpha1 = alpha
    val = astrohist(y, bins=bin_method, color=color, alpha=alpha1, ec="black")
    if type(yerr) != str:
        errors = []
        mean = []
        for j in range(len(val[1]) - 1):
            up = val[1][j + 1]
            low = val[1][j]
            errors.append(
                np.sum(
                    (norm.cdf(up, y, yerr) - norm.cdf(low, y, yerr))
                    * (1 - (norm.cdf(up, y, yerr) - norm.cdf(low, y, yerr)))
                )
            )
            mean.append(np.sum(norm.cdf(up, y, yerr) - norm.cdf(low, y, yerr)))
    else:
        errors = np.sqrt(val[0])
        mean = 0
    normalisation = 1
    if normed:
        val2 = astrohist(
            y, bins=bin_method, color=color, alpha=alpha, ec="black", density=True
        )
        normalisation = np.nanmedian(val[0] / val2[0])
        plt.ylim(
            0, np.max(val[0] / normalisation) + 0.25 * (np.max(val[0] / normalisation))
        )
    errors = np.array(errors)
    mean = np.array(mean)
    plt.errorbar(
        np.diff(val[1]) / 2 + val[1][0:-1],
        val[0] / normalisation,
        yerr=errors / normalisation,
        color="k",
    )
    return (
        np.diff(val[1]) / 2 + val[1][0:-1],
        val[0] / normalisation,
        errors / normalisation,
    )


def transit_proba(p, Rs=1, Ms=1):
    a = (Ms * Mass_sun * G_cst * (p * 24 * 3600) ** 2 / (4 * np.pi**2)) ** (1 / 3)
    # print('Distance Au : %.2f'%(a/au_m))
    proba = Rs * radius_sun / a
    return proba * 100


def transit_circular_dt(p, Rs=1, Ms=1):
    """Period in days, return dt in hours"""
    velocity = (2 * np.pi * G_cst * Mass_sun * Ms / (p * 24 * 3600)) ** (1 / 3)
    dt = 2 * Rs * radius_sun / (velocity)
    return dt / 3600.0


def transit_draw(P, T0, dt=0):
    ax = plt.gca()
    x1 = ax.get_xlim()[0]
    x2 = ax.get_xlim()[1]
    n1 = int((x1 - T0) / P)
    n2 = int((x2 - T0) / P)
    if n2 < n1:
        n1, n2 = n2, n1
    for j in np.arange(n1, n2 + 1):
        plt.axvline(x=j * P + T0, color="k", alpha=0.7)

        if dt is not None:
            plt.axvspan(
                xmin=j * P + T0 - dt / 2, xmax=j * P + T0 + dt / 2, color="k", alpha=0.4
            )


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


def unique_indexing(array):
    return find_nearest(np.unique(array), array)[0]


def IQ(array, axis=None):
    return np.nanpercentile(array, 75, axis=axis) - np.nanpercentile(
        array, 25, axis=axis
    )


def supSigma(array, axis=None):
    return np.nanpercentile(array, 84, axis=axis) - np.nanpercentile(
        array, 50, axis=axis
    )


def infSigma(array, axis=None):
    return np.nanpercentile(array, 50, axis=axis) - np.nanpercentile(
        array, 16, axis=axis
    )


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


def sigmoid(x, mu, sig):
    return (1 + np.exp(-sig * (x - mu))) ** -1


def bands_binning(gridx, vecx, vecy, vecyerr=None, binning="mean"):

    gridx = np.sort(gridx)

    m1 = (vecx < gridx[1:][:, np.newaxis]) * (vecx > gridx[0:-1][:, np.newaxis])
    m = (m1).astype("float")
    m[m == 0] = np.nan

    if vecyerr is None:
        vecyerr = np.ones(np.shape(vecy))

    w = np.nansum(m / vecyerr**2, axis=1)
    w[w == 0] = np.inf

    if binning == "mean":
        medx = np.nansum(m * vecx / vecyerr**2, axis=1) / w
        medy = np.nansum(m * vecy / vecyerr**2, axis=1) / w
    if binning == "sum":
        medx = np.nansum(m * vecx / vecyerr**2, axis=1)
        medy = np.nansum(m * vecy / vecyerr**2, axis=1)
    elif binning == "median":
        medx = np.nanmedian(m * vecx, axis=1)
        medy = np.nanmedian(m * vecy, axis=1)
    else:
        medx = np.nanmedian(m * vecx, axis=1)
        medy = np.nanpercentile(m * vecy, binning, axis=1)

    nb = np.nansum(m, axis=1)

    all_val = [np.ravel(medx), np.ravel(medy), np.ravel(nb)]

    all_val = np.array(all_val).T
    table_flat = pd.DataFrame(all_val, columns=["x", "y", "nb"])

    return table_flat


def find_borders_old(matrix, cluster_min_size, r_min=0):  # 21.6.21
    binary_matrix = (abs(matrix) > r_min).astype("int")
    borders = []
    mask_cluster = np.zeros(len(binary_matrix))
    i = 0
    for j in np.arange(1, len(binary_matrix)):
        if np.sum(binary_matrix[j, i:j]) < (j - i) * 0.5:
            if (j - i) > cluster_min_size:
                borders.append([i, j])
                mask_cluster[i:j] = 1
            i = j
    borders.append([borders[-1][1], len(binary_matrix) - 1])
    cluster_loc = np.unique(np.hstack(borders))
    mask_cluster = mask_cluster.astype("bool")
    return borders, cluster_loc, mask_cluster


def grid_binning_slow(gridx, gridy, vecx, vecy, vecz, Draw=True, cmap="plasma"):
    gridx = np.sort(gridx)
    gridy = np.sort(gridy)
    median = np.nanmedian(vecz)

    nb = []
    z = []
    for gx1, gx2 in zip(gridx[0:-1], gridx[1:]):
        for gy1, gy2 in zip(gridy[0:-1], gridy[1:]):
            mask = (vecx <= gx2) * (vecx >= gx1) * (vecy <= gy2) * (vecy >= gy1)
            nb.append(np.nansum(mask))
            z.append(np.nanmedian(vecz[mask]))

    nb = np.array(nb)
    z = np.array(z)
    x, y = np.meshgrid(0.5 * (gridx[1:] + gridx[0:-1]), 0.5 * (gridy[1:] + gridy[0:-1]))
    z[nb == 0] = median
    z = np.reshape(z, np.shape(x))

    all_val = [
        np.ravel(x),
        np.ravel(y),
        np.ravel(x),
        np.ravel(y),
        np.ravel(z),
        np.ravel(z),
        np.ravel(nb),
    ]

    all_val = np.array(all_val).T
    limite = np.nanpercentile(all_val[all_val[:, -1] != 0, -1], 5)
    table_flat = pd.DataFrame(
        all_val, columns=["x", "y", "binx", "biny", "binz", "binzerr", "nb"]
    )
    table = np.array([x, y, z, z])

    if Draw:
        vmin = np.nanpercentile(vecz, 16)
        vmax = np.nanpercentile(vecz, 84)
        plt.subplot(1, 2, 1)
        plt.scatter(vecx, vecy, c=vecz, vmin=vmin, vmax=vmax, cmap=cmap)
        ax = plt.gca()
        plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
        plt.xlim(np.min(gridx), np.max(gridx))
        plt.ylim(np.min(gridy), np.max(gridy))
        my_colormesh(
            np.unique(table_flat["x"]),
            np.unique(table_flat["y"]),
            z,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
    return table_flat, table


def grid_binning(
    gridx, gridy, vecx, vecy, vecz, veczerr=None, Draw=True, cmap="plasma"
):

    gridx = np.sort(gridx)
    gridy = np.sort(gridy)
    median = np.nanmedian(vecz)

    m1 = (vecx < gridx[1:][:, np.newaxis]) * (vecx > gridx[0:-1][:, np.newaxis])
    m2 = (vecy < gridy[1:][:, np.newaxis]) * (vecy > gridy[0:-1][:, np.newaxis])
    m = (m1 * m2[:, np.newaxis]).astype("float")
    m[m == 0] = np.nan

    if veczerr is None:
        veczerr = np.ones(np.shape(vecz))

    w = np.nansum(m / veczerr**2, axis=2)
    w[w == 0] = np.inf

    z = np.nansum(m * vecz / veczerr**2, axis=2) / w
    zerr = 1 / np.sqrt(w)
    medx = np.nansum(m * vecx / veczerr**2, axis=2) / w
    medy = np.nansum(m * vecy / veczerr**2, axis=2) / w
    nb = np.nansum(m, axis=2)

    x, y = np.meshgrid(0.5 * (gridx[1:] + gridx[0:-1]), 0.5 * (gridy[1:] + gridy[0:-1]))

    z[nb == 0] = median
    medy[nb == 0] = y[nb == 0]
    medx[nb == 0] = x[nb == 0]

    all_val = [
        np.ravel(x),
        np.ravel(y),
        np.ravel(medx),
        np.ravel(medy),
        np.ravel(z),
        np.ravel(zerr),
        np.ravel(nb),
    ]

    all_val = np.array(all_val).T
    limite = np.nanpercentile(all_val[all_val[:, -1] != 0, -1], 5)
    # print('Minimum number of element required : ',limite)
    table_flat = pd.DataFrame(
        all_val, columns=["x", "y", "binx", "biny", "binz", "binzerr", "nb"]
    )
    table_flat.loc[
        table_flat["nb"] <= limite, "binz"
    ] = median  # remove the 5percent distribution with lowest count for the statistic
    # x,y = np.meshgrid(np.unique(table_flat['x']),np.unique(table_flat['y']))
    # z = np.reshape(np.array(table_flat['binz']),np.shape(x))
    table = np.array([x, y, z, zerr])

    if Draw:
        vmin = np.nanpercentile(vecz, 16)
        vmax = np.nanpercentile(vecz, 84)
        plt.subplot(1, 2, 1)
        plt.scatter(vecx, vecy, c=vecz, vmin=vmin, vmax=vmax, cmap=cmap)
        ax = plt.gca()
        plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
        plt.xlim(np.min(gridx), np.max(gridx))
        plt.ylim(np.min(gridy), np.max(gridy))
        my_colormesh(
            np.unique(table_flat["x"]),
            np.unique(table_flat["y"]),
            z,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
    return table_flat, table


def poly2D_basis(x, y, deg=3, basis="legendre", Draw=False):

    if basis == "legendre":  # orthogonal between -1 and 1
        z = np.polynomial.legendre.legvander2d(x, y, [deg, deg])
    elif basis == "laguerre":  # orthogonal between 0 and inf
        z = np.polynomial.laguerre.lagvander2d(x, y, [deg, deg])
    elif basis == "chebyshev":  # orthogonal between -1 and 1 for sqrt(1-x**2) weighting
        z = np.polynomial.chebyshev.chebvander2d(x, y, [deg, deg])
    elif basis == "hermite":  # orthogonal between -inf and inf
        z = np.polynomial.hermite.hermvander2d(x, y, [deg, deg])
    elif basis == "polynomial":
        z = np.polynomial.polynomial.polyvander2d(x, y, [deg, deg])
    elif basis == "zernike":
        z = []
        for j in np.arange((deg + 1) ** 2):
            coeff = np.zeros((deg + 1) ** 2)
            coeff[j] = 1
            z.append(poly2D_zernike_square(x, y, coeff))
        z = np.array(z).T
    else:
        pass

    c = z.T
    c /= np.sum(c**2, axis=1)[:, np.newaxis]

    if Draw:
        plt.figure(figsize=(20, 10))
        for j in range((deg + 1) ** 2):
            plt.subplot(deg + 1, deg + 1, j + 1)
            plt.scatter(x, y, c=c[j], cmap="seismic")
        plt.subplots_adjust(
            left=0.08, right=0.97, top=0.96, bottom=0.08, hspace=0.3, wspace=0.3
        )
        plt.figure()
        plt.imshow(np.sum(c * c[:, np.newaxis], axis=2))

    return c


def poly2D_zernike_square(x2, y2, params):
    """from http://nopr.niscair.res.in/bitstream/123456789/24377/1/IJPAP%2051%2812%29%20837-843.pdf"""

    x = x2.copy()
    y = y2.copy()

    f = np.sqrt(np.pi) / 2

    x -= np.nanmin(x)
    y -= np.nanmin(y)
    x /= np.nanmax(x) / np.sqrt(np.pi)
    y /= np.nanmax(y) / np.sqrt(np.pi)
    x -= np.sqrt(np.pi) / 2
    y -= np.sqrt(np.pi) / 2

    r = np.sqrt(x**2 + y**2)

    z1 = 1 + 0 * x
    z2 = 1.95 * x
    z3 = 1.95 * y
    z4 = (-1.59) + (3.02 * r**2)
    z5 = 3.82 * x * y
    z6 = 3.03 * (x**2 - y**2)
    z7 = (-4.58 * x) + (6.24 * x * r**2)
    z8 = (-4.58 * y) + (6.24 * y * r**2)
    z9 = (1.90) - (9.36 * r**2) + (7.8 * r**4)
    z10 = (5.5 * x**2 * y) - (7.06 * y**3) + (1.9 * y)
    z11 = (7.11 * x**3) - (5.53 * y**2 * x) - (1.91 * x)
    z12 = 13.1 * x * y * (x**2 - y**2)
    z13 = 14.9 * (x**4 - y**4) - 10 * (x**2 - y**2)
    z14 = (13.2 * r**2 - 12.4) * x * y
    z15 = (
        (13 * x**4)
        + 4.39 * (-1.44 - 2.16 * y**2) * x**2
        - (5.97 * y**2)
        + (12.6 * y**4)
        + 0.71
    )
    z16 = (
        (18.1 * x**4 * y)
        - 36.2 * (x**2 * y**3)
        + (3.62 * y**5)
        - (2.92 * y)
        + (4.81 * x**2 * y)
        + (6.23 * y**3)
    )
    z17 = (
        (11.1 * x**5)
        - 22.2 * (x**3 * y**2)
        - (3.84 * x**3)
        + (33 * y**2 * x)
        - (33.3 * x * y**4)
        - (2.9 * x)
    )
    z18 = (
        (7.21 * y)
        - 29.2 * (x**2 * y)
        - (24.6 * y**3)
        + (32.1 * x**4 * y)
        + (16.8 * x**2 * y**3)
        + (22.7 * y**5)
    )
    # z19 = (7.9 * x) - (3.72 * x**3) - (6.77 * x * y**2) + (34.8 * x**5) + (16.4 * x**3 * y**2) - (8.4 * x * y**4) #function not ortthogonal
    z19 = (
        (8.36 * (x * f))
        - (42.6 * (x * f) ** 3)
        - (11 * (x * f) * (y / f) ** 2)
        + (43.2 * (x * f) ** 5)
        + (39.1 * (x * f) ** 3 * (y / f) ** 2)
        - (3.76 * (x * f) * (y / f) ** 4)
    )
    # z19 = (7.55 * (x/f)) - (28.1 * (x/f)**3) - (11.2 * (x/f) * (y*f)**2) + (21.5 * (x/f)**5) + (20.1 * (x/f)**3 * (y*f)**2) - (1.58 * (x/f) * (y*f)**4)
    z20 = (
        (-11.6 * x**2 * y)
        + 37 * (y**3)
        + (14.2 * x**4 * y)
        + (3.75 * x**2 * y**3)
        - (42.9 * y**5)
        - (5.22 * y)
    )
    z21 = (
        (30 * x**5)
        - (21.4 * x**3 * y**2)
        + (24.8 * x * y**4)
        + (3.07 * x)
        - (20.6 * x**3)
        - (6.59 * y**2 * x)
    )

    z = [
        z1,
        z2,
        z3,
        z4,
        z5,
        z6,
        z7,
        z8,
        z9,
        z10,
        z11,
        z12,
        z13,
        z14,
        z15,
        z16,
        z17,
        z18,
        z19,
        z20,
        z21,
    ]

    model = np.sum([p * z for p, z in zip(params, z)], axis=0)

    return model


def make_poly2D(n, m, maxi=False):
    matrix = np.reshape(np.ones((n + 1) * (m + 1)), (m + 1, n + 1))
    matrix_all = np.reshape(np.arange((n + 1) * (m + 1)), (m + 1, n + 1)).astype(
        "float"
    )
    matrix_cx, matrix_cy = np.meshgrid(np.arange(n + 1), np.arange(m + 1))

    if maxi:
        x_power = np.arange(n + 1)
        y_power = np.arange(m + 1)
        matrix2 = x_power + y_power[:, np.newaxis]
        maximum = np.max([n, m])
        matrix[matrix2 > maximum] = 0
        matrix_all[matrix2 > maximum] = np.nan
        matrix_all = matrix_all[~np.isnan(matrix_all)]
    nb_params = int(np.sum(matrix))
    matrix_all = np.ravel(matrix_all).astype("int")
    matrix_cx = np.ravel(matrix_cx)
    matrix_cy = np.ravel(matrix_cy)

    def Surface_test(X, *params):
        x, y = X
        if len(params) != nb_params:
            print("wrong number of parameters")
        x1 = np.array([x**p for p in range(n + 1)])
        y1 = np.array([y**p for p in range(m + 1)])

        z1 = np.zeros(np.shape(x1)[1])
        for num, j in enumerate(matrix_all):
            z1 = z1 + params[num] * x1[matrix_cx[j], :] * y1[matrix_cy[j], :]
        return z1

    def S_test(X, Y, params):
        if len(params) != nb_params:
            print("wrong number of parameters")
        x1 = np.array([X**p for p in range(n + 1)])
        y1 = np.array([Y**p for p in range(m + 1)])

        z1 = np.zeros((np.shape(x1)[1], np.shape(x1)[2]))
        for num, j in enumerate(matrix_all):
            z1 = z1 + params[num] * x1[matrix_cx[j]] * y1[matrix_cy[j]]
        return z1

    return nb_params, Surface_test, S_test


def make_sound(sentence, voice="Victoria"):
    if True:
        try:
            os.system('say -v %s "' % (voice) + sentence + '"')
        except:
            print("\7")
    else:
        print("\7")


def plot_copy_time(ax1=None, fmt="isot", time="x", split=0):
    if ax1 is None:
        ax1 = plt.gca()
    if time == "x":
        x = ax1.get_xticks()[1:-1]
    else:
        x0 = Time.Time(ax1.get_xticks()[0], format="mjd").decimalyear
        x1 = Time.Time(ax1.get_xticks()[-1], format="mjd").decimalyear
        tm = np.round(x1 - x0, 0)

        x = int(ax1.get_xticks()[0]) + 365.25 / (12 / tm) * np.arange(13)

    ax2 = ax1.twiny()

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x)
    if fmt == "deci":
        new_labels = Time.Time(x, format="mjd").decimalyear
        new_labels = ["%.2f" % (i) for i in new_labels]
    else:
        new_labels = Time.Time(x, format="mjd").isot
        new_labels = [i.split("T")[split] for i in new_labels]
    ax2.set_xticks(x)
    ax2.set_xticklabels(new_labels)
    return ax1


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


def randomly_shuffle_dataframe(dataframe, sort=0, axis=0):
    vec = np.ravel(dataframe.values)
    new_vec = np.random.choice(vec, len(vec), replace=False)
    new_matrix = np.reshape(new_vec, np.shape(dataframe))

    if sort == 1:
        new_matrix = np.sort(new_matrix, axis=axis)

    if sort == -1:
        new_matrix = np.sort(new_matrix, axis=axis)
        if axis:
            new_matrix = new_matrix[:, ::-1]
        else:
            new_matrix = new_matrix[::-1]

    new_dataframe = pd.DataFrame(new_matrix, columns=list(dataframe.keys()))

    return new_dataframe


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

    larger = (vec >= (cluster_output[:, 0][:, np.newaxis] - extended)).astype(
        "int"
    ) * elevation[:, np.newaxis]
    smaller = (vec <= (cluster_output[:, 1][:, np.newaxis] + 1 + extended)).astype(
        "int"
    ) * elevation[:, np.newaxis]
    flat = np.sqrt(np.sum(larger * smaller, axis=0))
    return flat


def bij_modulo(ang, tresh=3):
    offset = np.zeros(len(ang))
    for i in range(len(ang) - 1):
        if ang[i] - ang[i + 1] > tresh:
            offset[i + 1 :] = offset[i + 1 :] - 1
        elif ang[i] - ang[i + 1] < -tresh:
            offset[i + 1 :] = offset[i + 1 :] + 1

    return ang - offset * 2 * np.pi


def bij_modulo_merge(ang, max_cut=11, offset=np.pi):
    max_cut = 2 * int(max_cut / 2) + 1
    shift = max_cut // 2
    for i in range(len(ang) - 1):
        all_ang = ang[i + 1] + offset * np.arange(-shift, shift + 1)
        dist = ang[i] - all_ang
        min_idx = np.argmin(abs(dist))
        ang[i + 1 :] = ang[i + 1 :] + offset * np.arange(-shift, shift + 1)[min_idx]
    return ang


def dist_modulo(ang1, ang2):
    "for ang1§ and ang2 in radian, output the angle in degree"
    ang1 = ang1 * 180 / np.pi % 360 - 180
    ang2 = (ang2 * 180 / np.pi) % 360 - 180
    a = ang1 - ang2
    a = (a + 180) % 360 - 180
    return abs(a)


def map_unique(array):
    array = np.array(array)
    values = np.unique(array)
    values_int = np.arange(len(values))

    new_index = array.copy()
    for i, j in zip(values_int, values):
        new_index[np.where(array == j)[0]] = i

    return new_index.astype("int")


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


def fill_hole_mean(matrix, def_hole=0, axis=0):
    new_matrix = matrix.copy()
    matrix2 = np.roll(matrix, 1, axis=axis)
    matrix1 = np.roll(matrix, -1, axis=axis)

    mask = matrix == def_hole

    new_matrix[mask] = 0.5 * (matrix2[mask] + matrix1[mask])

    mask = (matrix2 * matrix1) == 0
    new_matrix[mask] = matrix[mask]

    return new_matrix


def detect_obs_season(time, min_gap=40):
    loc = np.where(np.diff(time) > min_gap)[0]
    loc = np.sort(np.array([0] + list(loc) + list(loc + 1) + [len(time) - 1]))
    borders = np.array([loc[::2], loc[1::2]]).T

    return borders


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


def match_unique_closest(array1, array2):
    """return a table [idx1,idx2,num1,num2,distance] matching the closest element from an array to the other, each pair is unique. Remark : algorithm very slow by conception if the arrays are too large."""
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

    m = array2_r - array1_r[:, np.newaxis]
    m_line = np.ones(len(array2_r)) * np.arange(len(array1_r))[:, np.newaxis]
    m_col = np.arange(len(array2_r)) * np.ones(len(array1_r))[:, np.newaxis]

    save = []

    for j in range(np.min([len(array1_r), len(array2_r)])):
        line, col = np.where(m == np.nanmin(abs(m)))
        if len(line):
            line = line[0]
            col = col[0]
            save.append(
                [
                    m_line[line][col],
                    m_col[line][col],
                    array1_r[line],
                    array2_r[col],
                    m[line][col],
                ]
            )

            m = np.delete(m, line, axis=0)
            m = np.delete(m, col, axis=1)

            m_col = np.delete(m_col, line, axis=0)
            m_col = np.delete(m_col, col, axis=1)

            m_line = np.delete(m_line, line, axis=0)
            m_line = np.delete(m_line, col, axis=1)
        else:
            break

    save = np.array(save)

    return save


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


def In_sigma(array, value):
    if (value < np.percentile(array, 84)) & (value > np.percentile(array, 16)):
        return 1
    if (value < np.percentile(array, 97.5)) & (value > np.percentile(array, 2.5)):
        return 2
    if (value < np.percentile(array, 99.5)) & (value > np.percentile(array, 0.5)):
        return 3
    else:
        return 99


#


def weighted_sigma(data, weights, Bessel=True):
    std = []
    if np.ndim(weights) == 1:
        weights = np.reshape(weights, (len(weights), 1))
    if np.ndim(data) == 1:
        data = np.reshape(data, (len(data), 1))
    for j in range(np.shape(weights)[1]):
        nonzeros = len(np.where(weights[:, j] != 0)[0])
        if Bessel == True:
            coeff = (nonzeros - 1) / nonzeros
        else:
            coeff = 1
        denominateur = coeff * np.sum(weights[:, j])
        weighted_average = np.average(data, axis=0, weights=weights)
        nominateur = np.sum(weights[:, j] * (data[:, j] - weighted_average[j]) ** 2)
        std.append(np.sqrt(nominateur / denominateur))
    return np.array(std)


#


def compute_CCF(
    l,
    spe,
    mask_ll,
    mask_W,
    RV_table,
    bin_width,
    normalize,
    excluded_ll,
    excluded_range,
    instrument="HARPS",
    rescale=0,
):

    """
    Compute the Cross Correlation Function for a given set of radial velocities and an 'observed spectrum'.

    The CCF is computed with an arbitrary binary mask rather than with a complete theoretical model. This apporach allows to build an
    average profile of the lines naturally because of the binarity of the mask (RV_table, CCF). It is average, because for each fixed
    RV you sum on all the lines in the line list.
    In the other approach (a la Snellen), the interpretation of the shape of (RV_table, CCF) is more complicated since it depends on
    the shape of the lines in the models we use for the cross-correlation.
    """

    # l, spe: spe(l) is the observed spectrum at wavelength l;
    # mask_ll: the line list (wavelength);
    # mask_W: weight of the single line (Pepe+, 2002). In transmission spectroscopy this is delicate, and thus set to 1 for every line.
    # RV_table: the table of radial velocities of the star behind the planet. Corresponds to a shift in the mask. Max CCF = true RV (theoretically)
    #           for HARPS, use -20/+20 km/sec in the planetary rest frame.
    # bin_width: the width of the mask bin in radial velocity space (= 1 pixel HARPS; too big: bending lines you lose information; too small: ????)
    #            Rs = 115000; dv_FWHM = c/Rs ~ 2.607 km/sec; dv_pix = dv_FWHM/sampling = 2.607/3.4 km/sec = 0.76673 km/sec
    # normalize: just for telluric correction. 1 = True
    # excluded_ll: used to exclude specific contaminating wavelengths; e.g. O2
    # excluded_range: same as before but with a range in the wavelength space
    #                 NOT SURE ABOUT THIS. From the code, it seems that a line in the 'excluded_ll' is rejected only if it falls inside this region.
    #                 I guess you may want to exclude the lines for a certain element only in certain regions of the spectrum, while you may want to
    #                 include them in others.
    # instrument: only important for HARPS, that has a gap between the CCDs
    # rescale: ????

    # This is just the spacing between the wavelengths, and a nice way to that by the way.
    dl = l[1:] - l[:-1]
    dl = np.concatenate((dl, np.array([dl[-1]])))

    # Centers of the wavelength bins (there is a reason for this, I don't remember which).
    l2 = l - dl / 2.0

    # CCF contains the CCF for each radial velocity (thus has the size of RV_table)
    CCF = np.zeros(len(RV_table), "d")
    rejected = 0

    # Cycle on the line list
    for j in range(len(mask_ll)):
        dCCF = np.zeros(len(RV_table), "d")
        # Jump the gap between the CCDs if the data comes from HARPS
        if (instrument == "HARPS") & (mask_ll[j] > 5299.0) & (mask_ll[j] < 5342.0):
            continue
            # 2.99792458e5 = c in km/sec
            # Exclude polluters.
        # if abs(excluded_ll-mask_ll[j]).min() < (mask_ll[j]*excluded_range/2.99792458e5):
        if np.min(abs(excluded_ll - mask_ll[j])) < (
            mask_ll[j] * excluded_range / 2.99792458e5
        ):

            rejected = rejected + 1
            continue
        # Compute the width of the mask in the wavelength space.
        # lambda/delta_lambda = c/delta_v ---> delta_lambda = lambda * delta_v /c
        mask_width = mask_ll[j] * bin_width / 2.99792458e5
        # For each line in the line list, cycle on the RVs and compute the contribution to the CCF.
        for k in range(len(RV_table)):
            # The planet moves with respect to the star. It absorbs in its rest frame, thus it is the 'observer' in this case -> Doppler formula with the plus.
            # Check: positive velocity is away from the star; negative is towards the star. In fact:
            # When the planet moves towards th star, RV < 0, thus the wavelength at which is absorbs is bluer than the original.
            lstart = mask_ll[j] * (1.0 + RV_table[k] / 2.99792458e5) - mask_width / 2.0
            lstop = mask_ll[j] * (1.0 + RV_table[k] / 2.99792458e5) + mask_width / 2.0

            # index1 is the index of the element in l2 such that if I insert lstart on its left the order is preserved. Thus, it is the index of the first wavelength
            # bin contained in the mask.
            # index2 is the index of the first wavelength bin not completely covered by the mask.
            index1 = np.searchsorted(l2, lstart)
            index2 = np.searchsorted(l2, lstop)

            # First term: all the bins completely contained in the mask. Second term: I have to add a piece of bin on the left of index1 which has been excluded in the
            # first term. Third term: the last bin contributes to the CCF only partially, remove the excess.
            dCCF[k] = (
                sum(spe[index1:index2])
                + spe[index1 - 1] * (l2[index1] - lstart) / dl[index1 - 1]
                - spe[index2 - 1] * (l2[index2] - lstop) / dl[index2 - 1]
            )

        if normalize == 1:
            index = len(RV_table) / 5
            #               slope = (dCCF[-index:].mean()-dCCF[:index].mean())/(RV_table[-index:].mean()-RV_table[:index].mean())
            #               c = dCCF[:index].mean() + slope*(RV_table-RV_table[:index].mean())
            c = np.concatenate((dCCF[:index], dCCF[-index:])).mean()
            if (dCCF.min() <= 0.0) | (c <= 0.0):
                continue
            dCCF = np.log(dCCF / c)
        if rescale == 1:
            index = np.searchsorted(l, mask_ll[j])
            dCCF = dCCF * dl[index] / mask_width

        # Sum the weighted contribution of each line to the CCF.
        CCF = CCF + mask_W[j] * dCCF
    if excluded_range > 0.0:
        print(
            "Rejected %i/%i lines in excluded wavelength range"
            % (rejected, len(mask_ll))
        )
    if rescale == 1:
        CCF = CCF / len(mask_ll)
    return CCF


def ccf2(wave, spec1, wave2, spec2, rv_max=30):
    "CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask"
    dwave = wave2[1] - wave2[0]
    dv = (10 ** (dwave) - 1) * 299.792e6 / 1000

    shift = np.arange(0, rv_max, dv)
    all_rv = np.hstack([-shift[::-1][:-1], shift])
    all_shift = np.hstack([-np.arange(len(shift))[::-1][:-1], np.arange(len(shift))])

    convolution = []
    for j in tqdm(all_shift):
        new_spec = interp1d(
            wave2,
            np.roll(spec2, j),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(wave)
        convolution.append(np.sum(new_spec * spec1))
    return all_rv, np.array(convolution)


def display_line(dataframe, wave, kw="wave"):
    idx = find_nearest(np.array(dataframe[kw]), wave)[0]
    return dataframe.index[idx]


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


def ccf(
    wave, spec1, spec2, extended=1500, rv_range=45, oversampling=10, spec1_std=None
):
    "CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask"
    dwave = np.median(np.diff(wave))

    if spec1_std is None:
        spec1_std = np.zeros(np.shape(spec1))

    if len(np.shape(spec1)) == 1:
        spec1 = spec1[:, np.newaxis].T
    if len(np.shape(spec1_std)) == 1:
        spec1_std = spec1_std[:, np.newaxis].T
    # spec1 = np.hstack([np.ones(extended),spec1,np.ones(extended)])

    spec1 = np.hstack(
        [np.ones((len(spec1), extended)), spec1, np.ones((len(spec1), extended))]
    )
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
                np.sqrt(np.abs(np.nansum(new_spec2 * spec1_std**2, axis=1)))
                / sum_spec
            )
            shift_save.append(j + k * dwave)
    shift_save = np.array(shift_save)
    sorting = np.argsort(shift_save)
    return (
        (299.792e6 * 10 ** shift_save[sorting]) - 299.792e6,
        np.array(convolution)[sorting],
        np.array(convolution_std)[sorting],
    )


def gauss_vincent(x, cen, cen2, amp, amp2, offset, wid, wid2):
    return (
        amp * np.exp(-0.5 * (x - cen) ** 2 / (2 * wid**2))
        + amp2 * np.exp(-0.5 * (x - cen2) ** 2 / (2 * wid2**2))
        + offset
    )


def gaussian(x, cen, amp, offset, wid):
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2)) + offset


def lorentzian(x, amp, cen, offset, wid):
    return amp * wid**2 / ((x - cen) ** 2 + wid**2) + offset


def checking(array1, array2, tresh=0.05):
    """checking(array1,array2,tresh) check if the items in array1 are in array2 inside a treshold and return the corresponding mask"""
    check = []
    for item in array1:
        if find_nearest(array2, item)[2] <= tresh:
            check.append(True)
        else:
            check.append(False)
    return np.array(check)


#
def substract_model(x, y, *par):
    if np.shape(par[0]) == ():
        a, b = par[0], par[1]
    else:
        a, b = par[0]
    model = a * x + b
    return y - model


#


def string_contained_in(array, string):
    array = np.array(array)
    split = np.array([len(i.split(string)) - 1 for i in array])
    return split.astype("bool"), array[split.astype("bool")]


def n_sigma_gaussian(dim, sigma=1.0, nb_point=1e8):
    a = np.random.randn(int(nb_point), dim)
    return np.sum(np.sqrt(np.sum(a**2, axis=1)) <= 1.0 * sigma) / nb_point


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
            mask = abs(array - np.nanmean(array, axis=axis)) <= m * np.nanstd(
                array, axis=axis
            )
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


#
def fit_line(x, y):
    return stats.linregress(x, y)[0], stats.linregress(x, y)[1]


def conv_void_air(wave):
    s2 = 1e4 / wave
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return wave / n


def conv_air_void(wave):
    s2 = 1e4 / wave
    n = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s2)
        + 0.0001599740894897 / (38.92568793293 - s2)
    )
    return wave * n


#
def slippery_filter(x, y, box=20, sigma=False):
    """Slippery median to filter a signal (flattening). slippery(x, y, box=size, sigma=val)."""
    grille = range(box, len(y) - box)
    slippery = np.array([np.median(y[j - box : j + box]) for j in grille])
    if sigma != False:
        enveloppe = np.array([np.std(y[j - box : j + box]) * sigma for j in grille])
        return x[grille], slippery, enveloppe
    else:
        return x[grille], slippery


def gaus(x, x0, sigma, norm=False):
    if not norm:
        return np.sqrt(2.0 * np.pi * sigma**2) * np.exp(
            -((x - x0) ** 2) / (2 * sigma**2)
        )
    if norm:
        return (
            np.sqrt(2.0 * np.pi * sigma**2)
            * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
            / np.max(
                np.sqrt(2.0 * np.pi * sigma**2)
                * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
            )
        )


def rand_unif(mini, maxi):
    return np.random.rand(1) * (maxi - mini) + mini


def gen_gaus(grid, number, xmin, xmax, sig_min, sig_max, amp_min, amp_max):
    synthetic = np.ones(len(grid))
    for j in range(number):
        synthetic *= 1 - rand_unif(amp_min, amp_max) * gaus(
            grid, rand_unif(xmin, xmax), rand_unif(sig_min, sig_max), norm=True
        )
    return synthetic


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
    std_cont = 1 / np.sqrt(
        np.percentile(spectrei[l3 - 8 * window : l4 + 8 * window], 95)
    )

    return (
        d1,
        d2,
        std_d1,
        std_d2,
        d1 / d2,
        std_cont * np.sqrt(2 + (1.0 / (1 - d1)) ** 2 + (1.0 / (1 - d2)) ** 2),
        np.sqrt((std_d1 / d2) ** 2 + (std_d2 * d1 / d2**2) ** 2),
    )


def broadGaussFast(x, y, sigma, edgeHandling=None, maxsig=None):
    dxs = x[1:] - x[0:-1]
    if maxsig is None:
        lx = len(x)
    else:
        lx = int(((sigma * maxsig) / dxs[0]) * 2.0) + 1
    nx = (np.arange(lx, dtype=np.int) - sum(divmod(lx, 2)) + 1) * dxs[0]
    e = gaus(nx, 0, sigma)
    e /= np.sum(e)
    if edgeHandling == "firstlast":
        nf = len(y)
        y = np.concatenate((np.ones(nf) * y[0], y, np.ones(nf) * y[-1]))
        result = np.convolve(y, e, mode="same")[nf:-nf]
    elif edgeHandling is None:
        result = np.convolve(y, e, mode="same")
    return result


def instrBroadGaussFast(
    wvl, flux, resolution, edgeHandling=None, fullout=False, maxsig=None
):
    meanWvl = np.mean(wvl)
    fwhm = 1.0 / float(resolution) * meanWvl
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    result = broadGaussFast(wvl, flux, sigma, edgeHandling=edgeHandling, maxsig=maxsig)

    if not fullout:
        return result
    else:
        return (result, fwhm)


def label_int(vec):
    new_vec = np.zeros(len(vec)).astype("int")
    for k, n in enumerate(np.unique(vec)):
        new_vec[np.where(vec == n)[0]] = k
    return new_vec.astype("int")


## Useful tools ##


def clip_vector(vectors, lower=None, upper=None):
    if lower is not None:
        mask1 = vectors > lower
    else:
        mask1 = np.ones(len(vectors)).astype("bool")
    if upper is not None:
        mask2 = vectors < upper
    else:
        mask2 = np.ones(len(vectors)).astype("bool")
    mask = mask1 & mask2
    return mask


def savefits(data1, data2, modele, name):
    new_hdul = fits.HDUList()
    new_hdul.append(fits.PrimaryHDU(header=modele[0].header))
    new_hdul.append(fits.ImageHDU(data1, header=modele[1].header, name=modele[1].name))
    new_hdul.append(fits.ImageHDU(data2, header=modele[2].header, name=modele[2].name))
    new_hdul.writeto(name, clobber=True)


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


def sphinx(sentence, rep=None, s2=""):
    answer = "-99.9"
    print(
        " ______________ \n\n --- SPHINX --- \n\n TTTTTTTTTTTTTT \n\n Question : "
        + sentence
        + "\n\n [Deafening silence ...] \n\n ______________ \n\n --- OEDIPE --- \n\n XXXXXXXXXXXXXX \n "
    )
    if rep != None:
        while answer not in rep:
            answer = input("Answer : " + s2)
    else:
        answer = input("Answer : " + s2)
    return answer


def savetxt(pathname, data, header, *format):
    """savetxt(pathname, data, header(as a string), format(as a list))"""
    underscore = ""
    for num, char in enumerate(header):
        if char != " ":
            underscore += "_"
        else:
            underscore += char

    header = header + "\n" + underscore + "\n"
    if format == ():
        np.savetxt(pathname, data, header=header, delimiter=" | ")
    elif (len(format) == 1) & (format[0] == "%s"):
        np.savetxt(pathname, data, header=header, fmt="%s", delimiter=" | ")
    else:
        np.savetxt(pathname, data, header=header, fmt=format[0], delimiter=" | ")


## console interactions ##


def print_pourcent(value, total):
    pourcentage = value * 100 / total
    return print(
        np.str(value)
        + "/"
        + np.str(total)
        + " ["
        + np.str(np.int(pourcentage))
        + " %]",
        end="\r",
    )


#
def print_warning():
    return print("\n ------ LOOK AT THERE ! ------- \n")


def print_box(sentence):
    print("\n")
    print("L" * len(sentence))
    print(sentence)
    print("T" * len(sentence))
    print("\n")


#
def blockPrint():
    """Disable the ability for the console to print something. Don't forget to restore it with the enablePrint function below."""
    sys.stdout = open(os.devnull, "w")


#
def enablePrint():
    #
    """Restore the ability for the terminal to print"""
    sys.stdout = sys.__stdout__


def writing_line(file, lenght=35):
    return file.write("\n" + "_" * 30 + "\n")


## Useful mathematical functions ##

#


def doppler_old(lamb, v):
    """Relativistic Doppler. Take (wavelenght, velocity[m/s]) and return lambda observed and lambda source"""
    c = 299.792e6
    factor = np.sqrt((1 + v / c) / (1 - v / c))
    lambo = lamb * factor
    lambs = lamb * (factor ** (-1))
    return lambo, lambs


def move_extract_dace(path):
    listOfFiles = [x[0] for x in os.walk(path)]
    for j in listOfFiles:
        os.system("mv " + j + "/*.fits " + path)


def plot_color_box(color="r", font="bold", lw=2):
    ax = plt.gca()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(lw)
        ax.spines[axis].set_color(color)

    ax.tick_params(axis="x", which="both", colors=color)
    ax.tick_params(axis="y", which="both", colors=color)

    if font == "bold":
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight("bold")

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight("bold")


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


#


def produce_dace_simbad_table():
    with open("/Users/cretignier/Documents/Python/Material/HARPN.json") as f:
        data_harpn = json.load(f)
    with open("/Users/cretignier/Documents/Python/Material/HARPS.json") as f:
        data_harps = json.load(f)

    data_harps = data_harps["values"]
    data_harpn = data_harpn["values"]

    table = []

    for j in np.arange(len(data_harps)):
        for k in data_harps[j]["obj_id_catname"]:
            table.append([k, data_harps[j]["obj_id_basename"]])

    for j in np.arange(len(data_harpn)):
        for k in data_harpn[j]["obj_id_catname"]:
            table.append([k, data_harpn[j]["obj_id_basename"]])

    table = np.array(table)

    pickle_dump(
        pd.DataFrame(table, columns=["dace", "simbad"]),
        open(root + "/Python/Material/Dace_to_simbad.p", "wb"),
    )


def crossmatch_dace_simbad(starname, direction="simbad"):
    table = pd.read_pickle(root + "/Python/Material/Dace_to_simbad.p")

    if direction == "simbad":
        name = np.array(table.loc[table["dace"] == starname, "simbad"])
    elif direction == "dace":
        name = np.array(table.loc[table["simbad"] == starname, "dace"])
    else:
        return None

    if len(name):
        return name[0]
    else:
        return None


def vec_oversamp(vec):
    return np.linspace(np.min(vec), np.max(vec), 100 * len(vec))


def conv_angle_corr(angle, angle_std=None):
    if angle_std is None:
        angle_std = np.zeros(np.shape(angle))

    angle = np.array(angle)
    angle_std = np.array(angle_std)

    angle_min = (angle - angle_std) % (2 * np.pi)
    angle_max = (angle + angle_std) % (2 * np.pi)
    angle_mean = (angle) % (2 * np.pi)

    angle_min = (angle_min + np.pi / 2) % (2 * np.pi) - np.pi
    angle_max = (angle_max + np.pi / 2) % (2 * np.pi) - np.pi
    angle_mean = (angle_mean + np.pi / 2) % (2 * np.pi) - np.pi

    angle_min = (angle_min <= 0).astype("int")
    angle_max = (angle_max <= 0).astype("int")
    angle_mean = (angle_mean <= 0).astype("int")

    r = 2 * ((angle_min + angle_max + angle_mean) / 3 - 0.5)
    r[abs(r) != 1] = 0
    return r


def AmpStar(ms, mp, periode, amplitude, i=90, e=0, code="Sun-Earth"):
    """(mass_star , mass_planet, period, amplitude , i=90 , e=0,[code]) return the unknown from vecteur (value fixed at 0) of the RV signal giving the star mass (solar mass) ans the period (year) amplitude in (meter/seconde). Mass can be given in solar, earth and jupitar mass with the code option Sun-Earth and Sun-Jupiter"""
    periode = periode / 365.25
    ms = Mass_sun * ms
    if type(mp) == int:
        if mp == 0:
            periode = periode
            amplitude = amplitude
            time = periode * 365.25 * 24 * 3600
            coeff = np.power(
                (
                    ms**2
                    * np.power(1 - e**2, 1.5)
                    * time
                    * amplitude**3
                    / (2 * np.pi * 6.67e-11)
                ),
                1 / 3.0,
            )
            if code == "Sun-Earth":
                return coeff / Mass_earth
            if code == "Sun-Jupiter":
                return coeff / Mass_jupiter
    if type(periode) == int:
        if periode == 0:
            mp = mp
            amplitude = amplitude
            if code == "Sun-Earth":
                mp = Mass_earth * mp
            if code == "Sun-Jupiter":
                mp = Mass_jupiter * mp
            mass_proj = mp * np.sin(i * np.pi / 180)
            coeff = (mass_proj**3 * 2 * np.pi * 6.67e-11) / (
                ms**2 * amplitude**3 * np.power(1 - e**2, 1.5)
            )
            time = coeff / (365.25 * 24 * 3600)
            return time
    if type(amplitude) == int:
        if amplitude == 0:
            mp = mp
            periode = periode
            if code == "Sun-Earth":
                mp = Mass_earth * mp
            if code == "Sun-Jupiter":
                mp = Mass_jupiter * mp
            mass_proj = mp * np.sin(i * np.pi / 180)
            time = periode * 365.25 * 24 * 3600
            coeff = (mass_proj**3 * 2 * np.pi * 6.67e-11) / (
                ms**2 * time * np.power(1 - e**2, 1.5)
            )
            return np.power(coeff, 1 / 3.0)


def PeriodSolarRV(period, code="day", position="inwards"):
    """Return the sinodyc period of a planet inside the Solar system see from the earth giving its period [days] (period, code='day')"""
    if code == "day":
        Earth_sidereal = 365.25
    if code == "year":
        Earth_sidereal = 1
    if position == "inwards":
        return abs(Earth_sidereal * period) / (Earth_sidereal - period)
    if position == "outwards":
        return abs(Earth_sidereal * period) / (period - Earth_sidereal)


def get_hill(a, e, mp, ms):
    return a * (1 - e) * ((mp * Mass_earth) / (3 * ms * Mass_sun)) ** (1 / 3)


def fit_planet(table, legend=True, alpha=1):
    curves = []

    if "P" in table.keys():
        table["p"] = table["P"]
    if "K" in table.keys():
        table["k"] = table["K"]
    if "w" in table.keys():
        table["peri"] = table["w"]
    if "L0" in table.keys():
        table["long"] = table["L0"]
    if "m_p" in table.keys():
        table["mass"] = table["m_p"]

    p_min = np.min(table["p"])
    p_max = np.max(table["p"])
    # t = np.arange(0, 3*p_max, p_min/72)
    t = np.arange(0, p_max, p_min / 36)
    color = -1
    for planet in table.index:

        color += 1
        semi_axis = table.loc[planet, "a"]
        period = table.loc[planet, "p"]
        ecc = table.loc[planet, "e"]
        node = table.loc[planet, "peri"]
        periastron = table.loc[planet, "long"]
        i = table.loc[planet, "i"]
        val = table.loc[planet, "mass"]

        if int(val) > 9999.9:
            val = 9999.9

        ke = pyasl.KeplerEllipse(semi_axis, period, ecc, Omega=node, i=i, w=periastron)
        pos = ke.xyzPos(t)
        if legend:
            plt.plot(
                pos[:, 0],
                pos[:, 1],
                label="%.1f $M_{\oplus}$ (%.1f days)" % (val, period),
                alpha=alpha,
                color=colors_cycle_mpl[color],
            )
        else:
            plt.plot(pos[:, 0], pos[:, 1], alpha=alpha, color=colors_cycle_mpl[color])
        plt.axis("equal")
        plt.scatter(0, 0, marker="*", color="yellow", ec="k", s=100)
        # plt.scatter(0,0,marker='x',color='k')
        plt.xlabel("X [AU]", fontsize=14)
        plt.ylabel("Y [AU]", fontsize=14)

        curves.append(pos)
    if legend:
        plt.legend(loc=1)
    curves = np.array(curves)
    return curves


def plot_hz(ms=1, ls_inf="-.", ls_sup=":", color_inf="r", color_sup="b"):
    """Kopparapu(2013)"""
    tab = pd.read_pickle(cwd + "/Material/HZ.p")
    index = find_nearest(tab["Ms"], ms)[0]
    hz_min = tab["HZ_inf"][index]
    hz_max = tab["HZ_sup"][index]

    plt.plot(
        hz_min * np.sin(np.linspace(0, 2 * np.pi, 100)),
        hz_min * np.cos(np.linspace(0, 2 * np.pi, 100)),
        color=color_inf,
        ls=ls_inf,
        label=r"$HZ_{inf}$",
    )
    plt.plot(
        hz_max * np.sin(np.linspace(0, 2 * np.pi, 100)),
        hz_max * np.cos(np.linspace(0, 2 * np.pi, 100)),
        color=color_sup,
        ls=ls_sup,
        label=r"$HZ_{sup}$",
    )


def plot_hill(x, y, hill_radius, color="k", ls=":", zorder=1):
    """Kopparapu(2013)"""

    plt.plot(
        x + hill_radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
        y + hill_radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
        color=color,
        ls=ls,
        zorder=zorder,
    )


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


def only_axis(color=None, lw=2, ax=None, side="all", ls="-"):
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if color is not None:
        plot_color_box(color=color, lw=lw, ax=ax, side=side, ls=ls)


def divide_0_0(a, b):
    eps = np.zeros(np.shape(a))
    eps[(a == 0) & (b == 0)] = 1
    return a / (b + eps)


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def compute_photon_noise_std(snr_continuum, kw="RV"):
    """From Photon_noise_CCF.py simulation"""

    dico = {
        "RV": [-0.9855573935046936, 0.2468210415895288],
        "Contrast": [-0.9765543442021598, -0.28227045485691565],
        "FWHM": [-0.9711086236761048, 0.32090083929671553],
        "Center": [-0.984651864233478, 0.49249441550671685],
        "Depth": [-0.9886403129578759, -0.2867164726734603],
        "EW": [-0.9808989674771276, -0.15466576617812083],
        "VSPAN": [-0.9839009757995953, 0.36998843415242944],
    }

    snr = np.log10(snr_continuum)
    coeff = dico[kw]

    return 10 ** (coeff[0] * snr + coeff[1])


def calc_Teff(t, sig):
    """
    Compute the effective time series length
    from the measurements times and errors.
    """
    w = 1.0 / (sig * sig)
    w /= np.sum(w)
    wt = w * t
    tm = np.sum(wt)
    t2m = np.sum(wt * t)
    return 2.0 * np.sqrt(np.pi * (t2m - tm * tm))


def fap(zmax, t, sig, freq_max, p0=1, Teff=None):
    """
    Compute the FAP at level zmax (highest periodogram peak),
    for a periodogram computed up to the frequency freq_max
    and with a base model with p0 parameters.

    If not provided, the effective time series length is computed
    with calc_Teff.
    """

    Nh = t.size - p0
    Nk = Nh - 2
    if Teff is None:
        Teff = calc_Teff(t, sig)
    W = freq_max * Teff
    chi2ratio = 1.0 - zmax
    if chi2ratio > 0:
        FapSingle = chi2ratio ** (Nk / 2.0)
        tau = W * FapSingle * np.sqrt(Nh * zmax / (2.0 * chi2ratio))
        Fap = FapSingle + tau
        if Fap > 1e-5:
            Fap = 1.0 - (1.0 - FapSingle) * np.exp(-tau)
        return Fap
    else:
        return 1


def merge_orders(wave, flux, flux_std, continuum=None, dwave=0.01, method="cubic"):
    wave_min = np.min(wave)
    wave_max = np.max(wave)
    new_wave = np.arange(wave_min, wave_max, dwave)

    nb_orders = np.shape(wave)[0]

    new_flux = np.zeros((nb_orders, len(new_wave)))
    new_flux_std = np.zeros((nb_orders, len(new_wave)))
    new_continuum = np.zeros((nb_orders, len(new_wave)))

    for j in range(nb_orders):
        flux_order = flux[j]
        flux_order[0:2] = 0
        flux_order[-2:] = 0

        flux_order_std = flux_std[j]
        flux_order_std[0:2] = 0
        flux_order_std[-2:] = 0

        flux2 = interp1d(
            wave[j],
            flux_order,
            kind=method,
            bounds_error=False,
            fill_value="extrapolate",
        )(new_wave)
        new_flux[j] = flux2

        flux2_std = interp1d(
            wave[j],
            flux_order_std,
            kind=method,
            bounds_error=False,
            fill_value="extrapolate",
        )(new_wave)
        new_flux_std[j] = flux2_std

        if continuum is not None:

            continuum_order = continuum[j]
            continuum_order[0:2] = 0
            continuum_order[-2:] = 0

            continuum2 = interp1d(
                wave[j],
                continuum_order,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate",
            )(new_wave)
            new_continuum[j] = continuum2

    new_flux[new_flux_std == 0] = np.nan
    new_continuum[new_flux_std == 0] = np.nan
    new_flux_std[new_flux_std == 0] = np.nan
    new_weight = 1 / new_flux_std**2

    merged = np.nansum(new_flux * new_weight, axis=0) / np.nansum(new_weight, axis=0)
    merged[np.isnan(merged)] = 0

    merged_std = 1 / np.sqrt(np.nansum(new_weight, axis=0))
    merged_std[np.isnan(merged_std)] = np.nanmax(merged_std)

    merged_continuum = np.nansum(new_continuum * new_weight, axis=0) / np.nansum(
        new_weight, axis=0
    )
    merged_continuum[np.isnan(merged_continuum)] = 0

    i = 0
    count = 0
    while i == 0:
        i = merged[count]
        count += 1

    new_wave = new_wave[count:]
    merged = merged[count:]
    merged_std = merged_std[count:]
    merged_continuum = merged_continuum[count:]

    i = 0
    count = -1
    while i == 0:
        i = merged[count]
        count -= 1

    new_wave = new_wave[:count]
    merged = merged[:count]
    merged_std = merged_std[:count]
    merged_continuum = merged_continuum[:count]
    merged_continuum[merged == 0] = 1

    return new_wave, merged, merged_std, merged_continuum


def star_year_observability(
    ra,
    dec,
    instrument="HARPS",
    Plot=False,
    airmass_min=2,
    new=True,
    nb_night=365,
    nb_pts=100,
    oversampling=5,
    return_curve=False,
    alt_sun_max=-12,
):
    if instrument == "HARPS":
        obs_loc = astrocoord.EarthLocation(
            lat=-29.260972 * u.deg, lon=-70.731694 * u.deg, height=2400
        )  # HARPN
        utcoffset = 0 * u.hour
    elif instrument == "HARPN":
        obs_loc = astrocoord.EarthLocation(
            lat=28.754000 * u.deg, lon=-17.889055 * u.deg, height=2387.2
        )  # HARPN
        utcoffset = 0 * u.hour
    elif instrument == "ESPRESSO":
        obs_loc = astrocoord.EarthLocation(
            lat=-24.627622 * u.deg, lon=-70.405075 * u.deg, height=2635
        )  # ESPRESSO
        utcoffset = 0 * u.hour
    elif instrument == "EXPRES":
        obs_loc = astrocoord.EarthLocation(
            lat=34.74444 * u.deg, lon=-68.578056 * u.deg, height=2360.0
        )  # LDT
        utcoffset = 0 * u.hour
    elif instrument == "CARMENES":
        obs_loc = astrocoord.EarthLocation(
            lat=37.223611 * u.deg, lon=2.546111 * u.deg, height=2168.0
        )  # Calar Alto
        utcoffset = 0 * u.hour
    elif instrument == "Geneva":
        obs_loc = astrocoord.EarthLocation(
            lat=46.204391 * u.deg, lon=6.143158 * u.deg, height=300
        )  # Geneva
        utcoffset = 0 * u.hour

    m33 = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

    t = np.linspace(2021, 2022, nb_night)

    midnight = Time.Time(t, format="decimalyear") - utcoffset

    nb = np.linspace(-12, 12, nb_pts)
    delta_midnight = nb * u.hour
    times = midnight + delta_midnight[:, np.newaxis]
    altazframe = AltAz(obstime=times, location=obs_loc)
    sunaltazs = get_sun(times).transform_to(altazframe)
    m33altazs = m33.transform_to(altazframe)

    def val(obj):
        mask = obj.alt.deg <= 0
        airmass = obj.secz
        airmass[mask] = 10
        return airmass

    star_airmass = val(m33altazs)

    # star_airmass[~mask] = 10

    dust, dust, sun_elev = my_colormesh(
        t,
        nb,
        np.array(sunaltazs.alt),
        zoom=oversampling,
        return_output=True,
        vmin=1,
        vmax=2,
        order=1,
    )

    new_t, new_nb, star_air = my_colormesh(
        t,
        nb,
        star_airmass,
        zoom=oversampling,
        return_output=True,
        vmin=1,
        vmax=2,
        order=1,
    )

    new_t = new_t[0]
    new_nb = new_nb[:, 0]

    mask = sun_elev < alt_sun_max

    star_air[~mask] = 5
    star_air[star_air >= 5] = 5

    min_airmass = np.min(star_air, axis=0)

    if Plot:
        if new:
            plt.figure()
        plt.title(
            "Instrument : %s\nObs time (Airmass<%.1f) : %.1f %%"
            % (
                instrument,
                airmass_min,
                np.sum(min_airmass < airmass_min) * 100 / len(new_t),
            )
        )
        plt.plot(new_t, min_airmass)
        plt.ylim(3, 1)
        plt.axhline(y=airmass_min, color="r")
        all_x = np.argsort(abs(min_airmass - airmass_min))[0:5]
        minx = new_t[np.min(all_x)]
        maxx = new_t[np.max(all_x)]
        if min_airmass[0] < airmass_min:
            plt.fill_betweenx(
                np.linspace(1, 3, 10), minx, x2=maxx, color="r", alpha=0.3
            )
        else:
            plt.fill_betweenx(
                np.linspace(1, 3, 10), 2020, x2=minx, color="r", alpha=0.3
            )
            plt.fill_betweenx(
                np.linspace(1, 3, 10), maxx, x2=2021, color="r", alpha=0.3
            )

        plt.xlabel("Time [year]", fontsize=14)
        plt.ylabel("Airmass", fontsize=14)

    if return_curve:
        return min_airmass, new_t
    else:
        return np.sum(min_airmass < airmass_min) * 100 / len(new_t)


def star_observability(
    ra,
    dec,
    utc_time="2020-9-16T00:00:00",
    instrument="HARPS",
    Plot=True,
    y_var="airmass",
    alt_sun_max=-12,
):
    if instrument == "HARPS":
        obs_loc = astrocoord.EarthLocation(
            lat=-29.260972 * u.deg, lon=-70.731694 * u.deg, height=2400
        )  # HARPN
        utcoffset = 0 * u.hour
    elif instrument == "HARPN":
        obs_loc = astrocoord.EarthLocation(
            lat=28.754000 * u.deg, lon=-17.889055 * u.deg, height=2387.2
        )  # HARPN
        utcoffset = 0 * u.hour
    elif instrument == "ESPRESSO":
        obs_loc = astrocoord.EarthLocation(
            lat=-24.627622 * u.deg, lon=-70.405075 * u.deg, height=2635
        )  # ESPRESSO
        utcoffset = 0 * u.hour
    elif instrument == "EXPRES":
        obs_loc = astrocoord.EarthLocation(
            lat=34.74444 * u.deg, lon=-68.578056 * u.deg, height=2360.0
        )  # LDT
        utcoffset = 0 * u.hour
    elif instrument == "CARMENES":
        obs_loc = astrocoord.EarthLocation(
            lat=37.223611 * u.deg, lon=2.546111 * u.deg, height=2168.0
        )  # Calar Alto
        utcoffset = 0 * u.hour
    elif instrument == "Geneva":
        obs_loc = astrocoord.EarthLocation(
            lat=46.204391 * u.deg, lon=6 * u.deg, height=300
        )  # Geneva
        utcoffset = 0 * u.hour

    m33 = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

    if type(utc_time) == str:
        midnight = Time.Time(utc_time, format="isot") - utcoffset
    elif type(utc_time) == float:
        midnight = Time.Time(utc_time, format="decimalyear") - utcoffset

    delta_midnight = np.linspace(-12, 12, 100) * u.hour
    times = midnight + delta_midnight
    altazframe = AltAz(obstime=times, location=obs_loc)
    sunaltazs = get_sun(times).transform_to(altazframe)
    moonaltazs = get_moon(times).transform_to(altazframe)
    moon = get_moon(times, location=obs_loc)

    dist = np.sqrt((moon.dec - m33.dec) ** 2 + (moon.ra - m33.ra) ** 2)

    m33altazs = m33.transform_to(altazframe)

    def val(obj):
        if y_var == "airmass":
            mask = obj.alt.deg <= 0
            airmass = obj.secz
            airmass[mask] = 10
            return airmass
        else:
            return obj.alt

    if Plot:
        plt.figure(figsize=(8, 6))
        plt.title(
            "Instrument : %s\nObstime : %s\nElevation : %.2f°\nAirmass : %.2f"
            % (
                instrument,
                utc_time,
                m33.transform_to(AltAz(obstime=midnight, location=obs_loc)).alt.deg,
                m33.transform_to(AltAz(obstime=midnight, location=obs_loc)).secz,
            )
        )
        plt.plot(delta_midnight, val(sunaltazs), color="y", label="Sun", zorder=10)

        plt.plot(delta_midnight, val(moonaltazs), color="r", label="Moon", zorder=10)
        plt.fill_between(
            delta_midnight.to("hr").value,
            0,
            90,
            sunaltazs.alt < -0 * u.deg,
            color="b",
            alpha=0.3,
        )
        plt.fill_between(
            delta_midnight.to("hr").value,
            0,
            90,
            sunaltazs.alt < alt_sun_max * u.deg,
            color="darkblue",
            alpha=0.5,
        )
        plt.plot(delta_midnight, val(m33altazs), color="k", lw=7)
        plt.scatter(
            delta_midnight,
            val(m33altazs),
            c=dist,
            label="Star",
            lw=0,
            s=8,
            cmap="YlOrRd_r",
            vmax=45,
            vmin=15,
            zorder=10,
        )
        ax = plt.colorbar()
        ax.ax.set_ylabel("Sky moon dist [°]")
        if y_var == "airmass":
            plt.ylabel("Airmass", fontsize=12)
            plt.ylim(2.5, 1)
        else:
            plt.ylabel("Elevation [°]", fontsize=12)
            plt.ylim(0, 90)
        plt.xlabel("Time - Obstime [hours]", fontsize=12)
        plt.legend(loc=3)
        plt.xlim(-12, 12)
        plt.axvline(x=0, color="k")
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(2))
        plt.grid()
        plt.subplots_adjust(left=0.09, right=0.99, top=0.85, bottom=0.08)


def sun_elevation(instrument="HARPS"):

    nb_pts = 24 * 60 + 1
    timess = np.linspace(2021, 2022, 365)

    if instrument == "HARPS":
        obs_loc = astrocoord.EarthLocation(
            lat=-29.260972 * u.deg, lon=-70.731694 * u.deg, height=2400
        )  # HARPN
        utcoffset = 0 * u.hour
    elif instrument == "HARPN":
        obs_loc = astrocoord.EarthLocation(
            lat=28.754000 * u.deg, lon=-17.889055 * u.deg, height=2387.2
        )  # HARPN
        utcoffset = 0 * u.hour
    elif instrument == "ESPRESSO":
        obs_loc = astrocoord.EarthLocation(
            lat=-24.627622 * u.deg, lon=-70.405075 * u.deg, height=2635
        )  # ESPRESSO
        utcoffset = 0 * u.hour
    elif instrument == "EXPRES":
        obs_loc = astrocoord.EarthLocation(
            lat=34.74444 * u.deg, lon=-68.578056 * u.deg, height=2360.0
        )  # LDT
        utcoffset = 0 * u.hour
    elif instrument == "CARMENES":
        obs_loc = astrocoord.EarthLocation(
            lat=37.223611 * u.deg, lon=2.546111 * u.deg, height=2168.0
        )  # Calar Alto
        utcoffset = 0 * u.hour
    elif instrument == "Geneva":
        obs_loc = astrocoord.EarthLocation(
            lat=46.204391 * u.deg, lon=6.143158 * u.deg, height=300
        )  # Geneva
        utcoffset = 0 * u.hour

    midnight = Time.Time(timess, format="decimalyear") - utcoffset

    nb = np.linspace(-12, 12, nb_pts)
    delta_midnight = nb * u.hour
    times = midnight + delta_midnight[:, np.newaxis]
    altazframe = AltAz(obstime=times, location=obs_loc)
    sunaltazs = get_sun(times).transform_to(altazframe)

    dico = {"time": timess, "hours": nb, "elevation": np.array(sunaltazs.alt)}

    pickle_dump(
        dico,
        open(
            "/Users/cretignier/Documents/Python/Material/Sun_elevation_"
            + instrument
            + ".p",
            "wb",
        ),
    )


def star_instrument_sky_observability(
    instrument="HARPS", z_max=1.5, d_delta=5, d_alpha=1, alt_sun_max=-12, texp=18
):
    """texp = exposure time in minutes"""
    oversampling = 10
    nb_night = 122
    nb_pts = 144

    if instrument == "HARPS":
        lat = -29.260972
        obs_loc = astrocoord.EarthLocation(
            lat=lat * u.deg, lon=-70.731694 * u.deg, height=2400
        )  # HARPN
        utcoffset = 0 * u.hour
    elif instrument == "HARPN":
        lat = 28.754000
        obs_loc = astrocoord.EarthLocation(
            lat=lat * u.deg, lon=-17.889055 * u.deg, height=2387.2
        )  # HARPN
        utcoffset = 0 * u.hour
    elif instrument == "ESPRESSO":
        lat = -24.627622
        obs_loc = astrocoord.EarthLocation(
            lat=lat * u.deg, lon=-70.405075 * u.deg, height=2635
        )  # ESPRESSO
        utcoffset = 0 * u.hour
    elif instrument == "EXPRES":
        lat = 34.74444
        obs_loc = astrocoord.EarthLocation(
            lat=34.74444 * u.deg, lon=-68.578056 * u.deg, height=2360.0
        )  # LDT
        utcoffset = 0 * u.hour
    elif instrument == "CARMENES":
        lat = 37.223611
        obs_loc = astrocoord.EarthLocation(
            lat=37.223611 * u.deg, lon=2.546111 * u.deg, height=2168.0
        )  # Calar Alto
        utcoffset = 0 * u.hour
    elif instrument == "Geneva":
        lat = 46.204391
        obs_loc = astrocoord.EarthLocation(
            lat=46.204391 * u.deg, lon=6.143158 * u.deg, height=300
        )  # Geneva
        utcoffset = 0 * u.hour

    lat_min = int(np.round((lat - 60), 0))
    lat_max = int(np.round((lat + 60), 0))
    lat_min = int(np.round((lat_min * 2 + 1) / 2, 0))
    lat_max = int(np.round((lat_max * 2 + 1) / 2, 0))

    lat_min = [-90, lat_min][int(lat_min > -90)]
    lat_max = [90, lat_max][int(lat_max < 90)]

    deltas = np.arange(lat_min, lat_max, d_delta)
    if d_alpha:
        alphas = np.arange(0, 24, d_alpha)
    else:
        alphas = [0]
    air_lim = z_max + 1

    t = np.linspace(2021, 2022, 365)[:: int(np.round(365 / nb_night, 0))]
    nb = np.linspace(-12, 12, nb_pts)
    midnight = Time.Time(t, format="decimalyear") - utcoffset

    delta_midnight = nb * u.hour

    times = midnight + delta_midnight[:, np.newaxis]
    altazframe = AltAz(obstime=times, location=obs_loc)

    sun = pd.read_pickle(
        "/Users/cretignier/Documents/Python/Material/Sun_elevation_" + instrument + ".p"
    )
    sun_elev = sun["elevation"]
    sun_t = sun["time"]
    sun_nb = sun["hours"]
    mask_sun = (sun_elev < alt_sun_max).astype("int")
    mask_sun2 = mask_sun.copy()

    for j in range(np.shape(mask_sun)[1]):
        v = mask_sun[:, j]
        val, borders = clustering(v, 0.5, 0.5)
        val2 = [np.product(i) for i in val]
        borders = borders[np.array(val2) == 1]
        mask_sun[:, j] = flat_clustering(len(v), borders, extended=-texp)
    mask_sun[0:texp, :] = mask_sun2[0:texp, :]
    mask_sun[-texp:, :] = mask_sun2[-texp:, :]

    def val(obj):
        mask = obj.alt.deg <= 0
        airmass = obj.secz
        airmass[mask] = 10
        return airmass

    visibility = []
    for delta in tqdm(deltas):
        visi = []
        for alpha in alphas:
            m33 = SkyCoord(alpha, delta, unit=(u.hourangle, u.deg))
            m33altazs = m33.transform_to(altazframe)
            star_airmass = val(m33altazs)
            visi.append(star_airmass)
        visibility.append(visi)

    visibility2 = []
    for i in tqdm(range(len(deltas))):
        visi2 = []
        for j in range(len(alphas)):
            new_t, new_nb, star_air = my_colormesh(
                t,
                nb,
                visibility[i][j],
                zoom=oversampling,
                return_output=True,
                vmin=1,
                vmax=2,
                order=1,
            )

            if (not i) & (not j):

                new_t = new_t[0]
                new_nb = new_nb[:, 0]

                loc1 = find_nearest(sun_nb, new_nb)[0]
                loc2 = find_nearest(sun_t, new_t)[0]

                mask_sun3 = mask_sun[loc1][:, loc2]

            star_air[star_air > 5] = 5
            star_air[mask_sun3 == 0] = 5

            curve = np.min(star_air, axis=0)

            curve = np.array(curve)
            curve[curve > air_lim - 1e-6] = air_lim - 1e-6
            airmass, cumu = transform_cumu(
                curve, limites=np.linspace(1, air_lim, int(1000 * (air_lim - 1) + 1))
            )
            visi2.append(cumu * 100 / len(curve))
        visibility2.append(visi2)

    vis = np.array(visibility2)

    def elev_airmass(airmass):
        return 180 * np.arcsin(1 / airmass) / np.pi

    for j in range(len(alphas)):
        visi = vis[:, j, :]
        plt.figure(figsize=(7, 7))
        plt.title("Star visibility for %s" % (instrument), fontsize=14)
        my_colormesh(airmass, deltas, visi, zoom=1, vmin=0, vmax=100, cmap="jet")
        plt.xlabel(r"$Z$ Airmass", fontsize=13)
        plt.ylabel(r" $\delta$ [°]", fontsize=13)
        plt.axvline(x=z_max, label=r"max $Z$", color="k", ls="-.")
        plt.axhline(y=lat, color="k", ls=":")

        ax = plt.colorbar()
        ax.ax.set_ylabel(r"Yearly number of nights with $Zi>Z$ [%]", fontsize=13)
        plt.grid()

        air_matrix, delta_matrix = np.meshgrid(airmass, deltas)
        cs = plt.contour(
            air_matrix, delta_matrix, visi, colors="k", levels=[50, 60, 70, 80]
        )
        cs.levels = ["%.0f%%" % (val2) for val2 in cs.levels]
        plt.clabel(cs, cs.levels, inline=True, fontsize=10)
        plt.legend()

        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax2.set_xlabel("Elevation [°]", fontsize=13)
        ticks = ax1.get_xticks()
        ticks = ticks[ticks > ax1.get_xlim()[0]]
        ticks = ticks[ticks < ax1.get_xlim()[1]]
        pos = (ticks - ax1.get_xlim()[0]) / (ax1.get_xlim()[1] - ax1.get_xlim()[0])
        lab = ["%.0f" % (i) for i in elev_airmass(ticks)]
        ax2.set_xticks(pos)
        ax2.set_xticklabels(lab)

        plt.subplots_adjust(top=0.90, right=0.97)

    if len(alphas):
        c = 0
        for j in [1.3, 1.5, 1.7, 1.9]:
            c += 1
            loc = find_nearest(airmass, j)[0][0]
            plt.subplot(2, 2, c)
            my_colormesh(
                alphas, deltas, vis[:, :, loc], vmin=50, vmax=80, cmap="viridis", zoom=5
            )
            plt.colorbar()
            plt.title("Z_lim = %.1f" % (j))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

    dico = {"delta": deltas, "alpha": alphas, "airmass": airmass, "visibility": vis}
    pickle_dump(
        dico,
        open(
            "/Users/cretignier/Documents/Python/Material/Visibility_declination_"
            + instrument
            + ".p",
            "wb",
        ),
    )


def getDuplicateColumns(df):
    """
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    """
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
    return list(duplicateColumnNames)


def symmetric_cut_functions(break_loc=0.5, length=100):
    alpha = np.zeros(length)
    cut_alpha = int(length * break_loc) - 1

    alpha[0:cut_alpha] = 1 - np.arange(cut_alpha) / cut_alpha
    alpha[length - cut_alpha :] = (
        np.arange(length - cut_alpha, length) - (length - cut_alpha)
    ) / cut_alpha
    return alpha


def film_change_variable(
    x,
    y,
    x2,
    y2,
    z=None,
    z2=None,
    cmap="jet",
    figsize=(10, 10),
    animation_time=3,
    break_label=0.5,
    xlabel="Old X var",
    xlabel2="New X var",
    ylabel="Old Y var",
    ylabel2="New Y var",
    zlabel="",
    zlabel2="",
    fontsize=15,
    lineplot=True,
):
    """for a scatter plot morphing animation"""

    dir_output = "/Users/cretignier/Documents/Python/temp_film/"

    if z is None:
        z = np.ones(len(x))
        cmap = "Greys"
        vmin = 0
        vmax = 1
    else:
        vmin = np.nanpercentile(z, 25)
        vmax = np.nanpercentile(z, 75)

    if z2 is None:
        z2 = z.copy()

    grad_x = x2 - x
    grad_y = y2 - y
    grad_z = z2 - z

    nb_frames = animation_time * 25

    alphas = symmetric_cut_functions(break_loc=break_label, length=nb_frames)

    for j in np.arange(nb_frames):
        alpha = alphas[j]
        new_x = x + (grad_x) * j / nb_frames
        new_y = y + (grad_y) * j / nb_frames
        new_z = z + (grad_z) * j / nb_frames

        fig = plt.figure(j + 1, figsize=figsize)
        if lineplot:
            plt.plot(new_x, new_y, color="k")
        plt.scatter(new_x, new_y, c=new_z, cmap=cmap, zorder=10, vmin=vmin, vmax=vmax)
        if j < (nb_frames / 2):
            plt.xlabel(xlabel, fontsize=fontsize, alpha=alpha)
            plt.ylabel(ylabel, fontsize=fontsize, alpha=alpha)
        else:
            plt.xlabel(xlabel2, fontsize=fontsize, alpha=alpha)
            plt.ylabel(ylabel2, fontsize=fontsize, alpha=alpha)
        plt.savefig(dir_output + "images" + str(j).zfill(4) + ".png")
        plt.close(fig)

    value = int(np.random.uniform(low=1, high=9999))
    os.system(
        "convert -delay 1 -quality 100 "
        + dir_output
        + "images*.png "
        + dir_output
        + "film_%s.mpeg" % (value)
    )
    print("Film saved in : %s" % (dir_output + "film_%s.mpeg" % (value)))
    os.system("rm " + dir_output + "images*.png")


# =============================================================================
# cross validation functions
# =============================================================================


def highlight_comp(var, nb_comp, legend=False, Plot=False, cluster=None, alpha=0.5):
    var = np.array(var)
    l = [np.where(var == "v%.0f" % (i))[0][0] for i in np.arange(1, nb_comp + 1)]
    l_comp = var[l]
    l_comp = [int(v[1:]) - 1 for v in l_comp]

    if Plot:
        for n, p in enumerate(l):
            plt.axvline(x=p, color="r", alpha=alpha)
            plt.axhline(y=p, color="r", alpha=alpha)
            plt.scatter(
                p, p, color=None, label="%.0f" % (n + 1), zorder=100, edgecolor="k"
            )
        if legend:
            plt.legend(loc=1)

    if cluster is not None:
        comp_inside_cluster = []
        clusters = []
        for j, k in zip(cluster[0:-1] - 0.5, cluster[1:] - 0.5):
            components = np.where((l > j) & (l < k))[0]
            if len(components):
                m, n = int(j + 0.5), int(k + 0.5)
                clusters.append([m, n])
                comp_inside_cluster.append(list(components))
        return l, l_comp, clusters, comp_inside_cluster
    else:
        return l, l_comp


def find_borders(
    matrix, cross_valid_size, frac_affected, r_min=0, Draw=False, loc_comp=None, alpha=0
):

    cluster_min_size = np.round(cross_valid_size * frac_affected, 0)

    binary_matrix = (abs(matrix) > r_min).astype("int")
    borders = []
    mask_cluster = np.zeros(len(binary_matrix))
    i = 0
    for j in np.arange(1, len(binary_matrix)):
        if (j - i) > cluster_min_size:
            if np.nansum(binary_matrix[j, i:j]) < (j - i) * 0.5:
                borders.append([i, j])
                mask_cluster[i:j] = 1
                i = j
    borders.append([borders[-1][1], len(binary_matrix)])
    cluster_loc = np.unique(np.hstack(borders))
    mask_cluster = mask_cluster.astype("bool")

    if Draw:
        for j, k in zip(cluster_loc[0:-1] - 0.5, cluster_loc[1:] - 0.5):
            plt.plot([j, j], [j, k], color="w")
            plt.plot([k, k], [j, k], color="w")
            plt.plot([j, k], [j, j], color="w")
            plt.plot([j, k], [k, k], color="w")
            if alpha:
                plt.axvline(x=j, color="white", alpha=alpha)
                plt.axvline(x=k, color="white", alpha=alpha)
                plt.axhline(y=j, color="white", alpha=alpha)
                plt.axhline(y=k, color="white", alpha=alpha)

    med_cluster = []
    inside = []
    for i, j in zip(cluster_loc[0:-1], cluster_loc[1:]):
        med_cluster.append(np.nanmedian(abs(matrix)[i:j, i:j]))
        if loc_comp is None:
            inside.append(1)
        else:
            inside.append(len(np.where((loc_comp > i - 0.5) & (loc_comp < j - 0.5))[0]))

    med_cluster = np.array(med_cluster)
    inside = np.array(inside)

    return borders, cluster_loc, mask_cluster, med_cluster, inside


def find_borders_it(
    matrix,
    cross_valid_size,
    frac_affected,
    cv_rm,
    Draw=True,
    selection=20,
    tresh_complete=0.25,
    loc_comp=None,
):

    cluster_min_size = np.round(cross_valid_size * frac_affected, 0)

    vecs = []
    for rcorr in np.arange(0.5, 0.91, 0.05):
        borders, cluster_loc, mask_cluster, med_cluster, inside = find_borders(
            matrix, cross_valid_size, frac_affected, r_min=rcorr, loc_comp=loc_comp
        )
        # vecs.append((np.sum(abs(cross_valid_size-np.diff(cluster_loc))==0),1-rcorr))
        criterion1 = (
            abs(cross_valid_size - np.diff(cluster_loc))
            <= cross_valid_size * cv_rm / 100 * tresh_complete
        )
        criterion2 = med_cluster > np.nanmax(med_cluster) / 2
        criterion3 = inside == 1
        vecs.append((np.sum(criterion1 & criterion2 & criterion3), np.round(rcorr, 2)))
    vecs.sort()
    print("Best combination for cluster detection : ", vecs[-1])
    borders, cluster_loc, mask_cluster, med_cluster, inside = find_borders(
        matrix,
        cross_valid_size,
        frac_affected,
        r_min=np.round(vecs[-1][-1], 2),
        loc_comp=loc_comp,
    )

    cluster_complete = (
        abs(cross_valid_size - (np.diff(cluster_loc) - 1))
        < cross_valid_size * cv_rm / 100 * tresh_complete
    )
    order_complete = np.where(cluster_complete)[0]
    mask_cluster_complete = np.zeros(len(matrix)).astype("bool")
    for i, j, k in zip(cluster_loc[0:-1], cluster_loc[1:], cluster_complete):
        if k:
            mask_cluster_complete[i : j + 1] = True

    if Draw:
        plt.imshow(matrix)
    med_cluster = []
    for j, k in zip(cluster_loc[0:-1] - 0.5, cluster_loc[1:] - 0.5):
        if (k - j) > cluster_min_size:
            med_cluster.append(
                np.nanmedian(
                    abs(matrix)[
                        int(j + 0.5) : int(k + 0.5), int(j + 0.5) : int(k + 0.5)
                    ]
                )
            )

    signi = [
        (i, j, k)
        for i, j, k in zip(
            1 - np.isnan(med_cluster).astype(int),
            med_cluster,
            np.arange(len(med_cluster)),
        )
    ]
    signi.sort()
    nb = [len(signi), selection][int(selection < len(signi))]

    order_signi = np.array([s[-1] for s in signi[::-1][0:selection]])
    cluster_signi = np.zeros(len(signi)).astype("bool")
    cluster_signi[order_signi] = True

    if Draw:
        for j, k, i in zip(
            cluster_loc[0:-1] - 0.5, cluster_loc[1:] - 0.5, cluster_signi
        ):
            if i:
                plt.plot([j, j], [j, k], color="w")
                plt.plot([k, k], [j, k], color="w")
                plt.plot([j, k], [j, j], color="w")
                plt.plot([j, k], [k, k], color="w")

    cluster_signi[order_complete] = False

    return (
        borders,
        cluster_loc,
        mask_cluster,
        med_cluster,
        cluster_signi,
        mask_cluster_complete,
    )


def block_matrix_iter(
    coeff, cross_valid_size, frac_affected, cv_rm, var_kept, selection, loc_comp
):

    cluster_min_size = np.round(cross_valid_size * frac_affected, 0)

    (
        borders,
        cluster_loc,
        mask_cluster,
        med_cluster,
        cluster_signi,
        mask_cluster_complete,
    ) = find_borders_it(
        coeff.matrix_corr.copy(),
        cross_valid_size,
        frac_affected,
        cv_rm,
        Draw=True,
        selection=selection,
        loc_comp=loc_comp,
    )

    var_classified = coeff.name_liste[mask_cluster_complete]
    print("Variable classified : %.0f" % (len(var_classified)))

    mapping = np.arange(len(coeff.matrix_corr.copy()))[mask_cluster]
    new_borders = find_nearest(mapping, np.array(borders)[:, 0])[0]

    mask_cluster_signi = np.zeros(len(mapping)).astype("bool")
    for j, k, i in zip(cluster_loc[0:-1] - 0.5, cluster_loc[1:] - 0.5, cluster_signi):
        if (k - j) > cluster_min_size:
            if i:
                mask_cluster_signi[int(j + 0.5) : int(k + 0.5)] = 1

    mask_cluster_signi = mask_cluster_signi.astype("bool")
    mapping2 = np.arange(len(mapping))[mask_cluster_signi]
    new_borders = np.unique(find_nearest(mapping2, new_borders)[0])
    coeff.r_matrix(
        name=var_kept[mask_cluster][mask_cluster_signi],
        absolute=True,
        Plot=False,
        rm_diagonal=True,
    )

    x, y = np.meshgrid(np.arange(len(mapping2)), np.arange(len(mapping2)))
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(abs(coeff.matrix_corr))
    save = grid_binning_slow(new_borders, new_borders, x, y, z, Draw=False)

    block_order2, dust = block_matrix2(save[1][2].copy())

    vec_ordered = var_kept[mask_cluster][mask_cluster_signi]

    new_vec_ordered = []
    for j in block_order2:
        new_vec_ordered.append(vec_ordered[new_borders[j] : new_borders[j + 1]])
    new_vec_ordered = np.hstack(new_vec_ordered)
    new_vec_ordered = np.hstack([var_classified, new_vec_ordered])
    final_vec_ordered = np.hstack(
        [new_vec_ordered, var_kept[~np.in1d(var_kept, new_vec_ordered)]]
    )

    coeff.r_matrix(name=final_vec_ordered, absolute=True, Plot=False, rm_diagonal=True)

    return coeff, final_vec_ordered


def convert_to_fits(
    files,
    ins="HARPN",
    kw_wave="wave",
    kw_flux="flux",
    kw_flux_err=None,
    ref="air",
    snr_ref=300,
):
    reference = fits.open(
        "/Users/cretignier/Documents/Python/Material/r.ESPRESSO_S1D_A_REFERENCE.fits"
    )
    files = np.sort(glob.glob(files))
    directory = "/".join(files[0].split("/")[:-1])
    last_dir = directory.split("/")[-1]

    nb_files = len(files)
    print("Number of files detected : %.0f" % (nb_files))

    os.system("mkdir " + directory.replace(last_dir, ins))
    os.system("mkdir " + directory.replace(last_dir, ins + "/DACE_TABLE"))

    zero = np.zeros(nb_files)
    reference2 = pd.read_csv(
        "/Users/cretignier/Documents/Python/Material/Dace_extracted_table.csv",
        index_col=0,
    )
    zero = zero * np.ones(len(np.array(reference2.columns)))[:, np.newaxis]
    new_tab = pd.DataFrame(zero.T, columns=reference2.columns)

    counter = -1
    for f in tqdm(files):
        counter += 1
        ext = f.split(".")[-1]
        if ext == "csv":
            file = pd.read_csv(f)
            wave = file[kw_wave]
            flux = file[kw_flux]
            if kw_flux_err is not None:
                flux_err = file[kw_flux_err]
            else:
                flux_err = np.sqrt(abs(flux))

        elif ext == "npz":
            file = np.load(f)
            wave = file[kw_wave]
            flux = file[kw_flux]
            if kw_flux_err is not None:
                flux_err = file[kw_flux_err]
            else:
                flux_err = np.sqrt(abs(flux))
                flux_err[flux < 1] = 1

        elif ext == "p":
            file = pd.read_pickle(f)
            wave = file[kw_wave]
            flux = file[kw_flux]
            if kw_flux_err is not None:
                flux_err = file[kw_flux_err]
            else:
                flux_err = np.sqrt(abs(flux))

        elif ext == "txt":
            file = np.loadtxt(f)
            wave = file[:, kw_wave]
            flux = file[:, kw_flux]

            if kw_flux_err is not None:
                flux_err = file[:, kw_flux_err]
            else:
                flux_err = np.sqrt(abs(flux))

        if ref == "air":
            wave_air = wave
            wave_void = conv_air_void(wave_air)
        else:
            wave_void = wave
            wave_air = conv_void_air(wave_void)

        data = np.rec.array(
            [
                (i1, i2, i3, i4, i5)
                for i1, i2, i3, i4, i5 in zip(
                    wave_void, wave_air, flux, flux_err, np.zeros(len(flux))
                )
            ],
            formats="float64,float64,float64,float64,float32",
            names="wavelength,wavelength_air,flux,error,quality",
        )

        reference[1] = fits.BinTableHDU(
            data, header=reference[1].header, name=reference[1].name
        )

        reference.writeto(f.replace("." + ext, ".fits"), overwrite=True)

        new_name = f.replace("." + ext, ".fits").replace(
            last_dir, last_dir + "/fits_output"
        )

        new_tab.loc[counter, "raw_file"] = new_name.split("/")[-1]
        new_tab.loc[counter, "fileroot"] = new_name
        new_tab.loc[counter, "rjd"] = counter
        new_tab.loc[counter, "mjd"] = counter

    os.system("mkdir " + directory + "/fits_output")

    os.system("mv " + directory + "/*.fits " + directory + "/fits_output")
    new_tab.to_csv(
        directory.replace(last_dir, ins + "/DACE_TABLE") + "/Dace_extracted_table.csv"
    )


def rsync_from_lesta(
    path_lesta="/hpcstorage/cretigni/Yarara/HD666/data/s1d/YARARA_LOGS",
    path_output=None,
    entry="2",
):
    if path_output is None:
        path_output = "."
    os.system(
        'rsync -av --progress -e "ssh -A cretigni@login0'
        + entry
        + '.astro.unige.ch ssh" cretigni@lesta0'
        + entry
        + ":"
        + path_lesta
        + " "
        + path_output
    )
