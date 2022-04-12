"""
This modules does XXX
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import savgol_filter
from scipy.stats import norm
from tqdm import tqdm


def mad(array, axis: int = 0, sigma_conv=True):
    """"""
    if axis == 0:
        step = abs(array - np.nanmedian(array, axis=axis))
    else:
        step = abs(array - np.nanmedian(array, axis=axis)[:, np.newaxis])
    return np.nanmedian(step, axis=axis) * [1, 1.48][int(sigma_conv)]


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


# stats
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
