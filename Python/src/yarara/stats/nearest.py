"""
This modules does XXX
"""
from typing import Any, Tuple, Union

import numpy as np
from numpy import float64, int64, ndarray
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm


# TODO: both to ArrayLike
def find_nearest(
    array: ndarray, value: Union[float, int64, int, float64, ndarray], dist_abs: bool = True
) -> Tuple[ndarray, ndarray, ndarray]:
    if type(array) != np.ndarray:
        array = np.array(array)  # TODO: -> np.asarray
    if type(value) != np.ndarray:
        value = np.array([value])

    array[np.isnan(array)] = 1e16

    idx = np.argmin(np.abs(array - value[:, np.newaxis]), axis=1)
    distance = abs(array[idx] - value)
    if dist_abs == False:
        distance = array[idx] - value
    return idx, array[idx], distance


def identify_nearest(array1: ndarray, array2: ndarray) -> ndarray:
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


def match_nearest(
    array1_: ndarray, array2_: ndarray, fast: bool = True, max_dist: None = None
) -> ndarray:
    """return a table [idx1,idx2,num1,num2,distance] matching the closest element from two arrays. Remark : algorithm very slow by conception if the arrays are too large."""
    array1 = np.array(array1_)  # TODO: asarray
    array2 = np.array(array2_)
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
