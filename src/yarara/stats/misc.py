"""
This modules does XXX
"""
from itertools import combinations
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mad(array: NDArray[np.float64], axis: int = 0, sigma_conv: bool = True) -> float:
    """"""
    if axis == 0:
        step = abs(array - np.nanmedian(array, axis=axis))
    else:
        step = abs(array - np.nanmedian(array, axis=axis)[:, np.newaxis])
    return np.nanmedian(step, axis=axis) * [1, 1.48][int(sigma_conv)]


def combination(items):
    output = sum([list(map(list, combinations(items, i))) for i in range(len(items) + 1)], [])
    return output


def local_max(spectre: NDArray[np.float64], vicinity: int) -> NDArray[np.float64]:
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


def IQ(array_: ArrayLike, axis: Optional[int] = None) -> Union[NDArray[np.float64], float]:
    array = np.array(array_)
    return np.nanpercentile(array, 75, axis=axis) - np.nanpercentile(array, 25, axis=axis)


def merge_borders(cluster_output: np.ndarray) -> np.ndarray:
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
        matrix[:, -1] = matrix[:, 1] - matrix[:, 0] + 1  # type: ignore
        matrix1 = matrix.copy()
    return matrix1
