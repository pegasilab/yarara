"""
This modules does XXX
"""
from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray


def flat_clustering(
    length: int,
    cluster_output: ndarray,
    extended: int = 0,
    elevation_: int = 1,  # TODO: really a integer and not an array
) -> ndarray:
    vec = np.arange(length)
    if isinstance(elevation_, int):
        elevation = np.ones(len(cluster_output)) * elevation_  # type: ignore
    else:
        elevation = elevation_
    larger = (vec >= (cluster_output[:, 0][:, np.newaxis] - extended)).astype("int") * elevation[
        :, np.newaxis
    ]
    smaller = (vec <= (cluster_output[:, 1][:, np.newaxis] + 1 + extended)).astype(
        "int"
    ) * elevation[:, np.newaxis]
    flat = np.sqrt(np.sum(larger * smaller, axis=0))
    return flat


def clustering(
    array: ndarray, tresh: Union[float, int], num: Union[float, int]
) -> Tuple[ndarray, ndarray]:
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

        kept: List[NDArray[np.float64]] = []
        for j in range(len(border)):
            if border[j, -1] >= num:
                kept.append(array[border[j, 0] : border[j, 1] + 2])
        return np.array(kept), border
    else:
        raise ValueError("No cluster found with such threshold")
