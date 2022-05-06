"""
This modules does XXX
"""
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def flat_clustering(
    length: int,
    cluster_output: np.ndarray,
    extended: int = 0,
    elevation_: Union[int, NDArray[np.int64]] = 1,
) -> np.ndarray:
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
    array: NDArray[np.float64], tresh: Union[int, float], num: Union[int, float]
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
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
