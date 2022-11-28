"""
This modules does XXX
"""
from typing import Literal, Tuple, Union, overload

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike, NDArray

from .. import Float
from ..util import assert_never


@overload
def rm_outliers(
    array_: ndarray,
    return_borders: bool = False,
    m: Union[Float, int] = 1.5,  # TODO: float
    kind: str = "sigma",
    axis: int = 0,
) -> Tuple[ndarray, ndarray]:
    pass


@overload
def rm_outliers(
    array_: ndarray,
    return_borders: bool,
    m: Union[Float, int] = 1.5,
    kind: str = "sigma",
    axis: int = 0,
) -> Tuple[ndarray, ndarray]:
    pass


def rm_outliers(
    array_: ndarray,
    return_borders: bool = False,
    m: Union[Float, int] = 1.5,
    kind: str = "sigma",
    axis: int = 0,
) -> Tuple[ndarray, ndarray]:
    array = np.array(array_)

    sup = np.inf
    inf = -np.inf
    if m != 0.0:
        array[array == np.inf] = np.nan
        # array[array!=array] = np.nan

        if kind == "inter":
            interquartile = np.nanpercentile(array, 75, axis=axis) - np.nanpercentile(
                array, 25, axis=axis
            )
            inf = np.nanpercentile(array, 25, axis=axis) - m * interquartile
            sup = np.nanpercentile(array, 75, axis=axis) + m * interquartile
            mask = (array >= inf) & (array <= sup)
        elif kind == "sigma":
            sup = np.nanmean(array, axis=axis) + m * np.nanstd(array, axis=axis)
            inf = np.nanmean(array, axis=axis) - m * np.nanstd(array, axis=axis)
            mask = abs(array - np.nanmean(array, axis=axis)) <= m * np.nanstd(array, axis=axis)
        elif kind == "mad":
            median = np.nanmedian(array, axis=axis)
            mad = np.nanmedian(abs(array - median), axis=axis)
            sup = median + m * mad * 1.48
            inf = median - m * mad * 1.48
            mask = abs(array - median) <= m * mad * 1.48
        else:
            assert_never(kind)
    else:
        mask = np.ones(len(array)).astype("bool")

    if return_borders:
        return mask, array[mask], sup, inf
    else:
        return mask, array[mask]
